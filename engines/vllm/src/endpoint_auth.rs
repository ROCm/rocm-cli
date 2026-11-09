// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Endpoint API-key enforcement for the vLLM server.
//!
//! Setting `VLLM_API_KEY` alone does not protect a vLLM endpoint: vLLM's own
//! check guards only a fixed set of path prefixes (`/v1` and a few others),
//! while the server also registers inference-capable routes outside them —
//! `/invocations`, `/tokenize`, `/detokenize` and more. So whenever an endpoint
//! key is in force, the server is additionally started with
//! `--middleware rocm_cli_vllm_auth.RequireEndpointApiKey`: a small ASGI
//! middleware, shipped inside this binary and written next to the engine's
//! data at launch, that requires the key on every route except `GET /health`.
//!
//! vLLM imports the middleware by module path, so the directory it is written
//! to is prepended to the server's `PYTHONPATH`. A failed import aborts vLLM's
//! startup, so a launch that asked for a key fails closed rather than coming up
//! with only the partial check.

use anyhow::{Context, Result};
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

/// Python module name the middleware is imported as.
const MODULE_NAME: &str = "rocm_cli_vllm_auth";

/// Source of the middleware module, compiled into the binary so the copy vLLM
/// imports always matches the adapter that launched it.
const MODULE_SOURCE: &str = include_str!("rocm_cli_vllm_auth.py");

/// The `--middleware` import path vLLM is given.
pub(crate) const MIDDLEWARE_IMPORT_PATH: &str = "rocm_cli_vllm_auth.RequireEndpointApiKey";

/// Environment variable carrying the key to both vLLM's own check and the
/// middleware. The middleware reads this exact name (`KEY_ENV` in the module).
pub(crate) const KEY_ENV: &str = "VLLM_API_KEY";

/// Extra `vllm serve` arguments that install the middleware.
pub(crate) fn middleware_args() -> [String; 2] {
    ["--middleware".to_owned(), MIDDLEWARE_IMPORT_PATH.to_owned()]
}

/// Directory the middleware module is written to under the engine's data
/// directory. Owned by the invoking user like the rest of that tree, so only
/// they can replace the code vLLM imports.
pub(crate) fn module_dir(paths: &rocm_core::AppPaths) -> PathBuf {
    paths.engine_dir(crate::ENGINE_NAME).join("python")
}

/// Write the middleware module into `dir`, replacing any earlier copy.
///
/// Written to a temporary file and renamed into place, so a concurrent launch
/// importing the module never reads a half-written file.
pub(crate) fn write_module(dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join(format!("{MODULE_NAME}.py"));
    let mut staged = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to stage a file in {}", dir.display()))?;
    staged
        .write_all(MODULE_SOURCE.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    staged
        .persist(&path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

/// Require `api_key` on every route of the vLLM server `command` will start.
///
/// Sets the key, writes the middleware module into `module_dir` and prepends
/// that directory to `PYTHONPATH`. Call after any other `PYTHONPATH` change
/// (see `apply_therock_env`): an entry `command` already sets is extended
/// rather than replaced. The `--middleware` argument itself comes from
/// [`middleware_args`] via `vllm_serve_args`, so the argv stays a pure
/// function of the launch request.
pub(crate) fn protect_endpoint(
    command: &mut ProcessCommand,
    api_key: &str,
    module_dir: &Path,
) -> Result<()> {
    write_module(module_dir)?;
    command.env(KEY_ENV, api_key);
    let pythonpath = middleware_pythonpath(
        module_dir,
        command_env(command, "PYTHONPATH"),
        std::env::var_os("PYTHONPATH"),
    )?;
    command.env("PYTHONPATH", pythonpath);
    Ok(())
}

/// How a [`ProcessCommand`] treats one environment variable.
#[derive(Debug, PartialEq, Eq)]
enum EnvOverride {
    /// The command leaves it alone; the child inherits this process's value.
    Inherited,
    /// The command removes it (`env_remove`).
    Removed,
    /// The command sets it to this value.
    Set(OsString),
}

/// The server's `PYTHONPATH`: `module_dir` first, then whatever the server
/// would otherwise have seen — the value the command sets, nothing if it
/// removes the variable, or `inherited` when it leaves it alone. Dropping the
/// inherited value would silently change which packages the server imports.
fn middleware_pythonpath(
    module_dir: &Path,
    command_value: EnvOverride,
    inherited: Option<OsString>,
) -> Result<OsString> {
    let current = match command_value {
        EnvOverride::Inherited => inherited,
        EnvOverride::Removed => None,
        EnvOverride::Set(value) => Some(value),
    };
    crate::process::prepend_path_entries(&[module_dir.to_path_buf()], current)
}

/// How `command` treats the environment variable `key`.
fn command_env(command: &ProcessCommand, key: &str) -> EnvOverride {
    command
        .get_envs()
        .find_map(|(name, value)| (name == key).then_some(value))
        .map_or(EnvOverride::Inherited, |value| match value {
            Some(value) => EnvOverride::Set(value.to_owned()),
            None => EnvOverride::Removed,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;

    fn env_of(command: &ProcessCommand, key: &str) -> Option<OsString> {
        match command_env(command, key) {
            EnvOverride::Set(value) => Some(value),
            EnvOverride::Inherited | EnvOverride::Removed => None,
        }
    }

    fn entries(value: &OsString) -> Vec<PathBuf> {
        std::env::split_paths(value).collect()
    }

    #[test]
    fn middleware_pythonpath_keeps_the_inherited_value_when_the_command_sets_none() -> Result<()> {
        let module_dir = PathBuf::from("/data/python");
        let composed = middleware_pythonpath(
            &module_dir,
            EnvOverride::Inherited,
            Some("/user/site".into()),
        )?;
        assert_eq!(
            entries(&composed),
            vec![module_dir, PathBuf::from("/user/site")],
            "the server must still see the PYTHONPATH it would have inherited"
        );
        Ok(())
    }

    #[test]
    fn middleware_pythonpath_prefers_the_command_value_over_the_inherited_one() -> Result<()> {
        let module_dir = PathBuf::from("/data/python");
        let composed = middleware_pythonpath(
            &module_dir,
            EnvOverride::Set("/sdk/amd_smi".into()),
            Some("/user/site".into()),
        )?;
        assert_eq!(
            entries(&composed),
            vec![module_dir, PathBuf::from("/sdk/amd_smi")]
        );
        Ok(())
    }

    #[test]
    fn middleware_pythonpath_honours_a_command_that_removes_the_variable() -> Result<()> {
        let module_dir = PathBuf::from("/data/python");
        let composed =
            middleware_pythonpath(&module_dir, EnvOverride::Removed, Some("/user/site".into()))?;
        assert_eq!(entries(&composed), vec![module_dir]);
        Ok(())
    }

    #[test]
    fn command_env_distinguishes_inherited_removed_and_set() {
        let mut command = ProcessCommand::new("vllm");
        assert_eq!(command_env(&command, "PYTHONPATH"), EnvOverride::Inherited);
        command.env_remove("PYTHONPATH");
        assert_eq!(command_env(&command, "PYTHONPATH"), EnvOverride::Removed);
        command.env("PYTHONPATH", "/x");
        assert_eq!(
            command_env(&command, "PYTHONPATH"),
            EnvOverride::Set("/x".into())
        );
    }

    #[test]
    fn protect_endpoint_sets_key_and_makes_the_middleware_importable() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let module_dir = temp.path().join("python");
        let mut command = ProcessCommand::new("vllm");
        command.env("PYTHONPATH", "/opt/existing");

        protect_endpoint(&mut command, "secret-key", &module_dir)?;

        assert_eq!(env_of(&command, KEY_ENV), Some("secret-key".into()));
        let pythonpath = env_of(&command, "PYTHONPATH").expect("PYTHONPATH set");
        let entries: Vec<PathBuf> = std::env::split_paths(&pythonpath).collect();
        assert_eq!(
            entries,
            vec![module_dir.clone(), PathBuf::from("/opt/existing")],
            "the middleware directory goes first and an earlier entry is kept"
        );
        let written = fs::read_to_string(module_dir.join(format!("{MODULE_NAME}.py")))?;
        assert_eq!(written, MODULE_SOURCE);
        Ok(())
    }

    #[test]
    fn write_module_replaces_a_stale_copy() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join(format!("{MODULE_NAME}.py"));
        fs::write(&path, "class RequireEndpointApiKey: pass\n")?;
        assert_eq!(write_module(temp.path())?, path);
        assert_eq!(fs::read_to_string(&path)?, MODULE_SOURCE);
        Ok(())
    }

    #[test]
    fn import_path_names_the_shipped_module_and_class() {
        let (module, class) = MIDDLEWARE_IMPORT_PATH
            .rsplit_once('.')
            .expect("module.Class");
        assert_eq!(module, MODULE_NAME);
        assert!(MODULE_SOURCE.contains(&format!("\nclass {class}:")));
        assert!(MODULE_SOURCE.contains(&format!("KEY_ENV = \"{KEY_ENV}\"")));
    }

    /// Drives the shipped middleware with a stdlib-only ASGI harness, the same
    /// way vLLM's `--middleware` loader mounts it (`app.add_middleware(cls)`
    /// wraps the app in `cls(app)`). Proves the routes vLLM leaves outside its
    /// own key check — `/invocations`, `/tokenize`, ... — answer 401 without
    /// the key, and that only `GET /health` stays open.
    ///
    /// Needs `python3` on PATH and fails loudly without it rather than passing
    /// vacuously. Linux-only: vLLM serving is Linux/WSL-only in rocm-cli, so
    /// the module never runs anywhere else.
    #[test]
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "vLLM serving (and so this middleware) is Linux/WSL-only"
    )]
    fn middleware_requires_the_key_on_every_route_but_health() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let module_dir = temp.path().join("python");
        write_module(&module_dir)?;
        let harness = temp.path().join("harness.py");
        fs::write(&harness, ASGI_HARNESS)?;

        let output = ProcessCommand::new("python3")
            .arg(&harness)
            .env("PYTHONPATH", &module_dir)
            .env(KEY_ENV, "secret-key")
            .stdin(Stdio::null())
            .output()
            .context("python3 is required to exercise the vLLM auth middleware")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "middleware harness failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(stdout.contains("ALL OK"), "harness stdout:\n{stdout}");
        Ok(())
    }

    /// Construction must fail when the key is empty, so a protected launch
    /// whose key went missing cannot answer a request without checking one.
    #[test]
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "vLLM serving (and so this middleware) is Linux/WSL-only"
    )]
    fn middleware_refuses_to_start_without_a_key() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_module(temp.path())?;
        let output = ProcessCommand::new("python3")
            .args([
                "-c",
                "import rocm_cli_vllm_auth as m; m.RequireEndpointApiKey(lambda *a: None)",
            ])
            .env("PYTHONPATH", temp.path())
            .env_remove(KEY_ENV)
            .stdin(Stdio::null())
            .output()
            .context("python3 is required to exercise the vLLM auth middleware")?;
        assert!(
            !output.status.success(),
            "constructing without a key must fail"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("VLLM_API_KEY is empty"),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    const ASGI_HARNESS: &str = r#"
import asyncio
import rocm_cli_vllm_auth as m

async def inner(scope, receive, send):
    if scope["type"] == "websocket":
        await send({"type": "websocket.accept"})
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"inner"})

async def call(app, method, path, headers=(), kind="http", root_path=""):
    sent = []
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        sent.append(message)
    scope = {"type": kind, "method": method, "path": path, "root_path": root_path,
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}
    await app(scope, receive, send)
    first = sent[0]
    if first["type"] == "http.response.start":
        return first["status"]
    return first["type"]

GOOD = [("Authorization", "Bearer secret-key")]
CASES = [
    # Routes outside vLLM's own guarded prefixes must need the key too.
    ("POST", "/invocations", (), 401),
    ("POST", "/tokenize", (), 401),
    ("POST", "/detokenize", (), 401),
    ("POST", "/generative_scoring", (), 401),
    ("POST", "/scale_elastic_ep", (), 401),
    ("GET", "/metrics", (), 401),
    ("GET", "/ping", (), 401),
    ("POST", "/v1/chat/completions", (), 401),
    ("GET", "/some/future/route", (), 401),
    # Wrong, malformed or non-bearer credentials.
    ("POST", "/invocations", [("Authorization", "Bearer wrong")], 401),
    ("POST", "/invocations", [("Authorization", "Bearer secret-key-and-more")], 401),
    ("POST", "/invocations", [("Authorization", "Basic secret-key")], 401),
    ("POST", "/invocations", [("Authorization", "secret-key")], 401),
    ("POST", "/invocations", [("X-Api-Key", "secret-key")], 401),
    # Only GET/HEAD /health is open, and only at exactly that path.
    ("GET", "/health", (), 200),
    ("HEAD", "/health", (), 200),
    ("POST", "/health", (), 401),
    ("GET", "/health/", (), 401),
    ("GET", "/healthz", (), 401),
    # A non-preflight OPTIONS is not exempt; a CORS preflight is.
    ("OPTIONS", "/invocations", (), 401),
    ("OPTIONS", "/invocations",
     [("Origin", "http://x"), ("Access-Control-Request-Method", "POST")], 200),
    # The right key opens every route, with any bearer-scheme casing.
    ("POST", "/invocations", GOOD, 200),
    ("POST", "/tokenize", GOOD, 200),
    ("GET", "/metrics", GOOD, 200),
    ("POST", "/v1/chat/completions", [("Authorization", "bearer secret-key")], 200),
]

async def main():
    app = m.RequireEndpointApiKey(inner)
    failures = []
    for method, path, headers, want in CASES:
        got = await call(app, method, path, headers)
        if got != want:
            failures.append(f"{method} {path} {headers!r}: want {want}, got {got}")
    # root_path is stripped before matching, as vLLM's own check does.
    got = await call(app, "GET", "/prefix/health", root_path="/prefix")
    if got != 200:
        failures.append(f"root_path /health: want 200, got {got}")
    # WebSocket routes are closed before the handshake without the key.
    got = await call(app, None, "/v1/realtime", kind="websocket")
    if got != "websocket.close":
        failures.append(f"websocket without key: want close, got {got}")
    got = await call(app, None, "/v1/realtime", GOOD, kind="websocket")
    if got != "websocket.accept":
        failures.append(f"websocket with key: want accept, got {got}")
    # Lifespan events pass straight through.
    seen = []
    async def lifespan_inner(scope, receive, send):
        seen.append(scope["type"])
    await m.RequireEndpointApiKey(lifespan_inner)({"type": "lifespan"}, None, None)
    if seen != ["lifespan"]:
        failures.append(f"lifespan not passed through: {seen}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("ALL OK")

asyncio.run(main())
"#;
}
