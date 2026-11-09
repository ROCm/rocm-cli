// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The endpoint-key middleware for vLLM's OpenAI server, shared by every engine
//! that runs one.
//!
//! Setting `VLLM_API_KEY` alone does not protect a vLLM endpoint: vLLM's own
//! check guards only a fixed set of path prefixes (`/v1` and a few others),
//! while the server also registers inference-capable routes outside them —
//! `/invocations`, `/tokenize`, `/detokenize` and more. So whenever an endpoint
//! key is in force, the server is additionally started with
//! `--middleware rocm_cli_vllm_auth.RequireEndpointApiKey`: the small ASGI
//! middleware in `rocm_cli_vllm_auth.py` beside this file, compiled
//! into the binary and written out at launch, that requires the key on every
//! route except `GET`/`HEAD /health`.
//!
//! This module owns the middleware itself — its source, its import path and
//! writing it to disk. How it reaches the server is each engine's business:
//! the native vLLM engine puts the directory on the server's `PYTHONPATH`
//! (`engines/vllm`'s `endpoint_auth.rs`), while the AIM engine bind-mounts it
//! into the container (`engines/aim`'s `endpoint_auth.rs`).

use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Python module name the middleware is imported as.
pub const MIDDLEWARE_MODULE_NAME: &str = "rocm_cli_vllm_auth";

/// Source of the middleware module, compiled into the binary so the copy vLLM
/// imports always matches the adapter that launched it.
pub const MIDDLEWARE_MODULE_SOURCE: &str = include_str!("rocm_cli_vllm_auth.py");

/// The `--middleware` import path vLLM is given.
pub const MIDDLEWARE_IMPORT_PATH: &str = "rocm_cli_vllm_auth.RequireEndpointApiKey";

/// Environment variable carrying the key to both vLLM's own check and the
/// middleware. The middleware reads this exact name (`KEY_ENV` in the module).
pub const MIDDLEWARE_KEY_ENV: &str = "VLLM_API_KEY";

/// Extra `vllm serve` arguments that install the middleware.
#[must_use]
pub fn middleware_args() -> [String; 2] {
    ["--middleware".to_owned(), MIDDLEWARE_IMPORT_PATH.to_owned()]
}

/// Write the middleware module into `dir`, replacing any earlier copy, and
/// return the file's path.
///
/// Written to a temporary file and renamed into place, so a concurrent launch
/// importing the module never reads a half-written file. The file holds no
/// secret, and on unix it and `dir` are made world-readable (`0644`/`0755`):
/// a container that bind-mounts `dir` may import it as a user other than the
/// one who wrote it.
pub fn write_middleware_module(dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join(format!("{MIDDLEWARE_MODULE_NAME}.py"));
    let mut staged = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to stage a file in {}", dir.display()))?;
    staged
        .write_all(MIDDLEWARE_MODULE_SOURCE.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        staged
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o644))
            .with_context(|| format!("failed to set permissions on {}", path.display()))?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o755))
            .with_context(|| format!("failed to set permissions on {}", dir.display()))?;
    }
    staged
        .persist(&path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    #[test]
    fn write_middleware_module_replaces_a_stale_copy() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join(format!("{MIDDLEWARE_MODULE_NAME}.py"));
        fs::write(&path, "class RequireEndpointApiKey: pass\n")?;
        assert_eq!(write_middleware_module(temp.path())?, path);
        assert_eq!(fs::read_to_string(&path)?, MIDDLEWARE_MODULE_SOURCE);
        Ok(())
    }

    /// A container bind-mounting the directory may run as another user, so
    /// both the directory and the module must be readable by anyone.
    #[cfg(unix)]
    #[test]
    fn write_middleware_module_leaves_the_module_readable_by_other_users() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("python");
        fs::create_dir_all(&dir)?;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        let path = write_middleware_module(&dir)?;
        assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, 0o644);
        assert_eq!(fs::metadata(&dir)?.permissions().mode() & 0o777, 0o755);
        Ok(())
    }

    #[test]
    fn import_path_names_the_shipped_module_and_class() {
        let (module, class) = MIDDLEWARE_IMPORT_PATH
            .rsplit_once('.')
            .expect("module.Class");
        assert_eq!(module, MIDDLEWARE_MODULE_NAME);
        assert!(MIDDLEWARE_MODULE_SOURCE.contains(&format!("\nclass {class}:")));
        assert!(MIDDLEWARE_MODULE_SOURCE.contains(&format!("KEY_ENV = \"{MIDDLEWARE_KEY_ENV}\"")));
    }

    #[test]
    fn middleware_args_install_the_shipped_class() {
        assert_eq!(
            middleware_args(),
            ["--middleware".to_owned(), MIDDLEWARE_IMPORT_PATH.to_owned()]
        );
    }

    /// Drives the shipped middleware with a stdlib-only ASGI harness, the same
    /// way vLLM's `--middleware` loader mounts it (`app.add_middleware(cls)`
    /// wraps the app in `cls(app)`). Proves the routes vLLM leaves outside its
    /// own key check — `/invocations`, `/tokenize`, ... — answer 401 without
    /// the key, and that only `GET`/`HEAD /health` and CORS preflights stay
    /// open.
    ///
    /// Needs `python3` on PATH and fails loudly without it rather than passing
    /// vacuously. Linux-only: vLLM serving (natively or in an AIM container) is
    /// Linux/WSL-only in rocm-cli, so the module never runs anywhere else.
    #[test]
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "vLLM serving (and so this middleware) is Linux/WSL-only"
    )]
    fn middleware_requires_the_key_on_every_route_but_health() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let module_dir = temp.path().join("python");
        write_middleware_module(&module_dir)?;
        let harness = temp.path().join("harness.py");
        fs::write(&harness, ASGI_HARNESS)?;

        let output = Command::new("python3")
            .arg(&harness)
            .env("PYTHONPATH", &module_dir)
            .env(MIDDLEWARE_KEY_ENV, "secret-key")
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
        write_middleware_module(temp.path())?;
        let output = Command::new("python3")
            .args([
                "-c",
                "import rocm_cli_vllm_auth as m; m.RequireEndpointApiKey(lambda *a: None)",
            ])
            .env("PYTHONPATH", temp.path())
            .env_remove(MIDDLEWARE_KEY_ENV)
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
    # A CORS preflight (OPTIONS carrying both Origin and
    # Access-Control-Request-Method) is the only other keyless request. Each
    # part alone is not one, and neither are those headers on another method.
    ("OPTIONS", "/invocations", (), 401),
    ("OPTIONS", "/invocations",
     [("Origin", "http://x"), ("Access-Control-Request-Method", "POST")], 200),
    ("OPTIONS", "/invocations", [("Origin", "http://x")], 401),
    ("OPTIONS", "/invocations", [("Access-Control-Request-Method", "POST")], 401),
    ("POST", "/invocations",
     [("Origin", "http://x"), ("Access-Control-Request-Method", "POST")], 401),
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
    # The /health exemption is for HTTP GET/HEAD only, not a WebSocket there.
    # This catches an exemption keyed on the path alone (for any scope type).
    # Dropping only the scope-type guard in `_exempt` would still pass here,
    # because a WebSocket scope carries no method for the GET/HEAD check.
    got = await call(app, None, "/health", kind="websocket")
    if got != "websocket.close":
        failures.append(f"websocket /health without key: want close, got {got}")
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
