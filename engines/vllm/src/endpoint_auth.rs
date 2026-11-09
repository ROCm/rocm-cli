// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Endpoint API-key enforcement for the native vLLM server.
//!
//! Setting `VLLM_API_KEY` alone does not protect a vLLM endpoint: vLLM's own
//! check guards only a fixed set of path prefixes (`/v1` and a few others),
//! while the server also registers inference-capable routes outside them —
//! `/invocations`, `/tokenize`, `/detokenize` and more. So whenever an endpoint
//! key is in force, the server is additionally started with
//! `--middleware rocm_cli_vllm_auth.RequireEndpointApiKey`: the ASGI middleware
//! shared with the AIM engine (`rocm_engine_protocol::endpoint_auth`), written
//! next to the engine's data at launch, that requires the key on every route
//! except `GET`/`HEAD /health` and CORS preflights.
//!
//! vLLM imports the middleware by module path, so the directory it is written
//! to is prepended to the server's `PYTHONPATH`. A failed import aborts vLLM's
//! startup, so a launch that asked for a key fails closed rather than coming up
//! with only the partial check.

use anyhow::Result;
use rocm_engine_protocol::endpoint_auth::{MIDDLEWARE_KEY_ENV, write_middleware_module};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

/// Directory the middleware module is written to under the engine's data
/// directory. Owned by the invoking user like the rest of that tree, so only
/// they can replace the code vLLM imports.
pub(crate) fn module_dir(paths: &rocm_core::AppPaths) -> PathBuf {
    paths.engine_dir(crate::ENGINE_NAME).join("python")
}

/// Require `api_key` on every route of the vLLM server `command` will start.
///
/// Sets the key, writes the middleware module into `module_dir` and prepends
/// that directory to `PYTHONPATH`. Call after any other `PYTHONPATH` change
/// (see `apply_therock_env`): an entry `command` already sets is extended
/// rather than replaced. The `--middleware` argument itself comes from
/// `rocm_engine_protocol::endpoint_auth::middleware_args` via
/// `vllm_serve_args`, so the argv stays a pure function of the launch request.
pub(crate) fn protect_endpoint(
    command: &mut ProcessCommand,
    api_key: &str,
    module_dir: &Path,
) -> Result<()> {
    write_middleware_module(module_dir)?;
    command.env(MIDDLEWARE_KEY_ENV, api_key);
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

/// The value `command` sets `key` to, if it sets one. Shared by this module's
/// tests and the launch-command tests in `process.rs`.
#[cfg(test)]
pub(crate) fn env_of(command: &ProcessCommand, key: &str) -> Option<OsString> {
    match command_env(command, key) {
        EnvOverride::Set(value) => Some(value),
        EnvOverride::Inherited | EnvOverride::Removed => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_engine_protocol::endpoint_auth::{MIDDLEWARE_MODULE_NAME, MIDDLEWARE_MODULE_SOURCE};
    use std::fs;

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

        assert_eq!(
            env_of(&command, MIDDLEWARE_KEY_ENV),
            Some("secret-key".into())
        );
        let pythonpath = env_of(&command, "PYTHONPATH").expect("PYTHONPATH set");
        let entries: Vec<PathBuf> = std::env::split_paths(&pythonpath).collect();
        assert_eq!(
            entries,
            vec![module_dir.clone(), PathBuf::from("/opt/existing")],
            "the middleware directory goes first and an earlier entry is kept"
        );
        let written = fs::read_to_string(module_dir.join(format!("{MIDDLEWARE_MODULE_NAME}.py")))?;
        assert_eq!(written, MIDDLEWARE_MODULE_SOURCE);
        Ok(())
    }
}
