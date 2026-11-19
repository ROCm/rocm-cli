// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Endpoint API-key enforcement for an AIM container.
//!
//! AIM images bundle vLLM's OpenAI server, so an endpoint key reaches it as
//! `VLLM_API_KEY` — and, as with the native vLLM engine, that check alone
//! guards only a fixed set of path prefixes (`/v1` and a few others) while the
//! server also answers `/invocations`, `/tokenize`, `/metrics` and more. A
//! keyed launch therefore also installs the shared every-route middleware
//! (`rocm_engine_protocol::endpoint_auth`):
//!
//! - the module is written under this engine's data directory and bind-mounted
//!   read-only into the container at [`CONTAINER_MODULE_DIR`];
//! - that directory is prepended to the container's `PYTHONPATH`, ahead of the
//!   value the caller or the image would otherwise have given it;
//! - `--middleware` is added through `AIM_ENGINE_ARGS`, the JSON object of
//!   extra `vllm serve` arguments the AIM runtime merges into the selected
//!   profile's own, keeping any arguments the caller passed there.
//!
//! All three travel in the container's env file, never on the `docker` command
//! line. Because the container's vLLM is not under this process's control,
//! the supervisor does not take the middleware's presence on trust: once the
//! container is ready it calls [`PROBE_ROUTE`] without the key and refuses to
//! serve unless that is refused ([`verify_every_route_guarded`]).

use anyhow::{Context, Result, bail};
use rocm_engine_protocol::endpoint_auth::{
    MIDDLEWARE_IMPORT_PATH, MIDDLEWARE_KEY_ENV, write_middleware_module,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Where the middleware directory is mounted inside the container.
pub(crate) const CONTAINER_MODULE_DIR: &str = "/opt/rocm-cli/python";

/// The AIM runtime's variable for extra engine arguments: a JSON object whose
/// keys are `vllm serve` flag names without the leading `--`.
pub(crate) const ENGINE_ARGS_ENV: &str = "AIM_ENGINE_ARGS";

const PYTHONPATH_ENV: &str = "PYTHONPATH";

/// The `AIM_ENGINE_ARGS` key that becomes vLLM's `--middleware`.
const MIDDLEWARE_ENGINE_ARG: &str = "middleware";

/// A route vLLM's own key check leaves open, called without the key once the
/// container is ready. vLLM's own check lets it through, so a refusal shows the
/// middleware is installed; it samples one route rather than proving every one.
pub(crate) const PROBE_ROUTE: &str = "/tokenize";

const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Host directory the middleware module is written to, under this engine's
/// data directory.
pub(crate) fn module_dir(paths: &rocm_core::AppPaths) -> PathBuf {
    paths.engine_dir(crate::ENGINE_NAME).join("python")
}

/// Write the middleware into `module_dir` and return that directory, the
/// source of the container's read-only bind mount.
pub(crate) fn stage_module(module_dir: &Path) -> Result<PathBuf> {
    write_middleware_module(module_dir)?;
    Ok(module_dir.to_path_buf())
}

/// Add the endpoint key and the middleware to a keyed container's `env`.
///
/// Sets `VLLM_API_KEY`, prepends [`CONTAINER_MODULE_DIR`] to `PYTHONPATH` and
/// adds `middleware` to `AIM_ENGINE_ARGS`. The values those two would
/// otherwise have had are kept: the caller's (`--engine-env`, where an empty
/// value means "forward it from this process's environment", resolved through
/// `host_env`), else the image's own (`image_env`, from its config) — setting
/// either in the env file replaces the image's value, so dropping it would
/// silently change what the server imports or how it runs.
///
/// Fails rather than launch when the `AIM_ENGINE_ARGS` it would extend — the
/// caller's, one forwarded from this process's environment, or the image's —
/// is not a JSON object or already names a middleware: neither can be
/// combined with the one the key needs. The refusal names the fix that suits
/// where the value came from ([`ValueSource`]).
pub(crate) fn protect_container_env(
    env: &mut BTreeMap<String, String>,
    api_key: &str,
    image_env: &BTreeMap<String, String>,
    host_env: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    let pythonpath = match inherited_value(env, image_env, PYTHONPATH_ENV, &host_env)? {
        Some(rest) if !rest.is_empty() => format!("{CONTAINER_MODULE_DIR}:{rest}"),
        _ => CONTAINER_MODULE_DIR.to_owned(),
    };
    let engine_args_source = match env.get(ENGINE_ARGS_ENV) {
        Some(value) if value.is_empty() => ValueSource::Host,
        Some(_) => ValueSource::Caller,
        None => ValueSource::Image,
    };
    let engine_args = engine_args_with_middleware(
        inherited_value(env, image_env, ENGINE_ARGS_ENV, &host_env)?.as_deref(),
        engine_args_source,
    )?;
    env.insert(MIDDLEWARE_KEY_ENV.to_owned(), api_key.to_owned());
    env.insert(PYTHONPATH_ENV.to_owned(), pythonpath);
    env.insert(ENGINE_ARGS_ENV.to_owned(), engine_args);
    Ok(())
}

/// The value `key` would have in the container without this module: the
/// caller's, else the image's.
fn inherited_value(
    env: &BTreeMap<String, String>,
    image_env: &BTreeMap<String, String>,
    key: &str,
    host_env: &impl Fn(&str) -> Option<String>,
) -> Result<Option<String>> {
    match env.get(key) {
        Some(value) if value.is_empty() => host_env(key).map(Some).with_context(|| {
            format!(
                "engine env `{key}` was passed without a value, but `{key}` is not set in this \
                 process's environment; export it before serving or pass `{key}=VALUE`"
            )
        }),
        Some(value) => Ok(Some(value.clone())),
        None => Ok(image_env.get(key).cloned()),
    }
}

/// Where an inherited `AIM_ENGINE_ARGS` came from, which decides how a user
/// gets rid of one that cannot carry the middleware.
#[derive(Debug, Clone, Copy)]
enum ValueSource {
    /// Set with a value by this launch (`--engine-env KEY=VALUE` or the
    /// engine recipe): the user changes it where they set it.
    Caller,
    /// Named with no value, so forwarded from this process's environment:
    /// the bad value lives there, not on the command line.
    Host,
    /// The image's own config: the user cannot remove it, only override it.
    Image,
}

impl ValueSource {
    fn remedy(self) -> String {
        match self {
            Self::Caller => format!(
                "change `{ENGINE_ARGS_ENV}` where this launch sets it (`--engine-env` or the \
                 engine recipe), or serve without an API key on a loopback host"
            ),
            Self::Host => format!(
                "`--engine-env {ENGINE_ARGS_ENV}` with no value forwards it from this process's \
                 environment, so set it there to a JSON object naming no \
                 `{MIDDLEWARE_ENGINE_ARG}`, or replace that flag with `--engine-env \
                 {ENGINE_ARGS_ENV}=<JSON object>`, or serve without an API key on a loopback host"
            ),
            Self::Image => format!(
                "the image sets this value, so override it with `--engine-env \
                 {ENGINE_ARGS_ENV}=<JSON object>` naming no `{MIDDLEWARE_ENGINE_ARG}` (it replaces \
                 the image's value whole, so repeat any of its other arguments you still want), \
                 or serve without an API key on a loopback host"
            ),
        }
    }
}

/// `existing` (a JSON object, or nothing) with `middleware` set to the
/// endpoint-key middleware, serialized for `AIM_ENGINE_ARGS`. `source` says
/// where `existing` came from, for the remedy a refusal names.
fn engine_args_with_middleware(existing: Option<&str>, source: ValueSource) -> Result<String> {
    let mut args = match existing.map(str::trim).filter(|value| !value.is_empty()) {
        None => serde_json::Map::new(),
        Some(text) => match serde_json::from_str::<serde_json::Value>(text) {
            Ok(serde_json::Value::Object(map)) => map,
            _ => bail!(
                "`{ENGINE_ARGS_ENV}` must be a JSON object to be combined with the endpoint \
                 API key's middleware, got `{text}`; {}",
                source.remedy()
            ),
        },
    };
    if args.contains_key(MIDDLEWARE_ENGINE_ARG) {
        bail!(
            "`{ENGINE_ARGS_ENV}` already sets `{MIDDLEWARE_ENGINE_ARG}`, which rocm-cli needs \
             to enforce the endpoint API key on every route; {}",
            source.remedy()
        );
    }
    args.insert(
        MIDDLEWARE_ENGINE_ARG.to_owned(),
        serde_json::Value::String(MIDDLEWARE_IMPORT_PATH.to_owned()),
    );
    Ok(serde_json::Value::Object(args).to_string())
}

/// Confirm a ready keyed container refuses a keyless call to [`PROBE_ROUTE`].
///
/// The middleware answers `401` before routing; a server without it answers
/// the request itself. Anything but `401`, or no answer, is an error, so a
/// container whose image ignored the middleware is not left serving with
/// [`PROBE_ROUTE`] open.
pub(crate) fn verify_every_route_guarded(container: &str, endpoint_url: &str) -> Result<()> {
    let response = rocm_core::http_post_json_with_auth(
        endpoint_url,
        PROBE_ROUTE,
        &serde_json::json!({}),
        None,
        PROBE_TIMEOUT,
    )
    .with_context(|| {
        format!(
            "could not confirm that AIM container `{container}` requires the endpoint API key \
             on every route"
        )
    })?;
    route_guard_verdict(container, response.status)
}

fn route_guard_verdict(container: &str, status: u16) -> Result<()> {
    if status == 401 {
        return Ok(());
    }
    bail!(
        "AIM container `{container}` answered `POST {PROBE_ROUTE}` without the endpoint API key \
         (HTTP {status}), so the key would not protect every route; refusing to serve it. The \
         image may not apply `{ENGINE_ARGS_ENV}` or vLLM's `--middleware`."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_host_env(_: &str) -> Option<String> {
        None
    }

    fn engine_args(env: &BTreeMap<String, String>) -> serde_json::Value {
        serde_json::from_str(&env[ENGINE_ARGS_ENV]).expect("AIM_ENGINE_ARGS is JSON")
    }

    #[test]
    fn a_keyed_container_gets_the_key_the_module_path_and_the_middleware() -> Result<()> {
        let mut env = BTreeMap::from([("AIM_PROFILE_ID".to_owned(), "p".to_owned())]);
        protect_container_env(&mut env, "k3y", &BTreeMap::new(), no_host_env)?;
        assert_eq!(env["VLLM_API_KEY"], "k3y");
        assert_eq!(env["PYTHONPATH"], CONTAINER_MODULE_DIR);
        assert_eq!(
            engine_args(&env),
            serde_json::json!({"middleware": MIDDLEWARE_IMPORT_PATH})
        );
        assert_eq!(env["AIM_PROFILE_ID"], "p", "other entries are untouched");
        Ok(())
    }

    #[test]
    fn the_images_pythonpath_and_engine_args_are_kept_behind_the_middleware() -> Result<()> {
        let image_env = BTreeMap::from([
            ("PYTHONPATH".to_owned(), "/workspace/aim-runtime".to_owned()),
            (
                ENGINE_ARGS_ENV.to_owned(),
                r#"{"max-model-len": 4096}"#.to_owned(),
            ),
        ]);
        let mut env = BTreeMap::new();
        protect_container_env(&mut env, "k3y", &image_env, no_host_env)?;
        assert_eq!(
            env["PYTHONPATH"],
            format!("{CONTAINER_MODULE_DIR}:/workspace/aim-runtime")
        );
        assert_eq!(
            engine_args(&env),
            serde_json::json!({"max-model-len": 4096, "middleware": MIDDLEWARE_IMPORT_PATH})
        );
        Ok(())
    }

    #[test]
    fn the_callers_values_win_over_the_images_and_are_kept() -> Result<()> {
        let image_env = BTreeMap::from([("PYTHONPATH".to_owned(), "/image".to_owned())]);
        let mut env = BTreeMap::from([
            ("PYTHONPATH".to_owned(), "/caller".to_owned()),
            (
                ENGINE_ARGS_ENV.to_owned(),
                r#"{"enforce-eager": null}"#.to_owned(),
            ),
        ]);
        protect_container_env(&mut env, "k3y", &image_env, no_host_env)?;
        assert_eq!(env["PYTHONPATH"], format!("{CONTAINER_MODULE_DIR}:/caller"));
        assert_eq!(
            engine_args(&env),
            serde_json::json!({"enforce-eager": null, "middleware": MIDDLEWARE_IMPORT_PATH})
        );
        Ok(())
    }

    #[test]
    fn a_forwarded_caller_value_is_read_from_this_process() -> Result<()> {
        let mut env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), String::new())]);
        protect_container_env(&mut env, "k3y", &BTreeMap::new(), |key| {
            (key == ENGINE_ARGS_ENV).then(|| r#"{"seed": 7}"#.to_owned())
        })?;
        assert_eq!(
            engine_args(&env),
            serde_json::json!({"seed": 7, "middleware": MIDDLEWARE_IMPORT_PATH})
        );

        let mut env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), String::new())]);
        let error = protect_container_env(&mut env, "k3y", &BTreeMap::new(), no_host_env)
            .expect_err("a forwarded variable that is unset cannot be merged");
        assert!(error.to_string().contains("not set"), "{error}");
        Ok(())
    }

    #[test]
    fn engine_args_that_cannot_carry_the_middleware_are_refused() {
        for existing in [
            r#"{"middleware": "other.Middleware"}"#,
            r#"["--seed", "7"]"#,
            "not json",
        ] {
            let mut env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), existing.to_owned())]);
            let error = protect_container_env(&mut env, "k3y", &BTreeMap::new(), no_host_env)
                .expect_err(existing);
            let text = error.to_string();
            assert!(text.contains(ENGINE_ARGS_ENV), "{text}");
            assert!(
                text.contains("change `AIM_ENGINE_ARGS` where this launch sets it"),
                "a caller's value is the caller's to change: {text}"
            );
            assert!(
                !env.contains_key("VLLM_API_KEY"),
                "nothing is set when the launch is refused"
            );
        }
    }

    #[test]
    fn an_images_engine_args_that_cannot_carry_the_middleware_name_the_override() -> Result<()> {
        for existing in [r#"{"middleware": "other.Middleware"}"#, "not json"] {
            let image_env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), existing.to_owned())]);
            let mut env = BTreeMap::new();
            let error = protect_container_env(&mut env, "k3y", &image_env, no_host_env)
                .expect_err(existing);
            let text = error.to_string();
            assert!(
                text.contains("the image sets this value")
                    && text.contains("--engine-env AIM_ENGINE_ARGS=")
                    && text.contains("replaces the image's value whole"),
                "an image's value can only be overridden, and the refusal must say how: {text}"
            );
            assert!(env.is_empty(), "nothing is set when the launch is refused");

            // The override the refusal names does clear it, replacing the
            // image's value rather than merging with it.
            let mut env =
                BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), r#"{"seed": 7}"#.to_owned())]);
            protect_container_env(&mut env, "k3y", &image_env, no_host_env)?;
            assert_eq!(
                engine_args(&env),
                serde_json::json!({"seed": 7, "middleware": MIDDLEWARE_IMPORT_PATH})
            );
        }
        Ok(())
    }

    #[test]
    fn a_forwarded_engine_args_that_cannot_carry_the_middleware_points_at_the_environment()
    -> Result<()> {
        let mut env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), String::new())]);
        let host = |key: &str| (key == ENGINE_ARGS_ENV).then(|| "not json".to_owned());
        let error = protect_container_env(&mut env, "k3y", &BTreeMap::new(), host)
            .expect_err("a forwarded value that is not a JSON object is refused");
        let text = error.to_string();
        assert!(
            text.contains("forwards it from this process's environment")
                && text.contains("set it there to a JSON object")
                && text.contains(
                    "replace that flag with `--engine-env AIM_ENGINE_ARGS=<JSON object>`"
                ),
            "the refusal must name both steps that clear it: {text}"
        );
        assert!(
            !env.contains_key("VLLM_API_KEY"),
            "nothing is set when the launch is refused"
        );

        // Each step it names clears the refusal. Setting the forwarded value to
        // a JSON object, keeping the bare flag:
        let fixed_host = |key: &str| (key == ENGINE_ARGS_ENV).then(|| r#"{"seed": 7}"#.to_owned());
        let mut env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), String::new())]);
        protect_container_env(&mut env, "k3y", &BTreeMap::new(), fixed_host)?;
        assert_eq!(
            engine_args(&env),
            serde_json::json!({"seed": 7, "middleware": MIDDLEWARE_IMPORT_PATH})
        );
        // Replacing the flag with an explicit value, the environment unchanged:
        let mut env = BTreeMap::from([(ENGINE_ARGS_ENV.to_owned(), r#"{"seed": 7}"#.to_owned())]);
        protect_container_env(&mut env, "k3y", &BTreeMap::new(), host)?;
        assert_eq!(
            engine_args(&env),
            serde_json::json!({"seed": 7, "middleware": MIDDLEWARE_IMPORT_PATH})
        );
        Ok(())
    }

    #[test]
    fn only_a_refusal_on_the_probe_route_counts_as_guarded() {
        assert!(route_guard_verdict("c", 401).is_ok());
        for status in [200, 400, 403, 404, 422, 500] {
            let error = route_guard_verdict("c", status).expect_err("not guarded");
            let text = error.to_string();
            assert!(text.contains(&format!("HTTP {status}")), "{text}");
            assert!(text.contains("refusing to serve"), "{text}");
        }
    }

    #[test]
    fn stage_module_writes_the_middleware_into_the_mount_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("python");
        assert_eq!(stage_module(&dir)?, dir);
        assert!(
            dir.join(format!(
                "{}.py",
                rocm_engine_protocol::endpoint_auth::MIDDLEWARE_MODULE_NAME
            ))
            .is_file()
        );
        Ok(())
    }
}
