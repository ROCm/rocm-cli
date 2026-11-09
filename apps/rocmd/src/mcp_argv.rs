// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Argument handling for the MCP tools that spawn `rocm`: the read-only
//! allowlist for the `rocm_command` tool and the builders that turn each
//! mutating tool's JSON arguments into a `rocm` argv. Nothing here spawns a
//! process or writes anything; [`crate::mcp`] is the only caller and owns the
//! dispatch.

use anyhow::{Context, Result, bail};
use rocm_core::DEFAULT_LOCAL_HOST;
use serde_json::Value;

pub(crate) fn normalized_rocm_command_args(
    arguments: &serde_json::Map<String, Value>,
) -> Result<Vec<String>> {
    let values = arguments
        .get("args")
        .and_then(Value::as_array)
        .context("rocm_command requires `args`")?;
    if values.is_empty() || values.len() > 64 {
        bail!("rocm_command `args` must contain 1 to 64 strings");
    }
    let mut args = Vec::with_capacity(values.len());
    for value in values {
        let arg = value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .context("rocm_command `args` entries must be non-empty strings")?;
        if arg.contains('\0') || arg.contains('\n') || arg.contains('\r') {
            bail!("rocm_command arguments must not contain control characters");
        }
        if arg.len() > 512 {
            bail!("rocm_command argument is too long");
        }
        args.push(arg.to_owned());
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("rocm"))
    {
        args.remove(0);
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfy"))
    {
        args[0] = "comfyui".to_owned();
    }
    if args.is_empty() {
        bail!("rocm_command args should omit the leading `rocm` program name");
    }
    Ok(args)
}

pub(crate) fn ensure_rocm_command_is_read_only(args: &[String]) -> Result<()> {
    let first = args.first().map(|value| value.to_ascii_lowercase());
    let second = args.get(1).map(|value| value.to_ascii_lowercase());
    let read_only = match first.as_deref() {
        Some("examine" | "version" | "model" | "models" | "daemon" | "logs") => true,
        Some("update") => !args.iter().any(|arg| arg == "--apply"),
        Some("runtimes") => {
            second.as_deref().is_none_or(|value| value == "list")
                || (second
                    .as_deref()
                    .is_some_and(|value| value == "uninstall" || value == "remove")
                    && args.iter().any(|arg| arg == "--dry-run"))
        }
        Some("engines") => second.as_deref().is_some_and(|value| value == "list"),
        Some("services") => second
            .as_deref()
            .is_none_or(|value| matches!(value, "list" | "logs")),
        Some("automations") => second.as_deref().is_none_or(|value| value == "list"),
        Some("config") => second.as_deref() == Some("show"),
        Some("comfyui") => second
            .as_deref()
            .is_none_or(|value| matches!(value, "status" | "logs" | "log")),
        Some("uninstall") => args.iter().any(|arg| arg == "--dry-run"),
        // `storage report` (the default subcommand) only measures folders. The
        // two `remove-*` verbs delete, so they stay off the read-only list.
        Some("storage") => second.as_deref().is_none_or(|value| value == "report"),
        // `setup status` reports first-time setup state (read-only); `setup reset`
        // clears the completion/dismissal state and is mutating (it does not by
        // itself reopen onboarding). Mirrors the bin's rocm_command classifier so
        // the read-only allowlist is consistent across binaries.
        Some("setup") => second.as_deref().is_none_or(|value| value == "status"),
        // `remote targets` reads the local tailnet, `doctor` fetches another
        // machine's state and scores it here, `status` probes sessions that
        // already exist. None of them change anything on either machine.
        // `serve`, `attach` and `stop` start, publish or tear down, so they stay
        // off the list and go through the approval UI like any other mutation.
        Some("remote") => second
            .as_deref()
            .is_some_and(|value| matches!(value, "targets" | "doctor" | "status")),
        _ => false,
    };
    if read_only {
        return Ok(());
    }
    bail!(
        "rocm_command changes local ROCm state or is unsupported here; request it through the ROCm CLI approval UI instead"
    )
}

pub(crate) fn build_install_sdk_args(
    arguments: &serde_json::Map<String, Value>,
    dry_run: bool,
) -> Result<Vec<String>> {
    let channel = arguments
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or("release");
    let format = arguments
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("wheel");
    let prefix = arguments.get("prefix").and_then(Value::as_str);
    let version = arguments.get("version").and_then(Value::as_str);
    let build_date = arguments.get("build_date").and_then(Value::as_str);
    let allow_system_prefix = arguments
        .get("allow_system_prefix")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if version.is_some() && build_date.is_some() {
        bail!("install_sdk accepts either `version` or `build_date`, not both");
    }

    let mut argv = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        channel.to_owned(),
        "--format".to_owned(),
        format.to_owned(),
    ];
    if let Some(prefix) = prefix {
        let prefix_path = std::path::Path::new(prefix);
        if system_prefix_requires_ack(prefix_path) && !allow_system_prefix {
            bail!(
                "install_sdk prefix `{}` is outside the user home; require `allow_system_prefix=true` before using system paths",
                prefix_path.display()
            );
        }
        argv.push("--prefix".to_owned());
        argv.push(prefix.to_owned());
    }
    if let Some(version) = version {
        if version.trim().is_empty() {
            bail!("install_sdk `version` cannot be empty");
        }
        argv.push("--version".to_owned());
        argv.push(version.to_owned());
    }
    if let Some(build_date) = build_date {
        if build_date.trim().is_empty() {
            bail!("install_sdk `build_date` cannot be empty");
        }
        argv.push("--build-date".to_owned());
        argv.push(build_date.to_owned());
    }
    if dry_run {
        argv.push("--dry-run".to_owned());
    } else {
        // `run_rocm_capture_for_paths` spawns `rocm` with null stdin, so
        // `interactive_terminal()` is false in the child and an active default
        // managed runtime would make the approval gate refuse with "re-run with
        // `--approve-replacing-active-default`" — a flag no MCP caller of this
        // tool can supply.
        //
        // Not `--yes` itself: that flag carries a second, unrelated consent —
        // approving required system-package installs, which run `sudo`. This
        // spawn has no terminal, so it could never answer a sudo password
        // prompt; granting that consent would make the vLLM/OpenMPI step attempt
        // an install it cannot complete and abort the engine auto-install that
        // previously warned and continued. `--approve-replacing-active-default`
        // grants only the runtime-displacement consent the gate asks for.
        //
        // Consent is not bypassed: `install_sdk` is in
        // `mcp_tool_requires_direct_approval`, so a direct `rocmd mcp-call`
        // needs `--allow-mutation` after an explicit user approval, and over the
        // MCP protocol the tool is annotated `destructiveHint` for the client's
        // approval UI. Mirrors the chat/MCP arm in `apps/rocm`. The dry-run
        // branch never reaches the gate (it returns earlier), so it stays bare.
        argv.push("--approve-replacing-active-default".to_owned());
    }
    Ok(argv)
}

pub(crate) fn build_install_engine_args(
    arguments: &serde_json::Map<String, Value>,
) -> Result<Vec<String>> {
    let engine = arguments
        .get("engine")
        .and_then(Value::as_str)
        .context("install_engine requires `engine`")?;
    let runtime_id = arguments
        .get("runtime_id")
        .and_then(Value::as_str)
        .unwrap_or("therock-release");
    let python_version = arguments.get("python_version").and_then(Value::as_str);
    let reinstall = arguments
        .get("reinstall")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut argv = vec![
        "engines".to_owned(),
        "install".to_owned(),
        engine.to_owned(),
        "--runtime-id".to_owned(),
        runtime_id.to_owned(),
    ];
    if let Some(python_version) = python_version {
        argv.push("--python-version".to_owned());
        argv.push(python_version.to_owned());
    }
    if reinstall {
        argv.push("--reinstall".to_owned());
    }
    Ok(argv)
}

pub(crate) fn build_launch_server_args(
    arguments: &serde_json::Map<String, Value>,
) -> Result<Vec<String>> {
    let model = arguments
        .get("model")
        .and_then(Value::as_str)
        .context("launch_server requires `model`")?;
    let host = arguments
        .get("host")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_LOCAL_HOST);
    let allow_public_bind = arguments
        .get("allow_public_bind")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "launch_server host `{host}` is not loopback; require `allow_public_bind=true` before binding a non-local interface"
        );
    }
    // A malformed `port` must be rejected here, not silently dropped by the
    // argv builder and left to fall through to automatic port selection.
    let port = rocm_core::launch_port_argument(arguments.get("port"))
        .map_err(|reason| anyhow::anyhow!("launch_server argument `port` {reason}"))?;
    // `rocm serve` proves a port is free only on the exact default host; every
    // other spelling (including the loopback aliases `localhost`/`::1` the
    // check above still accepts) needs an explicit port or the CLI call
    // refuses later with a less actionable error. See `is_canonical_loopback_host`.
    if port.is_none() && !rocm_core::is_canonical_loopback_host(host) {
        bail!(
            "launch_server host `{host}` needs an explicit `port`; automatic port selection is \
             only supported on the default host {DEFAULT_LOCAL_HOST}"
        );
    }

    let mut argv = vec!["serve".to_owned(), model.to_owned(), "--managed".to_owned()];
    if let Some(engine) = arguments.get("engine").and_then(Value::as_str) {
        argv.push("--engine".to_owned());
        argv.push(engine.to_owned());
    }
    if let Some(device) = arguments.get("device").and_then(Value::as_str) {
        argv.push("--device".to_owned());
        argv.push(device.to_owned());
    }
    if let Some(runtime_id) = arguments.get("runtime_id").and_then(Value::as_str) {
        argv.push("--runtime-id".to_owned());
        argv.push(runtime_id.to_owned());
    }
    if let Some(env_id) = arguments.get("env_id").and_then(Value::as_str) {
        argv.push("--env-id".to_owned());
        argv.push(env_id.to_owned());
    }
    argv.push("--host".to_owned());
    argv.push(host.to_owned());
    if allow_public_bind {
        argv.push("--allow-public-bind".to_owned());
    }
    if let Some(port) = port {
        argv.push("--port".to_owned());
        argv.push(port.to_string());
    }
    Ok(argv)
}

pub(crate) fn build_watcher_enable_args(
    arguments: &serde_json::Map<String, Value>,
) -> Result<Vec<String>> {
    let watcher = arguments
        .get("watcher")
        .and_then(Value::as_str)
        .context("watcher_enable requires `watcher`")?;
    let mut argv = vec![
        "automations".to_owned(),
        "enable".to_owned(),
        watcher.to_owned(),
    ];
    if let Some(mode) = arguments.get("mode").and_then(Value::as_str) {
        argv.push("--mode".to_owned());
        argv.push(mode.to_owned());
    }
    Ok(argv)
}

/// Inverse of [`rocm_engine_protocol::is_public_bind_host`], which owns the
/// policy so `rocm` and `rocmd` never classify the same host differently.
fn is_loopback_host(host: &str) -> bool {
    !rocm_engine_protocol::is_public_bind_host(host)
}

fn system_prefix_requires_ack(prefix: &std::path::Path) -> bool {
    match rocm_core::runtime_home_dir() {
        Some(home) => !rocm_core::runtime_path_is_same_or_inside(prefix, &home),
        None => true,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_read_only_verbs_are_allowed_and_mutating_ones_are_not() {
        let allow = |args: &[&str]| {
            let owned = args.iter().map(|a| (*a).to_owned()).collect::<Vec<_>>();
            super::ensure_rocm_command_is_read_only(&owned)
        };

        // These read: the local tailnet, another machine's state, sessions that
        // already exist. Rejecting them made the whole family unusable here even
        // though none of them change anything.
        for args in [
            &["remote", "targets"][..],
            &["remote", "targets", "--tag", "gpu"][..],
            &["remote", "doctor", "gpu-box"][..],
            &["remote", "status"][..],
        ] {
            allow(args).unwrap_or_else(|error| panic!("{args:?} should be read-only: {error:#}"));
        }

        // These start, publish or tear down, so they go through approval.
        for args in [
            &["remote", "serve", "gpu-box", "a-model"][..],
            &["remote", "attach", "sess"][..],
            &["remote", "stop", "sess"][..],
            &["remote"][..],
        ] {
            assert!(allow(args).is_err(), "{args:?} must not be read-only");
        }
    }

    #[test]
    fn rocm_command_helper_allows_only_read_only_rocm_commands() -> Result<()> {
        let status_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["rocm", "comfy", "status"]
            })
            .as_object()
            .expect("json object"),
        )?;
        assert_eq!(status_args, vec!["comfyui".to_owned(), "status".to_owned()]);
        ensure_rocm_command_is_read_only(&status_args).expect("ComfyUI status should be read-only");

        let log_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["comfyui", "logs"]
            })
            .as_object()
            .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&log_args).expect("ComfyUI logs should be read-only");

        let install_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["comfyui", "install"]
            })
            .as_object()
            .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&install_args)
            .expect_err("ComfyUI install must go through approval");
        assert!(error.to_string().contains("approval UI"));

        let shell_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["powershell", "-Command", "whoami"]
            })
            .as_object()
            .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&shell_args)
            .expect_err("non-rocm shell commands should be rejected");
        assert!(error.to_string().contains("approval UI"));
        Ok(())
    }

    #[test]
    fn rocm_command_helper_treats_setup_status_as_read_only_and_reset_as_mutating() -> Result<()> {
        // Mirrors the bin's rocm_command classifier so `setup status` is read-only
        // on every binary's tool surface while `setup reset` stays approval-gated.
        let bare_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup"] })
                .as_object()
                .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&bare_args).expect("bare setup should be read-only");

        let status_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup", "status"] })
                .as_object()
                .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&status_args).expect("setup status should be read-only");

        let reset_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup", "reset"] })
                .as_object()
                .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&reset_args)
            .expect_err("setup reset must go through approval");
        assert!(error.to_string().contains("approval UI"));
        Ok(())
    }

    #[test]
    fn rocm_command_helper_treats_runtimes_uninstall_dry_run_as_read_only() -> Result<()> {
        // Mirrors the bin's chat_rocm_command_action_from_args classifier so a
        // dry-run preview stays read-only on every binary's tool surface while
        // an actual uninstall/remove still requires approval.
        for verb in ["uninstall", "remove"] {
            let dry_run_args = normalized_rocm_command_args(
                serde_json::json!({ "args": ["runtimes", verb, "--dry-run"] })
                    .as_object()
                    .expect("json object"),
            )?;
            ensure_rocm_command_is_read_only(&dry_run_args)
                .unwrap_or_else(|_| panic!("runtimes {verb} --dry-run should be read-only"));

            let mutating_args = normalized_rocm_command_args(
                serde_json::json!({ "args": ["runtimes", verb] })
                    .as_object()
                    .expect("json object"),
            )?;
            let error = match ensure_rocm_command_is_read_only(&mutating_args) {
                Ok(()) => panic!("runtimes {verb} without --dry-run must go through approval"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("approval UI"));
        }
        Ok(())
    }

    #[test]
    fn storage_report_is_read_only_but_removal_is_not() -> Result<()> {
        for args in [vec!["storage"], vec!["storage", "report"]] {
            let normalized = normalized_rocm_command_args(
                serde_json::json!({ "args": args })
                    .as_object()
                    .expect("json object"),
            )?;
            ensure_rocm_command_is_read_only(&normalized)
                .unwrap_or_else(|_| panic!("storage {args:?} only measures folders"));
        }

        for verb in ["remove-old-installs", "remove-downloads"] {
            let normalized = normalized_rocm_command_args(
                serde_json::json!({ "args": ["storage", verb] })
                    .as_object()
                    .expect("json object"),
            )?;
            let error = ensure_rocm_command_is_read_only(&normalized)
                .expect_err("storage removal must go through approval");
            assert!(error.to_string().contains("approval UI"));
        }
        Ok(())
    }

    #[test]
    fn launch_server_rejects_public_bind_without_ack() {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
        ]);
        let error = build_launch_server_args(&arguments).unwrap_err();
        assert!(
            error.to_string().contains("allow_public_bind=true"),
            "{error:#}"
        );
    }

    #[test]
    fn launch_server_forwards_public_bind_ack() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
            ("allow_public_bind".to_owned(), Value::Bool(true)),
            ("port".to_owned(), Value::Number(11999.into())),
        ]);
        let args = build_launch_server_args(&arguments)?;
        assert!(args.contains(&"--allow-public-bind".to_owned()));
        Ok(())
    }

    #[test]
    fn launch_server_requires_an_explicit_port_for_non_canonical_loopback_hosts() {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("localhost".to_owned())),
        ]);
        let error = build_launch_server_args(&arguments).unwrap_err();
        assert!(error.to_string().contains("explicit `port`"), "{error:#}");
    }

    #[test]
    fn launch_server_rejects_a_malformed_port_instead_of_choosing_one_automatically() {
        // Each value would be dropped by a lenient argv builder and leave the
        // launch to automatic port selection on a host that cannot do it.
        for (port, expected) in [
            (Value::String("8000".to_owned()), "must be an integer"),
            (Value::Number(0.into()), "between 1 and 65535"),
            (Value::Number(65536.into()), "between 1 and 65535"),
        ] {
            let arguments = serde_json::Map::from_iter([
                ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
                ("host".to_owned(), Value::String("localhost".to_owned())),
                ("port".to_owned(), port.clone()),
            ]);
            let error = build_launch_server_args(&arguments).unwrap_err();
            assert!(error.to_string().contains(expected), "{port}: {error:#}");
        }
    }

    #[test]
    fn launch_server_treats_a_null_port_as_automatic() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("port".to_owned(), Value::Null),
        ]);
        let args = build_launch_server_args(&arguments)?;
        assert!(!args.contains(&"--port".to_owned()), "{args:?}");
        Ok(())
    }

    #[test]
    fn launch_server_allows_localhost_with_an_explicit_port() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("localhost".to_owned())),
            ("port".to_owned(), Value::Number(11999.into())),
        ]);
        let args = build_launch_server_args(&arguments)?;
        assert!(args.contains(&"11999".to_owned()));
        Ok(())
    }

    #[test]
    fn install_sdk_rejects_system_prefix_without_ack() {
        let arguments = serde_json::Map::from_iter([(
            "prefix".to_owned(),
            Value::String("/opt/rocm".to_owned()),
        )]);
        let error = build_install_sdk_args(&arguments, false).unwrap_err();
        assert!(
            error.to_string().contains("allow_system_prefix=true"),
            "{error:#}"
        );
    }

    /// The test above only ever hands `system_prefix_requires_ack` an
    /// already-canonical path, so it cannot catch the bug this crate's fix
    /// addresses: a `..`-respelled prefix that escapes `$HOME` used to compare
    /// equal to a path still inside it (`Path::ancestors()` treats `..` as an
    /// ordinary component), so acknowledgement was never required. Drive the
    /// same check with a prefix built by walking `..` out of the real home
    /// directory, which is exactly the shape the original bug let through.
    #[test]
    #[cfg(unix)]
    fn install_sdk_rejects_system_prefix_reached_by_escaping_home() {
        let home = rocm_core::runtime_home_dir().expect("a home directory");
        let escaped_prefix = format!("{}/../../usr", home.display());

        let arguments =
            serde_json::Map::from_iter([("prefix".to_owned(), Value::String(escaped_prefix))]);
        let error = build_install_sdk_args(&arguments, false).unwrap_err();
        assert!(
            error.to_string().contains("allow_system_prefix=true"),
            "{error:#}"
        );
    }

    /// The `install_sdk` MCP tool spawns `rocm` with null stdin, so a real
    /// install over an active default managed runtime would hit the approval
    /// gate's non-interactive refusal and bail asking for a flag no MCP caller
    /// can pass. The real-install argv must therefore carry the consent flag;
    /// the dry-run argv must not, because a dry run never reaches the gate and
    /// the flag there would claim an approval the caller did not give.
    ///
    /// It must be `--approve-replacing-active-default` and never `--yes`:
    /// `--yes` additionally approves running `sudo` for required system
    /// packages, and a null-stdin spawn has no terminal on which that password
    /// prompt could be answered.
    #[test]
    fn install_sdk_real_install_args_approve_only_the_runtime_replacement() -> Result<()> {
        let arguments = serde_json::Map::new();

        let real = build_install_sdk_args(&arguments, false)?;
        assert!(
            real.contains(&"--approve-replacing-active-default".to_owned()),
            "real install argv must approve the replacement for the null-stdin spawn: {real:?}"
        );
        assert!(
            !real.contains(&"--yes".to_owned()),
            "real install argv must not grant the system-package consent it cannot answer: {real:?}"
        );
        assert!(
            !real.contains(&"--dry-run".to_owned()),
            "real install argv must not be a dry run: {real:?}"
        );

        let dry = build_install_sdk_args(&arguments, true)?;
        assert!(
            !dry.contains(&"--approve-replacing-active-default".to_owned())
                && !dry.contains(&"--yes".to_owned()),
            "dry-run argv must not carry a consent flag: {dry:?}"
        );
        assert!(
            dry.contains(&"--dry-run".to_owned()),
            "dry-run argv must carry --dry-run: {dry:?}"
        );
        Ok(())
    }

    #[test]
    fn install_sdk_forwards_requested_build_date_and_rejects_conflict() -> Result<()> {
        let arguments = serde_json::Map::from_iter([(
            "build_date".to_owned(),
            Value::String("2026-06-05".to_owned()),
        )]);
        let argv = build_install_sdk_args(&arguments, true)?;
        assert_eq!(
            argv,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--build-date".to_owned(),
                "2026-06-05".to_owned(),
                "--dry-run".to_owned(),
            ]
        );

        let conflicting = serde_json::Map::from_iter([
            (
                "version".to_owned(),
                Value::String("7.13.0a20260605".to_owned()),
            ),
            (
                "build_date".to_owned(),
                Value::String("2026-06-05".to_owned()),
            ),
        ]);
        let error = build_install_sdk_args(&conflicting, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("either `version` or `build_date`"));
        Ok(())
    }

    #[test]
    fn watcher_enable_builds_mode_args() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            (
                "watcher".to_owned(),
                Value::String("server-recover".to_owned()),
            ),
            ("mode".to_owned(), Value::String("contained".to_owned())),
        ]);
        let argv = build_watcher_enable_args(&arguments)?;
        assert_eq!(
            argv,
            vec![
                "automations".to_owned(),
                "enable".to_owned(),
                "server-recover".to_owned(),
                "--mode".to_owned(),
                "contained".to_owned()
            ]
        );
        Ok(())
    }
}
