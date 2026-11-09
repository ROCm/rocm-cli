// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The background launch path: [`start_managed_service`] and the plain,
//! line-by-line rendering of its [`ManagedLaunchReport`].

use super::endpoint_auth::drop_orphaned_endpoint_key_on_already_running;
use super::readiness::{
    died_during_wait, record_wait_outcome, service_pid_is_alive, wait_for_record_ready,
    wait_on_existing_service,
};
#[cfg(feature = "e2e-oom-fault-injection")]
use super::readiness::{e2e_simulate_oom_launch, simulate_oom_managed_launch};
use super::spawn::{ManagedLaunchReport, ManagedSpawn, spawn_managed_engine_child};
use crate::serve_port;
use crate::{record_cli_audit_event, status_for_readiness};
use anyhow::Result;
use rocm_core::format_http_base_url;
use rocm_core::{AppPaths, EndpointReadiness};
use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint, ResolveModelResponse};
use std::fmt::Write as _;
#[cfg(windows)]
use std::thread;
use std::time::Duration;

/// Launch a managed service in the background: run the locked allocation and
/// spawn transaction, release the launch lock once the record is published,
/// then wait for HTTP readiness.
///
/// Drops the endpoint key `serve()` freshly staged for this attempt if the
/// launch fails before any `ManagedServiceRecord` exists to own it (port
/// reserved/in use, range exhausted, other bind error) — such a failure
/// returns before `abandon_failed_managed_launch` ever runs, so nothing else
/// would clear it. Scoped to the spawn call only, mirroring
/// [`run_attached_service`](super::run_attached_service)'s identical guard: a later failure (the readiness
/// wait, `record.write()`) leaves a record a live child actually owns, so it
/// must not also wipe that key.
#[allow(clippy::too_many_arguments)]
pub(crate) fn start_managed_service(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port_request: serve_port::PortRequest,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    endpoint_api_key: Option<&str>,
    launch_lock: rocm_core::FileLock,
    require_api_key: bool,
    gpu_refusal_waived_for_reuse: bool,
    wait_until_ready: bool,
    on_wait_begin: &mut dyn FnMut(&str, u16),
    on_wait_tick: &mut dyn FnMut(Duration),
) -> Result<ManagedLaunchReport> {
    #[cfg(feature = "e2e-oom-fault-injection")]
    if e2e_simulate_oom_launch() {
        return simulate_oom_managed_launch(
            paths,
            engine,
            service_id,
            requested_model,
            resolve,
            host,
            port_request
                .explicit()
                .unwrap_or(rocm_core::DEFAULT_LOCAL_PORT),
            device_policy,
            gpu_indices,
            runtime_id,
            env_id,
        );
    }
    let (mut record, child_pid) = match spawn_managed_engine_child(
        paths,
        engine,
        service_id,
        requested_model,
        resolve,
        host,
        port_request,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        engine_recipe,
        require_api_key,
        gpu_refusal_waived_for_reuse,
    )
    .inspect_err(|_| {
        drop_orphaned_endpoint_key_on_already_running(paths, service_id, endpoint_api_key);
    })? {
        ManagedSpawn::AlreadyRunning(report) => {
            // `--wait` applies to a service that is still starting too: the
            // natural retry after `rocm serve m` printed `starting`.
            drop(launch_lock);
            return wait_on_existing_service(
                paths,
                report,
                wait_until_ready,
                on_wait_begin,
                on_wait_tick,
            );
        }
        ManagedSpawn::Spawned { record, child_pid } => (*record, child_pid),
    };
    // The transaction resolved the request to one concrete port; everything
    // below — readiness, the endpoint URL, the audit line — uses that.
    let port = record.port;
    // The claiming service record is now persisted, so the selected GPU is
    // visible to any concurrent auto-selection. Release the launch lock before
    // the readiness wait below, which can block for many seconds — holding it
    // that long would needlessly serialize unrelated serves.
    drop(launch_lock);

    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));

    if wait_until_ready {
        on_wait_begin(service_id, port);
    }
    let readiness = wait_for_record_ready(
        paths,
        &record,
        wait_until_ready,
        endpoint_api_key,
        on_wait_tick,
    );
    if let Some(error) = died_during_wait(
        paths,
        service_id,
        wait_until_ready,
        readiness,
        service_pid_is_alive,
    ) {
        return Err(error);
    }
    let launch_status = status_for_readiness(readiness);
    record_wait_outcome(
        paths,
        &mut record,
        launch_status,
        readiness == EndpointReadiness::Serving,
    )?;
    let endpoint_url = format!("{}/v1", format_http_base_url(host, port));
    record_cli_audit_event(
        paths,
        "service",
        "managed_service_launch",
        "info",
        format!(
            "launched managed service engine={} model={} endpoint={} readiness={}",
            engine, resolve.canonical_model_id, endpoint_url, launch_status
        ),
        Some(service_id),
    );
    Ok(ManagedLaunchReport {
        service_id: service_id.to_owned(),
        endpoint_url,
        status: launch_status.to_owned(),
        already_running: false,
        child_pid: Some(child_pid),
        log_path: Some(record.log_path),
        manifest_path: Some(record.manifest_path),
    })
}

/// Reproduce the original plain, line-by-line managed-launch output. Used for
/// non-interactive output (piped, CI, the chat assistant's `serve --managed`),
/// where the animated summary is inappropriate. The interactive path renders the
/// summary table via [`crate::serve_summary`] instead.
pub(crate) fn print_managed_launch_plain(
    report: &ManagedLaunchReport,
    endpoint_api_key: Option<&str>,
) {
    if report.already_running {
        println!("managed service already running");
        println!("  service_id: {}", report.service_id);
        println!("  endpoint: {}", report.endpoint_url);
        println!("  status: {}", report.status);
        println!("  note: existing service detected; no second process spawned");
        return;
    }
    println!("managed service launched");
    println!("  service_id: {}", report.service_id);
    if let Some(child_pid) = report.child_pid {
        println!("  process_pid: {child_pid}");
    }
    println!("  endpoint: {}", report.endpoint_url);
    if let Some(key) = endpoint_api_key {
        // Intentional one-time display of a freshly generated API key to the
        // terminal so the user can copy it — the designed delivery channel
        // documented on `render_endpoint_client_config`, not a log. The tag below
        // is currently inert (Rust's CodeQL pack has no AlertSuppression.ql yet —
        // github/codeql#21637) but will start working once that lands, since the
        // tag must be the single line immediately before the flagged code.
        print!(
            // codeql[rust/cleartext-logging]
            "{}",
            render_endpoint_client_config(&report.endpoint_url, key)
        );
    }
    if let Some(log_path) = report.log_path.as_deref() {
        println!("  log_path: {}", log_path.display());
    }
    if let Some(manifest_path) = report.manifest_path.as_deref() {
        println!("  manifest_path: {}", manifest_path.display());
    }
    println!("  readiness: {}", report.status);
}

/// The client configuration to show when a `--wait` begins, so a key generated for
/// a public bind reaches the user even if the wait is interrupted or fails.
/// `None` unless the wait is on the launch that key belongs to: an attach to an
/// already-starting server (`waiting_on` differs from `service_id`) answers to
/// its own key, which is not shown here.
pub(crate) fn wait_begin_client_config(
    waiting_on: &str,
    service_id: &str,
    host: &str,
    port: u16,
    api_key: Option<&str>,
) -> Option<String> {
    let key = api_key.filter(|_| waiting_on == service_id)?;
    let endpoint_url = format!("{}/v1", format_http_base_url(host, port));
    Some(render_endpoint_client_config(&endpoint_url, key))
}

/// Render the one-time secure client configuration for a public, authenticated
/// endpoint. This is the *intended* channel for delivering the key to the user
/// (unlike logs/status, which must never contain it) — it prints the key once at
/// launch alongside a ready-to-use example. Callers only invoke this for a
/// non-loopback bind that generated/received a key.
pub(super) fn render_endpoint_client_config(endpoint_url: &str, api_key: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "  api key: {api_key}");
    let _ = writeln!(
        out,
        "  note: this key is shown only now - clients must send `Authorization: Bearer <key>`"
    );
    let _ = writeln!(
        out,
        "  example: curl -H \"Authorization: Bearer {api_key}\" {endpoint_url}/models"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint_keys;
    #[cfg(not(windows))]
    use crate::service_launch::run_attached_service;
    #[cfg(not(windows))]
    use crate::service_launch::spawn::TEST_SPAWN_HOOK;
    use crate::tests::test_paths;
    use rocm_core::ManagedServiceRecord;
    use std::fs;
    #[cfg(not(windows))]
    use std::io;

    #[test]
    fn wait_begin_shows_the_key_only_for_the_launch_it_belongs_to() {
        let own = wait_begin_client_config("svc-new", "svc-new", "0.0.0.0", 8000, Some("k-secret"))
            .expect("own launch shows its key");
        // No `{own}` in the failure messages: the text carries the dummy key,
        // and formatting it into an assert is what CodeQL flags as logging it.
        assert!(own.contains("api key: k-secret"), "key line missing");
        assert!(
            own.contains("http://0.0.0.0:8000/v1/models"),
            "models url missing"
        );
        // Attached to an already-starting server: its key is not this one.
        assert!(
            wait_begin_client_config("svc-old", "svc-new", "0.0.0.0", 8000, Some("k-secret"))
                .is_none()
        );
        // Loopback binds have no key to show.
        assert!(wait_begin_client_config("svc-new", "svc-new", "127.0.0.1", 8000, None).is_none());
    }

    #[test]
    fn endpoint_client_config_shows_key_once_with_bearer_guidance() {
        let rendered = render_endpoint_client_config("http://0.0.0.0:11435/v1", "secret-123");
        // Dummy literal exercising the intentional one-time key display documented
        // on `render_endpoint_client_config`; not a real credential or a log write.
        // codeql[rust/cleartext-logging]
        assert!(rendered.contains("secret-123"), "{rendered}");
        // codeql[rust/cleartext-logging]: see rationale above.
        assert!(rendered.contains("Authorization: Bearer"), "{rendered}");
        // codeql[rust/cleartext-logging]: see rationale above.
        assert!(rendered.contains("shown only now"), "{rendered}");
    }

    #[test]
    fn failed_port_reservation_leaves_no_orphaned_endpoint_key() -> Result<()> {
        // serve() stores the new attempt's endpoint key before calling
        // start_managed_service. If spawn_managed_engine_child fails inside
        // resolve_managed_port_in_transaction (explicit port already reserved
        // by another live service), it returns before any
        // ManagedServiceRecord exists for this attempt, so
        // abandon_failed_managed_launch never runs. Driving the real function
        // end to end (not a simulated call to
        // drop_orphaned_endpoint_key_on_already_running) proves
        // `start_managed_service`'s own `.inspect_err` actually wires the cleanup, mirroring how
        // attached_launch_failing_before_spawn_leaves_no_orphaned_endpoint_key
        // drives run_attached_service for the foreground path.
        let (root, paths) = test_paths("orphaned-key-on-reserved-port");
        paths.ensure()?;
        let mut other = ManagedServiceRecord::new(
            &paths,
            "lemonade-llama-1000",
            "lemonade",
            "llama",
            "llama-canonical",
            "127.0.0.1",
            11513,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        other.status = "ready".to_owned();
        other.engine_pid = Some(std::process::id());
        other.write()?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            served_model_id: None,
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };

        let service_id = "lemonade-qwen-2000";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "test-key")?;
        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let result = start_managed_service(
            &paths,
            "lemonade",
            service_id,
            "qwen",
            &resolve,
            "127.0.0.1",
            serve_port::PortRequest::Explicit(11513),
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            Some("test-key"),
            launch_lock,
            false,
            false,
            false,
            &mut |_, _| {},
            &mut |_elapsed| {},
        );
        let Err(error) = result else {
            let _ = fs::remove_dir_all(root);
            panic!("explicit port already reserved by a live service must be rejected")
        };
        assert!(
            error
                .to_string()
                .contains("already reserved by managed service"),
            "unexpected error: {error}"
        );

        assert!(
            endpoint_keys::endpoint_api_key(&paths, service_id).is_none(),
            "cleanup must drop the key this failed attempt staged before launching"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// The launch lock must still be held when the child is spawned: it is what
    /// serializes allocation against publication. Observed through the spawn
    /// hook, since releasing it early changes nothing else a test can see.
    #[cfg(not(windows))]
    #[test]
    fn start_and_attached_launches_hold_the_launch_lock_until_the_spawn() -> Result<()> {
        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            served_model_id: None,
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };
        for attached in [false, true] {
            let (root, paths) = test_paths("launch-lock-held-until-spawn");
            paths.ensure()?;
            let lock_path = paths.managed_launch_lock_path();
            let lock_was_held = std::rc::Rc::new(std::cell::Cell::new(false));
            let observed = lock_was_held.clone();
            TEST_SPAWN_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(std::rc::Rc::new(move |_command| {
                    observed.set(
                        rocm_core::FileLock::acquire_bounded(&lock_path, Duration::ZERO).is_err(),
                    );
                    Err(io::Error::other("spawn intercepted by test"))
                }));
            });
            let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
            let result = if attached {
                run_attached_service(
                    &paths,
                    "lemonade",
                    "lemonade-lock-attached",
                    "qwen",
                    &resolve,
                    "127.0.0.1",
                    serve_port::PortRequest::Auto,
                    &[],
                    None,
                    None,
                    None,
                    launch_lock,
                    false,
                    false,
                    |_, _| unreachable!("the spawn fails before a Ctrl-C could stop it"),
                )
            } else {
                start_managed_service(
                    &paths,
                    "lemonade",
                    "lemonade-lock-start",
                    "qwen",
                    &resolve,
                    "127.0.0.1",
                    serve_port::PortRequest::Auto,
                    &resolve.device_policy,
                    &[],
                    None,
                    None,
                    None,
                    None,
                    launch_lock,
                    false,
                    false,
                    false,
                    &mut |_, _| {},
                    &mut |_elapsed| {},
                )
                .map(|_| ())
            };
            TEST_SPAWN_HOOK.with(|hook| *hook.borrow_mut() = None);
            let _ = fs::remove_dir_all(root);
            assert!(
                format!("{:#}", result.expect_err("the intercepted spawn fails"))
                    .contains("spawn intercepted by test")
            );
            assert!(
                lock_was_held.get(),
                "the launch lock must be held at spawn (attached: {attached})"
            );
        }
        Ok(())
    }
}
