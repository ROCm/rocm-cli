// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The attached serve path (`--verbose`/`--foreground`): spawn the managed child
//! and stream its log in this terminal until the user detaches or stops it.

use super::endpoint_auth::drop_orphaned_endpoint_key_on_already_running;
use super::helper_autostart::ensure_background_helper_running_quiet;
use super::readiness::{
    StopWatch, managed_engine_exit_failure_detail, managed_service_stop_was_requested,
    mark_service_failed,
};
use super::spawn::{ManagedSpawn, spawn_managed_engine_child};
use super::start::render_endpoint_client_config;
use crate::record_cli_audit_event;
use crate::serve_port;
use crate::service_hints;
use crate::services_text::{
    render_service_action_result, render_service_action_result_with_helper_note,
    service_stop_unconfirmed,
};
use anyhow::{Context, Result, bail};
use rocm_core::AppPaths;
use rocm_core::format_http_base_url;
use rocm_engine_protocol::ResolveModelResponse;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::thread;
use std::time::Duration;

/// What ended an attached (`--verbose`/`--foreground`) streaming session. Kept
/// as a plain enum, separate from any terminal I/O, so the follow-up action
/// (detach note vs. stop the server) is unit-testable without a TTY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachOutcome {
    /// Ctrl-D: leave the server running and hand the terminal back.
    Detach,
    /// Ctrl-C: stop the server, then hand the terminal back.
    Stop,
    /// The engine process exited on its own while we were streaming its log.
    ServerExited,
}

/// Attached serve path for `--verbose`/`--foreground`: spawn the engine as a
/// detached managed child (the same child the background path spawns) and stream
/// its log in this terminal. Unlike the old in-process foreground, the server
/// survives the session — Ctrl-D detaches and leaves it running, Ctrl-C stops it.
///
/// `stop_service` is how Ctrl-C stops the server: the caller passes the managed
/// stop, so the `service_launch` module does not import the stop path, which
/// itself imports its spawn helpers. It returns the same `stop_server` answer
/// [`finish_attached_stop`] reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_attached_service(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port_request: serve_port::PortRequest,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    endpoint_api_key: Option<&str>,
    launch_lock: rocm_core::FileLock,
    require_api_key: bool,
    gpu_refusal_waived_for_reuse: bool,
    stop_service: impl FnOnce(&AppPaths, &str) -> Result<serde_json::Value>,
) -> Result<()> {
    let spawn = spawn_managed_engine_child(
        paths,
        engine,
        service_id,
        requested_model,
        resolve,
        host,
        port_request,
        &resolve.device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        resolve.engine_recipe.as_ref(),
        require_api_key,
        gpu_refusal_waived_for_reuse,
    )
    // Mirrors `start_managed_service`'s cleanup: a pre-spawn
    // failure (port reserved/in use, range exhausted, other bind error) returns
    // before any service record exists, so nothing else clears the key just
    // stored for this attempt.
    .inspect_err(|_| {
        drop_orphaned_endpoint_key_on_already_running(paths, service_id, endpoint_api_key);
    })?;
    // The claiming record is persisted (or an existing service was found), so the
    // selected GPU is now visible to concurrent auto-selection. Release the launch
    // lock before streaming logs, which blocks for the whole attached session.
    drop(launch_lock);

    // Attached mode is a managed record too, so it shares the transaction's
    // resolved concrete port.
    let (
        service_id,
        log_path,
        engine_state_path,
        port,
        child_pid,
        supervisor_start_ticks,
        mut liveness,
    ) = match spawn {
        // A server for this engine+model is already live. Don't fight it for the
        // port — point the user at the existing one instead of tailing a log we
        // did not start.
        ManagedSpawn::AlreadyRunning(report) => {
            println!("model already being served");
            println!("  service_id: {}", report.service_id);
            println!("  endpoint: {}", report.endpoint_url);
            println!("  status: {}", report.status);
            println!("  logs: rocm services logs {}", report.service_id);
            println!(
                "  stop: {}",
                service_hints::service_stop_hint(&report.service_id)
            );
            drop_orphaned_endpoint_key_on_already_running(paths, service_id, endpoint_api_key);
            return Ok(());
        }
        ManagedSpawn::Spawned {
            record,
            child_pid,
            liveness,
        } => (
            service_id.to_owned(),
            record.log_path.clone(),
            record.engine_state_path.clone(),
            record.port,
            child_pid,
            record.supervisor_start_ticks,
            liveness,
        ),
    };

    // The child is a managed service that outlives this session once detached.
    // The background helper adds one thing for it: with the `server-recover`
    // watcher enabled, it health-checks managed servers and restarts failed ones.
    // Every duty of the helper is watcher-gated, so with no watcher enabled no
    // helper is started. Nothing else depends on it: a self-exited server's
    // record is settled by its own supervisor (`supervise_service` writes the
    // exit via `record_supervised_exit` in `apps/rocmd/src/supervise.rs`) and on
    // every read (`refresh_managed_service_runtime_liveness`), and the dashboard
    // has its own embedded daemon. Match the background ordering (spawn, then
    // ensure the helper) and keep it quiet so no status line breaks into the
    // log stream.
    ensure_background_helper_running_quiet(true)?;

    // The resolution detail (model, engine, runtime, GPU, warnings) was already
    // printed as the "serve plan" block in `serve()`; extend it with the launch
    // coordinates and the streaming hint rather than repeating it.
    let endpoint = format!("{}/v1", format_http_base_url(host, port));
    println!("  service_id: {service_id}");
    println!("  endpoint: {endpoint}");
    if let Some(key) = endpoint_api_key {
        // Same intentional one-time key display as `print_managed_launch_plain`
        // in `service_launch/start.rs`; see its rationale for why the tag below is
        // currently inert.
        // codeql[rust/cleartext-logging]
        print!("{}", render_endpoint_client_config(&endpoint, key));
    }
    println!("  streaming engine logs - Ctrl-D detaches (leaves it running), Ctrl-C stops it");
    println!();

    // Sampled on every liveness poll for the whole attached session: an external
    // `rocm services stop` clears its own marker once it confirms, so reading it
    // only after the child is gone can miss it and report a deliberate stop as a
    // crash. See `StopWatch`.
    let mut stop_watch = StopWatch::default();
    let outcome = stream_attached_logs(&log_path, || {
        stop_watch.sample(paths, &service_id);
        liveness.is_running()
    })?;
    let stop_seen_while_attached = stop_watch.stop_was_requested();
    println!();

    match outcome {
        AttachOutcome::Detach => {
            // Audit before printing: a closed stdout makes `println!` panic,
            // which must not lose the event.
            record_cli_audit_event(
                paths,
                "service",
                "serve_detach",
                "info",
                format!("detached from attached serve service_id={service_id} endpoint={endpoint}"),
                Some(&service_id),
            );
            println!("detached - server still running");
            println!("  service_id: {service_id}");
            println!("  endpoint: {endpoint}");
            println!("  list: rocm services");
            println!("  logs: rocm services logs {service_id}");
            println!("  stop: {}", service_hints::service_stop_hint(&service_id));
            Ok(())
        }
        AttachOutcome::Stop => {
            println!("stopping server...");
            let stop = stop_service(paths, &service_id);
            finish_attached_stop(
                paths,
                &service_id,
                child_pid,
                stop,
                rocm_core::terminate_process_tree,
                &mut io::stdout().lock(),
                &mut io::stderr().lock(),
            )
        }
        AttachOutcome::ServerExited => {
            // `Stop` above is only this attached session's *own* keystroke
            // reader (Ctrl-C) or its own fallback — it says nothing about a
            // stop triggered any other way. `rocm services stop <id>` from
            // another shell and `rocm dash`'s stop action both kill the pid
            // this loop is watching by signalling it directly, which lands
            // here, in `ServerExited`, not `Stop`. So an exited child alone
            // does not mean a crash (ROCMAI-541): check the manifest's stop
            // marker (`stop_requested_unix_ms`, set before any such external
            // stop signals anything — see
            // `stop_internal_managed_service_confirmed_with`) and its
            // `stopped` status before concluding one — see
            // `managed_service_stop_was_requested`'s doc comment for why
            // `rocmd`'s own restart-recovery watcher needs no such carve-out
            // here. Only when neither is present is this an unexplained exit
            // — i.e. a crash.
            if stop_seen_while_attached || managed_service_stop_was_requested(paths, &service_id) {
                println!("server process exited");
                println!("  service_id: {service_id}");
                println!("  recent logs: rocm services logs {service_id}");
                return Ok(());
            }
            // Mark both the manifest and the engine state file `failed` (the
            // engine process is gone and cannot do it for itself) and fail
            // loudly instead of the previous "server process exited" / exit 0.
            // The manifest write is a best-effort merge, not a full
            // `ManagedServiceRecord::write()`: this attached session's own copy
            // of the record was captured at spawn time and may be stale by now
            // (a concurrent restart could have replaced it), so only the one
            // field this crash verdict owns is touched — never a write-back
            // that could clobber a newer record with an older snapshot.
            //
            // Guarded on the manifest still naming the exact *supervisor*
            // process this session spawned and has been watching — `child_pid`
            // is `supervisor_pid`/`supervisor_start_ticks`'s subject, the
            // `__engine-serve-http` process this loop's `liveness` tracks and
            // whose exit is what landed us here. `engine_pid` is the wrong
            // identity to gate on: `refresh_from_engine_state` overwrites it
            // with whatever the engine itself reports as `server_pid`, which
            // for an engine like Lemonade that spawns its own child server is
            // *that* child's pid, not this loop's `child_pid` — so comparing
            // on `engine_pid` is false for the exact case this whole fix is
            // for (a Lemonade-backed server that was running and then died),
            // and the manifest would never get stamped. `supervisor_pid`/
            // `supervisor_start_ticks` are never touched by
            // `refresh_from_engine_state` and ARE rewritten by a legitimate
            // restart (`record_supervisor_identity` at the top of this same
            // spawn path), so comparing on them actually distinguishes "this
            // manifest still names the child we're watching" from "a restart
            // already replaced it". The engine state file is guarded the same
            // way: its own `status` is exactly what `refresh_from_engine_state`
            // reads back into a newer manifest, so leaving this write
            // unguarded would let the clobber this guard prevents on the
            // manifest simply round-trip back in through that file instead.
            let still_watching_this_supervisor =
                crate::services_text::read_managed_service_manifest(paths, &service_id).is_ok_and(
                    |(current, _)| {
                        current.supervisor_pid == child_pid
                            && current.supervisor_start_ticks == supervisor_start_ticks
                    },
                );
            if still_watching_this_supervisor {
                mark_service_failed(
                    &paths.service_manifest_path(&service_id),
                    &engine_state_path,
                );
            }
            bail!(
                "{}",
                managed_engine_exit_failure_detail(&service_id, &log_path)
            );
        }
    }
}

/// Restores cooked terminal mode when dropped, so [`stream_attached_logs`] leaves
/// the terminal usable on every exit path (normal return, `?` error, or panic).
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Map a key press (control modifier + lowercased character) to the attach
/// action it triggers, if any. Factored out of the raw-mode reader loop so the
/// Ctrl-D/Ctrl-C mapping is unit-testable without a terminal.
const fn detach_key_outcome(ctrl: bool, ch: char) -> Option<AttachOutcome> {
    if !ctrl {
        return None;
    }
    match ch {
        'c' => Some(AttachOutcome::Stop),
        'd' => Some(AttachOutcome::Detach),
        _ => None,
    }
}

/// Follow `log_path` in the terminal until the user presses Ctrl-D (detach) or
/// Ctrl-C (stop), or the engine process exits. Uses crossterm raw mode to
/// capture the keys directly (in raw mode Ctrl-C does not raise SIGINT, so we see
/// it as a key event). When stdin is not a TTY (piped/CI), keystroke capture is
/// impossible, so we follow the log until the process exits instead.
///
/// `is_alive` is injected rather than this calling `process_is_running` on a
/// bare pid directly, so the real caller can back it with
/// [`rocm_core::ManagedChildLiveness`] — a `kill(pid, 0)` probe alone succeeds
/// for a zombie this process has not reaped yet on Unix (ROCMAI-541; see that
/// type's doc comment).
fn stream_attached_logs(
    log_path: &Path,
    mut is_alive: impl FnMut() -> bool,
) -> Result<AttachOutcome> {
    use std::io::IsTerminal as _;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    if !std::io::stdin().is_terminal() {
        return stream_attached_logs_no_tty(log_path, is_alive);
    }

    // Enter raw mode *before* spawning the key reader. In raw mode Ctrl-C arrives
    // as a key event instead of SIGINT; if the reader started first, a Ctrl-C in
    // that window would kill the CLI outright (leaving the detached child alive
    // but printing no detach/stop message). The guard restores cooked mode on
    // every exit path (normal return, `?` error, panic).
    crossterm::terminal::enable_raw_mode().context("failed to enter raw terminal mode")?;
    let _raw_guard = RawModeGuard;

    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<AttachOutcome>();

    let reader_stop = Arc::clone(&stop);
    let reader = thread::spawn(move || {
        use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
        while !reader_stop.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(Event::Key(key)) if key.kind != KeyEventKind::Release => {
                        let outcome = match key.code {
                            KeyCode::Char(ch) => detach_key_outcome(
                                key.modifiers.contains(KeyModifiers::CONTROL),
                                ch.to_ascii_lowercase(),
                            ),
                            _ => None,
                        };
                        if let Some(outcome) = outcome {
                            let _ = tx.send(outcome);
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });

    let mut stdout = io::stdout();
    let mut log_reader: Option<io::BufReader<fs::File>> = None;
    let mut line = String::new();
    let outcome = loop {
        if log_reader.is_none() {
            log_reader = fs::File::open(log_path).ok().map(io::BufReader::new);
        }
        if let Some(reader) = log_reader.as_mut() {
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        // Raw mode disables the terminal's own \n -> \r\n
                        // translation, so emit an explicit carriage return to
                        // keep the log left-aligned instead of stair-stepping.
                        let _ = write!(stdout, "{}\r\n", line.trim_end_matches('\n'));
                        let _ = stdout.flush();
                    }
                    Err(_) => break,
                }
            }
        }

        if let Ok(signal) = rx.try_recv() {
            break signal;
        }
        if !is_alive() {
            break AttachOutcome::ServerExited;
        }
        thread::sleep(Duration::from_millis(150));
    };

    stop.store(true, Ordering::Relaxed);
    let _ = reader.join();
    Ok(outcome)
}

/// Non-interactive fallback for [`stream_attached_logs`]: no keystroke capture,
/// so just follow the log until the (detached) engine process exits. A Ctrl-C
/// here delivers SIGINT to this process and leaves the managed server running.
///
/// This is the *only* possible outcome of a piped/non-TTY attached session
/// (there is no keystroke reader to ever produce `Detach`/`Stop` here), so
/// every externally-triggered stop during such a session reaches here as
/// `ServerExited` — making the caller's stop-marker check
/// (`managed_service_stop_was_requested`) load-bearing for every scripted or
/// CI `rocm serve --verbose` invocation, not just an edge case.
fn stream_attached_logs_no_tty(
    log_path: &Path,
    mut is_alive: impl FnMut() -> bool,
) -> Result<AttachOutcome> {
    let mut stdout = io::stdout();
    let mut log_reader: Option<io::BufReader<fs::File>> = None;
    let mut line = String::new();
    loop {
        if log_reader.is_none() {
            log_reader = fs::File::open(log_path).ok().map(io::BufReader::new);
        }
        if let Some(reader) = log_reader.as_mut() {
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        let _ = write!(stdout, "{line}");
                        let _ = stdout.flush();
                    }
                    Err(_) => break,
                }
            }
        }
        if !is_alive() {
            return Ok(AttachOutcome::ServerExited);
        }
        thread::sleep(Duration::from_millis(200));
    }
}

/// Finish the attached `rocm serve` Ctrl-C stop once the managed stop has
/// answered: what is printed where, what is signalled, what is audited, and
/// how the command exits. Everything but the stop itself and the real
/// signal is here, so a test can drive it.
///
/// - Confirmed: the report, with the background-helper note `rocm services
///   stop` also prints, goes to `out` and the command succeeds.
/// - Unconfirmed: the same report `rocm services stop` gives, check lines
///   included, becomes the error, so it reaches stderr and the command exits
///   non-zero. Nothing is signalled on the stop's behalf: the stop has already
///   done what it safely could.
/// - The stop itself failed: `signal_child` gets the child this session
///   launched, so Ctrl-C never leaves the model orphaned when the stop path
///   breaks. The stop's error, and the signal's if that fails too, are noted on
///   `err`, and the audit event records whether the signal worked. The command
///   still exits 0, as it always has here.
fn finish_attached_stop(
    paths: &AppPaths,
    service_id: &str,
    child_pid: u32,
    stop: Result<serde_json::Value>,
    signal_child: impl FnOnce(u32) -> Result<()>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<()> {
    match stop {
        Ok(result) if service_stop_unconfirmed(&result) => {
            record_cli_audit_event(
                paths,
                "service",
                "serve_stop",
                "error",
                format!("stop of attached serve not confirmed service_id={service_id}"),
                Some(service_id),
            );
            bail!(
                "{}",
                render_service_action_result("stop_server", &result).trim_end()
            )
        }
        Ok(result) => {
            record_cli_audit_event(
                paths,
                "service",
                "serve_stop",
                "info",
                format!("stopped attached serve service_id={service_id}"),
                Some(service_id),
            );
            write!(
                out,
                "{}",
                render_service_action_result_with_helper_note(paths, "stop_server", &result)
            )?;
            Ok(())
        }
        Err(error) => {
            let signal = signal_child(child_pid);
            // Record the audit event before writing to stderr, so a failed
            // write cannot skip it.
            let message = match &signal {
                Ok(()) => format!(
                    "stop failed; signaled attached serve child {child_pid} service_id={service_id}"
                ),
                Err(signal_error) => format!(
                    "stop failed; signaling attached serve child {child_pid} also failed: {signal_error:#} service_id={service_id}"
                ),
            };
            record_cli_audit_event(
                paths,
                "service",
                "serve_stop",
                if signal.is_ok() { "warn" } else { "error" },
                message,
                Some(service_id),
            );
            writeln!(err, "  note: {error:#}")?;
            if let Err(signal_error) = &signal {
                writeln!(
                    err,
                    "  note: signaling this session's server process {child_pid} directly also failed: {signal_error:#}"
                )?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint_keys;
    use crate::tests::test_paths;
    use rocm_core::ManagedServiceRecord;
    use rocm_engine_protocol::DevicePolicy;

    /// A `stop_server` answer whose one recorded process ended with `outcome`.
    fn stop_answer(confirmed: bool, outcome: &str) -> serde_json::Value {
        crate::services_text::internal_stop_server_value(serde_json::json!({
            "service_id": "svc-attached",
            "status": "ready",
            "signaled_pids": if outcome == "unverified" { vec![] } else { vec![4242] },
            "pid_outcomes": [{ "pid": 4242, "role": "supervisor_and_engine", "outcome": outcome }],
            // An attached serve on a public bind stores a key; an unconfirmed
            // stop keeps it.
            "endpoint_key_kept": !confirmed,
            "stopped": confirmed,
        }))
    }

    /// Run [`finish_attached_stop`] for `stop`, returning its result, what it
    /// wrote to stdout, the PID it signalled (if any) and the audit event.
    fn finish(
        name: &str,
        stop: Result<serde_json::Value>,
    ) -> (Result<()>, String, Option<u32>, serde_json::Value) {
        finish_signalling(name, stop, Ok(()))
    }

    /// [`finish`], with the direct signal answering `signal_result`.
    fn finish_signalling(
        name: &str,
        stop: Result<serde_json::Value>,
        signal_result: Result<()>,
    ) -> (Result<()>, String, Option<u32>, serde_json::Value) {
        let (result, out, _err, signalled, audit) = finish_capturing(name, stop, signal_result);
        (result, out, signalled, audit)
    }

    /// [`finish_signalling`], also returning what was written to stderr.
    fn finish_capturing(
        name: &str,
        stop: Result<serde_json::Value>,
        signal_result: Result<()>,
    ) -> (Result<()>, String, String, Option<u32>, serde_json::Value) {
        let (root, paths) = test_paths(name);
        paths.ensure().unwrap();
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut signalled = None;
        let result = finish_attached_stop(
            &paths,
            "svc-attached",
            777,
            stop,
            |pid| {
                signalled = Some(pid);
                signal_result
            },
            &mut out,
            &mut err,
        );
        let audit_text = fs::read_to_string(paths.audit_events_path()).expect("audit log");
        let audit = serde_json::from_str(audit_text.lines().last().expect("an audit event"))
            .expect("audit json");
        let _ = fs::remove_dir_all(root);
        (
            result,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
            signalled,
            audit,
        )
    }

    #[test]
    fn attached_ctrl_c_reports_a_confirmed_stop_on_stdout() {
        let (result, out, signalled, audit) =
            finish("attached-confirmed", Ok(stop_answer(true, "graceful")));
        result.expect("a confirmed stop succeeds");
        assert!(out.contains("Local server stopped"), "{out}");
        assert_eq!(signalled, None);
        assert_eq!(audit["level"], "info", "{audit}");
        assert_eq!(
            audit["message"],
            "stopped attached serve service_id=svc-attached"
        );
    }

    #[test]
    fn attached_ctrl_c_fails_an_unconfirmed_stop_like_services_stop_and_signals_nothing() {
        // The same report `rocm services stop` gives, as the error (stderr,
        // non-zero exit), naming no retry; nothing is signalled on the stop's
        // behalf and nothing reaches stdout.
        let answer = stop_answer(false, "timed_out");
        let (result, out, signalled, audit) = finish("attached-timed-out", Ok(answer.clone()));
        let error = format!("{:#}", result.expect_err("an unconfirmed stop fails"));
        assert_eq!(
            error,
            render_service_action_result("stop_server", &answer).trim_end(),
            "the report is `rocm services stop`'s own"
        );
        assert!(
            error.contains(
                "process 4242 (supervisor_and_engine) or a process under it did not exit"
            ),
            "{error}"
        );
        assert!(
            error.lines().any(|line| line
                == "  kept: the server record, which still names its processes, and its endpoint key, which the server may still be enforcing"),
            "{error}"
        );
        crate::services_text::tests::assert_no_retry_line(&error);
        assert!(out.is_empty(), "nothing on stdout: {out}");
        assert_eq!(signalled, None, "an unconfirmed stop signals nothing more");
        assert_eq!(audit["level"], "error", "{audit}");
        assert_eq!(
            audit["message"],
            "stop of attached serve not confirmed service_id=svc-attached"
        );
    }

    #[test]
    fn attached_ctrl_c_names_a_check_for_an_unverified_process() {
        let (result, out, signalled, audit) =
            finish("attached-unverified", Ok(stop_answer(false, "unverified")));
        let error = format!("{:#}", result.expect_err("an unverified stop fails"));
        assert!(error.contains("not signaled: process 4242"), "{error}");
        assert!(
            error.contains(&format!(
                "check: `{}`",
                crate::services_text::process_check_command(4242)
            )),
            "{error}"
        );
        crate::services_text::tests::assert_no_retry_line(&error);
        assert!(out.is_empty(), "{out}");
        assert_eq!(signalled, None);
        assert_eq!(audit["level"], "error", "{audit}");
    }

    #[test]
    fn attached_ctrl_c_signals_its_own_child_when_the_stop_itself_fails() {
        let (result, out, err, signalled, audit) = finish_capturing(
            "attached-stop-error",
            Err(anyhow::anyhow!("managed service `svc-attached` not found")
                .context("failed to stop svc-attached")),
            Ok(()),
        );
        result.expect("a failed stop falls back and still exits cleanly, as before");
        assert_eq!(signalled, Some(777));
        assert_eq!(
            err, "  note: failed to stop svc-attached: managed service `svc-attached` not found\n",
            "the stop's error goes to stderr, and a signal that worked adds nothing"
        );
        assert!(out.is_empty(), "nothing on stdout: {out}");
        assert_eq!(audit["level"], "warn", "{audit}");
        assert_eq!(
            audit["message"],
            "stop failed; signaled attached serve child 777 service_id=svc-attached"
        );

        // The direct signal failing too is printed on stderr and recorded, with
        // its error.
        let (result, _, err, signalled, audit) = finish_capturing(
            "attached-stop-error-signal-fails",
            Err(anyhow::anyhow!("managed service `svc-attached` not found")),
            Err(anyhow::anyhow!("no such process")),
        );
        result.expect("still exits 0, as on main");
        assert_eq!(
            err,
            "  note: managed service `svc-attached` not found\n  \
             note: signaling this session's server process 777 directly also failed: no such process\n",
            "both notes go to stderr"
        );
        assert_eq!(signalled, Some(777));
        assert_eq!(audit["level"], "error", "{audit}");
        assert_eq!(
            audit["message"],
            "stop failed; signaling attached serve child 777 also failed: no such process service_id=svc-attached"
        );
    }

    /// A writer whose every write fails, standing in for a closed stdout/stderr.
    struct BrokenWriter;

    impl Write for BrokenWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn attached_ctrl_c_records_the_audit_event_even_when_writing_the_outcome_fails() {
        // One case per branch that writes after deciding: a confirmed stop
        // (stdout) and a stop that itself failed (stderr notes). The write
        // error is returned, but the audit event is already on disk.
        let cases: [(&str, Result<serde_json::Value>, &str); 2] = [
            (
                "attached-broken-out",
                Ok(stop_answer(true, "graceful")),
                "stopped attached serve service_id=svc-attached",
            ),
            (
                "attached-broken-err",
                Err(anyhow::anyhow!("managed service `svc-attached` not found")),
                "stop failed; signaled attached serve child 777 service_id=svc-attached",
            ),
        ];
        for (name, stop, expected_message) in cases {
            let (root, paths) = test_paths(name);
            paths.ensure().unwrap();
            let result = finish_attached_stop(
                &paths,
                "svc-attached",
                777,
                stop,
                |_| Ok(()),
                &mut BrokenWriter,
                &mut BrokenWriter,
            );
            let audit_text = fs::read_to_string(paths.audit_events_path()).unwrap_or_default();
            let _ = fs::remove_dir_all(root);
            assert!(result.is_err(), "{name}: the write error is surfaced");
            let audit: serde_json::Value = serde_json::from_str(
                audit_text
                    .lines()
                    .last()
                    .unwrap_or_else(|| panic!("{name}: no audit event recorded")),
            )
            .expect("audit json");
            assert_eq!(audit["message"], expected_message, "{name}: {audit}");
        }
    }

    #[test]
    fn detach_key_ctrl_d_detaches_ctrl_c_stops() {
        assert_eq!(detach_key_outcome(true, 'd'), Some(AttachOutcome::Detach));
        assert_eq!(detach_key_outcome(true, 'c'), Some(AttachOutcome::Stop));
    }

    #[test]
    fn detach_key_ignores_plain_and_unrelated_keys() {
        // Without the control modifier, `d`/`c` are ordinary log-scroll input.
        assert_eq!(detach_key_outcome(false, 'd'), None);
        assert_eq!(detach_key_outcome(false, 'c'), None);
        // Other control combos are not detach/stop triggers.
        assert_eq!(detach_key_outcome(true, 'q'), None);
        assert_eq!(detach_key_outcome(true, 'z'), None);
    }

    #[test]
    fn attached_launch_failing_before_spawn_leaves_no_orphaned_endpoint_key() -> Result<()> {
        // The background path's cleanup is the `.inspect_err` in
        // start_managed_service. run_attached_service is the other caller of
        // spawn_managed_engine_child and needs the identical .inspect_err()
        // cleanup, not a simulated one: drive the real function end to end.
        let (root, paths) = test_paths("orphaned-key-attached-reserved-port");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let mut other = ManagedServiceRecord::new(
            &paths,
            "lemonade-llama-1000",
            "lemonade",
            "llama",
            "llama-canonical",
            "127.0.0.1",
            11514,
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

        let service_id = "lemonade-qwen-attached-2000";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "test-key")?;
        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let result = run_attached_service(
            &paths,
            "lemonade",
            service_id,
            "qwen",
            &resolve,
            "127.0.0.1",
            serve_port::PortRequest::Explicit(11514),
            &[],
            None,
            None,
            Some("test-key"),
            launch_lock,
            false,
            false,
            |_, _| unreachable!("the launch fails before a Ctrl-C could stop it"),
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
            "a pre-spawn failure must drop the key this attempt staged before launching"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
