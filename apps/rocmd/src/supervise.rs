// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::{AppPaths, ManagedServiceRecord, daemon_binary_path};
use std::fs;
use std::path::Path;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    let Some(raw) = value else {
        return Ok(Vec::new());
    };
    match rocm_engine_protocol::GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg)? {
        rocm_engine_protocol::GpuSelection::Auto => Ok(Vec::new()),
        rocm_engine_protocol::GpuSelection::Index(index) => Ok(vec![index]),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_service(
    paths: &AppPaths,
    service_id: String,
    engine: String,
    model_ref: String,
    canonical_model_id: String,
    runtime_id: Option<String>,
    env_id: Option<String>,
    host: String,
    port: u16,
    device_policy: String,
    gpu: Option<String>,
    engine_recipe_json: Option<String>,
) -> Result<()> {
    paths.ensure()?;
    fs::create_dir_all(paths.engine_logs_dir(&engine))?;
    fs::create_dir_all(paths.engine_state_dir(&engine))?;
    fs::create_dir_all(paths.services_dir())?;

    let gpu_indices = parse_gpu_indices_arg(gpu.as_deref())?;
    let _ = daemon_binary_path();

    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        engine.clone(),
        model_ref,
        canonical_model_id.clone(),
        host,
        port,
        "managed",
        std::process::id(),
        runtime_id.clone(),
        env_id.clone(),
        Some(device_policy.clone()),
    );
    // `new` recorded this PID; this pairs it with its start-time token.
    record.record_supervisor_identity(std::process::id());
    record.gpu_indices = gpu_indices;
    record.engine_recipe_json = engine_recipe_json.clone();
    // Carried over from whatever is on disk. `ManagedServiceRecord::new` starts
    // this false, so rebuilding a record here without restoring it would not
    // just skip the check now — it would write the weakened record back and
    // disarm every later `rocm services restart` as well.
    //
    // Propagated, not defaulted. This read arms the guard below, so it is not
    // best-effort the way an identical-looking call feeding a printed warning
    // would be. `load_managed_services` already *skips* unparseable records, so
    // an `Err` here is a real I/O failure — and a missing directory is `Ok`
    // anyway. Swallowing it would say "no service ever required a key", the
    // key-file fallback is false precisely when a service has been stopped, and
    // the weakened record would then be written back at the bottom of this
    // function. That is the outcome the comment above says must not happen.
    let existing = crate::persistence::load_managed_services(paths)
        .context(
            "could not read the service registry to check whether this service requires an \
             endpoint API key; refusing to recover it rather than assume it does not",
        )?
        .into_iter()
        .find(|existing| existing.service_id == record.service_id);
    let previously_required = existing
        .as_ref()
        .is_some_and(|existing| existing.requires_api_key);
    // Only what the registry recorded. The `|| key-file-is-present` clause that
    // used to be here re-derived the flag the same way `spawn_managed_engine_child`
    // did, and was wrong for the same reason: a public bind always has a key file
    // whether or not auth was ever demanded, so recovery re-armed this on services
    // that never asked for it and refused them with the wrong remediation.
    record.requires_api_key = previously_required;
    // A standing stop request is carried over too. The restart that launched
    // this supervisor clears the marker before it does so, so one found here was
    // set by a stop issued since — and `ManagedServiceRecord::new` starting it
    // at `None` would quietly withdraw that stop and hand the service straight
    // back to the daemon's recovery.
    record.stop_requested_unix_ms = existing.and_then(|existing| existing.stop_requested_unix_ms);
    // Refuse a keyless public respawn before the manifest write, so a refused
    // attempt leaves the recorded restart_count and timestamps intact instead of
    // clobbering them with a record no live process will ever back. The spawn
    // site below re-checks against the key actually threaded onto the command.
    crate::engine_ipc::ensure_public_service_has_endpoint_key(
        &record.host,
        rocm_engine_protocol::endpoint_key_file_if_present(paths, &record.service_id)
            .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
            .is_some(),
        record.requires_api_key,
    )?;
    record.write()?;

    let log_file = fs::File::create(&record.log_path)
        .with_context(|| format!("failed to create {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    let rocm_binary =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    let mut command = ProcessCommand::new(rocm_binary);
    command
        .args(engine_serve_http_args(
            &engine,
            &record.service_id,
            &canonical_model_id,
            &record.host,
            record.port,
            &device_policy,
            &record.gpu_indices,
            runtime_id.as_deref(),
            env_id.as_deref(),
            engine_recipe_json.as_deref(),
            &record.engine_state_path,
        ))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));
    // Re-thread the endpoint key file (public bind only) onto the engine child,
    // same as the initial `rocm serve` spawn. This path also runs on daemon
    // recovery (`restart_managed_service` re-execs `rocmd supervise`), so
    // without this a previously-authenticated public service would come back
    // up anonymous after a crash/recover cycle.
    // If the key is gone the child would listen on the recorded public host with
    // no auth, so fail closed instead — an unreachable service is recoverable,
    // an anonymous public one is not.
    let endpoint_key_applied =
        crate::engine_ipc::apply_endpoint_key_env(&mut command, paths, &record.service_id);
    crate::engine_ipc::ensure_public_service_has_endpoint_key(
        &record.host,
        endpoint_key_applied,
        record.requires_api_key,
    )?;
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn engine supervisor child for {engine}"))?;

    // Every write from here on goes through `update_supervised_record`: it
    // applies only this supervisor's own fields to the record as it is on disk
    // now, so a stop marker, or anything else another writer persisted meanwhile,
    // survives it.
    //
    // Captured while the child is known alive, so a later stop verifies this
    // exact process instead of whatever has since inherited its PID.
    let engine_pid = child.id();
    update_supervised_record(paths, &record, |current| {
        current.record_engine_identity(engine_pid);
        current.status = "running".to_owned();
    })?;

    let became_ready = wait_for_service_ready(
        paths,
        &record.engine,
        &record.service_id,
        &record.log_path,
        Duration::from_mins(3),
        |phase| {
            let _ = update_supervised_record(paths, &record, |current| {
                current.startup_phase = Some(phase.to_owned());
            });
        },
    );
    if became_ready {
        update_supervised_record(paths, &record, |current| {
            current.status = "ready".to_owned();
            // The phase only describes the coming-up window; clear it once ready.
            current.startup_phase = None;
        })?;
    }

    let exit_status = child.wait().context("failed waiting for engine child")?;
    record_supervised_exit(paths, &record, exit_status.success())?;

    if exit_status.success() {
        Ok(())
    } else {
        std::process::exit(exit_status.code().unwrap_or(1));
    }
}

/// Apply one of [`supervise_service`]'s own changes to its service's record as
/// it is on disk now, rather than writing back the copy the supervisor built at
/// startup.
///
/// `write` replaces the whole file, and the supervisor's copy is stale the
/// moment anyone else writes: a stop records `stop_requested_unix_ms` before it
/// signals anything, and `rocm`'s liveness refresh and the daemon's restart
/// write here too. Writing the startup snapshot back would erase that stop
/// marker — and a record reading `failed` with no marker is exactly what the
/// daemon's recovery restarts, undoing the stop. So `update` touches only the
/// fields the supervisor owns, and never the marker.
///
/// Nothing is written when the record no longer belongs to this supervisor:
///
/// - it is gone — `rocm services remove` deleted it, and writing would
///   resurrect a service the operator removed;
/// - it names a different supervisor process — a confirmed stop clears the
///   recorded PIDs, and a restart records its own. Either way the record now
///   describes something this supervisor is not, and its view of the service
///   (`running`, `failed`, …) would overwrite a truer one.
fn update_supervised_record(
    paths: &AppPaths,
    supervisor: &ManagedServiceRecord,
    update: impl FnOnce(&mut ManagedServiceRecord),
) -> Result<()> {
    if !paths.service_manifest_path(&supervisor.service_id).exists() {
        return Ok(());
    }
    let mut current = crate::persistence::load_service_record(paths, &supervisor.service_id)?;
    if current.supervisor_pid != supervisor.supervisor_pid
        || current.supervisor_start_ticks != supervisor.supervisor_start_ticks
    {
        return Ok(());
    }
    update(&mut current);
    current.write()
}

/// [`supervise_service`]'s last write: the engine child has exited, so record
/// how. Separate so a test can drive exactly this write.
fn record_supervised_exit(
    paths: &AppPaths,
    supervisor: &ManagedServiceRecord,
    success: bool,
) -> Result<()> {
    update_supervised_record(paths, supervisor, |current| {
        current.status = if success { "stopped" } else { "failed" }.to_owned();
    })
}

#[allow(clippy::too_many_arguments)]
fn engine_serve_http_args(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &str,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe_json: Option<&str>,
    state_path: &Path,
) -> Vec<String> {
    let mut args = vec![
        "__engine-serve-http".to_owned(),
        engine.to_owned(),
        service_id.to_owned(),
        canonical_model_id.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--device-policy".to_owned(),
        device_policy.to_owned(),
    ];
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(crate::common::optional_arg("--runtime-id", runtime_id));
    args.extend(crate::common::optional_arg("--env-id", env_id));
    args.extend(crate::common::optional_arg(
        "--engine-recipe-json",
        engine_recipe_json,
    ));
    args.extend(["--state-path".to_owned(), state_path.display().to_string()]);
    args
}

/// Keep only the final visible segment of a `\r`-redrawn progress line.
///
/// Progress tools (pip, tqdm, Hugging Face) redraw a line in place with a bare
/// carriage return and no newline, so the segment after the last `\r` is its
/// final visible state. Lines without `\r` pass through unchanged. (Same
/// collapse rule the dashboard job console applies to streamed job output.)
fn last_cr_segment(line: &str) -> &str {
    line.rsplit('\r').next().unwrap_or(line)
}

/// Classify a single serve-log line into a coarse startup phase token
/// (`downloading`/`loading`/`warmup`), or `None` when the line carries no phase
/// signal. Case-insensitive substring match over the common vLLM / llama.cpp /
/// Hugging Face startup vocabulary. Checked warmup → loading → downloading so
/// the latest lifecycle stage a line mentions wins.
fn classify_startup_phase(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("capturing cuda graph")
        || lower.contains("capturing the model")
        || lower.contains("warming up")
        || lower.contains("warmup")
    {
        Some("warmup")
    } else if lower.contains("loading weights")
        || lower.contains("loading model")
        || lower.contains("load_tensors")
        || lower.contains("llama_model_loader")
        || lower.contains("model loading took")
    {
        Some("loading")
    } else if lower.contains("downloading") || lower.contains("fetching") {
        Some("downloading")
    } else {
        None
    }
}

/// Read log bytes appended since `*pos`, advance `*pos`, and return the most
/// recent recognizable startup phase in that new output (later lines win, so a
/// download → load → warmup progression advances naturally).
///
/// Best-effort: any I/O error (file not created yet, transient read) yields
/// `None`. A shrunk file (rotation/truncation) resets the cursor to the top.
fn read_new_log_phase(log_path: &Path, pos: &mut u64) -> Option<&'static str> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(log_path).ok()?;
    let len = file.metadata().ok()?.len();
    if len < *pos {
        *pos = 0;
    }
    if len == *pos {
        return None;
    }
    file.seek(SeekFrom::Start(*pos)).ok()?;
    let mut bytes = Vec::new();
    let read = file.read_to_end(&mut bytes).ok()?;
    *pos += read as u64;
    let text = String::from_utf8_lossy(&bytes);
    let mut phase = None;
    for line in text.lines() {
        if let Some(found) = classify_startup_phase(last_cr_segment(line)) {
            phase = Some(found);
        }
    }
    phase
}

fn engine_healthcheck_ready(paths: &AppPaths, engine: &str, service_id: &str) -> Result<bool> {
    Ok(crate::engine_ipc::healthcheck_response_ready(
        &crate::engine_ipc::engine_healthcheck_response(paths, engine, service_id)?,
    ))
}

/// Poll a freshly-spawned service until its healthcheck reports ready (or the
/// timeout elapses), tailing its log file meanwhile and reporting each coarse
/// startup phase transition via `on_phase`.
fn wait_for_service_ready(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    log_path: &Path,
    timeout: Duration,
    mut on_phase: impl FnMut(&str),
) -> bool {
    let start = std::time::Instant::now();
    let mut log_pos: u64 = 0;
    let mut last_phase: Option<&'static str> = None;
    while start.elapsed() < timeout {
        if let Some(phase) = read_new_log_phase(log_path, &mut log_pos)
            && last_phase != Some(phase)
        {
            last_phase = Some(phase);
            on_phase(phase);
        }
        if engine_healthcheck_ready(paths, engine, service_id).unwrap_or(false) {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

pub(crate) fn restart_managed_service(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
) -> Result<()> {
    let rocmd_binary =
        std::env::current_exe().context("failed to resolve current rocmd executable path")?;
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    record.status = "recovering".to_owned();
    // Counts the restart and drops the previous run's inference verification.
    // The respawned child writes a fresh record of its own, and "recovering" is
    // outside the statuses that probe, so a stale verdict would not currently be
    // acted on — but this record is written again below, after the spawn, and
    // that write can land after the child's. Clearing here keeps the invariant
    // true at the one site that reuses a record across restarts.
    record.reset_for_restart();
    // No supervisor exists until the spawn below succeeds, so none is recorded.
    // Recording this daemon's own PID here, as a stand-in, left the manifest
    // naming the daemon as the service's supervisor whenever the spawn failed:
    // a stop run from any other process (`rocmd sandbox-tool stop_server`,
    // `rocm services stop`) verifies that identity, finds it live, and
    // terminates the daemon's whole process tree. The spawned supervisor's
    // identity is recorded right after the spawn.
    record.supervisor_pid = 0;
    record.supervisor_start_ticks = None;
    // The engine PID belonged to the run being replaced, and nothing here
    // starts an engine: the supervisor spawned below records its own, together
    // with its token. Carried forward, the old PID would sit beside a
    // supervisor it no longer matches — and on a record `rocm` wrote, with no
    // token at all, so the next stop could only signal it blind.
    record.engine_pid = None;
    record.engine_start_ticks = None;
    // Restarting is a request to run, so it supersedes any earlier stop request,
    // as `rocm services restart` treats it too.
    record.stop_requested_unix_ms = None;
    record.write()?;

    let mut child = detached_rocmd_command(&rocmd_binary)
        .args(recovery_supervise_args(record))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn recovery supervisor")?;

    record_spawned_supervisor(paths, record, child.id())?;

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check recovery supervisor startup state")?
    {
        // Not a write-back of `record`: a stop issued since the spawn has
        // persisted its marker, and a `failed` record without one is what
        // recovery restarts. Same discipline as the supervisor's own writes.
        update_supervised_record(paths, record, |current| {
            current.status = "failed".to_owned();
        })?;
        anyhow::bail!(
            "recovery supervisor exited immediately with status {status}; inspect {}",
            record.log_path.display()
        );
    }

    Ok(())
}

/// Record the supervisor [`restart_managed_service`] just spawned, as `pid`.
///
/// Between the restart's first write — which clears the recorded PIDs, since
/// no supervisor exists yet — and this one, the record names no process at
/// all. A stop that starts in that window persists its marker first and then
/// finds nothing to signal; writing the restart's whole copy back here would
/// erase that marker while the new supervisor comes up, and leave the stop's
/// own final write comparing against a record that no longer shows the
/// process it has to answer for. So only the supervisor's identity is applied,
/// to the record as it is on disk now, through the same ownership guard as the
/// supervisor's own writes: against the copy from before this assignment, so
/// it is the cleared PID that has to still be there. If the new supervisor has
/// already recorded itself, or the record was removed, nothing is written.
fn record_spawned_supervisor(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    pid: u32,
) -> Result<()> {
    let unrecorded = record.clone();
    record.record_supervisor_identity(pid);
    let (supervisor_pid, supervisor_start_ticks) =
        (record.supervisor_pid, record.supervisor_start_ticks);
    update_supervised_record(paths, &unrecorded, |current| {
        current.supervisor_pid = supervisor_pid;
        current.supervisor_start_ticks = supervisor_start_ticks;
    })
}

fn recovery_supervise_args(record: &ManagedServiceRecord) -> Vec<String> {
    let mut args = vec![
        "supervise".to_owned(),
        record.service_id.clone(),
        "--engine".to_owned(),
        record.engine.clone(),
        "--model-ref".to_owned(),
        record.model_ref.clone(),
        "--canonical-model-id".to_owned(),
        record.canonical_model_id.clone(),
        "--host".to_owned(),
        record.host.clone(),
        "--port".to_owned(),
        record.port.to_string(),
        "--device-policy".to_owned(),
        record
            .device_policy
            .as_deref()
            .unwrap_or("gpu_required")
            .to_owned(),
    ];
    args.extend(crate::common::optional_arg(
        "--runtime-id",
        record.runtime_id.as_deref(),
    ));
    args.extend(crate::common::optional_arg(
        "--env-id",
        record.env_id.as_deref(),
    ));
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&record.gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(crate::common::optional_arg(
        "--engine-recipe-json",
        record.engine_recipe_json.as_deref(),
    ));
    args
}

#[cfg(unix)]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("setsid");
    command.arg(rocmd_binary);
    command
}

#[cfg(not(unix))]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    ProcessCommand::new(rocmd_binary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::load_service_record;
    #[cfg(target_os = "linux")]
    use crate::test_support::UNCONFIRMED_STOP_PID;
    use crate::test_support::{identity_probe_record, temp_app_paths, unique_test_root};
    use crate::watchers::find_recoverable_service;

    /// Drive `supervise_service` far enough to reach the key guard, and return
    /// what it did.
    ///
    /// The guard sits before the manifest write and well before any spawn, so a
    /// refusal returns without starting a process — which is what makes the real
    /// call site testable at all. The arguments below are the shape a recovery
    /// re-exec passes: a loopback bind, no GPU, no recipe.
    ///
    /// This exists because testing `ensure_public_service_has_endpoint_key`
    /// directly with literal arguments cannot catch the defect that actually
    /// happened twice in this crate's history — the guard being *wired up* with
    /// the wrong value at its call site.
    ///
    /// Bounded, and the bound is the assertion. A guard that fails to refuse
    /// does not return an error — it falls through to the engine spawn and
    /// supervises a child that never exits, so an unbounded call would hang the
    /// suite instead of failing it. Both callers below are regression tests for
    /// a fail-*open*, which is exactly the shape that turns into a hang.
    fn supervise_at_the_guard(paths: &AppPaths, service_id: &str) -> Result<()> {
        let paths = paths.clone();
        let service_id = service_id.to_owned();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = supervise_service(
                &paths,
                service_id,
                "llamacpp".to_owned(),
                "a-model".to_owned(),
                "a-model".to_owned(),
                None,
                None,
                "127.0.0.1".to_owned(),
                11434,
                "gpu_required".to_owned(),
                None,
                None,
            );
            let _ = sender.send(outcome.map_err(|error| format!("{error:#}")));
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap_or_else(|_| {
                panic!(
                    "supervise_service did not return within 30s: the key guard let the call \
                     through and it reached the engine spawn, which is the fail-open this test \
                     exists to catch"
                )
            })
            .map_err(anyhow::Error::msg)
    }

    /// Write a service record into the registry the way a live service would
    /// have left it behind.
    fn seed_registry(paths: &AppPaths, service_id: &str, requires_api_key: bool) {
        fs::create_dir_all(paths.services_dir()).unwrap();
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id.to_owned(),
            "llamacpp".to_owned(),
            "a-model".to_owned(),
            "a-model".to_owned(),
            "127.0.0.1".to_owned(),
            11434,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        record.requires_api_key = requires_api_key;
        record.write().unwrap();
    }

    /// Drive `supervise_service` and report whether the key guard let it past.
    ///
    /// Decided on what the call *returns*, not on any file. The obvious
    /// observable — the manifest appearing — is useless here, because
    /// `seed_registry` has already written one, so polling for it passes
    /// whatever the guard does. That mistake was made first and caught by
    /// mutating the code the test claims to protect.
    ///
    /// A call the guard admits does not return: it carries on to the engine
    /// spawn. So the guard's refusal is the only thing that comes back quickly,
    /// and it is identified by its message rather than by the mere fact of an
    /// error — a later, unrelated failure must not read as a refusal.
    fn guard_admits(paths: &AppPaths, service_id: &str) -> bool {
        let owned_paths = paths.clone();
        let owned_id = service_id.to_owned();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = supervise_service(
                &owned_paths,
                owned_id,
                "llamacpp".to_owned(),
                "a-model".to_owned(),
                "a-model".to_owned(),
                None,
                None,
                "127.0.0.1".to_owned(),
                11434,
                "gpu_required".to_owned(),
                None,
                None,
            );
            let _ = sender.send(outcome.map_err(|error| format!("{error:#}")));
        });

        match receiver.recv_timeout(std::time::Duration::from_secs(10)) {
            // The key guard refused, by its own words.
            Ok(Err(rendered)) if rendered.contains("without authentication") => false,
            // Anything else means it got past the guard: it either finished, or
            // failed later for a reason that is not this guard, or is still
            // running because it reached the spawn.
            _ => true,
        }
    }

    #[test]
    fn a_service_that_never_required_a_key_is_not_refused_for_lacking_one() {
        // The other direction of the guard, and the one no test covered.
        // Hardcoding `record.requires_api_key = true` at the restore site passes
        // every other test in this crate, because they all seed a service that
        // *does* require a key. This is the case that catches it.
        //
        // Two records are seeded, not one: with a single record the
        // `existing.service_id == record.service_id` half of the lookup does
        // nothing, so dropping that comparison would go unnoticed and one
        // service's requirement would leak onto another's.
        let (root, paths) = temp_app_paths("supervise-no-key-needed");
        seed_registry(&paths, "svc-needs-key", true);
        seed_registry(&paths, "svc-plain", false);

        assert!(
            guard_admits(&paths, "svc-plain"),
            "a loopback service that never asked for a key must not be refused for lacking one"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn supervising_a_service_that_required_a_key_refuses_when_the_key_is_gone() {
        // The real call site, not the guard in isolation. `supervise_service`
        // rebuilds the record with `ManagedServiceRecord::new`, which starts
        // `requires_api_key` false, and restores it from the registry. Passing
        // the wrong value here — a literal, or the freshly-built field before it
        // is restored — is exactly the miswiring that shipped twice in this
        // crate and that a literal-argument unit test cannot see.
        let (root, paths) = temp_app_paths("supervise-requires-key");
        seed_registry(&paths, "svc-needs-key", true);

        let error = supervise_at_the_guard(&paths, "svc-needs-key")
            .expect_err("a service that required a key must not be recovered without one");
        assert!(
            format!("{error:#}").contains("without authentication"),
            "{error:#}"
        );

        // And the refusal must not have weakened what is on disk. The guard runs
        // before `record.write()` precisely so a refused attempt leaves the
        // recorded requirement armed for the next attempt.
        let stored = crate::persistence::load_managed_services(&paths).unwrap();
        let stored = stored
            .iter()
            .find(|candidate| candidate.service_id == "svc-needs-key")
            .expect("the seeded record must survive a refused recovery");
        assert!(
            stored.requires_api_key,
            "a refused recovery must not disarm the requirement"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_registry_that_cannot_be_read_refuses_recovery_rather_than_assuming_no_key() {
        // `load_managed_services` already skips records it cannot parse, so an
        // `Err` from it is a real I/O failure — and a missing directory is `Ok`.
        // Defaulting it away therefore says "no service ever required a key",
        // which is fail-open on an auth gate and, worse, gets written back.
        //
        // The failure is provoked portably: a directory named like a record
        // makes the `fs::read` inside the loop fail rather than the read_dir.
        let (root, paths) = temp_app_paths("supervise-unreadable-registry");
        fs::create_dir_all(paths.services_dir().join("not-a-record.json")).unwrap();

        let error = supervise_at_the_guard(&paths, "svc-unknown")
            .expect_err("an unreadable registry must refuse, not assume no key was required");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("could not read the service registry"),
            "{rendered}"
        );

        // Nothing was written: a registry we could not read is not a registry we
        // may add a weakened record to.
        assert!(
            !paths.service_manifest_path("svc-unknown").exists(),
            "a refused recovery must not persist a record"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn last_cr_segment_keeps_final_progress_redraw() {
        // A tqdm/HF-style in-place redraw collapses to its last segment.
        assert_eq!(
            last_cr_segment("Downloading:  10%\rDownloading:  55%\rDownloading: 100%"),
            "Downloading: 100%"
        );
        // A plain line is unchanged.
        assert_eq!(
            last_cr_segment("Loading model weights"),
            "Loading model weights"
        );
    }

    #[test]
    fn classify_startup_phase_maps_engine_vocabulary() {
        assert_eq!(
            classify_startup_phase("Downloading shards: 100%"),
            Some("downloading")
        );
        assert_eq!(
            classify_startup_phase("Fetching 12 files"),
            Some("downloading")
        );
        assert_eq!(
            classify_startup_phase("INFO: Loading model weights took 4.2s"),
            Some("loading")
        );
        assert_eq!(
            classify_startup_phase("llama_model_loader: loaded meta data"),
            Some("loading")
        );
        assert_eq!(
            classify_startup_phase("Capturing CUDA graph shapes"),
            Some("warmup")
        );
        assert_eq!(
            classify_startup_phase("Warming up the engine"),
            Some("warmup")
        );
        // Ordinary chatter carries no phase signal.
        assert_eq!(
            classify_startup_phase("Uvicorn running on http://..."),
            None
        );
    }

    #[test]
    fn classify_startup_phase_emits_only_dashboard_known_tokens() {
        // These tokens are the wire contract with the dashboard's
        // `StartupPhase::from_token` (rocm-dash-core); emitting anything else
        // would be silently dropped there. rocmd can't link that crate, so the
        // contract is pinned here by literal.
        for line in [
            "Downloading shards",
            "Loading model weights",
            "Capturing CUDA graph",
        ] {
            let token = classify_startup_phase(line).expect("line is a phase signal");
            assert!(
                matches!(token, "downloading" | "loading" | "warmup"),
                "token {token:?} must be one the dashboard understands"
            );
        }
    }

    #[test]
    fn read_new_log_phase_advances_and_tracks_latest() {
        use std::io::Write as _;
        // Workspace-local test root (rooted at CARGO_MANIFEST_DIR, not the
        // ambient temp dir) — same helper the other rocmd tests use.
        let dir = unique_test_root(&format!("rocmd-phase-{}", std::process::id()));
        let log = dir.join("svc.log");
        std::fs::write(&log, "boot\nDownloading shards: 100%\n").unwrap();

        let mut pos = 0_u64;
        assert_eq!(read_new_log_phase(&log, &mut pos), Some("downloading"));
        // No new bytes → no phase, cursor unchanged.
        let after_first = pos;
        assert_eq!(read_new_log_phase(&log, &mut pos), None);
        assert_eq!(pos, after_first);

        // Appending a later stage advances the phase.
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(f, "Loading model weights took 3s").unwrap();
        assert_eq!(read_new_log_phase(&log, &mut pos), Some("loading"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_serve_http_args_forward_engine_recipe_json() {
        let engine_recipe_json = r#"{"contract_version":"0.1.0","engine":"vllm","required_flags":["--enable-auto-tool-choice"]}"#;
        let args = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[],
            Some("therock-release:gfx120X-all"),
            Some("env-1"),
            Some(engine_recipe_json),
            Path::new("state.json"),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--engine-recipe-json" && pair[1] == engine_recipe_json)
        );
        assert!(
            args.windows(2).any(|pair| {
                pair[0] == "--runtime-id" && pair[1] == "therock-release:gfx120X-all"
            })
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--state-path" && pair[1] == "state.json")
        );
    }

    #[test]
    fn engine_serve_http_args_emit_gpu_indices_when_pinned() {
        let args = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[1],
            None,
            None,
            None,
            Path::new("state.json"),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--gpu" && pair[1] == "1")
        );

        let auto = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[],
            None,
            None,
            None,
            Path::new("state.json"),
        );
        assert!(!auto.iter().any(|arg| arg == "--gpu"));
    }

    /// The identity check is only as good as the token the launcher writes down.
    /// `rocm_core::identity_state` treats a missing token as a best-effort match
    /// — the legacy `(None, _)` branch — so a record written without one is
    /// signalled blind, which is the original defect restored in full. Every
    /// other test here hands itself a record with the token already set, and so
    /// proves nothing about the code that is supposed to put it there.
    ///
    /// This drives the real `supervise_service` and reads the manifest it
    /// persisted. It is stopped at that first write by making the service log
    /// path a directory, so the `fs::File::create` immediately after it fails:
    /// going further reaches the engine spawn, and the failure path past that
    /// calls `std::process::exit`, which inside a test binary would take the
    /// whole suite with it.
    ///
    /// Linux-only: `rocm_core::process_start_ticks` reads `/proc` and returns
    /// `None` everywhere else, so on other platforms there is no token to write.
    #[cfg(target_os = "linux")]
    #[test]
    fn supervise_service_persists_the_supervisor_identity_token() -> Result<()> {
        let (root, paths) = temp_app_paths("supervise-records-identity");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-supervise-identity";
        let log_path = identity_probe_record(&paths, service_id, 11443).log_path;
        // A directory where the supervisor expects to create a file.
        fs::create_dir_all(&log_path)?;

        let outcome = supervise_service(
            &paths,
            service_id.to_owned(),
            "llamacpp".to_owned(),
            "a-model".to_owned(),
            "a-model".to_owned(),
            None,
            None,
            "127.0.0.1".to_owned(),
            11443,
            "gpu_required".to_owned(),
            None,
            None,
        );
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let error =
            outcome.expect_err("the log path is a directory, so the engine spawn is unreachable");
        assert!(
            format!("{error:#}").contains("failed to create"),
            "stopped somewhere other than the log file: {error:#}"
        );
        assert!(
            persisted?.supervisor_start_ticks.is_some(),
            "supervise_service must persist the supervisor's start-time token beside its PID"
        );
        Ok(())
    }

    /// The supervisor's last write lands after its engine child exits — which
    /// is what a stop that could not verify the supervisor, but did kill the
    /// engine, leaves behind. Writing back the record the supervisor built at
    /// startup would erase the stop marker the stop persisted, and a `failed`
    /// record with no marker is exactly what the daemon's recovery restarts.
    #[test]
    fn supervisor_exit_write_keeps_a_standing_stop_marker() -> Result<()> {
        let (root, paths) = temp_app_paths("supervise-exit-keeps-marker");
        paths.ensure()?;
        let service_id = "svc-supervise-exit-marker";
        // The supervisor's own copy, as `supervise_service` builds it: no marker.
        let mut supervisor = identity_probe_record(&paths, service_id, 11446);
        supervisor.record_supervisor_identity(std::process::id());
        supervisor.status = "running".to_owned();
        // What is on disk once a stop has run: same processes, marker set.
        let mut on_disk = supervisor.clone();
        on_disk.stop_requested_unix_ms = Some(1);
        on_disk.write()?;

        record_supervised_exit(&paths, &supervisor, false)?;
        let persisted = load_service_record(&paths, service_id);
        let recoverable = find_recoverable_service(&paths);
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_eq!(
            persisted.status, "failed",
            "the supervisor's own field lands"
        );
        assert_eq!(
            persisted.stop_requested_unix_ms,
            Some(1),
            "the supervisor's exit write must not withdraw a standing stop"
        );
        assert!(
            recoverable?.is_none(),
            "a stopped service must not become a recovery candidate"
        );
        Ok(())
    }

    /// A record that names another supervisor — a confirmed stop cleared the
    /// PIDs, or a restart recorded its own — or that was removed outright, is
    /// no longer this supervisor's to write.
    #[test]
    fn supervisor_exit_write_leaves_a_record_it_no_longer_owns() -> Result<()> {
        let (root, paths) = temp_app_paths("supervise-exit-not-owner");
        paths.ensure()?;
        let service_id = "svc-supervise-exit-not-owner";
        let mut supervisor = identity_probe_record(&paths, service_id, 11447);
        supervisor.record_supervisor_identity(std::process::id());
        supervisor.status = "running".to_owned();
        // A confirmed stop's final write: status stopped, PIDs cleared.
        let mut stopped = supervisor.clone();
        stopped.status = "stopped".to_owned();
        stopped.supervisor_pid = 0;
        stopped.supervisor_start_ticks = None;
        stopped.write()?;

        record_supervised_exit(&paths, &supervisor, false)?;
        let after_stop = load_service_record(&paths, service_id);

        fs::remove_file(paths.service_manifest_path(service_id))?;
        record_supervised_exit(&paths, &supervisor, false)?;
        let resurrected = paths.service_manifest_path(service_id).exists();
        fs::remove_dir_all(root).ok();

        assert_eq!(after_stop?.status, "stopped");
        assert!(!resurrected, "a removed record must not be written back");
        Ok(())
    }

    /// The supervisor's first write rebuilds the record from its arguments.
    /// The restart that launched it cleared any stop marker first, so a marker
    /// on disk here was set by a stop issued since, and must survive.
    #[cfg(target_os = "linux")]
    #[test]
    fn supervise_service_first_write_keeps_a_standing_stop_marker() -> Result<()> {
        let (root, paths) = temp_app_paths("supervise-first-write-keeps-marker");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-supervise-first-write-marker";
        let mut seeded = identity_probe_record(&paths, service_id, 11448);
        seeded.stop_requested_unix_ms = Some(1);
        seeded.write()?;
        // Stop `supervise_service` right after its first write; see
        // `supervise_service_persists_the_supervisor_identity_token`.
        fs::create_dir_all(&seeded.log_path)?;

        let outcome = supervise_service(
            &paths,
            service_id.to_owned(),
            "llamacpp".to_owned(),
            "a-model".to_owned(),
            "a-model".to_owned(),
            None,
            None,
            "127.0.0.1".to_owned(),
            11448,
            "gpu_required".to_owned(),
            None,
            None,
        );
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(outcome.is_err(), "the log path is a directory");
        assert_eq!(persisted?.stop_requested_unix_ms, Some(1));
        Ok(())
    }

    #[test]
    fn recovery_supervise_args_preserve_engine_recipe_json() {
        let (_root, paths) = temp_app_paths("recovery-engine-recipe");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "managed",
            123,
            Some("therock-release:gfx120X-all".to_owned()),
            Some("env-1".to_owned()),
            Some("gpu_required".to_owned()),
        );
        let engine_recipe_json = r#"{"contract_version":"0.1.0","engine":"vllm","required_flags":["--enable-auto-tool-choice"]}"#;
        record.engine_recipe_json = Some(engine_recipe_json.to_owned());

        let args = recovery_supervise_args(&record);

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--engine-recipe-json" && pair[1] == engine_recipe_json)
        );
        assert!(
            args.windows(2)
                .any(|pair| { pair[0] == "--canonical-model-id" && pair[1] == "Qwen/Qwen3.5-4B" })
        );
    }

    /// The same guarantee for the daemon's recovery path, which records a PID of
    /// its own. See `supervise_service_persists_the_supervisor_identity_token`
    /// for why an unrecorded token is the defect rather than a cosmetic gap.
    ///
    /// The restart re-execs `rocmd supervise`, which here is the test binary:
    /// it rejects those arguments and exits at once, so the restart reports
    /// failure — after it has written the record this test reads.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_managed_service_persists_the_supervisor_identity_token() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-records-identity");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-identity";
        let mut record = identity_probe_record(&paths, service_id, 11444);
        record.status = "failed".to_owned();
        record.write()?;
        assert_eq!(
            load_service_record(&paths, service_id)?.supervisor_start_ticks,
            None,
            "precondition: the seeded record carries no token"
        );

        let _ = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(
            persisted?.supervisor_start_ticks.is_some(),
            "restart_managed_service must persist the supervisor's start-time token beside its PID"
        );
        Ok(())
    }

    /// A restart moves the supervisor PID; the engine PID it carried belonged
    /// to the run being replaced. On a record `rocm serve --background` wrote,
    /// that PID has no token, so carrying it past the point where the two PIDs
    /// diverge leaves the next stop an entry it can only signal blind.
    ///
    /// Linux-only for the same reason as the test above: the restart re-execs
    /// the test binary, which is only known to reject those arguments there.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_managed_service_drops_the_replaced_runs_engine_pid() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-drops-engine-pid");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-drops-engine-pid";
        let mut record = identity_probe_record(&paths, service_id, 11455);
        // The shape `rocm serve --background` writes: one PID in both roles.
        record.supervisor_pid = UNCONFIRMED_STOP_PID;
        record.engine_pid = Some(UNCONFIRMED_STOP_PID);
        record.engine_start_ticks = None;
        record.stop_requested_unix_ms = Some(1);
        record.status = "failed".to_owned();
        record.write()?;

        let _ = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_ne!(persisted.supervisor_pid, UNCONFIRMED_STOP_PID);
        assert_eq!(
            persisted.engine_pid, None,
            "the replaced run's engine PID must not outlive the restart"
        );
        assert_eq!(persisted.engine_start_ticks, None);
        assert_eq!(
            persisted.stop_requested_unix_ms, None,
            "a restart supersedes an earlier stop request"
        );
        Ok(())
    }

    /// A stop that lands while a restart waits to see whether its supervisor
    /// survived startup must outlive the restart's `failed` write: a `failed`
    /// record without the marker is exactly what recovery restarts.
    ///
    /// The restarted supervisor is the test binary, which rejects the
    /// `supervise` arguments and exits at once, so the restart reaches that
    /// write. A thread plays the stop: once the restart has recorded the
    /// spawned supervisor, it persists a marker, as a stop does first.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_failure_write_keeps_a_stop_marker_set_meanwhile() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-failure-keeps-marker");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-failure-marker";
        let mut record = identity_probe_record(&paths, service_id, 11456);
        record.status = "failed".to_owned();
        record.write()?;

        let stop = {
            let paths = paths.clone();
            thread::spawn(move || -> Result<bool> {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while std::time::Instant::now() < deadline {
                    // `write` is not atomic, so a read can land mid-write and
                    // see a truncated file; just look again.
                    let Ok(mut on_disk) = load_service_record(&paths, service_id) else {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    };
                    if on_disk.supervisor_pid != 0 && on_disk.supervisor_pid != std::process::id() {
                        on_disk.stop_requested_unix_ms = Some(1);
                        on_disk.write()?;
                        return Ok(true);
                    }
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(false)
            })
        };
        let outcome = restart_managed_service(&paths, &mut record);
        let stop_landed = stop.join().expect("stop thread");
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(
            stop_landed?,
            "precondition: the stop ran during the restart"
        );
        assert!(outcome.is_err(), "the test binary is not a supervisor");
        let persisted = persisted?;
        assert_eq!(persisted.status, "failed");
        assert_eq!(
            persisted.stop_requested_unix_ms,
            Some(1),
            "the restart's failure write must not withdraw a stop issued meanwhile"
        );
        Ok(())
    }

    /// Between a restart's first write, which clears the recorded PIDs, and the
    /// one recording the supervisor it spawned, a stop can persist its marker.
    /// Writing the restart's own copy back would erase it while the new
    /// supervisor comes up — so only the identity is applied.
    #[test]
    fn restart_records_its_supervisor_without_erasing_a_stop_marker() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-spawn-keeps-marker");
        paths.ensure()?;
        let service_id = "svc-restart-spawn-marker";
        // The restart's copy after its first write: no supervisor, no marker.
        let mut restarting = identity_probe_record(&paths, service_id, 11457);
        restarting.status = "recovering".to_owned();
        restarting.supervisor_pid = 0;
        restarting.supervisor_start_ticks = None;
        // A stop started since: same (empty) processes, marker set.
        let mut on_disk = restarting.clone();
        on_disk.stop_requested_unix_ms = Some(1);
        on_disk.write()?;

        record_spawned_supervisor(&paths, &mut restarting, std::process::id())?;
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_eq!(restarting.supervisor_pid, std::process::id());
        assert_eq!(persisted.supervisor_pid, std::process::id());
        assert_eq!(
            persisted.supervisor_start_ticks,
            restarting.supervisor_start_ticks
        );
        assert_eq!(
            persisted.stop_requested_unix_ms,
            Some(1),
            "recording the spawned supervisor must not withdraw a standing stop"
        );
        Ok(())
    }

    /// The spawned supervisor may record itself before the restart does, and
    /// the record may be removed outright; either way the restart's write is
    /// not the one to land.
    #[test]
    fn restart_leaves_a_record_naming_another_supervisor_or_none() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-spawn-not-owner");
        paths.ensure()?;
        let service_id = "svc-restart-spawn-not-owner";
        let mut restarting = identity_probe_record(&paths, service_id, 11458);
        restarting.supervisor_pid = 0;
        restarting.supervisor_start_ticks = None;
        let mut on_disk = restarting.clone();
        on_disk.supervisor_pid = 4_242_424;
        on_disk.write()?;

        record_spawned_supervisor(&paths, &mut restarting.clone(), std::process::id())?;
        let after_supervisor = load_service_record(&paths, service_id);

        fs::remove_file(paths.service_manifest_path(service_id))?;
        record_spawned_supervisor(&paths, &mut restarting, std::process::id())?;
        let resurrected = paths.service_manifest_path(service_id).exists();
        fs::remove_dir_all(root).ok();

        assert_eq!(after_supervisor?.supervisor_pid, 4_242_424);
        assert!(!resurrected, "a removed record must not be written back");
        Ok(())
    }
}
