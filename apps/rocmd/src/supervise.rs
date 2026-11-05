// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::persistence::load_service_record;
use crate::watchers::stop_requested;
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
pub(crate) fn update_supervised_record(
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

/// Respawn the supervisor for a managed service that needs recovery.
///
/// Recovery re-claims the concrete port already written in the record under
/// [`rocm_core::MANAGED_LAUNCH_LOCK_TIMEOUT`]-bounded acquisition of the same
/// cross-process launch lock a `rocm serve` launch holds (unbounded, via
/// [`rocm_core::FileLock::acquire`]). The two still can't publish competing
/// records for this port at the same instant — the lock is exclusive either
/// way. A concurrent `rocm serve` launch does still block for as long as
/// recovery holds the lock; what the bound buys is the other direction —
/// recovery never waits out an interactive launch indefinitely. If a `rocm
/// serve` call is already mid-transaction when this bounded acquire times
/// out, recovery is deferred (not retried in a busy loop) rather than
/// queuing behind the CLI.
pub(crate) fn restart_managed_service(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
) -> Result<()> {
    restart_managed_service_with(paths, record, None)
}

/// Replaces the real `command.spawn()` of a recovery supervisor: a test-only
/// seam for asserting on state at the exact instant production would spawn.
type SpawnOverride<'a> = dyn Fn(&mut ProcessCommand) -> std::io::Result<std::process::Child> + 'a;

/// [`restart_managed_service`], with the supervisor spawn interceptable via
/// `spawn_override` (the real `command.spawn()` when `None`).
fn restart_managed_service_with(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    spawn_override: Option<&SpawnOverride<'_>>,
) -> Result<()> {
    restart_managed_service_bounded(
        paths,
        record,
        spawn_override,
        rocm_core::MANAGED_LAUNCH_LOCK_TIMEOUT,
    )
}

/// [`restart_managed_service_with`] with the launch-lock wait injectable, so a
/// test can prove the wait is bounded without waiting out the real bound.
fn restart_managed_service_bounded(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    spawn_override: Option<&SpawnOverride<'_>>,
    lock_timeout: Duration,
) -> Result<()> {
    let _launch_lock =
        rocm_core::FileLock::acquire_bounded(paths.managed_launch_lock_path(), lock_timeout)?;

    // Re-check against the disk, now that the lock excludes a concurrent
    // `rocm serve` publish: the caller's `record` was read before this lock was
    // acquired, so a fresh launch could have claimed this exact (host, port)
    // in the meantime (the engine itself binds the port only after the model
    // loads, so a raw OS bind probe alone would not catch that — see
    // `crate::persistence::load_managed_services`'s callers for the shared
    // `is_live` reservation this mirrors). Refuse rather than spawn a
    // competitor; the next watcher tick reconsiders this service.
    let on_disk = crate::persistence::load_managed_services(paths)
        .context("failed to re-read managed services under the launch lock")?;
    if let Some(collision) = on_disk
        .iter()
        .find(|other| other.service_id != record.service_id && other.blocks_port(record.port))
    {
        anyhow::bail!(
            "cannot recover managed service {} on {}:{}: {} claimed that port while recovery \
             was waiting for the launch lock",
            record.service_id,
            record.host,
            record.port,
            collision.service_id
        );
    }

    // The same re-read must also prove this service's own record is still the
    // one the caller saw: a `rocm services restart` of this service can finish
    // while recovery waits for the lock, and writing the stale copy over it
    // would orphan the restarted engine or start a second supervisor.
    match load_service_record(paths, &record.service_id) {
        Ok(current) => {
            // Only a stop newer than the caller's copy counts: restarting a
            // record that already carried one deliberately supersedes it.
            if stop_requested(&current)
                && current.stop_requested_unix_ms != record.stop_requested_unix_ms
            {
                anyhow::bail!(
                    "cannot recover managed service {}: a stop was requested while recovery was \
                     waiting for the launch lock",
                    record.service_id
                );
            }
            if current.status != record.status
                || current.supervisor_pid != record.supervisor_pid
                || current.supervisor_start_ticks != record.supervisor_start_ticks
                || current.engine_pid != record.engine_pid
            {
                anyhow::bail!(
                    "cannot recover managed service {}: its record changed while recovery was \
                     waiting for the launch lock",
                    record.service_id
                );
            }
        }
        Err(_) if !paths.service_manifest_path(&record.service_id).exists() => {
            anyhow::bail!(
                "cannot recover managed service {}: it was removed while recovery was waiting \
                 for the launch lock",
                record.service_id
            );
        }
        Err(error) => return Err(error.context("failed to re-read the service record")),
    }

    // On the canonical loopback host, prove the recorded port is genuinely free
    // before publishing `recovering` or spawning anything: a supervisor that
    // cannot bind would replace a truthful record with a lie. Custom hosts keep
    // the existing engine-owned bind behavior — the record transition is still
    // serialized, but the bind is not preflighted.
    let lease = if rocm_core::is_canonical_loopback_host(&record.host) {
        Some(rocm_core::lease_loopback_port_or_bind_error(
            record.port,
            format_args!(
                "cannot recover managed service {} on {}:{}: its recorded port is in use",
                record.service_id, record.host, record.port
            ),
        )?)
    } else {
        None
    };

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

    // The truthful state to fall back to. Every failure after the `recovering`
    // write below (spawn, `try_wait`, recording the supervisor) restores this
    // through `fail_recovery` rather than leave a half-applied `recovering` row
    // that would keep reserving the port, and a supervisor that exists is never
    // left running unrecorded.
    let prior = record.clone();

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
    if let Err(error) = record.write() {
        // Nothing was spawned and the on-disk record was never replaced; just
        // undo the in-memory transition so the caller does not report recovery
        // that never happened.
        *record = prior;
        return Err(error.context("failed to publish the recovering service record"));
    }

    // Hand the port to the supervisor: close the probe socket immediately before
    // the spawn, still holding the launch lock.
    if let Some(lease) = lease {
        lease.release();
    }

    let recovering_snapshot = record.clone();
    let mut command = detached_rocmd_command(&rocmd_binary);
    command
        .args(recovery_supervise_args(record))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));
    let spawned = match spawn_override {
        Some(spawn) => spawn(&mut command),
        None => command.spawn(),
    };
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            let error = anyhow::Error::new(error).context("failed to spawn recovery supervisor");
            return Err(fail_recovery(
                paths,
                record,
                &recovering_snapshot,
                prior,
                error,
            ));
        }
    };

    if let Err(error) = record_spawned_supervisor(paths, record, child.id()) {
        // The supervisor is alive but unrecorded: it would hold the port while
        // being invisible to every stop/recovery path. Kill it first.
        terminate_recovery_supervisor(&mut child);
        let error = error.context("failed to publish the recovering service record");
        return Err(fail_recovery(
            paths,
            record,
            &recovering_snapshot,
            prior,
            error,
        ));
    }

    // The supervisor's identity is on disk now, so the rollback guard must
    // compare against this copy, not the one from before it was recorded.
    let recorded_snapshot = record.clone();
    thread::sleep(Duration::from_millis(200));
    let exited = match child.try_wait() {
        Ok(exited) => exited,
        Err(error) => {
            terminate_recovery_supervisor(&mut child);
            let error = anyhow::Error::new(error)
                .context("failed to check recovery supervisor startup state");
            return Err(fail_recovery(
                paths,
                record,
                &recorded_snapshot,
                prior,
                error,
            ));
        }
    };
    if let Some(status) = exited {
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

/// Undo [`restart_managed_service`]'s `recovering` publication after a later
/// step failed, and return the error to report.
///
/// Not a blind `record.write()` of `prior`: the same discipline as
/// `update_supervised_record` applies, so a stop marker a concurrent stop
/// persisted since the first write is not reverted. When the rollback itself
/// fails the on-disk record still claims `recovering` for a supervisor that no
/// longer exists, which is worse than the original failure and is reported with
/// it rather than swallowed.
fn fail_recovery(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    recovering_snapshot: &ManagedServiceRecord,
    prior: ManagedServiceRecord,
    error: anyhow::Error,
) -> anyhow::Error {
    if let Err(rollback) = rollback_recovering_record(paths, recovering_snapshot, &prior) {
        return error.context(format!(
            "rollback also failed: managed service {} is left recorded as recovering \
             with no supervisor ({rollback:#})",
            record.service_id
        ));
    }
    *record = prior;
    error
}

/// Grace given to a recovery supervisor that must be terminated because its
/// record could not be published. Short: it is milliseconds old.
const RECOVERY_CLEANUP_GRACE: Duration = Duration::from_secs(2);

// Test-only, thread-scoped call counter: `restart_managed_service`'s rollback
// branch asserts this to prove the kill is actually reached (not just that the
// record comes back truthful), since that branch's own supervisor is a
// re-exec'd test binary that exits on its own almost immediately regardless of
// whether it is killed, so `process_is_running` on it would not catch the call
// being dropped. Thread-local, not a global counter: cargo runs tests
// concurrently on separate threads, and this function has another, unrelated
// direct caller (`terminate_recovery_supervisor_kills_and_reaps_the_child`).
#[cfg(test)]
thread_local! {
    static TERMINATE_RECOVERY_SUPERVISOR_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Terminate and reap a recovery supervisor that could not be recorded.
///
/// Tree scope, forced: the supervisor may already have started an engine child,
/// and leaving either alive would keep the recorded port occupied by a process
/// no stop or recovery path can find. Reaping afterwards keeps the daemon free
/// of zombies.
fn terminate_recovery_supervisor(child: &mut std::process::Child) {
    #[cfg(test)]
    TERMINATE_RECOVERY_SUPERVISOR_CALLS.with(|calls| calls.set(calls.get() + 1));
    let _outcome = rocm_core::terminate_verified(
        &rocm_core::ProcessIdentity::capture(child.id()),
        rocm_core::KillScope::Tree,
        RECOVERY_CLEANUP_GRACE,
        true,
    );
    let _ = child.wait();
}

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

/// Restore only the fields [`restart_managed_service`]'s transaction owns,
/// after `record_spawned_supervisor` failed to publish them. The same
/// discipline as `update_supervised_record` elsewhere in that function: a
/// stop marker a concurrent stop persisted on the live record must survive
/// this rollback, not be overwritten by a blind copy of `restore`.
///
/// Differs from that helper in one case: when the manifest cannot be parsed
/// at all (as opposed to merely naming a different supervisor), there is no
/// live data left to selectively preserve, so this falls back to writing
/// `restore` in full rather than no-op'ing.
/// `restart_rolls_back_and_kills_the_supervisor_when_the_second_write_cannot_be_read`
/// exercises exactly that fallback.
fn rollback_recovering_record(
    paths: &AppPaths,
    recovering_snapshot: &ManagedServiceRecord,
    restore: &ManagedServiceRecord,
) -> Result<()> {
    let manifest_path = paths.service_manifest_path(&recovering_snapshot.service_id);
    if !manifest_path.exists() {
        return Ok(());
    }
    match load_service_record(paths, &recovering_snapshot.service_id) {
        Ok(mut current)
            if current.supervisor_pid == recovering_snapshot.supervisor_pid
                && current.supervisor_start_ticks == recovering_snapshot.supervisor_start_ticks =>
        {
            current.status = restore.status.clone();
            current.supervisor_pid = restore.supervisor_pid;
            current.supervisor_start_ticks = restore.supervisor_start_ticks;
            current.engine_pid = restore.engine_pid;
            current.engine_start_ticks = restore.engine_start_ticks;
            current.restart_count = restore.restart_count;
            current.last_restart_unix_ms = restore.last_restart_unix_ms;
            current.inference_verified_at_unix_ms = restore.inference_verified_at_unix_ms;
            current.inference_probe_attempted_at_unix_ms =
                restore.inference_probe_attempted_at_unix_ms;
            current.write()
        }
        Ok(_) => Ok(()),
        Err(_) => restore.write(),
    }
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
    use crate::test_support::{
        free_loopback_port, identity_probe_record, temp_app_paths, unique_test_root,
    };
    use crate::watchers::find_recoverable_service;
    use rocm_core::unix_time_millis;

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
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
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
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
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

    #[test]
    fn recovery_refuses_a_port_a_launch_claimed_while_it_waited_for_the_lock() -> Result<()> {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        // The gap this closes: a crashed service's port is simultaneously
        // "free" to a brand-new `rocm serve --gpu auto` launch (which only
        // checks *live* records) and "needs recovery" to this function (which
        // used to only probe the raw OS bind). Neither process's engine binds
        // the port until its model finishes loading, so a probe alone cannot
        // see a `starting` competitor that has not bound yet. Recreate that
        // exact disk state under the lock and require recovery to refuse,
        // rather than racing a second engine onto the same port.
        let (_root, paths) = temp_app_paths("recovery-launch-claimed-port");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let port = 21_777_u16;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-recover-3",
            "lemonade",
            "qwen",
            "qwen-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;

        let guard = rocm_core::FileLock::acquire_bounded(
            paths.managed_launch_lock_path(),
            rocm_core::MANAGED_LAUNCH_LOCK_TIMEOUT,
        )?;
        let finished = Arc::new(AtomicBool::new(false));
        let handle = {
            let paths = paths.clone();
            let finished = Arc::clone(&finished);
            thread::spawn(move || {
                let result = restart_managed_service(&paths, &mut record);
                finished.store(true, Ordering::SeqCst);
                result
            })
        };

        thread::sleep(Duration::from_millis(250));
        assert!(
            !finished.load(Ordering::SeqCst),
            "recovery must serialize on the shared launch lock"
        );

        // Simulate a `rocm serve --gpu auto` publishing a brand-new service on
        // this exact port while recovery was still waiting for the lock it
        // itself now holds — the one window the launch lock is supposed to
        // close.
        let mut competitor = ManagedServiceRecord::new(
            &paths,
            "lemonade-fresh-launch",
            "lemonade",
            "other-model",
            "other-model-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        competitor.status = "starting".to_owned();
        competitor.write()?;
        drop(guard);

        let error = handle
            .join()
            .expect("recovery thread")
            .expect_err("a live competitor on the same port must refuse recovery");
        let message = format!("{error:#}");
        assert!(message.contains("lemonade-fresh-launch"), "{message}");
        assert!(message.contains("claimed that port"), "{message}");

        let on_disk = crate::persistence::load_managed_services(&paths)?
            .into_iter()
            .find(|candidate| candidate.service_id == "lemonade-recover-3")
            .expect("record still present");
        assert_eq!(
            on_disk.status, "failed",
            "refused recovery must not overwrite the truthful record with `recovering`"
        );
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn recovery_is_not_blocked_by_a_stopped_or_failed_record_on_the_same_port() -> Result<()> {
        // The mirror image of the refusal above: `blocks_port` is
        // `is_live() && port == port`, so a `stopped` or `failed` record on
        // the same port must not count as a collision. Mutation-proof: if
        // `blocks_port` regressed to ignore status, either sibling below
        // would trip the "claimed that port" bail and this test would go red.
        let (root, paths) = temp_app_paths("recovery-non-live-same-port");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let port = free_loopback_port();
        let mut stopped_sibling = ManagedServiceRecord::new(
            &paths,
            "lemonade-stopped-sibling",
            "lemonade",
            "other-model",
            "other-model-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        stopped_sibling.status = "stopped".to_owned();
        stopped_sibling.write()?;

        let mut failed_sibling = ManagedServiceRecord::new(
            &paths,
            "lemonade-failed-sibling",
            "lemonade",
            "other-model-2",
            "other-model-2-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        failed_sibling.status = "failed".to_owned();
        failed_sibling.write()?;

        let mut record = identity_probe_record(&paths, "lemonade-recover-non-live", port);
        record.status = "failed".to_owned();
        record.write()?;

        // The re-exec'd "supervisor" is this very test binary, which cannot
        // actually run as `rocmd supervise` and exits almost immediately; the
        // same limitation `restart_managed_service_persists_the_supervisor_identity_token`
        // works around. What this proves is that the spawn was ever reached:
        // a wrongly-blocking `blocks_port` would bail before `record.write()`
        // below ever runs, leaving `supervisor_start_ticks` at `None`.
        let result = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, "lemonade-recover-non-live");
        fs::remove_dir_all(root).ok();

        if let Err(error) = &result {
            let message = format!("{error:#}");
            assert!(
                !message.contains("claimed that port"),
                "a stopped or failed sibling on the same port must not be treated as a collision: {message}"
            );
        }
        assert!(
            persisted?.supervisor_start_ticks.is_some(),
            "recovery must reach the spawn, proving neither sibling blocked it"
        );
        Ok(())
    }

    #[test]
    fn recovery_refuses_when_the_registry_cannot_be_re_read_leaving_the_prior_record_intact()
    -> Result<()> {
        use std::net::TcpListener;

        let (_root, paths) = temp_app_paths("recovery-publication-failure");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // A free recorded port, so a lease was never the obstacle under test.
        let probe = TcpListener::bind(("127.0.0.1", 0))?;
        let port = probe.local_addr()?.port();
        drop(probe);

        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-recover-2",
            "lemonade",
            "qwen",
            "qwen-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            4242,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;

        // Replace the manifest with a directory. `load_managed_services` scans
        // every `.json` entry under `services_dir()` to re-check for a
        // collision before publishing `recovering`, and this record's own
        // manifest is one of those entries — so the corruption trips that
        // re-read, not the publish a few lines later. That is the more
        // conservative of the two and still proves the invariant this test is
        // for: recovery must fail closed, not spawn a competitor, when it
        // cannot trust what else is live.
        fs::remove_file(&record.manifest_path)?;
        fs::create_dir_all(&record.manifest_path)?;

        let mut attempt = record.clone();
        let error = restart_managed_service(&paths, &mut attempt)
            .expect_err("an unreadable registry must refuse, not assume no collision");
        let message = format!("{error:#}");
        assert!(
            message.contains("failed to re-read managed services under the launch lock"),
            "{message}"
        );
        // Nothing was spawned and the on-disk row was never replaced, so the
        // error must NOT claim a failed rollback: that phrasing is reserved for
        // the post-spawn case where the record really is left mid-transition.
        assert!(!message.contains("rollback also failed"), "{message}");

        // The caller's record must not claim a recovery that never happened.
        assert_eq!(attempt.status, "failed");
        assert_eq!(attempt.restart_count, record.restart_count);
        assert_eq!(attempt.supervisor_pid, record.supervisor_pid);
        fs::remove_dir_all(&record.manifest_path)?;
        Ok(())
    }

    /// Mirrors `launch_releases_the_leased_port_before_spawning_the_child` in
    /// `apps/rocm/src/service_launch.rs`: the probe lease on the recorded port
    /// must already be released by the time the recovery supervisor is
    /// spawned, or the supervisor's own bind of that port would fail.
    #[test]
    fn restart_releases_the_leased_port_before_spawning_the_supervisor() -> Result<()> {
        use std::cell::Cell;
        use std::net::TcpListener;

        let (root, paths) = temp_app_paths("restart-releases-lease");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let probe = TcpListener::bind(("127.0.0.1", 0))?;
        let port = probe.local_addr()?.port();
        drop(probe);

        let service_id = "svc-restart-releases-lease";
        let mut record = identity_probe_record(&paths, service_id, port);
        record.status = "failed".to_owned();
        record.write()?;

        let bind_succeeded = Cell::new(false);
        let spawn_override =
            |command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
                // Retry briefly: a sibling test's concurrent fork can hold a
                // copy of the just-released socket until its exec closes it.
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while !bind_succeeded.get() && std::time::Instant::now() < deadline {
                    bind_succeeded.set(TcpListener::bind(("127.0.0.1", port)).is_ok());
                }
                command.spawn()
            };

        // The re-exec'd "supervise" args are rejected by this test binary,
        // which exits at once (see
        // restart_managed_service_persists_the_supervisor_identity_token), so
        // the restart itself is expected to report failure here.
        let _ = restart_managed_service_with(&paths, &mut record, Some(&spawn_override));
        fs::remove_dir_all(root).ok();

        assert!(
            bind_succeeded.get(),
            "the leased port must already be bindable (released) by the time the supervisor is spawned"
        );
        Ok(())
    }

    /// Recovery re-claims the concrete port already written in the record, so
    /// it must serialize on the same cross-process launch lock a fresh
    /// `rocm serve` launch takes, and must refuse a recorded port something
    /// else already holds rather than publish a `recovering` record that
    /// lies.
    #[test]
    fn recovery_waits_for_the_allocation_lock_and_refuses_an_occupied_port() -> Result<()> {
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let (root, paths) = temp_app_paths("recovery-port-collision");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // Something else already owns the port this service recorded.
        let occupant = TcpListener::bind(("127.0.0.1", 0))?;
        let port = occupant.local_addr()?.port();

        let mut record = identity_probe_record(&paths, "svc-recovery-port-collision", port);
        record.status = "failed".to_owned();
        record.write()?;

        let guard = rocm_core::FileLock::acquire_bounded(
            paths.managed_launch_lock_path(),
            rocm_core::MANAGED_LAUNCH_LOCK_TIMEOUT,
        )?;
        let finished = Arc::new(AtomicBool::new(false));
        let handle = {
            let paths = paths.clone();
            let finished = Arc::clone(&finished);
            let mut record = record.clone();
            thread::spawn(move || {
                let result = restart_managed_service(&paths, &mut record);
                finished.store(true, Ordering::SeqCst);
                result
            })
        };

        thread::sleep(Duration::from_millis(250));
        assert!(
            !finished.load(Ordering::SeqCst),
            "recovery must serialize on the shared launch lock"
        );
        drop(guard);

        let error = handle
            .join()
            .expect("recovery thread")
            .expect_err("an occupied recorded port must fail recovery");
        let message = format!("{error:#}");
        assert!(message.contains("recorded port is in use"), "{message}");

        // Nothing was spawned and the prior truthful record is untouched: it must
        // not have been rewritten as `recovering`.
        let on_disk = crate::persistence::load_managed_services(&paths)?
            .into_iter()
            .find(|candidate| candidate.service_id == "svc-recovery-port-collision")
            .expect("record still present");
        fs::remove_dir_all(root).ok();
        drop(occupant);

        assert_eq!(on_disk.status, "failed");
        assert_eq!(on_disk.restart_count, record.restart_count);
        Ok(())
    }

    #[test]
    #[ignore = "spawned as a child process by terminate_recovery_supervisor_kills_and_reaps_the_child"]
    fn recovery_cleanup_sleeper_child() {
        if std::env::var_os("ROCMD_TEST_SLEEPER_CHILD").is_none() {
            return;
        }
        thread::sleep(Duration::from_mins(2));
    }

    #[test]
    fn terminate_recovery_supervisor_kills_and_reaps_the_child() {
        // A real long-lived process stands in for a supervisor that was spawned
        // but whose record could not be published.
        let mut child = ProcessCommand::new(std::env::current_exe().expect("test binary"))
            .arg("watchers::tests::recovery_cleanup_sleeper_child")
            .arg("--exact")
            .arg("--ignored")
            .env("ROCMD_TEST_SLEEPER_CHILD", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleeper child");
        let child_pid = child.id();

        let started = std::time::Instant::now();
        terminate_recovery_supervisor(&mut child);
        // The sleeper sleeps for two minutes by itself: only a kill returns
        // well before that, so waiting it out must not pass.
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "terminate_recovery_supervisor took {:?}: the child was waited out, not killed",
            started.elapsed()
        );

        // The helper reaps, so the pid must be gone rather than a zombie.
        assert!(
            !rocm_core::process_is_running(child_pid),
            "an unrecordable supervisor must be terminated and reaped"
        );
        // The helper already reaped it; wait again so the child is provably
        // handled on every path in this function (and reaping is idempotent).
        assert!(child.wait().is_ok());
    }

    /// If publishing the `recovering` transition fails, nothing has been
    /// spawned yet: the rollback at that first write must undo the in-memory
    /// transition and leave the on-disk record exactly as it was, not a
    /// half-applied `recovering` row. The manifest write goes through
    /// [`crate::write_atomic`][rocm_core_write_atomic]-style write-then-rename,
    /// so a read-only *file* no longer blocks it (rename ignores the
    /// destination's own permission bits) — a read-only *directory* still does,
    /// since it can't hold the new temp file the rename needs.
    ///
    /// [rocm_core_write_atomic]: rocm_core
    ///
    /// Unix only and skipped as root, where a read-only mode is not enforced
    /// and the write would simply succeed.
    #[cfg(unix)]
    #[test]
    fn restart_rolls_back_the_in_memory_record_when_the_first_write_fails() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        if rocm_core::openmpi::running_as_root() {
            return Ok(());
        }

        let (root, paths) = temp_app_paths("restart-rollback-first-write");
        paths.ensure()?;
        let service_id = "svc-restart-rollback-first-write";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;

        let services_dir = paths.services_dir();
        // The launch lock and service log files are both created lazily, on
        // first acquire and on first open respectively; pre-create them so
        // making the directory read-only below blocks only the manifest's
        // write-then-rename, not those earlier opens.
        fs::File::create(paths.managed_launch_lock_path())?;
        fs::File::create(&record.log_path)?;
        fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o555))?;

        let result = restart_managed_service(&paths, &mut record);

        // Restore write access so the rest of the write (if any landed) can be
        // read back, and so cleanup can remove the directory.
        fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o755))?;
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let error = result.expect_err("a read-only manifest must fail the first write");
        assert!(
            format!("{error:#}").contains("failed to publish the recovering service record"),
            "{error:#}"
        );
        assert_eq!(
            record.status, "failed",
            "a failed first write must restore the in-memory record rather than leave it \
             \"recovering\""
        );
        assert_eq!(
            persisted?.status, "failed",
            "nothing must reach disk when the first write itself fails"
        );
        Ok(())
    }

    /// The first-write rollback above covers a failure before anything is
    /// spawned; this one covers the second write, after the supervisor exists.
    /// The spawn seam corrupts the manifest's content right after the first
    /// write (`recovering`, no supervisor yet), so
    /// `record_spawned_supervisor`'s read fails. That must kill the spawned
    /// supervisor and restore the prior record — and the rollback write itself
    /// must still land, since corrupting *content* leaves the file a writable
    /// regular file.
    ///
    /// The spawned supervisor here is a re-exec of this very test binary,
    /// which rejects its `supervise` arguments and exits on its own almost
    /// immediately regardless of whether anything kills it, so asserting
    /// `process_is_running` on it afterward would not catch the kill being
    /// dropped. `TERMINATE_RECOVERY_SUPERVISOR_CALLS` asserts the call itself
    /// is reached instead; `terminate_recovery_supervisor_kills_and_reaps_the_child`
    /// below proves what that call does against a real long-lived process.
    #[test]
    fn restart_rolls_back_and_kills_the_supervisor_when_the_second_write_cannot_be_read()
    -> Result<()> {
        let (root, paths) = temp_app_paths("restart-rollback-second-write-unreadable");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-rollback-second-write-unreadable";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;
        let prior = record.clone();

        let terminate_calls_before = TERMINATE_RECOVERY_SUPERVISOR_CALLS.with(std::cell::Cell::get);
        let manifest_path = paths.service_manifest_path(service_id);
        // The spawn seam runs after the restart's first write and before
        // `record_spawned_supervisor`, which is exactly the window to corrupt.
        let corrupt = |command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            fs::write(&manifest_path, b"not json")?;
            command.spawn()
        };
        let outcome = restart_managed_service_with(&paths, &mut record, Some(&corrupt));
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let error = outcome.expect_err("an unreadable second write must fail the restart");
        let message = format!("{error:#}");
        assert!(
            message.contains("failed to publish the recovering service record"),
            "{message}"
        );
        assert!(
            !message.contains("rollback also failed"),
            "corrupting content, not permissions, leaves the rollback write itself able to \
             land: {message}"
        );
        let terminate_calls_after = TERMINATE_RECOVERY_SUPERVISOR_CALLS.with(std::cell::Cell::get);
        assert_eq!(
            terminate_calls_after,
            terminate_calls_before + 1,
            "an unreadable second write must still kill the spawned supervisor"
        );
        assert_eq!(
            record.status, prior.status,
            "the in-memory record must be restored to what it was before this restart"
        );
        let persisted = persisted?;
        assert_eq!(
            persisted.status, prior.status,
            "the rollback write must overwrite the corrupted content with the prior record"
        );
        assert_eq!(persisted.supervisor_pid, prior.supervisor_pid);
        Ok(())
    }

    /// As above, but the rollback write itself also fails: a manifest made
    /// read-only after the first write still lets `record_spawned_supervisor`
    /// *read* it (ownership check passes), but both its write and the
    /// subsequent rollback write hit the same permission error. That must be
    /// reported as a rollback failure, not swallowed as the original one.
    ///
    /// The manifest writes go through write-then-rename, so blocking the
    /// *directory* (not the file) is what actually denies them.
    ///
    /// Unix only and skipped as root, where a read-only mode is not enforced.
    #[cfg(unix)]
    #[test]
    fn restart_reports_when_the_rollback_write_also_fails() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        if rocm_core::openmpi::running_as_root() {
            return Ok(());
        }

        let (root, paths) = temp_app_paths("restart-rollback-also-fails");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-rollback-also-fails";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;

        let services_dir = paths.services_dir();
        // Same seam as above: deny directory writes once the first write has
        // landed, so the second write and its rollback both fail.
        let lock = |command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o555))?;
            command.spawn()
        };
        let outcome = restart_managed_service_with(&paths, &mut record, Some(&lock));
        fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o755))?;
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let error = outcome.expect_err("a read-only manifest must fail both writes");
        let message = format!("{error:#}");
        assert!(
            message.contains("rollback also failed"),
            "both the second write and its rollback hit the same permission error: {message}"
        );
        assert!(
            message.contains(service_id),
            "the rollback-also-failed message must name the affected service: {message}"
        );
        // Left exactly as the second write's failure found it: `recovering`
        // with no supervisor, matching the "no supervisor" the error names.
        assert_eq!(persisted?.status, "recovering");
        Ok(())
    }

    /// A spawn that fails after the `recovering` write must not leave that
    /// write behind: `is_live` counts `recovering`, and rocmd reads records
    /// without refreshing them, so a stale one would reserve the port.
    #[test]
    fn restart_rolls_back_the_recovering_record_when_the_spawn_fails() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-spawn-fails-rolls-back");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-spawn-fails";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;

        let failing = |_command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            Err(std::io::Error::other("spawn refused by test"))
        };
        let outcome = restart_managed_service_with(&paths, &mut record, Some(&failing));
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let message = format!(
            "{:#}",
            outcome.expect_err("a failed spawn fails the restart")
        );
        assert!(
            message.contains("failed to spawn recovery supervisor"),
            "{message}"
        );
        assert_eq!(
            persisted?.status, "failed",
            "the on-disk record must not stay `recovering`"
        );
        assert_eq!(
            record.status, "failed",
            "the in-memory record must be restored too"
        );
        Ok(())
    }

    /// The disk re-check excludes the service's own record: rocmd reads records
    /// as-is, so a crashed service can still read `running` on its own port, and
    /// counting that as a competitor would make it impossible to ever recover.
    #[test]
    fn recovery_ignores_its_own_live_looking_record_on_its_own_port() -> Result<()> {
        use std::net::TcpListener;

        let (root, paths) = temp_app_paths("recovery-self-exclusion");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let probe = TcpListener::bind(("127.0.0.1", 0))?;
        let port = probe.local_addr()?.port();
        drop(probe);

        let service_id = "svc-recovery-self-exclusion";
        let mut record = identity_probe_record(&paths, service_id, port);
        record.status = "running".to_owned();
        record.write()?;

        let failing = |_command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            Err(std::io::Error::other("spawn refused by test"))
        };
        let outcome = restart_managed_service_with(&paths, &mut record, Some(&failing));
        fs::remove_dir_all(root).ok();

        let message = format!("{:#}", outcome.expect_err("the intercepted spawn fails"));
        assert!(
            message.contains("spawn refused by test"),
            "recovery must reach the spawn, not refuse over its own record: {message}"
        );
        Ok(())
    }

    /// A `rocm services restart` that finished while recovery waited for the
    /// launch lock changed this service's own record; recovery must refuse
    /// rather than write its stale copy over it.
    #[test]
    fn recovery_refuses_when_its_own_record_changed_while_waiting_for_the_lock() -> Result<()> {
        let (root, paths) = temp_app_paths("recovery-stale-own-record");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-recovery-stale-own-record";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;
        let mut stale = record.clone();

        // The restart that won the lock: same id, now starting under new pids.
        record.status = "starting".to_owned();
        record.supervisor_pid = 4_242_424;
        record.write()?;

        let never = |_command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            panic!("recovery must not spawn over a restarted record")
        };
        let outcome = restart_managed_service_with(&paths, &mut stale, Some(&never));
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let message = format!(
            "{:#}",
            outcome.expect_err("a changed record refuses recovery")
        );
        assert!(message.contains("record changed"), "{message}");
        let persisted = persisted?;
        assert_eq!(persisted.status, "starting");
        assert_eq!(persisted.supervisor_pid, 4_242_424);
        Ok(())
    }

    /// A stop requested while recovery waited for the lock must not be cleared.
    #[test]
    fn recovery_refuses_when_a_stop_was_requested_while_waiting_for_the_lock() -> Result<()> {
        let (root, paths) = temp_app_paths("recovery-stop-requested");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-recovery-stop-requested";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;
        let mut stale = record.clone();

        record.stop_requested_unix_ms = Some(unix_time_millis());
        record.write()?;

        let never = |_command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            panic!("recovery must not spawn for a service being stopped")
        };
        let outcome = restart_managed_service_with(&paths, &mut stale, Some(&never));
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let message = format!(
            "{:#}",
            outcome.expect_err("a pending stop refuses recovery")
        );
        assert!(message.contains("stop was requested"), "{message}");
        assert!(stop_requested(&persisted?));
        Ok(())
    }

    /// A manifest deleted while recovery waited must not be resurrected.
    #[test]
    fn recovery_refuses_when_the_record_was_removed_while_waiting_for_the_lock() -> Result<()> {
        let (root, paths) = temp_app_paths("recovery-removed-record");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-recovery-removed-record";
        let mut record = identity_probe_record(&paths, service_id, free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;
        let mut stale = record;
        fs::remove_file(paths.service_manifest_path(service_id))?;

        let never = |_command: &mut ProcessCommand| -> std::io::Result<std::process::Child> {
            panic!("recovery must not spawn for a removed service")
        };
        let outcome = restart_managed_service_with(&paths, &mut stale, Some(&never));
        let resurrected = paths.service_manifest_path(service_id).exists();
        fs::remove_dir_all(root).ok();

        let message = format!(
            "{:#}",
            outcome.expect_err("a removed record refuses recovery")
        );
        assert!(message.contains("was removed"), "{message}");
        assert!(!resurrected, "recovery must not rewrite a deleted manifest");
        Ok(())
    }

    /// Recovery waits a bounded time for the launch lock, then gives up instead
    /// of queuing behind an interactive launch.
    #[test]
    fn recovery_gives_up_when_the_launch_lock_stays_held_past_its_bound() -> Result<()> {
        let (root, paths) = temp_app_paths("recovery-lock-bound");
        paths.ensure()?;
        let mut record = identity_probe_record(&paths, "svc-lock-bound", free_loopback_port());
        record.status = "failed".to_owned();
        record.write()?;

        let guard = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
        let holder = thread::spawn(move || {
            thread::sleep(Duration::from_secs(3));
            drop(guard);
        });
        let started = std::time::Instant::now();
        let outcome =
            restart_managed_service_bounded(&paths, &mut record, None, Duration::from_millis(100));
        let waited = started.elapsed();
        holder.join().expect("holder thread");
        fs::remove_dir_all(root).ok();

        assert!(outcome.is_err(), "a held lock must defer recovery");
        assert!(
            waited < Duration::from_secs(2),
            "recovery waited {waited:?} for a lock held for 3s: the wait is not bounded"
        );
        Ok(())
    }

    /// The rollback restores the prior record when handed the snapshot taken
    /// after the supervisor was recorded, which is the one the `try_wait`
    /// failure arm passes. That arm cannot be driven from here, so this covers
    /// the rollback's guard, not the choice of snapshot at that call site.
    #[test]
    fn rollback_against_the_recorded_snapshot_restores_the_prior_record() -> Result<()> {
        let (root, paths) = temp_app_paths("rollback-recorded-snapshot");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let mut prior = identity_probe_record(&paths, "svc-rollback-recorded", 11466);
        prior.status = "failed".to_owned();
        let mut recorded = prior.clone();
        recorded.status = "recovering".to_owned();
        recorded.supervisor_pid = 4_242_424;
        recorded.write()?;

        rollback_recovering_record(&paths, &recorded, &prior)?;
        let persisted = load_service_record(&paths, "svc-rollback-recorded");
        fs::remove_dir_all(root).ok();

        assert_eq!(persisted?.status, "failed");
        Ok(())
    }

    /// The rollback merges onto the live record: a stop marker persisted since
    /// the first write survives, and the restart's own fields are restored.
    #[test]
    fn rollback_keeps_a_concurrent_stop_marker_and_restores_the_restarts_fields() -> Result<()> {
        let (root, paths) = temp_app_paths("rollback-merge");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let mut prior = identity_probe_record(&paths, "svc-rollback-merge", 11464);
        prior.status = "failed".to_owned();
        // Distinct values, so a restore that is skipped shows up.
        prior.restart_count = 7;
        prior.engine_pid = Some(4_111_111);
        prior.inference_verified_at_unix_ms = Some(5_555);
        let mut recovering = prior.clone();
        recovering.status = "recovering".to_owned();
        recovering.supervisor_pid = 0;
        recovering.restart_count = 8;
        recovering.engine_pid = None;
        recovering.inference_verified_at_unix_ms = None;
        // What a concurrent stop persisted on the live `recovering` record.
        let mut on_disk = recovering.clone();
        on_disk.stop_requested_unix_ms = Some(1_234);
        on_disk.write()?;

        rollback_recovering_record(&paths, &recovering, &prior)?;
        let persisted = load_service_record(&paths, "svc-rollback-merge");
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_eq!(persisted.status, "failed");
        assert_eq!(persisted.restart_count, 7);
        assert_eq!(persisted.engine_pid, Some(4_111_111));
        assert_eq!(persisted.inference_verified_at_unix_ms, Some(5_555));
        assert_eq!(
            persisted.stop_requested_unix_ms,
            Some(1_234),
            "a stop marker persisted meanwhile must survive the rollback"
        );
        Ok(())
    }

    /// If another writer has since recorded a different supervisor, that
    /// record is truer than this restart's view and the rollback leaves it.
    #[test]
    fn rollback_leaves_a_record_that_names_a_different_supervisor() -> Result<()> {
        let (root, paths) = temp_app_paths("rollback-other-supervisor");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let mut prior = identity_probe_record(&paths, "svc-rollback-other", 11465);
        prior.status = "failed".to_owned();
        let mut recovering = prior.clone();
        recovering.status = "recovering".to_owned();
        recovering.supervisor_pid = 0;
        let mut on_disk = recovering.clone();
        on_disk.supervisor_pid = 4_242_424;
        on_disk.write()?;

        rollback_recovering_record(&paths, &recovering, &prior)?;
        let persisted = load_service_record(&paths, "svc-rollback-other");
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_eq!(
            persisted.status, "recovering",
            "must not be blindly restored"
        );
        assert_eq!(persisted.supervisor_pid, 4_242_424);
        Ok(())
    }

    /// The lease branch above proves a bound loopback port refuses recovery;
    /// the doc comment on `restart_managed_service`'s lease also claims custom
    /// hosts skip that preflight entirely, relying on the engine's own bind
    /// error instead. Binding the exact recorded port on a custom host and
    /// still reaching a spawned supervisor is what proves that claim, rather
    /// than just trusting the comment.
    ///
    /// Linux-only and ignores the restart's own outcome for the same reason as
    /// `restart_managed_service_persists_the_supervisor_identity_token`: the
    /// restart re-execs the test binary, which rejects the `supervise`
    /// arguments and exits at once, so the restart itself still reports
    /// failure — after writing the supervisor identity this test reads.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_on_a_custom_host_skips_the_loopback_lease() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-custom-host-skips-lease");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-custom-host";
        let blocked_port = free_loopback_port();
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "llamacpp",
            "a-model",
            "a-model",
            "0.0.0.0",
            blocked_port,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "failed".to_owned();
        record.write()?;

        let _blocker = std::net::TcpListener::bind(("0.0.0.0", blocked_port))?;

        let outcome = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let message = outcome.err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(
            !message.contains("its recorded port is in use"),
            "a custom host must not preflight the port with a loopback lease: {message}"
        );
        assert!(
            persisted?.supervisor_start_ticks.is_some(),
            "recovery must still reach a spawned supervisor despite the bound port"
        );
        Ok(())
    }
}
