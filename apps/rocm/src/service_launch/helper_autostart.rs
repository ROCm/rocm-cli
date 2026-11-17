// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Starting the background automation helper (`rocm daemon`) when a watcher needs
//! it, with the autostart claim that keeps concurrent callers from spawning two.

#[cfg(not(windows))]
use super::spawn::{
    attach_background_stdio, detach_background_command, managed_service_process_command,
};
use crate::background_helper::background_helper_already_running;
use crate::engine_runtime_select::managed_service_launcher_path;
#[cfg(windows)]
use crate::engine_runtime_select::{app_path_env_var_refs, app_path_env_var_values};
use anyhow::{Context, Result};
use rocm_core::AppPaths;
use std::path::Path;
#[cfg(not(windows))]
use std::process::Stdio;

/// Shared daemon-lifecycle entrypoint: ensures the background automation helper
/// (`rocm daemon`) is running when an automation watcher needs it, spawning it
/// detached if not. Liveness is read from the file-based automation runtime
/// state. Intentionally `pub(crate)` — reused by both the `serve --managed` path
/// and `automations enable`. Only the spawn result itself (`command.spawn()` /
/// `spawn_detached_no_inherit`) is logged rather than propagated; setup errors
/// (path discovery, config load, stdio attach) still return `Err`.
///
/// The daemon's only work is running enabled automation watchers (server
/// recovery included — `server-recover` is one of them and is off by default),
/// and it exits on its own once none is enabled. So with no watcher enabled
/// nothing is spawned: a `serve` would otherwise leave a no-op daemon running
/// long after its service was stopped (ROCMAI-827).
pub(crate) fn ensure_background_helper_running() -> Result<()> {
    ensure_background_helper_running_quiet(false)
}

/// As [`ensure_background_helper_running`], but suppresses the stdout status line
/// when `quiet` is set. The interactive `rocm serve` summary path uses `quiet` so
/// the daemon-spawn note does not appear above the deployment summary table.
pub(crate) fn ensure_background_helper_running_quiet(quiet: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    ensure_background_helper_running_with(&paths, quiet, spawn_background_helper, &|| {})
}

/// The body of [`ensure_background_helper_running_quiet`], with the actual
/// process spawn injected so the decision around it is testable without
/// launching a daemon. `spawn` returns `Err` for a setup failure (propagated)
/// and `Ok(Err(..))` for a failed spawn (reported, not fatal). `on_lock_busy` is
/// called each time the autostart lock is found held by another process, which
/// lets a test wait for this call to be genuinely queued behind it.
fn ensure_background_helper_running_with(
    paths: &AppPaths,
    quiet: bool,
    spawn: impl FnOnce(&AppPaths) -> Result<Result<u32>>,
    on_lock_busy: &dyn Fn(),
) -> Result<()> {
    // An unreadable config is "no watcher": this also runs after `serve` has
    // already launched its child, where failing would report a launch that
    // happened as an error.
    if !rocm_core::RocmCliConfig::load(paths).is_ok_and(|config| config.automation_daemon_enabled())
    {
        return Ok(());
    }

    // Two concurrent callers (e.g. two `rocm serve`) can both read "not running"
    // and each spawn a daemon. Serialize the decision on a lock file — the first
    // holder spawns, later holders observe the now-running daemon and return
    // without spawning.
    //
    // The "already running" read is deliberately made only under the lock, with
    // no lock-free fast path. The daemon exits by itself once the last watcher
    // is disabled, and it decides that under this same lock and publishes
    // `running = false` before releasing it. Caller order is: save the enabling
    // config, then come here. A lock-free read could still see the daemon that is
    // about to exit as running, spawn nothing, and leave a watcher enabled with
    // no daemon; under the lock the daemon has either seen the new config and
    // stays, or has already published that it stopped.
    let _autostart_lock = loop {
        if let Some(lock) =
            rocm_core::FileLock::try_acquire(paths.automation_autostart_lock_path())?
        {
            break lock;
        }
        on_lock_busy();
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    if background_helper_already_running(paths)? {
        return Ok(());
    }

    // The lock alone does not close the window: the spawned daemon does not
    // publish its `running` runtime state until well after `spawn()` (clap parse,
    // runtime build, config load, banner flush). A second caller that acquires
    // this lock during that gap still sees "not running" and would spawn a
    // duplicate. Bridge the gap with a short-lived claim recording the child PID
    // and spawn time: a holder that finds a live, recent claim defers instead.
    let claim_path = paths.automation_autostart_claim_path();
    if autostart_spawn_in_flight(
        read_autostart_claim(&claim_path),
        now_unix_millis(),
        AUTOSTART_CLAIM_TTL_MS,
        rocm_core::process_is_running,
    ) {
        return Ok(());
    }

    match spawn(paths)? {
        Ok(daemon_pid) => {
            // Record the claim before returning (and thus releasing the lock) so a
            // concurrent holder in the spawn→publish window defers. Best-effort: a
            // failed write only reopens the original, already-tolerated race.
            let _ = write_autostart_claim(
                &claim_path,
                AutostartClaim {
                    daemon_pid,
                    spawned_at_ms: now_unix_millis(),
                },
            );
            if !quiet {
                println!("  helper: started background automation daemon");
            }
        }
        Err(error) if !quiet => {
            println!("  helper: could not start background automation daemon: {error}");
        }
        Err(_) => {}
    }
    Ok(())
}

/// Spawn `rocm daemon` detached from this process. The outer `Result` is a setup
/// failure (resolving the executable, attaching stdio); the inner one is the
/// spawn itself.
fn spawn_background_helper(paths: &AppPaths) -> Result<Result<u32>> {
    let exe = managed_service_launcher_path()
        .context("failed to resolve current rocm executable path")?;
    let args = vec!["daemon".to_owned()];
    #[cfg(windows)]
    let spawn_result = {
        let env_values = app_path_env_var_values(paths, None);
        let env_refs = app_path_env_var_refs(&env_values);
        rocm_core::spawn_detached_no_inherit(&exe, &args, &env_refs)
    };
    #[cfg(not(windows))]
    let spawn_result = {
        let mut command = managed_service_process_command(&exe, &args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, None)?;
        detach_background_command(&mut command);
        paths.apply_env(&mut command);
        command
            .spawn()
            .map(|child| child.id())
            .map_err(anyhow::Error::from)
    };
    Ok(spawn_result)
}

/// How long an autostart claim is honoured before it is treated as stale even if
/// its recorded PID is still alive. Comfortably longer than a cold daemon boot
/// (clap parse → runtime build → config load → state publish) yet short enough
/// that a crashed spawn cannot suppress autostart for long.
const AUTOSTART_CLAIM_TTL_MS: u128 = 30_000;

/// A just-spawned daemon's autostart claim: the child PID and the wall-clock time
/// (milliseconds since the Unix epoch) the spawn was recorded. It lets a
/// concurrent autostart holder distinguish a live, in-flight spawn from a stale
/// leftover. Serialized as a single `"<pid> <ms>"` line — no dependency and
/// trivially forward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutostartClaim {
    daemon_pid: u32,
    spawned_at_ms: u128,
}

/// Milliseconds since the Unix epoch, or `0` if the clock is before the epoch
/// (which only makes a fresh claim look old — safe, it just permits a respawn).
fn now_unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis())
}

/// Read an autostart claim, returning `None` when the file is absent or
/// unparseable (either is treated as "no claim", so a respawn is permitted).
fn read_autostart_claim(path: &Path) -> Option<AutostartClaim> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut parts = text.split_whitespace();
    let daemon_pid = parts.next()?.parse().ok()?;
    let spawned_at_ms = parts.next()?.parse().ok()?;
    Some(AutostartClaim {
        daemon_pid,
        spawned_at_ms,
    })
}

/// Write an autostart claim as `"<pid> <ms>"`. Best-effort at the call site.
fn write_autostart_claim(path: &Path, claim: AutostartClaim) -> std::io::Result<()> {
    std::fs::write(
        path,
        format!("{} {}", claim.daemon_pid, claim.spawned_at_ms),
    )
}

/// Whether an existing autostart `claim` means a daemon spawn is still in flight,
/// so the current lock holder should defer rather than spawn a duplicate. A claim
/// counts as in-flight only while its child PID is alive *and* it is younger than
/// `ttl_ms` — the TTL bounds how long a crashed spawn (or a PID later reused by an
/// unrelated process) can suppress autostart. `pid_alive` is injected so the
/// decision is unit-testable without a live process.
fn autostart_spawn_in_flight(
    claim: Option<AutostartClaim>,
    now_ms: u128,
    ttl_ms: u128,
    pid_alive: impl Fn(u32) -> bool,
) -> bool {
    claim.is_some_and(|claim| {
        now_ms.saturating_sub(claim.spawned_at_ms) < ttl_ms && pid_alive(claim.daemon_pid)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{runtime_state, set_helper_facts};
    use crate::tests::test_paths;
    use std::fs;

    /// ROCMAI-827: `rocm serve` started the background daemon on every launch,
    /// and with no automation watcher enabled that daemon had nothing to do yet
    /// kept running after every service was stopped. With the default config
    /// (no watcher enabled) the helper must not be spawned at all.
    #[test]
    fn background_helper_not_spawned_when_no_automation_is_enabled() {
        let (root, paths) = test_paths("helper-no-automation");
        paths.ensure().expect("create app dirs");
        // The stored flag alone is not "a watcher is enabled": nothing would run.
        let mut config = rocm_core::RocmCliConfig::default();
        config.automations.daemon_enabled = true;
        config.save(&paths).expect("save default config");

        let mut spawned = false;
        ensure_background_helper_running_with(
            &paths,
            true,
            |_| {
                spawned = true;
                Ok(Ok(4242))
            },
            &|| {},
        )
        .expect("ensure helper");

        let claim_written = paths.automation_autostart_claim_path().exists();
        let _ = fs::remove_dir_all(&root);
        assert!(
            !spawned,
            "no watcher enabled ⇒ the daemon has no work and must not be started"
        );
        assert!(!claim_written, "no spawn ⇒ no autostart claim");
    }

    /// The other half of the gate: once a watcher is enabled the helper is still
    /// started (exactly once) and its autostart claim recorded.
    #[test]
    fn background_helper_spawned_when_an_automation_is_enabled() {
        let (root, paths) = test_paths("helper-automation-enabled");
        paths.ensure().expect("create app dirs");
        let mut config = rocm_core::RocmCliConfig::default();
        config.watcher_config_mut("server-recover").enabled = true;
        config.save(&paths).expect("save config");

        let mut spawns = 0;
        ensure_background_helper_running_with(
            &paths,
            true,
            |_| {
                spawns += 1;
                Ok(Ok(4242))
            },
            &|| {},
        )
        .expect("ensure helper");

        let claim = read_autostart_claim(&paths.automation_autostart_claim_path());
        let _ = fs::remove_dir_all(&root);
        assert_eq!(spawns, 1, "an enabled watcher needs the daemon");
        assert_eq!(claim.map(|claim| claim.daemon_pid), Some(4242));
    }

    /// ROCMAI-827 race, enable side: the daemon that is about to exit still has
    /// `running = true` and a live pid in its state when `automations enable`
    /// (config already saved) asks for a helper. It decides under the autostart
    /// lock and publishes "stopped" before releasing it, so the enable must wait
    /// on that lock and then start a new helper. A lock-free "already running"
    /// read would return here without spawning, leaving a watcher with no daemon.
    #[test]
    fn background_helper_is_started_when_the_exiting_daemon_publishes_stopped() {
        let (root, paths) = test_paths("helper-exit-window");
        paths.ensure().expect("create app dirs");
        let mut config = rocm_core::RocmCliConfig::default();
        config.watcher_config_mut("server-recover").enabled = true;
        config.save(&paths).expect("save config");
        // The old daemon: alive (this process stands in) and still `running`.
        runtime_state(true, std::process::id())
            .write(&paths)
            .expect("write runtime state");

        // The exiting daemon holds the autostart lock while it decides.
        let exiting_daemon_lock =
            rocm_core::FileLock::acquire(paths.automation_autostart_lock_path())
                .expect("hold the lock");
        let enable_paths = paths.clone();
        let queued_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let queued = queued_flag.clone();
        let enable = std::thread::spawn(move || {
            let mut spawns = 0;
            ensure_background_helper_running_with(
                &enable_paths,
                true,
                |_| {
                    spawns += 1;
                    Ok(Ok(4242))
                },
                &|| queued.store(true, std::sync::atomic::Ordering::SeqCst),
            )
            .expect("ensure helper");
            spawns
        });
        // Wait until the enable path has actually found the lock held and queued
        // behind it, rather than assuming how long that takes.
        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(1);
        // A caller that skipped the lock finishes without ever queuing.
        while !queued_flag.load(std::sync::atomic::Ordering::SeqCst) && !enable.is_finished() {
            assert!(std::time::Instant::now() < deadline, "enable never queued");
            std::thread::yield_now();
        }
        runtime_state(false, std::process::id())
            .write(&paths)
            .expect("daemon publishes stopped");
        drop(exiting_daemon_lock);

        let spawns = enable.join().expect("enable thread");
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            spawns, 1,
            "the enable must start a helper once the old one has published it stopped"
        );
    }

    fn stop_result() -> serde_json::Value {
        serde_json::json!({"output": {"result": {
            "service": {"service_id": "svc-a", "status": "stopped"},
            "signaled_pids": [1]
        }}})
    }

    /// `rocm services stop` prints the note for a stop only, and only the variant
    /// that matches the state.
    #[test]
    fn stop_output_carries_the_helper_note_that_matches_the_state() {
        use crate::services_text::render_service_action_result_with_helper_note as render;

        let (root, paths) = test_paths("stop-helper-note");
        paths.ensure().expect("create app dirs");
        set_helper_facts(&paths, false, false);
        let quiet = render(&paths, "stop_server", &stop_result());
        set_helper_facts(&paths, true, true);
        let outlives = render(&paths, "stop_server", &stop_result());
        let restart = render(&paths, "restart_server", &stop_result());
        set_helper_facts(&paths, true, false);
        let not_running = render(&paths, "stop_server", &stop_result());
        let _ = fs::remove_dir_all(&root);

        assert!(!quiet.contains("background automation helper"), "{quiet}");
        assert!(
            outlives.contains("stopped processes: 1")
                && outlives.contains("keeps running while an automation is enabled")
                && outlives.contains("`rocm automations disable server-recover`"),
            "{outlives}"
        );
        assert!(
            not_running.contains("is not running") && !not_running.contains("keeps running"),
            "{not_running}"
        );
        assert!(
            !restart.contains("background automation helper"),
            "{restart}"
        );
    }

    /// `serve` calls the helper autostart after its child is already running. An
    /// unreadable config must not turn that launch into an error: it reads as "no
    /// watcher enabled" and nothing is spawned.
    #[test]
    fn background_helper_autostart_treats_an_unreadable_config_as_no_watcher() {
        let (root, paths) = test_paths("helper-bad-config");
        paths.ensure().expect("create app dirs");
        fs::write(paths.config_path(), b"{ this is not json").expect("write bad config");
        assert!(rocm_core::RocmCliConfig::load(&paths).is_err());

        let mut spawned = false;
        let result = ensure_background_helper_running_with(
            &paths,
            true,
            |_| {
                spawned = true;
                Ok(Ok(4242))
            },
            &|| {},
        );
        let _ = fs::remove_dir_all(&root);

        assert!(result.is_ok(), "{result:?}");
        assert!(!spawned);
    }

    #[test]
    fn autostart_spawn_in_flight_defers_only_for_a_live_recent_claim() {
        let claim = AutostartClaim {
            daemon_pid: 4242,
            spawned_at_ms: 1_000,
        };
        // A recent claim whose PID is still alive ⇒ a spawn is in flight, defer.
        assert!(autostart_spawn_in_flight(
            Some(claim),
            1_000 + 5_000,
            AUTOSTART_CLAIM_TTL_MS,
            |_| true,
        ));
        // Same claim but the child PID is gone (crashed spawn) ⇒ respawn.
        assert!(!autostart_spawn_in_flight(
            Some(claim),
            1_000 + 5_000,
            AUTOSTART_CLAIM_TTL_MS,
            |_| false,
        ));
        // Older than the TTL, even with a live PID (possible PID reuse) ⇒ respawn.
        assert!(!autostart_spawn_in_flight(
            Some(claim),
            1_000 + AUTOSTART_CLAIM_TTL_MS,
            AUTOSTART_CLAIM_TTL_MS,
            |_| true,
        ));
        // No claim at all ⇒ nothing in flight, spawn.
        assert!(!autostart_spawn_in_flight(
            None,
            1_000,
            AUTOSTART_CLAIM_TTL_MS,
            |_| true,
        ));
    }

    #[test]
    fn autostart_claim_round_trips_and_absent_or_garbage_reads_as_none() {
        let (root, paths) = test_paths("autostart-claim-roundtrip");
        let path = paths.automation_autostart_claim_path();
        fs::create_dir_all(path.parent().expect("claim parent")).expect("mkdir claim dir");
        // Absent file ⇒ no claim.
        assert_eq!(read_autostart_claim(&path), None);
        // Round-trip a written claim.
        let claim = AutostartClaim {
            daemon_pid: 9987,
            spawned_at_ms: 1_726_000_000_123,
        };
        write_autostart_claim(&path, claim).expect("write claim");
        assert_eq!(read_autostart_claim(&path), Some(claim));
        // Garbage / partial content ⇒ no claim (permits a respawn, never panics).
        fs::write(&path, "not-a-pid").expect("write garbage");
        assert_eq!(read_autostart_claim(&path), None);
        let _ = fs::remove_dir_all(&root);
    }
}
