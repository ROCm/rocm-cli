// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationRuntimeState, RocmCliConfig, WatcherRuntimeSnapshot, builtin_watchers,
    unix_time_millis,
};
use std::fs;
use std::io::{self, Write};
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
/// A pending [`acquire_autostart_lock`].
type AutostartLockWait<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Option<rocm_core::FileLock>>> + Send + 'a>,
>;

/// Pause between non-blocking attempts at the autostart lock.
const AUTOSTART_LOCK_POLL: Duration = Duration::from_millis(10);
/// The autostart lock is held only for a config read and a spawn, so a wait this
/// long means the holder is stuck; the exit attempt then gives up and the loop
/// stays up.
const AUTOSTART_LOCK_WAIT: Duration = Duration::from_secs(30);

/// Seams for the automation loop's tests: the tick period, and two callbacks
/// that let a test act at an exact point of a tick instead of racing a real
/// 5-second clock. Production passes [`LoopHooks::default`].
#[derive(Clone, Copy)]
pub(crate) struct LoopHooks<'a> {
    pub(crate) tick_interval: Duration,
    /// Called at the start of every tick, before the config is read.
    pub(crate) on_tick: Option<&'a (dyn Fn() + Sync)>,
    /// Called when a tick has read a config with no watcher enabled, before the
    /// exit decision is confirmed: the window an `automations enable` can land in.
    pub(crate) on_idle_observed: Option<&'a (dyn Fn() + Sync)>,
    /// Called each time the autostart lock is found held by someone else, so a
    /// test can wait for the loop to be genuinely waiting on it.
    pub(crate) on_lock_busy: Option<&'a (dyn Fn() + Sync)>,
    /// How long one exit attempt waits for the autostart lock before giving up
    /// (the loop stays up and decides again on a later tick). The wait runs as its
    /// own select arm, so it never delays webhook dispatch or shutdown.
    pub(crate) lock_wait: Duration,
    /// Stands in for Ctrl-C / SIGTERM, which a test cannot raise.
    pub(crate) stop: Option<&'a tokio::sync::Notify>,
}

impl Default for LoopHooks<'_> {
    fn default() -> Self {
        Self {
            tick_interval: WATCHER_TICK_INTERVAL,
            on_tick: None,
            on_idle_observed: None,
            on_lock_busy: None,
            lock_wait: AUTOSTART_LOCK_WAIT,
            stop: None,
        }
    }
}

pub(crate) async fn run_daemon(
    paths: &AppPaths,
    automations_enabled: bool,
    local_webhook_port: Option<u16>,
) -> Result<()> {
    run_daemon_with(
        paths,
        automations_enabled,
        local_webhook_port,
        LoopHooks::default(),
    )
    .await
}

pub(crate) async fn run_daemon_with(
    paths: &AppPaths,
    automations_enabled: bool,
    local_webhook_port: Option<u16>,
    hooks: LoopHooks<'_>,
) -> Result<()> {
    if local_webhook_port.is_some() && !automations_enabled {
        bail!("local webhook source requires --automations-enabled");
    }

    let config = RocmCliConfig::load(paths)?;
    let mut state = build_runtime_state(&config, automations_enabled);
    let local_webhook = if let Some(port) = local_webhook_port {
        Some(crate::webhook::start_local_webhook_source(port).await?)
    } else {
        None
    };
    let (local_webhook_endpoint, mut local_webhook_receiver, local_webhook_task) =
        match local_webhook {
            Some(source) => (
                Some(source.endpoint),
                Some(source.receiver),
                Some(source.task),
            ),
            None => (None, None, None),
        };
    state.local_webhook_endpoint = local_webhook_endpoint.clone();

    println!("rocmd run");
    println!("  automations enabled: {automations_enabled}");
    println!(
        "  lifecycle: {}",
        if automations_enabled {
            "while an automation watcher is enabled"
        } else {
            "on-demand"
        }
    );
    println!("  config: {}", paths.config_path().display());
    println!("  state: {}", paths.automation_state_path().display());
    println!(
        "  local_webhook_endpoint: {}",
        local_webhook_endpoint.as_deref().unwrap_or("disabled")
    );
    let enabled_count = state
        .active_watchers
        .iter()
        .filter(|watcher| watcher.enabled)
        .count();
    println!("  enabled watchers: {enabled_count}");
    // This banner is the foreground-loop readiness contract used by callers and
    // integration tests. Flush it before any persistent work so piped stdout on
    // Windows cannot retain the line in a userspace buffer indefinitely.
    io::stdout()
        .flush()
        .context("failed to flush rocmd run banner")?;

    if !automations_enabled {
        println!(
            "  note: rerun with --automations-enabled to run the watcher loop (it exits once no watcher is enabled)"
        );
        return Ok(());
    }

    paths.ensure()?;
    state.write(paths)?;
    crate::persistence::record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_start",
        "rocmd automation supervisor started",
        None,
    )?;
    state.write(paths)?;

    crate::watchers::evaluate_watchers(paths, &config, &mut state)?;
    state.last_tick_unix_ms = unix_time_millis();
    state.write(paths)?;

    let shutdown = async {
        match hooks.stop {
            Some(stop) => stop.notified().await,
            None => shutdown_signal().await,
        }
    };
    tokio::pin!(shutdown);

    // An idle tick's pending wait for the autostart lock; see its select arm.
    let mut exit_attempt: Option<AutostartLockWait<'_>> = None;
    let mut ticker = time::interval(hooks.tick_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Some(on_tick) = hooks.on_tick {
                    on_tick();
                }
                let config = RocmCliConfig::load(paths)?;
                // Persistent only while a watcher is enabled: once none is (none
                // was, or the last one was disabled) there is no work left, so
                // exit rather than idle forever as a detached orphan
                // (ROCMAI-827). This read is only a hint. `automations enable`
                // saves its config and then asks the autostart path whether a
                // daemon is running, so a watcher enabled after it would find
                // this process still "running" and spawn nothing. The decision
                // is therefore confirmed under the autostart lock (the
                // `exit_attempt` arm below), and "stopped" is published before
                // the lock is released: an enable either lands first (the
                // confirmation sees it and the loop stays) or finds the helper
                // stopped and starts a fresh one.
                if config.automation_daemon_enabled() {
                    // Work again: abandon any exit that was waiting for the lock.
                    exit_attempt = None;
                } else if exit_attempt.is_none() {
                    if let Some(on_idle_observed) = hooks.on_idle_observed {
                        on_idle_observed();
                    }
                    exit_attempt = Some(Box::pin(acquire_autostart_lock(paths, &hooks)));
                    continue;
                } else {
                    // Still waiting for the lock from an earlier tick.
                    continue;
                }
                crate::watchers::reconcile_watcher_snapshots(&config, &mut state);
                crate::watchers::evaluate_watchers(paths, &config, &mut state)?;
                state.last_tick_unix_ms = unix_time_millis();
                state.write(paths)?;
            }
            // The wait for the autostart lock is its own arm, not an await inside
            // the tick arm: while another process holds the lock (up to
            // `hooks.lock_wait`) the webhook and shutdown arms below stay
            // pollable. It is polled with try-lock, so no thread is parked either.
            attempt = async { exit_attempt.as_mut().expect("guarded by the arm condition").await }, if exit_attempt.is_some() => {
                exit_attempt = None;
                // `None`: the lock stayed held for the whole bounded wait. Stay
                // up; the next tick decides again.
                if let Some(_autostart_lock) = attempt? {
                    let config = RocmCliConfig::load(paths)?;
                    if !config.automation_daemon_enabled() {
                        println!("  exiting: no automation watcher is enabled");
                        io::stdout()
                            .flush()
                            .context("failed to flush rocmd exit note")?;
                        // Our own spawn claim would otherwise make the next
                        // autostart defer to this still-alive process.
                        release_own_autostart_claim(paths);
                        if let Some(task) = local_webhook_task.as_ref() {
                            task.abort();
                        }
                        return record_daemon_stop(paths, &mut state);
                    }
                }
            }
            event = crate::webhook::receive_local_webhook_event(&mut local_webhook_receiver) => {
                if let Some(event) = event {
                    let config = RocmCliConfig::load(paths)?;
                    crate::watchers::reconcile_watcher_snapshots(&config, &mut state);
                    crate::persistence::record_event(
                        paths,
                        &mut state,
                        "rocmd",
                        "info",
                        "local_webhook_event",
                        &format!(
                            "received local webhook event kind={} watcher_hint={}; dispatching through existing watcher policy; webhook payload grants no new action",
                            event.kind,
                            event.watcher_hint.as_deref().unwrap_or("<none>")
                        ),
                        event.service_id.clone(),
                    )?;
                    if let Err(error) =
                        crate::watchers::evaluate_watchers_for_events(paths, &config, &mut state, &[event])
                    {
                        crate::persistence::record_event(
                            paths,
                            &mut state,
                            "rocmd",
                            "error",
                            "local_webhook_dispatch_failed",
                            &format!(
                                "local webhook event could not be dispatched through watcher policy: {error}"
                            ),
                            None,
                        )?;
                    }
                    state.last_tick_unix_ms = unix_time_millis();
                    state.write(paths)?;
                } else {
                    local_webhook_receiver = None;
                    state.local_webhook_endpoint = None;
                    crate::persistence::record_event(
                        paths,
                        &mut state,
                        "rocmd",
                        "warn",
                        "local_webhook_stopped",
                        "local webhook source stopped; automation daemon continues without webhook ingestion",
                        None,
                    )?;
                    state.write(paths)?;
                }
            }
            () = &mut shutdown => {
                break;
            }
        }
    }

    record_daemon_stop(paths, &mut state)?;
    if let Some(task) = local_webhook_task {
        task.abort();
    }
    Ok(())
}

/// Take the autostart lock without blocking the runtime. `None` once
/// `hooks.lock_wait` has passed with the lock still held by someone else.
async fn acquire_autostart_lock(
    paths: &AppPaths,
    hooks: &LoopHooks<'_>,
) -> Result<Option<rocm_core::FileLock>> {
    let deadline = time::Instant::now() + hooks.lock_wait;
    loop {
        if let Some(lock) =
            rocm_core::FileLock::try_acquire(paths.automation_autostart_lock_path())?
        {
            return Ok(Some(lock));
        }
        if let Some(on_lock_busy) = hooks.on_lock_busy {
            on_lock_busy();
        }
        if time::Instant::now() >= deadline {
            return Ok(None);
        }
        time::sleep(AUTOSTART_LOCK_POLL).await;
    }
}

/// Publish that the loop is no longer running.
fn record_daemon_stop(paths: &AppPaths, state: &mut AutomationRuntimeState) -> Result<()> {
    state.running = false;
    state.last_tick_unix_ms = unix_time_millis();
    state.local_webhook_endpoint = None;
    crate::persistence::record_event(
        paths,
        state,
        "rocmd",
        "info",
        "daemon_stop",
        "rocmd automation supervisor stopped",
        None,
    )?;
    state.write(paths)
}

/// Remove the autostart claim (`"<pid> <ms>"`) if it names this process.
/// Best-effort: a leftover claim only delays a restart until its TTL.
fn release_own_autostart_claim(paths: &AppPaths) {
    let claim_path = paths.automation_autostart_claim_path();
    let own = std::fs::read_to_string(&claim_path).is_ok_and(|text| {
        text.split_whitespace()
            .next()
            .and_then(|pid| pid.parse::<u32>().ok())
            == Some(std::process::id())
    });
    if own {
        let _ = fs::remove_file(claim_path);
    }
}

pub(crate) fn print_status(paths: &AppPaths) -> Result<()> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    println!("rocmd status");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  data dir: {}", paths.data_dir.display());
    println!(
        "  policy: on-demand; the watcher loop runs only while an automation watcher is enabled"
    );
    println!(
        "  automations desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    match AutomationRuntimeState::load(paths)? {
        Some(state) => {
            println!(
                "  automations runtime: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
            println!(
                "  local_webhook_endpoint: {}",
                state
                    .local_webhook_endpoint
                    .as_deref()
                    .unwrap_or("disabled")
            );
            for watcher in state
                .active_watchers
                .into_iter()
                .filter(|watcher| watcher.enabled)
            {
                println!(
                    "  watcher {} mode={} last_event={}",
                    watcher.id,
                    watcher.mode.as_str(),
                    watcher.last_event.as_deref().unwrap_or("<none>")
                );
            }
        }
        None => println!("  automations runtime: inactive"),
    }
    println!(
        "  automation events: {}",
        paths.automation_events_path().display()
    );
    println!("  audit events: {}", paths.audit_events_path().display());

    let records = crate::persistence::load_managed_services(paths)?;
    if records.is_empty() {
        println!("  services: none");
        return Ok(());
    }

    for record in records {
        println!(
            "  service {} engine={} status={} endpoint={}",
            record.service_id, record.engine, record.status, record.endpoint_url
        );
    }

    Ok(())
}

fn build_runtime_state(
    config: &RocmCliConfig,
    automations_enabled: bool,
) -> AutomationRuntimeState {
    let now = unix_time_millis();
    let active_watchers = builtin_watchers()
        .iter()
        .map(|watcher| WatcherRuntimeSnapshot {
            id: watcher.id.to_owned(),
            enabled: config.watcher_enabled(watcher),
            mode: config.effective_watcher_mode(watcher),
            summary: watcher.summary.to_owned(),
            last_event: None,
            last_event_unix_ms: None,
        })
        .collect();
    AutomationRuntimeState {
        running: automations_enabled,
        automations_enabled,
        daemon_pid: std::process::id(),
        started_at_unix_ms: now,
        last_tick_unix_ms: now,
        local_webhook_endpoint: None,
        active_watchers,
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::thread;

    #[tokio::test]
    async fn local_webhook_requires_enabled_automation_loop() {
        let (_root, paths) = temp_app_paths("local-webhook-requires-loop");
        let error = run_daemon(&paths, false, Some(0)).await.unwrap_err();

        assert!(error.to_string().contains("requires --automations-enabled"));
    }

    /// ROCMAI-827: the automation loop used to run forever even with nothing to
    /// do, so a daemon `rocm serve` autostarted stayed alive after every service
    /// was stopped. With no automation watcher enabled it now exits by itself and
    /// records that it is no longer running.
    #[tokio::test]
    async fn automation_loop_exits_when_no_watcher_is_enabled() -> Result<()> {
        let (root, paths) = temp_app_paths("loop-exits-without-watchers");
        paths.ensure()?;
        RocmCliConfig::default().save(&paths)?;

        let finished =
            tokio::time::timeout(Duration::from_secs(30), run_daemon(&paths, true, None)).await;
        let state = AutomationRuntimeState::load(&paths)?;
        let _ = fs::remove_dir_all(root);

        finished.expect("the loop must exit on its own with no watcher enabled")?;
        let state = state.expect("the loop records its runtime state");
        assert!(!state.running, "an exited loop must not read as running");
        Ok(())
    }

    fn set_watcher(paths: &AppPaths, enabled: bool) {
        let mut config = RocmCliConfig::load(paths).expect("load config");
        config.watcher_config_mut("server-recover").enabled = enabled;
        config.automations.daemon_enabled = enabled;
        config.save(paths).expect("save config");
    }

    fn fast_ticks<'a>(
        on_tick: &'a (dyn Fn() + Sync),
        on_idle_observed: Option<&'a (dyn Fn() + Sync)>,
    ) -> LoopHooks<'a> {
        LoopHooks {
            tick_interval: Duration::from_millis(20),
            on_tick: Some(on_tick),
            on_idle_observed,
            ..LoopHooks::default()
        }
    }

    /// The loop's other half: while a watcher is enabled it keeps running across
    /// ticks, not just the first one (an interval fires immediately), and it still
    /// exits once the watcher is disabled. The test disables the watcher only
    /// after several ticks, so an exit check that misfired on a later tick (or a
    /// loop that stopped after the first) would show in the tick count.
    #[tokio::test]
    async fn automation_loop_keeps_running_while_automations_are_enabled() -> Result<()> {
        use std::sync::atomic::{AtomicU32, Ordering};

        const TICKS_WITH_WATCHER: u32 = 4;
        let (root, paths) = temp_app_paths("loop-persists-with-automations");
        paths.ensure()?;
        RocmCliConfig::default().save(&paths)?;
        set_watcher(&paths, true);

        let ticks = AtomicU32::new(0);
        let on_tick = || {
            if ticks.fetch_add(1, Ordering::SeqCst) + 1 > TICKS_WITH_WATCHER {
                set_watcher(&paths, false);
            }
        };
        let hooks = fast_ticks(&on_tick, None);
        let finished = time::timeout(
            Duration::from_secs(30),
            run_daemon_with(&paths, true, None, hooks),
        )
        .await;
        let state = AutomationRuntimeState::load(&paths)?;
        let _ = fs::remove_dir_all(root);

        finished.expect("the loop must exit once the watcher is disabled")?;
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            TICKS_WITH_WATCHER + 1,
            "the loop must run every tick while the watcher is enabled, and stop on the first tick after it is disabled"
        );
        assert!(!state.expect("runtime state").running);
        Ok(())
    }

    /// ROCMAI-827 race: the loop read "no watcher enabled", and an
    /// `automations enable` saved its config before the loop acted on that. The
    /// enable's autostart check would see this process as running and spawn
    /// nothing, so exiting here would leave a watcher enabled with no daemon. The
    /// exit must be re-decided against the current config and not happen.
    #[tokio::test]
    async fn automation_loop_stays_when_a_watcher_is_enabled_inside_the_exit_window() -> Result<()>
    {
        use std::sync::atomic::{AtomicU32, Ordering};

        let (root, paths) = temp_app_paths("loop-exit-window");
        paths.ensure()?;
        RocmCliConfig::default().save(&paths)?;
        set_watcher(&paths, true);

        let ticks = AtomicU32::new(0);
        let window_hits = AtomicU32::new(0);
        // Tick 2: the last watcher is disabled, so the loop observes "idle".
        // That observation is the window: the watcher is enabled again before
        // the loop confirms the exit. Tick 6 disables it for good (no window
        // re-enable) so the loop can end and the test can finish.
        let on_tick = || {
            let tick = ticks.fetch_add(1, Ordering::SeqCst) + 1;
            if tick == 2 || tick == 6 {
                set_watcher(&paths, false);
            }
        };
        let on_idle = || {
            if window_hits.fetch_add(1, Ordering::SeqCst) == 0 {
                set_watcher(&paths, true);
            }
        };
        let hooks = fast_ticks(&on_tick, Some(&on_idle));
        let finished = time::timeout(
            Duration::from_secs(30),
            run_daemon_with(&paths, true, None, hooks),
        )
        .await;
        let enabled_when_loop_ended = RocmCliConfig::load(&paths)?.automation_daemon_enabled();
        let _ = fs::remove_dir_all(root);

        finished.expect("the loop must end once the watcher stays disabled")?;
        assert_eq!(
            window_hits.load(Ordering::SeqCst),
            2,
            "exactly the window and the real exit"
        );
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            6,
            "the loop must survive the tick where a watcher was re-enabled in the window"
        );
        assert!(!enabled_when_loop_ended);
        Ok(())
    }

    /// Poll `condition` (yielding, no fixed pause) until it holds, so a test waits
    /// for an observed event rather than assuming how long it takes.
    fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_mins(1);
        while !condition() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            thread::yield_now();
        }
    }

    /// Run the loop on its own thread and runtime, next to a task that counts
    /// ticks of its own 1 ms timer. Returns the thread and that counter.
    fn spawn_daemon_beside_a_counter(
        paths: &AppPaths,
        hooks: LoopHooks<'static>,
    ) -> (
        thread::JoinHandle<Result<()>>,
        std::sync::Arc<std::sync::atomic::AtomicU32>,
    ) {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let beat = counter.clone();
        let paths = paths.clone();
        let daemon = thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(async move {
                    tokio::spawn(async move {
                        loop {
                            time::sleep(Duration::from_millis(1)).await;
                            beat.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    });
                    run_daemon_with(&paths, true, None, hooks).await
                })
        });
        (daemon, counter)
    }

    /// The exit is published under the autostart lock: while another caller holds
    /// it (an `automations enable` mid-decision) the loop does not report
    /// "stopped" or exit, and it clears its own spawn claim when it does. The wait
    /// is non-blocking: a task sharing the loop's single-threaded runtime keeps
    /// running meanwhile (a blocking acquire would stall it, and with it the
    /// webhook task). No fixed pause: the test waits until the loop has actually
    /// found the lock held.
    #[test]
    fn automation_loop_exit_waits_for_the_autostart_lock_and_clears_its_claim() -> Result<()> {
        use std::sync::atomic::{AtomicU32, Ordering};

        static BUSY: AtomicU32 = AtomicU32::new(0);
        static ON_BUSY: fn() = || {
            BUSY.fetch_add(1, Ordering::SeqCst);
        };
        BUSY.store(0, Ordering::SeqCst);

        let (root, paths) = temp_app_paths("loop-exit-lock");
        paths.ensure()?;
        RocmCliConfig::default().save(&paths)?;
        std::fs::write(
            paths.automation_autostart_claim_path(),
            format!("{} 1", std::process::id()),
        )?;

        let guard = rocm_core::FileLock::acquire(paths.automation_autostart_lock_path())?;
        let (daemon, beat) = spawn_daemon_beside_a_counter(
            &paths,
            LoopHooks {
                tick_interval: Duration::from_millis(20),
                on_lock_busy: Some(&ON_BUSY),
                ..LoopHooks::default()
            },
        );
        wait_until("the loop to find the lock held", || {
            BUSY.load(Ordering::SeqCst) >= 3
        });
        let before = beat.load(Ordering::SeqCst);
        wait_until("the runtime to keep running while waiting", || {
            beat.load(Ordering::SeqCst) >= before + 3
        });
        let finished_while_locked = daemon.is_finished();
        let running_while_locked =
            AutomationRuntimeState::load(&paths)?.is_some_and(|state| state.running);
        drop(guard);
        daemon.join().expect("daemon thread")?;
        let state = AutomationRuntimeState::load(&paths)?;
        let claim_left = paths.automation_autostart_claim_path().exists();
        let _ = fs::remove_dir_all(root);

        assert!(!finished_while_locked, "exit must wait for the lock");
        assert!(
            running_while_locked,
            "still published as running until then"
        );
        assert!(!state.expect("runtime state").running);
        assert!(!claim_left, "the exiting helper must clear its own claim");
        Ok(())
    }

    /// A holder that never lets go must not wedge the loop: after the bounded wait
    /// the tick gives up on exiting, the loop stays up and tries again on later
    /// ticks, and once the lock is free it exits.
    #[test]
    fn automation_loop_gives_up_a_stuck_lock_wait_and_exits_once_it_is_free() -> Result<()> {
        use std::sync::atomic::{AtomicU32, Ordering};

        static TICKS: AtomicU32 = AtomicU32::new(0);
        static ON_TICK: fn() = || {
            TICKS.fetch_add(1, Ordering::SeqCst);
        };
        TICKS.store(0, Ordering::SeqCst);

        let (root, paths) = temp_app_paths("loop-exit-stuck-lock");
        paths.ensure()?;
        RocmCliConfig::default().save(&paths)?;

        let guard = rocm_core::FileLock::acquire(paths.automation_autostart_lock_path())?;
        let (daemon, _beat) = spawn_daemon_beside_a_counter(
            &paths,
            LoopHooks {
                tick_interval: Duration::from_millis(20),
                on_tick: Some(&ON_TICK),
                lock_wait: Duration::from_millis(30),
                ..LoopHooks::default()
            },
        );
        // Several ticks have each waited out the bound and given up.
        wait_until("several timed-out exit attempts", || {
            TICKS.load(Ordering::SeqCst) >= 6
        });
        let finished_while_stuck = daemon.is_finished();
        drop(guard);
        daemon.join().expect("daemon thread")?;
        let state = AutomationRuntimeState::load(&paths)?;
        let _ = fs::remove_dir_all(root);

        assert!(!finished_while_stuck, "a stuck lock must not make it exit");
        assert!(!state.expect("runtime state").running);
        Ok(())
    }

    /// The wait for the autostart lock is its own select arm: while it is waiting
    /// on a held lock, a shutdown request is still acted on at once instead of
    /// after the (up to 30 s) wait. The lock stays held until the loop has
    /// returned, so the exit cannot have come from the lock being released.
    #[test]
    fn automation_loop_shuts_down_while_waiting_for_a_held_autostart_lock() -> Result<()> {
        use std::sync::atomic::{AtomicU32, Ordering};

        static BUSY: AtomicU32 = AtomicU32::new(0);
        static ON_BUSY: fn() = || {
            BUSY.fetch_add(1, Ordering::SeqCst);
        };
        static STOP: tokio::sync::Notify = tokio::sync::Notify::const_new();
        BUSY.store(0, Ordering::SeqCst);

        let (root, paths) = temp_app_paths("loop-shutdown-while-locked");
        paths.ensure()?;
        RocmCliConfig::default().save(&paths)?;

        let guard = rocm_core::FileLock::acquire(paths.automation_autostart_lock_path())?;
        let (daemon, _beat) = spawn_daemon_beside_a_counter(
            &paths,
            LoopHooks {
                tick_interval: Duration::from_millis(20),
                on_lock_busy: Some(&ON_BUSY),
                stop: Some(&STOP),
                // Longer than the test's own deadline: only a wait that really
                // runs beside the shutdown arm lets the loop stop with the lock held.
                lock_wait: Duration::from_mins(10),
                ..LoopHooks::default()
            },
        );
        wait_until("the loop to be waiting on the held lock", || {
            BUSY.load(Ordering::SeqCst) >= 2
        });
        STOP.notify_one();
        wait_until("the loop to shut down with the lock still held", || {
            daemon.is_finished()
        });
        drop(guard);
        daemon.join().expect("daemon thread")?;
        let state = AutomationRuntimeState::load(&paths)?;
        let _ = fs::remove_dir_all(root);

        assert!(!state.expect("runtime state").running);
        Ok(())
    }
}
