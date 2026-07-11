// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Client-side (CLI/TUI process) structured logging.
//!
//! Wires a file-only, non-blocking `tracing` subscriber so the `info!` /
//! `warn!` / `debug!` calls scattered across `rocm-dash-tui` and
//! `rocm-dash-daemon` (previously silently dropped — no subscriber was ever
//! installed) land in a real log file. CRITICAL: the writer must never touch
//! stdout/stderr — the TUI owns the raw-mode terminal, and a stray write
//! there would corrupt the display.

use std::path::Path;

use rocm_core::AppPaths;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

/// File name prefix for the daily-rotated client log
/// (`~/.rocm/logs/rocm-cli.log.<date>`), sibling to the daemon's
/// `rocmdashd.log` under the same canonical `AppPaths` root.
const LOG_FILE_PREFIX: &str = "rocm-cli.log";

/// Number of rotated daily log files retained; older ones are pruned on
/// startup so a long-lived install never grows the logs directory unbounded.
const MAX_RETAINED_LOGS: usize = 7;

/// Default filter applied when `RUST_LOG` is unset: `info` for our own
/// crates (loud enough to trace a hung chat request), `warn` for everything
/// else (dependency noise stays out of the file).
const DEFAULT_FILTER: &str = "warn,rocm=info,rocm_dash_tui=info,rocm_dash_daemon=info";

/// Initialize the process-wide `tracing` subscriber, writing to a daily-
/// rotated file under `paths.client_log_dir()`.
///
/// Returns the [`WorkerGuard`] the caller must keep alive for the process
/// lifetime — dropping it flushes and stops the non-blocking writer, so
/// letting it fall out of scope early silently truncates the log. Returns
/// `None` if the log directory couldn't be created or a global subscriber is
/// already installed (e.g. a second call in the same process); either case
/// degrades to no logging rather than a startup failure.
pub fn init(paths: &AppPaths) -> Option<WorkerGuard> {
    let log_dir = paths.client_log_dir();
    std::fs::create_dir_all(&log_dir).ok()?;
    prune_old_logs(&log_dir);

    let file_appender = tracing_appender::rolling::daily(&log_dir, LOG_FILE_PREFIX);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .ok()
        .map(|()| guard)
}

/// Remove rotated log files beyond [`MAX_RETAINED_LOGS`], keeping the logs
/// directory bounded for long-lived installs. Best-effort: any I/O error
/// just leaves the file in place for a future run to retry.
fn prune_old_logs(log_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(log_dir) else {
        return;
    };
    let mut candidates: Vec<_> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(LOG_FILE_PREFIX) && name != LOG_FILE_PREFIX)
        })
        .collect();
    if candidates.len() <= MAX_RETAINED_LOGS {
        return;
    }
    // Daily rotation suffixes sort lexicographically by date, so the oldest
    // files are simply the front of the sorted list.
    candidates.sort_by_key(std::fs::DirEntry::file_name);
    let excess = candidates.len() - MAX_RETAINED_LOGS;
    for entry in candidates.into_iter().take(excess) {
        let _ = std::fs::remove_file(entry.path());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths(name: &str) -> (std::path::PathBuf, AppPaths) {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("logging")
            .join(format!(
                "{name}-{}-{}",
                std::process::id(),
                rocm_core::unix_time_millis()
            ));
        let _ = std::fs::remove_dir_all(&root);
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    #[test]
    fn init_creates_log_dir_and_a_log_file() {
        let (root, paths) = temp_paths("init-creates-dir");
        let guard = init(&paths);
        assert!(guard.is_some(), "first init in-process should install ok");
        tracing::info!("hello from the test subscriber");
        // Drop the guard to flush the non-blocking writer before inspecting
        // the directory contents.
        drop(guard);

        let log_dir = paths.client_log_dir();
        assert!(log_dir.is_dir());
        let has_log_file = std::fs::read_dir(&log_dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(LOG_FILE_PREFIX))
            });
        assert!(
            has_log_file,
            "expected a rocm-cli.log.* file in {log_dir:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn init_is_idempotent_when_a_global_subscriber_already_exists() {
        let (root, paths) = temp_paths("init-idempotent");
        // A global tracing subscriber can only be installed once per
        // process; whichever test runs first wins. Either outcome here is a
        // successful directory/file setup, and the second `init` call in the
        // same process must not panic.
        let first = init(&paths);
        let second = init(&paths);
        assert!(second.is_none() || first.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn prune_old_logs_keeps_only_the_most_recent_files() {
        let (root, paths) = temp_paths("prune-old-logs");
        let log_dir = paths.client_log_dir();
        std::fs::create_dir_all(&log_dir).unwrap();
        for day in 0..(MAX_RETAINED_LOGS + 3) {
            std::fs::write(
                log_dir.join(format!("{LOG_FILE_PREFIX}.2026-01-{day:02}")),
                b"log line\n",
            )
            .unwrap();
        }

        prune_old_logs(&log_dir);

        let remaining = std::fs::read_dir(&log_dir).unwrap().count();
        assert_eq!(remaining, MAX_RETAINED_LOGS);

        let _ = std::fs::remove_dir_all(&root);
    }
}
