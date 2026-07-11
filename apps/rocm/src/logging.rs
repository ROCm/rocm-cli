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

use std::path::{Component, Path, PathBuf};

use rocm_core::AppPaths;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

/// File name prefix for the daily-rotated client log
/// (`~/.rocm/logs/rocm-cli.log.<date>`), sibling to the daemon's
/// `rocmdashd.log` under the same canonical `AppPaths` root.
const LOG_FILE_PREFIX: &str = "rocm-cli.log";

/// Total number of daily log files retained on disk, INCLUDING today's
/// actively-written file; older ones are pruned on startup so a long-lived
/// install never grows the logs directory unbounded. Pruning runs *after* the
/// appender opens today's file (see [`init`]), so this is the true on-disk
/// bound — a week of history.
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
    // Validate the resolved directory before any filesystem write: the base
    // path is derived from `$HOME` / `$ROCM_CLI_DATA_DIR` / config, so it must
    // be confirmed to live inside the `AppPaths` root and be free of `..`
    // traversal before it is handed to the appender or the pruner.
    let log_dir = validated_log_dir(paths)?;

    // Construct the appender first: `rolling::daily` opens today's file eagerly
    // at construction time. Prune AFTER that so today's active file is counted
    // in the retention window — otherwise the effective on-disk bound would be
    // MAX_RETAINED_LOGS + 1 (the pruned set plus the freshly-opened file).
    let file_appender = tracing_appender::rolling::daily(&log_dir, LOG_FILE_PREFIX);
    prune_old_logs(&log_dir);
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

/// Resolve, create, and validate the client log directory, breaking the
/// data-flow taint from `$HOME` / `$ROCM_CLI_DATA_DIR` / config into every
/// filesystem operation below.
///
/// The directory is always `<data_dir>/logs`. We create it, then canonicalize
/// both it and the `AppPaths` data-dir root and require the canonical log dir
/// to stay nested under the canonical root — so a symlink or `..` in the
/// resolved environment cannot redirect log writes outside `~/.rocm`. Returns
/// the validated, canonicalized directory (all subsequent fs ops run on this
/// value, not the raw env-derived path), or `None` — in which case [`init`]
/// degrades to no logging rather than writing to an unexpected location.
fn validated_log_dir(paths: &AppPaths) -> Option<PathBuf> {
    let root = paths.data_dir.as_path();
    let log_dir = paths.client_log_dir();
    std::fs::create_dir_all(&log_dir).ok()?;

    let canonical_root = std::fs::canonicalize(root).ok()?;
    let canonical_log_dir = std::fs::canonicalize(&log_dir).ok()?;

    // Reject any parent-dir traversal and require containment under the root.
    // Both conditions are belt-and-suspenders after canonicalization (which
    // already resolves `..` and symlinks) and act as the sanitizing barrier
    // for the tainted base path.
    let has_traversal = canonical_log_dir
        .components()
        .any(|c| matches!(c, Component::ParentDir));
    if has_traversal || !canonical_log_dir.starts_with(&canonical_root) {
        return None;
    }

    Some(canonical_log_dir)
}

/// Remove rotated log files beyond [`MAX_RETAINED_LOGS`], keeping the logs
/// directory bounded for long-lived installs. Best-effort: any I/O error
/// just leaves the file in place for a future run to retry.
///
/// `log_dir` must already have passed [`validated_log_dir`]; the caller only
/// ever passes that validated, canonicalized path.
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

    fn temp_paths(name: &str) -> (PathBuf, AppPaths) {
        // Use the OS temp dir (not the repo tree) so tests never litter the
        // working copy. Canonicalize the base first so the fixture paths are
        // derived from a resolved, symlink-free root (matches production's
        // `validated_log_dir` barrier and keeps assertions stable where
        // `$TMPDIR` is itself a symlink). Name-scoped with pid + millis to stay
        // unique across parallel test threads and repeated runs.
        let base = std::env::temp_dir();
        let base = std::fs::canonicalize(&base).unwrap_or(base);
        let root = base.join(format!(
            "rocm-cli-logging-test-{name}-{}-{}",
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
        // A global `tracing` subscriber can only be installed once per test
        // binary process, so whether THIS call wins the race is
        // nondeterministic under `cargo test`'s default parallel threads.
        // The rolling file appender opens its current-period file eagerly at
        // construction time, before the (possibly-losing) subscriber install
        // is attempted, so the directory/file assertions below hold either
        // way — only the `Some`/`None` guard outcome depends on the race.
        let guard = init(&paths);
        if guard.is_some() {
            tracing::info!("hello from the test subscriber");
        }
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

    #[test]
    fn prune_ordering_bounds_total_including_todays_active_file() {
        // Regression for the off-by-one: `init` opens today's file BEFORE
        // pruning, so pruning must count that active file within the retention
        // window. Pre-seed exactly MAX_RETAINED_LOGS dated files, then mirror
        // `init`'s ordering (open the daily appender, then prune) without
        // installing a global subscriber. The total on disk must stay at
        // MAX_RETAINED_LOGS — not MAX_RETAINED_LOGS + 1.
        let (root, paths) = temp_paths("prune-ordering");
        let log_dir = paths.client_log_dir();
        std::fs::create_dir_all(&log_dir).unwrap();
        for day in 1..=MAX_RETAINED_LOGS {
            std::fs::write(
                log_dir.join(format!("{LOG_FILE_PREFIX}.2020-01-{day:02}")),
                b"old\n",
            )
            .unwrap();
        }

        // Opens today's file eagerly (dated far after the 2020 seeds, so it
        // sorts newest and is never the one pruned).
        let appender = tracing_appender::rolling::daily(&log_dir, LOG_FILE_PREFIX);
        prune_old_logs(&log_dir);
        drop(appender);

        let remaining = std::fs::read_dir(&log_dir).unwrap().count();
        assert_eq!(
            remaining, MAX_RETAINED_LOGS,
            "today's active file must count toward the retention bound"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
