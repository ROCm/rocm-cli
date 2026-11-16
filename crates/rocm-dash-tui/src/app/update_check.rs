// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The background Updates-tile check: the tick-driven `rocm update --json` job
//! that feeds the Home tab's Updates tile, and the reducer that turns its output
//! into an `UpdateStatus`.
//!
//! A leaf module, split out of `app/event_loop.rs`: the loop only calls
//! `refresh_update_status` once per tick, and nothing here calls back into it.

use std::time::Duration;

use super::{AppState, UpdateStatus};

/// How often the background Updates-tile check re-runs.
const UPDATE_CHECK_INTERVAL: Duration = Duration::from_hours(6);

/// Job id for the periodic background update check driven off the tick loop.
/// Deliberately distinct from `update_manager`'s interactive `"update-check"`
/// so the two never clobber each other's job slot / console output.
/// `pub(crate)` so the Home tab's activity feed (`ui::tabs::home`) can filter
/// this job out — it is the tile's own plumbing, not user activity.
pub(crate) const HOME_UPDATE_CHECK_JOB_ID: &str = "home-update-check";

/// Bound on the background update check's own per-runtime index lookups, so a
/// slow/unreachable index can't leave the job running indefinitely — the same
/// principle as the CLI's own `STARTUP_UPDATE_CHECK_TIMEOUT_SECS`. The two
/// crates can't share the constant (`apps/rocm` depends on `rocm-dash-tui`,
/// not the reverse), so this value isn't required to match it.
const HOME_UPDATE_CHECK_TIMEOUT_SECS: u64 = 5;

/// Reduce a parsed `rocm update --json` document's `runtimes` array into an
/// [`UpdateStatus`]. Empty ⇒ nothing managed to check; any update-available or
/// repair-available row wins over up-to-date/error rows (the tile surfaces the
/// most actionable state — a repair is as actionable as an update); otherwise
/// `UpToDate` only if every row resolved cleanly — a mixed result (some rows
/// errored, some unrecognized) can't honestly assert freshness for the
/// runtimes that didn't resolve, so it's `Error` too.
pub(super) fn reduce_update_json(document: &serde_json::Value) -> UpdateStatus {
    fn status_of(row: &serde_json::Value) -> Option<&str> {
        row.get("status").and_then(serde_json::Value::as_str)
    }
    let Some(runtimes) = document
        .get("runtimes")
        .and_then(serde_json::Value::as_array)
    else {
        return UpdateStatus::Error;
    };
    if runtimes.is_empty() {
        return UpdateStatus::NoManagedRuntimes;
    }
    if let Some(row) = runtimes.iter().find(|row| {
        matches!(
            status_of(row),
            Some("update_available" | "repair_available")
        )
    }) {
        // A missing/null `latest_version` must not silently fall through to
        // the up-to-date/error checks below — that would misreport a real,
        // actionable update as "check failed". Fall back to a placeholder
        // instead of losing the actionable status.
        let latest_version = row
            .get("latest_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(version unknown)")
            .to_owned();
        return UpdateStatus::UpdateAvailable { latest_version };
    }
    if runtimes
        .iter()
        .all(|row| matches!(status_of(row), Some("up_to_date" | "ahead_of_index")))
    {
        return UpdateStatus::UpToDate;
    }
    UpdateStatus::Error
}

/// Spawn/consume the periodic `home-update-check` job that backs the Home
/// tab's Updates tile, and return any job-bridge side effects to pump.
///
/// Pure w.r.t. process I/O — like [`open_overlay_for_focus`], the caller runs
/// the returned effects through [`crate::jobs::run_effects`]. Called once per
/// tick from `event_loop`, skipped entirely under `state.simulated`.
pub(super) fn refresh_update_status(
    state: &mut AppState,
) -> Vec<rocm_dash_core::state::SideEffect> {
    if state.update_status_pending {
        let Some(job) = state.jobs.job(HOME_UPDATE_CHECK_JOB_ID) else {
            // Re-arm the same as the terminal-job path below: without this,
            // a vanished job would leave `update_check_due_at` in the past,
            // so every subsequent tick would spawn a new check immediately.
            state.update_status_pending = false;
            state.update_check_due_at = std::time::Instant::now() + UPDATE_CHECK_INTERVAL;
            return Vec::new();
        };
        if !job.is_terminal() {
            return Vec::new();
        }
        state.update_status = match &job.status {
            rocm_dash_core::state::JobStatus::Done { code: 0 } => job
                .output
                .iter()
                .rev()
                .find_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .map_or(UpdateStatus::Error, |doc| reduce_update_json(&doc)),
            _ => UpdateStatus::Error,
        };
        state.update_status_pending = false;
        state.update_check_due_at = std::time::Instant::now() + UPDATE_CHECK_INTERVAL;
        return Vec::new();
    }

    if std::time::Instant::now() < state.update_check_due_at {
        return Vec::new();
    }

    if std::env::var_os("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK").is_some() {
        return Vec::new();
    }

    let fx = state
        .jobs
        .apply(rocm_dash_core::state::StateEvent::StartJob {
            id: HOME_UPDATE_CHECK_JOB_ID.to_owned(),
            cmd: crate::ui::exec::resolve_exe(),
            args: vec![
                "update".to_owned(),
                "--json".to_owned(),
                "--timeout-secs".to_owned(),
                HOME_UPDATE_CHECK_TIMEOUT_SECS.to_string(),
            ],
        });
    if !fx.is_empty() {
        state.update_status_pending = true;
    }
    fx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::st;

    // --- Home tab update check (background job-bridge trigger) ---

    // Serializes every test in this group against
    // `refresh_update_status_skips_spawn_when_disabled_via_env`, which toggles
    // `ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK` — process env is shared across
    // test threads, so an unguarded test can observe the var mid-toggle and
    // spuriously see `refresh_update_status` skip the spawn it expects.
    static UPDATE_CHECK_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn refresh_update_status_spawns_on_first_due_tick() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        assert!(!s.update_status_pending);
        let fx = refresh_update_status(&mut s);
        assert!(!fx.is_empty(), "a due check spawns a job");
        assert!(s.update_status_pending);
        assert!(s.jobs.job(HOME_UPDATE_CHECK_JOB_ID).is_some());
    }

    #[test]
    fn refresh_update_status_does_not_duplicate_spawn_while_running() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let fx = refresh_update_status(&mut s);
        assert!(!fx.is_empty());
        assert!(s.update_status_pending);

        // Still pending, job still running (non-terminal) → no-op, no second spawn.
        let fx2 = refresh_update_status(&mut s);
        assert!(
            fx2.is_empty(),
            "no duplicate spawn while pending and running"
        );
        assert!(s.update_status_pending);
        assert_eq!(s.update_status, UpdateStatus::Unknown);
    }

    #[test]
    fn refresh_update_status_rearms_due_at_when_pending_job_vanishes() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        assert!(s.update_status_pending);

        // Not reachable today (jobs are never removed), but if it ever is,
        // `update_check_due_at` must still be pushed out — otherwise every
        // subsequent tick would spawn a new check immediately.
        s.jobs.jobs.remove(HOME_UPDATE_CHECK_JOB_ID);
        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty());
        assert!(!s.update_status_pending);
        assert!(
            s.update_check_due_at > std::time::Instant::now(),
            "due_at must be re-armed, not left in the past"
        );
    }

    #[test]
    fn refresh_update_status_resolves_from_terminal_success_json() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        assert!(s.update_status_pending);

        s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            line: serde_json::json!({
                "runtimes": [{
                    "runtime_key": "rocm",
                    "channel": "stable",
                    "family": "rocm",
                    "installed_version": "7.0.0",
                    "latest_version": "7.1.0",
                    "status": "update_available",
                    "message": null,
                }]
            })
            .to_string(),
        });
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            code: 0,
        });

        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty(), "resolving a terminal job spawns nothing");
        assert!(!s.update_status_pending);
        assert_eq!(
            s.update_status,
            UpdateStatus::UpdateAvailable {
                latest_version: "7.1.0".into()
            }
        );
    }

    #[test]
    fn refresh_update_status_resolves_to_error_on_terminal_failure() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        assert!(s.update_status_pending);

        // Nonzero exit → Error, regardless of any output on the ring.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            code: 1,
        });
        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty());
        assert!(!s.update_status_pending);
        assert_eq!(s.update_status, UpdateStatus::Error);
    }

    #[test]
    fn refresh_update_status_resolves_to_error_on_unparsable_success_output() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);

        // Exit 0 but no valid JSON line on the ring → Error, not a silent hang.
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobLine {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            line: "not json".into(),
        });
        s.jobs.apply(rocm_dash_core::state::StateEvent::JobDone {
            id: HOME_UPDATE_CHECK_JOB_ID.into(),
            code: 0,
        });
        let fx = refresh_update_status(&mut s);
        assert!(fx.is_empty());
        assert!(!s.update_status_pending);
        assert_eq!(s.update_status, UpdateStatus::Error);
    }

    #[test]
    fn refresh_update_status_spawn_args_include_bounded_timeout() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut s = st();
        let _ = refresh_update_status(&mut s);
        let job = s
            .jobs
            .job(HOME_UPDATE_CHECK_JOB_ID)
            .expect("job spawned on first due tick");
        // Pinned to a literal, not `HOME_UPDATE_CHECK_TIMEOUT_SECS`: comparing
        // the constant to itself can never catch an unintentional change to
        // its value. A literal forces a deliberate test update (and a second
        // thought) whenever the bound changes.
        assert_eq!(
            job.args.last().map(String::as_str),
            Some("5"),
            "the background check's timeout bound must stay a deliberate choice: {:?}",
            job.args
        );
        assert!(job.args.iter().any(|a| a == "--timeout-secs"));
    }

    #[test]
    fn refresh_update_status_skips_spawn_when_disabled_via_env() {
        let _guard = UPDATE_CHECK_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: serialized by `UPDATE_CHECK_ENV_TEST_LOCK`; no other thread
        // reads/writes this var concurrently.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut s = st();
            let fx = refresh_update_status(&mut s);
            assert!(fx.is_empty(), "a disabled check must not spawn a job");
            assert!(!s.update_status_pending);
            assert!(s.jobs.job(HOME_UPDATE_CHECK_JOB_ID).is_none());
        });
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK");
        }
        result.unwrap();
    }

    #[test]
    fn reduce_update_json_all_up_to_date_or_ahead_is_up_to_date() {
        let doc = serde_json::json!({
            "runtimes": [
                {"status": "up_to_date"},
                {"status": "ahead_of_index"},
            ]
        });
        assert_eq!(reduce_update_json(&doc), UpdateStatus::UpToDate);
    }

    #[test]
    fn reduce_update_json_mixed_up_to_date_and_error_is_error_not_up_to_date() {
        // One runtime resolved cleanly, one didn't — asserting "Up to date"
        // here would be a false claim about the runtime that errored.
        let doc = serde_json::json!({
            "runtimes": [
                {"status": "up_to_date"},
                {"status": "error", "message": "boom"},
            ]
        });
        assert_eq!(reduce_update_json(&doc), UpdateStatus::Error);
    }

    #[test]
    fn reduce_update_json_unrecognized_status_is_error() {
        let doc = serde_json::json!({
            "runtimes": [{"status": "something_new"}]
        });
        assert_eq!(reduce_update_json(&doc), UpdateStatus::Error);
    }

    #[test]
    fn reduce_update_json_repair_available_is_update_available_not_error() {
        // A same-version composition repair is as actionable as a version
        // bump — the tile must not report "check failed" for it.
        let doc = serde_json::json!({
            "runtimes": [{"status": "repair_available", "latest_version": "6.4.0"}]
        });
        assert_eq!(
            reduce_update_json(&doc),
            UpdateStatus::UpdateAvailable {
                latest_version: "6.4.0".to_owned()
            }
        );
    }

    #[test]
    fn reduce_update_json_missing_latest_version_is_still_update_available() {
        // A row with an actionable status but no `latest_version` must not be
        // silently skipped in favor of the up-to-date/error checks below it —
        // that would misreport a real update as "check failed".
        let doc = serde_json::json!({
            "runtimes": [{"status": "update_available"}]
        });
        assert_eq!(
            reduce_update_json(&doc),
            UpdateStatus::UpdateAvailable {
                latest_version: "(version unknown)".to_owned()
            }
        );
    }
}
