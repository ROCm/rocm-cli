// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! rocm-core/rocm-dash-daemon-free live managed-service-registry seam.
//!
//! `rocm-dash-tui` has no dependency on `rocm-core` or `rocm-dash-daemon` (see
//! AGENTS.md §6's crate-layering invariants), so the actual on-disk registry
//! read — `AppPaths::services_dir()` + `discover_managed_services` — lives in
//! the bin (`apps/rocm::dash`) and is handed in across this seam, mirroring
//! `tool_exec::RocmToolExecutor`.
//!
//! The probe is read live, at the moment the quit-confirm gate actually needs
//! an answer, rather than once at launch and cached. A one-shot, launch-time
//! read would freeze out two real cases: a model started mid-session via the
//! Serve wizard (its id would never be in a launch-time snapshot, so it would
//! never get a stop command or the `s`-to-open-Services hint), and a host
//! whose dashboard-embedded daemon client never connects at all (Windows —
//! `has_received_snapshot` never flips, so `AppState::instances` stays empty
//! for the life of the session). A live probe has neither problem — there is
//! no window for it to go stale in.

use std::collections::HashSet;
use std::sync::Arc;

/// Live classification of the managed-service registry, as of the moment
/// [`ManagedRegistryProbe::snapshot`] was called.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManagedRegistrySnapshot {
    /// Whether any managed-service record is currently a live scrape target —
    /// the same statuses `Instance::status.is_serving()` treats as "actively
    /// serving" (`ready`/`running`, excluding a `running` record still in a
    /// recognized `startup_phase` — see `instance_status_for_record`), and
    /// excluding one with no bound port (`port == 0`). An implementation
    /// deriving both fields of this snapshot from one discovery pass (as
    /// `LiveManagedRegistryProbe::snapshot` does) can never report this `true`
    /// with `managed_ids` empty: the same pass that counts a record here is
    /// the one that names its id there.
    pub has_live_service: bool,
    /// Ids of every managed-service record this host's own registry
    /// currently treats as a live scrape target — i.e. the ids
    /// `rocm services stop <id> --yes` can actually reach, as opposed to a
    /// Docker- or Lemonade-discovered instance from an external daemon, which
    /// that command cannot stop.
    pub managed_ids: HashSet<String>,
}

/// rocm-core/rocm-dash-daemon-free live registry-read boundary.
///
/// The bin implements this; the dash holds it as
/// `Option<Arc<dyn ManagedRegistryProbe>>`. The bin injects `Some` on every
/// path, including `--demo`/`--replay`/`--chat-mock` — unlike
/// `tool_exec::RocmToolExecutor`, which those paths null out. Demo/replay
/// safety instead comes from `AppState::simulated` short-circuiting
/// `has_live_instance` before the probe is ever consulted (reading a real
/// on-disk registry under `--demo`/`--replay` is harmless; it's just never
/// acted on). The `Debug` supertrait keeps `ResolvedArgs`/`AppState` deriving
/// `Debug`.
///
/// Implementations MUST stay cheap: a local JSON-file read only, never a
/// daemon round-trip or network I/O. In the dashboard, [`snapshot`](Self::snapshot)
/// is called every time something needs `live_registry_snapshot`'s answer:
/// `request_quit` on the quit keystroke, `quit_confirm_request` on every
/// render while the quit-confirm prompt is open, and
/// `quit_confirm_open_services_key` on the `s` keystroke while it's open.
/// The launcher's own `has_live_instance` check calls it on every render
/// too, but through `StaticManagedRegistrySnapshot` — a read cached once per
/// menu pass — so that per-frame call never reaches an implementation's
/// actual registry I/O; only the dashboard's calls above do.
pub trait ManagedRegistryProbe: std::fmt::Debug + Send + Sync {
    /// Read the registry now and classify it from the record's own `status`
    /// field, the same statuses `Instance::status.is_serving()` treats as
    /// live.
    ///
    /// This is **not** the same verdict `rocm services list` can give: that
    /// command also probes process liveness and the endpoint itself
    /// (`refresh_managed_service_runtime_liveness`), so it can demote a
    /// crashed service's stale on-disk `ready`/`running` status to `stopped`
    /// before this trait ever sees it. A local-only PID check was
    /// deliberately not added here, for the same reason
    /// `apps/rocm::dash::services_past_attempts` does not add one: it would
    /// duplicate the CLI's liveness classification in a second place that
    /// could drift from it, trading a crashed-but-not-yet-reaped service for
    /// whatever the duplicated check gets wrong instead. A caller using this
    /// for a blocking claim (e.g. the quit-confirm prompt) should treat it as
    /// "the registry has not been told this stopped" rather than "this is
    /// confirmed still running".
    fn snapshot(&self) -> ManagedRegistrySnapshot;
}

/// Arc-wrapped probe as stored in `ResolvedArgs`/`AppState`.
pub type SharedManagedRegistryProbe = Arc<dyn ManagedRegistryProbe>;

/// A [`ManagedRegistryProbe`] that always returns the same, already-computed
/// snapshot.
///
/// For a caller that already has a freshly-read classification in hand rather
/// than a live seam to call through — the pre-dashboard launcher
/// (`ui::launcher::launcher_state`) re-reads the registry once per
/// menu-loop pass (`apps/rocm::dash::run_launcher`'s own
/// `launcher_serving_instances` call) rather than on every render the way the
/// live dashboard's bin-injected probe does, so wrapping that one read is
/// exactly as live as the launcher itself ever is between keypresses.
#[derive(Debug, Clone, Default)]
pub struct StaticManagedRegistrySnapshot(pub ManagedRegistrySnapshot);

impl ManagedRegistryProbe for StaticManagedRegistrySnapshot {
    fn snapshot(&self) -> ManagedRegistrySnapshot {
        self.0.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_snapshot_returns_what_it_was_built_with() {
        let ids: HashSet<String> = std::iter::once("svc-a".to_string()).collect();
        let probe = StaticManagedRegistrySnapshot(ManagedRegistrySnapshot {
            has_live_service: true,
            managed_ids: ids.clone(),
        });
        let snap = probe.snapshot();
        assert!(snap.has_live_service);
        assert_eq!(snap.managed_ids, ids);
    }
}
