// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What the runner's discovery passes remember between ticks, and how a pass
//! reconciles what a source reports against it: the per-instance telemetry the
//! loop carries ([`TelemetryState`]), what each source has live
//! ([`DiscoveryState`]), how the managed-service registry's reads have gone
//! ([`RegistryHealth`]), and the upsert/announce/retire steps every source shares
//! ([`Reconcile`]). Split out of `runner.rs` to keep it within the module-size
//! limit; `run_loop` drives all of it.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rocm_dash_core::metrics::Instance;
use rocm_dash_core::observation::GenerationObservationTracker;
use rocm_dash_core::protocol::Event;
use rocm_dash_core::state::{State, StateEvent};
use rocm_dash_core::traits::DiscoveredService;
use tokio::sync::broadcast;
use tracing::{info, warn};

use super::{
    LatencyWindows, broadcast_and_persist, instance_from_discovered, merge_discovery_refresh,
};
use crate::persist::SessionWriter;

/// Per-instance telemetry the loop carries between scrapes, keyed by container id.
#[derive(Default)]
pub(super) struct TelemetryState {
    /// Generation-throughput trackers. Constructed lazily from
    /// `opts.instance_tick`; they survive scrape failures so the last-observed
    /// rate is held for the validity window before clearing.
    pub(super) gen_trackers: HashMap<String, GenerationObservationTracker>,
    /// Previous TTFT / TPOT histogram readings plus a freshness tracker per
    /// instance, for the windowed avg-ms derivation (mirrors `gen_trackers`).
    pub(super) prev_ttft: LatencyWindows,
    pub(super) prev_tpot: LatencyWindows,
}

impl TelemetryState {
    /// Remove all per-service telemetry state for `id`: tracker, TTFT, and
    /// TPOT baselines.  Call on explicit removal (Docker Gone, Lemonade Gone,
    /// managed Gone) and whenever an identity change requires a clean slate.
    pub(super) fn purge(&mut self, id: &str) {
        self.gen_trackers.remove(id);
        self.prev_ttft.remove(id);
        self.prev_tpot.remove(id);
    }

    /// Invalidate the generation tracker for `id` (reset baseline without
    /// removing the map entry) and clear the TTFT/TPOT latency windows.
    /// Called when an endpoint changes within the same container/service id so
    /// the new scrape address starts with a fresh rate baseline.
    pub(super) fn invalidate_on_endpoint_change(&mut self, id: &str) {
        if let Some(t) = self.gen_trackers.get_mut(id) {
            t.invalidate();
        }
        self.prev_ttft.remove(id);
        self.prev_tpot.remove(id);
    }
}

/// What each discovery source currently has live, so a pass can tell a new
/// service from a known one and a known one from one that has gone.
#[derive(Default)]
pub(super) struct DiscoveryState {
    /// Container ids the last Docker pass saw.
    pub(super) known_instances: HashSet<String>,
    /// Ids the last managed-service registry pass surfaced.
    pub(super) known_services: HashSet<String>,
    /// The subset of the registry ids whose engine is NOT vLLM, excluded from
    /// the vLLM Prometheus scrape so they aren't mis-parsed.
    pub(super) managed_non_vllm: HashSet<String>,
    /// Model identities of the externally discovered Lemonade rows.
    pub(super) lemonade_ids: HashSet<String>,
    /// Registry-managed Lemonade endpoints, scraped read-only for telemetry.
    pub(super) managed_lemonade: Vec<crate::registry::ManagedLemonadeTarget>,
    /// Ancestry roots of the managed services the registry surfaced, used to
    /// attribute GPU processes to a service when the cgroup join cannot (a
    /// managed service is not a container).
    pub(super) managed_roots: Vec<rocm_dash_collectors::ancestry::Root>,
    /// How the registry reads have been going, so a condition that lasts many
    /// passes is logged once rather than every discovery tick.
    pub(super) registry_health: RegistryHealth,
}

impl DiscoveryState {
    /// Record what a pass that read only part of the registry found. The services
    /// it saw join the known set, so a later complete read can retire them, and
    /// the known ones it did not see stay; the non-vLLM marks of the services it
    /// saw are refreshed and the others kept, so a kept non-vLLM service is still
    /// left out of the vLLM scrape.
    pub(super) fn keep_through_partial_registry(
        &mut self,
        disc: crate::registry::ManagedDiscovery,
    ) {
        self.managed_non_vllm.retain(|id| !disc.seen.contains(id));
        self.managed_non_vllm.extend(disc.non_vllm);
        // Roots of the services this pass saw are replaced; the unseen ones
        // keep theirs, as their services stay known.
        // A service this pass read and found stopped loses its roots too, or its
        // PID would stay matchable for as long as an unrelated record is broken.
        self.managed_roots
            .retain(|r| !disc.seen.contains(&r.owner) && !disc.retired.contains(&r.owner));
        self.managed_roots.extend(disc.roots);
        self.known_services.extend(disc.seen);
    }
}

/// What the managed-service registry's reads have reported so far. A failure
/// that lasts is logged on its first pass only, the recovery once, and a set of
/// malformed records each time it changes; the dashboard's warning badge, which
/// is cheap and does not scroll, carries the failure for as long as it lasts.
#[derive(Default)]
pub(super) struct RegistryHealth {
    /// The previous pass could not read the registry.
    pub(super) failing: bool,
    /// The malformed records the last successful pass reported, so an unchanged
    /// set is not reported again.
    pub(super) reported_malformed: Vec<PathBuf>,
}

/// What a discovery pass should say about the registry, apart from the services
/// it found. Kept as data so [`RegistryHealth::observe`] can be tested without a
/// log subscriber.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct RegistryChange {
    /// The first failed pass of a streak, with the error.
    pub(super) failed: Option<String>,
    /// The first successful pass after a streak of failures.
    pub(super) recovered: bool,
    /// The malformed records, when the set differs from the one last reported
    /// and is not empty.
    pub(super) malformed: Vec<PathBuf>,
}

impl RegistryHealth {
    pub(super) fn observe(
        &mut self,
        read: &Result<crate::registry::RegistryRead, crate::registry::RegistryReadError>,
    ) -> RegistryChange {
        let mut change = RegistryChange::default();
        match read {
            Err(e) => {
                if !self.failing {
                    self.failing = true;
                    change.failed = Some(e.to_string());
                }
            }
            Ok(read) => {
                if self.failing {
                    self.failing = false;
                    change.recovered = true;
                }
                if read.malformed != self.reported_malformed {
                    self.reported_malformed.clone_from(&read.malformed);
                    change.malformed.clone_from(&read.malformed);
                }
            }
        }
        change
    }
}

impl RegistryChange {
    pub(super) fn log(&self) {
        if let Some(error) = &self.failed {
            warn!(
                error = %error,
                "managed-service registry could not be read; keeping the known services until it can be"
            );
        }
        if self.recovered {
            info!("managed-service registry is readable again");
        }
        if !self.malformed.is_empty() {
            warn!(
                files = ?self.malformed,
                "skipping managed-service records that are not valid JSON"
            );
        }
    }
}

/// The header warning for a pass that could not read the registry.
pub(super) fn registry_read_warning(error: &crate::registry::RegistryReadError) -> String {
    format!("managed registry: {error}; keeping the last known services")
}

/// Which discovery source a service came from. It only selects the log wording
/// in [`Reconcile::announce`] and [`Reconcile::retire`]; those two steps behave
/// the same for every source. How a pass decides what is new, known or gone
/// differs: Docker and the registry go through `reconcile_set`, Lemonade through
/// `reconcile_lemonade`.
#[derive(Clone, Copy)]
pub(super) enum DiscoverySource {
    Docker,
    Lemonade,
    Managed,
}

impl DiscoverySource {
    pub(super) fn log_discovered(self, svc: &DiscoveredService) {
        match self {
            Self::Docker => info!(
                id = %svc.container_id,
                name = %svc.container_name,
                model = %svc.model_name,
                "instance discovered"
            ),
            Self::Lemonade => info!(
                id = %svc.container_id,
                model = %svc.model_name,
                "lemonade instance discovered"
            ),
            Self::Managed => info!(id = %svc.container_id, "managed service discovered"),
        }
    }

    pub(super) fn log_gone(self, id: &str) {
        match self {
            Self::Docker => info!(id = %id, "instance gone"),
            Self::Lemonade => info!(id = %id, "lemonade instance gone"),
            Self::Managed => info!(id = %id, "managed service gone"),
        }
    }
}

/// The loop state a discovery pass changes, and the stream it announces on.
pub(super) struct Reconcile<'a> {
    pub(super) state: &'a mut State,
    pub(super) telemetry: &'a mut TelemetryState,
    pub(super) tx: &'a broadcast::Sender<Event>,
    pub(super) persist: Option<&'a Arc<Mutex<SessionWriter>>>,
}

impl Reconcile<'_> {
    /// Upsert a (re)discovered service. Collector-owned fields carry over from
    /// the instance already in state unless `carry_live` is false, which is how
    /// a changed identity starts from a clean baseline.
    pub(super) fn upsert(&mut self, svc: &DiscoveredService, carry_live: bool) {
        let existing = self.state.instances.get(&svc.container_id).cloned();
        let inst = merge_discovery_refresh(svc, if carry_live { existing.as_ref() } else { None });
        self.state.apply(StateEvent::InstanceUpserted(inst));
    }

    /// Announce a newly discovered service. InstanceDiscovered is a
    /// discovery-metadata event: it carries a clean instance, with no prior
    /// collector-owned telemetry.
    pub(super) fn announce(&self, source: DiscoverySource, svc: &DiscoveredService) {
        source.log_discovered(svc);
        broadcast_and_persist(
            self.tx,
            self.persist,
            Event::InstanceDiscovered(instance_from_discovered(svc)),
        );
    }

    /// Drop a service that has gone: purge its telemetry, remove it from state
    /// and announce it.
    pub(super) fn retire(&mut self, source: DiscoverySource, id: &str) {
        source.log_gone(id);
        self.telemetry.purge(id);
        self.state
            .apply(StateEvent::InstanceRemoved(id.to_string()));
        broadcast_and_persist(
            self.tx,
            self.persist,
            Event::InstanceGone {
                container_id: id.to_string(),
            },
        );
    }

    /// Reconcile a source that reports its whole set each pass (Docker, the
    /// managed-service registry). `known` is what the previous pass saw and
    /// `seen` what this one saw; the caller stores `seen` as the next `known`.
    ///
    /// A service whose port changed under the same id has a new scrape address:
    /// its rate baseline and latency windows are reset and no live fields carry
    /// over. It is still the same service, so it is not announced again.
    pub(super) fn reconcile_set(
        &mut self,
        source: DiscoverySource,
        svcs: &[DiscoveredService],
        known: &HashSet<String>,
        seen: &HashSet<String>,
    ) {
        self.upsert_set(source, svcs, known);
        for gone in known.difference(seen) {
            self.retire(source, gone);
        }
    }

    /// The half of [`Self::reconcile_set`] that adds and refreshes: upsert every
    /// service in `svcs` and announce the ones `known` did not have. Retires
    /// nothing, which is what a pass that could only see part of its source needs
    /// (see [`DiscoveryState::keep_through_partial_registry`]).
    pub(super) fn upsert_set(
        &mut self,
        source: DiscoverySource,
        svcs: &[DiscoveredService],
        known: &HashSet<String>,
    ) {
        for svc in svcs {
            let same_endpoint = self
                .state
                .instances
                .get(&svc.container_id)
                .is_none_or(|ex| ex.port == svc.port);
            if !same_endpoint {
                self.telemetry
                    .invalidate_on_endpoint_change(&svc.container_id);
            }
            self.upsert(svc, same_endpoint);
            if !known.contains(&svc.container_id) {
                self.announce(source, svc);
            }
        }
    }

    /// Reconcile the probe-based Lemonade source: the models one endpoint
    /// reports loaded. An empty set means the endpoint did not answer (or is
    /// now registry-managed), which retires every current row.
    ///
    /// Each row's `container_id` encodes the host, the port and the model, so a
    /// host or port change yields new ids. The ids that went are retired first
    /// (their telemetry is purged, so a stale rate baseline, TTFT and TPOT do
    /// not bleed across; they leave the state and are announced gone), then the
    /// new ones are announced, so the state never carries a row for an endpoint
    /// that moved and the vLLM scrape never polls the old one. `current` is
    /// updated to the ids this pass saw.
    pub(super) fn reconcile_lemonade(
        &mut self,
        services: &[DiscoveredService],
        current: &mut HashSet<String>,
    ) {
        let seen: HashSet<String> = services
            .iter()
            .map(|svc| svc.container_id.clone())
            .collect();
        for gone in current.difference(&seen) {
            self.retire(DiscoverySource::Lemonade, gone);
        }
        self.upsert_set(DiscoverySource::Lemonade, services, current);
        *current = seen;
    }

    /// Start a managed Lemonade service clean when its registry target moved to
    /// another endpoint or model under the same service id: the rate baseline
    /// and latency windows are reset and no live fields carry over, as for a
    /// port change in [`Self::upsert_set`]. A target with no previous entry is
    /// new and needs nothing reset.
    pub(super) fn reset_moved_lemonade_targets(
        &mut self,
        svcs: &[DiscoveredService],
        previous: &[crate::registry::ManagedLemonadeTarget],
        current: &[crate::registry::ManagedLemonadeTarget],
    ) {
        for svc in svcs {
            let id = &svc.container_id;
            let Some(old) = previous.iter().find(|target| &target.service_id == id) else {
                continue;
            };
            let unchanged = current
                .iter()
                .find(|target| &target.service_id == id)
                .is_some_and(|new| {
                    crate::registry::endpoints_match(&old.host, old.port, &new.host, new.port)
                        && old.model_ref == new.model_ref
                        && old.canonical_model_id == new.canonical_model_id
                });
            if !unchanged && self.state.instances.contains_key(id) {
                self.telemetry.invalidate_on_endpoint_change(id);
                self.upsert(svc, false);
            }
        }
    }
}

/// The `(id, port)` pairs the per-instance vLLM Prometheus scrape polls: every
/// instance with a port except the discovered Lemonade rows (their endpoint
/// collector scrapes them) and the managed services whose engine is not vLLM.
pub(super) fn vllm_scrape_targets(
    instances: &HashMap<String, Instance>,
    lemonade_ids: &HashSet<String>,
    managed_non_vllm: &HashSet<String>,
) -> Vec<(String, u16)> {
    instances
        .values()
        .filter(|i| !lemonade_ids.contains(&i.container_id))
        .filter(|i| !managed_non_vllm.contains(&i.container_id))
        .filter_map(|i| i.port.map(|p| (i.container_id.clone(), p)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::tests::{at, baseline};
    use std::time::Duration;

    // -- Identity/endpoint cleanup helper tests -----------------------------------

    fn make_tracker(tick_secs: u64) -> GenerationObservationTracker {
        GenerationObservationTracker::new(Duration::from_secs(tick_secs))
    }

    /// `TelemetryState::purge` removes the tracker entry, TTFT baseline, and
    /// TPOT baseline for the target id and leaves other ids untouched.
    #[test]
    fn telemetry_state_purge_removes_all_three_maps_for_id() {
        let mut state = TelemetryState::default();

        // Populate two ids.
        state.gen_trackers.insert("a".into(), make_tracker(2));
        state.gen_trackers.insert("b".into(), make_tracker(2));
        state.prev_ttft.insert("a".into(), baseline(1.0, 1.0));
        state.prev_ttft.insert("b".into(), baseline(2.0, 2.0));
        state.prev_tpot.insert("a".into(), baseline(3.0, 3.0));
        state.prev_tpot.insert("b".into(), baseline(4.0, 4.0));

        state.purge("a");

        // "a" must be gone from all maps.
        assert!(
            !state.gen_trackers.contains_key("a"),
            "tracker for a must be removed"
        );
        assert!(
            !state.prev_ttft.contains_key("a"),
            "ttft for a must be removed"
        );
        assert!(
            !state.prev_tpot.contains_key("a"),
            "tpot for a must be removed"
        );
        // "b" must be unaffected.
        assert!(
            state.gen_trackers.contains_key("b"),
            "tracker for b must survive"
        );
        assert!(state.prev_ttft.contains_key("b"), "ttft for b must survive");
        assert!(state.prev_tpot.contains_key("b"), "tpot for b must survive");
    }

    /// `TelemetryState::invalidate_on_endpoint_change` invalidates the tracker
    /// (resets its window so subsequent snapshots return None) and removes the
    /// latency baselines, without removing the tracker entry itself.
    #[test]
    fn invalidate_on_endpoint_change_resets_tracker_and_clears_latency() {
        let mut state = TelemetryState::default();

        let mut tracker = make_tracker(2);
        // Give the tracker a reading so it has a non-None baseline.
        tracker.observe_counter(100.0, at(10));
        tracker.observe_counter(200.0, at(12));
        state.gen_trackers.insert("svc".into(), tracker);
        state.prev_ttft.insert("svc".into(), baseline(1.0, 10.0));
        state.prev_tpot.insert("svc".into(), baseline(0.5, 8.0));

        state.invalidate_on_endpoint_change("svc");

        // Tracker still exists but its window is reset: snapshot returns None.
        let (rate, _meta) = state
            .gen_trackers
            .get_mut("svc")
            .expect("tracker entry must remain")
            .snapshot(at(12));
        assert!(rate.is_none(), "invalidated tracker must yield None rate");
        // Latency baselines must be cleared.
        assert!(
            !state.prev_ttft.contains_key("svc"),
            "ttft baseline must be cleared on endpoint change"
        );
        assert!(
            !state.prev_tpot.contains_key("svc"),
            "tpot baseline must be cleared on endpoint change"
        );
    }

    // ── Discovery reconcile ─────────────────────────────────────────────────
    //
    // The Docker source cannot be driven end to end without a container
    // daemon, and `reconcile_set` is the code both it and the managed-service
    // registry run, so these tests exercise it directly. The registry and
    // Lemonade sources are also driven end to end in
    // `tests/discovery_reconcile.rs`, except the registry's retirement of a
    // vanished service and Lemonade's change of id, which only these tests
    // cover.

    fn discovered(id: &str, port: u16) -> DiscoveredService {
        DiscoveredService {
            container_id: id.into(),
            container_name: format!("name-{id}"),
            model_name: "model".into(),
            port: Some(port),
            ..Default::default()
        }
    }

    fn id_set(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| (*s).to_string()).collect()
    }

    /// The reducer state, telemetry and event stream a reconcile pass works on.
    struct Pass {
        state: State,
        telemetry: TelemetryState,
        tx: broadcast::Sender<Event>,
        rx: broadcast::Receiver<Event>,
    }

    impl Pass {
        fn new() -> Self {
            let (tx, rx) = broadcast::channel(64);
            Self {
                state: State::default(),
                telemetry: TelemetryState::default(),
                tx,
                rx,
            }
        }

        fn reconcile(&mut self) -> Reconcile<'_> {
            Reconcile {
                state: &mut self.state,
                telemetry: &mut self.telemetry,
                tx: &self.tx,
                persist: None,
            }
        }

        /// The announcements broadcast since the last call, as `discovered <id>`
        /// and `gone <id>` strings.
        fn announced(&mut self) -> Vec<String> {
            let mut out = Vec::new();
            while let Ok(ev) = self.rx.try_recv() {
                match ev {
                    Event::InstanceDiscovered(i) => {
                        out.push(format!("discovered {}", i.container_id));
                    }
                    Event::InstanceGone { container_id } => {
                        out.push(format!("gone {container_id}"));
                    }
                    _ => {}
                }
            }
            out
        }

        /// Give `id` a live KV reading and a latency baseline, as a scrape would.
        fn scrape(&mut self, id: &str) {
            let mut inst = self.state.instances.get(id).cloned().expect("instance");
            inst.kv_cache_usage_pct = Some(25.0);
            self.state.apply(StateEvent::InstanceUpserted(inst));
            self.telemetry
                .prev_ttft
                .insert(id.into(), baseline(1.0, 10.0));
            self.telemetry
                .prev_tpot
                .insert(id.into(), baseline(0.5, 8.0));
        }

        fn kv(&self, id: &str) -> Option<f32> {
            self.state
                .instances
                .get(id)
                .and_then(|i| i.kv_cache_usage_pct)
        }
    }

    #[test]
    fn reconcile_set_announces_a_new_service_once() {
        let mut pass = Pass::new();
        let svcs = [discovered("a", 8000)];

        pass.reconcile().reconcile_set(
            DiscoverySource::Docker,
            &svcs,
            &id_set(&[]),
            &id_set(&["a"]),
        );
        assert_eq!(pass.announced(), ["discovered a"]);
        assert!(pass.state.instances.contains_key("a"));

        pass.reconcile().reconcile_set(
            DiscoverySource::Docker,
            &svcs,
            &id_set(&["a"]),
            &id_set(&["a"]),
        );
        assert!(
            pass.announced().is_empty(),
            "a known service must not be announced again"
        );
    }

    #[test]
    fn reconcile_set_keeps_live_fields_while_the_endpoint_is_unchanged() {
        let mut pass = Pass::new();
        let svcs = [discovered("a", 8000)];
        pass.reconcile().reconcile_set(
            DiscoverySource::Docker,
            &svcs,
            &id_set(&[]),
            &id_set(&["a"]),
        );
        pass.scrape("a");

        pass.reconcile().reconcile_set(
            DiscoverySource::Docker,
            &svcs,
            &id_set(&["a"]),
            &id_set(&["a"]),
        );

        assert_eq!(pass.kv("a"), Some(25.0));
        assert!(pass.telemetry.prev_ttft.contains_key("a"));
        assert!(pass.telemetry.prev_tpot.contains_key("a"));
    }

    #[test]
    fn reconcile_set_starts_clean_when_the_port_moves_under_the_same_id() {
        let mut pass = Pass::new();
        pass.reconcile().reconcile_set(
            DiscoverySource::Docker,
            &[discovered("a", 8000)],
            &id_set(&[]),
            &id_set(&["a"]),
        );
        pass.scrape("a");
        pass.announced();

        pass.reconcile().reconcile_set(
            DiscoverySource::Docker,
            &[discovered("a", 9000)],
            &id_set(&["a"]),
            &id_set(&["a"]),
        );

        assert_eq!(pass.state.instances["a"].port, Some(9000));
        assert_eq!(pass.kv("a"), None, "the old endpoint's reading leaked");
        assert!(!pass.telemetry.prev_ttft.contains_key("a"));
        assert!(!pass.telemetry.prev_tpot.contains_key("a"));
        assert!(
            pass.announced().is_empty(),
            "a port change is not a new or a gone service"
        );
    }

    #[test]
    fn reconcile_set_retires_a_vanished_service_and_keeps_the_others() {
        let mut pass = Pass::new();
        let svcs = [discovered("a", 8000), discovered("b", 8001)];
        pass.reconcile().reconcile_set(
            DiscoverySource::Managed,
            &svcs,
            &id_set(&[]),
            &id_set(&["a", "b"]),
        );
        pass.scrape("a");
        pass.scrape("b");
        pass.announced();

        pass.reconcile().reconcile_set(
            DiscoverySource::Managed,
            &svcs[..1],
            &id_set(&["a", "b"]),
            &id_set(&["a"]),
        );

        assert_eq!(pass.announced(), ["gone b"]);
        assert!(!pass.state.instances.contains_key("b"));
        assert!(!pass.telemetry.prev_ttft.contains_key("b"));
        assert!(pass.state.instances.contains_key("a"));
        assert!(pass.telemetry.prev_ttft.contains_key("a"));
    }

    // ── Lemonade state matrix ───────────────────────────────────────────────
    //
    // One Lemonade endpoint is probed per discovery pass. Its id encodes the host
    // and port, so the states a probe can observe are: nothing ever answered; the
    // first answer; the same endpoint again; the host or the port changed; the
    // endpoint went quiet; it came back somewhere else; and the same answer
    // arriving twice in one tick. Each row below checks the announcements, the
    // instances a snapshot would carry, the telemetry kept per id, and the
    // `(id, port)` pairs the vLLM scrape would poll, because a stale instance is
    // visible on all four.

    fn lemonade_svc(host: &str, port: u16) -> DiscoveredService {
        DiscoveredService {
            container_id: format!("lemonade-{host}-{port}"),
            container_name: "lemonade".into(),
            model_name: "model".into(),
            port: Some(port),
            ..Default::default()
        }
    }

    impl Pass {
        /// The ids a snapshot carries. The loop builds a snapshot's instances from
        /// `state.instances`, so this is the dashboard's view of the pass.
        fn instance_ids(&self) -> Vec<String> {
            let mut ids: Vec<String> = self.state.instances.keys().cloned().collect();
            ids.sort();
            ids
        }

        /// The ids the vLLM Prometheus scrape would poll while `current` holds
        /// the live Lemonade ids.
        fn scrape_targets(&self, current: &HashSet<String>) -> Vec<String> {
            let mut ids: Vec<String> =
                vllm_scrape_targets(&self.state.instances, current, &HashSet::new())
                    .into_iter()
                    .map(|(id, _port)| id)
                    .collect();
            ids.sort();
            ids
        }

        /// A live KV reading, latency baselines and a rate tracker for `id`.
        fn give_live_telemetry(&mut self, id: &str) {
            self.scrape(id);
            self.telemetry
                .gen_trackers
                .insert(id.into(), make_tracker(2));
        }

        fn has_telemetry(&self, id: &str) -> bool {
            self.telemetry.gen_trackers.contains_key(id)
                || self.telemetry.prev_ttft.contains_key(id)
                || self.telemetry.prev_tpot.contains_key(id)
        }
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| (*s).to_string()).collect()
    }

    /// Discover `old`, give it live telemetry, then probe `new`. Returns the pass,
    /// the loop's current Lemonade ids and what the second probe announced.
    fn lemonade_identity_change(
        old: &DiscoveredService,
        new: &DiscoveredService,
    ) -> (Pass, HashSet<String>, Vec<String>) {
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();
        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(old), &mut current);
        pass.give_live_telemetry(&old.container_id);
        pass.announced();

        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(new), &mut current);
        let announced = pass.announced();
        (pass, current, announced)
    }

    #[test]
    fn lemonade_that_never_answers_leaves_no_trace() {
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();

        pass.reconcile().reconcile_lemonade(&[], &mut current);
        pass.reconcile().reconcile_lemonade(&[], &mut current);

        assert!(pass.announced().is_empty());
        assert!(current.is_empty());
        assert!(pass.instance_ids().is_empty());
        assert!(pass.scrape_targets(&current).is_empty());
    }

    #[test]
    fn lemonade_first_answer_is_announced_once_and_left_to_its_own_collector() {
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();

        pass.reconcile()
            .reconcile_lemonade(&[lemonade_svc("127.0.0.1", 13305)], &mut current);

        assert_eq!(pass.announced(), ["discovered lemonade-127.0.0.1-13305"]);
        assert_eq!(
            current,
            HashSet::from(["lemonade-127.0.0.1-13305".to_string()])
        );
        assert_eq!(pass.instance_ids(), ids(&["lemonade-127.0.0.1-13305"]));
        assert!(
            pass.scrape_targets(&current).is_empty(),
            "the live Lemonade instance is not a vLLM scrape target"
        );
    }

    #[test]
    fn lemonade_same_endpoint_again_keeps_its_instance_and_live_fields() {
        let svc = lemonade_svc("127.0.0.1", 13305);
        let id = svc.container_id.clone();
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();
        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(&svc), &mut current);
        pass.give_live_telemetry(&id);
        pass.announced();

        pass.reconcile().reconcile_lemonade(&[svc], &mut current);

        assert!(pass.announced().is_empty());
        assert_eq!(pass.instance_ids(), ids(&[&id]));
        assert_eq!(pass.kv(&id), Some(25.0));
        assert!(pass.has_telemetry(&id));
        assert!(pass.scrape_targets(&current).is_empty());
    }

    #[test]
    fn lemonade_host_change_retires_the_old_instance_and_announces_the_new_one() {
        let old = lemonade_svc("127.0.0.1", 13305);
        let new = lemonade_svc("10.0.0.7", 13305);
        let (pass, current, announced) = lemonade_identity_change(&old, &new);

        assert_eq!(current, HashSet::from([new.container_id]));
        assert_eq!(
            announced,
            [
                "gone lemonade-127.0.0.1-13305",
                "discovered lemonade-10.0.0.7-13305"
            ]
        );
        assert!(!pass.has_telemetry(&old.container_id));
        assert_eq!(pass.instance_ids(), ids(&["lemonade-10.0.0.7-13305"]));
        assert!(
            pass.scrape_targets(&current).is_empty(),
            "the old endpoint must not be polled as a vLLM target"
        );
    }

    #[test]
    fn lemonade_port_change_retires_the_old_instance_and_announces_the_new_one() {
        let old = lemonade_svc("127.0.0.1", 13305);
        let new = lemonade_svc("127.0.0.1", 13306);
        let (pass, current, announced) = lemonade_identity_change(&old, &new);

        assert_eq!(current, HashSet::from([new.container_id]));
        assert_eq!(
            announced,
            [
                "gone lemonade-127.0.0.1-13305",
                "discovered lemonade-127.0.0.1-13306"
            ]
        );
        assert!(!pass.has_telemetry(&old.container_id));
        assert_eq!(pass.instance_ids(), ids(&["lemonade-127.0.0.1-13306"]));
        assert!(
            pass.scrape_targets(&current).is_empty(),
            "the old endpoint must not be polled as a vLLM target"
        );
    }

    #[test]
    fn lemonade_endpoint_going_quiet_is_announced_gone_and_leaves_no_trace() {
        let svc = lemonade_svc("127.0.0.1", 13305);
        let id = svc.container_id.clone();
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();
        pass.reconcile().reconcile_lemonade(&[svc], &mut current);
        pass.give_live_telemetry(&id);
        pass.announced();

        pass.reconcile().reconcile_lemonade(&[], &mut current);

        assert_eq!(pass.announced(), [format!("gone {id}")]);
        assert!(current.is_empty());
        assert!(pass.instance_ids().is_empty());
        assert!(!pass.has_telemetry(&id));
        assert!(pass.scrape_targets(&current).is_empty());
    }

    #[test]
    fn lemonade_coming_back_elsewhere_after_going_quiet_is_the_only_instance() {
        let old = lemonade_svc("127.0.0.1", 13305);
        let new = lemonade_svc("127.0.0.1", 13306);
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();
        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(&old), &mut current);
        pass.give_live_telemetry(&old.container_id);
        pass.reconcile().reconcile_lemonade(&[], &mut current);
        assert_eq!(
            pass.announced(),
            [
                "discovered lemonade-127.0.0.1-13305",
                "gone lemonade-127.0.0.1-13305"
            ]
        );

        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(&new), &mut current);

        assert_eq!(pass.announced(), ["discovered lemonade-127.0.0.1-13306"]);
        assert_eq!(current, HashSet::from([new.container_id.clone()]));
        assert_eq!(pass.instance_ids(), ids(&[&new.container_id]));
        assert!(!pass.has_telemetry(&old.container_id));
        assert!(pass.scrape_targets(&current).is_empty());
    }

    #[test]
    fn lemonade_answer_repeated_within_one_tick_is_announced_once() {
        let svc = lemonade_svc("127.0.0.1", 13305);
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();

        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(&svc), &mut current);
        pass.reconcile().reconcile_lemonade(&[svc], &mut current);

        assert_eq!(pass.announced(), ["discovered lemonade-127.0.0.1-13305"]);
        assert_eq!(pass.instance_ids(), ids(&["lemonade-127.0.0.1-13305"]));
        assert!(pass.scrape_targets(&current).is_empty());
    }

    #[test]
    fn lemonade_endpoint_moving_away_and_back_is_retired_each_time() {
        let a = lemonade_svc("127.0.0.1", 13305);
        let b = lemonade_svc("127.0.0.1", 13306);
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();
        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(&a), &mut current);
        pass.announced();

        pass.reconcile().reconcile_lemonade(&[b], &mut current);
        assert_eq!(
            pass.announced(),
            [
                "gone lemonade-127.0.0.1-13305",
                "discovered lemonade-127.0.0.1-13306"
            ]
        );
        pass.reconcile()
            .reconcile_lemonade(std::slice::from_ref(&a), &mut current);

        assert_eq!(
            pass.announced(),
            [
                "gone lemonade-127.0.0.1-13306",
                "discovered lemonade-127.0.0.1-13305"
            ]
        );
        assert_eq!(current, HashSet::from([a.container_id.clone()]));
        assert_eq!(pass.instance_ids(), ids(&[&a.container_id]));
        assert!(pass.scrape_targets(&current).is_empty());
    }

    #[test]
    fn reconcile_lemonade_follows_one_endpoint_through_its_lifecycle() {
        let mut pass = Pass::new();
        let mut current: HashSet<String> = HashSet::new();

        pass.reconcile()
            .reconcile_lemonade(&[discovered("l1", 13305)], &mut current);
        assert_eq!(pass.announced(), ["discovered l1"]);
        assert_eq!(current, HashSet::from(["l1".to_string()]));

        pass.scrape("l1");
        pass.reconcile()
            .reconcile_lemonade(&[discovered("l1", 13305)], &mut current);
        assert!(
            pass.announced().is_empty(),
            "same id must not be re-announced"
        );
        assert_eq!(pass.kv("l1"), Some(25.0), "live fields carry over");

        // A new id (the host or port changed) retires the old id, telemetry
        // included, and announces the new one.
        pass.reconcile()
            .reconcile_lemonade(&[discovered("l2", 13306)], &mut current);
        assert_eq!(pass.announced(), ["gone l1", "discovered l2"]);
        assert_eq!(current, HashSet::from(["l2".to_string()]));
        assert!(!pass.telemetry.prev_ttft.contains_key("l1"));
        // The superseded instance leaves the state (and so every snapshot), not
        // only its telemetry.
        assert!(!pass.state.instances.contains_key("l1"));

        pass.reconcile().reconcile_lemonade(&[], &mut current);
        assert_eq!(pass.announced(), ["gone l2"]);
        assert!(current.is_empty());
        assert!(!pass.state.instances.contains_key("l2"));

        pass.reconcile().reconcile_lemonade(&[], &mut current);
        assert!(pass.announced().is_empty(), "nothing left to retire");
    }

    // Registry read health: one test per row of the state matrix (first
    // failure, repeated failure, recovery, empty, populated, malformed new,
    // malformed unchanged, malformed cleared).

    fn registry_read(
        malformed: &[&str],
    ) -> Result<crate::registry::RegistryRead, crate::registry::RegistryReadError> {
        Ok(crate::registry::RegistryRead {
            records: Vec::new(),
            malformed: malformed.iter().map(PathBuf::from).collect(),
        })
    }

    fn registry_failure(
        why: &str,
    ) -> Result<crate::registry::RegistryRead, crate::registry::RegistryReadError> {
        Err(crate::registry::RegistryReadError {
            path: PathBuf::from("services"),
            source: std::io::Error::other(why),
            partial: crate::registry::RegistryRead::default(),
        })
    }

    /// State matrix for what a partially read registry does to the discovery
    /// state, one assertion group per row: a service it saw joins the known set
    /// and its non-vLLM mark follows the record; one it did not see stays known
    /// and keeps its mark; nothing leaves either set.
    #[test]
    fn partial_registry_read_adds_what_it_saw_and_forgets_nothing() {
        let mut state = DiscoveryState {
            known_services: id_set(&["old-vllm", "old-lemonade", "flips"]),
            managed_non_vllm: id_set(&["old-lemonade", "flips"]),
            ..DiscoveryState::default()
        };

        state.keep_through_partial_registry(crate::registry::ManagedDiscovery {
            seen: id_set(&["new-lemonade", "new-vllm", "flips"]),
            non_vllm: id_set(&["new-lemonade"]),
            ..Default::default()
        });

        assert_eq!(
            state.known_services,
            id_set(&[
                "old-vllm",
                "old-lemonade",
                "flips",
                "new-lemonade",
                "new-vllm"
            ]),
            "the unseen services stay known and the new ones join them"
        );
        assert_eq!(
            state.managed_non_vllm,
            id_set(&["old-lemonade", "new-lemonade"]),
            "an unseen non-vLLM service keeps its mark, a seen one follows its record"
        );
    }

    #[test]
    fn partial_registry_read_that_saw_nothing_changes_nothing() {
        let mut state = DiscoveryState {
            known_services: id_set(&["a", "b"]),
            managed_non_vllm: id_set(&["b"]),
            ..DiscoveryState::default()
        };

        state.keep_through_partial_registry(crate::registry::ManagedDiscovery::default());

        assert_eq!(state.known_services, id_set(&["a", "b"]));
        assert_eq!(state.managed_non_vllm, id_set(&["b"]));
    }

    #[test]
    fn registry_health_reports_the_first_failure_of_a_streak_once() {
        let mut health = RegistryHealth::default();

        let first = health.observe(&registry_failure("disk gone"));
        assert_eq!(
            first.failed.as_deref(),
            Some("cannot read services: disk gone"),
            "the first failed pass names the error"
        );
        assert!(!first.recovered);

        for pass in 2..=5 {
            assert_eq!(
                health.observe(&registry_failure("disk gone")),
                RegistryChange::default(),
                "failed pass {pass} of the same streak must stay quiet"
            );
        }
    }

    #[test]
    fn registry_health_reports_recovery_once_and_a_new_streak_again() {
        let mut health = RegistryHealth::default();
        health.observe(&registry_failure("one"));
        health.observe(&registry_failure("one"));

        let recovered = health.observe(&registry_read(&[]));
        assert!(recovered.recovered);
        assert!(recovered.failed.is_none());
        assert_eq!(
            health.observe(&registry_read(&[])),
            RegistryChange::default(),
            "only the first good pass after a streak is a recovery"
        );

        let again = health.observe(&registry_failure("two"));
        assert_eq!(again.failed.as_deref(), Some("cannot read services: two"));
    }

    #[test]
    fn registry_health_is_quiet_for_an_empty_or_populated_registry() {
        let mut health = RegistryHealth::default();
        // A healthy read, empty or not, says nothing: there is no failure to
        // report and no failure to have recovered from.
        for _ in 0..3 {
            assert_eq!(
                health.observe(&registry_read(&[])),
                RegistryChange::default()
            );
        }
    }

    #[test]
    fn registry_health_reports_a_malformed_set_when_it_changes() {
        let mut health = RegistryHealth::default();

        let first = health.observe(&registry_read(&["a.json"]));
        assert_eq!(first.malformed, [PathBuf::from("a.json")]);
        assert_eq!(
            health.observe(&registry_read(&["a.json"])),
            RegistryChange::default(),
            "an unchanged malformed set is not reported again"
        );

        let grown = health.observe(&registry_read(&["a.json", "b.json"]));
        assert_eq!(
            grown.malformed,
            [PathBuf::from("a.json"), PathBuf::from("b.json")],
            "a newly malformed record reports the whole set"
        );

        let cleared = health.observe(&registry_read(&[]));
        assert!(
            cleared.malformed.is_empty(),
            "clearing the set is not a warning"
        );
        let back = health.observe(&registry_read(&["a.json"]));
        assert_eq!(
            back.malformed,
            [PathBuf::from("a.json")],
            "a record that goes bad again after being fixed is reported again"
        );
    }

    #[test]
    fn registry_health_keeps_the_malformed_set_across_a_read_failure() {
        let mut health = RegistryHealth::default();
        health.observe(&registry_read(&["a.json"]));
        health.observe(&registry_failure("blip"));

        let after = health.observe(&registry_read(&["a.json"]));
        assert!(after.recovered);
        assert!(
            after.malformed.is_empty(),
            "the same corrupt record is not news just because a read failed in between"
        );
    }

    #[test]
    fn partial_registry_read_refreshes_the_roots_it_saw_and_keeps_the_unseen() {
        use rocm_dash_collectors::ancestry::Root;
        let root = |pid, owner: &str| Root {
            pid,
            start_ticks: None,
            owner: owner.into(),
        };
        let mut state = DiscoveryState {
            known_services: id_set(&["seen", "unseen"]),
            managed_roots: vec![root(10, "seen"), root(20, "unseen")],
            ..DiscoveryState::default()
        };

        // `seen` was relaunched under a new engine pid; `unseen` was not read.
        state.keep_through_partial_registry(crate::registry::ManagedDiscovery {
            seen: id_set(&["seen"]),
            roots: vec![root(11, "seen")],
            ..Default::default()
        });

        let mut got: Vec<_> = state
            .managed_roots
            .iter()
            .map(|r| (r.pid, r.owner.as_str()))
            .collect();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![(11, "seen"), (20, "unseen")],
            "a seen service's roots are replaced, an unseen one's survive"
        );
    }

    #[test]
    fn partial_registry_read_drops_the_roots_of_a_service_it_saw_stop() {
        use rocm_dash_collectors::ancestry::Root;
        let root = |pid, owner: &str| Root {
            pid,
            start_ticks: None,
            owner: owner.into(),
        };
        let mut state = DiscoveryState {
            managed_roots: vec![root(10, "stopped"), root(20, "unseen")],
            ..DiscoveryState::default()
        };
        // `stopped` was read this pass and is no longer running; an unrelated
        // record failed to read, which is what makes this a partial pass.
        state.keep_through_partial_registry(crate::registry::ManagedDiscovery {
            retired: id_set(&["stopped"]),
            ..Default::default()
        });
        let owners: Vec<_> = state
            .managed_roots
            .iter()
            .map(|r| r.owner.as_str())
            .collect();
        assert_eq!(
            owners,
            vec!["unseen"],
            "a stopped service's PID must not stay matchable"
        );
    }

    #[test]
    fn partial_registry_read_that_saw_nothing_keeps_every_root() {
        use rocm_dash_collectors::ancestry::Root;
        let mut state = DiscoveryState {
            managed_roots: vec![Root {
                pid: 10,
                start_ticks: Some(5),
                owner: "a".into(),
            }],
            ..DiscoveryState::default()
        };
        state.keep_through_partial_registry(crate::registry::ManagedDiscovery::default());
        assert_eq!(state.managed_roots.len(), 1);
    }
}
