// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The quit-confirm prompt: the live-instance gates that decide whether quitting
//! needs one, and opening, answering and resolving it.
//!
//! A further `impl AppState` block, split out of `app/mod.rs` to keep the core
//! reducer focused. The gates read the live managed-registry snapshot (or the
//! first real snapshot once one lands); the prompt itself is rendered by the `ui`
//! layer (`quit_confirm_request`).

use super::{AppState, Modal};

impl AppState {
    /// True when *some* instance is actively serving from the user's point of
    /// view, whether this host's own registry manages it or it was
    /// Docker-/Lemonade-discovered from an external daemon — the gate for the
    /// confirm-before-quit prompt.
    ///
    /// Always `false` under `simulated` (`--demo`/`--replay`): a demo session
    /// synthesizes `Instance { status: Running, .. }` entries so the Serving
    /// tab has something to show, but nothing is actually running, so warning
    /// before quitting would print a false claim (AGENTS.md §3, issue #145's
    /// own prompt text: "it may still be running in the background after you
    /// quit").
    ///
    /// Otherwise ORs in a *live* registry read (`live_registry_snapshot`)
    /// until the daemon's first instance snapshot lands
    /// (`has_received_snapshot`): `instances` starts empty and is only
    /// populated once that snapshot arrives (see `push_snapshot`), so a `q`
    /// pressed in that startup window would otherwise see this return `false`
    /// even though a model actually was being served, simply because the live
    /// state hadn't arrived yet. On a host whose dashboard-embedded daemon
    /// client never connects at all (Windows — `client.rs`'s connect path is
    /// unix-only), `has_received_snapshot` never flips, so the live read
    /// stays authoritative for the entire session rather than only a brief
    /// startup window — the registry read is cheap and re-run every time this
    /// is called, so there is no staleness to outlive. Once a snapshot lands
    /// on a host where the daemon does connect, `instances` is authoritative
    /// and the read's *result* is no longer consulted for THIS question (a
    /// model that legitimately stops later in the session is still reflected
    /// correctly) — the read itself still runs unconditionally every call (it
    /// is cheap, per above), its answer is just ignored here once
    /// `has_received_snapshot` is `true`. `has_stoppable_managed_instance_with`
    /// below, in contrast, both reads the registry live AND consults that
    /// result unconditionally, for a different question (which ids are
    /// reachable).
    ///
    /// **Known gap:** pointing the dashboard at a non-local daemon via the
    /// `dashboard.tui.connect` config value, where that daemon does not
    /// discover *this* host's `services_dir`, is not accounted for. Once a
    /// snapshot lands, `instances` is sourced from that other daemon and this
    /// host's local registry read is ignored for this question (per above),
    /// so a model served here would not open the prompt. Not a case this
    /// gate is built to handle.
    pub(crate) fn has_live_instance(&self) -> bool {
        self.has_live_instance_with(&self.live_registry_snapshot())
    }

    /// Same question as [`has_live_instance`](Self::has_live_instance), against
    /// an already-fetched snapshot — for a caller that must ask this alongside
    /// another registry-backed question (e.g.
    /// [`has_stoppable_managed_instance_with`](Self::has_stoppable_managed_instance_with))
    /// and needs both answered from the same on-disk read, not two independent
    /// ones that a registry write in between could answer differently.
    pub(crate) fn has_live_instance_with(
        &self,
        snapshot: &crate::registry_probe::ManagedRegistrySnapshot,
    ) -> bool {
        if self.simulated {
            return false;
        }
        self.instances.values().any(|i| i.status.is_serving())
            || (!self.has_received_snapshot && snapshot.has_live_service)
    }

    /// Read the bin-injected live registry probe now, or a default (empty,
    /// nothing live) snapshot in the one path that has no probe to read: a
    /// test building `AppState` directly rather than through `resolved_args`,
    /// which injects `Some` on every real entry point, including
    /// demo/replay/mock. Single call site for the registry-read seam so
    /// `has_live_instance`,
    /// `has_stoppable_managed_instance_with`, `quit_confirm_open_services_key`,
    /// and `ui::quit_confirm::quit_confirm_body_with` all read the exact same live
    /// classification rather than each unwrapping `managed_registry_probe` by
    /// hand.
    pub(crate) fn live_registry_snapshot(&self) -> crate::registry_probe::ManagedRegistrySnapshot {
        self.managed_registry_probe
            .as_ref()
            .map_or_else(Default::default, |probe| probe.snapshot())
    }

    /// Whether at least one currently-serving instance is one this host's own
    /// registry manages (a live read — see `live_registry_snapshot`) — i.e.
    /// one `rocm services stop` can actually reach. Shared by
    /// `ui::quit_confirm::quit_confirm_body_with` (whether to show the "press s" hint) and
    /// `quit_confirm_open_services_key` (whether `s` does anything), so the
    /// two can't drift on when stopping is actually offered. Takes an
    /// already-fetched `snapshot` — see
    /// [`has_live_instance_with`](Self::has_live_instance_with) for why: both
    /// call sites also ask a sibling registry-backed question in the same
    /// decision and need the two answered from one read, not two independent
    /// ones a registry write in between could answer differently.
    pub(crate) fn has_stoppable_managed_instance_with(
        &self,
        snapshot: &crate::registry_probe::ManagedRegistrySnapshot,
    ) -> bool {
        self.instances
            .values()
            .any(|i| i.status.is_serving() && snapshot.managed_ids.contains(&i.container_id))
    }

    /// Whether the confirm-before-quit prompt is open. Mirrors
    /// [`approval_pending`](Self::approval_pending): the event loop gives it
    /// its own match arm above the normal per-tab dispatch, so no overlay can
    /// pre-empt the decision.
    pub(crate) const fn quit_confirm_pending(&self) -> bool {
        self.quit_confirm.is_some()
    }

    /// Open the confirm-before-quit prompt with the shared default choice
    /// (Approve — the user already asked to quit; this just confirms it).
    /// Closes any operational overlay first, mirroring `open_approval`, so the
    /// prompt owns focus alone. Also closes any open `modal` (Esc menu,
    /// Options, Detail, ThemePicker, Help, …): `q` reaches `KeyAction::Quit`
    /// from inside every one of those (see their per-modal arms above this
    /// impl in `actions.rs`), and declining must land back on the plain
    /// dashboard, not strand the user inside whatever was open when they
    /// pressed `q`. `close_overlays` also resets console scroll position —
    /// accepted here the same way it already is for `open_approval`, which
    /// has called it for exactly this reason since before issue #145.
    pub(crate) fn open_quit_confirm(&mut self) {
        self.close_overlays();
        self.modal = Modal::None;
        self.quit_confirm = Some(crate::ui::approval::ApprovalChoice::default());
    }

    /// Route a key to the open quit-confirm prompt: move the cursor and
    /// return a verdict if the key resolved one. No-op returning `None` when
    /// no prompt is open.
    pub(crate) fn on_quit_confirm_key(
        &mut self,
        code: crossterm::event::KeyCode,
    ) -> Option<crate::ui::approval::ApprovalVerdict> {
        let choice = self.quit_confirm.as_mut()?;
        let (new_choice, verdict) = crate::ui::approval::approval_key(code, *choice);
        *choice = new_choice;
        verdict
    }

    /// Resolve a key against the open quit-confirm prompt to completion:
    /// routes it via [`on_quit_confirm_key`](Self::on_quit_confirm_key), and
    /// on a verdict also applies the verdict's side effect — clearing the
    /// prompt on Deny/Cancel (Approve leaves it for the caller, which is
    /// about to exit anyway). Returns `Some(true)` on Approve (caller should
    /// exit), `Some(false)` on Deny/Cancel (prompt closed, caller continues),
    /// or `None` while the cursor merely moved / an irrelevant key arrived
    /// (prompt stays open). Shared by the dashboard event loop
    /// (`app::event_loop`) and the pre-dashboard launcher
    /// (`ui::launcher::handle_launcher_key`) so the two can't drift apart on
    /// what a quit-confirm verdict actually does.
    pub(crate) fn resolve_quit_confirm_key(
        &mut self,
        code: crossterm::event::KeyCode,
    ) -> Option<bool> {
        use crate::ui::approval::ApprovalVerdict;
        match self.on_quit_confirm_key(code)? {
            ApprovalVerdict::Approve => Some(true),
            ApprovalVerdict::Deny | ApprovalVerdict::Cancel => {
                self.quit_confirm = None;
                Some(false)
            }
        }
    }

    /// The quit-confirm prompt's `s` ("stop…") key (issue #145): rather than
    /// only warning that a model may still be running, jump straight to the Services
    /// overlay — pre-focused on a managed instance that's actually serving —
    /// so the user can stop it there through its own proven approval+job+
    /// console flow (`ui::services_manager`), instead of only being told the
    /// shell command to run themselves.
    ///
    /// Dashboard-only: the pre-dashboard launcher shares this prompt's text
    /// but has no live Services overlay to jump to (it isn't connected to a
    /// daemon and has no `services` field rendering path of its own), so its
    /// own key routing never calls this. Returns `true` when handled — the
    /// caller must not also resolve `code` through
    /// [`resolve_quit_confirm_key`](Self::resolve_quit_confirm_key) in that
    /// case. Returns `false` (prompt not open, key isn't `s`, or nothing
    /// stoppable is currently serving) when the caller should fall through to
    /// that normal resolution instead.
    ///
    /// The short-circuit above calls
    /// [`has_stoppable_managed_instance_with`](Self::has_stoppable_managed_instance_with)
    /// on exactly the same predicate the row-finding `position` closure below
    /// re-checks per row (`is_serving() && managed_ids.contains(..)`), so it
    /// cannot change which outcome this returns — only skip it sooner. Its
    /// value is avoiding `service_rows`'s allocation and sort on every `s`
    /// keypress when nothing is stoppable; it is not a correctness guard, and
    /// there is nothing for the two checks to drift on.
    pub(crate) fn quit_confirm_open_services_key(
        &mut self,
        code: crossterm::event::KeyCode,
    ) -> bool {
        if !self.quit_confirm_pending() || code != crossterm::event::KeyCode::Char('s') {
            return false;
        }
        // One read for both questions below: a registry write landing between
        // two separate reads could let the short-circuit see a stoppable
        // instance that the row lookup's own read no longer does, silently
        // swallowing the keypress (the prompt stays open with no feedback).
        let snapshot = self.live_registry_snapshot();
        if !self.has_stoppable_managed_instance_with(&snapshot) {
            return false;
        }
        let managed_ids = snapshot.managed_ids;
        let rows = crate::ui::services_manager::service_rows(&self.instances);
        let Some(selected) = rows.iter().position(|r| {
            managed_ids.contains(&r.id)
                && self
                    .instances
                    .get(&r.id)
                    .is_some_and(|i| i.status.is_serving())
        }) else {
            return false;
        };
        self.close_overlays();
        self.services = Some(crate::ui::services_manager::ServicesManagerState {
            selected,
            ..Default::default()
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::{st, with_registry_probe};
    use rocm_dash_core::metrics::Snapshot;
    use rocm_dash_core::protocol::Event;

    #[test]
    fn open_quit_confirm_closes_other_overlays() {
        // Mirrors open_approval's own close_overlays() call: the
        // confirm-before-quit prompt owns focus alone, the same way chat
        // approval does.
        let mut s = st();
        s.runtime_manager = Some(crate::ui::runtime_manager::RuntimeManagerState::default());
        s.open_quit_confirm();
        assert!(
            s.runtime_manager.is_none(),
            "open_quit_confirm must close any open overlay first"
        );
    }

    #[test]
    fn has_live_instance_trusts_a_live_registry_read_before_first_real_snapshot_lands() {
        // Issue #145 startup race: `instances` starts empty and is only
        // populated once the daemon's first snapshot arrives, so without a
        // live registry read a `q` pressed in that window saw no live
        // instance and quit with no confirm prompt even if a model actually
        // was being served. Also covers the Windows case (no daemon ever
        // connects, so no snapshot ever arrives): the live read stays
        // authoritative for the entire session, not just a startup window.
        let mut s = st();
        assert!(s.instances.is_empty());
        assert!(!s.has_live_instance(), "nothing seeded, nothing serving");

        with_registry_probe(&mut s, true, &[]);
        assert!(
            s.has_live_instance(),
            "the live registry read must stand in before the first snapshot"
        );
    }

    #[test]
    fn has_live_instance_stops_trusting_the_registry_read_once_a_real_snapshot_lands() {
        // The registry read must not outrank a real daemon snapshot once one
        // has arrived: with an empty (nothing-serving) snapshot, `instances`
        // is authoritative and the read's result is no longer consulted for
        // this question — even though the read itself still runs and would
        // still say something is live if its answer were asked for.
        let mut s = st();
        with_registry_probe(&mut s, true, &[]);
        assert!(s.has_live_instance());

        s.apply_event(Event::Snapshot(Snapshot::default()));
        assert!(
            !s.has_live_instance(),
            "a real (even empty) snapshot must retire the registry-read fallback"
        );
    }

    #[test]
    fn has_live_instance_is_always_false_under_simulated() {
        // `--demo`/`--replay` synthesize `Instance { status: Running, .. }`
        // entries (the Serving tab needs something to show), but nothing is
        // actually running, so the quit-confirm gate must never trust them —
        // otherwise it prints "it may still be running in the background"
        // when nothing does (AGENTS.md §3, issue #145).
        let mut s = st();
        s.simulated = true;
        s.instances.insert(
            "id0".to_string(),
            rocm_dash_core::metrics::Instance {
                container_id: "id0".to_string(),
                status: rocm_dash_core::metrics::InstanceStatus::Running,
                ..Default::default()
            },
        );
        with_registry_probe(&mut s, true, &[]);
        assert!(
            !s.has_live_instance(),
            "a simulated session must never report a live instance, \
             whether from a snapshot or the registry-read fallback"
        );
    }

    #[test]
    fn has_live_instance_ignores_a_non_serving_instance() {
        // Regression guard: a seeded instance whose status is not
        // Ready/Running (Starting/Stopped/Error/Unknown) must not make
        // `has_live_instance` report true — only `is_serving()` statuses
        // count, matching the quit-confirm prompt's own claim that something
        // is actually being served.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        for status in [
            InstanceStatus::Starting { phase: None },
            InstanceStatus::Stopped,
            InstanceStatus::Error,
            InstanceStatus::Unknown,
        ] {
            let mut s = st();
            s.instances.insert(
                "id0".to_string(),
                Instance {
                    container_id: "id0".to_string(),
                    status,
                    ..Default::default()
                },
            );
            assert!(
                !s.has_live_instance(),
                "status {status:?} must not count as a live instance"
            );
        }
    }

    #[test]
    fn has_stoppable_managed_instance_ignores_a_non_serving_managed_instance() {
        // Regression guard: a managed id in the live registry snapshot whose
        // instance status is not Ready/Running must not make this report
        // true — matching `has_live_instance`'s own `is_serving()` gate, so
        // the quit-confirm prompt never offers to jump to Services for
        // something that isn't actually running.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        for status in [
            InstanceStatus::Starting { phase: None },
            InstanceStatus::Stopped,
            InstanceStatus::Error,
            InstanceStatus::Unknown,
        ] {
            let mut s = st();
            s.instances.insert(
                "svc-managed".to_string(),
                Instance {
                    container_id: "svc-managed".to_string(),
                    status,
                    ..Default::default()
                },
            );
            with_registry_probe(&mut s, true, &["svc-managed"]);
            assert!(
                !s.has_stoppable_managed_instance_with(&s.live_registry_snapshot()),
                "status {status:?} must not count as stoppable"
            );
        }
    }

    #[test]
    fn quit_confirm_open_services_key_skips_a_non_serving_row_that_sorts_first() {
        // Regression guard: the row-finding `position` closure in
        // `quit_confirm_open_services_key` re-checks `is_serving()` per row
        // (see its doc comment). The sibling test above only ever seeds a
        // single non-serving instance, so `has_stoppable_managed_instance_with`'s
        // own short-circuit already returns `false` before `position` ever
        // runs — weakening the closure's `is_serving()` check to `.is_some()`
        // would still pass it. Seed a non-serving row ("svc-a", sorts first by
        // id, same key `services_manager::service_rows` sorts by) alongside a
        // serving one ("svc-b"), so `position` must actually skip row 0 to
        // land on row 1.
        use crossterm::event::KeyCode;
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut s = st();
        s.instances.insert(
            "svc-a".to_string(),
            Instance {
                container_id: "svc-a".to_string(),
                status: InstanceStatus::Stopped,
                ..Default::default()
            },
        );
        s.instances.insert(
            "svc-b".to_string(),
            Instance {
                container_id: "svc-b".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        with_registry_probe(&mut s, true, &["svc-a", "svc-b"]);
        s.open_quit_confirm();
        assert!(s.quit_confirm_open_services_key(KeyCode::Char('s')));
        let selected = s.services.expect("s must open Services").selected;
        assert_eq!(
            selected, 1,
            "row 0 (svc-a, stopped) must be skipped; row 1 (svc-b, running) must be selected"
        );
    }

    #[test]
    fn quit_confirm_open_services_key_ignores_a_non_serving_managed_instance() {
        // Regression guard, same predicate as the sibling test above but
        // through the actual `s`-key handler: a managed id that isn't
        // currently serving must not make `s` open Services.
        use crossterm::event::KeyCode;
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        for status in [
            InstanceStatus::Starting { phase: None },
            InstanceStatus::Stopped,
            InstanceStatus::Error,
            InstanceStatus::Unknown,
        ] {
            let mut s = st();
            s.instances.insert(
                "svc-managed".to_string(),
                Instance {
                    container_id: "svc-managed".to_string(),
                    status,
                    ..Default::default()
                },
            );
            with_registry_probe(&mut s, true, &["svc-managed"]);
            s.open_quit_confirm();
            assert!(
                !s.quit_confirm_open_services_key(KeyCode::Char('s')),
                "status {status:?} must not make `s` open Services"
            );
            assert!(s.services.is_none());
        }
    }

    #[test]
    fn quit_confirm_open_services_key_focuses_the_stoppable_managed_row() {
        use crossterm::event::KeyCode;
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut s = AppState::new("t".into(), "default-dark".into());
        // Sorts before "svc-managed" by id (`service_rows` sorts by id), so a
        // naive "just pick row 0" would wrongly focus this unmanaged one.
        s.instances.insert(
            "docker-abc".to_string(),
            Instance {
                container_id: "docker-abc".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        s.instances.insert(
            "svc-managed".to_string(),
            Instance {
                container_id: "svc-managed".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        with_registry_probe(&mut s, true, &["svc-managed"]);
        s.open_quit_confirm();

        assert!(s.quit_confirm_open_services_key(KeyCode::Char('s')));
        assert!(
            s.quit_confirm.is_none(),
            "the quit-confirm prompt must close once `s` is handled"
        );
        let services = s
            .services
            .as_ref()
            .expect("`s` must open the Services overlay");
        let rows = crate::ui::services_manager::service_rows(&s.instances);
        assert_eq!(
            rows[services.selected].id, "svc-managed",
            "must focus the stoppable managed row, not whichever sorts first: {rows:?}"
        );
    }

    #[test]
    fn quit_confirm_open_services_key_is_a_no_op_without_a_stoppable_managed_instance() {
        use crossterm::event::KeyCode;
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.instances.insert(
            "docker-abc".to_string(),
            Instance {
                container_id: "docker-abc".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        s.open_quit_confirm();

        assert!(!s.quit_confirm_open_services_key(KeyCode::Char('s')));
        assert!(
            s.quit_confirm.is_some(),
            "nothing to stop: the prompt must stay open for its normal y/n/Esc handling"
        );
        assert!(s.services.is_none());
    }

    #[test]
    fn quit_confirm_open_services_key_ignores_other_keys_and_a_closed_prompt() {
        use crossterm::event::KeyCode;
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(
            !s.quit_confirm_open_services_key(KeyCode::Char('s')),
            "must not fire while the prompt isn't open"
        );

        s.open_quit_confirm();
        assert!(
            !s.quit_confirm_open_services_key(KeyCode::Char('y')),
            "must not fire for any key other than `s`"
        );
        assert!(s.quit_confirm.is_some());
    }
}
