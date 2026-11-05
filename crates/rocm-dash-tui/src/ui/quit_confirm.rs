// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The quit-confirm prompt's content: the title, the body listing what is still
//! being served, and the optional "press s" stop hint.
//!
//! Split out of `ui/mod.rs`. `quit_confirm_request` is re-exported from
//! `ui/mod.rs`, so the pre-dashboard launcher's `crate::ui::quit_confirm_request`
//! path is unchanged. The gates that decide whether the prompt opens at all live
//! in `app/quit_confirm.rs`.

use crate::app::AppState;
use crate::ui::{approval, format};

/// Build the quit-confirm prompt's request (title + body) once, so the
/// dashboard's own render arm above and the pre-dashboard launcher's
/// (`ui::launcher::draw`) call the same construction instead of each
/// hand-copying the title string — the same launcher/dashboard drift
/// `quit_confirm_body`, `resolve_quit_confirm_key`, and `is_running` are each
/// also shared to avoid.
///
/// The title is derived from the same decision as the body
/// (`nothing_is_currently_being_served_with`) rather than hardcoded, so the two
/// can't contradict each other the way they used to when a model stopped
/// (or was never confirmed live) while the prompt sat open: "May still be
/// serving" above a body that says "Nothing is being served anymore."
/// underneath it.
///
/// `offer_stop_action` gates the trailing "press s" hint (issue #145): `true`
/// from the dashboard's own render arm, which can actually act on `s`
/// (`AppState::quit_confirm_open_services_key`); `false` from the
/// pre-dashboard launcher, which shares this prompt's text but has no live
/// Services overlay to jump to, so advertising a key it can't honor would be
/// exactly the kind of printed claim AGENTS.md §3 rules out.
pub(crate) fn quit_confirm_request(
    state: &AppState,
    offer_stop_action: bool,
) -> approval::ApprovalRequest {
    // One snapshot for the whole decision: the title, the body, and the
    // stop-hint below all read it instead of each doing their own
    // `live_registry_snapshot()` call, so a registry write mid-render can no
    // longer make them disagree the way "May still be serving" over a body
    // that says "Nothing is being served anymore." used to be possible (the doc comment
    // on `nothing_is_currently_being_served_with` below is only true because
    // of this single read).
    let snapshot = state.live_registry_snapshot();
    let title = if nothing_is_currently_being_served_with(state, &snapshot) {
        "Nothing to confirm — quit?"
    } else {
        "May still be serving — quit anyway?"
    };
    let mut body = quit_confirm_body_with(state, &snapshot);
    if offer_stop_action && state.has_stoppable_managed_instance_with(&snapshot) {
        body.push(String::new());
        body.push("Press s to open Services and stop it.".to_string());
    }
    approval::ApprovalRequest::new(title, body)
}

/// Whether the authoritative view — a real snapshot if one has landed,
/// otherwise a live registry read — says nothing is currently serving.
/// Shared by `quit_confirm_request` (title) and `quit_confirm_body_with` (its
/// own early-return branch) so the two can't independently decide this and
/// drift apart. Expressed directly via `AppState::has_live_instance_with` —
/// the exact same question, already answered there, against the caller's own
/// already-fetched `snapshot` rather than a fresh read — not re-deriving the
/// same `instances`/registry-fallback OR by hand a second time (the
/// `simulated` short-circuit that function also applies is a no-op here in
/// practice: the prompt can only be open at all once `has_live_instance` was
/// already `true` when it was opened, and `simulated` never changes
/// mid-session).
pub(super) fn nothing_is_currently_being_served_with(
    state: &AppState,
    snapshot: &crate::registry_probe::ManagedRegistrySnapshot,
) -> bool {
    !state.has_live_instance_with(snapshot)
}

/// Fetch-once convenience wrapper over
/// [`quit_confirm_body_with`] for a caller that only needs the body in
/// isolation (every unit test below) — `quit_confirm_request` calls the
/// `_with` form directly so it shares one snapshot across the title, body,
/// and stop-hint instead of each fetching its own.
#[cfg(test)]
fn quit_confirm_body(state: &AppState) -> Vec<String> {
    let snapshot = state.live_registry_snapshot();
    quit_confirm_body_with(state, &snapshot)
}

/// Body text for the quit-confirm prompt: name the models still serving
/// (mirroring the serving rows of the Serving tab's "Running now" listing in
/// `tabs::pane::live_lines`'s `OpenServices` arm, as plain lines instead of
/// styled spans; that listing also shows starting services, which this gate
/// deliberately does not count), and how to stop each one that this host's own registry
/// manages (a live read, passed in as `snapshot` so the caller controls how
/// many times the registry is actually read — see `quit_confirm_request`) —
/// not a blanket command, since a Docker- or Lemonade-discovered instance
/// from an external daemon cannot be stopped that way.
///
/// The trailer is hedged ("may still be running") rather than asserted
/// present-tense ("keeps running"), for every caller — not only the
/// pre-dashboard launcher. `state.instances` (the dashboard's own render arm)
/// looks daemon-verified but isn't: `rocm-dash-daemon`'s registry discovery
/// (`runner.rs`) promotes a record from `Starting` to `Ready` on a successful
/// scrape, but a *failed* scrape only logs and leaves the prior status in
/// place, and nothing on the `rocm dash` path calls the CLI's own liveness
/// reconcile (`refresh_managed_service_runtime_liveness`, wired only into
/// `load_managed_services`) — so a service that crashed since its last
/// successful scrape still reads `is_serving() == true` here, the same gap
/// the launcher's raw on-disk read already has. Both are now the identical
/// hedge instead of two different claims over equally-unverified data.
///
/// Iterates `state.instances.values()` more than once (via the `running`
/// closure below, and again inside `nothing_is_currently_being_served_with`)
/// rather than collecting into a `Vec` first: nothing here needs random
/// access, and a `HashMap`'s `values()` is cheap to re-create — this still
/// allocates a fresh `Vec<String>` for the returned body every render while
/// the prompt is open (an unavoidable allocation, not one this avoids); the
/// point of the double iteration is only to skip building an intermediate
/// `Vec<&Instance>` neither closure needs.
pub(super) fn quit_confirm_body_with(
    state: &AppState,
    snapshot: &crate::registry_probe::ManagedRegistrySnapshot,
) -> Vec<String> {
    let running = || state.instances.values().filter(|i| i.status.is_serving());
    let total = running().count();
    let managed_ids = &snapshot.managed_ids;
    if nothing_is_currently_being_served_with(state, snapshot) {
        // A real snapshot is authoritative here (or there was never a
        // startup fallback to begin with) and it says nothing is serving.
        // Can change between opening the prompt and this render (e.g. the
        // model exits on its own while the prompt is up) — not dead code.
        // The "it keeps running" trailer below would be a direct, printed
        // lie in this state (reachable whether or not the startup fallback
        // was ever active), so return before appending it rather than
        // widening the gate below.
        return vec!["Nothing is being served anymore.".to_string()];
    }
    let mut body = Vec::new();
    if total == 0 {
        // `total == 0` here only because `nothing_is_currently_being_served_with`
        // returned `false` above: either the startup-race fallback (issue
        // #145) is active — the daemon's first snapshot hasn't landed yet —
        // or the connection never will succeed at all (Windows: the
        // dashboard-embedded daemon client's connect path is unix-only, so
        // `instances` stays empty for the whole session). `instances` can't
        // name anything in either case, but the live registry read
        // (`managed_ids`) needs no daemon snapshot — it is this host's own
        // `rocm serve` registry, read straight off disk — so it can still
        // list each id this host can actually stop and the real command for
        // it; only the pretty model *name* `instances` would otherwise
        // supply is unavailable here. So this stays honest about not knowing
        // whether the listed ids are *still* serving, instead of claiming to
        // know either way — the trailer below is hedged to match, rather
        // than asserting present-tense that it does keep running.
        //
        // `managed_ids` is never empty here: `nothing_is_currently_being_served_with`
        // already returned `false` above, and with `total == 0` that can only be
        // because `snapshot.has_live_service` is `true` — every `ManagedRegistryProbe`
        // in this tree (`LiveManagedRegistryProbe::snapshot`, `apps/rocm::dash`) derives
        // both fields from the same `discover_managed_services` pass
        // (`crates/rocm-dash-daemon/src/registry.rs`), which inserts an id into `seen`
        // (-> `managed_ids`) for every record it also counts into `svcs` (->
        // `has_live_service`), so a probe can never report "live" with no id to name.
        // Asserted rather than branched on: a probe that someday broke that pairing
        // should fail loudly here instead of silently falling through with no ids
        // to list, leaving the "stop it first" guidance below with nothing to name.
        debug_assert!(
            !managed_ids.is_empty(),
            "a live registry snapshot reported serving with no managed id to name"
        );
        let mut ids: Vec<&String> = managed_ids.iter().collect();
        ids.sort();
        for id in ids.iter().take(5) {
            body.push(format!("• {id} — stop it first:"));
            body.push(format!("  rocm services stop {id} --yes"));
        }
        if let Some(line) = format::overflow_line(managed_ids.len(), 5) {
            body.push(line);
        }
    } else {
        // Sorted by `container_id` — the same key `services_manager::service_rows`
        // sorts by — rather than left in whichever order this `HashMap`'s
        // `values()` happened to yield, so the listing is at least stable
        // across renders. `s` does not focus "the Nth instance named here":
        // `quit_confirm_open_services_key` always focuses the FIRST managed,
        // still-serving row in `service_rows`' own order, which can include
        // an unmanaged (Docker/Lemonade) entry this listing also shows ahead
        // of it — so the row `s` focuses need not be the Nth line printed
        // below.
        let mut sorted: Vec<_> = running().collect();
        sorted.sort_by(|a, b| a.container_id.cmp(&b.container_id));
        for i in sorted.into_iter().take(5) {
            // The stop command is only ever printed next to a service this
            // host's own registry can actually reach: `instances` can also
            // hold a Docker- or Lemonade-discovered entry from an external
            // daemon, which `rocm services stop` cannot touch — printing it
            // unconditionally would be remediation advice the user cannot
            // follow for that instance (AGENTS.md §3).
            if managed_ids.contains(&i.container_id) {
                // The command sits on its own line rather than trailing the
                // model name on one long line: at the popup's narrowest width
                // (the launcher's copy, @id:dash-launcher-quit-confirm-when-serving
                // on an 80-column terminal) a combined line word-wraps the
                // command itself across two rows, which the e2e assertion
                // matches against the full flattened screen — a wrap there
                // lands the command's second half next to unrelated menu text
                // from outside the popup instead of the first half. Splitting
                // keeps the SHORT, fixture-sized id used in that scenario
                // (`e2e-mock`) from wrapping; it does not stop a real id
                // (`generate_service_id`, `rocm_core::generate_service_id`,
                // produces roughly 23-47 characters depending on engine and
                // model slug) from wrapping on its own line at a narrow
                // width. A real user never has to
                // reassemble a remediation command broken across the model
                // name's line, at least.
                body.push(format!("• {} — stop it first:", i.model_name));
                body.push(format!("  rocm services stop {} --yes", i.container_id));
            } else {
                body.push(format!("• {}", i.model_name));
            }
        }
        if let Some(line) = format::overflow_line(total, 5) {
            body.push(line);
        }
    }
    body.push(String::new());
    body.push("It may still be running in the background after you quit.".to_string());
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::test_support::with_registry_probe;

    use crate::ui::draw;

    #[test]
    fn quit_confirm_body_prints_the_stop_command_during_the_startup_fallback_when_the_registry_names_an_id()
     {
        // On a host whose daemon client never sends an instance snapshot
        // (Windows — it connects and streams live GPU/CPU/memory every tick,
        // but has no `rocmdashd` socket to source an instance list from, so
        // `has_received_snapshot` never flips for the whole session),
        // `instances` never fills, so without this the quit prompt
        // could only ever print the generic "a model was serving" sentence —
        // no model name (expected: `instances` is the only source for that),
        // but also no stop command and no id, even though the live registry
        // read already has one. The registry read needs no daemon snapshot,
        // so the stop command should be available here regardless.
        //
        // This only pins the printed string (AGENTS.md §3) — it does not run
        // the command. `dash-28`
        // (`@id:dash-quit-prompt-stop-command-actually-works`) is the test
        // that proves a real `rocm services stop <id> --yes` actually clears
        // the condition.
        let mut state = AppState::new("t".into(), "default-dark".into());
        with_registry_probe(&mut state, true, &["svc-managed"]);
        assert!(state.instances.is_empty());
        assert!(!state.has_received_snapshot);

        let body = quit_confirm_body(&state);
        assert!(
            body.iter()
                .any(|line| line.contains("rocm services stop svc-managed --yes")),
            "must print the real stop command from the live registry read even with no \
             daemon snapshot: {body:?}"
        );
        // Checked by requiring the hedge phrase itself, not by requiring the
        // absence of the retired present-tense one: production never prints
        // that exact retired string once hedged, so the negative alone can't
        // fail.
        assert!(
            body.iter()
                .any(|line| line.contains("may still be running")),
            "must still hedge — the registry read can't confirm current liveness, only that \
             the record hasn't been reconciled as stopped: {body:?}"
        );
    }

    #[test]
    fn quit_confirm_body_says_nothing_is_being_served_once_a_snapshot_confirms_it() {
        // Once a real (even empty) snapshot has landed, the startup fallback
        // is retired (`AppState::has_live_instance`'s own doc comment) and
        // the plain "nothing anymore" line is accurate again. Same class of
        // self-contradiction as the sibling test above, reached via a
        // different trigger: a model that was serving stops on its own while
        // the prompt sits open awaiting a keypress (the event loop drains
        // daemon snapshots on its own `select!` arm independent of the
        // pending prompt) — the body must not print "It keeps running in the
        // background" right after "Nothing is being served anymore."
        //
        // The registry is seeded to still (falsely, or just not yet reaped)
        // report a live, stoppable managed id, to prove that a leftover
        // registry entry alone can't make the body claim anything is
        // running once a real empty snapshot has landed — the snapshot's
        // early return in `quit_confirm_body` fires before the registry is
        // ever consulted, so a "stop it first" assertion here would be
        // vacuously true regardless of what's seeded (see the sibling tests
        // for actual stop-line coverage: `quit_confirm_body_names_the_real_stop_command_for_a_managed_instance`
        // and `quit_confirm_body_omits_the_stop_command_for_an_unmanaged_instance`).
        let mut state = AppState::new("t".into(), "default-dark".into());
        with_registry_probe(&mut state, true, &["svc-managed"]);
        state.has_received_snapshot = true;

        let body = quit_confirm_body(&state);
        assert!(
            body.iter()
                .any(|line| line.contains("Nothing is being served"))
        );
        // The early return in `quit_confirm_body_with` skips the trailer
        // entirely, so no hedge should leak through either — not just the
        // retired present-tense phrase, which production never prints once
        // hedged and so can't tell a leaked trailer from no trailer at all.
        assert!(
            !body
                .iter()
                .any(|line| line.contains("may still be running")),
            "must not print any continued-running trailer once a snapshot confirms nothing is \
             serving: {body:?}"
        );
    }

    #[test]
    fn quit_confirm_request_title_matches_the_body_when_nothing_is_being_served() {
        // The title used to be hardcoded to "Still serving" even in the exact
        // states `quit_confirm_body` itself renders as "Nothing is being
        // served anymore." — a visible contradiction between the two halves
        // of the same prompt. Covers both triggers for that state (plain
        // "nothing ever was", and the snapshot-confirms-it-stopped case) so a
        // fix for only one can't quietly leave the other still titled wrong.
        let plain = AppState::new("t".into(), "default-dark".into());
        let req = quit_confirm_request(&plain, true);
        assert_eq!(
            req.title, "Nothing to confirm — quit?",
            "title must not claim serving when nothing is"
        );
        assert!(
            req.body
                .iter()
                .any(|l| l.contains("Nothing is being served"))
        );

        let mut snapshot_confirmed = AppState::new("t".into(), "default-dark".into());
        with_registry_probe(&mut snapshot_confirmed, true, &[]);
        snapshot_confirmed.has_received_snapshot = true;
        let req = quit_confirm_request(&snapshot_confirmed, true);
        assert_eq!(
            req.title, "Nothing to confirm — quit?",
            "title must not claim serving once a snapshot confirms it stopped"
        );
    }

    #[test]
    fn quit_confirm_request_title_and_trailer_are_hedged_once_a_real_snapshot_has_landed() {
        // `state.instances` looks daemon-verified once a real snapshot has
        // landed, but isn't — a failed scrape in
        // `rocm-dash-daemon`'s `runner.rs` only logs and leaves the prior
        // status in place, with nothing on this path calling the CLI's own
        // liveness reconcile (`refresh_managed_service_runtime_liveness`,
        // wired only into `load_managed_services`). A crashed-but-still-
        // `ready` record therefore still reads `is_serving() == true` here,
        // so both the title and the trailer must stay hedged for the
        // dashboard's own render arm — not just for the pre-dashboard
        // launcher, which already got this hedge before this round.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.instances.insert(
            "svc-managed".to_string(),
            Instance {
                container_id: "svc-managed".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        state.has_received_snapshot = true;
        with_registry_probe(&mut state, true, &["svc-managed"]);

        let dashboard = quit_confirm_request(&state, true);
        assert_eq!(
            dashboard.title, "May still be serving — quit anyway?",
            "the dashboard's own title must hedge the same way the launcher's does: {:?}",
            dashboard.title
        );
        // Checked by requiring the hedge phrase itself: production never
        // prints the retired present-tense wording once hedged, so a bare
        // `!contains` against it can't fail.
        assert!(
            dashboard
                .body
                .iter()
                .any(|line| line.contains("may still be running")),
            "must not assert present-tense liveness the daemon never actually verified — the \
             trailer must hedge the same way the launcher's copy does: {:?}",
            dashboard.body
        );

        // The launcher's copy (`offer_stop_action: false`) must match: same
        // title, same hedge, derived from the exact same shared function.
        let launcher = quit_confirm_request(&state, false);
        assert_eq!(launcher.title, dashboard.title);
    }

    #[test]
    fn quit_confirm_request_offers_the_stop_hint_only_for_the_dashboard_with_something_stoppable() {
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.instances.insert(
            "svc-managed".to_string(),
            Instance {
                container_id: "svc-managed".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        with_registry_probe(&mut state, true, &["svc-managed"]);

        let dashboard = quit_confirm_request(&state, true);
        assert!(
            dashboard
                .body
                .iter()
                .any(|l| l.contains("Press s to open Services")),
            "the dashboard copy must offer the hint it can act on: {:?}",
            dashboard.body
        );

        let launcher = quit_confirm_request(&state, false);
        assert!(
            !launcher.body.iter().any(|l| l.contains("Press s")),
            "the launcher copy must not advertise a key it cannot honor: {:?}",
            launcher.body
        );

        // No managed instance serving (only an unmanaged/external one): the
        // dashboard copy must not offer the hint either, since `s` would have
        // nothing to focus.
        let mut unmanaged_only = AppState::new("t".into(), "default-dark".into());
        unmanaged_only.instances.insert(
            "docker-abc".to_string(),
            Instance {
                container_id: "docker-abc".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        let dashboard_unmanaged = quit_confirm_request(&unmanaged_only, true);
        assert!(
            !dashboard_unmanaged
                .body
                .iter()
                .any(|l| l.contains("Press s")),
            "must not offer to stop something `s` can't reach: {:?}",
            dashboard_unmanaged.body
        );
    }

    #[test]
    fn quit_confirm_body_names_the_real_stop_command_for_a_managed_instance() {
        // The prompt's remediation must name the id `rocm services stop`
        // actually needs, not a literal `<id>` placeholder, and must only
        // offer it for a service this host's own registry manages.
        //
        // This only pins the printed string (AGENTS.md §3) — a string match
        // cannot show the command "works". `dash-28`
        // (`@id:dash-quit-prompt-stop-command-actually-works`) is the test
        // that runs it for real and proves it does.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.instances.insert(
            "svc-managed".to_string(),
            Instance {
                container_id: "svc-managed".to_string(),
                model_name: "Qwen2.5-7B".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        with_registry_probe(&mut state, true, &["svc-managed"]);

        let body = quit_confirm_body(&state);
        assert!(
            body.iter()
                .any(|line| line == "• Qwen2.5-7B — stop it first:"),
            "expected the managed-instance bullet to name the serving model: {body:?}"
        );
        assert!(
            body.iter()
                .any(|line| line.contains("rocm services stop svc-managed --yes")),
            "expected a fillable stop command naming the real id: {body:?}"
        );
    }

    #[test]
    fn quit_confirm_body_omits_the_stop_command_for_an_unmanaged_instance() {
        // `instances` can hold a Docker- or Lemonade-discovered entry from an
        // external daemon when connected to one; `rocm services stop` cannot
        // reach those, so the prompt must not suggest it can.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("t".into(), "default-dark".into());
        state.instances.insert(
            "docker-abc123".to_string(),
            Instance {
                container_id: "docker-abc123".to_string(),
                model_name: "external-model".to_string(),
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        assert!(state.live_registry_snapshot().managed_ids.is_empty());

        let body = quit_confirm_body(&state);
        assert!(
            body.iter().any(|line| line.contains("external-model")),
            "the model must still be named: {body:?}"
        );
        assert!(
            !body.iter().any(|line| line.contains("rocm services stop")),
            "must not offer a stop command this host cannot honor: {body:?}"
        );
    }

    #[test]
    fn quit_confirm_body_caps_the_listing_at_five_with_an_overflow_line() {
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("t".into(), "default-dark".into());
        for n in 0..6 {
            state.instances.insert(
                format!("svc-{n}"),
                Instance {
                    container_id: format!("svc-{n}"),
                    model_name: format!("model-{n}"),
                    status: InstanceStatus::Running,
                    ..Default::default()
                },
            );
        }

        let body = quit_confirm_body(&state);
        let bullets = body.iter().filter(|line| line.starts_with('•')).count();
        assert_eq!(bullets, 5, "the listing must cap at 5 bullets: {body:?}");
        assert!(
            body.iter().any(|line| line.contains("…and 1 more")),
            "the sixth instance must be summarized by the overflow line: {body:?}"
        );
    }

    #[test]
    fn quit_confirm_body_lists_services_sorted_by_container_id() {
        // The body sorts by `container_id` — the same key
        // `services_manager::service_rows` sorts by — so the listing is
        // stable rather than left in whichever order a `HashMap`'s
        // `values()` happens to yield. This does not mean `s`
        // (`AppState::quit_confirm_open_services_key`) focuses the Nth
        // bullet printed here: it always focuses the FIRST managed,
        // still-serving row in `service_rows`' own order, which can include
        // an unmanaged entry this listing also shows.
        // `quit_confirm_body_caps_the_listing_at_five_with_an_overflow_line`
        // inserts ids (`svc-0`..`svc-5`) that already sort the same way
        // whether by id or insertion order, so it can't catch a regression
        // here — these ids are deliberately inserted out of sorted order.
        //
        // Eight ids, not three: with only 3 ids a `HashMap`'s random
        // iteration order matches the sorted order by chance often enough to
        // make this test flaky at catching its own regression — commenting
        // out the `sort_by` this asserts and re-running 9 times passed 4 and
        // failed 5. With 8 ids there are 8! = 40320 orderings, so a random
        // iteration order matching the fully-sorted one by chance is
        // negligible; a missing/wrong sort now fails reliably.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("t".into(), "default-dark".into());
        let ids = [
            "svc-golf",
            "svc-charlie",
            "svc-hotel",
            "svc-alpha",
            "svc-foxtrot",
            "svc-delta",
            "svc-bravo",
            "svc-echo",
        ];
        for id in ids {
            state.instances.insert(
                id.to_string(),
                Instance {
                    container_id: id.to_string(),
                    model_name: format!("model-{id}"),
                    status: InstanceStatus::Running,
                    ..Default::default()
                },
            );
        }

        let body = quit_confirm_body(&state);
        let bullets: Vec<&str> = body
            .iter()
            .filter(|line| line.starts_with('•'))
            .map(std::string::String::as_str)
            .collect();
        assert_eq!(
            bullets,
            vec![
                "• model-svc-alpha",
                "• model-svc-bravo",
                "• model-svc-charlie",
                "• model-svc-delta",
                "• model-svc-echo",
            ],
            "bullets must be sorted by container_id, not insertion/HashMap order \
             (capped at 5, so this also exercises the sort ahead of the cap): {body:?}"
        );
    }

    #[test]
    fn quit_confirm_popup_with_realistic_ids_does_not_panic_at_a_narrow_width() {
        // `generate_service_id` (`rocm_core::generate_service_id`) produces
        // `<engine>-<model-slug>-<unix-millis>` ids, roughly 23-47 characters,
        // not the short `e2e-mock`/`svc-managed` fixtures used elsewhere in
        // this file — a doc comment on the stop-command split used to claim
        // the split "keeps the command short enough to never need to wrap",
        // which is false for a real id at a narrow popup width. This doesn't
        // assert a specific layout (ratatui wraps however it wraps; asserting
        // exact column breaks here would just pin that behavior without
        // proving anything about it, and the clipping this can produce at 3+
        // services is a known, separately-tracked non-blocking gap, not what
        // this test is for) — it pins that several realistic-length,
        // realistic-*format* managed ids at the popup's narrowest supported
        // width render without panicking and still contain every id's text
        // somewhere in the frame.
        //
        // The fixture's `model_name` deliberately does NOT embed the id
        // (unlike an earlier version of this test): with a model name like
        // "model-<id>", `painted.contains(id)` could pass off that line
        // alone even if the "stop it first" bullet and its
        // `rocm services stop <id> --yes` line — the only other place the raw
        // id appears — were dropped entirely. With a plain name instead, the
        // id can only appear via the stop-command line, so this is no longer
        // vacuous.
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use rocm_dash_core::metrics::{Instance, InstanceStatus};

        let realistic_ids = [
            "vllm-llama-3-1-8b-instruct-1760345671001",
            "lemonade-qwen2-5-7b-instruct-1760345672002",
            "vllm-mixtral-8x7b-instruct-1760345673003",
        ];
        let model_names = ["Model A", "Model B", "Model C"];
        let mut state = AppState::new("t".into(), "default-dark".into());
        for (id, name) in realistic_ids.into_iter().zip(model_names) {
            state.instances.insert(
                id.to_string(),
                Instance {
                    container_id: id.to_string(),
                    model_name: name.to_string(),
                    status: InstanceStatus::Running,
                    ..Default::default()
                },
            );
        }
        with_registry_probe(&mut state, true, &realistic_ids);
        state.open_quit_confirm();

        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw(f, &mut state)).unwrap();
        let painted: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(
            painted.contains("rocm services stop"),
            "at least one stop command must render: {painted:?}"
        );
        for id in realistic_ids {
            assert!(
                painted.contains(id),
                "realistic id {id:?} must still appear somewhere in the \
                 rendered frame even if its line wraps — not vacuously \
                 satisfied by a model name that happens to embed it: {painted:?}"
            );
        }
    }
}
