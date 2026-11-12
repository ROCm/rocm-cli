// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The locked launch transaction around a managed service: the port allocation
//! decision against live records and the OS, the cleanup of a failed launch
//! attempt, and the reuse refusals (including the no-usable-GPU wording) that
//! decide whether a live service may answer a `serve` request.

use crate::endpoint_keys;
use crate::managed_service_status::{
    existing_live_managed_service_in, load_managed_services, managed_service_is_live,
};
use crate::serve_port;
use crate::{device_policy_name, record_cli_audit_event};
use anyhow::Result;
use rocm_core::{AppPaths, ManagedServiceRecord};
use rocm_engine_protocol::DevicePolicy;
use std::fs;
use std::time::Duration;

/// Outcome of the locked allocation step.
///
/// `Debug` so tests can assert on `Result::expect_err`.
#[derive(Debug)]
pub(super) enum ManagedPortDecision {
    /// An equivalent managed service is already live; nothing may be spawned.
    /// Boxed to keep the variants a similar size (clippy::large_enum_variant).
    AlreadyLive(Box<ManagedServiceRecord>),
    /// The concrete port this launch owns until it hands it to the child.
    Lease(rocm_core::LoopbackPortLease),
}

/// Live managed services, with liveness refreshed by [`load_managed_services`]
/// (dead PIDs demote to "stopped" and therefore neither satisfy the duplicate
/// check nor reserve a port).
///
/// A read failure degrades to "nothing is live": for an already-`ready`
/// service the OS preflight still guards its port (the engine holds it), and
/// refusing to serve because the services directory could not be listed would
/// be a worse failure than launching. That guard does not cover a `starting`
/// record: its port is released back to the OS in
/// `launch_managed_engine_child_locked` (see `lease.release()`), right before the
/// spawn, and stays unbound until the new engine child has actually bound it, so a concurrent launch that hits this
/// read failure in that narrow window can probe the same port successfully.
/// Accepted as a known gap — narrower than the general services-dir read
/// failure case, and only reachable by a second, near-simultaneous I/O
/// failure on top of one already in flight.
pub(super) fn live_managed_services(paths: &AppPaths) -> Vec<ManagedServiceRecord> {
    load_managed_services(paths)
        .unwrap_or_default()
        .into_iter()
        .filter(managed_service_is_live)
        .collect()
}

/// Ports that live managed services still own.
///
/// Conservative by design: on the default loopback host a record reserves its
/// port regardless of the host it recorded, because a `0.0.0.0` (or `localhost`)
/// listener also blocks the loopback address a new local server would bind.
/// Over-reserving costs one candidate in a 101-port scan; under-reserving costs
/// a collision. On a custom host the recorded host is compared (see
/// `serve_port::hosts_overlap`).
fn reserved_service_ports(live: &[ManagedServiceRecord]) -> Vec<serve_port::ReservedPort> {
    live.iter()
        .map(|record| serve_port::ReservedPort {
            port: record.port,
            host: record.host.clone(),
            service_id: record.service_id.clone(),
            status: record.status.clone(),
        })
        .collect()
}

/// The decision half of the launch transaction: refresh liveness, apply the
/// `(engine, canonical_model_id)` idempotency guard, then lease a concrete port
/// against both live records and the real OS.
///
/// Callers MUST hold [`rocm_core::AppPaths::managed_launch_lock_path`] across
/// this call *and* the record publication that follows, or two launches can
/// pick the same candidate.
/// `probe` is injected so tests can drive deterministic bind outcomes without a
/// GPU or a spawn.
pub(super) fn resolve_managed_port_in_transaction(
    paths: &AppPaths,
    engine: &str,
    canonical_model_id: &str,
    host: &str,
    request: serve_port::PortRequest,
    probe: &mut dyn FnMut(u16) -> std::io::Result<rocm_core::LoopbackPortLease>,
) -> Result<ManagedPortDecision> {
    let live = live_managed_services(paths);
    // Idempotency first: an already-live equivalent service means no port is
    // needed at all, so it must be answered before any candidate is leased.
    if let Some(existing) = existing_live_managed_service_in(&live, engine, canonical_model_id) {
        return Ok(ManagedPortDecision::AlreadyLive(Box::new(existing.clone())));
    }
    let reserved = reserved_service_ports(&live);
    serve_port::resolve_serve_port(host, request, &reserved, probe).map(ManagedPortDecision::Lease)
}

/// Grace given to a child that must be terminated because its record could not
/// be published. Short: the child is milliseconds old and has nothing to flush.
const MANAGED_LAUNCH_CLEANUP_GRACE: Duration = Duration::from_secs(2);

/// Whether this host records a start token for a process: Linux (the kernel
/// start-time) and Windows (the creation time). Anywhere else
/// `rocm_core::process_start_ticks` has nothing to read.
const fn host_records_start_tokens() -> bool {
    cfg!(any(target_os = "linux", windows))
}

/// The verdict that ends a failed launch's cleanup before anything is signalled:
/// `Some(Unverified)` for a child whose token was not captured on a host that
/// records tokens, because that child was gone before it could be captured and
/// its pid may now belong to another process. `None` otherwise: a captured token
/// is verified by the termination, and on a host with no tokens there is nothing
/// to verify, so the kill proceeds as it always did there.
const fn launch_cleanup_refusal(
    engine_start_ticks: Option<u64>,
    host_records_tokens: bool,
) -> Option<rocm_core::TerminationOutcome> {
    if engine_start_ticks.is_none() && host_records_tokens {
        Some(rocm_core::TerminationOutcome::Unverified)
    } else {
        None
    }
}

/// Undo a failed managed launch attempt before the launch lock is released,
/// so no failed transaction leaves a live reservation or a stray endpoint key.
///
/// Order matters: a spawned child is terminated *first*, so it is not still
/// holding the port or writing to the log by the time cleanup runs. The
/// termination outcome is audited but not acted on: the record is marked
/// abandoned (`stopped`, releasing the port) even when death was not confirmed. If no child was ever spawned (`record.engine_pid` is still `None`),
/// every artifact this attempt created is removed: there is nothing for
/// `rocm services`/`rocm logs` to show. If a child did run, its record and log
/// are kept instead (marked `status = "stopped"`, see
/// [`ManagedServiceRecord::mark_launch_abandoned`]), since the error that led here
/// already told the user to go inspect that log; only the endpoint key, which a
/// failed attempt has no use for, is cleared.
/// Every step is best-effort and idempotent; cleanup must never mask the
/// original failure with a second one.
///
/// `child_known_dead` is set when the startup check saw the child exit
/// (`EngineDiedAtStartup` in `spawn`). Such a child is not signalled at all: it is
/// gone, and its pid may already belong to an unrelated process. No start-time
/// token was captured for it before it died, so nothing else would stop the kill
/// from landing on that process. Its record and log are still kept, since the
/// error names that log.
///
/// The `stopped` mark is a guarded write ([`update_managed_record_on_disk`]):
/// a stop that settled the record, a restart that took it over, or a removal
/// is left as it is rather than overwritten or re-created.
pub(super) fn abandon_failed_managed_launch(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    child_known_dead: bool,
) {
    // The shared routine `rocm services stop` and `rocmd` use, not
    // `terminate_verified` alone: on Windows only the routine takes the
    // processes under the child down, so a failed launch cannot leave the engine
    // one level below the supervisor running.
    abandon_failed_managed_launch_with(paths, record, child_known_dead, |identity| {
        rocm_core::terminate_recorded_pid_within(identity, MANAGED_LAUNCH_CLEANUP_GRACE)
    });
}

/// [`abandon_failed_managed_launch`] with the termination of the child supplied
/// by the caller, so a test sees what the cleanup asks it to terminate, and
/// whether it asks at all.
fn abandon_failed_managed_launch_with(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    child_known_dead: bool,
    terminate: impl Fn(&rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome,
) {
    let spawned_pid = record.engine_pid.filter(|pid| *pid != 0);
    if child_known_dead || spawned_pid.is_some() {
        if let Some(pid) = spawned_pid.filter(|_| !child_known_dead) {
            terminate_abandoned_launch_child(paths, record, pid, terminate);
        }
        let _ = update_managed_record_on_disk(
            paths,
            &record.service_id,
            |current| launch_still_owns(current, record),
            ManagedServiceRecord::mark_launch_abandoned,
        );
        endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
        return;
    }
    for path in [
        record.manifest_path.as_path(),
        record.log_path.as_path(),
        record.engine_state_path.as_path(),
    ] {
        let _ = fs::remove_file(path);
    }
    endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
}

/// Terminate the child of an abandoned launch whose state is unknown, and audit
/// the outcome.
fn terminate_abandoned_launch_child(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    pid: u32,
    terminate: impl Fn(&rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome,
) {
    // The token captured at spawn, not a fresh `capture`: when the child
    // already exited, a fresh one would describe a recycled pid's new owner
    // and the forced tree kill would hit it. Where start tokens exist
    // (Linux, and Windows, which records the process creation time), a
    // missing one means the child was gone before it could be captured, so
    // the pid is unverifiable and nothing is signalled (`terminate_verified`
    // alone would "best-effort proceed"). A token that was captured is
    // verified by the termination on both.
    //
    // The tradeoff on Windows, chosen deliberately: a child whose creation
    // time could not be read at capture is not killed, so if `OpenProcess`
    // failed for a child that is in fact alive (it is this account's own
    // fresh child, so that is not expected), a failed launch leaves its
    // engine running and the audit event says `unverified`. The alternative
    // is a forced tree kill of a pid that may by now belong to another
    // process, which cannot be undone. The spawn primitive returns only the
    // pid and closes its handle, so a handle-based identity would mean
    // changing it; that is not done here. The refusal decision
    // (`launch_cleanup_refusal`) and what the cleanup asks the termination to
    // do are under tests that run on Windows; the cleanup of a real Windows
    // child is run by no test.
    let outcome = if let Some(refusal) =
        launch_cleanup_refusal(record.engine_start_ticks, host_records_start_tokens())
    {
        refusal
    } else {
        terminate(&rocm_core::ProcessIdentity::new(
            pid,
            record.engine_start_ticks,
        ))
    };
    record_cli_audit_event(
        paths,
        "service",
        "managed_service_launch_cleanup",
        "warn",
        format!(
            "terminated engine child pid={pid} after a failed launch publication outcome={}",
            outcome.as_str()
        ),
        Some(&record.service_id),
    );
}

/// Apply one change to a managed service's record as it is on disk now, rather
/// than writing back a copy the caller holds.
///
/// `ManagedServiceRecord::write` replaces the whole file, and a stop records
/// `stop_requested_unix_ms` and then `"stopped"` without taking the launch lock.
/// A copy read before the stop and written back after it would erase both,
/// bringing back as live a service the stop just reported stopped, and with it
/// the marker that keeps `rocmd`'s recovery off it.
///
/// Returns `Ok(None)`, writing nothing, when the record is gone or
/// `still_owned` refuses the record as it is now: whoever wrote since holds a
/// truer view. This narrows the lost-update window; it does not close it, since
/// the read and the write are separate steps and a stop takes no lock on the
/// manifest.
pub(super) fn update_managed_record_on_disk(
    paths: &AppPaths,
    service_id: &str,
    still_owned: impl FnOnce(&ManagedServiceRecord) -> bool,
    update: impl FnOnce(&mut ManagedServiceRecord),
) -> Result<Option<ManagedServiceRecord>> {
    if !paths.service_manifest_path(service_id).exists() {
        return Ok(None);
    }
    let mut current = ManagedServiceRecord::read_manifest(paths, service_id)?;
    if !still_owned(&current) {
        return Ok(None);
    }
    update(&mut current);
    current.write()?;
    Ok(Some(current))
}

/// Whether the launch that wrote `launched` may still write `current`, the same
/// service's record as it is on disk now: it still reads as live, and names the
/// supervisor the launch last recorded — 0 for the claim, before any child is
/// on record. A confirmed stop writes `"stopped"`; a restart records its own
/// supervisor. Either way the record is no longer the launch's to write.
pub(super) fn launch_still_owns(
    current: &ManagedServiceRecord,
    launched: &ManagedServiceRecord,
) -> bool {
    crate::managed_service_status::managed_service_is_live(current)
        && current.supervisor_pid == launched.supervisor_pid
        && current.supervisor_start_ticks == launched.supervisor_start_ticks
}

/// The ways a live managed service that already matches on liveness, engine and
/// canonical model id can still be unusable by this invocation.
///
/// Those three terms are necessary but not sufficient: three further terms can
/// still refuse, one variant each. [`managed_service_reuse_refusal`] is the
/// single decision procedure both callers ask, so the two cannot answer
/// differently — which they previously did; see that function for what the
/// disagreement cost.
pub(crate) enum ManagedReuseRefusal {
    /// The live service was started with different serve options.
    RecipeMismatch,
    /// `--require-api-key` was asked for and the live service has no key.
    ///
    /// The engine reads its key once, at launch, from the environment the spawn
    /// builds — so a server started without one keeps serving anonymously no
    /// matter what is written afterwards. Upgrading the record would be worse
    /// than doing nothing: it would claim auth the live process does not
    /// enforce, and `ensure_public_service_has_endpoint_key` would pass on the
    /// strength of a key file nothing reads. `rocm remote serve` relies on this:
    /// it publishes a loopback port onto the tailnet and prints "the API key
    /// above is what stops anyone else calling it", which reusing an
    /// unauthenticated service silently would make false about an endpoint the
    /// whole tailnet can reach. Refusing is the only answer that fails closed.
    MissingApiKey,
    /// The live service runs under a device policy that does not satisfy this
    /// invocation's. Reusing a `cpu_only` service to answer a `gpu_required`
    /// serve is the silent CPU fallback AGENTS.md section 6 forbids.
    DevicePolicyTooWeak,
}

/// Why this invocation cannot reuse `existing` — or `None` when it genuinely
/// can. See [`ManagedReuseRefusal`] for the terms this answers on.
///
/// This exists because `serve()`'s reuse pre-gate lets a matching live service
/// skip the no-usable-AMD-GPU refusal. Deciding that bypass on a *weaker* test
/// than the spawn then uses to accept reuse produced a dead end: on a GPU-less
/// host (`HIP_VISIBLE_DEVICES=` with a server live from another shell) a serve
/// with different options bypassed the refusal, reached the spawn, and was told
/// to stop the server and run the command again — and the rerun, with nothing
/// left to reuse, hit the no-GPU refusal. The advice named a command that could
/// not clear the condition, which AGENTS.md section 3 forbids. Sharing the
/// predicate means a service that cannot actually be reused never buys the
/// bypass, so the user gets the accurate refusal first and keeps their server.
///
/// `requested_device_policy` is taken by value, not as an `Option`, on purpose.
/// It was optional so the spawn could opt out of the device-policy term, and
/// that left two holes at once: the spawn's `DevicePolicyTooWeak` arm was
/// unreachable (an untestable bail), and `serve()` — the caller that *does*
/// gate the GPU refusal on this answer — could silently drop the term by
/// passing `None`, with the whole suite still green. Both callers have a
/// `DevicePolicy` in hand, so requiring it costs nothing and makes that
/// mutation a compile error. It widens nothing observable today either: the
/// only policy a record can carry is the one `parse_device_policy` admits
/// (see below), so `existing.device_policy == Some("cpu_only")` is not
/// reachable from any `rocm serve` this CLI can run.
/// The `rocm serve` no-usable-GPU refusal, worded once.
///
/// Two places raise it and they must say the same thing: `serve()`'s fail-fast
/// pre-flight, and [`spawn_managed_engine_child`](super::spawn::spawn_managed_engine_child)
/// when the reuse that waived that pre-flight turns out not to be available after
/// all.
pub(crate) fn no_usable_gpu_refusal(device_policy: &DevicePolicy) -> String {
    // Keyed on the same `is_wsl_host()` the usable-GPU probe branches on, so the
    // advice follows the platform whose answer produced the refusal.
    no_usable_gpu_refusal_text(device_policy_name(device_policy), rocm_core::is_wsl_host())
}

/// The refusal text for `policy`, split on whether the probe that refused was
/// the WSL one.
///
/// WSL reaches the GPU through `/dev/dxg` and ROCDXG, and has no `/dev/kfd`, so
/// telling a WSL user to confirm that device is present names something that can
/// never clear the condition. There the advice points at the status `rocm
/// examine` reports for the path the probe actually asked about
/// ([`rocm_core::WSL_ROCDXG_READY`]), and at `rocm diagnose`, which names what is
/// missing.
fn no_usable_gpu_refusal_text(policy: &str, on_wsl: bool) -> String {
    let remedy = if on_wsl {
        format!(
            "WSL reaches the GPU through /dev/dxg and ROCDXG rather than /dev/kfd: run \
             `rocm examine` and check that `driver_status` is `{ready}` (`rocm diagnose` \
             names what is missing; a ROCDXG library the loader reaches only through \
             LD_LIBRARY_PATH needs that variable exported in this shell, see docs/wsl.md), and ensure HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES \
             are not masking every device.",
            ready = rocm_core::WSL_ROCDXG_READY
        )
    } else {
        "Check the driver with `rocm examine`, confirm /dev/kfd is present, and ensure \
         HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES are not masking every device."
            .to_owned()
    };
    format!(
        "no usable AMD GPU detected; `rocm serve` requires a GPU under the {policy} policy \
         and does not fall back to CPU. {remedy}"
    )
}

pub(crate) fn managed_service_reuse_refusal(
    existing: &ManagedServiceRecord,
    requested_recipe_json: Option<&str>,
    require_api_key: bool,
    requested_device_policy: &DevicePolicy,
) -> Option<ManagedReuseRefusal> {
    if existing.engine_recipe_json.as_deref() != requested_recipe_json {
        return Some(ManagedReuseRefusal::RecipeMismatch);
    }
    if require_api_key && !existing.requires_api_key {
        return Some(ManagedReuseRefusal::MissingApiKey);
    }
    // Only `gpu_required` can be let down by the record's policy, and the
    // enumeration is exhaustive rather than narrow: `parse_device_policy`
    // (`main.rs`) maps `auto`, `gpu`, `gpu_required` *and* `gpu_preferred` onto
    // `GpuRequired`, and rejects `cpu_only` with an error, so `GpuRequired` is
    // the only policy `serve()` can hand this predicate — pinned by
    // `parse_device_policy_defaults_to_gpu_required_without_cpu_fallback`. There
    // is deliberately no wider match on "anything but `CpuOnly`": it would be
    // identical at runtime and would read as a guard against a `gpu_preferred`
    // bypass that no CLI invocation can reach. A `cpu_only` invocation would be
    // satisfied by anything anyway and never reaches the bail. README.md's
    // "that is the only policy pair refused" is the contract, and this matches
    // it exactly.
    if matches!(requested_device_policy, DevicePolicy::GpuRequired)
        && existing.device_policy.as_deref() == Some("cpu_only")
    {
        return Some(ManagedReuseRefusal::DevicePolicyTooWeak);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_hints;
    use crate::service_launch::test_support::sleeper_test_name;
    use crate::service_launch::test_support::{reusable_record, write_live_service_record};
    use crate::tests::test_paths;
    use rocm_core::DEFAULT_LOCAL_HOST;
    #[cfg(not(windows))]
    use std::process::Command as ProcessCommand;
    use std::thread;

    #[test]
    fn the_no_gpu_refusal_sends_a_bare_metal_user_to_dev_kfd() {
        let text = super::no_usable_gpu_refusal_text("gpu_required", false);
        assert!(text.contains("confirm /dev/kfd is present"), "{text}");
        assert!(!text.contains("ROCDXG"), "{text}");
    }

    #[test]
    fn the_no_gpu_refusal_does_not_send_a_wsl_user_to_dev_kfd() {
        // WSL has no /dev/kfd, so advice to confirm it can never clear the
        // condition. The text names the status `examine` reports for the path
        // the probe asked about, taken from the constant the producer uses, so
        // the two cannot drift. That the status really is the one a refused
        // host shows is asserted end to end by serve-42.
        let text = super::no_usable_gpu_refusal_text("gpu_required", true);
        assert!(!text.contains("/dev/kfd is present"), "{text}");
        assert!(text.contains("rocm examine"), "{text}");
        assert!(text.contains("rocm diagnose"), "{text}");
        // The advice is true only because `rocdxg_ready` counts a library on
        // LD_LIBRARY_PATH; `a_rocdxg_only_on_the_loader_path_is_ready` in
        // `rocm-core`'s host_gpu/wsl.rs proves that behavior.
        assert!(text.contains("LD_LIBRARY_PATH"), "{text}");
        assert!(
            text.contains(&format!("`{}`", rocm_core::WSL_ROCDXG_READY)),
            "{text}"
        );
    }

    #[test]
    fn the_no_gpu_refusal_names_the_policy_on_both_platforms() {
        for on_wsl in [false, true] {
            let text = super::no_usable_gpu_refusal_text("gpu_required", on_wsl);
            assert!(
                text.starts_with("no usable AMD GPU detected; `rocm serve` requires a GPU under the gpu_required policy"),
                "{text}"
            );
        }
    }

    #[test]
    fn reuse_is_refused_on_every_term_the_spawn_and_the_gpu_bail_depend_on() {
        let (root, paths) = test_paths("managed-reuse-predicate");
        paths.ensure().expect("paths");
        let base = reusable_record(&paths);

        // The baseline must be reusable, or the negative cases below could pass
        // for the wrong reason.
        assert!(
            managed_service_reuse_refusal(&base, None, true, &DevicePolicy::GpuRequired).is_none(),
            "a matching service with no recipe, a key, and a gpu_required policy is reusable"
        );

        // Recipe mismatch — the spawn's first refusal.
        let mut mismatched = base.clone();
        mismatched.engine_recipe_json = Some("{\"engine\":\"vllm\"}".to_owned());
        assert!(matches!(
            managed_service_reuse_refusal(&mismatched, None, true, &DevicePolicy::GpuRequired),
            Some(ManagedReuseRefusal::RecipeMismatch)
        ));

        // Unmet `--require-api-key` — the spawn's second refusal. Also pins that
        // it is *only* refused when the key is actually demanded.
        let mut anonymous = base.clone();
        anonymous.requires_api_key = false;
        assert!(matches!(
            managed_service_reuse_refusal(&anonymous, None, true, &DevicePolicy::GpuRequired),
            Some(ManagedReuseRefusal::MissingApiKey)
        ));
        assert!(
            managed_service_reuse_refusal(&anonymous, None, false, &DevicePolicy::GpuRequired)
                .is_none(),
            "an anonymous service is fine when no key was demanded"
        );

        // Device policy — the term `serve()` depends on, because it is the one
        // gating the no-usable-GPU bail on the answer. Reusing a `cpu_only`
        // service to answer a `gpu_required` serve is the silent CPU fallback.
        let mut cpu_only = base;
        cpu_only.device_policy = Some("cpu_only".to_owned());
        assert!(matches!(
            managed_service_reuse_refusal(&cpu_only, None, true, &DevicePolicy::GpuRequired),
            Some(ManagedReuseRefusal::DevicePolicyTooWeak)
        ));
        // `GpuRequired` is the only policy the CLI can produce — `--device
        // gpu_preferred`/`auto`/`gpu` all parse to it and `cpu_only` is rejected
        // (`parse_device_policy_defaults_to_gpu_required_without_cpu_fallback`) —
        // so there is no second policy arm to discriminate here, and asserting
        // one would pin an input `serve()` never produces. Both callers pass
        // the same policy; the parameter is not optional, so neither can drop
        // this term without a compile error.
        //
        // A cpu_only *invocation* never reaches the bail, so nothing to refuse.
        assert!(
            managed_service_reuse_refusal(&cpu_only, None, true, &DevicePolicy::CpuOnly).is_none()
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn duplicate_managed_launch_detected_across_distinct_service_ids() -> Result<()> {
        // `generate_service_id` embeds a timestamp, so a second launch for the
        // same engine+model has a DIFFERENT service_id. The guard must still
        // detect the live service by (engine, canonical_model_id), and return
        // the newest live match. `starting` skips the endpoint probe; the
        // current process id is a guaranteed-live PID.
        let (root, paths) = test_paths("dup-managed-distinct-ids");
        paths.ensure()?;

        // Older, dead manifest for the same engine+model (distinct service_id).
        let mut dead = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-1000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11500,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        dead.status = "ready".to_owned();
        dead.engine_pid = Some(999_999_999);
        dead.created_at_unix_ms = 1000;
        dead.write()?;

        // Newer, live manifest for the same engine+model (distinct service_id).
        let mut live = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-2000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11501,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        live.status = "starting".to_owned();
        live.engine_pid = Some(std::process::id());
        live.created_at_unix_ms = 2000;
        live.write()?;

        let live_services = live_managed_services(&paths);
        let found =
            existing_live_managed_service_in(&live_services, "lemonade", "qwen-canonical").cloned();
        let _ = fs::remove_dir_all(root);

        let found = found.expect("a live managed service should be detected by engine+model");
        assert_eq!(
            found.service_id, "lemonade-qwen-2000",
            "should return the newest live match"
        );
        assert!(managed_service_is_live(&found));
        Ok(())
    }

    #[test]
    fn dead_managed_service_allows_relaunch() -> Result<()> {
        // A stale manifest with dead PIDs must NOT block a relaunch: liveness
        // refresh demotes it to "stopped", so the guard returns None.
        let (root, paths) = test_paths("dup-managed-dead");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-3000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11502,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.engine_pid = Some(999_999_999);
        record.write()?;

        let live = live_managed_services(&paths);
        let found = existing_live_managed_service_in(&live, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            found.is_none(),
            "a dead managed service must not block relaunch"
        );
        Ok(())
    }

    #[test]
    fn live_service_for_other_model_does_not_block() -> Result<()> {
        // A live service for a DIFFERENT model must not match — the guard keys
        // on the model, not just the engine.
        let (root, paths) = test_paths("dup-managed-other-model");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-other-1",
            "lemonade",
            "other",
            "other-canonical",
            "127.0.0.1",
            11503,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.engine_pid = Some(std::process::id());
        record.write()?;

        let live = live_managed_services(&paths);
        let found = existing_live_managed_service_in(&live, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            found.is_none(),
            "a live service for a different model must not match"
        );
        Ok(())
    }

    #[test]
    fn missing_manifest_allows_launch() {
        // No services dir / manifests → nothing to detect, launch proceeds.
        let (root, paths) = test_paths("dup-managed-missing");
        let live = live_managed_services(&paths);
        let found = existing_live_managed_service_in(&live, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        assert!(found.is_none());
    }

    fn never_probe(_port: u16) -> std::io::Result<rocm_core::LoopbackPortLease> {
        panic!("no port may be probed on this path")
    }

    fn always_free(port: u16) -> std::io::Result<rocm_core::LoopbackPortLease> {
        Ok(rocm_core::LoopbackPortLease::engine_owned(port))
    }

    #[test]
    fn allocation_transaction_answers_an_equivalent_live_service_before_leasing() -> Result<()> {
        let (root, paths) = test_paths("alloc-txn-already-live");
        paths.ensure()?;
        let existing = write_live_service_record(
            &paths,
            "lemonade-qwen-9000",
            "qwen-canonical",
            11501,
            "starting",
        )?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "qwen-canonical",
            "127.0.0.1",
            serve_port::PortRequest::Auto,
            &mut never_probe,
        )?;
        let _ = fs::remove_dir_all(root);

        match decision {
            ManagedPortDecision::AlreadyLive(record) => {
                assert_eq!(record.service_id, existing.service_id);
                assert_eq!(record.port, existing.port);
            }
            ManagedPortDecision::Lease(_) => {
                panic!("an equivalent live service must be answered, not leased a new port")
            }
        }
        Ok(())
    }

    #[test]
    fn allocation_transaction_skips_ports_reserved_by_other_live_services() -> Result<()> {
        let (root, paths) = test_paths("alloc-txn-skips-reserved");
        paths.ensure()?;
        write_live_service_record(&paths, "lemonade-a", "model-a", 11_435, "running")?;
        write_live_service_record(&paths, "lemonade-b", "model-b", 11_436, "recovering")?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-c",
            "127.0.0.1",
            serve_port::PortRequest::Auto,
            &mut always_free,
        )?;
        let _ = fs::remove_dir_all(root);

        match decision {
            ManagedPortDecision::Lease(lease) => assert_eq!(lease.port(), 11_437),
            ManagedPortDecision::AlreadyLive(record) => {
                panic!("model-c has no live service, expected a lease, got {record:?}")
            }
        }
        Ok(())
    }

    #[test]
    fn allocation_transaction_ignores_stopped_records() -> Result<()> {
        let (root, paths) = test_paths("alloc-txn-ignores-stopped");
        paths.ensure()?;
        let mut stale =
            write_live_service_record(&paths, "lemonade-stale", "model-stale", 11_435, "running")?;
        // A dead pid demotes the record to stopped when liveness is refreshed, so
        // it must not reserve its port.
        stale.engine_pid = Some(999_999_999);
        stale.supervisor_pid = 999_999_999;
        stale.write()?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-fresh",
            "127.0.0.1",
            serve_port::PortRequest::Auto,
            &mut always_free,
        )?;
        let _ = fs::remove_dir_all(root);

        match decision {
            ManagedPortDecision::Lease(lease) => assert_eq!(lease.port(), 11_435),
            ManagedPortDecision::AlreadyLive(record) => {
                panic!("a stopped record must not block a fresh launch, got {record:?}")
            }
        }
        Ok(())
    }

    #[test]
    fn stopping_the_reserving_record_frees_its_explicit_port_for_reuse() -> Result<()> {
        // Proves the mechanism `rocm services stop` relies on: it never touches
        // the port directly, it only flips the stopped record's `status` (see
        // the `all_stopped` branch in `main.rs`'s stop handler). This recreates
        // that exact disk state and checks the same explicit port a live
        // competitor refused now resolves, closing the loop the E2E
        // `assert_port_reserved_message` step only checks by wording.
        let (root, paths) = test_paths("alloc-explicit-port-freed-by-stop");
        paths.ensure()?;
        let mut holder = ManagedServiceRecord::new(
            &paths,
            "lemonade-a-1",
            "lemonade",
            "model-a",
            "model-a".to_owned(),
            DEFAULT_LOCAL_HOST,
            11_435,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        holder.status = "running".to_owned();
        holder.engine_pid = Some(std::process::id());
        holder.write()?;

        let still_reserved = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-b",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Explicit(11_435),
            &mut always_free,
        );
        // The refusal names the service AND the command that frees the port;
        // the stop below is that command's own implementation, and the grant
        // after it is the outcome, so advice and result are pinned together.
        let stop_hint = service_hints::service_stop_hint("lemonade-a-1");
        assert!(
            matches!(still_reserved, Err(ref error) if error.to_string().contains("already reserved by managed service lemonade-a-1")
                && error.to_string().contains(&stop_hint)),
            "unexpected result while still running: {still_reserved:?}"
        );

        // Go through the real stop path (`rocm services stop <id> --yes`'s own
        // implementation) rather than writing `status = "stopped"` to disk
        // directly, so this test actually proves the advice serve-45 pins: that
        // running the stop command frees the port. The termination closure is
        // the same seam production injects a real signal through; here it just
        // confirms every recorded process gone without touching the test
        // process itself.
        crate::service_control::stop_internal_managed_service_with(&paths, "lemonade-a-1", |_| {
            Vec::new()
        })?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-b",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Explicit(11_435),
            &mut always_free,
        );
        let _ = fs::remove_dir_all(root);

        match decision? {
            ManagedPortDecision::Lease(lease) => assert_eq!(lease.port(), 11_435),
            ManagedPortDecision::AlreadyLive(existing) => {
                panic!(
                    "a stopped record must not block reuse: {}",
                    existing.service_id
                )
            }
        }
        Ok(())
    }

    #[test]
    fn allocation_transaction_rejects_an_explicit_port_a_recovering_service_holds() -> Result<()> {
        let (root, paths) = test_paths("alloc-txn-rejects-explicit-collision");
        paths.ensure()?;
        let existing = write_live_service_record(
            &paths,
            "lemonade-recovering",
            "model-recovering",
            11_435,
            "recovering",
        )?;

        let result = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-other",
            "127.0.0.1",
            serve_port::PortRequest::Explicit(11_435),
            &mut never_probe,
        );
        let _ = fs::remove_dir_all(root);

        let Err(error) = result else {
            panic!("an explicit port a recovering service holds must be refused")
        };
        let message = error.to_string();
        assert!(
            message.contains(&existing.service_id),
            "the refusal must name the service holding the port: {message}"
        );
        assert!(
            message.contains("recovering"),
            "the refusal must say why the port is unavailable: {message}"
        );
        Ok(())
    }

    #[test]
    fn concurrent_auto_transactions_publish_distinct_ports() -> Result<()> {
        let (root, paths) = test_paths("alloc-txn-concurrent-distinct");
        paths.ensure()?;
        let paths = std::sync::Arc::new(paths);

        let spawn_one = |model: &'static str| {
            let paths = std::sync::Arc::clone(&paths);
            thread::spawn(move || -> Result<u16> {
                let _launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
                let lease = match resolve_managed_port_in_transaction(
                    &paths,
                    "lemonade",
                    model,
                    "127.0.0.1",
                    serve_port::PortRequest::Auto,
                    &mut serve_port::loopback_bind_probe,
                )? {
                    ManagedPortDecision::Lease(lease) => lease,
                    ManagedPortDecision::AlreadyLive(record) => {
                        panic!("fresh model must not already be live: {record:?}")
                    }
                };
                let port = lease.port();
                write_live_service_record(&paths, model, model, port, "running")?;
                lease.release();
                Ok(port)
            })
        };

        let first = spawn_one("model-one");
        let second = spawn_one("model-two");
        let first_port = first.join().expect("thread one panicked")?;
        let second_port = second.join().expect("thread two panicked")?;
        let _ = fs::remove_dir_all(root);

        assert_ne!(first_port, second_port);
        for port in [first_port, second_port] {
            assert!(
                (serve_port::AUTO_PORT_FIRST..=serve_port::AUTO_PORT_LAST).contains(&port),
                "leased port {port} must fall within the auto-select range"
            );
        }
        Ok(())
    }

    #[test]
    fn concurrent_equivalent_transactions_converge_on_one_record() -> Result<()> {
        let (root, paths) = test_paths("alloc-txn-concurrent-converge");
        paths.ensure()?;
        let paths = std::sync::Arc::new(paths);

        let race_once = || {
            let paths = std::sync::Arc::clone(&paths);
            thread::spawn(move || -> Result<bool> {
                let _launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
                match resolve_managed_port_in_transaction(
                    &paths,
                    "lemonade",
                    "shared-model",
                    "127.0.0.1",
                    serve_port::PortRequest::Auto,
                    &mut serve_port::loopback_bind_probe,
                )? {
                    ManagedPortDecision::Lease(lease) => {
                        let port = lease.port();
                        write_live_service_record(
                            &paths,
                            "lemonade-shared",
                            "shared-model",
                            port,
                            "running",
                        )?;
                        lease.release();
                        Ok(true)
                    }
                    ManagedPortDecision::AlreadyLive(_) => Ok(false),
                }
            })
        };

        let first = race_once();
        let second = race_once();
        let first_launched = first.join().expect("thread one panicked")?;
        let second_launched = second.join().expect("thread two panicked")?;

        let live_count = live_managed_services(&paths).len();
        let _ = fs::remove_dir_all(root);

        assert_ne!(
            first_launched, second_launched,
            "exactly one racing transaction must launch"
        );
        assert_eq!(live_count, 1, "only one record may end up live");
        Ok(())
    }

    #[test]
    fn abandoned_launch_leaves_no_record_log_or_endpoint_key() -> Result<()> {
        let (root, paths) = test_paths("abandon-launch-cleanup");
        paths.ensure()?;
        let mut record = write_live_service_record(
            &paths,
            "lemonade-abandon",
            "abandon-model",
            11_490,
            "running",
        )?;
        fs::create_dir_all(
            record
                .engine_state_path
                .parent()
                .expect("engine state parent"),
        )?;
        fs::write(&record.engine_state_path, "{}")?;
        endpoint_keys::store_endpoint_api_key(&paths, &record.service_id, "secret-key")?;
        // Simulate a pre-spawn failure: nothing ever ran.
        record.engine_pid = None;

        abandon_failed_managed_launch(&paths, &record, false);

        assert!(!record.manifest_path.exists());
        assert!(!record.log_path.exists());
        assert!(!record.engine_state_path.exists());
        assert_eq!(
            endpoint_keys::endpoint_api_key(&paths, &record.service_id),
            None
        );
        assert!(live_managed_services(&paths).is_empty());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// What the cleanup asks of the termination, through the seam and so on every
    /// host: the child's PID with the token captured at spawn, and nothing at all
    /// for a child with no token where tokens are recorded. The record is marked
    /// stopped either way and the audit event carries the verdict the termination
    /// gave.
    #[test]
    fn abandoned_launch_asks_the_termination_for_the_spawned_child_with_its_token() -> Result<()> {
        use std::cell::RefCell;
        let cases = [
            // (token captured, whether the cleanup is expected to ask)
            (Some(77_u64), true),
            (None, !host_records_start_tokens()),
        ];
        for (token, expect_call) in cases {
            let (root, paths) = test_paths("abandon-launch-asks-termination");
            paths.ensure()?;
            let mut record = write_live_service_record(
                &paths,
                "lemonade-abandon-seam",
                "abandon-seam-model",
                11_494,
                "running",
            )?;
            record.engine_pid = Some(4242);
            record.engine_start_ticks = token;
            let asked = RefCell::new(Vec::new());

            abandon_failed_managed_launch_with(&paths, &record, false, |identity| {
                asked.borrow_mut().push(*identity);
                rocm_core::TerminationOutcome::Forced
            });

            let asked = asked.into_inner();
            let persisted = load_managed_services(&paths)?
                .into_iter()
                .find(|candidate| candidate.service_id == record.service_id)
                .expect("a launch that spawned a child keeps its record");
            let _ = fs::remove_dir_all(root);
            let expected = if expect_call {
                vec![rocm_core::ProcessIdentity::new(4242, token)]
            } else {
                Vec::new()
            };
            assert_eq!(asked, expected, "token {token:?}");
            assert_eq!(persisted.status, "stopped", "token {token:?}");
        }
        Ok(())
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn abandoned_launch_never_signals_a_pid_that_no_longer_matches_the_spawn_token() -> Result<()> {
        // The child exited and its pid now belongs to an unrelated live
        // process: the start token recorded at spawn no longer matches, so the
        // forced tree kill must not touch it.
        let (root, paths) = test_paths("abandon-launch-recycled-pid");
        paths.ensure()?;
        let mut record = write_live_service_record(
            &paths,
            "lemonade-abandon-recycled",
            "abandon-recycled-model",
            11_493,
            "running",
        )?;
        let current_exe = std::env::current_exe().expect("current test exe");
        let mut bystander = ProcessCommand::new(current_exe)
            .arg("--exact")
            .arg(sleeper_test_name())
            .arg("--ignored")
            .arg("--test-threads=1")
            .env("ROCM_TEST_SLEEPER_CHILD", "1")
            .spawn()
            .expect("spawn bystander");
        record.record_engine_identity(bystander.id());
        record.engine_start_ticks = record.engine_start_ticks.map(|ticks| ticks + 1);

        abandon_failed_managed_launch(&paths, &record, false);

        let mismatched_left_alone = bystander.try_wait().expect("poll bystander").is_none();
        // A child that was gone before its token could be captured leaves none:
        // that must not fall back to "best-effort" signalling either.
        record.engine_start_ticks = None;
        abandon_failed_managed_launch(&paths, &record, false);
        let still_running =
            mismatched_left_alone && bystander.try_wait().expect("poll bystander").is_none();
        let _ = bystander.kill();
        let _ = bystander.wait();
        let _ = fs::remove_dir_all(root);
        assert!(
            still_running,
            "a pid with a mismatched token must be left alone"
        );
        Ok(())
    }

    #[test]
    #[cfg(not(windows))]
    fn abandoned_launch_terminates_the_spawned_child() -> Result<()> {
        let (root, paths) = test_paths("abandon-launch-terminates-child");
        paths.ensure()?;
        let mut record = write_live_service_record(
            &paths,
            "lemonade-abandon-child",
            "abandon-child-model",
            11_491,
            "running",
        )?;

        let current_exe = std::env::current_exe().expect("current test exe");
        let mut child = ProcessCommand::new(current_exe)
            .arg("--exact")
            .arg(sleeper_test_name())
            .arg("--ignored")
            .arg("--test-threads=1")
            .env("ROCM_TEST_SLEEPER_CHILD", "1")
            .spawn()
            .expect("spawn sleeper child");
        let child_pid = child.id();
        record.record_engine_identity(child_pid);

        abandon_failed_managed_launch(&paths, &record, false);

        let wait_result = child.wait();

        match wait_result {
            Ok(status) => assert!(
                !status.success(),
                "the sleeper child must be terminated, not exit cleanly"
            ),
            Err(error) => panic!("failed to wait for sleeper child: {error}"),
        }
        let persisted = load_managed_services(&paths)
            .expect("re-read manifests")
            .into_iter()
            .find(|candidate| candidate.service_id == record.service_id);
        let _ = fs::remove_dir_all(root);
        let persisted = persisted.expect(
            "a launch that spawned a child must keep its record so `rocm services`/`rocm logs` can show the attempt",
        );
        // `stopped`, not `failed`: `rocmd`'s recovery restarts a `failed` record
        // that carries no stop marker, which would respawn a launch the CLI just
        // reported as failed. The rocmd side of the pair is
        // `abandoned_launch_record_is_not_a_recovery_candidate` in
        // `apps/rocmd/src/watchers.rs`.
        assert_eq!(
            persisted.status, "stopped",
            "a spawned child's record must be kept, but in a status recovery ignores"
        );
        Ok(())
    }

    /// A sleeper the test owns, standing in for an unrelated process that now
    /// holds a pid the launch once recorded. Spawned with `std`'s `Command`
    /// directly, so the test runs on Windows too, where this matters most: no
    /// start-time token guards a kill there.
    fn spawn_bystander() -> std::process::Child {
        std::process::Command::new(std::env::current_exe().expect("current test exe"))
            .arg("--exact")
            .arg(sleeper_test_name())
            .arg("--ignored")
            .arg("--test-threads=1")
            .env("ROCM_TEST_SLEEPER_CHILD", "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn bystander")
    }

    #[test]
    fn an_engine_seen_dead_at_startup_is_never_signalled() -> Result<()> {
        // The startup check saw the engine exit, so its pid is free. Here a
        // bystander holds it. Cleanup must not signal it — not even through a
        // start-time token, which on Windows would be read only after the
        // engine had died and so would describe the bystander — and must
        // still keep the record and log of the failed launch for the error
        // that names them.
        let (root, paths) = test_paths("abandon-known-dead-engine");
        paths.ensure()?;
        let mut record = write_live_service_record(
            &paths,
            "lemonade-abandon-known-dead",
            "abandon-known-dead-model",
            11_495,
            "starting",
        )?;
        fs::write(&record.log_path, "engine: bad flag\n")?;
        let mut bystander = spawn_bystander();
        record.record_engine_identity(bystander.id());

        abandon_failed_managed_launch(&paths, &record, true);

        let left_alone = bystander.try_wait().expect("poll bystander").is_none();
        let _ = bystander.kill();
        let _ = bystander.wait();
        let persisted = ManagedServiceRecord::read_manifest(&paths, &record.service_id);
        let log_kept = record.log_path.exists();
        let _ = fs::remove_dir_all(root);
        assert!(
            left_alone,
            "a pid the dead engine freed must not be signalled"
        );
        assert_eq!(persisted?.status, "stopped");
        assert!(log_kept, "the log the error names must be kept");
        Ok(())
    }

    #[test]
    fn abandoning_a_launch_does_not_recreate_a_removed_record() -> Result<()> {
        let (root, paths) = test_paths("abandon-removed-record");
        paths.ensure()?;
        let record = write_live_service_record(
            &paths,
            "lemonade-abandon-removed",
            "abandon-removed-model",
            11_496,
            "starting",
        )?;
        fs::remove_file(&record.manifest_path)?;

        abandon_failed_managed_launch(&paths, &record, true);

        let recreated = record.manifest_path.exists();
        let _ = fs::remove_dir_all(root);
        assert!(!recreated, "cleanup must not bring back a removed record");
        Ok(())
    }

    #[test]
    fn the_launch_does_not_publish_over_a_record_a_restart_took_over() -> Result<()> {
        // A restart records its own supervisor on a record that still reads as
        // live. The launch's claim names supervisor 0, so it is no longer the
        // launch's to write.
        let (root, paths) = test_paths("launch-owns-supervisor");
        paths.ensure()?;
        let launched = write_live_service_record(
            &paths,
            "lemonade-owns-supervisor",
            "owns-supervisor-model",
            11_497,
            "starting",
        )?;
        let mut taken = ManagedServiceRecord::read_manifest(&paths, &launched.service_id)?;
        taken.supervisor_pid = launched.supervisor_pid.wrapping_add(7);
        taken.write()?;
        assert!(crate::managed_service_status::managed_service_is_live(
            &taken
        ));

        assert!(!launch_still_owns(&taken, &launched));
        let written = update_managed_record_on_disk(
            &paths,
            &launched.service_id,
            |current| launch_still_owns(current, &launched),
            |current| current.status = "running".to_owned(),
        )?;
        let after = ManagedServiceRecord::read_manifest(&paths, &launched.service_id)?;
        let _ = fs::remove_dir_all(root);
        assert!(
            written.is_none(),
            "the launch wrote over a restart's record"
        );
        assert_eq!(after.status, "starting");
        assert_eq!(after.supervisor_pid, taken.supervisor_pid);
        // Same supervisor pid but a different start token: a restart whose new
        // supervisor reused the pid. Not the launch's either.
        let mut reused = launched.clone();
        reused.supervisor_start_ticks = Some(launched.supervisor_start_ticks.unwrap_or(0) + 1);
        assert!(!launch_still_owns(&reused, &launched));
        // Same supervisor, same start token, still live: the launch owns it.
        assert!(launch_still_owns(&launched, &launched));
        Ok(())
    }

    #[test]
    fn an_engine_seen_dead_before_it_was_recorded_keeps_its_record_and_log() -> Result<()> {
        // The Windows shape: the startup check saw the engine exit before
        // `record_engine_identity` ran, so no engine pid is on record. Nothing
        // is spawned-and-recorded, yet the failure names the log, so the record
        // and log must survive, marked `stopped`.
        let (root, paths) = test_paths("abandon-dead-engine-never-recorded");
        paths.ensure()?;
        let mut record = write_live_service_record(
            &paths,
            "lemonade-abandon-never-recorded",
            "abandon-never-recorded-model",
            11_498,
            "starting",
        )?;
        record.engine_pid = None;
        record.supervisor_pid = 0;
        fs::write(&record.log_path, "engine: bad flag\n")?;
        record.write()?;

        abandon_failed_managed_launch(&paths, &record, true);

        let persisted = ManagedServiceRecord::read_manifest(&paths, &record.service_id);
        let log_kept = record.log_path.exists();
        let _ = fs::remove_dir_all(root);
        assert_eq!(persisted?.status, "stopped");
        assert!(log_kept, "the log the error names must be kept");
        Ok(())
    }

    /// A failed launch's cleanup refuses to signal a child whose token was not
    /// captured on any host that records tokens. Windows is one of them, since
    /// it records the creation time: a missing token there means the child was
    /// gone before it could be captured, and its pid may now be another
    /// process's, so the kill must not proceed as it did when Windows had none.
    #[test]
    fn launch_cleanup_refuses_a_child_without_a_token_only_where_tokens_exist() {
        use rocm_core::TerminationOutcome::Unverified;

        // A host that records tokens: a missing one is unverifiable, a captured
        // one goes on to be verified by the termination.
        assert_eq!(launch_cleanup_refusal(None, true), Some(Unverified));
        assert_eq!(launch_cleanup_refusal(Some(7), true), None);
        // A host with no tokens has nothing to verify and proceeds as before.
        assert_eq!(launch_cleanup_refusal(None, false), None);
        assert_eq!(launch_cleanup_refusal(Some(7), false), None);
    }

    /// Ties the cleanup's idea of "this host records tokens" to what
    /// `process_start_ticks` really does, on the host the test runs on: Linux and
    /// Windows read a token for a live process (this one), anywhere else there
    /// is none. If a platform gains or loses tokens without the cleanup being
    /// told, this fails there.
    #[test]
    fn the_cleanup_guard_agrees_with_whether_this_host_reads_tokens() {
        assert_eq!(
            host_records_start_tokens(),
            rocm_core::process_start_ticks(std::process::id()).is_some()
        );
    }
}
