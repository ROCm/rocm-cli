// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! GPU/device-policy parsing, `--gpu`/`--device` selection, and per-GPU VRAM
//! usage telemetry/warnings: parsing the `--device`/`--gpu` CLI and internal
//! arguments into `DevicePolicy`/`GpuSelection`, validating and
//! auto-selecting GPU ordinals (mask- and busy-service-aware) under the
//! managed-launch lock, reading per-GPU VRAM occupancy (`amd-smi metric
//! --json`, with an amdgpu DRM sysfs fallback), and building the serve-plan
//! low-VRAM warning from it.
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-214) of the `main.rs` modularization (see
//! `docs/architecture.md`). `dispatch()` calls directly into this module at
//! one point — the `Command::EngineServeHttp` arm's
//! `parse_gpu_indices_arg(gpu.as_deref())?` — re-imported via `use
//! crate::gpu_policy::parse_gpu_indices_arg;`; `dispatch()`'s own body is
//! otherwise unaffected.
//!
//! This module owns: `parse_device_policy`, `parse_gpu_selection`,
//! `parse_gpu_indices_arg`, the managed-launch-lock GPU selection
//! (`select_gpu_indices_under_launch_lock`, `validate_pinned_gpu_index`,
//! `effective_gpu_count`, `auto_select_gpu_indices`, `select_auto_gpu_index`,
//! `busy_gpu_indices`), per-GPU VRAM telemetry (the `GpuVramUsage` type,
//! `gpu_vram_usage`, `simulated_low_vram_usage`, `gpu_vram_usage_amd_smi`,
//! `gpu_vram_usage_sysfs`, `read_drm_vram_usage`, `read_sysfs_u64`,
//! `drm_device_is_amd`, `parse_gpu_vram_usage`), the serve-plan low-VRAM
//! warning (`gpu_low_memory_warning`, `vram_capacity_is_meaningful`,
//! `serve_gpu_low_memory_warning`), and the best-effort `amd-smi`-based GPU
//! count (`detect_gpu_count`).
//!
//! One item physically interleaved in this span in `main.rs` stayed there
//! instead of moving: `const fn device_policy_name` sits immediately after
//! `detect_gpu_count` and before the next cluster's (ROCMAI-215)
//! `EngineSelection`/`resolve_engine_selection`, but it already had callers
//! outside this cluster — `engine_ipc.rs` and `service_launch.rs` reach it
//! unqualified via `use crate::{...}`, same as `serve_cmd.rs` does — plus its
//! own test, so it is not part of this cluster's relocation.
//!
//! `main.rs` calls into it for `GpuVramUsage`, `gpu_vram_usage`,
//! `parse_device_policy` (from `restart_child_argv`), `parse_gpu_indices_arg`
//! (the `dispatch()` call site above), `serve_gpu_low_memory_warning`, and
//! `vram_capacity_is_meaningful`; `serve_cmd.rs` calls into it for
//! `detect_gpu_count`, `gpu_vram_usage`, `parse_device_policy`,
//! `parse_gpu_selection`, `select_gpu_indices_under_launch_lock`,
//! `serve_gpu_low_memory_warning`, and `validate_pinned_gpu_index`;
//! `engine_ipc.rs` calls into it for `parse_device_policy` (from
//! `run_builtin_engine_serve_http`, itself reached from `dispatch()`'s own
//! `__engine-serve-http` arm one level further down). `effective_gpu_count`,
//! `auto_select_gpu_indices`, `select_auto_gpu_index`,
//! `simulated_low_vram_usage`, `gpu_vram_usage_amd_smi`,
//! `gpu_vram_usage_sysfs`, `read_drm_vram_usage`, `read_sysfs_u64`,
//! `drm_device_is_amd`, `parse_gpu_vram_usage`, `gpu_low_memory_warning`, and
//! `busy_gpu_indices` have no callers outside this module and stay private;
//! `GpuVramUsage` and its fields are `pub(crate)` for the same reason —
//! `main.rs`'s `classify_accelerator_memory`/`collect_serve_notes` and their
//! own tests construct and read them directly.
//!
//! `gpu_vram_usage_sysfs` has two definitions, `#[cfg(target_os = "linux")]`
//! and `#[cfg(not(target_os = "linux"))]` — both moved intact, cfg-gates
//! unchanged, no platform behavior change.

use std::fs;
use std::path::Path;
use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Result, bail};
use rocm_core::AppPaths;
use rocm_engine_protocol::{DevicePolicy, GpuSelection};

use crate::managed_service_status::load_managed_services;

pub(crate) fn parse_device_policy(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "auto" | "gpu" | "gpu_required" | "gpu_preferred" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => bail!(
            "rocm serve requires ROCm GPU execution; CPU mode is not a fallback path in rocm-cli"
        ),
        other => bail!("unsupported device policy: {other}"),
    }
}

/// Parse the user-facing `--gpu` value (`auto` or a single index like `1`)
/// into a `GpuSelection`. Comma lists are rejected (single-GPU serving only).
/// `None` defaults to `auto`.
pub(crate) fn parse_gpu_selection(value: Option<&str>) -> Result<GpuSelection> {
    let Some(raw) = value else {
        return Ok(GpuSelection::Auto);
    };
    GpuSelection::parse_cli_value(raw).map_err(|message| anyhow::anyhow!(message))
}

/// Parse the internal `--gpu` csv passed to the hidden `__engine-serve-http`
/// subcommand back into explicit device ordinals (empty when absent/`auto`).
pub(crate) fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    match parse_gpu_selection(value)? {
        GpuSelection::Auto => Ok(Vec::new()),
        GpuSelection::Index(index) => Ok(vec![index]),
    }
}

/// Take the managed-launch lock and decide, under it, which device ordinal this
/// server pins. This is the whole select-then-claim critical section's entry
/// half: the returned [`rocm_core::FileLock`] must be held by the caller until
/// the claiming service record is persisted, at which point the choice is
/// visible to any concurrent auto-selection and the lock is dropped (in
/// `start_managed_service`/`run_attached_service`, before the readiness wait).
/// Acquiring the lock and reading busy-GPU state are one operation here on
/// purpose: split apart, two concurrent `rocm serve --gpu auto` both read the
/// same GPU as free and land on it.
///
/// `pinned` is the already-validated explicit `--gpu <index>` (see
/// [`validate_pinned_gpu_index`], which `serve()` applies before runtime
/// resolution so a bad ordinal is refused without a runtime). Explicit-index and
/// CPU-only launches take the lock too — their selection is fixed, but their
/// claim-record write must stay serialized against a concurrent auto-select.
/// Only `--gpu auto` (`cpu_only` false, `pinned` `None`) reads live state under
/// it, picking the lowest-numbered GPU that is idle (high free VRAM) and not
/// already serving a rocm-cli managed/foreground model. The result holds at most
/// one ordinal; serving across multiple GPUs is not supported.
///
/// Ordinal semantics: the index produced here is fed to engines via
/// `HIP_VISIBLE_DEVICES`, but it is sourced from `amd-smi`'s `gpu` index
/// (probing/busy detection). On a normal single-host ROCm install without GPU
/// partitioning these orderings coincide, but they are not guaranteed to match
/// when `ROCR_VISIBLE_DEVICES`, CPX/partition modes, or non-default device
/// enumeration are in play. Validate on multi-GPU hardware before relying on a
/// specific `--gpu <index>` mapping in those configurations.
///
/// Selection is mask-aware: `visible` is the set of HIP ordinals still usable
/// after the active visibility mask (see [`rocm_core::usable_amd_gpu_indices`]) —
/// already in the space the choice is exported through, since a
/// `ROCR_VISIBLE_DEVICES` mask has its survivors re-indexed to `0..N`.
/// Auto-selection is restricted to it, so serve never targets a masked-out
/// device. `None` (an unprobeable host) keeps the previous mask-unaware
/// behavior.
///
/// The lock is held for as long as the caller keeps the guard, which includes
/// the already-running check inside `spawn_managed_engine_child` — that refreshes
/// managed-service liveness and can issue per-service endpoint probes, so an
/// unrelated concurrent serve can wait on the order of seconds. `FileLock`
/// blocks without a timeout; the slow first-use install is deliberately kept
/// outside this section so it is not also serialized.
///
/// That span is relied on by a second caller: because the guard outlives both
/// `store_endpoint_api_key` and the first `record.write()`, it also covers the
/// interval in which a launch's 0600 endpoint key sits on disk with no manifest
/// beside it. [`prune_managed_service_records`] acquires the same lock so its
/// leftover sweep cannot read the directory in that state. Shortening the
/// guard's life — releasing it before the record is persisted — would silently
/// reopen that window as well as the GPU double-booking one.
pub(crate) fn select_gpu_indices_under_launch_lock(
    paths: &AppPaths,
    cpu_only: bool,
    pinned: Option<Vec<u32>>,
    detect_count: impl FnOnce() -> Option<usize>,
    visible: Option<&[u32]>,
    vram: Option<&[GpuVramUsage]>,
) -> Result<(Vec<u32>, rocm_core::FileLock)> {
    let lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
    let indices = if cpu_only {
        Vec::new()
    } else if let Some(indices) = pinned {
        indices
    } else {
        // `detect_count` is invoked only here: it shells out to amd-smi, and the
        // CPU-only and explicit-index paths have no use for the count, so they do
        // not pay for that subprocess inside the critical section.
        auto_select_gpu_indices(paths, detect_count(), visible, vram)
    };
    Ok((indices, lock))
}

/// Validate an explicit `--gpu <index>` against the active visibility set,
/// returning the single pinned ordinal. `visible` is the set of HIP ordinals
/// still usable after the visibility mask (see
/// [`rocm_core::usable_amd_gpu_indices`]) — already in the same space the index
/// is exported through — so it is the authoritative answer whenever it could be
/// enumerated. `detected` is only a fallback range check for hosts where the
/// visible set is unknown (`visible` is `None`). Both `None` (an unprobeable
/// host) is allowed through so serving can still proceed.
pub(crate) fn validate_pinned_gpu_index(
    index: u32,
    detected: Option<usize>,
    visible: Option<&[u32]>,
    mask_active: bool,
) -> Result<Vec<u32>> {
    // Prefer the visibility-resolved set. It is authoritative and already in the
    // HIP-ordinal space `--gpu` is exported through, so it settles both range and
    // mask membership in one comparison — avoiding the earlier bug of checking
    // the index against `detected` (a differently-probed physical count) and
    // `visible` (HIP space) as if they shared one ordinal space.
    //
    // `mask_active` only shapes the message, not the accept/reject decision: with
    // no mask set the visible set is just `0..present`, so an index outside it is
    // simply absent from the host, and blaming HIP/ROCR variables the user never
    // set sends them chasing an environment problem that does not exist.
    if let Some(visible) = visible {
        if visible.is_empty() {
            // An empty set is authoritative "no GPU usable" (every device masked
            // out, or none present), not "could not enumerate" — that is `None`.
            // serve()'s no-usable-GPU fail-fast normally reports this first with a
            // fuller message; reject here too so a bad `--gpu` can never slip
            // through.
            if mask_active {
                bail!(
                    "no usable AMD GPU is available under the active visibility mask \
                     (HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES); `--gpu {index}` cannot be honoured"
                );
            }
            bail!("no usable AMD GPU is present on this host; `--gpu {index}` cannot be honoured");
        }
        if !visible.contains(&index) {
            let list = visible
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            if mask_active {
                bail!(
                    "--gpu index {index} is not available under the active visibility mask \
                     (HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES); usable GPU indices: [{list}]"
                );
            }
            bail!("--gpu index {index} is not present on this host; usable GPU indices: [{list}]");
        }
        return Ok(vec![index]);
    }
    // No visible set could be enumerated: fall back to the raw detected count so
    // an obviously out-of-range index is still refused.
    if let Some(count) = detected
        && (index as usize) >= count
    {
        bail!(
            "--gpu index {index} is out of range; {count} GPU(s) detected (valid indices 0..{})",
            count - 1
        );
    }
    Ok(vec![index])
}

/// The GPU count to rank `--gpu auto` candidates over. Prefers the `amd-smi`
/// device count, but only when that count is usable — `None` (amd-smi
/// unavailable) and `Some(0)` (amd-smi ran and saw no devices) both fall through
/// to the DRM sysfs VRAM fallback's row count, so auto-selection can still
/// consider devices amd-smi did not enumerate. Used only to decide whether
/// `--gpu auto` has anything at all to rank, never as a `--gpu <index>`
/// validation bound — see [`validate_pinned_gpu_index`]. `None` when neither
/// source yields a non-zero count.
fn effective_gpu_count(detected: Option<usize>, vram: Option<&[GpuVramUsage]>) -> Option<usize> {
    detected
        .filter(|&count| count > 0)
        .or_else(|| vram.map(<[GpuVramUsage]>::len).filter(|&count| count > 0))
}

/// A GPU's local VRAM occupancy as reported by `amd-smi metric --json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuVramUsage {
    pub(crate) index: u32,
    pub(crate) used_mb: u64,
    pub(crate) total_mb: u64,
}

impl GpuVramUsage {
    /// Fraction of total VRAM that is currently free (`0.0`..=`1.0`). Returns
    /// `None` when the total is unknown so callers do not divide by zero.
    fn free_fraction(self) -> Option<f64> {
        if self.total_mb == 0 {
            return None;
        }
        Some(self.total_mb.saturating_sub(self.used_mb) as f64 / self.total_mb as f64)
    }

    /// Absolute VRAM currently free, in MiB. Used to compare GPUs of differing
    /// total capacity: a higher free *fraction* on a smaller GPU can still mean
    /// less free memory than a larger GPU, so auto-selection ranks by this.
    const fn free_mb(self) -> u64 {
        self.total_mb.saturating_sub(self.used_mb)
    }
}

/// A GPU is treated as "free" for `--gpu auto` when at least this fraction of
/// its VRAM is unused (i.e. it is effectively idle).
const AUTO_FREE_VRAM_FRACTION: f64 = 0.90;

/// Pick a GPU ordinal for `--gpu auto`. Prefers the lowest-numbered GPU that is
/// idle (free VRAM at or above [`AUTO_FREE_VRAM_FRACTION`]) and not pinned by a
/// running rocm-cli service; otherwise falls back to the non-busy GPU with the
/// most free VRAM, then to the first non-busy GPU. When neither the amd-smi
/// device count nor the DRM sysfs VRAM fallback yields any device it returns no
/// selection rather than assuming device 0 — the engine's device probe then pins
/// the first present GPU or fails fast under the GPU-required policy.
///
/// Selection is mask-aware: candidates are restricted to `visible`, the HIP
/// ordinals still usable after the active visibility mask (see
/// [`rocm_core::usable_amd_gpu_indices`]), so auto-select never lands on a
/// masked-out GPU the engine would then reject. `None` (an unprobeable host)
/// leaves the detected range unrestricted, as before.
///
/// Under a `ROCR_VISIBLE_DEVICES` mask the *choice* is correct — `visible` is
/// already re-indexed into the HIP space the selection is exported through — but
/// the `vram` rows come from amd-smi, which does not honour that mask, so the
/// idleness ranking can read a different device's occupancy than the one the
/// ordinal binds. That is a best-effort heuristic, not a binding hazard: it can
/// pick a less idle visible GPU, never a hidden one. Translating amd-smi rows
/// through a ROCR mask is deferred (needs multi-GPU hardware to verify).
///
/// This reads service state (`busy_gpu_indices`) but does not lock on its own.
/// Concurrency safety is the caller's responsibility: `serve()` holds the
/// managed-launch lock (`AppPaths::managed_launch_lock_path`) across this
/// selection and the claiming record write, so two near-concurrent `--gpu auto`
/// launches are serialized and cannot land on the same idle GPU. Callers that
/// select outside that lock get only best-effort de-confliction from the
/// VRAM-occupancy fallback and the start-time low-memory warning.
fn auto_select_gpu_indices(
    paths: &AppPaths,
    detected: Option<usize>,
    visible: Option<&[u32]>,
    vram: Option<&[GpuVramUsage]>,
) -> Vec<u32> {
    let busy = busy_gpu_indices(paths);
    select_auto_gpu_index(detected, visible, &busy, vram)
}

/// Pure auto-selection used by [`auto_select_gpu_indices`], split out so the
/// preference order can be unit-tested without amd-smi or service state.
fn select_auto_gpu_index(
    detected: Option<usize>,
    visible: Option<&[u32]>,
    busy: &[u32],
    vram: Option<&[GpuVramUsage]>,
) -> Vec<u32> {
    // The amd-smi `list` device count is the primary source, but on a shared
    // node without amd-smi it is `None` even though the DRM sysfs fallback probe
    // may still have populated `vram`. Derive the count from those rows in that
    // case so `--gpu auto` can rank GPUs by free VRAM instead of returning no
    // selection and landing on a busy device 0.
    let count = effective_gpu_count(detected, vram).unwrap_or(0);
    if count == 0 {
        // No GPU count from amd-smi and no VRAM telemetry (both unavailable, or
        // genuinely zero devices). Do not assume device 0 exists: return no
        // selection and let the engine's device probe pin the first present GPU
        // or fail fast under the GPU-required policy (no GPU-0 fallback).
        return Vec::new();
    }
    // Ordinals this selection may hand out, lowest first. When VRAM telemetry is
    // present, drive the scan from the *actual* row indices it reports rather
    // than a synthetic `0..count` range: amd-smi metric (and the DRM sysfs
    // fallback) can report non-contiguous ordinals when devices are hidden by a
    // visibility mask, so `0..count` would look up rows that do not exist and
    // miss the real ones. Without telemetry, fall back to the dense detected
    // range so service-state-only selection still has candidates.
    //
    // Either way the set is then narrowed to the ordinals still visible under
    // the active mask, so auto-select never targets a masked-out GPU. `visible`
    // is in HIP space (a ROCR mask's survivors are re-indexed to `0..N`), which
    // is the space the selection is exported through, so the retained ordinals
    // are directly selectable. When the host is unprobeable (`visible` is
    // `None`) the set is left unrestricted (mask-unaware, as before).
    //
    // A short row set is not always a mask, though: `parse_gpu_vram_usage`
    // drops any device entry missing `/mem_usage/used_vram/value` or
    // `/mem_usage/total_vram/value`, so a device the lighter `list` enumeration
    // counts can simply have no row. Taking rows-only there would shrink the
    // candidate set for a reason that has nothing to do with visibility, and if
    // every *reported* device is busy the terminal fallback would hand back a
    // busy GPU while an idle, merely untelemetried one went unconsidered —
    // exactly the "serve pinned to an occupied GPU" fault this selection exists
    // to avoid. So the `0..count` range is unioned back in — but only where it
    // is evidence rather than invention, which is decided by the rows, not by
    // `visible`:
    //
    //   * `detected` must be `Some(n > 0)`, so `count` is the `list` device
    //     count — an independent source asserting those ordinals exist. When it
    //     is absent `effective_gpu_count` falls back to `rows.len()`, which is
    //     only the rows restating their own size and confirms nothing they omit.
    //   * every row index must be below `count`. A row at or above it is the one
    //     available proof that the ordinal space is re-indexed or sparse rather
    //     than dense from 0 (`[2, 3]` with `count == 2`), and that is exactly the
    //     masked shape the rows-only construction was introduced for, so there
    //     the rows stay authoritative. Below the count the rows are a subset of a
    //     dense range the count already asserts, and `0..count` is no stronger an
    //     assumption than the telemetry-less arm below already makes.
    //
    // Gating on `visible.is_some()` instead would have left the bug standing
    // wherever the mask is unknown, which is not a corner: `usable_amd_gpu_indices`
    // is `None` unconditionally off Linux, so `visible` is always `None` on
    // Windows. The retain below still runs on top whenever a mask *is* known, so
    // no synthesised ordinal survives that the mask contradicts.
    let mut reported: Vec<u32> = match vram {
        Some(rows) => {
            let mut indices: Vec<u32> = rows.iter().map(|row| row.index).collect();
            let count_is_independently_sourced = detected.is_some_and(|detected| detected > 0);
            let rows_are_a_dense_range_subset =
                indices.iter().all(|&index| (index as usize) < count);
            if count_is_independently_sourced && rows_are_a_dense_range_subset {
                indices.extend(0..count as u32);
            }
            indices
        }
        None => (0..count as u32).collect(),
    };
    reported.sort_unstable();
    // The union above can repeat an ordinal that both sources name.
    reported.dedup();
    if let Some(visible) = visible {
        reported.retain(|index| visible.contains(index));
    }
    if reported.is_empty() {
        // Every reported device is masked out (or telemetry reported no rows at
        // all). Do not assume device 0 exists: return no selection and let the
        // engine's device probe pin the first present GPU or fail fast under the
        // GPU-required policy (no GPU-0 fallback).
        return Vec::new();
    }
    // The subset still free to claim. Kept separate from `reported` so the
    // terminal all-busy fallback below can still name a device that was actually
    // reported instead of fabricating one.
    let candidate_indices: Vec<u32> = reported
        .iter()
        .copied()
        .filter(|index| !busy.contains(index))
        .collect();
    let usage_for = |index: u32| vram.and_then(|rows| rows.iter().find(|row| row.index == index));

    if vram.is_some() {
        // Pass 1: lowest-index idle GPU.
        for &index in &candidate_indices {
            if let Some(free) = usage_for(index).and_then(|usage| usage.free_fraction())
                && free >= AUTO_FREE_VRAM_FRACTION
            {
                return vec![index];
            }
        }
        // Pass 2: the non-busy GPU with the most free VRAM in absolute terms
        // (not free percentage, which can favor a smaller GPU on heterogeneous
        // VRAM systems). `usage_for` is only *partial* over `candidate_indices` —
        // the union above puts back ordinals the `list` count confirms but
        // telemetry never reported — yet no "does this ordinal have a row?" guard
        // is needed, because the comparator already orders them last: the keys are
        // `Option<u64>`, and `None` sorts below every `Some`, including `Some(0)`.
        // So a rowless ordinal can only be the maximum when *no* candidate has a
        // row; then every key is `None`, every comparison falls through to the
        // index tie-break, and the winner is the lowest candidate index — which is
        // precisely what Pass 3 would return from an ascending `reported`. Adding
        // the guard back would therefore change no outcome, only skip a pass that
        // already agrees.
        if let Some(&index) = candidate_indices.iter().max_by(|left, right| {
            let left_free = usage_for(**left).map(|usage| usage.free_mb());
            let right_free = usage_for(**right).map(|usage| usage.free_mb());
            left_free
                .cmp(&right_free)
                // Break ties toward the lowest index.
                .then(right.cmp(left))
        }) {
            return vec![index];
        }
    }

    // Pass 3: no VRAM telemetry; first GPU not pinned by a managed service.
    if let Some(&index) = candidate_indices.first() {
        return vec![index];
    }
    // Every reported, visible GPU is pinned by a running service. Return the
    // lowest of them (no CPU fallback is ever used); the caller surfaces a
    // low-memory warning so the user can free a device or pick another `--gpu`.
    // It must come from `reported`, not a hardcoded 0: once candidates are
    // sourced from the VRAM rows those ordinals can be sparse (a visibility mask
    // can leave only `[2, 3]`), so a literal 0 would pin a device nothing ever
    // reported — exported straight into `HIP_VISIBLE_DEVICES` as a silently
    // wrong choice. `reported` is non-empty here, guaranteed by the check above.
    vec![reported[0]]
}

/// Best-effort per-GPU VRAM occupancy. Prefers `amd-smi metric --json`; when
/// amd-smi is not installed (e.g. a shared node or container that ships no
/// amd-smi — this never tries `rocm-smi`) it falls back to the amdgpu DRM sysfs
/// VRAM counters so `--gpu auto` selection and the low-VRAM warning still have
/// telemetry to work with on a single-GPU host (see
/// [`read_drm_vram_usage`] for why the sysfs fallback withholds telemetry on a
/// multi-GPU one). Returns `None` only when neither source is readable
/// (callers then fall back to service-state-only auto-selection).
pub(crate) fn gpu_vram_usage() -> Option<Vec<GpuVramUsage>> {
    #[cfg(feature = "e2e-test-hooks")]
    if let Some(forced) = simulated_low_vram_usage() {
        return Some(forced);
    }
    gpu_vram_usage_amd_smi().or_else(gpu_vram_usage_sysfs)
}

/// E2E hook: when `ROCM_E2E_FORCE_LOW_VRAM` names a GPU ordinal, report that GPU
/// as almost entirely occupied, overriding the real telemetry.
///
/// The serve-plan low-VRAM OOM warning (and, for vLLM, its
/// `--gpu-memory-utilization` note) only fires when a *selected* GPU is below
/// [`AUTO_FREE_VRAM_FRACTION`] free. The GPU lane's real cards are comfortably
/// free, so without this seam the branch is unreachable in E2E and the note has
/// no live coverage. Synthesizing one near-full single-GPU reading makes that
/// path deterministic on the vLLM GPU lane (a non-APU host, so
/// [`vram_capacity_is_meaningful`] still holds) without touching the device the
/// engine actually launches on. Compiled only into the `e2e-test-hooks` binary;
/// production builds never see it.
#[cfg(feature = "e2e-test-hooks")]
fn simulated_low_vram_usage() -> Option<Vec<GpuVramUsage>> {
    let index: u32 = std::env::var("ROCM_E2E_FORCE_LOW_VRAM")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    // A single 24 GiB device with ~0.5 GiB free is well below the 0.90 free
    // threshold, so `gpu_low_memory_warning` fires; one GPU keeps
    // `vram_capacity_is_meaningful` honest on the non-APU vLLM lane.
    let total_mb = 24 * 1024;
    Some(vec![GpuVramUsage {
        index,
        used_mb: total_mb - 512,
        total_mb,
    }])
}

/// Per-GPU VRAM occupancy via `amd-smi metric --json`. Returns `None` when
/// amd-smi is unavailable or its output cannot be parsed.
fn gpu_vram_usage_amd_smi() -> Option<Vec<GpuVramUsage>> {
    let binary = rocm_core::resolve_amd_smi_binary();
    let output = ProcessCommand::new(&binary)
        .arg("metric")
        .arg("--json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let rows = parse_gpu_vram_usage(&value);
    if rows.is_empty() { None } else { Some(rows) }
}

/// Fallback per-GPU VRAM occupancy read straight from the amdgpu DRM sysfs
/// counters (`/sys/class/drm/card*/device/mem_info_vram_{total,used}`), used when
/// amd-smi is not installed. Linux-only; other platforms expose no such
/// interface and return `None`.
#[cfg(target_os = "linux")]
fn gpu_vram_usage_sysfs() -> Option<Vec<GpuVramUsage>> {
    let rows = read_drm_vram_usage(Path::new("/sys/class/drm"));
    if rows.is_empty() { None } else { Some(rows) }
}

#[cfg(not(target_os = "linux"))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn gpu_vram_usage_sysfs() -> Option<Vec<GpuVramUsage>> {
    None
}

/// Read per-GPU VRAM occupancy from the amdgpu DRM sysfs tree rooted at
/// `drm_dir`. Enumerates primary `card<N>` nodes (skipping connector sub-nodes
/// like `card0-DP-1`), keeps only AMD devices exposing readable
/// `mem_info_vram_{total,used}` counters, and assigns HIP ordinal 0 to the sole
/// AMD card to mirror HIP device ordering on a standard single-GPU host install.
///
/// HIP ordinals come from the KFD *compute* topology, not DRM card numbers, and
/// the two enumerations are only guaranteed to agree when there is exactly one
/// AMD DRM card: an APU passes the same AMD filter as a discrete GPU, so a host
/// pairing an APU with a dGPU can order `card<N>` differently than HIP's node
/// order, and that ordinal is fed to `HIP_VISIBLE_DEVICES` downstream. Rather
/// than guess, this returns no rows at all when more than one AMD card is
/// *found* — counted before the telemetry filters below, so a second AMD card
/// with an unreadable counter still trips the guard instead of leaving the
/// survivor mislabelled ordinal 0. Callers then fall back to service-state-only
/// auto-selection. Split from [`gpu_vram_usage_sysfs`] so it can be tested
/// against a planted sysfs layout.
#[cfg(any(target_os = "linux", test))]
fn read_drm_vram_usage(drm_dir: &Path) -> Vec<GpuVramUsage> {
    const BYTES_PER_MIB: u64 = 1024 * 1024;
    let Ok(entries) = fs::read_dir(drm_dir) else {
        return Vec::new();
    };
    // Every AMD card is counted here, *before* the telemetry filters below can
    // drop it, so the multi-card ordinal-ambiguity guard fires on how many AMD
    // cards exist rather than on how many produced readable counters.
    let mut amd_cards_found = 0usize;
    let mut cards: Vec<GpuVramUsage> = Vec::new();
    for entry in entries.flatten() {
        let card_path = entry.path();
        let Some(name) = card_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        if name
            .strip_prefix("card")
            .and_then(|digits| digits.parse::<u32>().ok())
            .is_none()
        {
            continue;
        }
        let device_dir = card_path.join("device");
        if !drm_device_is_amd(&device_dir) {
            continue;
        }
        amd_cards_found += 1;
        let Some(total_bytes) = read_sysfs_u64(&device_dir.join("mem_info_vram_total")) else {
            continue;
        };
        // An unreadable `used` counter is unknown occupancy, not zero: treating
        // it as `0` would make the card look 100% free, which is exactly what
        // `--gpu auto` prefers first. Skip the card instead of guessing.
        let Some(used_bytes) = read_sysfs_u64(&device_dir.join("mem_info_vram_used")) else {
            continue;
        };
        cards.push(GpuVramUsage {
            // The sole surviving AMD card is HIP ordinal 0 (guaranteed by the
            // multi-card guard below).
            index: 0,
            used_mb: used_bytes / BYTES_PER_MIB,
            total_mb: total_bytes / BYTES_PER_MIB,
        });
    }
    // More than one AMD card found: ascending `card<N>` order is not guaranteed
    // to match HIP's ordinal order (see doc comment above), so do not hand out
    // ordinals that might feed the wrong device into `HIP_VISIBLE_DEVICES`.
    if amd_cards_found > 1 {
        return Vec::new();
    }
    cards
}

/// Read a sysfs file whose entire contents are a single unsigned integer.
#[cfg(any(target_os = "linux", test))]
fn read_sysfs_u64(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

/// Whether a DRM `device` directory belongs to an AMD GPU.
///
/// Delegates rather than re-implementing. This probe and `rocm-core`'s KFD/DRM
/// count authority have to agree on what counts as an AMD card — if this one
/// recognises fewer, it under-counts and the multi-card ordinal guard it feeds
/// silently narrows — and a second copy of the test with a comment claiming the
/// two match is not agreement, it is a promise nothing enforces. Calling the
/// same function is.
#[cfg(any(target_os = "linux", test))]
fn drm_device_is_amd(device_dir: &Path) -> bool {
    rocm_core::is_amdgpu_device(device_dir)
}

/// Parse `amd-smi metric --json` output into per-GPU VRAM usage. Accepts both
/// the `{"gpu_data": [...]}` envelope and a bare top-level array, mirroring the
/// schema variance handled by the dashboard amd-smi collector.
///
/// A row reporting `total_vram: 0` is dropped rather than kept as a measurement
/// of an empty pool. `amd-smi` emits that shape when it enumerated the device
/// but could not read its memory controller (a driver hiccup, not a 0 GiB
/// GPU), and every other reader of this figure already treats zero as "not a
/// real reading": [`GpuVramUsage::free_fraction`] returns `None` on it rather
/// than reporting the GPU fully occupied. Keeping the row here would let
/// [`host_accelerator_memory`] construct `AcceleratorMemory::Dedicated(0.0)`,
/// which reads as "measured, and it is nothing" rather than "could not
/// measure" -- turning a diagnostic gap into a confident `Blocked` verdict.
fn parse_gpu_vram_usage(value: &serde_json::Value) -> Vec<GpuVramUsage> {
    let entries = value
        .get("gpu_data")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.as_array());
    let Some(entries) = entries else {
        return Vec::new();
    };
    entries
        .iter()
        .enumerate()
        .filter_map(|(position, entry)| {
            let index = entry
                .get("gpu")
                .and_then(serde_json::Value::as_u64)
                .map_or(position as u32, |id| id as u32);
            let used_mb = entry
                .pointer("/mem_usage/used_vram/value")
                .and_then(serde_json::Value::as_u64)?;
            let total_mb = entry
                .pointer("/mem_usage/total_vram/value")
                .and_then(serde_json::Value::as_u64)?;
            if total_mb == 0 {
                return None;
            }
            Some(GpuVramUsage {
                index,
                used_mb,
                total_mb,
            })
        })
        .collect()
}

/// Build a serve-plan warning when a selected GPU is already heavily used, so
/// the user knows to free it or pick another `--gpu` before the engine fails on
/// insufficient VRAM. Returns `None` when telemetry is missing or the selected
/// GPUs are comfortably free.
fn gpu_low_memory_warning(gpu_indices: &[u32], vram: Option<&[GpuVramUsage]>) -> Option<String> {
    let vram = vram?;
    for &index in gpu_indices {
        // Skip indices without telemetry rather than abandoning the whole scan,
        // so a missing row for one GPU does not suppress warnings for others.
        let Some(usage) = vram.iter().find(|row| row.index == index) else {
            continue;
        };
        let Some(free) = usage.free_fraction() else {
            continue;
        };
        if free < AUTO_FREE_VRAM_FRACTION {
            let free_gib = (usage.total_mb.saturating_sub(usage.used_mb)) as f64 / 1024.0;
            let total_gib = usage.total_mb as f64 / 1024.0;
            return Some(format!(
                "warning: GPU {index} has only {free_gib:.1} GiB of {total_gib:.1} GiB free; \
                 serving may fail on VRAM. Free it or pick another with `--gpu <index|auto>`."
            ));
        }
    }
    None
}

/// Whether the reported VRAM capacity describes the memory the engine will
/// actually draw on.
///
/// An APU has no private VRAM. `amd-smi` reports the fixed BIOS carveout (often
/// ~4 GiB) as `total_vram`, while the allocator serves the model out of
/// GTT-backed system RAM — so on a 128 GiB Strix Halo, "only 1.6 GiB of 4.0 GiB
/// free" is measuring the wrong pool. It fires on healthy machines and would
/// stay quiet on a genuinely starved one, so a low-memory warning built on it
/// carries no information and is better withheld.
///
/// The single-GPU condition is what keeps this honest.
/// [`rocm_core::gfx_is_apu_family`] classifies a *part*, and host detection
/// yields one target for the whole machine, so on a laptop pairing an APU with
/// a discrete card an APU verdict would otherwise suppress a legitimate warning
/// about the discrete card. When more than one GPU is present we cannot
/// attribute the target to the selected ordinal, so the warning stands.
pub(crate) fn vram_capacity_is_meaningful(gfx_target: Option<&str>, gpu_count: usize) -> bool {
    if gpu_count != 1 {
        return true;
    }
    !gfx_target.is_some_and(rocm_core::gfx_is_apu_family)
}

/// The serve plan's low-VRAM warning, withheld on hosts where the VRAM figures
/// do not describe the memory the engine uses (see
/// [`vram_capacity_is_meaningful`]).
///
/// Resolves the gfx target only once a warning would actually be emitted. Host
/// GPU detection can probe sysfs/WSL — `serve` deliberately gates its own call
/// on whether engine selection needs it — so the quiet path must not start
/// paying for a lookup it will never use.
pub(crate) fn serve_gpu_low_memory_warning(
    gpu_indices: &[u32],
    vram: Option<&[GpuVramUsage]>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
) -> Option<String> {
    let warning = gpu_low_memory_warning(gpu_indices, vram)?;
    let gpu_count = vram.map_or(0, <[GpuVramUsage]>::len);
    let gfx_target = match host_gpu_summary {
        Some(summary) => summary.gfx_target.clone(),
        None => rocm_core::detect_host_gfx_target(),
    };
    vram_capacity_is_meaningful(gfx_target.as_deref(), gpu_count).then_some(warning)
}

/// GPU ordinals currently pinned by running rocm-cli managed/foreground
/// services, used to skip busy devices during `--gpu auto` selection.
fn busy_gpu_indices(paths: &AppPaths) -> Vec<u32> {
    let Ok(records) = load_managed_services(paths) else {
        return Vec::new();
    };
    let mut busy = Vec::new();
    for record in records {
        if !matches!(
            record.status.as_str(),
            "starting" | "running" | "recovering" | "ready"
        ) {
            continue;
        }
        for index in record.gpu_indices {
            if !busy.contains(&index) {
                busy.push(index);
            }
        }
    }
    busy
}

/// Best-effort count of local AMD GPUs via `amd-smi`. Returns `None` when
/// amd-smi is unavailable or its output cannot be parsed (callers then fall
/// back to conservative defaults).
pub(crate) fn detect_gpu_count() -> Option<usize> {
    let binary = rocm_core::resolve_amd_smi_binary();
    let output = ProcessCommand::new(&binary)
        .arg("list")
        .arg("--json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let count = value.as_array().map(Vec::len)?;
    if count == 0 { None } else { Some(count) }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rocm_core::ManagedServiceRecord;
    use serde_json::json;

    use super::*;
    use crate::DevicePolicyArg;
    #[cfg(feature = "e2e-test-hooks")]
    use crate::tests::ScopedTestEnv;
    use crate::tests::test_paths;

    /// Persist a live-looking managed record claiming `gpu` — the same shape a
    /// real launch writes, with the current process id as the supervisor so the
    /// liveness refresh in `load_managed_services` keeps it "starting" (and thus
    /// counted by `busy_gpu_indices`).
    fn write_claiming_record(paths: &AppPaths, service_id: &str, port: u16, gpu: &[u32]) {
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "starting".to_owned();
        record.gpu_indices = gpu.to_vec();
        record.write().expect("write claiming record");
    }

    #[test]
    fn launch_lock_makes_gpu_select_and_claim_atomic() {
        // Regression for the serve read-select-launch race: the busy-GPU read and
        // the claiming record write must happen under one lock, or two concurrent
        // `--gpu auto` serves both read the same GPU as free and land on it.
        //
        // The test does NOT take the lock itself — that would only prove
        // `FileLock` excludes (already covered by
        // `file_lock_serializes_concurrent_holders` in rocm-core). It calls
        // `select_gpu_indices_under_launch_lock`, the production helper `serve()`
        // uses, whose contract is that it returns the guard *it* acquired together
        // with the selection; the test holds that guard across the claim exactly
        // as `serve()` holds it until `spawn_managed_engine_child` persists the
        // record. Delete the `FileLock::acquire` from that helper and this test
        // goes red: both threads then select GPU 0.
        //
        // Determinism: the barrier releases both threads together and each sleeps
        // between select and claim, so an unlocked helper double-books GPU 0
        // regardless of scheduling skew, while the locked helper forces the second
        // thread to observe the first thread's claim.
        let (root, paths) = test_paths("launch-lock-atomic-claim");
        paths.ensure().expect("prepare paths");
        let detected = Some(2_usize);

        let barrier = std::sync::Barrier::new(2);
        let selections = std::thread::scope(|scope| {
            let handles: Vec<_> = [("svc-race-a", 21001_u16), ("svc-race-b", 21002_u16)]
                .into_iter()
                .map(|(service_id, port)| {
                    let paths = &paths;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        // The exact call `serve()` makes: the helper acquires the
                        // launch lock and selects under it, handing the guard back.
                        // `None` visibility keeps selection mask-unaware for the
                        // test host; `pinned` `None` + `cpu_only` false is the
                        // `--gpu auto` path that reads live busy-GPU state.
                        let (gpu, lock) = select_gpu_indices_under_launch_lock(
                            paths,
                            false,
                            None,
                            || detected,
                            None,
                            None,
                        )
                        .expect("auto GPU selection under launch lock");
                        // Widen the select→claim window so an unlocked helper
                        // deterministically double-books GPU 0; under the lock the
                        // second thread cannot enter until we claim.
                        std::thread::sleep(Duration::from_millis(50));
                        write_claiming_record(paths, service_id, port, &gpu);
                        drop(lock);
                        gpu
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("selection thread joins"))
                .collect::<Vec<_>>()
        });

        let mut picked: Vec<u32> = selections.into_iter().flatten().collect();
        picked.sort_unstable();
        assert_eq!(
            picked,
            vec![0, 1],
            "serialized select-then-claim must hand out distinct GPUs, got {picked:?}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_device_policy_defaults_to_gpu_required_without_cpu_fallback() -> Result<()> {
        assert_eq!(parse_device_policy(None)?, DevicePolicy::GpuRequired);
        assert_eq!(parse_device_policy(Some("gpu"))?, DevicePolicy::GpuRequired);
        assert_eq!(
            parse_device_policy(Some("gpu_preferred"))?,
            DevicePolicy::GpuRequired
        );
        let cpu = parse_device_policy(Some("cpu")).unwrap_err().to_string();
        assert!(cpu.contains("CPU mode is not a fallback path"));
        Ok(())
    }

    fn vram(index: u32, used_mb: u64, total_mb: u64) -> GpuVramUsage {
        GpuVramUsage {
            index,
            used_mb,
            total_mb,
        }
    }

    #[test]
    fn auto_selection_prefers_lowest_index_idle_gpu() {
        // GPU 0 busy (only 5% free), GPU 1 idle, GPU 2 idle.
        let usage = [
            vram(0, 182_000, 192_000),
            vram(1, 1_000, 192_000),
            vram(2, 500, 192_000),
        ];
        assert_eq!(
            select_auto_gpu_index(Some(3), None, &[], Some(&usage)),
            vec![1],
            "should skip the busy GPU 0 and pick the lowest idle GPU"
        );
    }

    #[test]
    fn auto_selection_skips_managed_and_busy_gpus_then_picks_most_free() {
        // GPU 0 pinned by a managed service; GPU 1 partly used; GPU 2 more free
        // but none is fully idle, so pass 2 (most free) applies.
        let usage = [
            vram(0, 10_000, 192_000),
            vram(1, 120_000, 192_000),
            vram(2, 60_000, 192_000),
        ];
        assert_eq!(
            select_auto_gpu_index(Some(3), None, &[0], Some(&usage)),
            vec![2],
            "with no idle GPU, pick the non-busy GPU with the most free VRAM"
        );
    }

    #[test]
    fn auto_selection_pass_two_ranks_by_absolute_free_vram() {
        // Heterogeneous VRAM with no fully-idle GPU (so pass 2 applies):
        // GPU 0 is a small card with a high free *fraction* (75%) but little
        // absolute free memory; GPU 1 is large with a lower fraction (~48%)
        // but far more free memory. Auto-selection must prefer GPU 1.
        let usage = [vram(0, 6_000, 24_000), vram(1, 100_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(Some(2), None, &[], Some(&usage)),
            vec![1],
            "pass 2 should rank by absolute free VRAM, not free percentage"
        );
    }

    #[test]
    fn auto_selection_falls_back_to_first_non_busy_without_vram() {
        assert_eq!(select_auto_gpu_index(Some(4), None, &[0, 1], None), vec![2]);
        // Unknown GPU count: no GPU-0 fallback — defer to the engine device probe.
        assert_eq!(
            select_auto_gpu_index(None, None, &[], None),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn auto_selection_restricts_candidates_to_visible_set() {
        // GPUs 0 and 1 are masked out (only 2 and 3 visible). With no VRAM data
        // auto-select must pick the lowest *visible* ordinal, never a hidden one.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[2, 3]), &[], None),
            vec![2],
            "selection must stay inside the visibility mask"
        );
        // The lowest visible GPU (2) is busy, so fall through to the next visible.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[2, 3]), &[2], None),
            vec![3]
        );
    }

    #[test]
    fn auto_selection_all_visible_busy_falls_back_to_lowest_visible_not_zero() {
        // Every visible GPU is pinned; the fallback must be the lowest *visible*
        // ordinal (2), not a hardcoded 0 that the mask has hidden.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[2, 3]), &[2, 3], None),
            vec![2]
        );
    }

    #[test]
    fn auto_selection_returns_none_when_every_device_masked_out() {
        // An empty visible set means no candidate survives the mask: return no
        // selection rather than assuming device 0.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[]), &[], None),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn auto_selection_uses_vram_row_count_when_amd_smi_count_is_unknown() {
        // `detect_gpu_count()` is `None` while VRAM telemetry is present. With
        // *several* rows that is the amd-smi shape of the state -- `metric`
        // answered while `list` did not -- not the DRM sysfs fallback, which
        // withholds telemetry entirely once a second AMD card is found and so
        // can never produce more than one row (`read_drm_vram_usage`). Either
        // way auto-selection must derive the count from the rows and skip the
        // busy GPU 0 rather than returning no selection (which would land the
        // engine on GPU 0).
        let usage = [
            vram(0, 182_000, 192_000),
            vram(1, 1_000, 192_000),
            vram(2, 500, 192_000),
        ];
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&usage)),
            vec![1],
            "with no amd-smi count, rank the reported VRAM rows and pick the first idle GPU"
        );
        // The state the DRM sysfs fallback can actually reach: exactly one row.
        // It is still ranked (and still selected) rather than discarded for
        // being a single device.
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&[vram(0, 1_000, 192_000)])),
            vec![0],
            "the single row the sysfs fallback can emit must still be selectable"
        );
        // Still nothing to go on when neither the count nor telemetry is present.
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&[])),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn auto_selection_ranks_the_actual_reported_row_indices() {
        // A visibility mask hides GPUs 0 and 1, so amd-smi metric reports only
        // the surviving devices with their absolute, non-contiguous ordinals
        // [2, 3] -- there is no row at index 0 or 1. Auto-selection must scan
        // those real indices, not a synthetic `0..count` range (which would
        // look up absent rows and fall through to a bogus GPU 0).
        let usage = [vram(2, 182_000, 192_000), vram(3, 1_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&usage)),
            vec![3],
            "should skip the busy GPU 2 and pick the idle GPU 3 by its real ordinal"
        );
        // With GPU 3 also pinned by a managed service, pass 3 still returns a
        // real reported ordinal (GPU 2), never a fabricated index 0.
        assert_eq!(
            select_auto_gpu_index(None, None, &[3], Some(&usage)),
            vec![2],
            "the sole non-busy reported GPU must be selected by its real ordinal"
        );
    }

    #[test]
    fn auto_selection_all_reported_busy_falls_back_to_a_reported_ordinal() {
        // Same sparse-ordinal host as above ([2, 3] reported, nothing at 0 or 1)
        // but now BOTH reported GPUs are pinned by a running service, so every
        // pass falls through to the terminal "all busy, pick one anyway"
        // fallback. That fallback used to hand back a hardcoded `0` — safe only
        // while candidates were a dense `0..count`, and simply wrong once they
        // come from the reported rows: index 0 is a device nothing reported, yet
        // it would be exported verbatim as `HIP_VISIBLE_DEVICES`. It must return
        // the lowest ordinal that was actually reported instead.
        let usage = [vram(2, 182_000, 192_000), vram(3, 190_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(None, None, &[2, 3], Some(&usage)),
            vec![2],
            "the all-busy fallback must name a reported GPU, never a fabricated index 0"
        );
        // "Lowest reported" has to mean lowest by ordinal, not first in the
        // rows: `amd-smi` orders its output by enumeration, not by index, so
        // feed the same two rows in descending order. Varying the *busy* slice's
        // order instead would prove nothing — it is only ever read through
        // `.contains()`, so that assertion was a duplicate of the one above.
        let descending = [vram(3, 190_000, 192_000), vram(2, 182_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(None, None, &[2, 3], Some(&descending)),
            vec![2],
            "the fallback must take the lowest reported ordinal, not the first row"
        );
    }

    #[test]
    fn auto_selection_considers_a_detected_gpu_that_reported_no_vram_row() {
        // A short row set is not always a visibility mask. `parse_gpu_vram_usage`
        // drops any device entry missing its `used_vram`/`total_vram` pointers, so
        // here `list` counts two GPUs, both are visible, but only GPU 0 produced a
        // row — and GPU 0 is pinned by a managed service. Driving candidates from
        // the rows alone leaves `[0]`, the busy filter empties it, every pass
        // iterates nothing and the terminal fallback hands back GPU 0: `serve`
        // pinned to an already-occupied GPU, the exact failure this selection
        // exists to prevent. The untelemetried GPU 1 is confirmed by both the
        // count and the visible set, so it must be a candidate and must win.
        assert_eq!(
            select_auto_gpu_index(
                Some(2),
                Some(&[0, 1]),
                &[0],
                Some(&[vram(0, 182_000, 192_000)])
            ),
            vec![1],
            "a detected, visible GPU with no VRAM row must be preferred over a busy reported one"
        );
        // The same shape without telemetry for the busy device being conclusive:
        // GPU 1 is still the only non-busy ordinal either source confirms.
        assert_eq!(
            select_auto_gpu_index(
                Some(2),
                Some(&[0, 1]),
                &[0],
                Some(&[vram(0, 1_000, 192_000)])
            ),
            vec![1],
            "an idle-looking but service-pinned GPU 0 must still lose to the free GPU 1"
        );
        // The masking behaviour this rows-only construction was introduced for is
        // untouched: rows `[2, 3]` against `count == 2` put a row index at or above
        // the count, which is the proof the ordinal space is re-indexed, so the
        // union never happens and the rows stand alone. (The visible retain would
        // also have dropped `0`/`1` here — this pins the row-index rule itself, so
        // the masked case survives on a host where the mask is unknown too.)
        let masked = [vram(2, 182_000, 192_000), vram(3, 190_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(Some(2), Some(&[2, 3]), &[2, 3], Some(&masked)),
            vec![2],
            "a masked host must not gain candidates 0/1 from the detected count"
        );
    }

    #[test]
    fn auto_selection_considers_an_untelemetried_gpu_when_the_visible_set_is_unknown() {
        // Same shape as the test above, but `visible` is `None`. That is not an
        // exotic host: `probe_usable_amd_gpu_indices` returns `None`
        // *unconditionally* off Linux, and `serve` threads that straight into
        // `visible`, so every Windows run takes this path — as does any Linux host
        // whose KFD topology and DRM cards are both unreadable. Gating the
        // `0..count` union on `visible.is_some()` therefore left the whole bug
        // standing on a supported platform.
        //
        // Nothing about the mask is needed to justify GPU 1 here: the amd-smi
        // `list` count asserts two devices and the rows name only ordinals below
        // that count, so the rows are a subset of a dense range, not a re-indexed
        // one. GPU 0 is pinned by a managed service, so the untelemetried GPU 1 is
        // the only free ordinal and must win over the busy reported fallback.
        assert_eq!(
            select_auto_gpu_index(Some(2), None, &[0], Some(&[vram(0, 182_000, 192_000)])),
            vec![1],
            "an unprobeable host must still consider a counted GPU that reported no VRAM row"
        );
        // A row index at or above the count is the only evidence available that the
        // ordinal space is *not* a dense `0..count`, so there the rows stay
        // authoritative even with no mask to retain against — otherwise an
        // unprobeable masked host would have `[0, 1]` invented for it.
        let masked = [vram(2, 182_000, 192_000), vram(3, 190_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(Some(2), None, &[2, 3], Some(&masked)),
            vec![2],
            "re-indexed rows must not gain unconfirmed candidates 0/1 from the detected count"
        );
        // The count must come from the `list` enumeration to confirm anything.
        // With `detected` absent it is `rows.len()`, i.e. the rows restating their
        // own size, and `parse_gpu_vram_usage` can repeat an ordinal: it falls back
        // to the array position only when an entry has no `gpu` field, so a payload
        // naming `"gpu": 0` twice yields two index-0 rows. That makes `rows.len()`
        // 2 with every index below it, and dropping the `detected` half of the
        // condition would invent a GPU 1 nothing ever enumerated.
        assert_eq!(
            select_auto_gpu_index(
                None,
                None,
                &[0],
                Some(&[vram(0, 182_000, 192_000), vram(0, 182_000, 192_000)])
            ),
            vec![0],
            "a row-derived count must not synthesise an ordinal no source reported"
        );
    }

    #[test]
    fn validate_pinned_gpu_index_rejects_out_of_range() {
        // Index equal to or beyond the detected count is rejected.
        let error = validate_pinned_gpu_index(4, Some(4), None, false)
            .expect_err("index 4 is out of range");
        assert!(error.to_string().contains("out of range"));
        assert!(validate_pinned_gpu_index(9, Some(2), None, false).is_err());
    }

    #[test]
    fn validate_pinned_gpu_index_accepts_in_range_or_unknown_count() {
        // In-range index pins exactly that ordinal.
        assert_eq!(
            validate_pinned_gpu_index(0, Some(1), None, false).unwrap(),
            vec![0]
        );
        assert_eq!(
            validate_pinned_gpu_index(3, Some(4), None, false).unwrap(),
            vec![3]
        );
        // Unknown count (amd-smi unavailable) is allowed through unvalidated.
        assert_eq!(
            validate_pinned_gpu_index(7, None, None, false).unwrap(),
            vec![7]
        );
    }

    #[test]
    fn effective_gpu_count_prefers_amd_smi_then_falls_back_to_vram_rows() {
        let usage = [vram(0, 1_000, 192_000), vram(1, 1_000, 192_000)];
        // amd-smi's count wins when it is known, even over a differing row count.
        assert_eq!(effective_gpu_count(Some(4), Some(&usage)), Some(4));
        // amd-smi unavailable (`None`): the sysfs VRAM fallback row count stands in.
        assert_eq!(effective_gpu_count(None, Some(&usage)), Some(2));
        // The second, previously untested fallback trigger: amd-smi ran and
        // reported zero devices. `Some(0)` must fall through to the VRAM rows
        // just as `None` does, or a host amd-smi cannot enumerate but sysfs can
        // would have nothing for `--gpu auto` to rank. Dropping the
        // `filter(|&count| count > 0)` on `detected` turns this line red.
        assert_eq!(effective_gpu_count(Some(0), Some(&usage)), Some(2));
        // Neither source available.
        assert_eq!(effective_gpu_count(None, None), None);
        assert_eq!(effective_gpu_count(None, Some(&[])), None);
        // Both present but both empty: still nothing to rank.
        assert_eq!(effective_gpu_count(Some(0), Some(&[])), None);
        assert_eq!(effective_gpu_count(Some(0), None), None);
    }

    #[test]
    fn validate_pinned_gpu_index_rejects_masked_out_device() {
        // The visible set is authoritative: an index outside it is rejected early
        // with the usable set, rather than deferring to a late engine error. A
        // mask is set here, so the message names the visibility variables.
        let error = validate_pinned_gpu_index(0, Some(4), Some(&[2, 3]), true)
            .expect_err("masked-out device must be rejected");
        let message = error.to_string();
        assert!(message.contains("not available under the active visibility mask"));
        assert!(message.contains("[2, 3]"));
        // A visible index still pins exactly that ordinal.
        assert_eq!(
            validate_pinned_gpu_index(2, Some(4), Some(&[2, 3]), true).unwrap(),
            vec![2]
        );
        // An empty visible set is authoritative "no GPU usable" (every device
        // masked out), not "could not enumerate" — the latter is `None`, per the
        // `usable_amd_gpu_indices` contract. So it rejects rather than falling
        // through to the count check; the detected count must not override it.
        let masked_out = validate_pinned_gpu_index(1, Some(4), Some(&[]), true)
            .expect_err("an all-masked host must reject an explicit --gpu index");
        assert!(
            masked_out
                .to_string()
                .contains("no usable AMD GPU is available")
        );
    }

    #[test]
    fn validate_pinned_gpu_index_absent_index_without_mask_reads_as_not_present() {
        // No visibility mask is set, yet amd-smi counts more devices than the
        // KFD/DRM probe finds, so the authoritative visible set (0..present) is
        // shorter than `detected`. An index in that gap is genuinely absent from
        // the host — the rejection must say so and must NOT blame HIP/ROCR
        // variables the user never set (that message sends them debugging a mask
        // that does not exist).
        let error = validate_pinned_gpu_index(2, Some(4), Some(&[0, 1]), false)
            .expect_err("an index past the visible set must be refused");
        let message = error.to_string();
        assert!(
            message.contains("not present on this host"),
            "unmasked rejection should read as not-present, got: {message}"
        );
        assert!(
            !message.contains("visibility mask"),
            "must not blame a mask when none is set, got: {message}"
        );
        assert!(message.contains("[0, 1]"));

        // The all-empty visible set with no mask is a not-present message too,
        // never a spurious mask advisory.
        let empty = validate_pinned_gpu_index(0, Some(2), Some(&[]), false)
            .expect_err("no present GPU must refuse an explicit --gpu index");
        assert!(empty.to_string().contains("present on this host"));
        assert!(!empty.to_string().contains("visibility mask"));
    }

    #[test]
    fn validate_pinned_gpu_index_prefers_visible_set_over_detected_count() {
        // detected and visible come from different probes. When both are known the
        // visible set wins, so an index inside the detected count but outside the
        // HIP-visible set is still rejected (and vice versa), instead of the two
        // being compared as one ordinal space.
        assert_eq!(
            validate_pinned_gpu_index(1, Some(4), Some(&[0, 1]), true).unwrap(),
            vec![1]
        );
        assert!(
            validate_pinned_gpu_index(2, Some(4), Some(&[0, 1]), true).is_err(),
            "index within the detected count but outside the visible set must reject"
        );
    }

    #[test]
    fn parse_gpu_vram_usage_reads_gpu_data_envelope() {
        let value = json!({
            "gpu_data": [
                {"gpu": 0, "mem_usage": {"used_vram": {"value": 1000}, "total_vram": {"value": 192_000}}},
                {"gpu": 1, "mem_usage": {"used_vram": {"value": 50000}, "total_vram": {"value": 192_000}}}
            ]
        });
        let rows = parse_gpu_vram_usage(&value);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].index, 0);
        assert_eq!(rows[0].used_mb, 1000);
        assert_eq!(rows[1].index, 1);
        assert!((rows[1].free_fraction().unwrap() - (142_000.0 / 192_000.0)).abs() < 1e-9);
    }

    #[test]
    fn read_drm_vram_usage_reads_a_single_amdgpu_card_and_assigns_ordinal_zero() {
        // Plant a minimal `/sys/class/drm`-shaped tree: one AMD card, a connector
        // sub-node that must be skipped, and a non-AMD card that must be ignored.
        let root = std::env::temp_dir().join(format!("rocm-drm-vram-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let plant_amd = |card: &str, total: u64, used: u64| {
            let device = root.join(card).join("device");
            fs::create_dir_all(&device).unwrap();
            fs::write(device.join("vendor"), "0x1002\n").unwrap();
            fs::write(device.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
            fs::write(device.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
        };
        // 192 GiB total, mostly free (values in bytes).
        plant_amd("card0", 206_158_430_208, 1_073_741_824);
        // A connector sub-node under card0 — must be skipped, not parsed as a card.
        fs::create_dir_all(root.join("card0-DP-1")).unwrap();
        // A non-AMD primary card — different vendor, must be ignored.
        let intel = root.join("card1").join("device");
        fs::create_dir_all(&intel).unwrap();
        fs::write(intel.join("vendor"), "0x8086\n").unwrap();
        fs::write(intel.join("mem_info_vram_total"), "1000000\n").unwrap();

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert_eq!(rows.len(), 1, "only the one AMD card counts: {rows:?}");
        assert_eq!(rows[0].index, 0);
        // 206_158_430_208 bytes / 1 MiB == 196_608 MiB.
        assert_eq!(rows[0].total_mb, 196_608);
        assert!(rows[0].free_fraction().unwrap() > AUTO_FREE_VRAM_FRACTION);
    }

    #[test]
    fn read_drm_vram_usage_withholds_telemetry_when_multiple_amd_cards_are_present() {
        // Ascending `card<N>` order is not guaranteed to match HIP's compute
        // ordinal once more than one AMD card exists (e.g. an APU alongside a
        // dGPU), so the fallback must not hand out ordinals it cannot vouch for.
        let root = std::env::temp_dir().join(format!("rocm-drm-vram-multi-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let plant_amd = |card: &str, total: u64, used: u64| {
            let device = root.join(card).join("device");
            fs::create_dir_all(&device).unwrap();
            fs::write(device.join("vendor"), "0x1002\n").unwrap();
            fs::write(device.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
            fs::write(device.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
        };
        plant_amd("card0", 206_158_430_208, 1_073_741_824);
        plant_amd("card1", 206_158_430_208, 189_284_651_008);

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            rows.is_empty(),
            "multi-card sysfs telemetry must be withheld: {rows:?}"
        );
    }

    #[test]
    fn read_drm_vram_usage_withholds_telemetry_when_a_second_amd_card_has_no_counter() {
        // The guard must count AMD cards *found*, not surviving rows: a genuine
        // two-AMD-card host where the second card's `used` counter is unreadable
        // would otherwise slip past a `rows.len() > 1` check and mislabel the
        // survivor ordinal 0 — the exact APU+dGPU misattribution the guard exists
        // to prevent.
        let root =
            std::env::temp_dir().join(format!("rocm-drm-vram-partial-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        // card0: fully readable.
        let card0 = root.join("card0").join("device");
        fs::create_dir_all(&card0).unwrap();
        fs::write(card0.join("vendor"), "0x1002\n").unwrap();
        fs::write(card0.join("mem_info_vram_total"), "206158430208\n").unwrap();
        fs::write(card0.join("mem_info_vram_used"), "1073741824\n").unwrap();
        // card1: AMD, but its `used` counter cannot be read.
        let card1 = root.join("card1").join("device");
        fs::create_dir_all(&card1).unwrap();
        fs::write(card1.join("vendor"), "0x1002\n").unwrap();
        fs::write(card1.join("mem_info_vram_total"), "206158430208\n").unwrap();

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            rows.is_empty(),
            "a second AMD card must trip the guard even without readable counters: {rows:?}"
        );
    }

    #[test]
    fn read_drm_vram_usage_skips_a_card_with_an_unreadable_used_counter() {
        // A card whose `mem_info_vram_used` cannot be read must not be treated as
        // 0 bytes used (100% free) — that would make a broken counter look like
        // the ideal `--gpu auto` pick. It should be skipped entirely instead.
        let root =
            std::env::temp_dir().join(format!("rocm-drm-vram-unreadable-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let device = root.join("card0").join("device");
        fs::create_dir_all(&device).unwrap();
        fs::write(device.join("vendor"), "0x1002\n").unwrap();
        fs::write(device.join("mem_info_vram_total"), "206158430208\n").unwrap();
        // No `mem_info_vram_used` file written at all.

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            rows.is_empty(),
            "a card with no readable used counter must be skipped: {rows:?}"
        );
    }

    #[test]
    fn read_drm_vram_usage_is_empty_without_a_drm_tree() {
        let missing = std::env::temp_dir().join("rocm-drm-vram-absent-98765");
        let _ = fs::remove_dir_all(&missing);
        assert!(read_drm_vram_usage(&missing).is_empty());
    }

    #[test]
    fn gpu_low_memory_warning_flags_busy_selected_gpu() {
        let usage = [vram(0, 182_000, 192_000), vram(1, 1_000, 192_000)];
        let warning = gpu_low_memory_warning(&[0], Some(&usage)).expect("warning for busy GPU 0");
        assert!(warning.contains("GPU 0"));
        assert!(warning.contains("free"));
        assert!(gpu_low_memory_warning(&[1], Some(&usage)).is_none());
        assert!(gpu_low_memory_warning(&[0], None).is_none());
    }

    fn host_gpu(gfx_target: &str) -> rocm_core::HostGpuSummary {
        rocm_core::HostGpuSummary {
            gfx_target: Some(gfx_target.to_owned()),
            ..rocm_core::HostGpuSummary::default()
        }
    }

    #[test]
    fn unified_memory_apu_suppresses_the_vram_capacity_warning() {
        // Strix Halo as reported by amd-smi: a 4 GiB BIOS carveout with 1.6 GiB
        // free (40%, under the 90% bar) on a machine whose engine actually
        // serves out of ~128 GiB of shared system RAM. The old reading —
        // "only 1.6 GiB of 4.0 GiB free" — describes a pool the allocator does
        // not use, so it must not reach the user.
        let carveout = [vram(0, 2_458, 4_096)];
        assert!(
            gpu_low_memory_warning(&[0], Some(&carveout)).is_some(),
            "the underlying threshold still trips; only the serve-plan wrapper withholds it"
        );
        assert_eq!(
            serve_gpu_low_memory_warning(&[0], Some(&carveout), Some(&host_gpu("gfx1151"))),
            None
        );
    }

    #[test]
    fn discrete_gpu_still_warns_when_genuinely_busy() {
        let busy = [vram(0, 182_000, 192_000)];
        let warning = serve_gpu_low_memory_warning(&[0], Some(&busy), Some(&host_gpu("gfx1100")))
            .expect("a discrete card that is 95% full still warrants a warning");
        assert!(warning.contains("GPU 0"));
        assert!(warning.contains("`--gpu <index|auto>`"));

        // Baseline: the same discrete card, mostly free, stays quiet.
        let idle = [vram(0, 1_000, 192_000)];
        assert_eq!(
            serve_gpu_low_memory_warning(&[0], Some(&idle), Some(&host_gpu("gfx1100"))),
            None
        );
    }

    #[test]
    fn apu_verdict_does_not_silence_a_second_gpu() {
        // `gfx_is_apu_family` classifies a part, while host detection reports one
        // target for the whole machine. On an APU+dGPU laptop that target cannot
        // be attributed to the selected ordinal, so the warning must survive
        // rather than be suppressed on the discrete card's behalf.
        let hybrid = [vram(0, 2_458, 4_096), vram(1, 182_000, 192_000)];
        assert!(
            serve_gpu_low_memory_warning(&[1], Some(&hybrid), Some(&host_gpu("gfx1151"))).is_some()
        );
        assert!(vram_capacity_is_meaningful(Some("gfx1151"), 2));
        // An unknown target is never treated as unified memory.
        assert!(vram_capacity_is_meaningful(None, 1));
    }

    /// The `ROCM_E2E_FORCE_LOW_VRAM` hook must synthesize a reading that actually
    /// trips the serve-plan low-VRAM warning on a non-APU (vLLM-lane) host, and
    /// stay inert when the var is unset. This is what the `@requires-gpu`
    /// `serve-vllm-low-vram-oom-guidance` scenario relies on to fire the note
    /// deterministically on cards that are really free.
    #[cfg(feature = "e2e-test-hooks")]
    #[test]
    fn forced_low_vram_hook_trips_the_serve_plan_warning() {
        let mut env = ScopedTestEnv::new();

        // Unset: the hook contributes nothing and real telemetry is consulted.
        env.clear("ROCM_E2E_FORCE_LOW_VRAM");
        assert!(
            simulated_low_vram_usage().is_none(),
            "no override without the env var"
        );

        // Set to ordinal 0: a single near-full device that the serve-plan wrapper
        // reports on a discrete (non-APU) host but that stays honest there.
        env.set("ROCM_E2E_FORCE_LOW_VRAM", "0");
        let forced = simulated_low_vram_usage().expect("override present when the var is set");
        assert_eq!(
            forced.len(),
            1,
            "single synthetic GPU keeps the APU guard honest"
        );
        assert_eq!(forced[0].index, 0);
        assert!(
            forced[0]
                .free_fraction()
                .is_some_and(|f| f < AUTO_FREE_VRAM_FRACTION),
            "the synthetic reading must be below the free-VRAM bar so the warning fires"
        );
        let warning = serve_gpu_low_memory_warning(&[0], Some(&forced), Some(&host_gpu("gfx1100")))
            .expect("a near-full discrete card warrants the serve-plan warning");
        assert!(warning.contains("GPU 0"));
    }

    /// A row reporting `total_vram: 0` is not a measurement of an empty GPU; it
    /// is `amd-smi` enumerating a device it could not read the memory
    /// controller for. Keeping the row would let `host_accelerator_memory`
    /// report `AcceleratorMemory::Dedicated(0.0)`, which the readiness verdict
    /// then treats as "measured, and it is nothing" (`Blocked`) rather than
    /// "could not measure" (`Undetermined`).
    #[test]
    fn parse_gpu_vram_usage_drops_a_zero_total_row() {
        let value = json!({
            "gpu_data": [
                {
                    "gpu": 0,
                    "mem_usage": {
                        "used_vram": {"value": 0},
                        "total_vram": {"value": 0},
                    },
                },
                {
                    "gpu": 1,
                    "mem_usage": {
                        "used_vram": {"value": 512},
                        "total_vram": {"value": 16384},
                    },
                },
            ],
        });

        let rows = parse_gpu_vram_usage(&value);

        assert_eq!(
            rows,
            vec![GpuVramUsage {
                index: 1,
                used_mb: 512,
                total_mb: 16384,
            }],
            "a zero-total row must be dropped, not kept as a 0 GiB measurement: {rows:?}"
        );
    }

    /// All rows reporting a zero total collapses to no telemetry at all, so
    /// `gpu_vram_usage()`'s own `is_empty` check turns it into `None` --
    /// "could not measure" -- rather than a `Vec` of one unusable row.
    #[test]
    fn parse_gpu_vram_usage_returns_nothing_when_every_row_is_zero_total() {
        let value = json!({
            "gpu_data": [{
                "gpu": 0,
                "mem_usage": {
                    "used_vram": {"value": 0},
                    "total_vram": {"value": 0},
                },
            }],
        });

        assert!(parse_gpu_vram_usage(&value).is_empty());
    }

    #[test]
    fn device_policy_arg_maps_through_parse_device_policy() {
        // `as_policy_str` feeds the parsed variant back through
        // `parse_device_policy`, the single choke point enforcing the strict
        // no-CPU policy. Guard the mapping so a mis-typed arm (e.g. `CpuOnly =>
        // "gpu_required"`) cannot silently turn the deliberate CPU rejection into
        // a GPU-required serve while every other test stays green.
        assert_eq!(
            parse_device_policy(Some(DevicePolicyArg::GpuRequired.as_policy_str()))
                .expect("gpu_required parses"),
            DevicePolicy::GpuRequired
        );
        assert_eq!(
            parse_device_policy(Some(DevicePolicyArg::GpuPreferred.as_policy_str()))
                .expect("gpu_preferred parses"),
            DevicePolicy::GpuRequired
        );
        // `cpu_only` must still be rejected outright rather than mapped to a
        // GPU policy.
        assert!(
            parse_device_policy(Some(DevicePolicyArg::CpuOnly.as_policy_str())).is_err(),
            "cpu_only must be rejected"
        );
    }
}
