// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which TheRock package family a request targets: resolving it from an
//! explicit `--family`, the environment, a managed runtime or the host GPU, the
//! refusals for a GPU that has no ROCm build, recovering the exact arch behind a
//! grouped family, and the recovery hint appended when no runtime is found.
//!
//! Builds on the `source` types; nothing in `source` depends on this module.

use super::source::{FamilyResolution, TheRockChannel};
use anyhow::{Result, bail};
use rocm_core::{
    AppPaths, GpuSupportLevel, VULKAN_ONLY_SUPPORT_CLAIM, VULKAN_UNCONFIRMED_CLAIM,
    VULKAN_UNCONFIRMED_NEXT_STEP, VulkanStatus, detect_host_gfx_target,
    detect_host_gpu_diagnostics, detect_managed_therock_family, detect_vulkan_support,
    gpu_support_level, known_therock_families, normalize_therock_family, runtime_is_windows,
};
use std::fmt::Write as _;

/// The family override to actually resolve a next-layout wheel request with,
/// given a (possibly grouped) family and an exact arch recorded or requested
/// alongside it.
///
/// A grouped family (e.g. `gfx125X-dcgpu`) carries no exact arch, so passing
/// it straight through would leave the next layout's device target
/// undetermined and the whole resolve would bail — on the very host the
/// runtime is already installed on. `candidate_arch` is the exact arch a more
/// authoritative source already recorded for this same family; when it
/// agrees, pass it as the override itself, which `resolve_family` also
/// normalizes back to this same family. Used by both halves of updating a
/// next-layout manifest: `resolve_latest_for_manifest` (planning) and
/// `install_wheel_runtime` via `device_target_override` (applying) — a fix to
/// one without the other leaves the update path half-working.
pub(super) fn family_override_or_recovered_arch(
    family: &str,
    candidate_arch: Option<&str>,
) -> String {
    raw_arch_agreeing_with_family(candidate_arch.map(str::to_owned), family)
        .unwrap_or_else(|| family.to_owned())
}

/// Whether `value` is an exact GFX arch code (`gfx1200`, `gfx90a`) rather than
/// a grouped family label (`gfx120X-all`) or prose with a gfx token in it.
///
/// Deliberately stricter than [`extract_first_gfx_token`], which digs a target
/// out of noisy KFD output: a `--family` value is only evidence of an exact arch
/// when the user typed exactly one, and `gfx120X-all` normalizing to a family
/// must not be mistaken for naming a member of it.
///
/// [`extract_first_gfx_token`]: rocm_core::extract_first_gfx_token
fn is_raw_gfx_arch_code(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    let Some(rest) = lower.strip_prefix("gfx") else {
        return false;
    };
    let Some(last) = rest.len().checked_sub(1) else {
        return false;
    };
    rest.bytes()
        .enumerate()
        .all(|(index, byte)| byte.is_ascii_digit() || (index == last && byte.is_ascii_lowercase()))
}

fn family_override_raw_arch(value: &str) -> Option<String> {
    let trimmed = value.trim();
    is_raw_gfx_arch_code(trimmed).then(|| trimmed.to_ascii_lowercase())
}

/// A host-probed arch, kept only when it belongs to the family some more
/// authoritative source already resolved.
///
/// A managed runtime's manifest records the family but never the arch, so the
/// arch can only be recovered by probing this host. That probe is trustworthy
/// exactly when it agrees: a second card, or a machine the runtime was moved to,
/// reports an arch from a different family, and installing its device payload
/// would produce a runtime that cannot run the family's kernels.
fn raw_arch_agreeing_with_family(raw_arch: Option<String>, family: &str) -> Option<String> {
    raw_arch.filter(|raw| normalize_therock_family(raw).as_deref() == Some(family))
}

pub(super) fn resolve_family(
    paths: &AppPaths,
    family_override: Option<&str>,
) -> Result<FamilyResolution> {
    let env_override = std::env::var("ROCM_CLI_THEROCK_FAMILY").ok();

    // An explicitly asked-for target that ROCm publishes no build for is
    // refused here rather than normalized. `normalize_therock_family` answers
    // `None` for gfx90c, and `None` from this branch means "fall through", so
    // without this the user's `--family gfx90c` would be dropped and whatever
    // GPU this host happens to have would be installed for instead.
    // Unrecognized spellings keep falling through to detection, which is the
    // long-standing behavior for a typo.
    //
    // The guard sits inside each source's own branch, not ahead of both, so a
    // source is only ever refused when it is actually the one being consulted:
    // a valid `--family` must resolve even if a stale `ROCM_CLI_THEROCK_FAMILY`
    // left over from an earlier session happens to name gfx90c, and the
    // reverse for an env value once `--family` isn't the one in play.
    if let Some(value) = family_override {
        if target_is_vulkan_only(value) {
            bail!(unsupported_requested_target_message(
                value.trim(),
                "--family"
            ));
        }
        if let Some(family) = normalize_therock_family(value) {
            return Ok(FamilyResolution {
                family,
                source: "manifest".to_owned(),
                raw_arch: family_override_raw_arch(value),
            });
        }
    }

    if let Some(value) = env_override.as_deref() {
        if target_is_vulkan_only(value) {
            bail!(unsupported_requested_target_message(
                value.trim(),
                "ROCM_CLI_THEROCK_FAMILY"
            ));
        }
        if let Some(family) = normalize_therock_family(value) {
            return Ok(FamilyResolution {
                family,
                source: "env".to_owned(),
                raw_arch: family_override_raw_arch(value),
            });
        }
    }

    if let Some(family) = detect_managed_therock_family(paths) {
        let raw_arch = raw_arch_agreeing_with_family(detect_host_gfx_target(), &family);
        return Ok(FamilyResolution {
            family,
            source: "managed-runtime".to_owned(),
            raw_arch,
        });
    }

    let host_arch = detect_host_gfx_target();
    if let Some(raw_arch) = host_arch.as_deref()
        && let Some(family) = normalize_therock_family(raw_arch)
    {
        return Ok(FamilyResolution {
            family,
            source: "host".to_owned(),
            raw_arch: Some(raw_arch.to_owned()),
        });
    }

    // A GPU was detected that is known to have no package family: the gfx90c
    // Vega iGPU in Renoir/Cezanne APUs. Listing the recognized families would
    // invite installing a build for different silicon, so say plainly that
    // there is nothing to install for this GPU. Other unmapped targets keep the
    // generic error below, which points at `--family`.
    if let Some(raw_arch) = host_arch.as_deref()
        && target_is_vulkan_only(raw_arch)
    {
        // The probe is paid for *here*, inside the branch that is already
        // failing, and not by `target_is_vulkan_only` above: the refusal is a
        // fact about the target, and only the sentence that follows it is a
        // claim about this host. Keeping the two apart is what lets the gate
        // stay exactly what it was while the advice stops over-promising. No
        // successful resolve runs this, so `install sdk`'s hot path is
        // unchanged, and `detect_host_gpu_diagnostics` on the next line already
        // costs more than one `vulkaninfo` attempt.
        let message = unsupported_host_gpu_message(raw_arch, detect_vulkan_support().status);
        let diagnostics = detect_host_gpu_diagnostics();
        bail!("{message}\n\n{diagnostics}");
    }

    bail!(
        "unable to resolve a supported TheRock GPU family for this host.\n\
         Re-run with an explicit package family: `rocm install sdk --family <FAMILY>`.\n\
         Recognized families: {}.\n\n{}",
        known_therock_families().join(", "),
        detect_host_gpu_diagnostics()
    )
}

/// Whether `value` is a GPU target the CLI recognizes and knows has no ROCm
/// build — [`GpuSupportLevel::VulkanOnly`] — which is what `install sdk`
/// refuses with the two messages below.
///
/// Asked of [`gpu_support_level`] rather than re-tested here. The install path
/// used to call `gfx_target_has_no_rocm_build` directly, which was *exactly*
/// equivalent, and still wrong to keep: it
/// made this a second, independent definition of the tier whose claim these
/// messages then go on to make, so a change to the classification would have
/// left `rocm examine` and `rocm install sdk` disagreeing about the same GPU
/// while both looked correct in isolation. One classifier, one answer.
///
/// The family argument is computed rather than passed as `None` because this
/// is also asked of a target the *user* named, before any host probe: for
/// `--family gfx1100` the covering family is what makes the answer `RocmHip`,
/// and hard-coding `None` here would classify every requested target as if no
/// family covered it. On the host branch the caller has already established
/// that `normalize_therock_family` answers `None`, so the extra call there is
/// a no-op that costs a string compare.
///
/// The Vulkan argument is [`rocm_core::VulkanStatus::Unknown`] — "nobody
/// looked" — and deliberately not this host's probe result. `value` may be a
/// target the user *named*, which can be a GPU that is not in this machine, so
/// a Vulkan driver found here would say nothing about it; and on the host
/// branch, probing would make `install sdk` refuse or allow based on a Vulkan
/// fact that has no bearing on whether a ROCm SDK exists. `Unknown` keeps the
/// `VulkanOnly` tier for every target the catalog confirms has no ROCm build,
/// which is exactly the set this gate refused before the probe existed — the
/// refusals are unchanged.
///
/// This is only the *gate*, not the sentence. Whether a target has a ROCm build
/// is a fact about the catalog, and probing Vulkan to answer it would make
/// `install sdk` refuse or allow on an unrelated fact. What the refusal may
/// then *say* about Vulkan is a different question, and the two messages below
/// answer it differently: [`unsupported_host_gpu_message`] is about the GPU in
/// this machine, so it grades its claim by this host's probe, while
/// [`unsupported_requested_target_message`] is about a GPU the user named,
/// which may not be here at all, so it keeps the unconditional
/// `VULKAN_ONLY_SUPPORT_CLAIM` — there is no host whose probe would bear on it.
fn target_is_vulkan_only(value: &str) -> bool {
    matches!(
        gpu_support_level(
            Some(value),
            normalize_therock_family(value).as_deref(),
            VulkanStatus::Unknown,
        ),
        GpuSupportLevel::VulkanOnly
    )
}

/// The install error for a detected GPU target that no TheRock package family
/// covers. It deliberately does not list `--family` values: none of them was
/// built for this GPU.
///
/// **The Vulkan half of this message is graded by the probe, not asserted.**
/// It used to close, unconditionally, with "the Vulkan backend can still be
/// used - try `rocm serve <model> --engine lemonade`", on a host nobody had
/// looked at. That is the same unearned claim the Vulkan probe exists to
/// remove, and because this branch runs only after *this machine's* GPU has
/// been detected, there is a probe result that bears on it directly: the same
/// session could have `rocm examine` saying "no Vulkan driver was found, so no
/// GPU backend can run here" while `rocm install sdk` sent the user to serve a
/// model on that backend. AGENTS.md §3 asks that a named command be able to
/// clear the condition it was printed for, and on a `NoGpuBackend` host that
/// `rocm serve` cannot.
///
/// So `vulkan` is routed through [`gpu_support_level`] — the same classifier
/// `rocm examine`'s `rocm_support` line uses, with the same status — and the
/// closing sentence follows the level. The claims themselves come from
/// `rocm-core`'s shared constants, so this message and that line cannot drift
/// into describing one host two ways. Only `VulkanOnly` keeps the Lemonade
/// recommendation; `VulkanUnconfirmed` makes it conditional on what
/// `vulkaninfo` reports, and `NoGpuBackend` does not make it at all.
///
/// The *refusal* is unchanged by any of this: the gate is
/// [`target_is_vulkan_only`], which does not probe. A host with no Vulkan
/// driver was refused before and is refused now; it is only told something true
/// afterwards.
///
/// The remediation pins the engine with `--engine lemonade` rather than
/// suggesting a bare `rocm serve <model>`. This branch knows the *host* has no
/// ROCm build, which is not the same as knowing which engine `serve` would
/// pick: `select_serve_engine` (`serve_cmd.rs`) ranks an explicit `--engine`
/// first, then a configured `default_engine`, and only then the host's own
/// preference — so a user who once ran `rocm config set-default-engine vllm`
/// (never validated against the GPU) would have the bare command attempt vLLM
/// on a host with no ROCm, and fail under the no-CPU-fallback policy. Naming
/// the engine makes the sentence true whatever the config says, because the
/// explicit flag out-ranks every other selection source.
///
/// **The named command is not refused on the host that is shown it.** `serve`'s
/// no-usable-GPU pre-flight is `host_has_no_usable_gpu` (`serve_cmd.rs`) over
/// `rocm_core::usable_amd_gpu_indices`, and on Linux that composes the KFD
/// compute-node count with the DRM `amdgpu` card count: a *zero* KFD count
/// falls back to DRM rather than refusing. A gfx90c APU whose kernel exposes no
/// compute node can only have been named by the DRM `ip_discovery` read, which
/// walks the same `card*/device` entries that count — so this branch is
/// reachable only on a host with at least one such card, and the pre-flight
/// therefore sees ordinal 0. `rocm_core`'s
/// `a_gfx90c_host_the_kernel_gives_no_compute_node_still_has_a_usable_gpu`
/// drives both halves off one planted `/sys`, and serve-67 runs the real binary
/// against that same host shape on a GPU-less lane.
fn unsupported_host_gpu_message(raw_arch: &str, vulkan: VulkanStatus) -> String {
    // The same three arguments `target_is_vulkan_only` passes, with this host's
    // real status in place of `Unknown`. The caller has already established
    // that the first two make the target Vulkan-only, so only the three Vulkan
    // tiers are reachable here.
    let level = gpu_support_level(
        Some(raw_arch),
        normalize_therock_family(raw_arch).as_deref(),
        vulkan,
    );
    let vulkan_sentence = match level {
        // A device answered, a driver is installed, or the platform cannot be
        // probed at all. The last of those keeps the benefit of the doubt it
        // had before any probe existed — see `GpuSupportLevel::VulkanOnly`.
        GpuSupportLevel::VulkanOnly => format!(
            "The GPU is detected but is {VULKAN_ONLY_SUPPORT_CLAIM}. \
             Try `rocm serve <model> --engine lemonade` - it runs on Lemonade's \
             Vulkan backend instead."
        ),
        // Looked at, nothing settled. The Lemonade suggestion survives, but as
        // a consequence of what `vulkaninfo` reports rather than as a promise:
        // this level is reached routinely by a working Adrenalin laptop, so
        // withholding it entirely would be as wrong as asserting it.
        GpuSupportLevel::VulkanUnconfirmed => format!(
            "The GPU is detected but is not supported by ROCm, and \
             {VULKAN_UNCONFIRMED_CLAIM} - {VULKAN_UNCONFIRMED_NEXT_STEP}. \
             If it reports a device, `rocm serve <model> --engine lemonade` runs \
             on Lemonade's Vulkan backend."
        ),
        // Both halves confirmed: no ROCm build, and the probe looked for a
        // Vulkan driver and found none. Naming `rocm serve` here would name a
        // command that cannot clear what printed it, so the only step left is
        // the one that can.
        GpuSupportLevel::NoGpuBackend => {
            "The GPU is detected but is not supported by ROCm, and no Vulkan GPU driver was \
             found here either, so there is no GPU backend to serve a model on. Install one \
             (Mesa RADV on Linux, the AMD display driver on Windows) and re-run `rocm \
             examine` to confirm it."
                .to_owned()
        }
        // Unreachable: the caller gated on `target_is_vulkan_only`, which asks
        // `gpu_support_level` the same first two questions, so the catalog has
        // already confirmed a recognized target with no ROCm build and no
        // covering family. Spelled out rather than folded into a `_` so that a
        // new `GpuSupportLevel` is a compile error here instead of silently
        // inheriting a Vulkan claim, and worded so that if one ever did arrive
        // it would assert nothing.
        GpuSupportLevel::RocmHip | GpuSupportLevel::Unknown => {
            "Run `rocm examine` to see what this host supports.".to_owned()
        }
    };
    format!(
        "no ROCm SDK build exists for the detected GPU target `{raw_arch}`, so \
         `rocm install sdk` has nothing to install for it.\n{vulkan_sentence}"
    )
}

/// The install error for a target the user asked for by name that no TheRock
/// package family covers.
///
/// Separate from [`unsupported_host_gpu_message`] because nothing was detected
/// here: the user named the target, so the error names where it came from and
/// does not claim a GPU is present. Like that message it offers no `--family`
/// values, since none of them was built for this GPU.
///
/// That same distinction is why this one must not borrow its sibling's closing
/// sentence. `unsupported_host_gpu_message` can promise that `rocm serve` runs
/// on Lemonade's Vulkan backend because it is only reached once host detection
/// has itself come back with a target that has no ROCm build. This message is
/// raised from the `--family` / `ROCM_CLI_THEROCK_FAMILY` branches at the top of
/// [`resolve_family`], before any host probe runs, so the rejected target says
/// nothing about the GPU in this machine: a stale env value naming gfx90c
/// refuses just the same on a host whose own GPU is fully ROCm-capable, and
/// there `rocm serve` would run on ROCm/HIP. So it points at `rocm examine`,
/// which reports this host's real support level, instead of asserting one.
fn unsupported_requested_target_message(value: &str, source: &str) -> String {
    format!(
        "no ROCm SDK build exists for the GPU target `{value}` requested by \
         {source}, so `rocm install sdk` has nothing to install for it.\n\
         That GPU is {VULKAN_ONLY_SUPPORT_CLAIM}. Nothing was installed and no \
         other family was substituted. \
         Run `rocm examine` to see what this host itself supports."
    )
}

pub(super) fn family_resolution_hint(
    source: &str,
    family: &str,
    channel: TheRockChannel,
    format: &str,
) -> String {
    let families = known_therock_families().join(", ");
    let auto_detected = matches!(source, "host" | "managed-runtime");
    let mut hint = String::new();

    if auto_detected {
        let _ = write!(
            hint,
            "no installable TheRock {} runtime was found for the auto-detected GPU family `{family}`.\n\
             Re-run with an explicit package family: `rocm install sdk --family <FAMILY>`.\n\
             Recognized families: {families}.",
            channel.as_str()
        );
    } else {
        let _ = write!(
            hint,
            "no installable TheRock {} runtime was found for the requested package family `{family}`.\n\
             Recognized families: {families}.",
            channel.as_str()
        );
    }

    let other_channel = match channel {
        TheRockChannel::Release => "nightly",
        TheRockChannel::Nightly => "release",
    };
    let alternate_format = match format {
        "wheel" if !runtime_is_windows() => Some("tarball"),
        "tarball" => Some("wheel"),
        _ => None,
    };
    match alternate_format {
        Some(alternate_format) => {
            let _ = write!(
                hint,
                "\nIf your GPU is newer than the {} packages, try `--channel {other_channel}` or `--format {alternate_format}`.",
                channel.as_str()
            );
        }
        None => {
            let _ = write!(
                hint,
                "\nIf your GPU is newer than the {} packages, try `--channel {other_channel}`.",
                channel.as_str()
            );
        }
    }

    hint
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::{
        test_paths, test_runtime_manifest, write_test_runtime_manifest,
    };
    use std::fs;

    #[test]
    fn family_override_raw_arch_accepts_only_exact_arch_codes() {
        assert_eq!(
            family_override_raw_arch("gfx1200"),
            Some("gfx1200".to_owned())
        );
        assert_eq!(
            family_override_raw_arch(" gfx90a "),
            Some("gfx90a".to_owned())
        );
        assert_eq!(
            family_override_raw_arch("GFX1250"),
            Some("gfx1250".to_owned())
        );
        assert_eq!(family_override_raw_arch("gfx120X-all"), None);
        assert_eq!(family_override_raw_arch("gfx103X-dgpu"), None);
        assert_eq!(
            family_override_raw_arch("gpu target gfx1030 detected"),
            None
        );
        assert_eq!(family_override_raw_arch("gfx"), None);
        assert_eq!(family_override_raw_arch(""), None);
    }

    /// A managed runtime records only its family, so the arch has to come from
    /// a host probe — and is worth nothing when the host is a different card.
    #[test]
    fn managed_family_keeps_only_a_host_arch_from_the_same_family() {
        assert_eq!(
            raw_arch_agreeing_with_family(Some("gfx1030".to_owned()), "gfx103X-dgpu"),
            Some("gfx1030".to_owned())
        );
        assert_eq!(
            raw_arch_agreeing_with_family(Some("gfx90a".to_owned()), "gfx103X-dgpu"),
            None
        );
        assert_eq!(raw_arch_agreeing_with_family(None, "gfx103X-dgpu"), None);
    }

    /// The gate on both install refusals is the classifier's `VulkanOnly`
    /// tier, not a private re-derivation of it.
    ///
    /// Every case is one `gpu_support_level` answers, so a change there has to
    /// come through here: `gfx90c` is the recognized-with-no-ROCm-build tier
    /// and is refused; a covered target is `RocmHip` and is not; an
    /// unrecognized one is `Unknown`, a gap in this build's catalog rather
    /// than a fact about the card, and refusing it would be the exact false
    /// claim `GpuSupportLevel`'s doc comment exists to prevent.
    #[test]
    fn the_install_refusal_gate_is_the_classifiers_vulkan_only_tier() {
        for refused in ["gfx90c", "GFX90C", "gfx90c:xnack-", " gfx90c "] {
            assert!(
                target_is_vulkan_only(refused),
                "`{refused}` is the Vulkan-only tier and install must refuse it"
            );
        }
        for allowed in ["gfx1100", "gfx1151", "gfx90a", "gfx950-dcgpu"] {
            assert!(
                !target_is_vulkan_only(allowed),
                "`{allowed}` has a ROCm build; install must not refuse it"
            );
        }
        for unknown in ["gfx803", "not-a-target", ""] {
            assert!(
                !target_is_vulkan_only(unknown),
                "`{unknown}` is unrecognized, which is a gap in this catalog and \
                 not a confirmed absence of a ROCm build: refusing it would claim \
                 something about hardware nobody probed"
            );
        }
    }

    /// Both refusals make the same claim about a Vulkan-only GPU, so they
    /// render it from the same string.
    ///
    /// Asserting the constant is *in* each message is what makes it one
    /// claim; the wording itself is pinned word-for-word in `rocm-core`
    /// (`the_vulkan_only_claim_is_pinned_word_for_word`), so neither test
    /// alone can be satisfied by quietly rewriting the other's expectation.
    ///
    /// `rocm examine`'s `rocm_support` line is deliberately not in this list.
    /// It grades the claim finer than three tiers, naming the device the probe
    /// enumerated - see `rocm_support_line` in `rocm-core`'s
    /// `host_gpu/render.rs`.
    ///
    /// The host-side refusal is asked here at the status where it *does* make
    /// the claim. Which statuses those are is
    /// `the_detected_gpu_refusal_grades_its_vulkan_claim_by_the_probe`'s
    /// question; this one is only that the wording comes from the shared
    /// constant and not from a second copy.
    #[test]
    fn both_install_refusals_render_the_shared_vulkan_only_claim() {
        for message in [
            unsupported_host_gpu_message("gfx90c", VulkanStatus::Available),
            unsupported_requested_target_message("gfx90c", "--family"),
        ] {
            assert!(
                message.contains(VULKAN_ONLY_SUPPORT_CLAIM),
                "the claim must come from the shared constant, or install and \
                 examine can drift apart while both look right: {message}"
            );
        }
    }

    /// The detected-GPU refusal must say about Vulkan only what this host's
    /// probe established.
    ///
    /// The regression this pins is a contradiction visible inside one session:
    /// on a host with no AMD Vulkan driver, `rocm examine` reported "no Vulkan
    /// GPU driver was found" while `rocm install sdk` closed with "the Vulkan
    /// backend can still be used; try `rocm serve <model> --engine lemonade`".
    /// Both read as facts about the same machine, and the command the second
    /// one named could not clear the condition it was printed for, which
    /// AGENTS.md §3 requires of remediation advice.
    ///
    /// Every `VulkanStatus` is covered, so a status that stops being routed
    /// through `gpu_support_level` fails here. `Unknown` is in the
    /// claim-keeping list deliberately: a platform the probe cannot read at
    /// all keeps the benefit of the doubt it had before the probe existed,
    /// which is the one exception `GpuSupportLevel::VulkanOnly` documents.
    #[test]
    fn the_detected_gpu_refusal_grades_its_vulkan_claim_by_the_probe() {
        // A device answered, a driver is installed, or nobody could look: all
        // three may offer Lemonade, and all three say so with the shared claim.
        for status in [
            VulkanStatus::Available,
            VulkanStatus::DriverPresentDevicesUnverified,
            VulkanStatus::Unknown,
        ] {
            let message = unsupported_host_gpu_message("gfx90c", status);
            assert!(
                message.contains(VULKAN_ONLY_SUPPORT_CLAIM),
                "{status:?} is the tier that has earned the claim: {message}"
            );
            assert!(
                message.contains("rocm serve <model> --engine lemonade"),
                "{status:?}: {message}"
            );
        }

        // Looked at and nothing settled. The question is open, so the message
        // must say so rather than answer it in either direction, and the
        // Lemonade suggestion survives only as a consequence of what
        // `vulkaninfo` reports.
        let unconfirmed = unsupported_host_gpu_message("gfx90c", VulkanStatus::Inconclusive);
        assert!(
            unconfirmed.contains(VULKAN_UNCONFIRMED_CLAIM),
            "a host that was probed and showed nothing is owed the hedged claim, the same one \
             `rocm examine` gives it: {unconfirmed}"
        );
        assert!(
            !unconfirmed.contains(VULKAN_ONLY_SUPPORT_CLAIM),
            "nothing here established a Vulkan backend: {unconfirmed}"
        );
        assert!(
            unconfirmed.contains("vulkaninfo"),
            "the one thing that would settle it has to be named: {unconfirmed}"
        );
        assert!(
            unconfirmed.contains("If it reports a device, `rocm serve <model> --engine lemonade`"),
            "the serve suggestion must be conditional on the probe, not a promise: {unconfirmed}"
        );

        // Both halves confirmed. `rocm serve` cannot clear this, so it must not
        // be named at all.
        let absent = unsupported_host_gpu_message("gfx90c", VulkanStatus::Absent);
        assert!(
            !absent.contains(VULKAN_ONLY_SUPPORT_CLAIM),
            "`rocm examine` tells this host no GPU backend can run here; `install sdk` must not \
             tell it the opposite in the same session: {absent}"
        );
        assert!(
            !absent.contains("rocm serve"),
            "the probe looked and found no Vulkan driver, so serving on one is a command that \
             cannot clear the condition it is printed for (AGENTS.md §3): {absent}"
        );
        assert!(
            !absent.contains("lemonade") && !absent.contains("Lemonade"),
            "naming the engine makes the same promise as naming the command: {absent}"
        );
        assert!(
            absent.contains("no Vulkan GPU driver was found"),
            "the host is still owed what was established about it: {absent}"
        );
        assert!(
            absent.contains("Mesa RADV") && absent.contains("AMD display driver"),
            "and the one step that can clear it: {absent}"
        );

        // The target and the opening sentence are the same whatever Vulkan
        // reported: the refusal itself is a fact about the catalog, and only
        // the sentence after it is a claim about this host.
        for status in [
            VulkanStatus::Available,
            VulkanStatus::DriverPresentDevicesUnverified,
            VulkanStatus::Inconclusive,
            VulkanStatus::Absent,
            VulkanStatus::Unknown,
        ] {
            let message = unsupported_host_gpu_message("gfx90c", status);
            assert!(
                message.contains("no ROCm SDK build exists for the detected GPU target `gfx90c`"),
                "{status:?}: {message}"
            );
            assert!(
                target_is_vulkan_only("gfx90c"),
                "and the gate that printed it does not consult the probe at all"
            );
        }
    }

    #[test]
    fn unsupported_gpu_error_names_the_gpu_and_does_not_suggest_families() {
        let message = unsupported_host_gpu_message("gfx90c", VulkanStatus::Available);
        assert!(message.contains("no ROCm SDK build"), "{message}");
        assert!(message.contains("`gfx90c`"), "{message}");
        assert!(message.contains("Vulkan"), "{message}");
        assert!(message.contains("rocm serve <model>"), "{message}");
        assert!(!message.contains("--family"), "{message}");
        for family in known_therock_families() {
            assert!(
                !message.contains(family),
                "the error must not offer `{family}` for an unsupported GPU: {message}"
            );
        }
    }

    /// Knowing the host has no ROCm build is not knowing which engine `serve`
    /// would pick.
    ///
    /// `select_serve_engine` ranks a configured `default_engine` above the
    /// host's own preference, and `rocm config set-default-engine` never
    /// validates that choice against the GPU. So on a gfx90c host whose config
    /// says `vllm`, a bare `rocm serve <model>` attempts vLLM and fails under
    /// the no-CPU-fallback policy — while this message promised Lemonade's
    /// Vulkan backend. Pinning `--engine lemonade` makes the sentence true for
    /// every config, because an explicit `--engine` out-ranks all of it.
    #[test]
    fn unsupported_gpu_error_pins_the_engine_a_configured_default_could_override() {
        let message = unsupported_host_gpu_message("gfx90c", VulkanStatus::Available);

        assert!(
            message.contains("rocm serve <model> --engine lemonade"),
            "the remediation must name the engine it promises, or a configured \
             `default_engine` makes it false: {message}"
        );
        // Guards the exact regression: the bare command with the claim still
        // attached. `--engine lemonade` keeps `rocm serve <model>` as a prefix,
        // so the check has to look at what directly follows it.
        assert!(
            !message.contains("rocm serve <model>` "),
            "a bare `rocm serve <model>` cannot carry this promise: {message}"
        );
    }

    /// The requested-target refusal must not promise which backend `rocm serve`
    /// will use, because it never looked at this host.
    ///
    /// Both refusals used to close with the same "`rocm serve <model>` runs on
    /// Lemonade's Vulkan backend" sentence. In `unsupported_host_gpu_message`
    /// that is a fact: it is only reached after `detect_host_gfx_target` returns
    /// a target `target_is_vulkan_only` confirms. This one fires from the
    /// `--family` / `ROCM_CLI_THEROCK_FAMILY` branches before any probe runs —
    /// see `cli_family_overrides_a_stale_env_value_naming_an_unsupported_gpu`
    /// for an override that deliberately disagrees with its host — so on a
    /// ROCm-capable machine it told the user Vulkan when serving would really
    /// run on ROCm/HIP. It now points at `rocm examine` instead, which is true
    /// on every host and still leaves somewhere to go.
    #[test]
    fn requested_target_error_does_not_claim_which_backend_serving_will_use() {
        for source in ["--family", "ROCM_CLI_THEROCK_FAMILY"] {
            let message = unsupported_requested_target_message("gfx90c", source);
            assert!(
                !message.contains("rocm serve"),
                "the requested target says nothing about this host's GPU, so the \
                 refusal must not predict what `rocm serve` would run on: {message}"
            );
            assert!(
                !message.contains("Lemonade"),
                "naming the engine makes the same unchecked promise about this \
                 host as naming the command: {message}"
            );
            assert!(
                message.contains("rocm examine"),
                "dropping the claim must still leave the user a next step: {message}"
            );
        }

        // The sibling path did look at this host, so it may make the claim when
        // the probe backs it: this is the asymmetry, not a blanket ban on the
        // sentence. It pins the engine for a different reason — see
        // `unsupported_gpu_error_pins_the_engine_a_configured_default_could_override`
        // — and which probe results earn it is
        // `the_detected_gpu_refusal_grades_its_vulkan_claim_by_the_probe`'s.
        let detected = unsupported_host_gpu_message("gfx90c", VulkanStatus::Available);
        assert!(
            detected.contains("rocm serve <model> --engine lemonade"),
            "{detected}"
        );
        assert!(detected.contains("Vulkan"), "{detected}");
    }

    /// Asking for gfx90c by name must fail, not quietly install for some other
    /// GPU.
    ///
    /// `normalize_therock_family("gfx90c")` is `None`, and in `resolve_family`
    /// `None` from the override branch means "keep looking" — so before the
    /// explicit-target guard, `--family gfx90c` was dropped on the floor and
    /// the host's own family was installed instead. That is worse than the
    /// data-center mapping this change removed: the user named a target and
    /// got a different one with no warning.
    ///
    /// Reaching detection at all would make this test depend on the machine it
    /// runs on, so a failure here is also the signal that the guard moved below
    /// the branches it has to precede.
    #[test]
    fn requesting_gfx90c_by_name_fails_instead_of_installing_another_family() {
        let (_root, paths) = test_paths("resolve-family-gfx90c-override");
        let error = resolve_family(&paths, Some("gfx90c"))
            .expect_err("--family gfx90c must not resolve to any package family");
        let message = error.to_string();
        assert!(message.contains("no ROCm SDK build"), "{message}");
        assert!(message.contains("`gfx90c`"), "{message}");
        assert!(message.contains("--family"), "{message}");
        assert!(
            message.contains("no other family was substituted"),
            "{message}"
        );
        for family in known_therock_families() {
            assert!(
                !message.contains(family),
                "asking for gfx90c must not resolve or offer `{family}`: {message}"
            );
        }
    }

    /// A valid `--family` must resolve even if `ROCM_CLI_THEROCK_FAMILY` is
    /// stale and happens to name gfx90c.
    ///
    /// The no-build guard used to run ahead of both sources regardless of
    /// which one would actually be consulted, so a leftover env value with no
    /// bearing on this invocation could refuse an otherwise-valid `--family`.
    /// The guard now lives inside each source's own branch, so it only fires
    /// for the source actually being read.
    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn cli_family_overrides_a_stale_env_value_naming_an_unsupported_gpu() {
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let old_env = std::env::var_os("ROCM_CLI_THEROCK_FAMILY");
        unsafe {
            std::env::set_var("ROCM_CLI_THEROCK_FAMILY", "gfx90c");
        }

        let (_root, paths) = test_paths("resolve-family-cli-overrides-stale-env");
        let resolution = resolve_family(&paths, Some("gfx1103"));

        unsafe {
            match old_env {
                Some(value) => std::env::set_var("ROCM_CLI_THEROCK_FAMILY", value),
                None => std::env::remove_var("ROCM_CLI_THEROCK_FAMILY"),
            }
        }

        let resolution = resolution
            .expect("a valid --family must resolve even with an unrelated gfx90c env override set");
        assert_eq!(resolution.family, "gfx110X-all");
        assert_eq!(resolution.source, "manifest");
    }

    #[test]
    fn requested_target_error_names_where_the_target_came_from() {
        let message = unsupported_requested_target_message("gfx90c", "--family");
        assert!(message.contains("`gfx90c`"), "{message}");
        assert!(message.contains("--family"), "{message}");
        // The user named it; nothing was detected, so the error must not say so.
        assert!(!message.contains("detected"), "{message}");

        let message = unsupported_requested_target_message("gfx90c", "ROCM_CLI_THEROCK_FAMILY");
        assert!(message.contains("ROCM_CLI_THEROCK_FAMILY"), "{message}");
    }

    #[test]
    fn resolve_family_uses_managed_runtime_before_host_detection() -> Result<()> {
        if std::env::var("ROCM_CLI_THEROCK_FAMILY")
            .ok()
            .and_then(|value| normalize_therock_family(&value))
            .is_some()
        {
            return Ok(());
        }

        let (root, paths) = test_paths("resolve-family-managed-runtime");
        let manifest = test_runtime_manifest("active", "therock-release:gfx120X-all", 1);
        write_test_runtime_manifest(&paths, &manifest)?;

        let resolution = resolve_family(&paths, None)?;

        assert_eq!(resolution.family, "gfx120X-all");
        assert_eq!(resolution.source, "managed-runtime");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn family_resolution_hint_for_auto_detected_points_at_family_flag() {
        let hint = family_resolution_hint("host", "gfx950-dcgpu", TheRockChannel::Release, "wheel");

        assert!(hint.contains("auto-detected GPU family `gfx950-dcgpu`"));
        assert!(hint.contains("--family <FAMILY>"));
        // Lists recognized families the user can pass instead.
        assert!(hint.contains("gfx110X-all"));
        // Points at the other channel as an escape hatch.
        assert!(hint.contains("--channel nightly"));
    }

    #[test]
    fn family_resolution_hint_for_user_supplied_frames_around_requested_family() {
        let hint =
            family_resolution_hint("manifest", "gfx110X-all", TheRockChannel::Release, "wheel");

        assert!(hint.contains("requested package family `gfx110X-all`"));
        // A family the user named themselves is not described as auto-detected.
        assert!(!hint.contains("auto-detected"));
        assert!(!hint.contains("--family <FAMILY>"));
        assert!(hint.contains("--channel nightly"));
    }

    #[test]
    fn family_resolution_hint_suggests_release_channel_from_nightly() {
        let hint = family_resolution_hint("host", "gfx950-dcgpu", TheRockChannel::Nightly, "wheel");

        assert!(hint.contains("--channel release"));
    }
}
