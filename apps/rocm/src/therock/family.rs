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
    AppPaths, detect_host_gfx_target, detect_host_gpu_diagnostics, detect_managed_therock_family,
    gfx_target_has_no_rocm_build, known_therock_families, normalize_therock_family,
    runtime_is_windows,
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
        if gfx_target_has_no_rocm_build(value) {
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
        if gfx_target_has_no_rocm_build(value) {
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
        && gfx_target_has_no_rocm_build(raw_arch)
    {
        let message = unsupported_host_gpu_message(raw_arch);
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

/// The install error for a detected GPU target that no TheRock package family
/// covers. It deliberately does not list `--family` values: none of them was
/// built for this GPU.
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
fn unsupported_host_gpu_message(raw_arch: &str) -> String {
    format!(
        "no ROCm SDK build exists for the detected GPU target `{raw_arch}`, so \
         `rocm install sdk` has nothing to install for it.\n\
         The GPU is detected but is not supported by ROCm; \
         the Vulkan backend can still be used. \
         Try `rocm serve <model> --engine lemonade` - it runs on Lemonade's \
         Vulkan backend instead."
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
         That GPU is not supported by ROCm; the Vulkan backend can still be \
         used. Nothing was installed and no other family was substituted. \
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

    #[test]
    fn unsupported_gpu_error_names_the_gpu_and_does_not_suggest_families() {
        let message = unsupported_host_gpu_message("gfx90c");
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
        let message = unsupported_host_gpu_message("gfx90c");

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
    /// a target `gfx_target_has_no_rocm_build` confirms. This one fires from the
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

        // The sibling path did look at this host, so it keeps the claim: this is
        // the asymmetry, not a blanket ban on the sentence. It pins the engine
        // for a different reason — see
        // `unsupported_gpu_error_pins_the_engine_a_configured_default_could_override`.
        let detected = unsupported_host_gpu_message("gfx90c");
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
