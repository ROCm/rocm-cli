// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Model fit: answering "can this host run this model" for `rocm diagnose
//! --model`. `assess_model_on_this_host` and `assess_model_with_host_paths`
//! gather what the verdict needs (the examination, system RAM, the model
//! catalog, the saved config and the host GPU summary) and hand it to
//! `assess_model_for_host`, which picks the engine through
//! `serve_cmd::select_serve_engine` and asks `rocm_core`'s readiness check for
//! the verdict. `host_accelerator_memory`, `classify_accelerator_memory` and
//! `examination_reports_amd_gpu` turn the examination and per-GPU VRAM
//! readings into the accelerator-memory fact, and `PlatformRuleOut`,
//! `engine_ruled_out_by_platform` and `unsupported_here_for` rule an engine
//! out for the platform (native Windows for vLLM and AIM, an unreachable Docker
//! daemon for AIM) and word the reason.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. It is a module of its own, rather than part of the
//! `rocm diagnose` handler, so `examine_report_text.rs` and
//! `model_catalog_text.rs`, which read `PlatformRuleOut` and
//! `engine_ruled_out_by_platform`, do not have to depend on a command module. Its own dependencies are `serve_cmd.rs` and
//! `gpu_policy.rs`; neither depends on this module. `assess_model_on_this_host`
//! (called from `diagnose()` in `diagnose_cmd.rs`), `PlatformRuleOut` and
//! `engine_ruled_out_by_platform` are `pub(crate)`; `main.rs` re-imports the
//! last two so `examine_report_text.rs` and `model_catalog_text.rs` keep
//! reaching them as `crate::<name>`.
//! Every other item has no caller outside this module and stays private.
//!
//! The cluster's regression tests live in this module's own `mod tests`.
//! `engine_ruled_out_by_platform_gates_aim_on_docker_reachability` re-runs the
//! test binary on `docker_unreachable_subprocess_helper` by its full path,
//! `model_fit::tests::docker_unreachable_subprocess_helper`; that path moved
//! with the helper.

use rocm_core::model_readiness::{
    AcceleratorMemory, HostEngineChoice, HostFacts, ModelCatalogSource, ModelReadiness,
};
use rocm_core::{
    AppPaths, ModelRecipeRecord, RocmCliConfig, detect_host_gpu_summary, load_model_recipe_registry,
};

use crate::gpu_policy::{GpuVramUsage, gpu_vram_usage, vram_capacity_is_meaningful};
use crate::serve_cmd::select_serve_engine;

/// Answer `--model` for the machine running the command.
///
/// Gathers only what the verdict needs, and keeps every "could not read it"
/// apart from every "it is not there": an unreadable recipe index becomes
/// `Unreachable` rather than an empty catalog, and GPU memory that `amd-smi`
/// could not report becomes `Unknown` rather than zero. Both collapses would
/// turn a gap in what the CLI can see into a confident refusal of the model.
pub(crate) fn assess_model_on_this_host(
    model_ref: &str,
    examination: &rocm_core::Examination,
) -> ModelReadiness {
    let paths = AppPaths::discover().ok();
    assess_model_with_host_paths(model_ref, examination, paths.as_ref())
}

/// Testable core of [`assess_model_on_this_host`].
///
/// Takes the discovered `AppPaths` as an explicit parameter so the GPU-summary
/// lookup and the config lookup read the same directories, instead of each
/// racing its own `AppPaths::discover()` call. That used to be two separate
/// calls: one (implicitly `None`) feeding [`detect_host_gpu_summary`], another
/// feeding the config lookup a few lines later. Both resolve identically today
/// because `AppPaths::discover()` is deterministic within a process, but
/// `detect_host_gpu_summary`'s paths-driven TheRock-manifest gfx-target
/// fallback only ever sees a path when one is passed in -- silently skipping
/// it here is exactly the kind of drift that would let this command name a
/// different engine than `rocm serve` for the same model.
fn assess_model_with_host_paths(
    model_ref: &str,
    examination: &rocm_core::Examination,
    paths: Option<&AppPaths>,
) -> ModelReadiness {
    let host_gpu_summary = detect_host_gpu_summary(paths);
    let host = HostFacts {
        accelerator_memory: host_accelerator_memory(&host_gpu_summary, examination),
        system_ram_gib: rocm_core::detect_system_ram_gib(),
    };
    let configured_engine = paths
        .and_then(|paths| RocmCliConfig::load(paths).ok())
        .and_then(|config| config.default_engine);
    match load_model_recipe_registry() {
        Ok(registry) => assess_model_for_host(
            model_ref,
            &ModelCatalogSource::Available(&registry),
            &host,
            configured_engine.as_deref(),
            Some(&host_gpu_summary),
        ),
        Err(error) => assess_model_for_host(
            model_ref,
            &ModelCatalogSource::Unreachable {
                detail: format!("{error:#}"),
            },
            &host,
            configured_engine.as_deref(),
            Some(&host_gpu_summary),
        ),
    }
}

/// The memory pool an engine on this host would allocate a model from.
///
/// The APU case is why this is not simply a sum of `total_vram`. An APU has no
/// private VRAM; `amd-smi` reports the fixed BIOS carve-out (often ~4 GiB) while
/// the allocator serves the model out of GTT-backed system RAM. Comparing a
/// recipe minimum against the carve-out would refuse a 22 GiB model on a 128 GiB
/// Strix Halo, which serves it fine. [`vram_capacity_is_meaningful`] is the same
/// test `serve`'s low-VRAM warning uses, so the two cannot disagree about which
/// hosts the figure describes.
///
/// The examination is what makes "there is no GPU" reachable at all. `amd-smi`
/// answers "how much memory", and its absence is ambiguous — no GPU, or no
/// `amd-smi`. The examination reads the devices themselves, so it can tell those
/// two apart, and they deserve different answers: one is a machine that needs a
/// GPU, the other a machine whose GPU could not be measured.
/// Whether the examination has positive or unresolved evidence of an AMD GPU,
/// used to tell "no GPU" apart from "GPU present but VRAM unmeasured" when
/// `amd-smi` produced nothing.
///
/// `examination.has_amd_gpu` cannot answer this on WSL: the hardware probes
/// that set it are skipped there (see [`rocm_core::Examination::probe_with_interpreter`]),
/// so the probe leaves it `false` on every WSL host, healthy or not. This
/// command reads the probe directly, never the `rocm examine --json` document
/// that fills the field in afterwards from the Windows-side GPU target, so it
/// would report a working machine as having no GPU at all -- `Blocked`
/// instead of the `Undetermined` a genuinely unmeasurable host deserves.
/// `rocm_sees_gpu`, not `has_amd_gpu`, is the WSL-native signal, per the same
/// reasoning `fix-wsl-6-host-driver-too-old` already uses in `diagnose.rs`.
/// Only `Some(false)` -- rocminfo positively enumerating no device -- counts
/// as "no GPU"; `None` means rocminfo was absent so the question went unasked,
/// which is not evidence of anything and must not read as a confident no.
fn examination_reports_amd_gpu(examination: &rocm_core::Examination) -> bool {
    match examination.wsl.as_ref() {
        Some(wsl) => wsl.rocm_sees_gpu != Some(false),
        None => examination.has_amd_gpu,
    }
}

fn host_accelerator_memory(
    host_gpu_summary: &rocm_core::HostGpuSummary,
    examination: &rocm_core::Examination,
) -> AcceleratorMemory {
    classify_accelerator_memory(
        gpu_vram_usage().as_deref(),
        host_gpu_summary.gfx_target.as_deref(),
        examination_reports_amd_gpu(examination),
        rocm_core::usable_amd_gpu_indices().as_deref(),
    )
}

/// The pure classification behind [`host_accelerator_memory`], split out so it
/// can be exercised with fixed input instead of a live `amd-smi` subprocess.
/// Deciding whether a host is an APU and which figure represents its memory is
/// the actual judgement this command depends on; entangled with the
/// subprocess call it was untestable, and a checker that silently regressed
/// to always reporting `Dedicated` would leave every test green.
fn classify_accelerator_memory(
    vram: Option<&[GpuVramUsage]>,
    gfx_target: Option<&str>,
    gpu_reported: bool,
    usable_indices: Option<&[u32]>,
) -> AcceleratorMemory {
    let Some(vram) = vram else {
        if gpu_reported {
            return AcceleratorMemory::Unknown;
        }
        return AcceleratorMemory::None;
    };
    if vram_capacity_is_meaningful(gfx_target, vram.len()) {
        // `rocm serve` pins exactly one ordinal -- there is no tensor-parallel
        // path anywhere in this binary -- so summing every card's VRAM into one
        // figure can report a model READY when it would OOM on every individual
        // card. Narrow to the mask-visible rows (a `HIP_VISIBLE_DEVICES` mask
        // hides devices the same way it does for `rocm serve` itself) and take
        // the most capable one: that is the best case `--gpu auto` could
        // actually land on, so it is the right upper bound for "would this
        // fit", even though it cannot name which ordinal the verdict answers
        // for. `usable_indices` of `None` means the mask could not be probed,
        // so every reported row is treated as visible rather than refusing to
        // answer.
        let max_gib = vram
            .iter()
            .filter(|usage| usable_indices.is_none_or(|indices| indices.contains(&usage.index)))
            .map(|usage| usage.total_mb as f64 / 1024.0)
            .fold(0.0_f64, f64::max);
        return AcceleratorMemory::Dedicated(max_gib);
    }
    // This is an APU: the carve-out `amd-smi` reports above is the wrong pool,
    // per `vram_capacity_is_meaningful`'s own doc. The right pool is the GTT
    // aperture, and nothing in this binary can read it -- `amd-smi` does not
    // report it, and no code path here calls `rocm-smi`, the tool that does.
    // Total system RAM is not a substitute: the GTT aperture is capped well
    // below installed RAM by BIOS/kernel policy, and asserting installed RAM
    // as the engine's allocation pool trades the carve-out's under-estimate
    // for an over-estimate in exactly the direction that produces a false
    // `Ready` for a model that does not actually fit -- the wrong answer this
    // command exists to prevent. This is deliberately not `Unknown`: that
    // variant means no telemetry at all, for which "run `amd-smi metric
    // --json`" is a real next step. Here `amd-smi` already ran and answered --
    // it just named the wrong pool -- so the same remediation would send a
    // Strix Halo user in a circle forever. Until the real aperture can be
    // read, this reports `UnifiedMemoryUnreadable`, so the verdict reads
    // `Undetermined` with no dead-end command attached, rather than a
    // confident wrong one.
    AcceleratorMemory::UnifiedMemoryUnreadable
}

/// Assess a model against this host, with `serve`'s own engine decision.
///
/// The engine is not re-derived here. `select_serve_engine` is the one place
/// that answers "which engine serves this model on this host", and passing it in
/// is what keeps `rocm diagnose --model` from confidently naming an engine
/// `rocm serve` would never pick. An engine ruled out by the platform gate is
/// reported as such rather than silently swapped for another: `serve` does not
/// fall back either, and a verdict that pretended otherwise would be wrong in
/// the user's favour.
fn assess_model_for_host(
    model_ref: &str,
    catalog: &ModelCatalogSource<'_>,
    host: &HostFacts,
    configured_default_engine: Option<&str>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
) -> ModelReadiness {
    let engine_for = |recipe: &ModelRecipeRecord| {
        let selection = select_serve_engine(
            None,
            configured_default_engine,
            Some(recipe),
            host_gpu_summary,
        );
        HostEngineChoice {
            unsupported_here: unsupported_here_for(
                &selection.engine,
                engine_ruled_out_by_platform(&selection.engine),
            ),
            engine: selection.engine,
            source: selection.source.to_owned(),
        }
    };
    rocm_core::model_readiness::assess_model_readiness(model_ref, catalog, host, &engine_for)
}

/// Why the platform gate rules an engine out on this host, if it does.
///
/// Returned by [`engine_ruled_out_by_platform`], the one place that decides
/// *why*, so the `/model` adapter-availability note and the readiness verdict
/// -- which only decide how to word it -- cannot disagree about the same host
/// and engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PlatformRuleOut {
    Allowed,
    NativeWindowsVllm,
    /// AIM's Docker GPU passthrough exists only for Linux and WSL2.
    NativeWindowsAim,
    AimDockerUnavailable,
}

/// Whether, and why, the platform gate rules this engine out on this host.
pub(crate) fn engine_ruled_out_by_platform(engine: &str) -> PlatformRuleOut {
    engine_ruled_out_by_platform_with(engine, rocm_engine_aim::builtin_docker_daemon_reachable)
}

/// [`engine_ruled_out_by_platform`] with the Docker probe injected, so a caller
/// that asks about many engines in one pass (the `/model` listing) can answer
/// the same `docker info` once instead of once per recipe.
pub(crate) fn engine_ruled_out_by_platform_with(
    engine: &str,
    docker_reachable: impl FnOnce() -> bool,
) -> PlatformRuleOut {
    if rocm_core::runtime_is_windows() && engine.eq_ignore_ascii_case("vllm") {
        PlatformRuleOut::NativeWindowsVllm
    } else if rocm_core::runtime_is_windows() && engine.eq_ignore_ascii_case("aim") {
        PlatformRuleOut::NativeWindowsAim
    } else if engine.eq_ignore_ascii_case("aim") && !docker_reachable() {
        PlatformRuleOut::AimDockerUnavailable
    } else {
        PlatformRuleOut::Allowed
    }
}

/// The `unsupported_here` message for an engine, given why the platform gate
/// already ruled it out.
///
/// Split out from the `engine_for` closure in [`assess_model_for_host`] so it
/// can be unit-tested on both branches directly, instead of only indirectly
/// through whichever platform the test happens to run on. `ruled_out` is
/// passed in rather than re-derived from `engine` here, so a test can
/// exercise the "ruled out" branch on Linux CI and the "allowed" branch on a
/// Windows runner without either one being unreachable.
fn unsupported_here_for(engine: &str, ruled_out: PlatformRuleOut) -> Option<String> {
    match ruled_out {
        PlatformRuleOut::Allowed => None,
        PlatformRuleOut::AimDockerUnavailable => Some(format!(
            "{engine} needs a reachable Docker daemon to serve; start Docker or choose a different engine"
        )),
        PlatformRuleOut::NativeWindowsVllm | PlatformRuleOut::NativeWindowsAim => Some(format!(
            "{engine} has no adapter on native Windows; serve it from WSL or Linux"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hosts whose engine choice differs, so the assertion below is exercised on
    /// more than one branch of `select_serve_engine`.
    fn engine_selection_hosts() -> Vec<(&'static str, rocm_core::HostGpuSummary)> {
        vec![
            (
                "Instinct MI300X",
                rocm_core::HostGpuSummary {
                    name: Some("AMD Instinct MI300X".to_owned()),
                    gfx_target: Some("gfx942".to_owned()),
                    therock_family: Some("gfx94X-dcgpu".to_owned()),
                },
            ),
            (
                "Strix Halo",
                rocm_core::HostGpuSummary {
                    name: Some("AMD Radeon 8060S".to_owned()),
                    gfx_target: Some("gfx1151".to_owned()),
                    therock_family: Some("gfx1151".to_owned()),
                },
            ),
            ("no detected GPU", rocm_core::HostGpuSummary::default()),
        ]
    }

    /// I3 — `rocm diagnose --model` never names an engine `rocm serve` would not
    /// select.
    ///
    /// Two layers agreeing about one decision. The defect this catches is not a
    /// wrong `select_serve_engine`, it is the readiness path re-deriving the
    /// choice from `recipe.preferred_engines` and drifting: that reads correctly
    /// and is wrong on exactly the hosts where serve overrides the recipe. So
    /// both real call sites are driven with the same inputs and compared, rather
    /// than either being called with literal arguments.
    #[test]
    fn the_engine_doctor_names_is_the_engine_serve_would_select() {
        let registry = rocm_core::builtin_model_recipe_registry();
        let host = HostFacts {
            accelerator_memory: AcceleratorMemory::Dedicated(192.0),
            system_ram_gib: Some(1024.0),
        };

        // Non-vacuity: at least one pair must be a case where serve OVERRIDES the
        // recipe's own first preference, because that is the only case a
        // re-derivation would get wrong. Native Windows has no such case --
        // `preferred_serve_engine_for_host_gpu_summary` never prefers vLLM there
        // -- so the check is stated where it exists and the equality assertion
        // below still runs everywhere.
        if !rocm_core::runtime_is_windows() {
            let overridden = engine_selection_hosts().into_iter().any(|(_, summary)| {
                registry.recipes.iter().any(|recipe| {
                    let selected = select_serve_engine(None, None, Some(recipe), Some(&summary));
                    recipe
                        .preferred_engines
                        .first()
                        .is_some_and(|preferred| !preferred.eq_ignore_ascii_case(&selected.engine))
                })
            });
            assert!(
                overridden,
                "no host/recipe pair here exercises serve overriding the recipe's preferred \
                 engine, so the comparison below cannot catch the readiness path re-deriving \
                 the choice itself"
            );
        }

        for (label, summary) in engine_selection_hosts() {
            for recipe in &registry.recipes {
                let expected = select_serve_engine(None, None, Some(recipe), Some(&summary));
                let report = assess_model_for_host(
                    &recipe.canonical_model_id,
                    &ModelCatalogSource::Available(&registry),
                    &host,
                    None,
                    Some(&summary),
                );
                assert_eq!(
                    report.engine.as_deref(),
                    Some(expected.engine.as_str()),
                    "on {label}, `rocm diagnose --model {}` names {:?} but `rocm serve` would \
                     select `{}`",
                    recipe.canonical_model_id,
                    report.engine,
                    expected.engine
                );
            }
        }
    }

    /// When the platform gate rules an engine out, the message names it and
    /// points at WSL/Linux -- the actual escape hatch, not a guess.
    #[test]
    fn unsupported_here_for_names_the_engine_when_ruled_out() {
        let message = unsupported_here_for("vllm", PlatformRuleOut::NativeWindowsVllm)
            .expect("a ruled-out engine gets a message");
        assert!(
            message.contains("vllm"),
            "the message has to name the engine it is talking about: {message:?}"
        );
        assert!(
            message.to_lowercase().contains("wsl") || message.to_lowercase().contains("linux"),
            "the message has to point at the actual escape hatch: {message:?}"
        );
    }

    /// AIM's Docker GPU passthrough only exists on Linux/WSL2, so native
    /// Windows must be ruled out (the Docker-reachability gate alone would pass
    /// on Docker Desktop and launch on CPU).
    #[test]
    fn unsupported_here_for_names_wsl_for_aim_on_native_windows() {
        let message = unsupported_here_for("aim", PlatformRuleOut::NativeWindowsAim)
            .expect("a ruled-out engine gets a message");
        assert!(
            message.contains("aim") && message.contains("WSL"),
            "{message:?}"
        );
    }

    /// When the platform gate allows an engine, there is nothing to report --
    /// asserted directly against the pure helper, not by relying on whichever
    /// OS this test happens to run on.
    #[test]
    fn unsupported_here_for_is_none_when_allowed() {
        assert_eq!(unsupported_here_for("vllm", PlatformRuleOut::Allowed), None);
        assert_eq!(
            unsupported_here_for("llamacpp", PlatformRuleOut::Allowed),
            None
        );
    }

    /// `aim` is ruled out for a different reason than `vllm` on Windows
    /// (Docker, not the OS), so the message has to say Docker, not WSL.
    ///
    /// Pins the wording only; that the gate actually fires on an unreachable
    /// daemon is proven by
    /// `engine_ruled_out_by_platform_gates_aim_on_docker_reachability`.
    #[test]
    fn unsupported_here_for_names_docker_for_aim() {
        let message = unsupported_here_for("aim", PlatformRuleOut::AimDockerUnavailable)
            .expect("a ruled-out engine gets a message");
        assert!(
            message.to_lowercase().contains("docker"),
            "the message has to name the actual reason, Docker: {message:?}"
        );
    }

    /// Child half of [`engine_ruled_out_by_platform_gates_aim_on_docker_reachability`].
    ///
    /// Runs in its own process so the hostile `PATH` (a fake `docker` that
    /// always fails) can be set with `Command::env` -- mutating `PATH`
    /// in-process would need `unsafe`, which the workspace denies.
    #[test]
    fn docker_unreachable_subprocess_helper() {
        if std::env::var_os("ROCM_AIM_DOCKER_UNREACHABLE_TEST_DIR").is_none() {
            return;
        }
        assert_eq!(
            engine_ruled_out_by_platform("aim"),
            PlatformRuleOut::AimDockerUnavailable,
            "docker is unreachable, so the platform gate must rule aim out"
        );
        let message =
            unsupported_here_for("aim", engine_ruled_out_by_platform("aim")).expect("ruled out");
        assert!(
            message.to_lowercase().contains("docker"),
            "the gate and its message must agree: {message:?}"
        );
    }

    /// Fix for the review finding that `unsupported_here_for_names_docker_for_aim`
    /// only pinned the message's wording, never the real
    /// `engine_ruled_out_by_platform("aim")` gate it is paired with in
    /// `assess_model_for_host`. This drives the gate itself against a `docker`
    /// that always fails, rather than hand-feeding it `ruled_out: true`.
    #[cfg(unix)]
    #[test]
    fn engine_ruled_out_by_platform_gates_aim_on_docker_reachability() {
        let dir = std::env::temp_dir().join(format!(
            "rocm-aim-docker-unreachable-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fake_bin = dir.join("bin");
        std::fs::create_dir_all(&fake_bin).unwrap();
        rocm_core::fake_executable::write_fake_executable(
            &fake_bin.join("docker"),
            "#!/bin/sh\nexit 1\n",
        )
        .unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "model_fit::tests::docker_unreachable_subprocess_helper",
                "--nocapture",
            ])
            .env(
                "ROCM_AIM_DOCKER_UNREACHABLE_TEST_DIR",
                std::path::PathBuf::from("1"),
            )
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "helper subprocess failed");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for the engine-consistency bug: `assess_model_on_this_host`
    /// used to call `AppPaths::discover()` twice -- once (implicitly `None`) for
    /// the GPU summary, once for the config lookup a few lines later -- so the
    /// two could read different directories. Writing a `default_engine` to one
    /// known `AppPaths` and driving both the config lookup and the GPU-summary
    /// lookup off that same value pins it down: if a future change reintroduces
    /// a second, independent `AppPaths::discover()` for either half, the
    /// configured engine this test writes to disk stops reaching the verdict.
    #[test]
    fn the_configured_default_engine_reaches_the_readiness_verdict_from_the_same_paths_as_the_gpu_summary()
     {
        let dir = std::env::temp_dir().join(format!(
            "rocm-diagnose-model-engine-consistency-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = AppPaths {
            config_dir: dir.clone(),
            data_dir: dir.clone(),
            cache_dir: dir.clone(),
        };
        let config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..Default::default()
        };
        config
            .save(&paths)
            .expect("write a config.json under the temp AppPaths");

        let registry = rocm_core::builtin_model_recipe_registry();
        let recipe = registry
            .recipes
            .first()
            .expect("the builtin registry ships at least one recipe");
        let examination = rocm_core::Examination::default();

        // `configured_default` outranks the host GPU summary and the recipe's own
        // preference in `select_serve_engine`, so this stays deterministic
        // regardless of what this test happens to run on.
        let summary = detect_host_gpu_summary(Some(&paths));
        let expected = select_serve_engine(None, Some("vllm"), Some(recipe), Some(&summary));

        let readiness =
            assess_model_with_host_paths(&recipe.canonical_model_id, &examination, Some(&paths));

        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            readiness.engine.as_deref(),
            Some(expected.engine.as_str()),
            "the config written under this AppPaths said default_engine = \"vllm\", but the \
             readiness verdict named {:?} -- the config lookup and the GPU-summary lookup must \
             read the same discovered AppPaths",
            readiness.engine
        );
    }

    /// A single dedicated GPU reports its measured total, unmodified.
    #[test]
    fn classify_accelerator_memory_reports_dedicated_vram_for_a_discrete_gpu() {
        let vram = [GpuVramUsage {
            index: 0,
            used_mb: 1024,
            total_mb: 16384,
        }];
        let result = classify_accelerator_memory(Some(&vram), Some("gfx942"), true, None);
        assert_eq!(result, AcceleratorMemory::Dedicated(16.0));
    }

    /// The defect this guards against: an APU's `amd-smi` carve-out figure is
    /// not the pool the engine actually allocates from (the real pool is the
    /// GTT aperture, which nothing in this binary can read), so asserting
    /// installed system RAM as a stand-in used to fabricate a number that
    /// could read `Ready` for a model that does not actually fit. The correct,
    /// honest answer is `UnifiedMemoryUnreadable`, not a number of any kind,
    /// and not the plain `Unknown` used for genuine no-telemetry hosts either
    /// -- confirming this never again quietly regresses to reporting a
    /// fabricated figure (or worse, the APU's own tiny carve-out as
    /// `Dedicated`), and never again collapses back into the generic
    /// `Unknown`, whose `amd-smi` remediation would be a dead end here.
    #[test]
    fn classify_accelerator_memory_reports_unifiedmemoryunreadable_not_total_ram_for_an_apu() {
        let apu_carveout = [GpuVramUsage {
            index: 0,
            used_mb: 2048,
            total_mb: 4096,
        }];
        let result = classify_accelerator_memory(Some(&apu_carveout), Some("gfx1151"), true, None);
        assert_eq!(
            result,
            AcceleratorMemory::UnifiedMemoryUnreadable,
            "an APU's memory pool must read UnifiedMemoryUnreadable, not a fabricated figure, \
             and not the generic Unknown either: {result:?}"
        );
    }

    /// No VRAM telemetry at all, but the examination positively saw an AMD
    /// GPU: unmeasurable, not absent.
    #[test]
    fn classify_accelerator_memory_reports_unknown_when_gpu_present_but_unmeasured() {
        let result = classify_accelerator_memory(None, None, true, None);
        assert_eq!(result, AcceleratorMemory::Unknown);
    }

    /// No VRAM telemetry and no evidence of an AMD GPU: there is nothing to
    /// allocate a model from.
    #[test]
    fn classify_accelerator_memory_reports_none_when_no_gpu_is_present() {
        let result = classify_accelerator_memory(None, None, false, None);
        assert_eq!(result, AcceleratorMemory::None);
    }

    /// An APU paired with a discrete card (more than one GPU reported) keeps
    /// the discrete card's warning meaningful, per `vram_capacity_is_meaningful`
    /// -- the multi-GPU case cannot attribute the gfx target to one ordinal, so
    /// it is never treated as the unified-memory case even when the target
    /// looks like an APU family.
    #[test]
    fn classify_accelerator_memory_treats_multi_gpu_as_dedicated_even_with_apu_gfx_target() {
        let vram = [
            GpuVramUsage {
                index: 0,
                used_mb: 1024,
                total_mb: 4096,
            },
            GpuVramUsage {
                index: 1,
                used_mb: 2048,
                total_mb: 8192,
            },
        ];
        let result = classify_accelerator_memory(Some(&vram), Some("gfx1151"), true, None);
        assert_eq!(result, AcceleratorMemory::Dedicated(8.0));
    }

    /// The defect a reviewer caught: `rocm serve` pins exactly one GPU
    /// ordinal, never the sum of every card, so summing VRAM across GPUs can
    /// report a model READY when it would OOM on every individual card. On an
    /// 8x192 GiB host a sum of ~1536 GiB would clear even the largest curated
    /// recipe's minimum; the true per-card figure (192 GiB) is what the
    /// verdict must compare against. Were this still summing, the assertion
    /// below would see `Dedicated(1536.0)`, not `Dedicated(192.0)`.
    #[test]
    fn classify_accelerator_memory_takes_the_largest_card_not_the_sum() {
        let vram: Vec<GpuVramUsage> = (0..8)
            .map(|index| GpuVramUsage {
                index,
                used_mb: 0,
                total_mb: 192 * 1024,
            })
            .collect();
        let result = classify_accelerator_memory(Some(&vram), Some("gfx942"), true, None);
        assert_eq!(
            result,
            AcceleratorMemory::Dedicated(192.0),
            "a homogeneous 8-GPU host must report one card's capacity, not the sum of all \
             eight, or a model that OOMs on every card reads READY: {result:?}"
        );
    }

    /// A `HIP_VISIBLE_DEVICES`-style mask hides a GPU from `rocm serve` the
    /// same way it hides one from auto-selection (see
    /// `select_auto_gpu_index`'s own mask handling) -- but `amd-smi` enumerates
    /// at the driver level and knows nothing about that mask, so its rows
    /// still include the hidden card. Narrowing by `usable_indices` is what
    /// keeps a masked-out GPU's capacity from inflating the verdict for a
    /// model that could never actually land on it.
    #[test]
    fn classify_accelerator_memory_ignores_a_gpu_masked_out_by_the_visibility_filter() {
        let vram = [
            GpuVramUsage {
                index: 0,
                used_mb: 0,
                total_mb: 8 * 1024,
            },
            GpuVramUsage {
                index: 1,
                used_mb: 0,
                total_mb: 192 * 1024,
            },
        ];
        // Only index 0 is visible: the 192 GiB card at index 1 is masked out.
        let result = classify_accelerator_memory(Some(&vram), Some("gfx942"), true, Some(&[0]));
        assert_eq!(
            result,
            AcceleratorMemory::Dedicated(8.0),
            "a masked-out GPU's VRAM must not count toward the verdict: {result:?}"
        );
    }

    /// The end-to-end proof, one layer above `classify_accelerator_memory`'s own
    /// unit tests: not just that the function picks the right number, but that
    /// picking the wrong one would have changed the verdict a user actually
    /// sees. Two 8 GiB cards sum to 16 GiB, comfortably above `qwen3.5-4b`'s 12
    /// GiB minimum -- a verdict built on the sum would read READY, and `rocm
    /// serve` pins exactly one ordinal, so that model would OOM on every card
    /// this host has. No single card clears 12 GiB, so the correct verdict is
    /// BLOCKED. Were `classify_accelerator_memory` still summing, this would
    /// observe `Ready`, not `Blocked`.
    #[test]
    fn a_host_whose_vram_sum_clears_the_minimum_but_no_single_card_does_is_blocked_not_ready() {
        let vram = [
            GpuVramUsage {
                index: 0,
                used_mb: 0,
                total_mb: 8 * 1024,
            },
            GpuVramUsage {
                index: 1,
                used_mb: 0,
                total_mb: 8 * 1024,
            },
        ];
        let accelerator_memory =
            classify_accelerator_memory(Some(&vram), Some("gfx942"), true, None);
        // Premise check: if this is ever 16.0 (the sum) instead of 8.0 (the
        // largest single card), the scenario below no longer tests what it
        // claims to.
        assert_eq!(accelerator_memory, AcceleratorMemory::Dedicated(8.0));

        let host = HostFacts {
            accelerator_memory,
            system_ram_gib: Some(1024.0),
        };
        let registry = rocm_core::builtin_model_recipe_registry();
        let report = assess_model_for_host(
            "qwen3.5-4b",
            &ModelCatalogSource::Available(&registry),
            &host,
            None,
            None,
        );
        assert_eq!(
            report.verdict,
            rocm_core::model_readiness::ModelVerdict::Blocked,
            "qwen3.5-4b needs 12 GiB; two 8 GiB cards sum to 16 GiB (which would clear it) but \
             no single card does (8 GiB each) -- the verdict must be BLOCKED, not READY, or \
             `rocm serve` would pin one 8 GiB card to a model that does not fit on it: {report:#?}"
        );
    }

    /// `has_amd_gpu` is `false` on a WSL examination as the probe returns it
    /// (the probes that set it are skipped there), and what the WSL host's
    /// display adapter says is not what ROCm can enumerate, so
    /// `host_accelerator_memory` must read `rocm_sees_gpu` instead on a WSL
    /// host -- and only a confirmed `Some(false)` counts as
    /// "no GPU"; `None` means rocminfo was absent, which is not evidence of
    /// anything, per the same reasoning `fix-wsl-6-host-driver-too-old` in
    /// `diagnose.rs` already uses.
    #[test]
    fn examination_reports_amd_gpu_reads_rocm_sees_gpu_on_wsl() {
        let wsl_examination = |rocm_sees_gpu: Option<bool>| rocm_core::Examination {
            is_wsl: true,
            wsl: Some(rocm_core::examine::WslFacts {
                rocm_sees_gpu,
                ..Default::default()
            }),
            ..rocm_core::Examination::default()
        };

        assert!(
            !examination_reports_amd_gpu(&wsl_examination(Some(false))),
            "rocminfo positively enumerating no device must read as no GPU"
        );
        assert!(
            examination_reports_amd_gpu(&wsl_examination(Some(true))),
            "rocminfo positively enumerating a device must read as a GPU present"
        );
        assert!(
            examination_reports_amd_gpu(&wsl_examination(None)),
            "rocminfo absent (the question went unasked) must not read as a confident no"
        );
    }

    /// Off WSL, the bare-metal probes are what set `has_amd_gpu`, so this must
    /// keep using it rather than a WSL-only signal that was never populated.
    #[test]
    fn examination_reports_amd_gpu_reads_has_amd_gpu_off_wsl() {
        let mut examination = rocm_core::Examination::default();
        assert!(!examination_reports_amd_gpu(&examination));
        examination.has_amd_gpu = true;
        assert!(examination_reports_amd_gpu(&examination));
    }
}
