// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Will this model run on this machine?
//!
//! `examine` answers what is on the host and `diagnose` answers what went wrong
//! after a failure. This answers the question that comes before both, from data
//! the CLI already has: a curated [`ModelRecipeRecord`] (dtype, quantization,
//! minimum GPU memory, device policy, preferred engines, declared alternatives)
//! composed with what the host can offer. **It never fetches weights** — which
//! is the whole point: a user should not learn a model will not run by starting
//! a download and waiting for it to fail. The curated path reads nothing beyond
//! what the caller hands over. A GGUF reference *outside* the catalog is the one
//! exception, and a small one: [`attach_context_estimate`] reads the file's
//! header (for a remote file: one repository listing, then a few megabytes by
//! range request) and sizes the
//! model from it. See [`crate::gguf_fit`].
//!
//! Three distinctions carry the design, and collapsing any of them produces a
//! confident wrong answer:
//!
//! - A catalog that could not be **read** is not a model that does not **fit**.
//!   [`ModelCatalogSource`] keeps them apart and [`ModelVerdict::Undetermined`]
//!   is where the first one lands.
//! - Memory the CLI **could not measure** is not memory the host **does not
//!   have**. See [`AcceleratorMemory`].
//! - A measured **ceiling** on memory shared with the operating system is not a
//!   dedicated pool. An APU's engine allocates from system memory, and the
//!   figure that bounds it is readable ([`AcceleratorMemory::Unified`]) — but it
//!   describes what the machine would allow at best, not what is free, so a
//!   model that fits under it is [`ModelVerdict::Degraded`], never
//!   [`ModelVerdict::Ready`]. Where even that figure is missing, the `amd-smi`
//!   carve-out reading is still the wrong pool and still has no remediation, so
//!   it stays apart from "no telemetry at all":
//!   [`AcceleratorMemory::UnifiedMemoryUnreadable`].
//!
//! A model outside the catalog has no recipe, so it is reported
//! [`UndeterminedReason::ModelNotCurated`], never blocked. When it is a GGUF the
//! header allows an *estimate* (weights, KV cache at a context size, compute
//! buffer) against dedicated VRAM, carried in [`ModelReadiness::estimate`] under
//! its own label rather than as a verdict: an estimate from a header is not a
//! recipe's tested minimum.

use crate::ModelRecipeRecord;
use crate::ModelRecipeRegistry;
use crate::diagnose::{Fix, Route, upstream_route};
use crate::gguf::parse_gguf_header;
use crate::gguf_fit::{
    CannotEstimateReason, ContextEstimate, ESTIMATE_CAVEAT, EstimateOutcome, FitRequest,
    KvCacheType, SERVE_CONTEXT_NOTE, SERVE_OPTIONS_NOTE, estimate_fit, evidence_lines, tokens,
};
use crate::gguf_source::{GgufRef, GgufSource, ProjectorNamed, parse_gguf_ref};
use serde::{Deserialize, Serialize};

/// How many fallback alternatives to offer when a recipe declares none that fit.
///
/// Matches what `rocm model --verbose` has always offered; the cap exists so a
/// refusal stays readable, not because more could not be found.
const MAX_FALLBACK_ALTERNATIVES: usize = 3;

/// Whether the model will run here.
///
/// Four states, deliberately not three. Folding `Undetermined` into `Blocked`
/// would report "this machine cannot serve that model" when the truth is "the
/// CLI could not find out", which reads exactly like a real answer and sends the
/// user looking for hardware they may not need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVerdict {
    /// It will run, on the named engine.
    Ready,
    /// It will run, but below what the recipe recommends.
    Degraded,
    /// It will not run, and the evidence says why.
    Blocked,
    /// The CLI could not find out. See [`ModelReadiness::undetermined_reason`].
    Undetermined,
}

/// Why no verdict about the model could be reached.
///
/// Serialized so a caller can tell a local misconfiguration (the first two) from
/// a gap in host telemetry (the third) without parsing prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndeterminedReason {
    /// The recipe catalog itself could not be read or verified.
    CatalogUnreachable,
    /// The catalog was read and carries no recipe for this model.
    ModelNotCurated,
    /// The host's accelerator memory could not be measured.
    AcceleratorMemoryUnknown,
    /// The host is an APU, and the figure for the pool its engine allocates
    /// from was absent or unparseable on this machine.
    ///
    /// The pool itself *is* readable -- `mem_info_gtt_total` on Linux,
    /// `DXGI_ADAPTER_DESC1::SharedSystemMemory` on Windows, both read by
    /// [`crate::host_gpu::detect_host_unified_memory_gib`] -- so a host that
    /// lands here has a specific, local gap: an older kernel that does not
    /// expose the file, a container that does not pass the DRM device through,
    /// or a driver that answered with a figure too small to be a pool. The
    /// `amd-smi` carve-out reading is still there and still names the wrong
    /// pool, which is why this stays distinct from
    /// [`Self::AcceleratorMemoryUnknown`]: running `amd-smi` again would answer
    /// identically forever, so that remediation is not offered here.
    UnifiedMemoryUnreadable,
}

/// The memory pool an engine would allocate this model from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcceleratorMemory {
    /// Dedicated VRAM, in GiB. What a discrete card reports is what it has.
    Dedicated(f64),
    /// No GPU is visible to ROCm on this host.
    None,
    /// There is a GPU, but its memory could not be read at all -- no
    /// telemetry, positive or otherwise.
    ///
    /// Distinct from [`AcceleratorMemory::None`] on purpose: one is a fact about
    /// the machine and the other is a gap in what the CLI can see, and they do
    /// not deserve the same verdict.
    Unknown,
    /// An APU, whose engine allocates from a pool of system memory shared with
    /// the operating system rather than from private VRAM.
    ///
    /// `gtt_total_gib` is that pool's measured ceiling: the kernel's
    /// `mem_info_gtt_total` aperture cap on Linux, the WDDM
    /// `DXGI_ADAPTER_DESC1::SharedSystemMemory` budget on Windows. Both are read
    /// by [`crate::host_gpu::detect_host_unified_memory_gib`], and both are
    /// measurements of driver and BIOS policy on this machine -- not installed
    /// RAM, and not a fraction of it.
    ///
    /// `vram_carveout_gib` is what `amd-smi` reported as `total_vram`, kept only
    /// so a report can show the two side by side. It is the fixed BIOS carve-out
    /// (512 MiB on a Ryzen 7 4700U, ~4 GiB on a Strix Halo) and is never the
    /// figure a recipe minimum is compared against: doing so refuses nearly
    /// every curated model on machines that serve them.
    ///
    /// A ceiling is not an allocation. The operating system, the compositor and
    /// every other process draw on the same memory, so this can never reach
    /// [`ModelVerdict::Ready`] -- see `assess_model_readiness`.
    Unified {
        gtt_total_gib: f64,
        vram_carveout_gib: Option<f64>,
    },
    /// An APU whose pool could not be read on this particular host: the figure
    /// was absent or unparseable. See
    /// [`UndeterminedReason::UnifiedMemoryUnreadable`] for what that means and
    /// why it is still not [`AcceleratorMemory::Unknown`].
    UnifiedMemoryUnreadable,
}

impl AcceleratorMemory {
    /// The figure to compare a recipe minimum against, when there is one.
    ///
    /// A unified pool answers with its ceiling. That is a real measurement and a
    /// comparison against it is worth making -- a model above the aperture
    /// cannot fit however free the machine is -- but clearing it does not make
    /// the model `Ready`, because the pool is shared. The verdict side of that
    /// is in `assess_model_readiness`, not here.
    pub(crate) const fn measured_gib(self) -> Option<f64> {
        match self {
            Self::Dedicated(gib)
            | Self::Unified {
                gtt_total_gib: gib, ..
            } => Some(gib),
            Self::None | Self::Unknown | Self::UnifiedMemoryUnreadable => None,
        }
    }

    /// Whether the figure above is a ceiling on memory shared with the
    /// operating system rather than a dedicated pool.
    pub(crate) const fn is_shared_pool(self) -> bool {
        matches!(self, Self::Unified { .. })
    }
}

/// The engine `serve` would select for a recipe on this host.
///
/// Supplied by the caller rather than derived here. Engine selection is one
/// decision with one implementation (`select_serve_engine` in the `rocm`
/// binary); re-deriving it from `preferred_engines` reads correctly and is wrong
/// on exactly the hosts where serve overrides the recipe's preference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostEngineChoice {
    pub engine: String,
    /// Why that engine, in serve's own words.
    pub source: String,
    /// Set when the platform gate rules this engine out here (for example vLLM
    /// on native Windows). `serve` does not silently pick another one, so
    /// neither does this.
    pub unsupported_here: Option<String>,
}

/// What the host can offer, as far as this question needs.
#[derive(Debug, Clone, PartialEq)]
pub struct HostFacts {
    pub accelerator_memory: AcceleratorMemory,
    /// Host system RAM in GiB, or `None` when it could not be read. Advisory:
    /// it can only soften a verdict to [`ModelVerdict::Degraded`], never decide
    /// one, so not knowing it does not make the answer undeterminable.
    pub system_ram_gib: Option<f64>,
}

/// The recipe catalog, or the reason there isn't one.
pub enum ModelCatalogSource<'a> {
    Available(&'a ModelRecipeRegistry),
    /// The index could not be read, parsed, or signature-verified.
    Unreachable {
        detail: String,
    },
}

/// A curated model offered in place of one that will not run here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelAlternative {
    /// The reference to pass to `rocm serve`.
    pub model_ref: String,
    pub required_gpu_memory_gib: Option<f64>,
    pub engine: String,
}

/// The answer to "will this model run here".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelReadiness {
    /// What the user asked about, verbatim.
    pub model_ref: String,
    /// Set once a recipe was matched; `None` means no recipe was ever read, so
    /// nothing below describes the model itself.
    pub canonical_model_id: Option<String>,
    pub verdict: ModelVerdict,
    /// What the verdict was reached from, one fact per line.
    pub evidence: Vec<String>,
    pub engine: Option<String>,
    pub engine_source: Option<String>,
    pub required_gpu_memory_gib: Option<f64>,
    pub available_gpu_memory_gib: Option<f64>,
    pub alternatives: Vec<ModelAlternative>,
    /// What to do about it, in the same shape `rocm diagnose` uses for a cause.
    ///
    /// Carries no `fix_id`: there is no catalog entry behind it, so `rocm fix`
    /// would reject one, and rendering an `apply with:` line here would name a
    /// command that cannot run.
    pub fix: Option<Fix>,
    /// Where to report it, when the answer is one this CLI cannot give.
    pub route: Option<Route>,
    pub undetermined_reason: Option<UndeterminedReason>,
    /// The recipe's own warnings, passed through unchanged.
    ///
    /// Informational — every curated recipe carries at least one — so they do
    /// not move the verdict. Degradation is decided from measurements.
    pub warnings: Vec<String>,
    /// A memory estimate from a GGUF file's header, for a model outside the
    /// catalog. Absent for curated models and for references that are not GGUF.
    ///
    /// Deliberately not a [`ModelVerdict`]: `verdict` stays `undetermined`, and
    /// this carries its own outcome, so no reader mistakes an estimate for a
    /// recipe's tested minimum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate: Option<ContextEstimate>,
}

impl ModelReadiness {
    /// An answer reached before any recipe was read.
    ///
    /// Every field describing the comparison is left empty, including the host's
    /// own measured memory: they are read as one set, and a figure sitting
    /// beside a null requirement invites a reader to supply the missing half
    /// themselves. `AcceleratorMemoryUnknown` does not come through here — a
    /// recipe *was* read in that case, and what it needs is worth saying even
    /// when the host side of the comparison is missing.
    fn undetermined(model_ref: &str, reason: UndeterminedReason, evidence: Vec<String>) -> Self {
        Self {
            model_ref: model_ref.to_owned(),
            canonical_model_id: None,
            verdict: ModelVerdict::Undetermined,
            evidence,
            engine: None,
            engine_source: None,
            required_gpu_memory_gib: None,
            available_gpu_memory_gib: None,
            alternatives: Vec::new(),
            fix: None,
            route: None,
            undetermined_reason: Some(reason),
            warnings: Vec::new(),
            estimate: None,
        }
    }
}

/// Pick curated models to offer in place of one that will not run.
///
/// The recipe's declared `manual_alternatives` first, in the order it lists
/// them; only when none of those survive `fits` does it fall back to other
/// curated recipes for the same task, capped at
/// [`MAX_FALLBACK_ALTERNATIVES`]. The cap applies to the fallback alone, because
/// a recipe author listing four alternatives meant all four.
///
/// `fits` is the caller's, because callers mean different things by it: `rocm
/// model --verbose` asks only whether the GPU-memory minimum is met, while
/// `rocm diagnose --model` asks the whole readiness question. Sharing the
/// *selection policy* while keeping the predicates apart is deliberate — what
/// must not exist twice is the ordering and the fallback rule.
///
/// Returns each candidate paired with the reference to name it by: the string
/// the recipe declared, or the candidate's own display alias for fallbacks.
#[must_use]
pub fn curated_alternatives<'a>(
    recipe: Option<&'a ModelRecipeRecord>,
    catalog: &'a [ModelRecipeRecord],
    fits: &dyn Fn(&ModelRecipeRecord) -> bool,
) -> Vec<(&'a str, &'a ModelRecipeRecord)> {
    let declared = recipe
        .map(|recipe| {
            recipe
                .manual_alternatives
                .iter()
                .filter_map(|candidate_ref| {
                    catalog
                        .iter()
                        .find(|candidate| candidate.matches_ref(candidate_ref))
                        .map(|candidate| (candidate_ref.as_str(), candidate))
                })
                .filter(|(_, candidate)| fits(candidate))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !declared.is_empty() {
        return declared;
    }
    catalog
        .iter()
        .filter(|candidate| {
            recipe.is_none_or(|recipe| candidate.canonical_model_id != recipe.canonical_model_id)
        })
        .filter(|candidate| recipe.is_none_or(|recipe| candidate.task == recipe.task))
        .filter(|candidate| fits(candidate))
        .take(MAX_FALLBACK_ALTERNATIVES)
        .map(|candidate| (recipe_display_ref(candidate), candidate))
        .collect()
}

/// The shortest reference a user can type for a recipe.
#[must_use]
pub fn recipe_display_ref(recipe: &ModelRecipeRecord) -> &str {
    recipe
        .aliases
        .first()
        .map_or(recipe.canonical_model_id.as_str(), String::as_str)
}

/// Whether this recipe needs a GPU at all.
fn requires_gpu(recipe: &ModelRecipeRecord) -> bool {
    recipe.device_policy != "cpu_only"
}

/// Format a GiB figure for a verdict's evidence text.
///
/// The fractional branch floors to one decimal rather than rounding to
/// nearest. A measured figure (`available`/`actual`) sits just under its
/// nameplate size -- an 8 GiB card reports 8176 MiB, which is 7.9844 GiB -- and
/// rounding to nearest would print "8.0 GiB" against an "8 GiB" recipe minimum
/// on a BLOCKED verdict: evidence that reads as "the host has exactly what the
/// recipe needs" while the verdict says otherwise. `required`/`recommended`
/// values come from `u32` catalog fields, so they are always exact integers
/// and never reach this branch; flooring only ever shades a measured figure
/// down, never a minimum.
pub(crate) fn format_gib(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0} GiB")
    } else {
        format!("{:.1} GiB", (value * 10.0).floor() / 10.0)
    }
}

/// Answer "will this model run here", from the catalog and the host alone.
///
/// Reads no file the caller did not already hand over and opens no socket. The
/// `engine_for` closure is how the caller's engine decision reaches this without
/// being copied into it.
#[must_use]
pub fn assess_model_readiness(
    model_ref: &str,
    catalog: &ModelCatalogSource<'_>,
    host: &HostFacts,
    engine_for: &dyn Fn(&ModelRecipeRecord) -> HostEngineChoice,
) -> ModelReadiness {
    assess(model_ref, catalog, host, engine_for, true)
}

/// The body of [`assess_model_readiness`], with alternatives made optional.
///
/// `offer_alternatives` is what terminates the recursion, and it is not an
/// optimisation. Deciding whether an alternative is worth offering means asking
/// the same question about it, and the catalog's declared alternatives point
/// both ways — `qwen` offers `qwen-tiny`, `qwen-tiny` offers `qwen`. On a host
/// where both are blocked, an assessment that recursed freely would follow that
/// cycle until the stack ran out. A candidate is therefore assessed as a model
/// and never as a source of further candidates: the recursion is exactly one
/// level deep by construction.
fn assess(
    model_ref: &str,
    catalog: &ModelCatalogSource<'_>,
    host: &HostFacts,
    engine_for: &dyn Fn(&ModelRecipeRecord) -> HostEngineChoice,
    offer_alternatives: bool,
) -> ModelReadiness {
    let registry = match catalog {
        // The first of the three collapses this module exists to prevent. A
        // signed index that is missing, unreadable or fails verification says
        // nothing whatsoever about the model, so nothing about the model is
        // reported -- not even the minimum it requires, which was never read.
        ModelCatalogSource::Unreachable { detail } => {
            let mut readiness = ModelReadiness::undetermined(
                model_ref,
                UndeterminedReason::CatalogUnreachable,
                vec![
                    format!("the model recipe catalog could not be read: {detail}"),
                    format!(
                        "this is a fact about the catalog, not about `{model_ref}`; the recipe \
                         was never read, so nothing here says whether the model fits"
                    ),
                ],
            );
            readiness.fix = Some(Fix {
                summary: "restore the recipe catalog, then ask again".to_owned(),
                verify: "rocm model".to_owned(),
                notes: vec![
                    "ROCM_CLI_MODEL_RECIPE_INDEX_PATH selects a signed index; unset it to fall \
                     back to the catalog built into this binary"
                        .to_owned(),
                    "a configured index also needs \
                     ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH, and its signature sidecar \
                     beside it"
                        .to_owned(),
                ],
                ..Fix::default()
            });
            return readiness;
        }
        ModelCatalogSource::Available(registry) => *registry,
    };

    let Some(recipe) = registry
        .recipes
        .iter()
        .find(|recipe| recipe.matches_ref(model_ref))
    else {
        // Not "this model will not run" -- the catalog simply has no recipe for
        // it, and answering for an arbitrary hub model is out of scope. What can
        // honestly be offered is what the catalog does carry that runs here.
        let mut readiness = ModelReadiness::undetermined(
            model_ref,
            UndeterminedReason::ModelNotCurated,
            vec![
                format!("`{model_ref}` is not in the curated recipe catalog"),
                "there is no recipe for it, so no tested GPU memory minimum to compare; a GGUF \
                 reference (a local .gguf path, or `owner/repo:QUANT` on Hugging Face) is \
                 estimated from its file header instead"
                    .to_owned(),
            ],
        );
        if offer_alternatives {
            readiness.alternatives = alternatives_for(None, registry, host, engine_for);
        }
        readiness.route = Some(upstream_route("rocm-core"));
        readiness.fix = Some(Fix {
            summary: "ask about a curated recipe, or serve this model directly and read the \
                      engine's own refusal"
                .to_owned(),
            commands: vec!["rocm model".to_owned()],
            verify: format!("rocm serve {model_ref}"),
            notes: vec![
                "`rocm serve` accepts models outside this catalog; for one that is not a GGUF \
                 reference this cannot say in advance whether it will fit"
                    .to_owned(),
            ],
            ..Fix::default()
        });
        return readiness;
    };

    let choice = engine_for(recipe);
    let required = recipe.min_gpu_mem_gb.map(f64::from);
    let available = host.accelerator_memory.measured_gib();
    let mut evidence = Vec::new();
    evidence.push(format!(
        "recipe {} ({}, {})",
        recipe.canonical_model_id,
        recipe.dtype,
        recipe
            .quantization
            .as_deref()
            .unwrap_or("no quantization declared")
    ));
    evidence.push(format!(
        "{} would serve it ({})",
        choice.engine, choice.source
    ));
    let mut readiness = ModelReadiness {
        model_ref: model_ref.to_owned(),
        canonical_model_id: Some(recipe.canonical_model_id.clone()),
        verdict: ModelVerdict::Ready,
        evidence,
        engine: Some(choice.engine.clone()),
        engine_source: Some(choice.source.clone()),
        required_gpu_memory_gib: required,
        available_gpu_memory_gib: available,
        alternatives: Vec::new(),
        fix: None,
        route: None,
        undetermined_reason: None,
        warnings: recipe.warnings.clone(),
        estimate: None,
    };

    // Ordered by what pre-empts what. The engine gate comes first because a
    // model that fits in memory still will not run on an engine this platform
    // has no adapter for, and reporting the memory verdict there would be true
    // and useless.
    if let Some(reason) = &choice.unsupported_here {
        readiness.verdict = ModelVerdict::Blocked;
        readiness.evidence.push(reason.clone());
        if offer_alternatives {
            readiness.alternatives = alternatives_for(Some(recipe), registry, host, engine_for);
        }
        readiness.fix = Some(blocked_fix(
            "serve this model from a platform that has the engine",
            &readiness.alternatives,
            model_ref,
            vec![reason.clone()],
        ));
        return readiness;
    }

    if requires_gpu(recipe) && matches!(host.accelerator_memory, AcceleratorMemory::None) {
        readiness.verdict = ModelVerdict::Blocked;
        readiness.evidence.push(format!(
            "no GPU is visible to ROCm, and this recipe's device policy is `{}`; there is no CPU \
             fallback",
            recipe.device_policy
        ));
        readiness.fix = Some(no_gpu_fix(format!("rocm diagnose --model {model_ref}")));
        return readiness;
    }

    match (required, available) {
        // The second collapse: memory that could not be measured is not memory
        // the host does not have. Blocking here would name the model as the
        // problem when the gap is in the host telemetry.
        (Some(required), None)
            if matches!(
                host.accelerator_memory,
                AcceleratorMemory::UnifiedMemoryUnreadable
            ) =>
        {
            // The APU case where even the aperture could not be read. The
            // `amd-smi` telemetry is not missing, it names the wrong pool, and
            // offering "run amd-smi again" would be false -- it already ran,
            // answered, and will answer identically forever.
            readiness.verdict = ModelVerdict::Undetermined;
            readiness.undetermined_reason = Some(UndeterminedReason::UnifiedMemoryUnreadable);
            readiness.evidence.push(format!(
                "this recipe needs {}, but this host has no dedicated VRAM and the figure for \
                 the memory pool its engine allocates from could not be read here; there is \
                 nothing to run today that would change this answer",
                format_gib(required)
            ));
            readiness.fix = None;
        }
        (Some(required), None) => {
            readiness.verdict = ModelVerdict::Undetermined;
            readiness.undetermined_reason = Some(UndeterminedReason::AcceleratorMemoryUnknown);
            readiness.evidence.push(format!(
                "this recipe needs {}, but the GPU memory on this host could not be read; that \
                 is a gap in what the CLI can see, not a verdict about the model",
                format_gib(required)
            ));
            readiness.fix = Some(read_gpu_memory_fix(model_ref));
        }
        (Some(required), Some(available)) if available < required => {
            readiness.verdict = ModelVerdict::Blocked;
            readiness.evidence.push(format!(
                "this recipe needs {}, and this host offers {}",
                format_gib(required),
                format_gib(available)
            ));
            // A ceiling below the minimum blocks as firmly as a small card
            // does: nothing frees memory past an aperture the driver will not
            // hand out. Said plainly so the figure is not mistaken for
            // occupancy a reboot would clear.
            if host.accelerator_memory.is_shared_pool() {
                readiness.evidence.push(
                    "that figure is the whole pool this host's integrated GPU may draw on, so \
                     freeing memory would not raise it"
                        .to_owned(),
                );
            }
            if offer_alternatives {
                readiness.alternatives = alternatives_for(Some(recipe), registry, host, engine_for);
            }
            readiness.fix = Some(blocked_fix(
                "serve a curated model that fits this machine",
                &readiness.alternatives,
                model_ref,
                Vec::new(),
            ));
        }
        (Some(required), Some(available)) => {
            readiness.evidence.push(format!(
                "this recipe needs {}, and this host offers {}",
                format_gib(required),
                format_gib(available)
            ));
        }
        (None, _) => {
            readiness.evidence.push(format!(
                "this recipe declares no GPU memory minimum (device policy `{}`)",
                recipe.device_policy
            ));
        }
    }

    if readiness.verdict == ModelVerdict::Ready
        && let (Some(recommended), Some(actual)) = (
            recipe.recommended_system_ram_gb.map(f64::from),
            host.system_ram_gib,
        )
        && actual < recommended
    {
        readiness.verdict = ModelVerdict::Degraded;
        readiness.evidence.push(format!(
            "this recipe recommends {} of system RAM and this host has {}; it will run, with \
             slower loading and less headroom",
            format_gib(recommended),
            format_gib(actual)
        ));
        readiness.fix = Some(Fix {
            summary: "expect a slower load, or free system RAM before serving".to_owned(),
            verify: format!("rocm diagnose --model {model_ref}"),
            ..Fix::default()
        });
    }

    // A measured ceiling on a shared pool is not an allocation. The operating
    // system, the compositor and every other process draw on the same memory,
    // and nothing here measured how much of it is free, so clearing the
    // comparison above is as far as this can go. `Degraded` is exactly the word
    // for "it will run, but below what the recipe assumes it has", and unlike
    // `Ready` it does not promise a load that may in fact fail.
    //
    // Last, so it never speaks over `Blocked` or `Undetermined` -- those
    // already answered -- and so a host the system-RAM check above has already
    // softened still gets this said about it, keeping that check's own
    // remediation rather than replacing it with one about a different problem.
    if host.accelerator_memory.is_shared_pool()
        && matches!(
            readiness.verdict,
            ModelVerdict::Ready | ModelVerdict::Degraded
        )
    {
        let was_ready = readiness.verdict == ModelVerdict::Ready;
        readiness.verdict = ModelVerdict::Degraded;
        readiness.evidence.push(
            "this host's integrated GPU has no dedicated VRAM: that figure is the ceiling on a \
             pool it shares with the operating system, not memory set aside for the model, so \
             how much of it is free depends on what else is running"
                .to_owned(),
        );
        if was_ready {
            readiness.fix = Some(Fix {
                summary: "close what else is using memory before serving; the engine, not this \
                          check, has the last word on whether it loads"
                    .to_owned(),
                verify: format!("rocm diagnose --model {model_ref}"),
                ..Fix::default()
            });
        }
    }

    readiness
}

/// A remediation for a model that will not run, built from what would.
fn blocked_fix(
    summary: &str,
    alternatives: &[ModelAlternative],
    model_ref: &str,
    mut notes: Vec<String>,
) -> Fix {
    if alternatives.is_empty() {
        notes.push(
            "no curated recipe for this task runs on this host either; `rocm model` lists the \
             whole catalog with its requirements"
                .to_owned(),
        );
        return Fix {
            summary: summary.to_owned(),
            commands: vec!["rocm model".to_owned()],
            verify: format!("rocm diagnose --model {model_ref}"),
            notes,
            ..Fix::default()
        };
    }
    Fix {
        summary: summary.to_owned(),
        commands: alternatives
            .iter()
            .map(|alternative| format!("rocm serve {}", alternative.model_ref))
            .collect(),
        // Whichever one the user picks, this is how they confirm it before
        // committing to a download -- the command this whole feature exists to
        // put in front of that decision.
        verify: format!("rocm diagnose --model {}", alternatives[0].model_ref),
        notes,
        ..Fix::default()
    }
}

/// Curated models that would actually run here.
///
/// The predicate is the whole readiness question, asked against the same host:
/// an alternative is only worth naming if asking about it would come back ready.
/// See [`assess`] for why the inner call is the one that does not recurse.
fn alternatives_for(
    recipe: Option<&ModelRecipeRecord>,
    registry: &ModelRecipeRegistry,
    host: &HostFacts,
    engine_for: &dyn Fn(&ModelRecipeRecord) -> HostEngineChoice,
) -> Vec<ModelAlternative> {
    let verdict_for = |candidate: &ModelRecipeRecord| {
        assess(
            &candidate.canonical_model_id,
            &ModelCatalogSource::Available(registry),
            host,
            engine_for,
            false,
        )
        .verdict
    };
    // Ready first. Degraded is a legitimate suggestion -- it runs -- but on a
    // host where something runs well, saying so is better than offering a
    // compromise. Blocked and Undetermined candidates are never offered at all:
    // pointing a user at a second model they also cannot run is the failure this
    // whole branch exists to avoid.
    let ready = curated_alternatives(recipe, &registry.recipes, &|candidate| {
        verdict_for(candidate) == ModelVerdict::Ready
    });
    let chosen = if ready.is_empty() {
        curated_alternatives(recipe, &registry.recipes, &|candidate| {
            matches!(
                verdict_for(candidate),
                ModelVerdict::Ready | ModelVerdict::Degraded
            )
        })
    } else {
        ready
    };
    chosen
        .into_iter()
        .map(|(candidate_ref, candidate)| ModelAlternative {
            model_ref: candidate_ref.to_owned(),
            required_gpu_memory_gib: candidate.min_gpu_mem_gb.map(f64::from),
            engine: engine_for(candidate).engine,
        })
        .collect()
}

/// The context size an estimate is made for when none was asked for.
pub const DEFAULT_CONTEXT_SIZE: u64 = 4096;

/// What the user asked of a context-size estimate.
///
/// Each field is `None` when the flag was not passed, which is not the same as
/// passing the default: it decides whether a curated model, which has no
/// context estimate, should say so.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContextRequest {
    pub context_size: Option<u64>,
    pub kv_cache_type: Option<KvCacheType>,
}

/// Size a GGUF model outside the catalog from its file header.
///
/// Acts only on an answer of [`UndeterminedReason::ModelNotCurated`] whose
/// reference names a GGUF. Every other answer keeps its verdict, plan and
/// alternatives; the only thing added to it is, when `--context-size` or
/// `--kv-cache-type` was passed, an evidence line saying why the flag did not
/// apply. The verdict stays `Undetermined`: what is added is [`ModelReadiness::estimate`]
/// and the evidence that goes with it. `source` is the only thing that reads a
/// file or the network, and it reads the header, never the weights.
///
/// Three steps, each its own function: [`plan_context_estimate`] decides whether
/// there is a file to read (and, when there is not, what to say about flags that
/// were passed), [`read_context_estimate`] reads the header and sizes it, and
/// [`apply_context_estimate`] rewrites the readiness for what came back.
pub fn attach_context_estimate(
    readiness: &mut ModelReadiness,
    request: &ContextRequest,
    host: &HostFacts,
    source: &dyn GgufSource,
) {
    match plan_context_estimate(readiness, request) {
        ContextPlan::LeaveAsIs { note } => readiness.evidence.extend(note),
        ContextPlan::Estimate {
            reference,
            fit_request,
        } => {
            let (estimate, refused_models) =
                read_context_estimate(&reference, &readiness.model_ref, &fit_request, host, source);
            apply_context_estimate(
                readiness,
                &reference,
                estimate,
                refused_models.as_deref(),
                request,
            );
        }
    }
}

/// What [`plan_context_estimate`] decided for an answer.
#[derive(Debug, PartialEq)]
enum ContextPlan {
    /// Nothing is read and the answer keeps its shape. `note` is the evidence
    /// line that says why a context flag the user passed was not applied; `None`
    /// when no flag was passed, or for an answer that owes no explanation.
    LeaveAsIs { note: Option<String> },
    /// Read this GGUF's header and size it for this request.
    Estimate {
        reference: GgufRef,
        fit_request: FitRequest,
    },
}

/// Decide whether `readiness` is owed an estimate, without reading anything.
fn plan_context_estimate(readiness: &ModelReadiness, request: &ContextRequest) -> ContextPlan {
    let asked = request.context_size.is_some() || request.kv_cache_type.is_some();
    let leave = |note: String| ContextPlan::LeaveAsIs {
        note: asked.then_some(note),
    };
    if readiness.canonical_model_id.is_some() {
        return leave(
            "a context size and KV cache type do not apply to a curated recipe, whose GPU memory \
             minimum is fixed; estimates for them cover GGUF models outside the catalog"
                .to_owned(),
        );
    }
    if readiness.undetermined_reason == Some(UndeterminedReason::CatalogUnreachable) {
        // Whether the reference names a curated model is exactly what could not be
        // found out, so there is no telling which answer it is owed: say that the
        // flags were not applied rather than dropping them without a word.
        return leave(
            "the context size and KV cache type were not applied: the recipe catalog could \
             not be read, so it is not known whether this is a curated model or a GGUF \
             outside the catalog"
                .to_owned(),
        );
    }
    // Every answer that reaches here today is a catalog miss: the other
    // undetermined answers carry the recipe they were about, and are left above.
    // This keeps a future recipe-less answer from being estimated by accident.
    if readiness.undetermined_reason != Some(UndeterminedReason::ModelNotCurated) {
        return ContextPlan::LeaveAsIs { note: None };
    }
    let Some(reference) = parse_gguf_ref(&readiness.model_ref) else {
        return leave(
            "a context size and KV cache type are estimated for GGUF models only: a local \
             .gguf path, or `owner/repo:QUANT` on Hugging Face"
                .to_owned(),
        );
    };
    // `owner/repo` with no quantization names a repository, not a file, and a
    // repository is not read (every file in it is a different model). Without a
    // context flag the answer is the plain "not in the catalog" it always was,
    // reached without touching the network; with one, the user is told what
    // would make it answerable.
    if matches!(reference, GgufRef::Hub { variant: None, .. }) {
        return leave(format!(
            "`{}` names a repository, not a model file: add `:QUANT` (for example \
             `:Q4_K_M`) to estimate one of its GGUF files",
            readiness.model_ref
        ));
    }
    ContextPlan::Estimate {
        reference,
        fit_request: FitRequest {
            context_size: request.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE),
            context_size_defaulted: request.context_size.is_none(),
            kv_cache_type: request.kv_cache_type.unwrap_or(KvCacheType::F16),
        },
    }
}

/// Read `reference`'s header through `source` and size it against `host`. A
/// header that cannot be reached or read is an estimate with no figures, named
/// after `model_ref` as the user wrote it. When the name was refused as a
/// multimodal projector, the repository's model files come back as well, for
/// the plan to offer one.
fn read_context_estimate(
    reference: &GgufRef,
    model_ref: &str,
    fit_request: &FitRequest,
    host: &HostFacts,
    source: &dyn GgufSource,
) -> (ContextEstimate, Option<Vec<String>>) {
    let read = source.open(reference).and_then(|mut opened| {
        let header =
            parse_gguf_header(&mut opened.reader).map_err(|error| anyhow::anyhow!("{error}"))?;
        Ok((header, opened.size, opened.description))
    });
    match read {
        Ok((header, size, description)) => (
            estimate_fit(
                &header,
                size,
                &description,
                fit_request,
                host.accelerator_memory,
                host.system_ram_gib,
            ),
            None,
        ),
        Err(error) => {
            let refused_models = error
                .downcast_ref::<ProjectorNamed>()
                .map(|refusal| refusal.models.clone());
            (
                ContextEstimate::unreadable(model_ref, fit_request, &error),
                refused_models,
            )
        }
    }
}

/// Put `estimate` on `readiness`, and rewrite the parts of the answer that were
/// written for a model nobody had looked at: the route, the fix, the
/// alternatives and the warnings, each according to what the estimate found.
fn apply_context_estimate(
    readiness: &mut ModelReadiness,
    reference: &GgufRef,
    estimate: ContextEstimate,
    refused_models: Option<&[String]>,
    request: &ContextRequest,
) {
    readiness.evidence.extend(evidence_lines(&estimate));
    let ask_again = |model_ref: &str| rediagnose(model_ref, request);
    // The route sends a user to the upstream tracker because the catalog lacks
    // the model. A GGUF was looked at directly, so that is no longer the
    // question -- and for a path that does not exist or a hub that did not
    // answer, the tracker is the wrong place to take it.
    readiness.route = None;
    if estimate.total_gib.is_some() {
        // The header was read and sized, so the "serve it directly and read the
        // engine's refusal" plan answered a question that no longer applies --
        // whether or not this host could be compared against. A header that could
        // not be read gets a plan of its own below (check the path, or the name
        // and the network), as does a hub name refused as a projector; one that
        // was read but could not be sized keeps the "serve it" plan, pointed at a
        // command that accepts the reference.
        // The host gaps with a next step get the plan a curated model gets:
        // memory that could not be read is looked into with `amd-smi`, and no
        // visible GPU with `rocm diagnose`. An APU's unreadable pool has no
        // command that changes the answer, and gets none.
        readiness.fix = match estimate.cannot_estimate_reason {
            Some(CannotEstimateReason::AcceleratorMemoryUnknown) => {
                let mut fix = read_gpu_memory_fix(&readiness.model_ref);
                fix.verify = ask_again(&readiness.model_ref);
                Some(fix)
            }
            Some(CannotEstimateReason::NoGpu) => Some(no_gpu_fix(ask_again(&readiness.model_ref))),
            // A layout whose figure can be too low (state-space, hybrid,
            // mixture-of-experts, or a header missing a width) was sized and
            // compared, and its detail tells the reader to compare the total
            // with the GPU's memory and leave headroom. No command would turn it
            // into a verdict, so there is no plan to offer.
            Some(
                CannotEstimateReason::UnmeasuredLayout | CannotEstimateReason::MissingMetadata,
            ) => None,
            // An APU's pool cannot be read and no command changes that; a fit or
            // a misfit is the answer itself, with its suggestions.
            _ => None,
        };
        if estimate.outcome == EstimateOutcome::Fits {
            readiness.alternatives.clear();
        }
        readiness.warnings.push(ESTIMATE_CAVEAT.to_owned());
    } else if estimate.cannot_estimate_reason == Some(CannotEstimateReason::HeaderUnreadable)
        && matches!(reference, GgufRef::Local(_))
    {
        // A path that could not be read is a path to check. "Serve it directly and
        // read the engine's refusal" is advice for a model nobody has looked at,
        // and `rocm serve <path>` on a file that is not there only repeats this.
        // A URL is read as a path, and the "could not open" it gets does not say
        // why; the form that is read is `owner/repo:QUANT`.
        let lower = readiness.model_ref.to_ascii_lowercase();
        let notes = if lower.starts_with("http://") || lower.starts_with("https://") {
            vec![
                "a URL is not read: name a Hugging Face file as `owner/repo:QUANT` (or \
                 `owner/repo:file.gguf`), or download it and pass the local path"
                    .to_owned(),
            ]
        } else {
            Vec::new()
        };
        readiness.fix = Some(Fix {
            summary: "check the path, and that the file is a GGUF model, then ask again".to_owned(),
            verify: ask_again(&readiness.model_ref),
            notes,
            ..Fix::default()
        });
    } else if estimate.cannot_estimate_reason == Some(CannotEstimateReason::HeaderUnreadable)
        && matches!(reference, GgufRef::Hub { .. })
    {
        // The header could not be fetched or read: a name that does not exist, a
        // gated or private repository, no network, a busy hub, or a file that is
        // not a GGUF. The estimate's detail says which. Serving it would start a
        // download of the whole file to learn the same thing, so the plan is to
        // act on that detail and ask again.
        readiness.fix = Some(Fix {
            summary: "act on the reason above (a misspelt name or quantization, no route to \
                      Hugging Face, a busy hub to wait for, or a file that is not a GGUF), \
                      then ask again"
                .to_owned(),
            verify: ask_again(&readiness.model_ref),
            notes: vec![
                "a gated or private repository cannot be read, because no token is sent: \
                 download the file yourself and ask about its local path"
                    .to_owned(),
            ],
            ..Fix::default()
        });
    } else if estimate.cannot_estimate_reason == Some(CannotEstimateReason::NotAModel) {
        readiness.fix = Some(projector_fix(reference, refused_models, request));
    } else if let Some(fix) = &mut readiness.fix {
        point_fix_at_unsized_gguf(fix, reference, &ask_again(&readiness.model_ref));
    }
    // A compared estimate is sized for a context `rocm serve` would not pass,
    // so the answer says so, once: with the options a suggestion names, or on
    // its own.
    if matches!(
        estimate.outcome,
        EstimateOutcome::Fits | EstimateOutcome::DoesNotFit
    ) {
        let note = if estimate.suggestions.is_empty() {
            SERVE_CONTEXT_NOTE
        } else {
            SERVE_OPTIONS_NOTE
        };
        readiness.warnings.push(note.to_owned());
    }
    readiness.estimate = Some(estimate);
}

/// The command that asks the same GGUF question again: the same reference,
/// quoted for this platform's shell when it needs to be, and the same
/// `--context-size` and `--kv-cache-type` when they were passed, so following
/// the plan does not quietly answer a different question.
fn rediagnose(model_ref: &str, request: &ContextRequest) -> String {
    rediagnose_for(model_ref, request, cfg!(windows))
}

/// [`rediagnose`] for a chosen platform, so both are tested on either.
///
/// A reference made only of characters no shell treats specially is printed
/// as it is, so a plain Windows path (`C:\models\m.gguf`) still works in both
/// `cmd.exe` and PowerShell; anything else is quoted for PowerShell on Windows
/// and for a POSIX shell elsewhere. A reference that starts with `-` is joined
/// to the flag (`--model=-x.gguf`), because the argument parser would read it
/// as another flag.
fn rediagnose_for(model_ref: &str, request: &ContextRequest, windows: bool) -> String {
    use std::fmt::Write as _;
    let quoted = shell_word(model_ref, windows);
    let mut command = if model_ref.starts_with('-') {
        format!("rocm diagnose --model={quoted}")
    } else {
        format!("rocm diagnose --model {quoted}")
    };
    if let Some(size) = request.context_size {
        let _ = write!(command, " --context-size {size}");
    }
    if let Some(kind) = request.kv_cache_type {
        let _ = write!(command, " --kv-cache-type {kind}");
    }
    command
}

/// `model_ref` as one word of a command for this platform's shell: as it is
/// when no shell would split or expand it, quoted for PowerShell on Windows and
/// for a POSIX shell elsewhere otherwise.
fn shell_word(model_ref: &str, windows: bool) -> String {
    // `~` is plain after the first character, where neither shell expands it;
    // Windows short names (`ADMINI~1`) carry one in the middle of a path.
    let plain = !model_ref.is_empty()
        && !model_ref.starts_with('~')
        && model_ref.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, '/' | ':' | '.' | '_' | '-' | '=' | '~')
                || (windows && c == '\\')
        });
    if plain {
        model_ref.to_owned()
    } else if windows {
        crate::quoting::powershell_quote(model_ref)
    } else {
        crate::quoting::posix_shell_quote(model_ref)
    }
}

/// The plan when no GPU is visible to ROCm: find out why, then ask again with
/// `verify`. Shared by a curated model that needs a GPU and a GGUF estimate.
fn no_gpu_fix(verify: String) -> Fix {
    Fix {
        summary: "make a GPU visible to ROCm, then ask again".to_owned(),
        commands: vec!["rocm diagnose".to_owned()],
        verify,
        notes: vec![
            "`rocm diagnose` matches this machine against the known reasons a GPU does not show \
             up"
            .to_owned(),
        ],
        ..Fix::default()
    }
}

/// The plan when this host's GPU memory could not be read: look at what
/// `amd-smi` reports, then ask again. Shared by a curated model and a GGUF
/// estimate, which hit the same gap.
fn read_gpu_memory_fix(model_ref: &str) -> Fix {
    Fix {
        summary: "let the CLI read this host's GPU memory, then ask again".to_owned(),
        commands: vec!["amd-smi metric --json".to_owned()],
        verify: format!("rocm diagnose --model {model_ref}"),
        notes: vec![
            "the reading comes from `amd-smi`; if it is missing or failing, the ROCm install is \
             what needs attention first"
                .to_owned(),
        ],
        ..Fix::default()
    }
}

/// The plan for a name refused as a multimodal projector. "Serve it and read
/// the engine's refusal" would point back at the file just refused, so the plan
/// is to ask about the model instead: by its exact file name when the
/// repository holds one model, and with no command when it holds several (the
/// estimate's detail lists them) or none.
fn projector_fix(reference: &GgufRef, models: Option<&[String]>, request: &ContextRequest) -> Fix {
    let summary = "name the model's own GGUF file or quantization, not its multimodal projector, \
                   then ask again"
        .to_owned();
    // The model's name comes from the hub's listing, not from the user, so it is
    // offered only when it reads back as that same hub file: a name the parser
    // would take for a local path, or that a shell would expand, is not printed
    // as a command.
    let verify = match (reference, models) {
        (GgufRef::Hub { owner, repo, .. }, Some([only])) => {
            let candidate = format!("{owner}/{repo}:{only}");
            let reads_back = parse_gguf_ref(&candidate)
                == Some(GgufRef::Hub {
                    owner: owner.clone(),
                    repo: repo.clone(),
                    variant: Some(only.clone()),
                });
            if reads_back {
                rediagnose(&candidate, request)
            } else {
                String::new()
            }
        }
        _ => String::new(),
    };
    Fix {
        summary,
        verify,
        ..Fix::default()
    }
}

/// Rewrite the "serve it and see" plan for a GGUF reference whose header was
/// reached for but not sized.
fn point_fix_at_unsized_gguf(fix: &mut Fix, reference: &GgufRef, verify: &str) {
    // The plan was written for a model that is not a GGUF reference, and its
    // note says so. This one is, and the estimate says why it could not be
    // sized, so the note says that instead of contradicting it.
    fix.notes = vec![
        "`rocm serve` accepts models outside this catalog; the estimate above could not size \
         this GGUF, so this cannot say in advance whether it will fit"
            .to_owned(),
    ];
    // `rocm serve` takes `owner/repo:QUANT`, whose label has no `/`; the
    // `owner/repo:dir/file.gguf` form selects an exact file for this estimate
    // only, and serve would not read it as a checkpoint. Naming it in a
    // `rocm serve` command would hand the user a line that fails.
    if matches!(reference, GgufRef::Hub { variant: Some(variant), .. } if variant.contains('/')) {
        verify.clone_into(&mut fix.verify);
        fix.notes.push(
            "`rocm serve` does not accept an exact file name after the colon; it takes \
             `owner/repo:QUANT`"
                .to_owned(),
        );
    }
    // `rocm serve` does take a local file, and serves it as it is; the path is
    // quoted for the shell. One that starts with `-` would be read as a flag,
    // so that one is asked about again instead.
    if let GgufRef::Local(path) = reference {
        let path = path.to_string_lossy();
        if path.starts_with('-') {
            verify.clone_into(&mut fix.verify);
        } else {
            fix.verify = format!("rocm serve {}", shell_word(&path, cfg!(windows)));
        }
    }
}

/// What an estimate says when a device-visibility variable is set on a host
/// whose GPUs differ in memory.
pub const MASK_UNVERIFIED_NOTE: &str = "a device-visibility variable (HIP_VISIBLE_DEVICES, \
     ROCR_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES) is set and this host's GPUs do not all \
     have the same memory; how the mask's numbering maps onto amd-smi's has not been verified \
     on such a host, so the memory compared with may be that of a different card (or the host \
     may be mistaken for an integrated GPU)";

/// Add [`MASK_UNVERIFIED_NOTE`] to a compared estimate when `mask_set` and the
/// GPUs' totals (`gpu_totals_mb`, one per amd-smi row) are not all equal.
///
/// Which row a mask selects is not something this binary has verified on a
/// multi-GPU machine, so the visible rows are not guessed at: if any two cards
/// differ, whichever one the mask picked might not be the one compared. An
/// estimate that compared nothing has no figure to doubt and is left alone.
pub fn note_unverified_mask(readiness: &mut ModelReadiness, mask_set: bool, gpu_totals_mb: &[u64]) {
    let differ = gpu_totals_mb.windows(2).any(|pair| pair[0] != pair[1]);
    if !(mask_set && differ) {
        return;
    }
    if let Some(estimate) = readiness.estimate.as_mut()
        && estimate.available_gib.is_some()
    {
        estimate.notes.push(MASK_UNVERIFIED_NOTE.to_owned());
    }
}

/// The word this verdict is reported under.
const fn verdict_label(verdict: ModelVerdict) -> &'static str {
    match verdict {
        ModelVerdict::Ready => "READY",
        ModelVerdict::Degraded => "DEGRADED",
        ModelVerdict::Blocked => "BLOCKED",
        ModelVerdict::Undetermined => "UNDETERMINED",
    }
}

/// The word the report opens with. An estimate that compared its figures with
/// the GPU is labelled as one, never as a verdict; every other answer -- an
/// estimate that could not compare included -- keeps the verdict's own word.
fn header_label(readiness: &ModelReadiness) -> String {
    match &readiness.estimate {
        Some(estimate) if estimate.outcome == EstimateOutcome::Fits => format!(
            "ESTIMATED: fits at {} (calibrated on one MI300X, not a tested minimum)",
            tokens(estimate.context_size)
        ),
        Some(estimate) if estimate.outcome == EstimateOutcome::DoesNotFit => format!(
            "ESTIMATED: does not fit at {} (calibrated on one MI300X, not a tested minimum)",
            tokens(estimate.context_size)
        ),
        _ => verdict_label(readiness.verdict).to_owned(),
    }
}

/// Render the human-facing model verdict.
///
/// Deliberately laid out like `diagnose::render_report_text`: the same `   - `
/// for evidence, `     $ ` for a runnable command and `   verify after fix: `
/// for the check afterwards. Those prefixes are how anything post-processing the
/// report tells prose from commands, and a second report shape would quietly
/// exempt this one from it.
#[must_use]
pub fn render_model_readiness_text(readiness: &ModelReadiness) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "rocm diagnose --model {}: {}",
        readiness.model_ref,
        header_label(readiness)
    );
    if let Some(canonical) = &readiness.canonical_model_id {
        let _ = writeln!(out, "   recipe: {canonical}");
    }
    for line in &readiness.evidence {
        let _ = writeln!(out, "   - {line}");
    }
    if let Some(estimate) = &readiness.estimate
        && !estimate.suggestions.is_empty()
    {
        let _ = writeln!(out, "   what would make it fit:");
        for suggestion in &estimate.suggestions {
            let _ = writeln!(out, "     - {}", suggestion.text);
        }
    }
    if !readiness.alternatives.is_empty() {
        let _ = writeln!(out, "   what would run here instead:");
        for alternative in &readiness.alternatives {
            let requirement = alternative.required_gpu_memory_gib.map_or_else(
                || "no GPU minimum".to_owned(),
                |value| format!("{} minimum", format_gib(value)),
            );
            let _ = writeln!(
                out,
                "     {} ({requirement}, {})",
                alternative.model_ref, alternative.engine
            );
        }
    }
    if let Some(fix) = &readiness.fix {
        let _ = writeln!(out, "   plan: {}", fix.summary);
        for command in &fix.commands {
            let _ = writeln!(out, "     $ {command}");
        }
        for note in &fix.notes {
            let _ = writeln!(out, "   note: {note}");
        }
        if !fix.verify.is_empty() {
            let _ = writeln!(out, "   verify after fix: {}", fix.verify);
        }
    }
    if let Some(route) = &readiness.route {
        let _ = writeln!(out, "   report it: {:>12}: {}", route.target, route.url);
    }
    // The recipe's own warnings come last and carry no verdict weight. Putting
    // them above the plan would read as reasons for it. The estimate's notes
    // are about the figures, not a recipe, and are labelled as such.
    if let Some(estimate) = &readiness.estimate {
        for note in &estimate.notes {
            let _ = writeln!(out, "   note: {note}");
        }
    }
    for warning in &readiness.warnings {
        let label = if readiness.estimate.is_some() {
            "note"
        } else {
            "recipe note"
        };
        let _ = writeln!(out, "   {label}: {warning}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_model_recipe_registry;

    /// A recipe with a minimum far above anything a test host offers, so the
    /// "would have been blocked" premise of the unreadable-catalog test cannot
    /// quietly stop holding.
    const LARGE_MODEL_REF: &str = "qwen3-32b-fp8";

    /// The smallest curated recipe: 2 GiB minimum, 4 GiB recommended RAM.
    const SMALL_MODEL_REF: &str = "qwen-smoke";

    /// Stands in for the host's engine decision. The invariant under test here is
    /// about verdicts, not about which engine was picked — that is I3, and it is
    /// asserted where the real decision lives.
    fn engine_choice(recipe: &ModelRecipeRecord) -> HostEngineChoice {
        HostEngineChoice {
            engine: recipe
                .preferred_engines
                .first()
                .cloned()
                .unwrap_or_else(|| "lemonade".to_owned()),
            source: "test host".to_owned(),
            unsupported_here: None,
        }
    }

    fn host_with(gpu_gib: f64, ram_gib: f64) -> HostFacts {
        HostFacts {
            accelerator_memory: AcceleratorMemory::Dedicated(gpu_gib),
            system_ram_gib: Some(ram_gib),
        }
    }

    /// I1 — a catalog the CLI could not read never produces a compatibility
    /// verdict.
    ///
    /// The host here is one that *would* be told the model does not fit: the
    /// paired assertion below proves it, and without that pairing the
    /// undetermined assertion would pass against a host incapable of producing
    /// the wrong answer in the first place. The failure this guards against is
    /// not "no verdict at all", it is an unreadable source being reported with
    /// the same word a real incompatibility gets.
    #[test]
    fn an_unreadable_catalog_is_never_reported_as_an_incompatible_model() {
        let registry = builtin_model_recipe_registry();
        let starved = host_with(0.0, 64.0);

        let readable = assess_model_readiness(
            LARGE_MODEL_REF,
            &ModelCatalogSource::Available(&registry),
            &starved,
            &engine_choice,
        );
        assert_eq!(
            readable.verdict,
            ModelVerdict::Blocked,
            "premise failed: this host must be one that a readable catalog reports as blocked, \
             otherwise the undetermined assertion below proves nothing. Got {:?} with evidence {:?}",
            readable.verdict,
            readable.evidence
        );

        let detail = "index /nonexistent/model-index.json could not be read";
        let unreachable = assess_model_readiness(
            LARGE_MODEL_REF,
            &ModelCatalogSource::Unreachable {
                detail: detail.to_owned(),
            },
            &starved,
            &engine_choice,
        );
        assert_eq!(
            unreachable.verdict,
            ModelVerdict::Undetermined,
            "an unreadable catalog was reported as {:?}, which is a verdict about the model; \
             the CLI does not know anything about the model here. Evidence: {:?}",
            unreachable.verdict,
            unreachable.evidence
        );
        assert_eq!(
            unreachable.undetermined_reason,
            Some(UndeterminedReason::CatalogUnreachable),
            "the reason given was {:?}, not the unreachable source",
            unreachable.undetermined_reason
        );
        assert!(
            unreachable
                .evidence
                .iter()
                .any(|line| line.contains(detail)),
            "the source failure `{detail}` appears nowhere in the evidence: {:?}",
            unreachable.evidence
        );
        assert_eq!(
            unreachable.required_gpu_memory_gib, None,
            "a requirement of {:?} was reported for a model whose recipe was never read",
            unreachable.required_gpu_memory_gib
        );
    }

    /// I2 — an alternative offered is never one that is itself not runnable here.
    ///
    /// Every alternative is fed back through the same assessment with the same
    /// host, rather than through the filter helper with literal arguments: the
    /// failure mode is the filter being handed the wrong host, not the filter
    /// being wrong.
    #[test]
    fn every_alternative_offered_would_itself_run_on_this_host() {
        let registry = builtin_model_recipe_registry();
        // Non-vacuity. The loop below asserts something about each alternative
        // offered, so an assessment that offers none passes it while proving
        // nothing -- which is what the first run of this test actually did
        // against the stub. This host is one where the catalog's largest recipe
        // cannot run and smaller ones can, so a correct assessment has to offer
        // something here.
        let crowded = host_with(24.0, 512.0);
        let largest = assess_model_readiness(
            "glm5",
            &ModelCatalogSource::Available(&registry),
            &crowded,
            &engine_choice,
        );
        assert!(
            !largest.alternatives.is_empty(),
            "a 905 GiB recipe on a 24 GiB host offered nothing that would work instead, so the \
             per-alternative assertions below check nothing. Verdict was {:?}: {:?}",
            largest.verdict,
            largest.evidence
        );

        for gpu_gib in [4.0_f64, 8.0, 24.0, 48.0] {
            let host = host_with(gpu_gib, 512.0);
            for recipe in &registry.recipes {
                let report = assess_model_readiness(
                    &recipe.canonical_model_id,
                    &ModelCatalogSource::Available(&registry),
                    &host,
                    &engine_choice,
                );
                for alternative in &report.alternatives {
                    let offered = assess_model_readiness(
                        &alternative.model_ref,
                        &ModelCatalogSource::Available(&registry),
                        &host,
                        &engine_choice,
                    );
                    assert!(
                        !matches!(
                            offered.verdict,
                            ModelVerdict::Blocked | ModelVerdict::Undetermined
                        ),
                        "on a {gpu_gib} GiB host, `{}` was offered as what would work instead of \
                         `{}`, but asking about `{}` reports {:?}: {:?}",
                        alternative.model_ref,
                        recipe.canonical_model_id,
                        alternative.model_ref,
                        offered.verdict,
                        offered.evidence
                    );
                    // Anchored against the host fact rather than against another
                    // assessment. The two assertions either side of this one ask
                    // `assess_model_readiness` whether what it offered is
                    // runnable, so they share a definition of "runnable" with the
                    // code under test: an implementation that compared the wrong
                    // memory figure everywhere would be wrong consistently, and
                    // they would both stay green. This one names GPU memory
                    // itself, so it fails when the comparison drifts onto another
                    // field. The hosts above pair a small GPU with 512 GiB of
                    // system RAM precisely so the two figures cannot be confused
                    // for one another.
                    if let Some(needs) = alternative.required_gpu_memory_gib {
                        assert!(
                            needs <= gpu_gib,
                            "on a {gpu_gib} GiB GPU, `{}` was offered instead of `{}`, but it \
                             declares a need for {needs} GiB of GPU memory",
                            alternative.model_ref,
                            recipe.canonical_model_id,
                        );
                    }
                    // The host above has ample system RAM, so nothing offered on
                    // it has an excuse to be merely degraded. Asserted separately
                    // from the invariant because the invariant has to keep
                    // holding on a RAM-starved host, where degraded is honest.
                    assert_eq!(
                        offered.verdict,
                        ModelVerdict::Ready,
                        "on a {gpu_gib} GiB host with ample system RAM, `{}` was offered instead \
                         of `{}` but is only {:?}: {:?}",
                        alternative.model_ref,
                        recipe.canonical_model_id,
                        offered.verdict,
                        offered.evidence
                    );
                }
            }
        }
    }

    /// Memory that could not be measured is not memory the host does not have,
    /// and neither is the same as there being no GPU. Three inputs a two-state
    /// `Option<f64>` would have collapsed into two answers.
    #[test]
    fn unmeasured_memory_absent_gpu_and_ample_memory_are_three_different_answers() {
        let registry = builtin_model_recipe_registry();
        let catalog = ModelCatalogSource::Available(&registry);
        let ask = |memory| {
            assess_model_readiness(
                SMALL_MODEL_REF,
                &catalog,
                &HostFacts {
                    accelerator_memory: memory,
                    system_ram_gib: Some(64.0),
                },
                &engine_choice,
            )
        };

        let unknown = ask(AcceleratorMemory::Unknown);
        assert_eq!(unknown.verdict, ModelVerdict::Undetermined);
        assert_eq!(
            unknown.undetermined_reason,
            Some(UndeterminedReason::AcceleratorMemoryUnknown),
            "a GPU whose memory could not be read must not be reported as one that is too \
             small: {:?}",
            unknown.evidence
        );

        let absent = ask(AcceleratorMemory::None);
        assert_eq!(
            absent.verdict,
            ModelVerdict::Blocked,
            "a gpu_required recipe on a host with no GPU does not run, and there is no CPU \
             fallback: {:?}",
            absent.evidence
        );

        assert_eq!(
            ask(AcceleratorMemory::Dedicated(64.0)).verdict,
            ModelVerdict::Ready
        );
    }

    /// An APU's `amd-smi` telemetry names the wrong pool (the BIOS carve-out,
    /// not what the engine allocates from), and on *this* host the right pool
    /// could not be read either -- an older kernel with no `mem_info_gtt_total`,
    /// a container with no DRM device passed through. This is `Undetermined`,
    /// not `Blocked` -- the model might well fit -- and it must not be conflated
    /// with the no-telemetry-at-all case, whose remediation
    /// (`amd-smi metric --json`) would send an APU user in a circle forever.
    /// The host whose pool IS readable is
    /// [`an_apu_with_a_readable_pool_is_degraded_never_ready`].
    #[test]
    fn an_apu_with_no_readable_pool_is_undetermined_with_no_dead_end_command() {
        let registry = builtin_model_recipe_registry();
        let catalog = ModelCatalogSource::Available(&registry);
        let strix_halo = HostFacts {
            accelerator_memory: AcceleratorMemory::UnifiedMemoryUnreadable,
            system_ram_gib: Some(128.0),
        };
        let readiness = assess_model_readiness("qwen3.6", &catalog, &strix_halo, &engine_choice);
        assert_eq!(
            readiness.verdict,
            ModelVerdict::Undetermined,
            "no readable figure means no verdict, not a confident one: {:?}",
            readiness.evidence
        );
        assert_eq!(
            readiness.undetermined_reason,
            Some(UndeterminedReason::UnifiedMemoryUnreadable),
            "the reason has to name the APU-specific gap, not the generic unmeasured one: {:?}",
            readiness.undetermined_reason
        );
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("no dedicated VRAM")),
            "the report has to say why, in terms an APU user can act on: {:?}",
            readiness.evidence
        );
        assert!(
            readiness.fix.is_none(),
            "there is no command that makes this readable today; offering one would be a dead \
             end: {:?}",
            readiness.fix
        );
    }

    /// An APU whose pool WAS read is judged on it -- and on that figure alone,
    /// never on the carve-out beside it -- but cannot be called `Ready`.
    ///
    /// The three answers this pins apart are the whole point of the variant: a
    /// model under the ceiling is `Degraded` (it will run, but the pool is
    /// shared with the operating system and nothing measured what is free), one
    /// over it is `Blocked` (no amount of freeing raises an aperture the driver
    /// will not hand out), and the same arithmetic on a dedicated card is
    /// `Ready`. Without the last of the three, `Degraded` could be coming from
    /// the system-RAM softening instead and the test would prove nothing.
    #[test]
    fn an_apu_with_a_readable_pool_is_degraded_never_ready() {
        let registry = builtin_model_recipe_registry();
        let catalog = ModelCatalogSource::Available(&registry);
        let ask = |accelerator_memory| {
            assess_model_readiness(
                SMALL_MODEL_REF,
                &catalog,
                &HostFacts {
                    accelerator_memory,
                    // Above every recipe's recommendation, so the system-RAM
                    // softening cannot be what produces `Degraded` below.
                    system_ram_gib: Some(4096.0),
                },
                &engine_choice,
            )
        };

        let ample = ask(AcceleratorMemory::Unified {
            gtt_total_gib: 60.0,
            // A carve-out far below the recipe's minimum: were THIS the figure
            // compared against, the verdict would be `Blocked`.
            vram_carveout_gib: Some(0.5),
        });
        assert_eq!(
            ample.verdict,
            ModelVerdict::Degraded,
            "a pool shared with the operating system is a ceiling, not an allocation, so it \
             cannot promise READY: {:?}",
            ample.evidence
        );
        assert_eq!(
            ample.available_gpu_memory_gib,
            Some(60.0),
            "the verdict is reached against the pool, not the carve-out beside it"
        );
        assert!(
            ample
                .evidence
                .iter()
                .any(|line| line.contains("shares with the operating system")),
            "`degraded` without the reason reads as a property of the model: {:?}",
            ample.evidence
        );

        // The premise: the same arithmetic on a dedicated card IS ready, so the
        // assertion above is about the shared pool and not about this recipe.
        assert_eq!(
            ask(AcceleratorMemory::Dedicated(60.0)).verdict,
            ModelVerdict::Ready
        );

        // And a ceiling genuinely below the minimum blocks, with the reason:
        // this is not occupancy a reboot would clear.
        let starved = ask(AcceleratorMemory::Unified {
            gtt_total_gib: 0.25,
            vram_carveout_gib: Some(0.5),
        });
        assert_eq!(
            starved.verdict,
            ModelVerdict::Blocked,
            "{:?}",
            starved.evidence
        );
        assert!(
            starved
                .evidence
                .iter()
                .any(|line| line.contains("freeing memory would not raise it")),
            "a refusal on an aperture has to say freeing memory will not help, or the user \
             closes applications and asks again: {:?}",
            starved.evidence
        );
    }

    /// The sibling gap -- no telemetry at all -- keeps its own remediation.
    /// Pinned alongside the APU case above so the two branches of the same
    /// match arm cannot collapse back into each other unnoticed.
    #[test]
    fn no_telemetry_at_all_still_points_at_amd_smi() {
        let registry = builtin_model_recipe_registry();
        let readiness = assess_model_readiness(
            SMALL_MODEL_REF,
            &ModelCatalogSource::Available(&registry),
            &HostFacts {
                accelerator_memory: AcceleratorMemory::Unknown,
                system_ram_gib: Some(64.0),
            },
            &engine_choice,
        );
        assert_eq!(readiness.verdict, ModelVerdict::Undetermined);
        assert_eq!(
            readiness.undetermined_reason,
            Some(UndeterminedReason::AcceleratorMemoryUnknown)
        );
        let fix = readiness
            .fix
            .as_ref()
            .expect("a genuine telemetry gap has a real next step: run amd-smi");
        assert!(
            fix.commands
                .iter()
                .any(|command| command.contains("amd-smi")),
            "the remediation for a true telemetry gap has to name the tool that would fill it: \
             {:?}",
            fix.commands
        );
    }

    /// Below the recipe's recommended system RAM it still runs, and saying it
    /// does not would be wrong in the direction that costs the user the model.
    #[test]
    fn a_host_below_the_recommended_system_ram_is_degraded_not_blocked() {
        let registry = builtin_model_recipe_registry();
        let readiness = assess_model_readiness(
            SMALL_MODEL_REF,
            &ModelCatalogSource::Available(&registry),
            &HostFacts {
                accelerator_memory: AcceleratorMemory::Dedicated(64.0),
                system_ram_gib: Some(2.0),
            },
            &engine_choice,
        );
        assert_eq!(
            readiness.verdict,
            ModelVerdict::Degraded,
            "GPU memory is ample and only the RAM recommendation is missed: {:?}",
            readiness.evidence
        );
    }

    /// An engine the platform rules out pre-empts the memory comparison. A model
    /// that fits perfectly still will not start on an engine with no adapter
    /// here, and reporting the memory verdict instead would be true and useless.
    #[test]
    fn an_engine_the_platform_rules_out_blocks_a_model_that_would_otherwise_fit() {
        let registry = builtin_model_recipe_registry();
        let ruled_out = |recipe: &ModelRecipeRecord| HostEngineChoice {
            unsupported_here: Some("vllm has no adapter on native Windows".to_owned()),
            ..engine_choice(recipe)
        };
        let readiness = assess_model_readiness(
            SMALL_MODEL_REF,
            &ModelCatalogSource::Available(&registry),
            &HostFacts {
                accelerator_memory: AcceleratorMemory::Dedicated(512.0),
                system_ram_gib: Some(512.0),
            },
            &ruled_out,
        );
        assert_eq!(
            readiness.verdict,
            ModelVerdict::Blocked,
            "512 GiB is ample for a 2 GiB recipe, so only the engine gate can be blocking it: \
             {:?}",
            readiness.evidence
        );
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("no adapter on native Windows")),
            "the refusal has to name the engine gate, not leave it reading as a memory problem: \
             {:?}",
            readiness.evidence
        );
    }

    /// A model the catalog does not carry is a gap in the catalog, not a fact
    /// about the machine. It is also the one case where the CLI genuinely has
    /// nothing to say, so it routes onward the way `rocm diagnose` does.
    #[test]
    fn a_model_outside_the_catalog_is_undetermined_and_routed_onward() {
        let registry = builtin_model_recipe_registry();
        let readiness = assess_model_readiness(
            "some-lab/never-curated-7b",
            &ModelCatalogSource::Available(&registry),
            &host_with(64.0, 64.0),
            &engine_choice,
        );
        assert_eq!(readiness.verdict, ModelVerdict::Undetermined);
        assert_eq!(
            readiness.undetermined_reason,
            Some(UndeterminedReason::ModelNotCurated)
        );
        assert!(
            readiness.canonical_model_id.is_none() && readiness.required_gpu_memory_gib.is_none(),
            "no recipe was read, so nothing may be reported about the model itself: {readiness:?}"
        );
        let route = readiness
            .route
            .expect("an unanswerable question needs a route onward");
        assert!(
            route.url.starts_with("http"),
            "the route has to be somewhere the user can actually go: {route:?}"
        );
    }

    /// The renderer is driven for real rather than having its shape assumed.
    ///
    /// The prefixes are a coupling `diagnose`'s renderer already has and anything
    /// post-processing a report depends on: `     $ ` marks a runnable command
    /// and `   - ` marks prose. A second report shape that looked similar but
    /// differed would silently exempt this half of the output from it.
    #[test]
    fn the_rendered_answer_marks_commands_the_way_the_diagnosis_does() {
        let registry = builtin_model_recipe_registry();
        let readiness = assess_model_readiness(
            LARGE_MODEL_REF,
            &ModelCatalogSource::Available(&registry),
            &host_with(4.0, 64.0),
            &engine_choice,
        );
        assert_eq!(readiness.verdict, ModelVerdict::Blocked);
        let rendered = render_model_readiness_text(&readiness);
        assert!(
            rendered.starts_with(&format!("rocm diagnose --model {LARGE_MODEL_REF}: BLOCKED")),
            "the verdict has to lead: {rendered}"
        );
        let commands = rendered
            .lines()
            .filter(|line| line.starts_with("     $ "))
            .count();
        assert_eq!(
            commands,
            readiness.fix.as_ref().map_or(0, |fix| fix.commands.len()),
            "every command in the plan has to reach the report behind the same prefix a \
             diagnosis uses:\n{rendered}"
        );
        for alternative in &readiness.alternatives {
            assert!(
                rendered.contains(&alternative.model_ref),
                "`{}` was chosen as an alternative but never rendered:\n{rendered}",
                alternative.model_ref
            );
        }
    }

    /// The defect a reviewer caught: a card's real VRAM total sits just under
    /// its nameplate size (an 8 GiB card reports 8176 MiB, i.e. 7.9844 GiB), so
    /// rounding to nearest prints "8.0 GiB" against an "8 GiB" recipe minimum --
    /// evidence that reads as "the host has exactly what it needs" on a verdict
    /// that says BLOCKED. Flooring instead keeps the displayed figure honestly
    /// below the minimum it failed to meet. Were this still rounding to
    /// nearest, the assertion below would see "8.0 GiB", not "7.9 GiB".
    #[test]
    fn format_gib_floors_a_measured_figure_rather_than_rounding_up_to_the_minimum() {
        let real_eight_gib_card_mib = 8176.0 / 1024.0;
        assert_eq!(format_gib(real_eight_gib_card_mib), "7.9 GiB");
    }

    /// A whole-number figure (every `required`/`recommended` minimum, and any
    /// measured figure that happens to land exactly on an integer) is
    /// unaffected by the floor: it still renders with no decimal at all.
    #[test]
    fn format_gib_renders_a_whole_number_with_no_decimal() {
        assert_eq!(format_gib(8.0), "8 GiB");
    }

    // ----- GGUF estimate (models outside the catalog) -----------------------

    use crate::gguf::test_builder::llama_like;
    use crate::gguf_fit::SuggestionKind;
    use crate::gguf_source::{GgufRef, OpenedGguf};
    use std::io::Cursor;
    use std::sync::Mutex;

    /// Serves one in-memory GGUF for every reference and records what it was
    /// asked to open, so a test can show the source was (or was not) reached.
    struct FakeSource {
        file: Option<Vec<u8>>,
        /// The size the source reports for the file: the header plus the
        /// weights it pretends to carry, so no test allocates a real model.
        size: u64,
        opened: Mutex<Vec<GgufRef>>,
        /// When set, opening refuses the name as a multimodal projector, as
        /// file selection does, listing these model files.
        refusal: Option<Vec<String>>,
    }

    impl FakeSource {
        fn with(file: Vec<u8>) -> Self {
            Self {
                size: file.len() as u64,
                file: Some(file),
                opened: Mutex::new(Vec::new()),
                refusal: None,
            }
        }

        /// A llama-shaped file whose weights are `weights_mib` MiB: 8192
        /// tokens of f16 cache is exactly 1 GiB for it.
        fn weights(weights_mib: u64) -> Self {
            let header = llama_like(10).file(0);
            Self {
                size: header.len() as u64 + weights_mib * 1024 * 1024,
                file: Some(header),
                opened: Mutex::new(Vec::new()),
                refusal: None,
            }
        }

        fn failing() -> Self {
            Self {
                file: None,
                size: 0,
                opened: Mutex::new(Vec::new()),
                refusal: None,
            }
        }

        /// A hub that refuses the name as a projector, with `models` beside it.
        fn refusing_a_projector(models: &[&str]) -> Self {
            Self {
                refusal: Some(models.iter().map(|&model| model.to_owned()).collect()),
                ..Self::failing()
            }
        }

        fn calls(&self) -> usize {
            self.opened.lock().unwrap().len()
        }
    }

    impl GgufSource for FakeSource {
        fn open(&self, reference: &GgufRef) -> anyhow::Result<OpenedGguf> {
            self.opened.lock().unwrap().push(reference.clone());
            if let (Some(models), GgufRef::Hub { variant, .. }) = (&self.refusal, reference) {
                return Err(anyhow::Error::new(ProjectorNamed {
                    variant: variant.clone().unwrap_or_default(),
                    projectors: vec![variant.clone().unwrap_or_default()],
                    models: models.clone(),
                }));
            }
            let Some(file) = &self.file else {
                anyhow::bail!("the hub could not be reached");
            };
            Ok(OpenedGguf {
                size: self.size,
                reader: Box::new(Cursor::new(file.clone())),
                description: "fake.gguf".to_owned(),
            })
        }
    }

    fn not_curated(model_ref: &str) -> ModelReadiness {
        let registry = builtin_model_recipe_registry();
        assess_model_readiness(
            model_ref,
            &ModelCatalogSource::Available(&registry),
            &host_with(24.0, 64.0),
            &engine_choice,
        )
    }

    fn asked(context: u64) -> ContextRequest {
        ContextRequest {
            context_size: Some(context),
            kv_cache_type: None,
        }
    }

    /// The decide step on its own: a GGUF outside the catalog is planned for a
    /// read with the defaults filled in, and an answer that is not one is left as
    /// it is, with an explanation only when a context flag was passed.
    #[test]
    fn the_plan_reads_only_a_gguf_outside_the_catalog_and_fills_in_the_defaults() {
        let gguf = not_curated("some-lab/never-curated-7b:Q4_K_M");
        assert_eq!(
            plan_context_estimate(&gguf, &ContextRequest::default()),
            ContextPlan::Estimate {
                reference: GgufRef::Hub {
                    owner: "some-lab".to_owned(),
                    repo: "never-curated-7b".to_owned(),
                    variant: Some("Q4_K_M".to_owned()),
                },
                fit_request: FitRequest {
                    // 4096, written out: `--help` and the README state it
                    // independently of the constant.
                    context_size: 4096,
                    context_size_defaulted: true,
                    kv_cache_type: KvCacheType::F16,
                },
            }
        );
        let not_gguf = not_curated("some-lab/never-curated-7b");
        assert_eq!(
            plan_context_estimate(&not_gguf, &ContextRequest::default()),
            ContextPlan::LeaveAsIs { note: None }
        );
        assert!(matches!(
            plan_context_estimate(&not_gguf, &asked(8192)),
            ContextPlan::LeaveAsIs { note: Some(note) } if note.contains("add `:QUANT`")
        ));
        // Flags that were passed are carried as given, not replaced by defaults.
        let both = ContextRequest {
            context_size: Some(32768),
            kv_cache_type: Some(KvCacheType::Q8_0),
        };
        assert!(matches!(
            plan_context_estimate(&gguf, &both),
            ContextPlan::Estimate { fit_request, .. } if fit_request == FitRequest {
                context_size: 32768,
                context_size_defaulted: false,
                kv_cache_type: KvCacheType::Q8_0,
            }
        ));
        // A curated model is never read, and is told why only when it asked.
        let curated = not_curated(SMALL_MODEL_REF);
        assert_eq!(
            plan_context_estimate(&curated, &ContextRequest::default()),
            ContextPlan::LeaveAsIs { note: None }
        );
        assert!(matches!(
            plan_context_estimate(&curated, &both),
            ContextPlan::LeaveAsIs { note: Some(note) } if note.contains("curated recipe")
        ));
    }

    /// A name refused as a projector is `not_a_model`, and the plan points at
    /// the model rather than at serving the projector it refused: by its exact
    /// name when there is one model, and with no command to run otherwise.
    #[test]
    fn a_refused_projector_is_not_offered_back_as_a_serve_command() {
        let refused = |models: &[&str]| {
            let mut readiness = not_curated("some-lab/vision-7b:mmproj-vision-7b-f16.gguf");
            attach_context_estimate(
                &mut readiness,
                &ContextRequest::default(),
                &host_with(24.0, 64.0),
                &FakeSource::refusing_a_projector(models),
            );
            readiness
        };
        let one = refused(&["vision-7b-Q4_K_M.gguf"]);
        let estimate = one.estimate.as_ref().expect("an estimate");
        assert_eq!(
            estimate.cannot_estimate_reason,
            Some(CannotEstimateReason::NotAModel)
        );
        assert!(
            estimate.detail.as_deref().is_some_and(|detail| detail
                .starts_with("`mmproj-vision-7b-f16.gguf` is a multimodal projector")),
            "{:?}",
            estimate.detail
        );
        let fix = one.fix.as_ref().expect("a plan for the user");
        assert!(
            fix.summary.contains("not its multimodal projector"),
            "{fix:?}"
        );
        assert_eq!(
            fix.verify,
            "rocm diagnose --model some-lab/vision-7b:vision-7b-Q4_K_M.gguf"
        );
        // Asked with a context size, the command for the model asks with it too.
        let mut with_flags = not_curated("some-lab/vision-7b:mmproj-vision-7b-f16.gguf");
        attach_context_estimate(
            &mut with_flags,
            &ContextRequest {
                context_size: Some(16_384),
                kv_cache_type: Some(KvCacheType::Q4_0),
            },
            &host_with(24.0, 64.0),
            &FakeSource::refusing_a_projector(&["vision-7b-Q4_K_M.gguf"]),
        );
        assert_eq!(
            with_flags.fix.as_ref().expect("a plan").verify,
            "rocm diagnose --model some-lab/vision-7b:vision-7b-Q4_K_M.gguf --context-size \
             16384 --kv-cache-type q4_0"
        );
        let text = render_model_readiness_text(&one);
        assert!(!text.contains("header could not be read"), "{text}");
        // No header was read, so the report does not say the figures came from one.
        assert!(!text.contains("estimated from the GGUF header"), "{text}");
        assert!(!text.contains("rocm serve"), "{text}");
        // A listed name the parser would not read back as that hub file (a shell
        // expansion, a space) is not printed as a command.
        for unsafe_name in ["x$(id).gguf", "my model.gguf"] {
            let odd = refused(&[unsafe_name]);
            assert_eq!(
                odd.fix.as_ref().map(|fix| fix.verify.as_str()),
                Some(""),
                "{unsafe_name}"
            );
        }
        // Several models: no single command to name, so none is given.
        let several = refused(&["vision-7b-Q4_K_M.gguf", "vision-7b-Q8_0.gguf"]);
        assert_eq!(
            several.fix.as_ref().map(|fix| fix.verify.as_str()),
            Some("")
        );
        // A projector-like name that failed for another reason (an unreachable
        // hub) is not blamed on the name: it keeps the unreadable answer.
        let mut unreachable = not_curated("some-lab/vision-7b:mmproj-vision-7b-f16.gguf");
        attach_context_estimate(
            &mut unreachable,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        assert_eq!(
            unreachable
                .estimate
                .and_then(|estimate| estimate.cannot_estimate_reason),
            Some(CannotEstimateReason::HeaderUnreadable)
        );
        assert!(
            unreachable
                .fix
                .is_some_and(|fix| !fix.summary.contains("multimodal projector")),
        );
    }

    /// A sized estimate that could not be compared keeps a next step only where
    /// one exists, the one a curated model gets: memory that could not be read
    /// is looked into with `amd-smi`, no visible GPU with `rocm diagnose`; an
    /// APU has no command that would change the answer.
    #[test]
    fn a_sized_estimate_that_could_not_compare_keeps_a_plan_only_when_one_exists() {
        for (memory, expected) in [
            (AcceleratorMemory::Unknown, Some("amd-smi metric --json")),
            (AcceleratorMemory::None, Some("rocm diagnose")),
            (AcceleratorMemory::UnifiedMemoryUnreadable, None),
        ] {
            let host = HostFacts {
                accelerator_memory: memory,
                system_ram_gib: Some(64.0),
            };
            let registry = builtin_model_recipe_registry();
            let mut readiness = assess_model_readiness(
                "some-lab/never-curated-7b:Q4_K_M",
                &ModelCatalogSource::Available(&registry),
                &host,
                &engine_choice,
            );
            attach_context_estimate(
                &mut readiness,
                &ContextRequest {
                    context_size: Some(32_768),
                    kv_cache_type: None,
                },
                &host,
                &FakeSource::weights(64),
            );
            let estimate = readiness.estimate.as_ref().expect("an estimate");
            assert!(
                estimate.total_gib.is_some(),
                "{memory:?}: the header was sized"
            );
            let commands = readiness
                .fix
                .as_ref()
                .map(|fix| fix.commands.clone())
                .unwrap_or_default();
            match expected {
                Some(command) => {
                    assert_eq!(commands, vec![command.to_owned()], "{memory:?}");
                    assert_eq!(
                        readiness.fix.as_ref().map(|fix| fix.verify.as_str()),
                        Some(
                            "rocm diagnose --model some-lab/never-curated-7b:Q4_K_M \
                             --context-size 32768"
                        ),
                        "{memory:?}"
                    );
                }
                None => assert!(readiness.fix.is_none(), "{memory:?}: {:?}", readiness.fix),
            }
        }
    }

    #[test]
    fn a_gguf_outside_the_catalog_gets_an_estimate_and_stays_undetermined() {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        // The catalog-miss answer offers curated models that would run here; an
        // estimate that says the model fits has no use for them.
        assert!(
            !readiness.alternatives.is_empty(),
            "the catalog-miss answer carries curated alternatives to clear"
        );
        let source = FakeSource::weights(2048);
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host_with(24.0, 64.0),
            &source,
        );
        let estimate = readiness.estimate.as_ref().expect("no estimate attached");
        assert_eq!(estimate.outcome, EstimateOutcome::Fits);
        assert!(
            readiness.alternatives.is_empty(),
            "ESTIMATED: fits offered alternatives: {:?}",
            readiness.alternatives
        );
        assert!(!render_model_readiness_text(&readiness).contains("what would run here instead"),);
        assert_eq!(estimate.context_size, 8192);
        // The closed-set verdict is not widened: an estimate is not a recipe.
        assert_eq!(readiness.verdict, ModelVerdict::Undetermined);
        assert_eq!(
            readiness.undetermined_reason,
            Some(UndeterminedReason::ModelNotCurated)
        );
        assert!(
            readiness.canonical_model_id.is_none() && readiness.required_gpu_memory_gib.is_none()
        );
        assert!(readiness.fix.is_none() && readiness.route.is_none());
        assert_eq!(source.calls(), 1);
        let text = render_model_readiness_text(&readiness);
        assert!(
            text.starts_with("rocm diagnose --model some-lab/never-curated-7b:Q4_K_M: ESTIMATED: fits at 8192 tokens"),
            "{text}"
        );
        assert!(
            !text.contains("READY") && !text.contains("BLOCKED"),
            "{text}"
        );
        // Labelled `note:`, as a figure's caveat, not `recipe note:`: there is no
        // recipe behind an estimate.
        assert!(
            text.lines()
                .any(|line| line.starts_with("   note: these are estimates")),
            "{text}"
        );
        assert!(!text.contains("recipe note"), "{text}");
    }

    #[test]
    fn a_gguf_that_does_not_fit_says_what_would_make_it_fit() {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        // 2 GiB of weights, 1 GiB of cache at 8192, ~1.4 GiB for the compute
        // buffer and runtime: 4.4 GiB against 4, with room for ~5000 tokens.
        let host = host_with(4.0, 64.0);
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host,
            &FakeSource::weights(2048),
        );
        let estimate = readiness.estimate.as_ref().unwrap();
        assert_eq!(estimate.outcome, EstimateOutcome::DoesNotFit);
        // A model that does not fit keeps the curated models that would run here:
        // they answer the question the user is left with.
        assert!(
            !readiness.alternatives.is_empty(),
            "ESTIMATED: does not fit dropped the alternatives"
        );
        let kinds: Vec<_> = estimate.suggestions.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SuggestionKind::SmallerContext), "{kinds:?}");
        assert!(kinds.contains(&SuggestionKind::KvCacheType), "{kinds:?}");
        let text = render_model_readiness_text(&readiness);
        assert!(
            text.contains("ESTIMATED: does not fit at 8192 tokens"),
            "{text}"
        );
        assert!(text.contains("what would make it fit:"), "{text}");
        // Nothing printed as a command: `rocm serve` takes none of these options,
        // and a runnable-looking line for them would be false.
        assert!(!text.contains("     $ "), "{text}");
        assert!(!text.contains("rocm serve some-lab"), "{text}");
        assert!(!text.contains("rocm serve --"), "{text}");
        assert!(
            text.contains("`rocm serve` does not take a context size"),
            "{text}"
        );
    }

    #[test]
    fn a_bare_repository_is_answered_as_before_and_read_nowhere() {
        // `owner/repo` with no quantization names a repository, not a file. With
        // no context flag the answer is exactly the plain "not in the catalog"
        // one, reached without touching the network or the disk.
        let plain = not_curated("some-lab/never-curated-7b");
        let mut readiness = plain.clone();
        let source = FakeSource::weights(64);
        attach_context_estimate(
            &mut readiness,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &source,
        );
        assert_eq!(readiness, plain);
        assert_eq!(source.calls(), 0);
        assert!(readiness.estimate.is_none() && readiness.route.is_some());
        // With a flag, the user is told what would make it answerable -- still
        // without reading anything.
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host_with(24.0, 64.0),
            &source,
        );
        assert_eq!(source.calls(), 0);
        assert!(readiness.estimate.is_none());
        let text = render_model_readiness_text(&readiness);
        assert!(
            text.contains("names a repository, not a model file: add `:QUANT`"),
            "{text}"
        );
    }

    /// The command a GGUF plan offers asks the same question again: the flags
    /// that were passed come with it, and a path a shell would split is quoted.
    #[test]
    fn the_command_to_ask_again_keeps_the_flags_and_quotes_the_path() {
        let mut readiness = not_curated("my models/nosuch.gguf");
        attach_context_estimate(
            &mut readiness,
            &ContextRequest {
                context_size: Some(32_768),
                kv_cache_type: Some(KvCacheType::Q8_0),
            },
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        assert_eq!(
            readiness.fix.as_ref().expect("a plan").verify,
            "rocm diagnose --model 'my models/nosuch.gguf' --context-size 32768 \
             --kv-cache-type q8_0"
        );
        let plain = ContextRequest::default();
        assert_eq!(
            rediagnose("o/r:Q4_K_M", &plain),
            "rocm diagnose --model o/r:Q4_K_M"
        );
        assert_eq!(
            rediagnose_for("it's here.gguf", &plain, false),
            r"rocm diagnose --model 'it'\''s here.gguf'"
        );
        // Windows: a plain path is left as it is, so cmd.exe reads it too;
        // one with a space is quoted for PowerShell, which doubles a `'`.
        assert_eq!(
            rediagnose_for(r"C:\models\m.gguf", &plain, true),
            r"rocm diagnose --model C:\models\m.gguf"
        );
        assert_eq!(
            rediagnose_for(r"C:\my models\it's.gguf", &plain, true),
            r"rocm diagnose --model 'C:\my models\it''s.gguf'"
        );
        // A Windows short name keeps a path plain on either shell; a leading
        // `~` is quoted, since a shell would expand it.
        assert_eq!(
            rediagnose_for(r"C:\Users\ADMINI~1\Temp\m.gguf", &plain, true),
            r"rocm diagnose --model C:\Users\ADMINI~1\Temp\m.gguf"
        );
        assert_eq!(
            rediagnose_for("~/m.gguf", &plain, false),
            "rocm diagnose --model '~/m.gguf'"
        );
        // A backslash is not plain to a POSIX shell.
        assert_eq!(
            rediagnose_for(r"a\b.gguf", &plain, false),
            r"rocm diagnose --model 'a\b.gguf'"
        );
        // A reference that starts with `-` is joined to the flag.
        assert_eq!(
            rediagnose_for("-x.gguf", &plain, false),
            "rocm diagnose --model=-x.gguf"
        );
        assert_eq!(
            rediagnose("$HOME/m.gguf", &plain),
            "rocm diagnose --model '$HOME/m.gguf'"
        );
    }

    /// A local file that was read but cannot be sized (here one shard of a
    /// split model) is offered to `rocm serve`, which takes a local path, with
    /// the path quoted for the shell where it needs to be.
    #[test]
    fn an_unsized_local_gguf_is_offered_to_rocm_serve_quoted() {
        let mut readiness = not_curated("models/split.gguf");
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host_with(24.0, 64.0),
            &FakeSource::with(llama_like(10).u32("split.count", 3).file(0)),
        );
        assert_eq!(
            readiness.estimate.as_ref().unwrap().cannot_estimate_reason,
            Some(CannotEstimateReason::SplitFile)
        );
        let fix = readiness.fix.as_ref().expect("a plan");
        assert_eq!(fix.verify, "rocm serve models/split.gguf");
        let mut spaced = not_curated("my models/split.gguf");
        attach_context_estimate(
            &mut spaced,
            &asked(8192),
            &host_with(24.0, 64.0),
            &FakeSource::with(llama_like(10).u32("split.count", 3).file(0)),
        );
        assert_eq!(
            spaced.fix.as_ref().expect("a plan").verify,
            "rocm serve 'my models/split.gguf'"
        );
    }

    /// A context flag with a reference that is neither curated nor a GGUF says
    /// the estimate is for GGUF models only, and reads nothing.
    #[test]
    fn a_context_flag_on_a_reference_that_is_not_a_gguf_says_why_it_was_not_applied() {
        let mut readiness = not_curated("some-bare-model-name");
        let source = FakeSource::weights(64);
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host_with(24.0, 64.0),
            &source,
        );
        assert_eq!(source.calls(), 0);
        assert!(readiness.estimate.is_none());
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("estimated for GGUF models only")),
            "{:?}",
            readiness.evidence
        );
    }

    /// A hub header that could not be fetched is not answered with `rocm serve`,
    /// which would download the whole file to fail the same way: the plan is to
    /// check the name and the network, and ask again.
    #[test]
    fn an_unreadable_hub_header_is_not_answered_with_a_download() {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        let fix = readiness.fix.as_ref().expect("a plan");
        assert!(fix.summary.contains("no route to Hugging Face"), "{fix:?}");
        assert_eq!(
            fix.verify,
            "rocm diagnose --model some-lab/never-curated-7b:Q4_K_M --context-size 8192"
        );
        assert!(
            fix.notes.iter().any(|note| note.contains("local path")),
            "{fix:?}"
        );
        assert!(!render_model_readiness_text(&readiness).contains("rocm serve some-lab"));
    }

    /// A pasted URL is read as a path and cannot be opened; the plan says which
    /// form is read instead.
    #[test]
    fn a_url_is_told_which_form_is_read() {
        let mut readiness = not_curated("https://huggingface.co/o/r/resolve/main/m.gguf");
        attach_context_estimate(
            &mut readiness,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        let fix = readiness.fix.as_ref().expect("a plan");
        assert!(
            fix.notes
                .iter()
                .any(|note| note.contains("`owner/repo:QUANT`")),
            "{fix:?}"
        );
    }

    /// A layout that gets figures and no verdict is sized and compared, so the
    /// "serve it and see" plan and the tracker route no longer apply, and no
    /// command would turn it into a verdict: there is no plan, and the detail
    /// says to compare the total with the GPU's memory.
    #[test]
    fn a_layout_without_a_verdict_gets_its_figures_and_no_plan() {
        for (layout, source) in [
            (
                "mixture-of-experts",
                FakeSource::with(llama_like(10).u32("llama.expert_count", 8).file(64)),
            ),
            (
                "state-space",
                FakeSource::with(llama_like(10).u32("llama.ssm.state_size", 16).file(64)),
            ),
        ] {
            let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
            attach_context_estimate(
                &mut readiness,
                &asked(4096),
                &host_with(24.0, 64.0),
                &source,
            );
            let estimate = readiness.estimate.as_ref().expect("an estimate");
            assert_eq!(
                estimate.cannot_estimate_reason,
                Some(CannotEstimateReason::UnmeasuredLayout),
                "{layout}"
            );
            assert!(estimate.total_gib.is_some(), "{layout}");
            assert!(readiness.fix.is_none(), "{layout}: {:?}", readiness.fix);
            assert!(readiness.route.is_none(), "{layout}");
            let text = render_model_readiness_text(&readiness);
            assert!(
                text.contains("compare it with this GPU's"),
                "{layout}: {text}"
            );
        }
    }

    #[test]
    fn a_gguf_that_cannot_be_read_is_not_sent_to_the_upstream_tracker() {
        // A typo'd path: the tracker cannot help with it, and the plan is to
        // check the path.
        let mut readiness = not_curated("models/nosuch.gguf");
        assert!(
            readiness.route.is_some(),
            "the catalog-miss route is the default"
        );
        attach_context_estimate(
            &mut readiness,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        let estimate = readiness.estimate.as_ref().unwrap();
        assert_eq!(
            estimate.cannot_estimate_reason,
            Some(CannotEstimateReason::HeaderUnreadable)
        );
        assert!(readiness.route.is_none());
        // A typo'd local path is a path to check, not a model to "serve directly".
        let fix = readiness.fix.as_ref().expect("a plan is still given");
        assert!(fix.summary.contains("check the path"), "{}", fix.summary);
        assert!(fix.commands.is_empty());
        assert_eq!(fix.verify, "rocm diagnose --model models/nosuch.gguf");
        let text = render_model_readiness_text(&readiness);
        assert!(!text.contains("report it:"), "{text}");
        assert!(
            !text.contains("serve this model directly") && !text.contains("rocm serve"),
            "{text}"
        );
    }

    fn compared_estimate() -> ModelReadiness {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut readiness,
            &asked(8192),
            &host_with(24.0, 64.0),
            &FakeSource::weights(2048),
        );
        assert!(readiness.estimate.as_ref().unwrap().available_gib.is_some());
        readiness
    }

    #[test]
    fn a_mask_on_a_host_whose_gpus_differ_is_said_to_be_unverified() {
        let notes = |readiness: &ModelReadiness| readiness.estimate.as_ref().unwrap().notes.clone();
        // Mask set, cards of two sizes: the figure may be for the other card.
        let mut readiness = compared_estimate();
        note_unverified_mask(&mut readiness, true, &[24_576, 8192]);
        assert!(
            notes(&readiness).iter().any(|n| n == MASK_UNVERIFIED_NOTE),
            "{:?}",
            notes(&readiness)
        );
        let text = render_model_readiness_text(&readiness);
        assert!(text.contains("has not been verified"), "{text}");
        // Whichever order the rows come in.
        let mut readiness = compared_estimate();
        note_unverified_mask(&mut readiness, true, &[8192, 24_576]);
        assert!(notes(&readiness).iter().any(|n| n == MASK_UNVERIFIED_NOTE));
        // No mask: nothing to doubt. Same cards: whichever is picked, the figure is the same.
        for (mask_set, totals) in [
            (false, vec![24_576_u64, 8192]),
            (true, vec![24_576, 24_576]),
            (true, vec![24_576]),
            (true, vec![]),
        ] {
            let mut readiness = compared_estimate();
            note_unverified_mask(&mut readiness, mask_set, &totals);
            assert!(
                !notes(&readiness).iter().any(|n| n == MASK_UNVERIFIED_NOTE),
                "{mask_set} {totals:?}"
            );
        }
        // An estimate that compared nothing has no figure to doubt.
        let mut unmeasured = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut unmeasured,
            &asked(8192),
            &HostFacts {
                accelerator_memory: AcceleratorMemory::Unknown,
                system_ram_gib: Some(64.0),
            },
            &FakeSource::weights(2048),
        );
        note_unverified_mask(&mut unmeasured, true, &[24_576, 8192]);
        assert!(
            !notes(&unmeasured).iter().any(|n| n == MASK_UNVERIFIED_NOTE),
            "{:?}",
            notes(&unmeasured)
        );
    }

    /// `owner/repo:dir/file.gguf` selects an exact file for the estimate; `rocm
    /// serve` reads only `owner/repo:QUANT`, so a verify line naming it would fail.
    /// Shown on a file that was read but could not be sized (one shard of a
    /// split model), where serving it is still the plan.
    #[test]
    fn the_verify_command_is_one_rocm_serve_accepts() {
        let verify = |model_ref: &str| {
            let mut readiness = not_curated(model_ref);
            attach_context_estimate(
                &mut readiness,
                &ContextRequest::default(),
                &host_with(24.0, 64.0),
                &FakeSource::with(llama_like(10).u32("split.count", 3).file(64)),
            );
            readiness
                .fix
                .expect("the plan stays for a file that could not be sized")
                .verify
        };
        assert_eq!(
            verify("some-lab/never-curated-7b:Q4_K_M"),
            "rocm serve some-lab/never-curated-7b:Q4_K_M"
        );
        assert_eq!(
            verify("some-lab/never-curated-7b:sub/model-Q4_K_M.gguf"),
            "rocm diagnose --model some-lab/never-curated-7b:sub/model-Q4_K_M.gguf"
        );
    }

    #[test]
    fn the_default_context_is_labelled_as_a_default() {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut readiness,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &FakeSource::weights(64),
        );
        let estimate = readiness.estimate.as_ref().unwrap();
        assert_eq!(estimate.context_size, 4096, "the default --help documents");
        assert!(estimate.context_size_defaulted);
        assert_eq!(estimate.kv_cache_type, KvCacheType::F16);
        let text = render_model_readiness_text(&readiness);
        assert!(
            text.contains("(the default; pass --context-size to change it)"),
            "{text}"
        );
    }

    /// The three states of host memory, for a model that would otherwise fit:
    /// none produces a comparison, each says its own reason, and the figures
    /// the file allows are still shown.
    #[test]
    fn host_memory_that_cannot_be_compared_never_yields_a_fits_or_blocked_answer() {
        for (memory, reason) in [
            (AcceleratorMemory::None, CannotEstimateReason::NoGpu),
            (
                AcceleratorMemory::Unknown,
                CannotEstimateReason::AcceleratorMemoryUnknown,
            ),
            (
                AcceleratorMemory::UnifiedMemoryUnreadable,
                CannotEstimateReason::UnifiedMemoryUnreadable,
            ),
        ] {
            let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
            let host = HostFacts {
                accelerator_memory: memory,
                system_ram_gib: Some(64.0),
            };
            attach_context_estimate(
                &mut readiness,
                &asked(4096),
                &host,
                &FakeSource::weights(64),
            );
            let estimate = readiness.estimate.as_ref().unwrap();
            assert_eq!(
                estimate.outcome,
                EstimateOutcome::CannotEstimate,
                "{memory:?}"
            );
            assert_eq!(estimate.cannot_estimate_reason, Some(reason), "{memory:?}");
            assert!(
                estimate.total_gib.is_some(),
                "{memory:?}: the need was not shown"
            );
            assert_eq!(readiness.verdict, ModelVerdict::Undetermined);
            // Sized, so the "serve it directly" plan no longer applies; any plan
            // left is the host's own (see the test after this one).
            assert!(readiness.route.is_none(), "{memory:?}");
            assert!(
                readiness
                    .fix
                    .as_ref()
                    .is_none_or(|fix| !fix.summary.contains("serve this model directly")),
                "{memory:?}: {:?}",
                readiness.fix
            );
            let text = render_model_readiness_text(&readiness);
            assert!(text.contains(": UNDETERMINED"), "{memory:?}: {text}");
            assert!(!text.contains("ESTIMATED:"), "{memory:?}: {text}");
            assert!(text.contains("KV cache (f16)"), "{memory:?}: {text}");
        }
    }

    #[test]
    fn an_unreachable_or_unreadable_file_is_a_gap_in_the_estimate_not_a_verdict_on_the_model() {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut readiness,
            &asked(4096),
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        let estimate = readiness.estimate.as_ref().unwrap();
        assert_eq!(
            estimate.cannot_estimate_reason,
            Some(CannotEstimateReason::HeaderUnreadable)
        );
        assert!(
            estimate
                .detail
                .as_deref()
                .unwrap()
                .contains("could not be reached")
        );
        assert_eq!(estimate.total_gib, None);
        // The plan is to act on that reason and ask again, not to serve a file
        // whose header could not even be read.
        let fix = readiness.fix.as_ref().expect("a plan");
        assert!(fix.summary.contains("act on the reason above"), "{fix:?}");
        assert!(fix.verify.starts_with("rocm diagnose --model "), "{fix:?}");

        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut readiness,
            &asked(4096),
            &host_with(24.0, 64.0),
            &FakeSource::with(b"not a gguf file at all".to_vec()),
        );
        let estimate = readiness.estimate.as_ref().unwrap();
        assert_eq!(
            estimate.cannot_estimate_reason,
            Some(CannotEstimateReason::HeaderUnreadable)
        );
        assert!(
            estimate
                .detail
                .as_deref()
                .unwrap()
                .contains("not a GGUF file")
        );
    }

    /// The plan's note was written for a model that is not a GGUF reference. For
    /// one that is, and whose header could not be sized, it must not say so.
    #[test]
    fn the_plan_note_never_claims_a_gguf_is_not_one() {
        let note_of = |source: &FakeSource| {
            let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
            attach_context_estimate(&mut readiness, &asked(4096), &host_with(24.0, 64.0), source);
            let text = render_model_readiness_text(&readiness);
            assert!(!text.contains("not a GGUF reference"), "{text}");
            readiness.fix.expect("the plan stays").notes.join("\n")
        };
        // Unreadable (its own plan, about the name and the network), and read
        // but not sizeable (a shard of a split model).
        let unreadable = note_of(&FakeSource::failing());
        assert!(unreadable.contains("no token is sent"), "{unreadable}");
        let shard = note_of(&FakeSource::with(
            llama_like(10).u32("split.count", 3).file(64),
        ));
        assert!(shard.contains("could not size this GGUF"), "{shard}");
        // Sized: there is no plan left to carry a note.
        let mut sized = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut sized,
            &asked(4096),
            &host_with(24.0, 64.0),
            &FakeSource::weights(64),
        );
        assert!(sized.fix.is_none());
        // Without a context flag nothing is read and the original note stands:
        // that reference is not asked about as a GGUF at all.
        let mut plain = not_curated("lab/plain-name");
        attach_context_estimate(
            &mut plain,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &FakeSource::failing(),
        );
        assert!(plain.estimate.is_none());
        assert!(
            plain
                .fix
                .unwrap()
                .notes
                .join("\n")
                .contains("for one that is not a GGUF reference")
        );
    }

    #[test]
    fn flags_are_never_dropped_in_silence_when_the_catalog_could_not_be_read() {
        let unreachable = || {
            assess_model_readiness(
                "/models/local.gguf",
                &ModelCatalogSource::Unreachable {
                    detail: "no index".to_owned(),
                },
                &host_with(24.0, 64.0),
                &engine_choice,
            )
        };
        let source = FakeSource::weights(64);
        let mut readiness = unreachable();
        let before = readiness.evidence.clone();
        attach_context_estimate(
            &mut readiness,
            &asked(4096),
            &host_with(24.0, 64.0),
            &source,
        );
        assert!(readiness.estimate.is_none());
        assert_eq!(source.calls(), 0, "which answer it is owed is not known");
        assert_eq!(readiness.evidence.len(), before.len() + 1);
        assert!(
            readiness
                .evidence
                .last()
                .unwrap()
                .contains("were not applied"),
            "{:?}",
            readiness.evidence
        );
        assert_eq!(
            readiness.undetermined_reason,
            Some(UndeterminedReason::CatalogUnreachable)
        );
        // No flag, no line: the answer is exactly what it was.
        let mut readiness = unreachable();
        let before = readiness.clone();
        attach_context_estimate(
            &mut readiness,
            &ContextRequest::default(),
            &host_with(24.0, 64.0),
            &source,
        );
        assert_eq!(readiness, before);
    }

    /// A compared estimate says once that `rocm serve` would not use the size it
    /// was sized for: with the suggestions' options when it does not fit, on its
    /// own when it fits; an estimate that compared nothing says neither.
    #[test]
    fn a_compared_estimate_says_rocm_serve_passes_no_context() {
        let warnings = |memory_gib: f64| {
            let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
            attach_context_estimate(
                &mut readiness,
                &asked(8192),
                &host_with(memory_gib, 64.0),
                &FakeSource::weights(2048),
            );
            readiness.warnings
        };
        let fits = warnings(24.0);
        assert!(fits.iter().any(|w| w == SERVE_CONTEXT_NOTE), "{fits:?}");
        assert!(!fits.iter().any(|w| w == SERVE_OPTIONS_NOTE), "{fits:?}");
        let misfit = warnings(2.0);
        assert!(misfit.iter().any(|w| w == SERVE_OPTIONS_NOTE), "{misfit:?}");
        assert!(
            !misfit.iter().any(|w| w == SERVE_CONTEXT_NOTE),
            "{misfit:?}"
        );
        let mut uncompared = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut uncompared,
            &asked(8192),
            &HostFacts {
                accelerator_memory: AcceleratorMemory::Unknown,
                system_ram_gib: None,
            },
            &FakeSource::weights(2048),
        );
        assert!(
            !uncompared
                .warnings
                .iter()
                .any(|w| w == SERVE_CONTEXT_NOTE || w == SERVE_OPTIONS_NOTE)
        );
    }

    #[test]
    fn a_one_token_context_reads_as_one_token() {
        let mut readiness = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut readiness,
            &asked(1),
            &host_with(24.0, 64.0),
            &FakeSource::weights(64),
        );
        let text = render_model_readiness_text(&readiness);
        assert!(
            text.contains(": ESTIMATED: fits at 1 token (calibrated"),
            "{text}"
        );
        assert!(!text.contains("1 tokens"), "{text}");
    }

    #[test]
    fn only_a_model_outside_the_catalog_that_names_a_gguf_reaches_the_source() {
        let source = FakeSource::weights(64);
        let host = host_with(24.0, 64.0);
        // A curated model: no network, no estimate; and an explicit context
        // flag is answered with why it does not apply, not silently ignored.
        let mut curated = not_curated(SMALL_MODEL_REF);
        attach_context_estimate(&mut curated, &asked(8192), &host, &source);
        assert!(curated.estimate.is_none());
        assert!(
            curated
                .evidence
                .iter()
                .any(|line| line.contains("do not apply to a curated recipe")),
            "{:?}",
            curated.evidence
        );
        let mut curated = not_curated(SMALL_MODEL_REF);
        let before = curated.evidence.clone();
        attach_context_estimate(&mut curated, &ContextRequest::default(), &host, &source);
        assert_eq!(
            curated.evidence, before,
            "an unasked curated answer must not change"
        );
        // A name that is not a GGUF reference.
        let mut other = not_curated("never-curated");
        let before = other.evidence.len();
        attach_context_estimate(&mut other, &ContextRequest::default(), &host, &source);
        assert!(other.estimate.is_none() && other.evidence.len() == before);
        attach_context_estimate(&mut other, &asked(4096), &host, &source);
        assert!(other.estimate.is_none());
        assert!(
            other
                .evidence
                .iter()
                .any(|line| line.contains("GGUF models only"))
        );
        // A reference of the right shape, but the catalog could not be read:
        // the answer is about the catalog, and an estimate would contradict it.
        let mut unreachable = assess_model_readiness(
            "some-lab/never-curated-7b:Q4_K_M",
            &ModelCatalogSource::Unreachable {
                detail: "gone".to_owned(),
            },
            &host,
            &engine_choice,
        );
        attach_context_estimate(&mut unreachable, &asked(4096), &host, &source);
        assert!(unreachable.estimate.is_none());
        assert_eq!(
            source.calls(),
            0,
            "the source was reached for a model that did not qualify"
        );
    }

    #[test]
    fn an_answer_without_an_estimate_serializes_exactly_as_it_did_before() {
        let readiness = not_curated("some-lab/never-curated-7b");
        let json = serde_json::to_value(&readiness).unwrap();
        assert!(json.get("estimate").is_none(), "{json}");
        // And a JSON answer from before the field existed still reads.
        let round_trip: ModelReadiness = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, readiness);
        let mut with = not_curated("some-lab/never-curated-7b:Q4_K_M");
        attach_context_estimate(
            &mut with,
            &asked(4096),
            &host_with(24.0, 64.0),
            &FakeSource::weights(64),
        );
        let json = serde_json::to_value(&with).unwrap();
        assert_eq!(json["estimate"]["outcome"], "fits");
        assert_eq!(json["estimate"]["kv_cache_type"], "f16");
        assert_eq!(json["verdict"], "undetermined");
    }
}
