// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Will this model run on this machine?
//!
//! `examine` answers what is on the host and `diagnose` answers what went wrong
//! after a failure. This answers the question that comes before both, from data
//! the CLI already has: a curated [`ModelRecipeRecord`] (dtype, quantization,
//! minimum GPU memory, device policy, preferred engines, declared alternatives)
//! composed with what the host can offer. **It fetches nothing** — no weights,
//! no metadata probe, no network call of any kind — which is the whole point:
//! today a user learns a model will not run by starting a download and waiting
//! for it to fail.
//!
//! Three distinctions carry the design, and collapsing any of them produces a
//! confident wrong answer:
//!
//! - A catalog that could not be **read** is not a model that does not **fit**.
//!   [`ModelCatalogSource`] keeps them apart and [`ModelVerdict::Undetermined`]
//!   is where the first one lands.
//! - Memory the CLI **could not measure** is not memory the host **does not
//!   have**. See [`AcceleratorMemory`].
//! - The memory an APU *reports* is not the memory its engine *allocates from*.
//!   See [`AcceleratorMemory::UnifiedSystemMemory`].
//!
//! Curated recipes only. Answering for an arbitrary hub model needs a metadata
//! probe and a table mapping quantization schemes to available kernels, neither
//! of which has a source of truth here; such a model is reported
//! [`UndeterminedReason::ModelNotCurated`], never blocked.

use crate::ModelRecipeRecord;
use crate::ModelRecipeRegistry;
use crate::diagnose::{Fix, Route, upstream_route};
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
}

/// The memory pool an engine would allocate this model from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcceleratorMemory {
    /// Dedicated VRAM, in GiB. What a discrete card reports is what it has.
    Dedicated(f64),
    /// GTT-backed system memory, in GiB, on a host with no private VRAM.
    ///
    /// An APU has none. `amd-smi` reports the fixed BIOS carve-out (often ~4
    /// GiB) as `total_vram` while the allocator serves the model out of system
    /// RAM, so on a 128 GiB Strix Halo the carve-out is measuring the wrong
    /// pool — blocking a 22 GiB recipe against it would refuse a machine that
    /// runs it comfortably.
    UnifiedSystemMemory(f64),
    /// No GPU is visible to ROCm on this host.
    None,
    /// There is a GPU, but its memory could not be read.
    ///
    /// Distinct from [`AcceleratorMemory::None`] on purpose: one is a fact about
    /// the machine and the other is a gap in what the CLI can see, and they do
    /// not deserve the same verdict.
    Unknown,
}

impl AcceleratorMemory {
    /// The figure to compare a recipe minimum against, when there is one.
    const fn measured_gib(self) -> Option<f64> {
        match self {
            Self::Dedicated(gib) | Self::UnifiedSystemMemory(gib) => Some(gib),
            Self::None | Self::Unknown => None,
        }
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

fn format_gib(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
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
                "`rocm diagnose --model` answers for curated recipes only: judging an arbitrary \
                 model needs metadata this CLI does not fetch"
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
                "`rocm serve` accepts models outside this catalog; it just cannot say in advance \
                 whether they will fit"
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
    if let AcceleratorMemory::UnifiedSystemMemory(gib) = host.accelerator_memory {
        evidence.push(format!(
            "this host has no dedicated VRAM; its engine allocates from {} of GTT-backed system \
             memory, which is the figure compared below",
            format_gib(gib)
        ));
    }

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
        readiness.fix = Some(Fix {
            summary: "make a GPU visible to ROCm, then ask again".to_owned(),
            commands: vec!["rocm diagnose".to_owned()],
            verify: format!("rocm diagnose --model {model_ref}"),
            notes: vec![
                "`rocm diagnose` matches this machine against the known reasons a GPU does not \
                 show up"
                    .to_owned(),
            ],
            ..Fix::default()
        });
        return readiness;
    }

    match (required, available) {
        // The second collapse: memory that could not be measured is not memory
        // the host does not have. Blocking here would name the model as the
        // problem when the gap is in the host telemetry.
        (Some(required), None) => {
            readiness.verdict = ModelVerdict::Undetermined;
            readiness.undetermined_reason = Some(UndeterminedReason::AcceleratorMemoryUnknown);
            readiness.evidence.push(format!(
                "this recipe needs {}, but the GPU memory on this host could not be read; that \
                 is a gap in what the CLI can see, not a verdict about the model",
                format_gib(required)
            ));
            readiness.fix = Some(Fix {
                summary: "let the CLI read this host's GPU memory, then ask again".to_owned(),
                commands: vec!["amd-smi metric --json".to_owned()],
                verify: format!("rocm diagnose --model {model_ref}"),
                notes: vec![
                    "the reading comes from `amd-smi`; if it is missing or failing, the ROCm \
                     install is what needs attention first"
                        .to_owned(),
                ],
                ..Fix::default()
            });
        }
        (Some(required), Some(available)) if available < required => {
            readiness.verdict = ModelVerdict::Blocked;
            readiness.evidence.push(format!(
                "this recipe needs {}, and this host offers {}",
                format_gib(required),
                format_gib(available)
            ));
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

/// The word this verdict is reported under.
const fn verdict_label(verdict: ModelVerdict) -> &'static str {
    match verdict {
        ModelVerdict::Ready => "READY",
        ModelVerdict::Degraded => "DEGRADED",
        ModelVerdict::Blocked => "BLOCKED",
        ModelVerdict::Undetermined => "UNDETERMINED",
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
        verdict_label(readiness.verdict)
    );
    if let Some(canonical) = &readiness.canonical_model_id {
        let _ = writeln!(out, "   recipe: {canonical}");
    }
    for line in &readiness.evidence {
        let _ = writeln!(out, "   - {line}");
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
    // them above the plan would read as reasons for it.
    for warning in &readiness.warnings {
        let _ = writeln!(out, "   recipe note: {warning}");
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

    /// An APU reports a BIOS carve-out as its VRAM while allocating from system
    /// RAM. Comparing a recipe minimum against the carve-out refuses a machine
    /// that runs the model comfortably, so the pool the engine actually uses is
    /// what gets compared.
    #[test]
    fn an_apu_is_judged_on_the_memory_its_engine_allocates_from() {
        let registry = builtin_model_recipe_registry();
        let catalog = ModelCatalogSource::Available(&registry);
        // A 22 GiB recipe against a 4 GiB carve-out on a 128 GiB machine.
        let strix_halo = HostFacts {
            accelerator_memory: AcceleratorMemory::UnifiedSystemMemory(128.0),
            system_ram_gib: Some(128.0),
        };
        let readiness = assess_model_readiness("qwen3.6", &catalog, &strix_halo, &engine_choice);
        assert_eq!(
            readiness.verdict,
            ModelVerdict::Ready,
            "a 22 GiB recipe runs on a 128 GiB unified-memory host: {:?}",
            readiness.evidence
        );
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("GTT-backed system memory")),
            "the report has to say which pool it compared, or the figure reads as VRAM: {:?}",
            readiness.evidence
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
}
