// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! ROCm failure-mode diagnosis.
//!
//! Rust port of the `rocm-doctor` skill's `diagnose.py`. It matches an
//! [`Examination`] (plus optional user symptom text) against a **closed list**
//! of known misconfigurations and returns ranked [`Diagnosis`] results, each
//! with the evidence it used and a [`Fix`] (plan + verify step). When nothing
//! matches it routes the user upstream rather than guessing.
//!
//! The catalog is deliberately closed: new failure modes are added here, not
//! invented at runtime. Keyword tables, thresholds, and tracker URLs are the
//! data; the per-check logic mirrors `diagnose.py` field-for-field so the two
//! stay behaviorally identical.

use crate::examine::Examination;
use host::{
    check_1_arch_not_in_wheel, check_2_hsa_override_unneeded, check_6_path_missing,
    check_8_wheel_rocm_mismatch, check_9_igpu_dgpu_collision, check_20_gpu_unsupported_by_rocm,
};
use linux::{
    check_3_rocm_kernel_unsupported, check_4_render_group, check_5_amdgpu_blacklisted,
    check_7_stale_repos, check_10_container_devices, check_11_iommu_hang,
    check_12_repo_native_broken,
};
use regex::Regex;
use report::route_when_no_match;
use scoring::{
    KEYWORDS_COMGR_CONFLICT, KEYWORDS_SHM_TOO_SMALL, KEYWORDS_TORCH_DLPACK_CUDA_VARIANT,
    KEYWORDS_VLLM_OOM, finalize, keyword_score, keyword_score_collapsing_overlaps, zero,
};
use serde::{Deserialize, Serialize};
use windows::{
    check_13_hip_sdk_missing, check_14_adrenalin_too_old, check_15_msvc_redist,
    check_21_msvc_redist_outdated, check_22_display_driver_old,
};
use wsl::{
    check_wsl_1_gpu_not_exposed, check_wsl_2_dxcore_missing, check_wsl_3_rocdxg_missing,
    check_wsl_4_rocdxg_not_linked, check_wsl_5_distro_too_old, check_wsl_6_host_driver_too_old,
    check_wsl_7_wsl1,
};

mod host;
mod linux;
mod report;
mod scoring;
#[cfg(test)]
mod test_support;
mod windows;
mod wsl;

pub use linux::dkms_driver_plan_is_supported;
pub use report::{render_report_text, upstream_route};

/// At/above this score a diagnosis is treated as a real match.
pub const MIN_SCORE_FOR_MATCH: i32 = 50;

/// At/above this score the agent may propose the fix immediately.
pub const HIGH_CONFIDENCE: i32 = 75;

/// A proposed remediation for a [`Diagnosis`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fix {
    pub summary: String,
    pub commands: Vec<String>,
    pub needs_sudo: bool,
    pub needs_reboot: bool,
    pub needs_relogin: bool,
    pub fix_id: String,
    /// Whether `rocm fix <fix_id>`, run with no extra arguments, will change
    /// the machine.
    ///
    /// Kept alongside [`Fix::class`], which it is derived from, because it is
    /// the field published consumers already read. It is not redundant detail
    /// so much as a narrower question: `class` says *what* the CLI will do,
    /// this says only whether the machine is about to change.
    pub auto_applicable: bool,
    /// What the CLI will do with this entry on the examined machine.
    ///
    /// `#[serde(default)]` so a payload written before this field existed still
    /// deserializes; the default understates rather than overstates.
    #[serde(default)]
    pub class: crate::fix::FixClass,
    /// The argument this entry is waiting for, when `class` is
    /// [`crate::fix::FixClass::NeedsArgument`]. Read from the catalog beside
    /// `class` so the flags line renders identically from `rocm fix` and here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needs: Option<String>,
    pub notes: Vec<String>,
    pub verify: String,
}

/// A single scored match against the failure-mode catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Diagnosis {
    pub id: String,
    pub title: String,
    pub score: i32,
    pub evidence: Vec<String>,
    pub fix: Option<Fix>,
}

/// Where to send a report when no catalog entry matches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Route {
    pub target: String,
    pub url: String,
}

/// The full diagnosis output (mirrors `diagnose.py --json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnoseReport {
    /// All nonzero-score diagnoses, highest score first.
    ///
    /// Every checker that fired at all lands here, including ones scoring below
    /// [`MIN_SCORE_FOR_MATCH`] — so a non-empty `matched` does NOT mean a cause
    /// was established. Read [`DiagnoseReport::has_match`] for that.
    pub matched: Vec<Diagnosis>,
    /// Whether any entry in `matched` cleared [`MIN_SCORE_FOR_MATCH`].
    ///
    /// Serialized because a JSON consumer cannot otherwise tell a real cause
    /// from a weak signal, and the obvious substitute — "is `matched` empty?" —
    /// is wrong. Several checkers open with a nonzero base score for a
    /// situation that is merely *potentially* relevant (being in a container,
    /// having an APU alongside a discrete GPU), so a perfectly healthy host
    /// produces a non-empty `matched` full of sub-threshold entries. A caller
    /// gating on emptiness proposes a fix for a machine with nothing wrong, and
    /// never routes the user to `route_when_no_match`.
    ///
    /// Computed at construction; [`DiagnoseReport::has_match`] recomputes from
    /// `matched` and stays the authority for Rust callers.
    #[serde(default)]
    pub has_match: bool,
    pub min_score_for_match: i32,
    pub high_confidence_threshold: i32,
    pub route_when_no_match: Route,
    /// Set when the catalog has no entries at all for the running platform, and
    /// carrying the routing note that says so.
    ///
    /// Decided purely by whether the catalog covers the running platform — never
    /// by a score. When this is `Some`, no checker ran, so `matched` is empty and
    /// `has_match` is false; a consumer must not wait for sub-threshold rows
    /// alongside it. Pinned by
    /// `an_out_of_scope_report_never_carries_matched_entries`.
    #[serde(default)]
    pub out_of_scope: Option<String>,
    /// The verdict on a model, when `--model` named one.
    ///
    /// Attached by the caller after the fact rather than produced by
    /// [`diagnose`]: answering it needs the host's GPU memory and the engine
    /// `serve` would select, neither of which an [`Examination`] carries. It
    /// rides on this report rather than replacing it because the environment
    /// answer stays true and useful either way — a blocked model on a host whose
    /// driver is also misconfigured is two findings, not one.
    #[serde(default)]
    pub model: Option<crate::model_readiness::ModelReadiness>,
}

/// Whether any diagnosis cleared [`MIN_SCORE_FOR_MATCH`].
///
/// One place the rule is written, so the serialized `has_match` field and the
/// [`DiagnoseReport::has_match`] accessor cannot answer differently.
fn any_cleared_threshold(matched: &[Diagnosis]) -> bool {
    matched.iter().any(|d| d.score >= MIN_SCORE_FOR_MATCH)
}

impl DiagnoseReport {
    /// Whether at least one diagnosis cleared [`MIN_SCORE_FOR_MATCH`].
    #[must_use]
    pub fn has_match(&self) -> bool {
        any_cleared_threshold(&self.matched)
    }
}

// ---------------------------------------------------------------------------
// Per-misconfiguration checkers (1:1 with diagnose.py)
//
// A new checker goes at the END of this section, next to where its `CHECKERS`
// entry was appended. Inserting one between an existing function and the doc
// comment above it silently rebinds that comment to the new item: Rust joins a
// run of `///` lines with no blank line between them, so the old rationale ends
// up documenting the new constant and the old function is left with none. That
// compiles, and clippy, rustfmt and every test pass.
// ---------------------------------------------------------------------------

/// A required anchor before [`KEYWORDS_VLLM_OOM`] is even scored: the word
/// "vllm" itself, or its distinctive VRAM-reservation flag. Without one of
/// these, a generic HIP/CUDA/PyTorch OOM string is not evidence of *this*
/// failure mode -- see the comment on [`KEYWORDS_VLLM_OOM`]. `tensor[-_]parallel`
/// is deliberately NOT an anchor: it is a Megatron/DeepSpeed term (rocm-cli does
/// not serve one model across GPUs), so anchoring on it would misattribute those
/// frameworks' OOMs to vLLM.
///
/// `gpu[-_]memory[-_]utilization` is also a (20-point) entry in
/// [`KEYWORDS_VLLM_OOM`], so a line that only echoes the flag -- a config dump,
/// or a paste of rocm-cli's own low-VRAM hint, which prints
/// `--gpu-memory-utilization` -- is both anchored and self-scoring. That is
/// harmless *because* only anchored lines are scored: such a line is worth 20,
/// far below [`MIN_SCORE_FOR_MATCH`], so it can surface as a weak signal in
/// `matched` but can never carry a verdict, and it can no longer lend its anchor
/// to an OOM elsewhere in the paste. It stays an anchor because the flag is
/// vLLM-specific, and because a real vLLM OOM often names it on the failing line.
const VLLM_ANCHOR_PATTERN: &str = r"vllm|gpu[-_]memory[-_]utilization";

/// A canonical vLLM-OOM `--symptom` string guaranteed to clear
/// [`MIN_SCORE_FOR_MATCH`].
///
/// It carries two independent signals (the torch exception class *and* the HIP
/// allocator message) on one anchored line, so it scores even after overlap
/// de-duplication. Both OOM surfaces — the vLLM engine's startup-failure hint
/// and the `rocm serve` summary note — fall back to this when the user's actual
/// failing line cannot be rendered as one intact single-quoted shell argument
/// (`quotable_in_single_quotes` rejects it), so the `rocm diagnose` command they
/// print always reports a cause. Note the condition is *quotability*, not
/// diagnosability: `vllm_oom_diagnose_symptom` returns the raw line with its
/// control bytes intact, and an escape-bearing line scores on its rendered rows
/// but is refused by the quoting guard, which is what sends it here.
pub const VLLM_OOM_CANONICAL_SYMPTOM: &str = "vllm: torch.OutOfMemoryError: HIP out of memory";

/// The lines of `symptom` that carry the vLLM OOM anchor (`vllm` or
/// `gpu_memory_utilization`), joined by newlines — the only text
/// [`check_16_vllm_oom`] is allowed to score.
///
/// Co-occurrence anywhere in a pasted log is too loose -- a stray `vllm` mention
/// re-enables the very misattribution the anchor exists to prevent (a
/// `llama.cpp` OOM in a paste that also names vLLM). Returning the anchored
/// lines rather than a yes/no gate is what makes that true: regex matching
/// ignores line boundaries, so scoring the whole symptom behind a boolean gate
/// still counted keyword hits from *unanchored* lines at full weight, and one
/// benign `gpu_memory_utilization` config echo was enough to hand another
/// framework's OOM a HIGH_CONFIDENCE vLLM verdict
/// (`only_the_anchored_lines_are_scored_not_the_whole_paste`).
///
/// The join is by `\n` so that no pattern can straddle two lines: the regexes
/// here are literals or use `.*`, which does not match a newline. Dropping the
/// unanchored lines therefore cannot fabricate a match across the seam.
///
/// The same-line "anchor + OOM token" requirement falls out of this: an anchored
/// line with no OOM token contributes nothing, so the score is 0 and the checker
/// returns no diagnosis. The engine's emitted `vllm: <failing line>` keeps the
/// anchor and the error on one line by construction.
///
/// The split is [`crate::terminal::rendered_lines`], not `str::lines()` and not
/// a set of separator characters. `lines()` splits on `\n` and `\r\n` only, so a
/// lone `\r` was ordinary text to it — and a bare CR is what every progress bar
/// in this ecosystem emits to repaint its line (tqdm, pip, huggingface). A raw
/// terminal capture pasted into `rocm diagnose --symptom '...'` therefore
/// collapsed into one giant "line", the anchor matched somewhere in it, and
/// every keyword in the paste scored — exactly the whole-paste scoring this
/// function exists to prevent, reinstated by one byte
/// (`any_line_advance_is_a_boundary_not_scoreable_text`).
///
/// Adding `\r` to the split fixed that one byte and nothing else, because a
/// two-character allowlist is not the boundary a terminal renders. Every other
/// way of starting a new line — `ESC E` (`NEL`), `ESC D` (`IND`), `CSI n B`
/// (`CUD`), which are terminfo's `nel`/`ind`/`cud1` and ordinary output from
/// curses- and `rich`-style progress UIs — collapsed the paste just as
/// completely, as did `\x0b`, `\x0c`, `\u{85}`, `U+2028`/`U+2029` and any stray
/// control byte. Each one scored a `llama.cpp` OOM at 95 against a `vllm` anchor
/// from a different rendered line, past a `HIGH_CONFIDENCE` threshold of 75, and
/// cited the other engine's tokens as its evidence.
///
/// What this is *not* is a terminal emulator. The contract is one-sided, and
/// [`crate::terminal::rendered_lines`] is where to read it — that doc carries
/// the contract itself and points on to the module section holding the detail
/// behind it, including the single shape in which two rendered rows still come
/// back as one segment. It is deliberately not restated here: a second copy is
/// a stale copy waiting to happen, and this function is where a stale one would
/// do the damage. What matters at this call site is the direction of the error
/// that remains: a capture whose cursor addressing that walk does not model
/// loses a diagnosis rather than inventing one, and the canonical-symptom
/// fallback already covers that. The misattribution is the failure mode with a
/// user-visible cost, and this is what narrows it to the one shape that
/// contract calls out.
///
/// Sharing that walk with the vLLM engine's sanitizer is also what keeps the
/// `SGR` exception right. A colourised logger emits `ESC [ 31 m` *inside* a
/// line, so a splitter that treats every escape as a boundary cuts a genuine
/// `vllm: ... torch.OutOfMemoryError` between its anchor and its error token and
/// silently stops diagnosing it (measured: 95 → 0).
///
/// `\r\n` yields an empty segment, which carries no anchor and is dropped.
fn vllm_anchored_lines(symptom: &str) -> String {
    let Ok(anchor) = Regex::new(VLLM_ANCHOR_PATTERN) else {
        return String::new();
    };
    crate::terminal::rendered_lines(symptom)
        .into_iter()
        .filter(|line| anchor.is_match(&line.to_lowercase()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether `symptom` clears [`MIN_SCORE_FOR_MATCH`] for the vLLM-OOM failure
/// mode.
///
/// This is the OOM *classifier* the whole stack scores against. It is not what
/// the engine calls directly any more: `rocm_core::vllm_oom_diagnose_symptom`
/// owns line selection for both surfaces and reaches this per rendered row, and
/// what those surfaces then branch on before printing is
/// `quotable_in_single_quotes` — a line this function accepts is still replaced
/// by [`VLLM_OOM_CANONICAL_SYMPTOM`] when it cannot be rendered as one intact
/// single-quoted argument. Scoring here and admissibility there are deliberately
/// separate questions; keeping them straight is what stops a colourised line
/// being printed as a runnable lookalike of the user's error.
#[must_use]
pub fn vllm_oom_symptom_is_diagnosable(symptom: &str) -> bool {
    scores_on_every_engine_host(symptom, "fix-16-vllm-oom", run_all_checks)
}

/// vLLM ran the GPU out of memory. Keyword-only: an [`Examination`] carries no
/// per-GPU VRAM or tenancy fields, so nothing structural can corroborate this —
/// the user must pass the error via `--symptom`, or arrive from the serve
/// failure note that points here. `e` is therefore unused.
fn check_16_vllm_oom(_e: &Examination, symptom: &str) -> Diagnosis {
    // Score the anchored lines only, never the whole paste. The anchor is the
    // *only* thing separating this failure mode from any other framework's
    // PyTorch/HIP OOM, so a boolean gate over the whole symptom is not enough:
    // keyword hits from unanchored lines would still count at full weight. Any
    // future change here must keep the scored text restricted to anchored lines.
    let anchored = vllm_anchored_lines(symptom);
    if anchored.is_empty() {
        // No vLLM anchor anywhere: don't attribute a bare framework OOM to vLLM.
        return zero("fix-16-vllm-oom", "vLLM GPU out of memory");
    }
    let (score, evidence) = keyword_score_collapsing_overlaps(&anchored, KEYWORDS_VLLM_OOM);
    if score <= 0 {
        return zero("fix-16-vllm-oom", "vLLM GPU out of memory");
    }
    // Two different faults share this error, and the remediation splits on which
    // one it is: on a shared/busy GPU vLLM's fixed ~90% reservation collides
    // with memory already in use, so lowering the reservation is the fix; for a
    // model that genuinely does not fit, lowering it only trades an earlier OOM
    // for a later one, so the wording must not present the knob as the answer in
    // that case.
    let summary = format!(
        "{} If instead the model genuinely does not fit in this GPU's VRAM, lowering the \
         reservation will not help - use a smaller or quantized model (rocm-cli serves one \
         model on a single GPU; it does not shard a model across GPUs).",
        crate::VLLM_GPU_MEMORY_UTILIZATION_HINT
    );
    let fix = Fix {
        summary,
        // Byte-identical to the `fix-16-vllm-oom` catalog recipe's block, and
        // pinned there by `the_oom_plan_matches_the_catalog_copy`. The two are
        // one plan in two modules and a user may meet either copy.
        commands: vec![
            "# Case 1 -- shared/busy GPU (tenancy collision): lower the reservation,".to_owned(),
            "# or steer vLLM onto a less-busy device:".to_owned(),
            "rocm serve <model> --gpu-memory-utilization 0.5".to_owned(),
            "rocm serve <model> --gpu <index>".to_owned(),
            "# Case 2 -- the model genuinely does not fit: the reservation is not the".to_owned(),
            "# problem; pick a smaller or quantized model (single-GPU serving only).".to_owned(),
        ],
        fix_id: "fix-16-vllm-oom".to_owned(),
        verify: "rocm serve <model> <case-appropriate options above>   # re-run and watch for a clean startup".to_owned(),
        notes: vec![
            "Only lower --gpu-memory-utilization when the GPU is shared or already busy; on a GPU dedicated to this server it cannot create the room a too-large model needs.".to_owned(),
            "This entry is keyword-matched from the error text: `rocm diagnose` cannot see per-GPU VRAM or tenancy, so pass the failure with --symptom (or arrive from the `rocm serve` failure note).".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-16-vllm-oom",
        "vLLM ran the GPU out of memory at startup",
        score,
        evidence,
        fix,
    )
}

/// A code object manager library that does not belong to the active runtime.
///
/// The rule is symmetric, and that is what makes it safe. Rather than asking
/// "is this the wheel copy?" and then carving out an exception for the managed
/// runtime, it asks one question of both libraries: **does the copy of
/// `libamd_comgr` that would load belong to the same installation as the copy of
/// `libamdhip64` that would load?** If it does, there is nothing wrong, whichever
/// installation that is.
///
/// Every case falls out of that instead of being legislated. A healthy managed
/// install takes both libraries from the managed runtime, so nothing differs and
/// nothing is reported -- no special case required. A user who has put a wheel's
/// library directory on the search path while the runtime still resolves to the
/// system install takes them from two different places, and that is the failure.
///
/// A control written as an exception is a rule that has not been stated
/// correctly yet.
fn check_18_comgr_conflict(e: &Examination, symptom: &str) -> Diagnosis {
    const ID: &str = "fix-18-comgr-conflict";
    const TITLE: &str = "code object manager library does not belong to the active HIP runtime";

    let (Some(comgr), Some(hip)) = (e.comgr_selected.as_ref(), e.hip_selected.as_ref()) else {
        // Nothing to compare. A machine with no ROCm at all is not a machine
        // with a conflict, and saying otherwise would be the worst kind of false
        // report: confident, and about something that is not there.
        return zero(ID, TITLE);
    };
    // The same rule `probe_comgr` uses to compute `comgr_matches_runtime`. Calling
    // it here rather than restating the conditions is what keeps `rocm examine
    // --json` and this finding from disagreeing about one machine: an
    // unattributed copy, a runtime with no code object manager of its own to
    // prefer, or the two already agreeing are all `None`/`Some(true)` here and
    // "nothing to report" below, exactly as they are for that field.
    if crate::examine::comgr_matches_runtime(&e.comgr_paths, Some(comgr), Some(hip)) != Some(false)
    {
        return zero(ID, TITLE);
    }
    let Some(matching) = e
        .comgr_paths
        .iter()
        .find(|copy| copy.install_root == hip.install_root)
    else {
        // Unreachable: `comgr_matches_runtime` only returns `Some(false)` when
        // this search succeeds. Kept as a guard rather than an `unwrap` so a
        // future change to either function fails safely instead of panicking.
        return zero(ID, TITLE);
    };

    let mut score = 60;
    let mut evidence = vec![
        format!(
            "the code object manager that would load is {} (from {})",
            comgr.path, comgr.install_root
        ),
        format!(
            "the HIP runtime that would load is {} (from {})",
            hip.path, hip.install_root
        ),
        format!(
            "{} also ships a code object manager at {}",
            hip.install_root, matching.path
        ),
    ];
    if !comgr.version.is_empty() && !matching.version.is_empty() {
        evidence.push(format!(
            "versions differ: {} would load, the runtime ships {}",
            comgr.version, matching.version
        ));
    }
    // Said out loud because the answer depends on which process asks, and
    // because that answer changes what the `export` below is worth.
    // `find_library_copies` checks `active-runtime` directories first -- that
    // is a served child's search order (`rocm serve`/`rocm chat`), not a
    // plain shell's. When neither selection came from there, the measurement
    // really is the plain shell the user's own command ran in, and the
    // `export` is the whole fix. When either did, that half of the
    // measurement reflects what the active managed runtime (and anything it
    // serves) loads, not a plain shell -- and a shell `export` is not
    // guaranteed to reach a served child that prepends its own runtime's
    // directories ahead of `LD_LIBRARY_PATH`.
    let active_runtime_involved =
        comgr.source == "active-runtime" || hip.source == "active-runtime";
    evidence.push(if active_runtime_involved {
        "an active managed runtime decided part of this: rocm serve/rocm chat would see this \
         result, a plain shell might not"
            .to_owned()
    } else {
        "this describes the environment as it stands outside the CLI's managed runtimes".to_owned()
    });

    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_COMGR_CONFLICT);
    score += kw_score;
    evidence.extend(kw_ev);

    let mut notes = vec![crate::fix::COMGR_CONFLICT_NEITHER_OPTION_RECOMMENDED.to_owned()];
    if active_runtime_involved {
        notes.push(
            "rocm serve/rocm chat prepend the active managed runtime's own directories ahead \
             of LD_LIBRARY_PATH, so the export below may not change what they load; \
             reinstalling or repairing that runtime so it ships its own code object manager is \
             the fix that reaches it directly."
                .to_owned(),
        );
    }

    let fix = Fix {
        summary:
            "Make the code object manager and the HIP runtime come from the same installation."
                .to_owned(),
        commands: vec![
            format!("# The library that would load: {}", comgr.real_path),
            format!("# The runtime that would load: {}", hip.real_path),
            format!("# The runtime's own copy:      {}", matching.real_path),
            "# Either keep one stack and remove the other, or order the search".to_owned(),
            "# path so the runtime's own copy is found first:".to_owned(),
            format!(
                "export LD_LIBRARY_PATH=\"{}:$LD_LIBRARY_PATH\"",
                std::path::Path::new(&matching.real_path)
                    .parent()
                    .map_or_else(String::new, |dir| dir.to_string_lossy().into_owned())
            ),
        ],
        fix_id: ID.to_owned(),
        verify: "python -c \"import torch; torch.zeros(1, device='cuda')\"".to_owned(),
        notes,
        ..Fix::default()
    };
    finalize(ID, TITLE, score, evidence, fix)
}

/// A canonical engine-startup import-failure `--symptom` string guaranteed to
/// clear [`MIN_SCORE_FOR_MATCH`].
///
/// `libtorch_cuda.so` is the one token in that traceback that cannot mean
/// anything else (a ROCm build of torch ships `libtorch_hip.so` and never that
/// file), and it is worth exactly the threshold on its own on purpose; see
/// [`KEYWORDS_TORCH_DLPACK_CUDA_VARIANT`]. The vLLM engine falls back to this when
/// the user's actual failing line would not itself be diagnosable, so the
/// `rocm diagnose` command it prints always reports a cause.
pub const VLLM_IMPORT_CANONICAL_SYMPTOM: &str =
    "OSError: libtorch_cuda.so: cannot open shared object file";

/// Whether `symptom` clears [`MIN_SCORE_FOR_MATCH`] for the engine-startup
/// import failure.
///
/// The vLLM engine calls this to decide whether the user's actual failing line
/// is worth routing into a `--symptom` example, or whether to fall back to
/// [`VLLM_IMPORT_CANONICAL_SYMPTOM`] so the printed command is guaranteed to
/// report a cause rather than reading as "the tool checked and found nothing".
///
/// The hint fires on every host the vLLM engine runs on, which is Linux and
/// WSL2. The symptom therefore has to clear the threshold on each of those
/// families, run through the same registry `rocm diagnose` uses, so a check
/// that is registered for one family and not the other cannot make the engine
/// name a command that finds nothing.
#[must_use]
pub fn vllm_import_symptom_is_diagnosable(symptom: &str) -> bool {
    scores_on_every_engine_host(symptom, "fix-17-torch-dlpack", run_all_checks)
}

/// Whether the entry `id` clears [`MIN_SCORE_FOR_MATCH`] for `symptom` on every
/// host family the vLLM engine runs on (Linux and WSL2). `run` is the registry
/// run for one host. It is a parameter so a test can hand in a registry where
/// the entry is missing on one family.
fn scores_on_every_engine_host(
    symptom: &str,
    id: &str,
    run: impl Fn(&Examination, &str) -> Vec<Diagnosis>,
) -> bool {
    let linux = Examination {
        os_family: "linux".to_owned(),
        ..Examination::default()
    };
    let wsl = Examination {
        is_wsl: true,
        ..linux.clone()
    };
    [linux, wsl].iter().all(|host| {
        run(host, symptom)
            .iter()
            .any(|d| d.id == id && d.score >= MIN_SCORE_FOR_MATCH)
    })
}

/// The vLLM engine-startup import failure (EAI-8012).
///
/// Keyword-only, and not by preference. The fact that decides this failure is
/// the torch version inside the *managed runtime*, and `Examination`'s framework
/// probe imports torch from the ambient interpreter instead — on an affected
/// host it reports `framework: unknown` with a `torch import failed` note, so
/// the deciding fact is absent from the examination entirely. Nothing structural
/// can fire until that probe targets the active runtime's interpreter, which is
/// why the parameter is unused: the user has to supply the error text through
/// `rocm diagnose --symptom "<pasted error>"`.
///
/// That is the only route to this entry from the symptom. The vLLM engine's own
/// startup-failure message now points at it (`import_failure_hint` in
/// `engines/vllm`, which prints `rocm diagnose --symptom '<the error line>'`),
/// but only for that engine. A service that died at startup for any other reason
/// still renders a `logs:` and a `restart:` hint and no more, because adding a
/// diagnose hint there changes a shared surface for every failed service
/// regardless of cause and belongs in its own change. The recipe is also
/// reachable by name (`rocm fix` lists it, `rocm fix fix-17-torch-dlpack` prints
/// it), but that needs the id rather than the symptom.
fn check_17_torch_dlpack_cuda_variant(_e: &Examination, symptom: &str) -> Diagnosis {
    let (score, evidence) = keyword_score(symptom, KEYWORDS_TORCH_DLPACK_CUDA_VARIANT);
    if score <= 0 {
        return zero(
            "fix-17-torch-dlpack",
            "torch-c-dlpack-ext loads its CUDA variant on ROCm",
        );
    }
    let fix = Fix {
        summary: "Only if the engine's runtime holds a ROCm build of torch in the 2.4-2.9 range with torch-c-dlpack-ext installed: reinstall the engine so its pinned torch is restored, which moves torch off the versions the extension ships prebuilts for.".to_owned(),
        // Three labelled groups, because the steps run in three different places
        // and the report renders them as one undifferentiated `$`-prefixed list.
        // Unlabelled, a user pasting the block wholesale is relying on terminal
        // stdin buffering to land the probes in the subshell -- and on the
        // reinstall NOT landing there, since it replaces the very environment
        // that shell is standing in.
        commands: vec![
            "# --- step 1 of 3, in YOUR shell ---".to_owned(),
            "# Opens an INTERACTIVE subshell with the engine's environment active,".to_owned(),
            "# and does not return until you leave it. Run this line on its own.".to_owned(),
            "rocm engines shell vllm".to_owned(),
            "# --- step 2 of 3, INSIDE the subshell step 1 opened ---".to_owned(),
            "# Confirm the trigger before changing anything. It has to be the".to_owned(),
            "# ENGINE's interpreter, not the one on your PATH -- they are different".to_owned(),
            "# interpreters, and only the engine's decides this failure.".to_owned(),
            "python -c \"import torch; print(torch.__version__, torch.version.hip)\"".to_owned(),
            "python -c \"import importlib.metadata as m; print(m.version('torch-c-dlpack-ext'))\""
                .to_owned(),
            "# This entry applies ONLY when torch.version.hip is set, torch.__version__".to_owned(),
            "# is in the 2.4-2.9 range, and torch-c-dlpack-ext is installed. Outside".to_owned(),
            "# that range the extension raises a handled ImportError and this is not".to_owned(),
            "# the failure you are looking at. Then leave the subshell:".to_owned(),
            "exit".to_owned(),
            "# --- step 3 of 3, back in YOUR OWN shell ---".to_owned(),
            "# If all three held, put the engine's pinned torch back. Do NOT run".to_owned(),
            "# this from inside the subshell: it replaces the environment that".to_owned(),
            "# shell is standing in.".to_owned(),
            "rocm engines install vllm --reinstall".to_owned(),
        ],
        fix_id: "fix-17-torch-dlpack".to_owned(),
        verify: "rocm serve <model> --engine vllm   # then `rocm services list --all` and `rocm services logs <service-id>` to confirm the import no longer aborts".to_owned(),
        notes: vec![
            "Running vLLM on ROCm is not by itself a reason to apply this. The trigger is narrow: a ROCm build of torch in the 2.4-2.9 range (the versions torch-c-dlpack-ext ships prebuilts for), torch without a native __dlpack_c_exchange_api__, and torch-c-dlpack-ext present -- it arrives as a transitive dependency of tilelang, which vLLM pins.".to_owned(),
            "The usual way a runtime lands in that range is `rocm install sdk` being re-run after the engine was installed, which overwrites the engine's pinned torch. Reinstalling the engine puts the pin back.".to_owned(),
            "The defect is upstream and there is nothing to correct locally: torch-c-dlpack-ext picks its variant from torch.cuda.is_available(), which is True on ROCm because PyTorch reuses the torch.cuda namespace for HIP, and it ships no ROCm variant to pick. tvm_ffi imports it as optional but guards only ImportError/AttributeError, while ctypes.CDLL raises OSError -- so an explicitly optional import kills the process.".to_owned(),
            "A service that failed at startup is hidden from a plain `rocm services list`; pass --all to recover its id.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-17-torch-dlpack",
        "vLLM engine start aborts on torch-c-dlpack-ext loading its CUDA variant",
        score,
        evidence,
        fix,
    )
}

/// The shared-memory allowance below which a serving workload is in trouble.
///
/// Chosen to separate the failure from the healthy majority rather than to
/// describe what a workload wants. A container's default is 64 MB; an ordinary
/// Linux host gives `/dev/shm` half its RAM, clearing this on anything with 2 GB
/// or more. WSL2 ships the same 64 MB default a container does.
///
/// This deliberately under-reports. A container given 2 GB is still short for a
/// large model and will not be flagged here. That is the right way to be wrong:
/// a diagnosis that fires on healthy machines is one people stop reading, and a
/// threshold set at what a workload *wants* would trip every ordinary laptop.
const SHM_MIN_BYTES: u64 = 1024 * 1024 * 1024;

/// What the catalog already tells users to ask for. Quoted rather than restated
/// so the two cannot drift: `fix-10-container` prints `--shm-size=8g`.
const SHM_RECOMMENDED: &str = "8g";

/// `/dev/shm` too small for a serving workload.
///
/// Established from the state of the machine, not from the error text. That
/// ordering is forced by the failure itself: the crash never names shared
/// memory, which is exactly why the user cannot get from the message to the
/// cause on their own.
fn check_19_shm_too_small(e: &Examination, symptom: &str) -> Diagnosis {
    const ID: &str = "fix-19-shm-too-small";
    const TITLE: &str = "shared memory allowance too small for a serving workload";

    // Unmeasured is not short. `None` means the path was absent or the query
    // failed, and reporting a shortage on that basis would be a finding about
    // the probe rather than about the machine.
    let Some(total) = e.shm_total_bytes else {
        return zero(ID, TITLE);
    };
    if total >= SHM_MIN_BYTES {
        return zero(ID, TITLE);
    }

    let mut score = 60;
    let mut evidence = vec![format!(
        "{} is {}, and a serving workload needs gigabytes",
        crate::disk_space::format_bytes(total),
        "the whole shared-memory allowance"
    )];
    if let Some(available) = e.shm_available_bytes
        && available != total
    {
        evidence.push(format!(
            "{} of it is free",
            crate::disk_space::format_bytes(available)
        ));
    }
    if e.in_container {
        // Not more certain that the allowance is small -- that is measured. More
        // certain about the cause and the remedy, because a container's default
        // is exactly this, and restarting it with a larger one is a known step.
        score += 10;
        evidence.push(format!(
            "this is a {} container, whose default allowance is 64 MiB",
            // Matches the generic `probe_container` writes when it cannot name
            // the runtime. Saying "linux container" invented a kind the probe
            // never reports.
            if e.container_kind.is_empty() {
                "container"
            } else {
                &e.container_kind
            }
        ));
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_SHM_TOO_SMALL);
    score += kw_score;
    evidence.extend(kw_ev);

    let fix = Fix {
        summary: "Raise the shared-memory allowance before running the workload.".to_owned(),
        commands: vec![
            "# In a container: restart it with a larger allowance.".to_owned(),
            format!("#   docker run --shm-size={SHM_RECOMMENDED} ...    # see fix-10-container"),
            "# On a host: remount it, and make that survive a reboot.".to_owned(),
            format!("sudo mount -o remount,size={SHM_RECOMMENDED} /dev/shm"),
            format!("# /etc/fstab:  tmpfs  /dev/shm  tmpfs  defaults,size={SHM_RECOMMENDED}  0 0"),
        ],
        needs_sudo: true,
        fix_id: ID.to_owned(),
        verify: "df -h /dev/shm".to_owned(),
        notes: vec![
            "A running container cannot have its allowance changed; it has to be started again."
                .to_owned(),
            // The threshold under-reports on purpose, and until now that was
            // said only in the source. A reader who is told nothing reads
            // silence as a clean bill of health.
            format!(
                "This is reported below {}. Silence is not proof of enough: a container given \
                 2 GiB clears that bar and can still be too small for a large model.",
                crate::disk_space::format_bytes(SHM_MIN_BYTES)
            ),
        ],
        ..Fix::default()
    };
    finalize(ID, TITLE, score, evidence, fix)
}

/// A checker plus the OS families it applies to.
type Checker = (fn(&Examination, &str) -> Diagnosis, &'static [&'static str]);

// The `wsl` entries below run on less evidence than on bare metal: WSL collects
// no GPU topology (`gpus` is empty, because the probes that fill it read KFD and
// DRM), so the parts of those checks that compare a detected gfx target against a
// wheel's arch list cannot contribute. They still fire on the evidence WSL does
// have -- environment, ROCm install, symptom keywords -- which is strictly better
// than the previous behaviour of not running at all.
//
// The degradation is one-directional and must stay that way: less evidence means
// a check may miss a real fault, never that it invents one. Pinned by
// `the_shared_checks_under_report_on_wsl_rather_than_over_report`.
const CHECKERS: &[Checker] = &[
    (check_1_arch_not_in_wheel, &["linux", "windows", "wsl"]),
    (check_2_hsa_override_unneeded, &["linux", "windows", "wsl"]),
    (check_3_rocm_kernel_unsupported, &["linux"]),
    (check_4_render_group, &["linux"]),
    (check_5_amdgpu_blacklisted, &["linux"]),
    (check_6_path_missing, &["linux", "windows", "wsl"]),
    (check_7_stale_repos, &["linux"]),
    (check_8_wheel_rocm_mismatch, &["linux", "windows", "wsl"]),
    // Not "wsl": WSL2 exposes no per-device topology to collide over.
    (check_9_igpu_dgpu_collision, &["linux", "windows"]),
    (check_10_container_devices, &["linux"]),
    (check_11_iommu_hang, &["linux"]),
    (check_12_repo_native_broken, &["linux"]),
    (check_13_hip_sdk_missing, &["windows"]),
    (check_14_adrenalin_too_old, &["windows"]),
    (check_15_msvc_redist, &["windows"]),
    (check_16_vllm_oom, &["linux", "wsl"]),
    // Linux and WSL2: `libtorch_cuda.so` is an ELF name, and the vLLM engine is
    // gated off native Windows but runs on WSL2. The engine's startup-failure
    // hint fires on both, so the entry it points at must be reachable on both.
    // 17 rather than 16 because the vLLM out-of-memory entry above already
    // holds 16; the number is a stable handle, so the two do not get to share
    // one.
    (check_17_torch_dlpack_cuda_variant, &["linux", "wsl"]),
    (check_wsl_1_gpu_not_exposed, WSL_ONLY),
    (check_wsl_2_dxcore_missing, WSL_ONLY),
    (check_wsl_3_rocdxg_missing, WSL_ONLY),
    (check_wsl_4_rocdxg_not_linked, WSL_ONLY),
    (check_wsl_5_distro_too_old, WSL_ONLY),
    (check_wsl_6_host_driver_too_old, WSL_ONLY),
    (check_wsl_7_wsl1, WSL_ONLY),
    // Both families, opting in explicitly as the platform split requires. Which
    // copy of a library the loader picks is not a question about the amdgpu
    // module or the Windows host driver -- a wheel copy and a system copy
    // collide on WSL2 exactly as they do on bare metal. Windows is deferred
    // until this has proved the approach.
    (check_18_comgr_conflict, &["linux", "wsl"]),
    // Both families, for the same reason. The shortage has nothing to do with
    // `amdgpu` or `/dev/kfd` -- it is the size of a tmpfs -- and WSL2 ships the
    // same 64 MB default a container does, so leaving this tagged `linux` alone
    // would silence it on one of the two platforms most likely to have it.
    (check_19_shm_too_small, &["linux", "wsl"]),
    // Not "wsl": WSL2 reports no GPU topology (`gpus` is empty), so this check
    // has nothing to read there and would never fire.
    (check_20_gpu_unsupported_by_rocm, &["linux", "windows"]),
    // Windows only: the runtime is read from the Windows System32 folder. A WSL
    // distro is examined as a distro, so it has no such runtime to read.
    (check_21_msvc_redist_outdated, &["windows"]),
    // Windows only: the driver date comes from the Windows display adapter.
    // WSL2 reads the host driver through its own entries.
    (check_22_display_driver_old, &["windows"]),
];

const WSL_ONLY: &[&str] = &["wsl"];

/// The platform family a catalog entry is selected by.
///
/// This is deliberately NOT `Examination::os_family`. WSL2 reports an `os_family`
/// of `linux` and must keep doing so — `install`, `serve` and the engine crates
/// branch on it — but it is a different platform for diagnosis: it reaches the
/// GPU through `/dev/dxg` and the Windows host driver, with no `amdgpu` module
/// and no `/dev/kfd`.
///
/// Resolving the family here rather than in each check is what makes the split
/// safe. Every bare-metal entry is tagged `linux` only, so it stops applying on
/// WSL automatically; an entry that genuinely applies to both opts in by naming
/// `wsl` as well. That preserves the property the old wholesale WSL short-circuit
/// bought — no `fix-4-render-group` on a healthy WSL box — without also
/// suppressing the checks that were always valid there.
const fn platform_family(e: &Examination) -> &str {
    if e.is_wsl {
        return "wsl";
    }
    if e.os_family.is_empty() {
        return "linux";
    }
    e.os_family.as_str()
}

/// Run every applicable checker, drop zero-score results, sort by score
/// descending (stable, so ties keep catalog order).
fn run_all_checks(e: &Examination, symptom: &str) -> Vec<Diagnosis> {
    let family = platform_family(e);
    let mut results: Vec<Diagnosis> = CHECKERS
        .iter()
        .filter(|(_, applicable)| applicable.contains(&family))
        .map(|(check, _)| check(e, symptom))
        .filter(|d| d.score > 0)
        .collect();
    // Stable sort by score descending: ties keep catalog order.
    results.sort_by_key(|d| std::cmp::Reverse(d.score));
    results
}

/// Whether any catalog entry at all applies to this platform.
fn catalog_covers(e: &Examination) -> bool {
    let family = platform_family(e);
    CHECKERS
        .iter()
        .any(|(_, applicable)| applicable.contains(&family))
}

/// Diagnose an examination against the closed catalog.
#[must_use]
pub fn diagnose(e: &Examination, symptom: &str) -> DiagnoseReport {
    // WSL2 used to be short-circuited here as out of scope. It is a real platform
    // in the catalog now, with its own entries; what keeps the bare-metal checks
    // off it is `platform_family`, not a special case at this level.
    //
    // `fix-16-vllm-oom` and `fix-17-torch-dlpack` opt into `wsl` in CHECKERS for
    // that reason: they are keyword-only checks whose failure modes are just as
    // real under WSL2, so they still answer there alongside the WSL entries.
    //
    // `out_of_scope` still exists, for the platforms that genuinely have no
    // entries. That case used to fall through to an empty catalog and report "no
    // known misconfiguration", which reads as "your machine looks fine" when the
    // truth is that nothing was ever checked.
    let out_of_scope = uncovered_platform_message(e);
    let mut matched = if out_of_scope.is_some() {
        Vec::new()
    } else {
        run_all_checks(e, symptom)
    };
    take_applicability_from_the_catalog(&mut matched, platform_family(e));
    DiagnoseReport {
        has_match: any_cleared_threshold(&matched),
        matched,
        min_score_for_match: MIN_SCORE_FOR_MATCH,
        high_confidence_threshold: HIGH_CONFIDENCE,
        route_when_no_match: route_when_no_match(e),
        out_of_scope,
        model: None,
    }
}

/// Fill each finding's applicability from the catalog, for the operating system
/// of the machine that was examined.
///
/// One place, deliberately. Every checker used to state `auto_applicable` by
/// hand, which meant the catalog's answer existed twice in two modules with
/// nothing comparing them — and they had already drifted: `fix-9-igpu-dgpu` was
/// auto-applicable in the catalog and, in the Linux arm of its own checker,
/// not. Deriving it here makes that disagreement unrepresentable rather than
/// something a test has to go looking for.
///
/// Keyed on the examined machine's platform *family*, not the running host and
/// not `os_family`. WSL reports an `os_family` of `linux` but is its own family
/// for catalog purposes, so reading `os_family` here would look up every WSL
/// recipe under the wrong platform and silently report them all as print-only.
fn take_applicability_from_the_catalog(matched: &mut [Diagnosis], platform_family: &str) {
    for diagnosis in matched.iter_mut() {
        let Some(fix) = diagnosis.fix.as_mut() else {
            continue;
        };
        // An entry the catalog does not list for this OS keeps the default:
        // the CLI will not act on it here, which is exactly what the OS gate in
        // `fix::apply` would tell the user.
        let class = crate::fix::class_on(&fix.fix_id, platform_family).unwrap_or_default();
        fix.class = class;
        fix.auto_applicable = class.applies_itself();
        fix.needs = crate::fix::needs_on(&fix.fix_id, platform_family).map(ToOwned::to_owned);
    }
}

/// Say so when the catalog has no entries for the running platform.
///
/// Returning `None` here means the platform is covered, not that it is healthy.
fn uncovered_platform_message(e: &Examination) -> Option<String> {
    if catalog_covers(e) {
        return None;
    }
    Some(format!(
        "rocm diagnose covers Linux, Windows and WSL2. This host reports '{}', which the \
         catalog has no entries for, so nothing was checked -- this is not a clean bill of \
         health. Run `rocm examine --json` and report the platform upstream.",
        e.os_family
    ))
}

#[cfg(test)]
mod tests {
    use super::test_support::{igpu_dgpu_host, linux_base, windows_driver_exam, wsl_base};
    use super::windows::DISPLAY_DRIVER_OLD_SCORE;
    use super::*;
    use crate::examine::Gpu;

    /// `take_applicability_from_the_catalog` is the only place that may fill in
    /// `Fix::auto_applicable`. Every checker used to set it by hand instead --
    /// twenty of them were converted to leave it at `Fix::default()` and let the
    /// catalog fill it in, but the shared-memory and WSL checkers kept the
    /// inline `auto_applicable: false,` a while longer. It never produced wrong
    /// output, because the catalog pass overwrites whatever a checker sets --
    /// which is exactly the problem: a checker disagreeing with the catalog
    /// again (the way `fix-9-igpu-dgpu` once did) would be silently masked
    /// rather than caught. A behavioral test can't see this, since the output
    /// is correct either way; this reads the checkers' own source instead, so a
    /// hand-set boolean is caught before it can start drifting unnoticed.
    #[test]
    fn no_checker_hand_sets_auto_applicable() {
        // The checkers live in this file and in the modules under `diagnose/`.
        // Every file that holds one is listed here: a checker that moves into a
        // file missing from this list would leave the guard passing vacuously.
        let sources = [
            include_str!("diagnose.rs"),
            include_str!("diagnose/host.rs"),
            include_str!("diagnose/linux.rs"),
            include_str!("diagnose/report.rs"),
            include_str!("diagnose/scoring.rs"),
            include_str!("diagnose/windows.rs"),
            include_str!("diagnose/wsl.rs"),
        ];
        let offenders: Vec<&str> = sources
            .into_iter()
            .flat_map(|source| source.lines())
            .filter(|line| {
                let trimmed = line.trim_start();
                trimmed.starts_with("auto_applicable: true")
                    || trimmed.starts_with("auto_applicable: false")
            })
            .collect();
        assert!(
            offenders.is_empty(),
            "a Fix literal is hand-setting `auto_applicable` again -- only \
             `take_applicability_from_the_catalog` may set this field; every checker must \
             leave it at `Fix::default()` and let the catalog fill it in, or the catalog and \
             the checker can silently drift the way fix-9-igpu-dgpu once did:\n{offenders:#?}"
        );
    }

    /// A pasted HIP-loading error still opens fix-13 when no AMD GPU was
    /// detected at all — deliberately, and this is the test that says so.
    ///
    /// `symptom_names_hip_loading` is the one of `needs_hip`'s three disjuncts
    /// not gated on `e.has_amd_gpu`, and the asymmetry is the point. The other
    /// two read a *probe* ("PyTorch here is a HIP build"), which is only
    /// meaningful beside a GPU. This one reads a *fact the user supplied*: a
    /// process on this machine tried to load HIP and failed. A GPU probe
    /// answering "nothing detected" is itself one of the things a broken HIP
    /// install causes, so letting it veto the user's own error would silence
    /// the entry on the hosts most likely to need it — a headless box whose
    /// driver did not load, or a WSL guest where the probe sees no card.
    /// Installing the HIP SDK is the right next step either way; if there
    /// genuinely is no AMD GPU, fix-13 ranks beside the other causes rather
    /// than suppressing them.
    #[test]
    fn fix_13_still_answers_a_pasted_hip_error_when_no_amd_gpu_was_detected() {
        let no_gpu = Examination {
            os_family: "windows".to_owned(),
            // The probe found nothing: no `has_amd_gpu`, no `gpus` entries.
            ..Examination::default()
        };
        let finding = check_13_hip_sdk_missing(&no_gpu, "amdhip64.dll was not found");
        assert_eq!(
            finding.score, 85,
            "the same score as the host with a detected GPU: the pasted error is the \
             evidence, and an absent-GPU probe does not weaken it: {finding:?}"
        );
        assert!(
            finding
                .evidence
                .iter()
                .any(|ev| ev.contains("No HIP SDK install found")),
            "and the SDK's absence is still named as the thing to put right: {finding:?}"
        );
        // Nothing here is an unsupported-GPU host, so the entry that outranks
        // fix-13 on the ticket's laptop must stay silent rather than firing on
        // a machine whose hardware was never identified.
        assert_eq!(
            check_20_gpu_unsupported_by_rocm(&no_gpu, "").score,
            0,
            "an undetected GPU is not a confirmed-unsupported one"
        );
    }

    fn shm_finding(report: &DiagnoseReport) -> Option<&Diagnosis> {
        report
            .matched
            .iter()
            .find(|d| d.id == "fix-19-shm-too-small")
    }

    #[test]
    fn a_tiny_shared_memory_allowance_is_reported_before_the_workload_runs() {
        // The container default, which is also what WSL2 ships. The point of the
        // entry is that it fires on machine state alone: the crash this prevents
        // never names shared memory, so a user who pasted it would get nothing.
        let mut e = linux_base();
        e.shm_total_bytes = Some(64 * 1024 * 1024);
        e.shm_available_bytes = Some(64 * 1024 * 1024);

        let report = diagnose(&e, "");
        let finding = shm_finding(&report).expect("a 64 MiB allowance must be reported");

        assert!(
            finding.score >= MIN_SCORE_FOR_MATCH,
            "the shortage is established by measurement, so it has to clear the \
             threshold with no help from the symptom text: {}",
            finding.score
        );
        let evidence = finding.evidence.join("\n");
        assert!(
            evidence.contains("64"),
            "the report has to state what is actually available:\n{evidence}"
        );
        let fix = finding.fix.as_ref().expect("the finding must carry a plan");
        assert!(
            !fix.auto_applicable,
            "raising the allowance means restarting a container or remounting; \
             neither is something to do on the user's behalf"
        );
        assert!(
            fix.commands.iter().any(|c| c.contains("8g")),
            "the steps have to name a size to raise it to:\n{:#?}",
            fix.commands
        );
    }

    #[test]
    fn the_threshold_is_pinned_at_its_edge_rather_than_somewhere_between() {
        // 64 MiB and 8 GiB leave everything in between unpinned: a threshold
        // silently moved to 512 MiB or 2 GiB would pass both. The boundary is
        // the only value worth asserting, because it is the only one a change
        // to the constant has to cross.
        let at_threshold = shm_report(SHM_MIN_BYTES);
        assert!(
            shm_finding(&at_threshold).is_none(),
            "exactly the threshold is enough; the rule is 'below', not 'at or below'"
        );
        let just_under = shm_report(SHM_MIN_BYTES - 1);
        assert!(
            shm_finding(&just_under).is_some(),
            "one byte under the threshold has to report, or the constant means nothing"
        );

        // Both assertions above are written in terms of the constant, so they
        // move with it: they pin the rule ("below", not "at or below") and say
        // nothing about the value. The value is a promise now -- the fix notes
        // tell the user the figure -- so pin it literally. This is meant to fail
        // when someone changes it, which makes the change deliberate and forces
        // the note to move with it.
        assert_eq!(
            SHM_MIN_BYTES,
            1024 * 1024 * 1024,
            "the threshold is published to users in the fix notes; changing it means \
             changing what they were told, so update both together"
        );
        // And a value from the middle of the range, which neither 64 MiB nor
        // 8 GiB constrains: an ordinary host with 4 GB of RAM gets 2 GiB here
        // and must stay silent.
        assert!(
            shm_finding(&shm_report(2 * 1024 * 1024 * 1024)).is_none(),
            "2 GiB is what a modest but healthy host provides"
        );
    }

    #[test]
    fn the_symptom_text_raises_the_finding_without_being_needed_for_it() {
        // Both halves matter. The keywords must earn their place, and they must
        // not be load-bearing: the whole premise is that the crash never names
        // shared memory, so a user who pastes an unrelated error still gets the
        // finding.
        let machine_only = shm_finding(&shm_report(64 * 1024 * 1024))
            .expect("machine state alone reports")
            .score;

        for symptom in [
            "RuntimeError: DataLoader worker (pid 12) is killed by signal: Bus error",
            "OSError: [Errno 28] No space left on device: '/dev/shm/torch_abc'",
            "failed to allocate shared memory",
        ] {
            let mut e = linux_base();
            e.shm_total_bytes = Some(64 * 1024 * 1024);
            let report = diagnose(&e, symptom);
            let scored = shm_finding(&report)
                .expect("still reports with a symptom")
                .score;
            assert!(
                scored > machine_only,
                "{symptom:?} should raise the finding above the machine-state score \
                 ({scored} vs {machine_only})"
            );
        }
    }

    /// A library copy belonging to `install_root`, sitting at `path`.
    fn copy_in(install_root: &str, path: &str, version: &str) -> crate::examine::LibraryCopy {
        crate::examine::LibraryCopy {
            path: path.to_owned(),
            real_path: path.to_owned(),
            version: version.to_owned(),
            source: "test".to_owned(),
            install_root: install_root.to_owned(),
        }
    }

    /// Like `copy_in`, but with an explicit loader-search source instead of
    /// the default `"test"` -- needed to exercise the branches in fix-18 that
    /// key off which tier a selection came from.
    fn copy_in_with_source(
        source: &str,
        install_root: &str,
        path: &str,
        version: &str,
    ) -> crate::examine::LibraryCopy {
        crate::examine::LibraryCopy {
            source: source.to_owned(),
            ..copy_in(install_root, path, version)
        }
    }

    fn comgr_finding(report: &DiagnoseReport) -> Option<&Diagnosis> {
        report
            .matched
            .iter()
            .find(|d| d.id == "fix-18-comgr-conflict")
    }

    #[test]
    fn a_code_object_manager_from_another_installation_is_reported() {
        // The failure this entry exists for: the runtime resolves to the system
        // install while the library that compiles device code for it comes from
        // a wheel, and the error the user sees names neither.
        let mut e = linux_base();
        let wheel = copy_in("/wheel", "/wheel/lib/libamd_comgr.so.2", "2.8.0");
        let system = copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3.0.0");
        e.comgr_selected = Some(wheel.clone());
        e.comgr_paths = vec![wheel, system];
        e.hip_selected = Some(copy_in(
            "/opt/rocm",
            "/opt/rocm/lib/libamdhip64.so.6",
            "6.4",
        ));

        let report = diagnose(&e, "");
        let finding = comgr_finding(&report).expect("the mismatch must be reported");

        assert!(
            finding.score >= MIN_SCORE_FOR_MATCH,
            "a mismatch established from machine state alone has to clear the \
             threshold without help from the symptom text: {}",
            finding.score
        );
        let fix = finding.fix.as_ref().expect("the finding must carry a plan");
        assert!(
            !fix.auto_applicable,
            "both remedies can break a working Python environment, so nothing here \
             may be applied for the user"
        );
        let evidence = finding.evidence.join("\n");
        for expected in [
            "/wheel/lib/libamd_comgr.so.2",
            "/opt/rocm",
            "2.8.0",
            "3.0.0",
        ] {
            assert!(
                evidence.contains(expected),
                "the report has to name each copy, where it came from and its \
                 version -- `{expected}` is missing:\n{evidence}"
            );
        }
    }

    #[test]
    fn a_partly_used_allowance_reports_what_is_left_as_well_as_the_size() {
        // The two numbers answer different questions, and the evidence only
        // mentions what is free when it differs from the total -- otherwise it
        // would repeat itself on an idle machine.
        let mut e = linux_base();
        e.shm_total_bytes = Some(64 * 1024 * 1024);
        e.shm_available_bytes = Some(2 * 1024 * 1024);

        let report = diagnose(&e, "");
        let evidence = shm_finding(&report)
            .expect("a short allowance reports")
            .evidence
            .join("\n");
        assert!(
            evidence.contains("2.0 MiB"),
            "a mostly-full allowance has to say how little is left:\n{evidence}"
        );
    }

    /// A Linux machine whose only notable property is its shared-memory size.
    fn shm_report(total: u64) -> DiagnoseReport {
        let mut e = linux_base();
        e.shm_total_bytes = Some(total);
        e.shm_available_bytes = Some(total);
        diagnose(&e, "")
    }

    #[test]
    fn an_ordinary_shared_memory_allowance_is_not_reported() {
        // The control. An ordinary Linux host gives /dev/shm half its RAM, so
        // this is the common case -- and an entry that fired here would fire on
        // nearly every machine, which is how a catalog stops being read.
        let mut e = linux_base();
        e.shm_total_bytes = Some(8 * 1024 * 1024 * 1024);
        e.shm_available_bytes = Some(8 * 1024 * 1024 * 1024);

        assert!(
            shm_finding(&diagnose(&e, "")).is_none(),
            "8 GiB is what the catalog itself tells users to ask for"
        );
    }

    #[test]
    fn a_healthy_managed_installation_raises_no_report() {
        // The control, and the reason the rule is stated symmetrically. A managed
        // runtime spreads its libraries across separate `_rocm_sdk_*` packages
        // that sit *beside* each other under one `site-packages` -- not nested
        // under the runtime's own root (`_rocm_sdk_devel`) at all -- so the two
        // copies sit in different directories of ONE installation. Anything
        // that decided ownership by walking up from the file would call these
        // two installations and fire on the most common install we ship. (A
        // fixture nesting `_rocm_sdk_core` under the root, the shape the
        // installer never produces, would pass by construction without
        // proving that -- this one matches `apps/rocm/src/therock/probe.rs`'s
        // `ROCM_SDK_PROBE_SCRIPT` instead.)
        let mut e = linux_base();
        let root = "/data/runtimes/therock/_rocm_sdk_devel";
        let comgr = copy_in(
            root,
            "/data/runtimes/therock/_rocm_sdk_core/lib/libamd_comgr.so.2",
            "2.8.0",
        );
        e.comgr_selected = Some(comgr.clone());
        e.comgr_paths = vec![comgr];
        e.hip_selected = Some(copy_in(
            root,
            "/data/runtimes/therock/_rocm_sdk_devel/lib/libamdhip64.so.6",
            "6.4",
        ));

        assert!(
            comgr_finding(&diagnose(&e, "")).is_none(),
            "both libraries come from one installation, so there is no conflict"
        );
    }

    #[test]
    fn an_unmeasured_shared_memory_allowance_is_not_reported() {
        // Unknown is not zero. `None` means the path was absent or the query
        // failed, and a shortage reported on that basis would be a finding about
        // the probe rather than about the user's machine.
        let mut e = linux_base();
        e.shm_total_bytes = None;
        e.shm_available_bytes = None;

        assert!(
            shm_finding(&diagnose(&e, "")).is_none(),
            "a machine that could not be measured is not a machine with a shortage"
        );
    }

    #[test]
    fn a_second_copy_that_does_not_load_raises_no_report() {
        // Holding two copies is not a fault. Only the one that loads matters.
        let mut e = linux_base();
        let winner = copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3.0.0");
        let loser = copy_in("/wheel", "/wheel/lib/libamd_comgr.so.2", "2.8.0");
        e.comgr_selected = Some(winner.clone());
        e.comgr_paths = vec![winner, loser];
        e.hip_selected = Some(copy_in(
            "/opt/rocm",
            "/opt/rocm/lib/libamdhip64.so.6",
            "6.4",
        ));

        assert!(
            comgr_finding(&diagnose(&e, "")).is_none(),
            "the copy that loads belongs to the runtime, so nothing is wrong"
        );
        assert_eq!(
            e.comgr_paths.len(),
            2,
            "the machine report still lists both copies; only the diagnosis stays quiet"
        );
    }

    #[test]
    fn being_in_a_container_raises_the_finding_without_creating_it() {
        // The container flag says the cause and the remedy are known exactly, so
        // it raises confidence. It must not be able to conjure a finding on a
        // machine whose allowance is fine.
        let mut healthy = linux_base();
        healthy.shm_total_bytes = Some(8 * 1024 * 1024 * 1024);
        healthy.in_container = true;
        healthy.container_kind = "docker".to_owned();
        assert!(
            shm_finding(&diagnose(&healthy, "")).is_none(),
            "a container with a healthy allowance has nothing wrong with it"
        );

        let mut short = linux_base();
        short.shm_total_bytes = Some(64 * 1024 * 1024);
        let outside = shm_finding(&diagnose(&short, ""))
            .expect("short allowance reports")
            .score;
        short.in_container = true;
        short.container_kind = "docker".to_owned();
        let inside = shm_finding(&diagnose(&short, ""))
            .expect("short allowance reports")
            .score;
        assert!(
            inside > outside,
            "knowing it is a container makes the cause certain, so it should rank \
             higher there: {inside} vs {outside}"
        );
    }

    #[test]
    fn nothing_is_reported_when_there_is_nothing_to_compare() {
        // A machine with no ROCm is not a machine with a conflict. Reporting one
        // would be the worst kind of false finding: confident, and about
        // something that is not there.
        let cases = [
            ("neither library found", None, None),
            (
                "no runtime to attribute the library to",
                Some(copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3")),
                None,
            ),
            (
                "a copy no known installation claims",
                Some(copy_in("", "/somewhere/libamd_comgr.so.3", "3")),
                Some(copy_in(
                    "/opt/rocm",
                    "/opt/rocm/lib/libamdhip64.so.6",
                    "6.4",
                )),
            ),
        ];
        for (case, comgr, hip) in cases {
            let mut e = linux_base();
            e.comgr_paths = comgr.iter().cloned().collect();
            e.comgr_selected = comgr;
            e.hip_selected = hip;
            assert!(
                comgr_finding(&diagnose(&e, "")).is_none(),
                "{case}: reported a conflict it could not have established"
            );
        }
    }

    #[test]
    fn a_runtime_with_no_copy_of_its_own_is_not_a_conflict() {
        // The second half of the rule. Without a copy belonging to the runtime
        // there is nothing to switch to, so the advice would be empty -- and the
        // machine is simply one where the only copy lives elsewhere.
        let mut e = linux_base();
        let only = copy_in("/wheel", "/wheel/lib/libamd_comgr.so.2", "2.8.0");
        e.comgr_selected = Some(only.clone());
        e.comgr_paths = vec![only];
        e.hip_selected = Some(copy_in(
            "/opt/rocm",
            "/opt/rocm/lib/libamdhip64.so.6",
            "6.4",
        ));

        assert!(
            comgr_finding(&diagnose(&e, "")).is_none(),
            "the runtime ships no code object manager of its own, so there is no \
             alternative to point the user at"
        );
    }

    #[test]
    fn an_active_runtime_s_own_hip_without_its_own_comgr_notes_the_scope_limit() {
        // Reachable, not theoretical: a managed runtime can ship its own HIP
        // runtime while leaning on the system's code object manager. Then
        // `hip_selected` comes from the `active-runtime` tier -- the order a
        // served child (`rocm serve`/`rocm chat`) consults, not a plain
        // shell's -- so the finding must say so instead of claiming the
        // plain-shell sentence, and the `export` remedy must carry a caveat
        // that it may not reach that served child.
        let mut e = linux_base();
        let system = copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3.0.0");
        let runtime_comgr = copy_in(
            "/data/runtimes/therock",
            "/data/runtimes/therock/lib/libamd_comgr.so.2",
            "2.8.0",
        );
        e.comgr_selected = Some(system.clone());
        e.comgr_paths = vec![system, runtime_comgr];
        e.hip_selected = Some(copy_in_with_source(
            "active-runtime",
            "/data/runtimes/therock",
            "/data/runtimes/therock/lib/libamdhip64.so.6",
            "6.4",
        ));

        let report = diagnose(&e, "");
        let finding = comgr_finding(&report).expect("the mismatch must be reported");
        let evidence = finding.evidence.join("\n");
        assert!(
            evidence.contains("active managed runtime decided part of this"),
            "the evidence must own up to the active-runtime tier rather than claim a \
             plain shell measured it:\n{evidence}"
        );
        assert!(
            !evidence.contains("outside the CLI's managed runtimes"),
            "the plain-shell sentence must not be printed when the measurement is \
             not a plain shell's:\n{evidence}"
        );
        let notes = finding
            .fix
            .as_ref()
            .expect("the finding must carry a plan")
            .notes
            .join("\n");
        assert!(
            notes.contains("may not change what they load"),
            "the remedy has to say it might not reach a served child:\n{notes}"
        );
    }

    #[test]
    fn comgr_matches_runtime_never_disagrees_with_the_diagnosis() {
        // `rocm examine --json`'s `comgr_matches_runtime` and `rocm diagnose`'s
        // fix-18 finding answer the same question about the same machine. Both
        // are driven by `comgr_matches_runtime` in `examine/library_copies.rs`, so this checks
        // the two surfaces stay in lockstep across every shape the catalog
        // itself tests -- including the case that used to disagree: a runtime
        // whose own installation ships no code object manager at all.
        let cases: Vec<(&str, Examination)> = vec![
            ("neither library found", linux_base()),
            {
                let mut e = linux_base();
                e.comgr_selected =
                    Some(copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3"));
                e.comgr_paths = vec![e.comgr_selected.clone().unwrap()];
                ("no runtime to attribute the library to", e)
            },
            {
                let mut e = linux_base();
                let unattributed = copy_in("", "/somewhere/libamd_comgr.so.3", "3");
                e.comgr_selected = Some(unattributed.clone());
                e.comgr_paths = vec![unattributed];
                e.hip_selected = Some(copy_in(
                    "/opt/rocm",
                    "/opt/rocm/lib/libamdhip64.so.6",
                    "6.4",
                ));
                ("a copy no known installation claims", e)
            },
            {
                // The disagreement this test guards against: the runtime's own
                // install ships no comgr copy of its own to switch to.
                let mut e = linux_base();
                let only = copy_in("/wheel", "/wheel/lib/libamd_comgr.so.2", "2.8.0");
                e.comgr_selected = Some(only.clone());
                e.comgr_paths = vec![only];
                e.hip_selected = Some(copy_in(
                    "/opt/rocm",
                    "/opt/rocm/lib/libamdhip64.so.6",
                    "6.4",
                ));
                ("a runtime with no copy of its own", e)
            },
            {
                let mut e = linux_base();
                let winner = copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3.0.0");
                let loser = copy_in("/wheel", "/wheel/lib/libamd_comgr.so.2", "2.8.0");
                e.comgr_selected = Some(winner.clone());
                e.comgr_paths = vec![winner, loser];
                e.hip_selected = Some(copy_in(
                    "/opt/rocm",
                    "/opt/rocm/lib/libamdhip64.so.6",
                    "6.4",
                ));
                ("the copy that loads already belongs to the runtime", e)
            },
            {
                let mut e = linux_base();
                let wheel = copy_in("/wheel", "/wheel/lib/libamd_comgr.so.2", "2.8.0");
                let system = copy_in("/opt/rocm", "/opt/rocm/lib/libamd_comgr.so.3", "3.0.0");
                e.comgr_selected = Some(wheel.clone());
                e.comgr_paths = vec![wheel, system];
                e.hip_selected = Some(copy_in(
                    "/opt/rocm",
                    "/opt/rocm/lib/libamdhip64.so.6",
                    "6.4",
                ));
                ("a genuine conflict", e)
            },
        ];

        for (case, e) in cases {
            let matches_runtime = crate::examine::comgr_matches_runtime(
                &e.comgr_paths,
                e.comgr_selected.as_ref(),
                e.hip_selected.as_ref(),
            );
            let diagnosis_fires = comgr_finding(&diagnose(&e, "")).is_some();
            assert_eq!(
                matches_runtime == Some(false),
                diagnosis_fires,
                "{case}: comgr_matches_runtime={matches_runtime:?} but the \
                 diagnosis {}",
                if diagnosis_fires {
                    "fired"
                } else {
                    "did not fire"
                }
            );
        }
    }

    #[test]
    fn the_engine_import_failure_outranks_the_render_group_false_lead() {
        // The reported case, reduced to what makes the false lead fire: the user
        // is outside the render and video groups, so the catalog's best answer
        // to this symptom was fix-4 at 45 (35 render + 10 video), and following
        // it meant a usermod, a re-login, and no progress. The fixture leaves
        // `kfd` unset rather than modelling the reported host's permissions,
        // because fix-4's score here comes from the group membership alone.
        // The fixture reproduces that false lead, so the assertion is about the
        // ranking and not only about the new entry's score.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);

        let report = diagnose(
            &e,
            "vllm engine fails to start: OSError: libtorch_cuda.so: cannot open shared object file",
        );
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-17-torch-dlpack");
        assert!(top.score >= MIN_SCORE_FOR_MATCH, "score was {}", top.score);
        assert!(report.has_match());

        let render_group = report
            .matched
            .iter()
            .position(|d| d.id == "fix-4-render-group")
            .expect("the fixture must still produce the false lead this outranks");
        assert!(
            render_group > 0,
            "the render-group suggestion must no longer be the top answer here"
        );
    }

    #[test]
    fn the_engine_import_plan_says_which_shell_each_step_runs_in() {
        // The plan `diagnose` attaches to the finding is a second copy of the
        // catalog recipe's command block, and it is the copy the reported user
        // actually saw. Hold it to the same boundary the recipe is held to, so
        // the two cannot drift into disagreeing about which shell runs what.
        let report = diagnose(
            &linux_base(),
            "OSError: libtorch_cuda.so: cannot open shared object file",
        );
        let fix = report
            .matched
            .iter()
            .find(|d| d.id == "fix-17-torch-dlpack")
            .and_then(|d| d.fix.as_ref())
            .expect("the finding must carry a plan");
        let commands: Vec<&str> = fix.commands.iter().map(String::as_str).collect();
        crate::fix::assert_engine_shell_boundary_is_labelled(&fix.fix_id, &commands);
        // The boundary check alone leaves the wording free to drift, so pin the
        // two copies to each other line for line as well.
        crate::fix::assert_plan_matches_the_catalog_copy(&fix.fix_id, &commands);
    }

    #[test]
    fn the_oom_plan_matches_the_catalog_copy() {
        // Same two-copies-of-one-plan risk as `fix-17-torch-dlpack`, and it had
        // already opened: the executable lines agreed, so the cross-check over
        // the fixes shared with the catalog
        // (`diagnosis_remediation_matches_the_fix_catalog_for_shared_fix_ids`)
        // saw nothing -- it filters `#` lines by construction -- while the comments explaining
        // *which of the two faults each step addresses* had drifted apart. That
        // prose is the whole point of this entry: the two faults need opposite
        // responses, so a user meeting the `diagnose` copy and a user meeting
        // the `rocm fix` copy must be told the same thing.
        let report = diagnose(&linux_base(), VLLM_OOM_CANONICAL_SYMPTOM);
        let fix = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .and_then(|d| d.fix.as_ref())
            .expect("the finding must carry a plan");
        let commands: Vec<&str> = fix.commands.iter().map(String::as_str).collect();
        crate::fix::assert_plan_matches_the_catalog_copy(&fix.fix_id, &commands);
    }

    #[test]
    fn the_extension_name_alone_does_not_establish_the_variant_failure() {
        // The extension appearing in a traceback says it is involved, not that
        // it loaded the CUDA variant. Holding this under the threshold is what
        // stops the entry from answering every vLLM import error.
        let report = diagnose(
            &linux_base(),
            "ImportError raised from torch_c_dlpack_ext during startup",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-17-torch-dlpack")
            .expect("the keyword should still register as a weak signal");
        assert!(
            hit.score < MIN_SCORE_FOR_MATCH,
            "score was {}, which would promote a weak signal to an established cause",
            hit.score
        );
    }

    /// The gate the rocm-doctor skill tells an agent to read, and the reason it
    /// is not "is `matched` empty?".
    ///
    /// This cannot be asserted from the e2e suite: `diagnose` scores several
    /// checkers from host state alone, with no symptom keyword involved, so on a
    /// runner that happens to have (say) `amdgpu` blacklisted the catalog
    /// explains the host no matter what symptom is passed. Constructing the
    /// `Examination` is the only way to hold the premise still.
    #[test]
    fn sub_threshold_causes_leave_has_match_false_and_route_upstream() {
        // Missing both groups scores 45 -- worth listing, not enough to
        // establish. `matched` is NOT empty here, which is the whole point: an
        // agent gating on emptiness would propose this fix for a host where
        // nothing was established, and never route the user anywhere.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "the office printer keeps jamming on page three");

        assert!(
            !report.matched.is_empty(),
            "this test is pointless unless something sub-threshold was listed"
        );
        assert!(
            report
                .matched
                .iter()
                .all(|d| d.score < report.min_score_for_match),
            "expected every cause below {}, got {:?}",
            report.min_score_for_match,
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
        assert!(
            !report.has_match,
            "nothing cleared the threshold, so has_match must be false -- \
             skills/rocm-doctor/ tells an agent to route upstream on exactly this"
        );
        assert!(
            report.route_when_no_match.url.starts_with("http"),
            "the skill's rule is to hand over this tracker when nothing was \
             established, so it must name somewhere to go: {:?}",
            report.route_when_no_match
        );
    }

    /// The other half: the flag has to discriminate, or asserting it is free.
    #[test]
    fn an_established_cause_sets_has_match() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "cannot open /dev/kfd: permission denied");
        assert!(
            report.has_match,
            "a symptom that matches the catalog must set has_match: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    /// Every diagnosis must agree with the fix catalog about whether the CLI
    /// can apply the fix itself.
    ///
    /// An agent following the rocm-doctor skill branches on `auto_applicable`
    /// from `diagnose --json` to decide whether to offer to run `rocm fix` or
    /// merely print the plan, while `fix::apply` dispatches on `RECIPES`. The
    /// two are written in different files and nothing but this test holds them
    /// together, so a drift makes the CLI contradict itself: it would advertise
    /// a fix as one it can run and then refuse, or the reverse.
    ///
    /// Both OS families are exercised because the checkers build their `Fix`
    /// per-OS and only one branch is taken per run, so a Linux-only test would
    /// leave the Windows branch free to drift unobserved.
    #[test]
    fn every_diagnosis_agrees_with_the_fix_catalog_on_auto_applicability() {
        for os in ["linux", "windows"] {
            let mut e = Examination {
                os_family: os.to_owned(),
                ..Examination::default()
            };
            // Trip several checkers at once so this covers more of the catalog
            // than a single fix.
            e.in_render_group = Some(false);
            e.in_video_group = Some(false);
            e.has_apu = true;
            e.has_discrete_amd = true;
            e.gpus = vec![
                Gpu {
                    gfx_target: "gfx1103".to_owned(),
                    is_amd: true,
                    is_apu: Some(true),
                    ..Gpu::default()
                },
                Gpu {
                    gfx_target: "gfx1100".to_owned(),
                    is_amd: true,
                    is_apu: Some(false),
                    ..Gpu::default()
                },
            ];

            let report = diagnose(&e, "torch crashes with a segfault");
            assert!(
                !report.matched.is_empty(),
                "{os}: expected at least one diagnosis to compare against the catalog"
            );
            for d in &report.matched {
                let fix = d
                    .fix
                    .as_ref()
                    .unwrap_or_else(|| panic!("{os}/{}: matched with no fix", d.id));
                let catalog = crate::fix::class_on(&fix.fix_id, os)
                    .unwrap_or_else(|| {
                        panic!("{os}/{}: emitted a fix-id not in the catalog", fix.fix_id)
                    })
                    .applies_itself();
                assert_eq!(
                    fix.auto_applicable, catalog,
                    "{os}/{}: diagnose reports auto_applicable={}, but the fix catalog \
                     (what `rocm fix` actually does) says {catalog}",
                    fix.fix_id, fix.auto_applicable
                );
            }
        }
    }

    /// ROCMAI-543: every string `rocm diagnose` can put on a redirectable
    /// stream has to be plain ASCII, or a Windows console on the OEM code page
    /// (437) renders UTF-8 punctuation as mojibake like `ΓÇö`.
    ///
    /// Driven off `CHECKERS` rather than off a list of individual messages.
    /// The catalog *is* the registry of everything this command can say, so a
    /// check added later is covered the day it lands and no single sentence
    /// needs a pinned copy here -- a test that restates the constant it checks
    /// protects very little. Both surfaces are asserted because both are
    /// redirectable: the rendered text report, and the `--json` payload, which
    /// `serde_json` emits as raw UTF-8 rather than `\u` escapes.
    ///
    /// Two of this fix's conversions sit on branches a default `Examination`
    /// never reaches -- the dGPU/APU pairing in `check_9_igpu_dgpu_collision`
    /// and `check_16_vllm_oom`'s "does not fit" wording -- so the corpus lights
    /// them deliberately and the coverage guard at the end fails if it stops.
    #[test]
    fn every_message_the_diagnose_catalog_can_emit_is_ascii() {
        let corpus: Vec<(&str, Examination, &str)> = vec![
            ("linux, no symptom", linux_base(), ""),
            ("wsl, no symptom", wsl_base(), ""),
            (
                "linux apu+dgpu, crash",
                igpu_dgpu_host("linux"),
                "torch crashes with a segfault",
            ),
            (
                "windows apu+dgpu, crash",
                igpu_dgpu_host("windows"),
                "torch crashes with a segfault",
            ),
            ("linux vllm oom", linux_base(), VLLM_OOM_CANONICAL_SYMPTOM),
            ("wsl vllm oom", wsl_base(), VLLM_OOM_CANONICAL_SYMPTOM),
        ];

        let mut everything_seen = String::new();
        for (label, examination, symptom) in &corpus {
            // Every catalog entry, not only the ones that scored high enough to
            // be reported on this host: a message is still shipped text when the
            // host it describes is not the one under test.
            for (check, _) in CHECKERS {
                let diagnosis = check(examination, symptom);
                let json = serde_json::to_string(&diagnosis).expect("a Diagnosis serializes");
                assert!(
                    json.is_ascii(),
                    "{label}: diagnosis `{}` carries non-ASCII punctuation, which renders as \
                     mojibake on a Windows console using the OEM code page (ROCMAI-543): {json}",
                    diagnosis.id
                );
                everything_seen.push_str(&json);
            }

            let report = diagnose(examination, symptom);
            let text = render_report_text(&report, report.matched.len());
            assert!(
                text.is_ascii(),
                "{label}: `rocm diagnose` text output must be ASCII-only (ROCMAI-543):\n{text}"
            );
            let json = serde_json::to_string(&report).expect("a DiagnoseReport serializes");
            assert!(
                json.is_ascii(),
                "{label}: `rocm diagnose --json` output must be ASCII-only (ROCMAI-543): {json}"
            );
            everything_seen.push_str(&text);
        }

        // An invariant test that quietly stops reaching the branches it was
        // written for passes for the wrong reason. These markers sit beside the
        // converted punctuation without containing it, so they stay true
        // whichever way the dash goes and only fail if the corpus rots.
        assert!(
            everything_seen.contains("Discrete GPU(s):"),
            "the corpus no longer reaches check_9's dGPU/APU branch, so its wording is unguarded"
        );
        assert!(
            everything_seen.contains("reservation will not help"),
            "the corpus no longer reaches check_16's does-not-fit wording, so it is unguarded"
        );
    }

    #[test]
    fn wsl2_never_runs_the_bare_metal_catalog() {
        // The property the old wholesale WSL short-circuit bought, kept after
        // WSL became a real platform in the catalog. Every signal below WOULD
        // fire a bare-metal check; none of them mean anything on WSL2, where
        // there is no amdgpu module, no /dev/kfd and no render group.
        let mut e = wsl_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        e.amdgpu_loaded = Some(false);
        e.rocm_version = "6.4.1".to_owned();
        e.amdgpu_blacklisted_in = vec!["/etc/modprobe.d/blacklist.conf".to_owned()];
        e.rocm_repos_seen = vec![
            "repo.radeon.com/rocm/6.2".to_owned(),
            "repo.radeon.com/rocm/6.4".to_owned(),
        ];
        let report = diagnose(&e, "unable to open /dev/kfd permission denied");
        for bare_metal in [
            "fix-3-rocm-kernel",
            "fix-4-render-group",
            "fix-5-amdgpu-load",
            "fix-7-stale-repos",
            "fix-10-container",
            "fix-11-iommu",
            "fix-12-installer",
        ] {
            assert!(
                !report.matched.iter().any(|d| d.id == bare_metal),
                "{bare_metal} must not fire on WSL2"
            );
        }
    }

    #[test]
    fn wsl_checks_never_fire_on_bare_metal() {
        // The converse guard. A bare-metal host has no WSL facts at all, and the
        // WSL checks read those facts as false -- so without the family gate they
        // would report a missing /dev/dxg on every ordinary Linux box.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "no ROCm-capable device is detected");
        assert!(
            !report.matched.iter().any(|d| d.id.starts_with("fix-wsl-")),
            "no WSL entry may fire on bare metal: {:?}",
            report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_shared_checks_under_report_on_wsl_rather_than_over_report() {
        // WSL does not collect GPU topology: `gpus` stays empty because the
        // probes that fill it read KFD and DRM, which do not exist there. The
        // cross-platform checks enabled on WSL read those fields, so they run on
        // less evidence here than on bare metal.
        //
        // That degradation has to be one-directional. Missing topology must mean
        // a check cannot reach its threshold on structure alone -- never that it
        // invents a fault. This pins the direction: on a healthy host with a
        // framework installed, no shared check may fire at all.
        let mut e = wsl_base();
        e.framework = "pytorch".to_owned();
        e.framework_version = "2.6.0".to_owned();
        e.framework_rocm_version = "6.4".to_owned();
        e.rocm_version = "6.4.1".to_owned();
        e.framework_arch_list = vec!["gfx1100".to_owned(), "gfx1151".to_owned()];
        e.rocm_path = "/opt/rocm".to_owned();
        e.env
            .insert("PATH".to_owned(), "/opt/rocm/bin:/usr/bin".to_owned());
        assert!(e.gpus.is_empty(), "WSL collects no GPU topology");

        let report = diagnose(&e, "");
        assert!(
            report.matched.is_empty(),
            "no shared check may fire without topology: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_platform_with_no_catalog_entries_says_nothing_was_checked() {
        // Previously only WSL set this. An unsupported OS fell through to an
        // empty catalog and reported "no known misconfiguration", which reads as
        // a clean bill of health when in truth nothing ran.
        let mut e = linux_base();
        e.os_family = "other".to_owned();
        let report = diagnose(&e, "anything at all");
        let reason = report
            .out_of_scope
            .as_deref()
            .expect("an uncovered platform must say so");
        assert!(reason.contains("other"), "must name the platform: {reason}");
        assert!(
            reason.contains("not a clean bill of health"),
            "must not be mistaken for a pass: {reason}"
        );
        assert!(report.matched.is_empty());
        assert!(!report.has_match());
    }

    #[test]
    fn an_out_of_scope_report_never_carries_matched_entries() {
        // The serialized JSON contract on `out_of_scope`: when it is set the
        // catalog did not run, so `matched` cannot hold anything -- not even a
        // weak sub-threshold signal. One direction only: a covered platform with
        // nothing wrong also has an empty `matched`, with `out_of_scope` unset.
        // Nothing else in the tree pins the pair, which is how the field's own
        // doc comment came to claim the opposite. Two independent mechanisms
        // uphold it today -- `diagnose` clears `matched` when `out_of_scope` is
        // set, and `run_all_checks` filters on the same platform family
        // `catalog_covers` consults -- so this fails once both are gone, which
        // is exactly the "a checker somehow ran on an uncovered host" shape the
        // JSON contract must not develop.
        let mut uncovered = linux_base();
        uncovered.os_family = "other".to_owned();
        let mut windows = linux_base();
        windows.os_family = "windows".to_owned();
        for e in [linux_base(), wsl_base(), windows, uncovered] {
            // A symptom that scores for at least one entry where the catalog
            // applies, so "matched is empty" is a real consequence of the
            // platform being uncovered rather than of a silent symptom.
            let report = diagnose(&e, VLLM_OOM_CANONICAL_SYMPTOM);
            assert!(
                report.out_of_scope.is_none() || report.matched.is_empty(),
                "out_of_scope set with matched entries on {}: {:?}",
                e.os_family,
                report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
            );
            assert!(
                report.out_of_scope.is_none() || !report.has_match(),
                "out_of_scope set alongside has_match on {}",
                e.os_family
            );
        }
    }

    #[test]
    fn non_wsl_still_diagnoses_normally() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "");
        assert!(report.out_of_scope.is_none());
        assert_eq!(report.matched[0].id, "fix-4-render-group");
    }

    #[test]
    fn the_serialized_verdict_agrees_with_the_accessor() {
        // Rust callers read the method, tooling reads the field. They answer the
        // same question, so a host where they disagree would hand the two
        // audiences different verdicts.
        let mut in_container = linux_base();
        in_container.in_container = true;
        let mut render_group_missing = linux_base();
        render_group_missing.in_render_group = Some(false);
        render_group_missing.in_video_group = Some(false);

        // The fixtures above all land BELOW the threshold, so on their own they
        // would only ever exercise the `false` branch -- and a field that is
        // always false serializes correctly by accident. The last one carries a
        // symptom that pushes the same finding past HIGH_CONFIDENCE, so `true`
        // reaches the wire here too and not only on a bare-metal e2e lane.
        let mut real_fault = linux_base();
        real_fault.in_render_group = Some(false);

        let cases = [
            (linux_base(), ""),
            (in_container, ""),
            (render_group_missing, ""),
            (real_fault, "RuntimeError: unable to open /dev/kfd"),
        ];
        assert!(
            cases
                .iter()
                .any(|(e, symptom)| diagnose(e, symptom).has_match),
            "at least one fixture must clear the threshold, or this only ever \
             proves the false branch"
        );

        for (e, symptom) in cases {
            let report = diagnose(&e, symptom);
            assert_eq!(
                report.has_match,
                report.has_match(),
                "field and accessor disagree for {:?}",
                report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
            );
            let json: serde_json::Value =
                serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
            assert_eq!(
                json.get("has_match").and_then(serde_json::Value::as_bool),
                Some(report.has_match),
                "the verdict must survive into the emitted document"
            );
        }
    }

    #[test]
    fn windows_only_checks_skipped_on_linux() {
        let e = linux_base();
        let report = diagnose(&e, "vcruntime140.dll is missing");
        // fix-15 is windows-only; must not appear on a linux exam.
        assert!(report.matched.iter().all(|d| d.id != "fix-15-msvc-redist"));
    }

    #[test]
    fn vllm_oom_signature_is_a_high_confidence_match() {
        // The distinctive vLLM startup OOM: torch/HIP allocation failure, with
        // the vLLM anchor that tells this apart from an arbitrary PyTorch OOM.
        let report = diagnose(
            &linux_base(),
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
        );
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-16-vllm-oom");
        assert!(top.score >= HIGH_CONFIDENCE, "score was {}", top.score);
        assert!(report.has_match());
        // The remediation must stay conditional: the reservation knob is right
        // for a tenancy collision and wrong for a model that does not fit.
        let fix = top.fix.as_ref().unwrap();
        assert!(!fix.auto_applicable, "OOM workaround must be print-only");
        assert!(fix.summary.contains("--gpu-memory-utilization"));
        assert!(
            fix.summary.contains("does not fit"),
            "the 'genuinely does not fit' caveat must survive: {}",
            fix.summary
        );
    }

    #[test]
    fn generic_pytorch_oom_without_a_vllm_signal_does_not_match() {
        // The reviewed false positive: `torch.OutOfMemoryError: CUDA out of
        // memory` scores 45+45=90 under the keyword table alone, but nothing
        // about that string is vLLM-specific -- any ROCm PyTorch job emits the
        // identical shape (ROCm's PyTorch build reports the CUDA-compat name).
        // Without an explicit vLLM anchor, this failure mode must not fire.
        let report = diagnose(&linux_base(), "torch.OutOfMemoryError: CUDA out of memory");
        assert!(
            report.matched.iter().all(|d| d.id != "fix-16-vllm-oom"),
            "a bare framework OOM with no vLLM signal must not be attributed to vLLM: {:?}",
            report.matched
        );
        assert!(!report.has_match());
    }

    #[test]
    fn torch_oom_class_alone_is_not_a_vllm_match() {
        // `outofmemory` is nested inside the exception class name. It must not
        // count as a second, independent signal and turn one token into a
        // high-confidence vLLM diagnosis for an arbitrary PyTorch workload.
        let report = diagnose(&linux_base(), "vllm: torch.OutOfMemoryError");
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("the weak keyword signal remains visible");
        assert!(oom.score < MIN_SCORE_FOR_MATCH, "score was {}", oom.score);
        assert!(!report.has_match());
    }

    #[test]
    fn a_bare_out_of_memory_stays_below_the_match_threshold() {
        // Without the vLLM/HIP shape, "out of memory" alone must not claim this
        // failure mode -- it scores (the `vllm` anchor is present and "out of
        // memory" is a keyword), but stays below MIN_SCORE_FOR_MATCH.
        let report = diagnose(&linux_base(), "vllm: the process was killed: out of memory");
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("the weak `out of memory` keyword signal must remain visible in `matched`");
        assert!(
            oom.score < MIN_SCORE_FOR_MATCH,
            "a bare OOM must stay sub-threshold, got {}",
            oom.score
        );
    }

    #[test]
    fn vllm_oom_is_not_available_on_native_windows() {
        // vLLM is Linux/WSL-only (native Windows is unsupported), so the checker
        // must not fire on a Windows examination even with the error text.
        let e = Examination {
            os_family: "windows".to_owned(),
            ..Examination::default()
        };
        let report = diagnose(&e, "vllm: torch.OutOfMemoryError: HIP out of memory.");
        assert!(report.matched.iter().all(|d| d.id != "fix-16-vllm-oom"));
    }

    #[test]
    fn vllm_oom_keyword_diagnosis_is_available_on_wsl() {
        let mut e = linux_base();
        e.is_wsl = true;
        let report = diagnose(
            &e,
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
        );
        assert!(report.out_of_scope.is_none());
        assert!(report.has_match());
        assert_eq!(report.matched[0].id, "fix-16-vllm-oom");
    }

    #[test]
    fn wsl_sub_threshold_vllm_signal_is_preserved_in_matched() {
        // A weak (sub-threshold) vLLM OOM signal on WSL must still surface in
        // `matched` per the `DiagnoseReport::matched` contract: `run_all_checks`
        // drops only zero-score results, so a score-25 mention is reported as a
        // WEAK row rather than silently discarded.
        //
        // This used to also assert that the weak hit did not bury a WSL
        // out-of-scope routing note. There is no such note any more -- WSL2 is a
        // covered platform with its own catalog entries -- so what is left to pin
        // is the half that still has teeth: the entry survives in `matched`, and
        // it does not promote itself into a match.
        //
        // `wsl_base()` (a healthy WSL GPU stack) rather than a default
        // examination: a default one has no /dev/dxg, so fix-wsl-1 would fire at
        // 55 and the report would match for reasons that have nothing to do with
        // the signal under test.
        let report = diagnose(&wsl_base(), "vllm: out of memory");
        assert!(!report.has_match());
        assert!(
            report.out_of_scope.is_none(),
            "WSL2 is a covered platform, not an out-of-scope one"
        );
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("a nonzero sub-threshold hit must stay in `matched`, not be dropped");
        assert!(oom.score < MIN_SCORE_FOR_MATCH, "score was {}", oom.score);
    }

    #[test]
    fn a_genuine_vllm_oom_match_on_wsl_is_surfaced_not_routed_out_of_scope() {
        // The other side of the has_match gate: a real, at-or-above-threshold
        // vLLM OOM on WSL is surfaced, and ranks top on a host whose own WSL GPU
        // stack is healthy -- so the verdict is the OOM, not an incidental WSL
        // finding.
        let report = diagnose(&wsl_base(), VLLM_OOM_CANONICAL_SYMPTOM);
        assert!(report.has_match());
        assert!(report.out_of_scope.is_none());
        assert_eq!(report.matched[0].id, "fix-16-vllm-oom");
    }

    #[test]
    fn hip_error_out_of_memory_status_is_a_match() {
        // Regression: `RuntimeError: hipErrorOutOfMemory` previously scored 0
        // (`\boutofmemory\b` has no word boundary after `hipError`), so the
        // engine printed a diagnose command that reported nothing.
        let report = diagnose(&linux_base(), "vllm: RuntimeError: hipErrorOutOfMemory");
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("hipErrorOutOfMemory must now be recognized");
        assert!(
            oom.score >= MIN_SCORE_FOR_MATCH,
            "hipErrorOutOfMemory should clear the threshold, got {}",
            oom.score
        );
    }

    #[test]
    fn torch_cuda_module_path_oom_class_is_recognized() {
        // Regression: ROCm's PyTorch reports the class under the `.cuda.` module
        // path (`torch.cuda.OutOfMemoryError`); the class pattern must match it.
        // It is still the (sub-threshold) class name, not a stronger message.
        let report = diagnose(&linux_base(), "vllm: torch.cuda.OutOfMemoryError");
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("the torch.cuda.* class name must be recognized as evidence");
        assert!(oom.score > 0, "score was {}", oom.score);
        assert!(
            oom.score < MIN_SCORE_FOR_MATCH,
            "the bare class name must stay sub-threshold, got {}",
            oom.score
        );
    }

    #[test]
    fn hip_out_of_memory_yields_a_single_evidence_line() {
        // Overlap guard: "HIP out of memory" must not be counted both as the
        // 50-point HIP message and the 25-point `out of memory` nested inside
        // it. One phrase, one evidence bullet.
        let report = diagnose(&linux_base(), "vllm: HIP out of memory");
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("HIP out of memory must match");
        let fix = oom.fix.as_ref().unwrap();
        let evidence_lines = oom
            .evidence
            .iter()
            .filter(|e| e.to_lowercase().contains("out of memory"))
            .count();
        assert_eq!(
            evidence_lines, 1,
            "one phrase must yield one evidence line, got {:?}",
            oom.evidence
        );
        assert_eq!(oom.score, 50, "the HIP message carries the match alone");
        assert!(fix.summary.contains("--gpu-memory-utilization"));
    }

    #[test]
    fn vllm_anchor_must_be_on_the_same_line_as_the_oom_token() {
        // Co-occurrence anywhere is too loose: a paste that mentions vLLM on one
        // line and an unrelated framework's OOM on another must not attribute
        // the OOM to vLLM.
        let report = diagnose(
            &linux_base(),
            "I installed vllm last week.\nToday my llama.cpp run died: CUDA out of memory",
        );
        assert!(
            report.matched.iter().all(|d| d.id != "fix-16-vllm-oom"),
            "an OOM on a different line from the vLLM mention must not match: {:?}",
            report.matched
        );
        // The engine's emitted `vllm: <failing line>` keeps them on one line and
        // still matches.
        let anchored = diagnose(&linux_base(), "vllm: CUDA out of memory");
        assert!(anchored.matched.iter().any(|d| d.id == "fix-16-vllm-oom"));
    }

    #[test]
    fn only_the_anchored_lines_are_scored_not_the_whole_paste() {
        // Regression: the anchor was a boolean *gate*, after which the entire
        // symptom was scored. Regex matching ignores line boundaries, so a
        // benign vLLM config echo on one line unlocked full-weight scoring of
        // another framework's OOM on a different line: `cuda out of memory`
        // (50) + `torch.outofmemoryerror` (45) do not overlap, so both survived
        // collapsing for 95 -- above HIGH_CONFIDENCE -- and the user got a
        // confident "vLLM ran the GPU out of memory" verdict for a llama.cpp
        // failure. Only the anchored lines may be scored.
        let report = diagnose(
            &linux_base(),
            "vllm config: gpu_memory_utilization=0.9\n\
             llama.cpp: torch.OutOfMemoryError: CUDA out of memory",
        );
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("the anchored config line is still a (weak) vLLM signal");
        assert!(
            oom.score < MIN_SCORE_FOR_MATCH,
            "an unanchored framework's OOM must not score for vLLM, got {} from {:?}",
            oom.score,
            oom.evidence
        );
        assert!(
            oom.evidence
                .iter()
                .all(|e| !e.to_lowercase().contains("out of memory")),
            "evidence must come from the anchored line only: {:?}",
            oom.evidence
        );

        // Same shape with the anchored line carrying no OOM token at all: the
        // unanchored OOM must not be borrowed to reach a verdict either.
        let report = diagnose(
            &linux_base(),
            "starting vllm serve on gpu 0\n\
             llama.cpp: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.21 GiB.",
        );
        assert!(
            report.matched.iter().all(|d| d.id != "fix-16-vllm-oom"),
            "an OOM on an unanchored line must not be attributed to vLLM: {:?}",
            report.matched
        );

        // The control: once the OOM is on the anchored line, it still matches at
        // full strength, so the scoping did not simply disable the checker.
        let report = diagnose(
            &linux_base(),
            "vllm config: gpu_memory_utilization=0.9\n\
             vllm: torch.OutOfMemoryError: CUDA out of memory",
        );
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("an anchored OOM line must still match");
        assert!(oom.score >= HIGH_CONFIDENCE, "score was {}", oom.score);
    }

    /// The `fix-16-vllm-oom` score for `symptom`, or 0 when it did not match.
    fn vllm_oom_score(symptom: &str) -> i32 {
        diagnose(&linux_base(), symptom)
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .map_or(0, |d| d.score)
    }

    #[test]
    fn any_line_advance_is_a_boundary_not_scoreable_text() {
        // Regression, in two rounds.
        //
        // The filter first used `str::lines()`, which splits on `\n` and `\r\n`
        // only -- a lone `\r` is ordinary text to it. Every progress bar in this
        // ecosystem (tqdm, pip, huggingface) repaints with a bare CR, and a
        // pasted terminal capture is the expected way to use `--symptom`, so the
        // whole capture collapsed into one "line": the vLLM anchor matched
        // somewhere in it and another framework's OOM elsewhere in it scored at
        // full weight. That is
        // `only_the_anchored_lines_are_scored_not_the_whole_paste` defeated by
        // one byte, and neither existing guard saw it because both use `\n`-only
        // fixtures.
        //
        // Adding `\r` to the split fixed that byte and left the hole open for
        // every other one. Measured on the round that did so: each separator
        // below put the anchor and a llama.cpp OOM on genuinely separate
        // rendered lines, and each scored 95 -- past a HIGH_CONFIDENCE of 75,
        // citing the llama.cpp line's tokens as the evidence for a vLLM verdict.
        // Hence a table rather than one test per byte: the property is "a line
        // advance is a line advance", and enumerating it is what stops the next
        // separator from reopening this.
        for (label, sep) in [
            ("LF", "\n"),
            ("CR", "\r"),
            ("CRLF", "\r\n"),
            ("ESC E (NEL)", "\u{1b}E"),
            ("ESC D (IND)", "\u{1b}D"),
            ("CSI 1 B (CUD)", "\u{1b}[1B"),
            ("CSI H (CUP)", "\u{1b}[2;1H"),
            ("VT", "\u{b}"),
            ("FF", "\u{c}"),
            ("C1 NEL", "\u{85}"),
            ("U+2028 LINE SEPARATOR", "\u{2028}"),
            ("U+2029 PARAGRAPH SEPARATOR", "\u{2029}"),
            ("a stray control byte", "\u{1}"),
        ] {
            let capture = format!(
                "Downloading shards:  10%{sep}vllm serve starting up{sep}\
                 Downloading shards:  90%{sep}\
                 llama.cpp: torch.OutOfMemoryError: CUDA out of memory. \
                 Tried to allocate 7.21 GiB.{sep}"
            );
            assert_eq!(
                vllm_oom_score(&capture),
                0,
                "a {label}-separated llama.cpp OOM must not be attributed to vLLM"
            );
        }

        // The control, per separator: a capture whose *anchored* segment carries
        // the OOM must still match, so the boundary set did not simply blind the
        // checker to every capture that contains one.
        for (label, sep) in [
            ("CR", "\r"),
            ("ESC E (NEL)", "\u{1b}E"),
            ("VT", "\u{b}"),
            ("a stray control byte", "\u{1}"),
        ] {
            let capture = format!(
                "Downloading shards:  90%{sep}\
                 vllm: torch.OutOfMemoryError: HIP out of memory{sep}"
            );
            let score = vllm_oom_score(&capture);
            assert!(
                score >= MIN_SCORE_FOR_MATCH,
                "an anchored OOM segment must still match across {label}: score was {score}"
            );
        }
    }

    #[test]
    fn colour_codes_inside_a_line_do_not_split_a_genuine_vllm_oom() {
        // The other half of the boundary question, and the reason this is a
        // grammar walk and not `char::is_control()`. A colourised logger emits
        // SGR *inside* a line -- level name in colour, message plain -- so a
        // splitter that treats every escape, or every control character, as a
        // boundary cuts a real `vllm: ... torch.OutOfMemoryError` between its
        // anchor and its error token. Measured on the one-line
        // `is_control()`-split candidate: this case went from 95 to 0, trading
        // the false positive above for a silent miss on the most ordinary real
        // input there is.
        let plain = "vllm: torch.OutOfMemoryError: HIP out of memory";
        let coloured =
            "\u{1b}[31mvllm: \u{1b}[1mtorch.OutOfMemoryError: HIP out of memory\u{1b}[0m";
        assert_eq!(
            vllm_oom_score(coloured),
            vllm_oom_score(plain),
            "SGR inside a line must not change the verdict"
        );
        let score = vllm_oom_score(coloured);
        assert!(score >= HIGH_CONFIDENCE, "score was {score}");

        // And the same line after a progress-bar repaint, which is how it
        // actually arrives: the CR is a boundary, the SGR around it is not.
        assert!(
            vllm_oom_score(&format!("Loading:  90%\r{coloured}\n")) >= HIGH_CONFIDENCE,
            "a colourised OOM after a CR repaint must still match"
        );
    }

    #[test]
    fn rocm_cli_s_own_low_vram_hint_is_never_a_verdict_on_its_own() {
        // `gpu[-_]memory[-_]utilization` is both an anchor and a scoring entry,
        // so a line that merely echoes the flag anchors itself. rocm-cli's own
        // low-VRAM hint prints `--gpu-memory-utilization`, and users paste it
        // back in. That must stay a weak signal, never a diagnosis: worth 20,
        // far below MIN_SCORE_FOR_MATCH, with nothing else on the line to
        // corroborate it.
        let report = diagnose(&linux_base(), crate::VLLM_GPU_MEMORY_UTILIZATION_HINT);
        let oom = report
            .matched
            .iter()
            .find(|d| d.id == "fix-16-vllm-oom")
            .expect("the flag mention stays visible as a weak signal");
        assert!(
            oom.score < MIN_SCORE_FOR_MATCH,
            "a bare flag echo must not diagnose an OOM, got {} from {:?}",
            oom.score,
            oom.evidence
        );
    }

    #[test]
    fn canonical_vllm_import_symptom_is_diagnosable() {
        // The engine's fallback symptom must always clear the threshold, so the
        // `rocm diagnose` command it prints never reports nothing.
        assert!(vllm_import_symptom_is_diagnosable(
            VLLM_IMPORT_CANONICAL_SYMPTOM
        ));
        // A frame that names only the shim's module, and not the missing library,
        // is not enough on its own, and the engine has to know that.
        assert!(!vllm_import_symptom_is_diagnosable(
            "File \"tvm_ffi/_optional_torch_c_dlpack.py\", line 42, in <module>"
        ));
        assert!(!vllm_import_symptom_is_diagnosable("an ordinary failure"));
    }

    #[test]
    fn vllm_import_keyword_diagnosis_is_available_on_wsl() {
        // The engine's import hint fires on WSL2 too, so `rocm diagnose
        // --symptom` has to answer there. Check 17 used to be registered for
        // Linux only, and this report had no match.
        let report = diagnose(&wsl_base(), VLLM_IMPORT_CANONICAL_SYMPTOM);
        assert!(report.out_of_scope.is_none());
        assert!(report.has_match());
        assert_eq!(report.matched[0].id, "fix-17-torch-dlpack");
    }

    #[test]
    fn the_engine_host_guard_requires_the_entry_to_score_on_every_host_family() {
        // A registry stand-in: `listed_on` says which hosts list the entry.
        fn registry(
            id: &'static str,
            listed_on_wsl: bool,
            listed_on_linux: bool,
        ) -> impl Fn(&Examination, &str) -> Vec<Diagnosis> {
            move |host, _symptom| {
                let listed = if host.is_wsl {
                    listed_on_wsl
                } else {
                    listed_on_linux
                };
                if !listed {
                    return Vec::new();
                }
                vec![Diagnosis {
                    id: id.to_owned(),
                    score: 100,
                    ..Diagnosis::default()
                }]
            }
        }
        let guard = |run| scores_on_every_engine_host("x", "fix-17-torch-dlpack", run);
        assert!(guard(registry("fix-17-torch-dlpack", true, true)));
        assert!(
            !guard(registry("fix-17-torch-dlpack", false, true)),
            "an entry missing on WSL must not count"
        );
        assert!(
            !guard(registry("fix-17-torch-dlpack", true, false)),
            "an entry missing on Linux must not count"
        );
        assert!(
            !guard(registry("fix-16-vllm-oom", true, true)),
            "a different entry scoring is not this entry scoring"
        );
        // And the real registry lists both entries on both families.
        assert!(vllm_import_symptom_is_diagnosable(
            VLLM_IMPORT_CANONICAL_SYMPTOM
        ));
        assert!(vllm_oom_symptom_is_diagnosable(VLLM_OOM_CANONICAL_SYMPTOM));
        // An ISA error scores on another check, not on this one.
        assert!(!vllm_import_symptom_is_diagnosable(
            "hipErrorNoBinaryForGpu: Unable to find code object for all current devices"
        ));
    }

    #[test]
    fn canonical_vllm_oom_symptom_is_diagnosable() {
        // The engine's fallback symptom must always clear the threshold, so the
        // `rocm diagnose` command it prints never reports nothing.
        assert!(vllm_oom_symptom_is_diagnosable(VLLM_OOM_CANONICAL_SYMPTOM));
        // And a bare framework mention the engine would reject is not diagnosable.
        assert!(!vllm_oom_symptom_is_diagnosable("vllm: out of memory"));
    }

    #[test]
    fn every_checker_platform_is_covered_by_its_recipe() {
        // `render_report_text` ends a matched diagnosis with `apply with: rocm
        // fix {id}`, and `fix::apply` then gates that id on the recipe's
        // `applies_on` against the *running* OS. So any platform family a
        // checker is registered for but its recipe omits is a platform where the
        // tool names a command and then refuses to run it -- which is exactly
        // what `fix-16-vllm-oom` did on WSL2, where the checker opts into `wsl`
        // and the recipe was LINUX_ONLY.
        //
        // Only one direction is an error. A recipe may legitimately apply more
        // widely than its checker answers (a user can reach `rocm fix <id>`
        // directly, without a diagnosis), so this asserts containment rather
        // than equality.
        //
        // No exemptions, including for the informational `info-20` entry: it is
        // a `DIAGNOSE-ONLY` member of the same catalog, so `rocm fix` answers
        // for it on both platforms its checker is registered for, and the
        // `apply with:` line diagnose prints for it is live like every other.
        for (check, families) in CHECKERS {
            let id = check(&Examination::default(), "").id;
            let applies_on = crate::fix::recipe_applies_on(&id).unwrap_or_else(|| {
                panic!(
                    "checker id `{id}` has no recipe in the fix catalog, so the \
                     `apply with: rocm fix {id}` line diagnose prints is dead"
                )
            });
            for family in *families {
                assert!(
                    applies_on.contains(family),
                    "`{id}` is diagnosed on `{family}` but its recipe only applies on \
                     {applies_on:?}; `rocm fix {id}` would print the plan and then exit 3 there"
                );
            }
        }
    }

    #[test]
    fn diagnosis_remediation_matches_the_fix_catalog_for_shared_fix_ids() {
        // fix-13, fix-14, fix-15, fix-16, fix-21, info-20 and fix-22 live in two places: the `rocm fix`
        // catalog (crate::fix::RECIPES) and the Fix these checks embed in a
        // Diagnosis. The prose *framing* legitimately differs by surface (a
        // catalog rationale vs a diagnosis summary), but the actionable
        // remediation must not silently diverge -- the verify command, the
        // notes, and the executable command lines have to agree. This is the
        // guard that they do; it caught a two-vs-three-space `verify` and a
        // reworded note.
        //
        // info-20 is the same dual-representation shape even though it has
        // nothing to run: its notes are the whole of its remedy (that
        // HSA_OVERRIDE_GFX_VERSION is not a workaround, and that `rocm serve`
        // already picks Vulkan), so the two copies drifting would leave one
        // surface giving advice the other contradicts. Its empty `commands` and
        // `verify` have to stay agreed on too, since a command appearing on one
        // side only is how a DIAGNOSE-ONLY entry silently stops being one.
        let vllm_oom = check_16_vllm_oom(
            &Examination {
                os_family: "linux".to_owned(),
                ..Examination::default()
            },
            VLLM_OOM_CANONICAL_SYMPTOM,
        );
        let msvc = check_15_msvc_redist(
            &Examination {
                os_family: "windows".to_owned(),
                msvc_redist_present: Some(false),
                ..Examination::default()
            },
            "ImportError: DLL load failed: vcruntime140_1.dll not found",
        );
        let msvc_outdated = check_21_msvc_redist_outdated(
            &Examination {
                os_family: "windows".to_owned(),
                msvc_redist_version: "14.29.30133.0".to_owned(),
                ..Examination::default()
            },
            "",
        );
        let unsupported_gpu = check_20_gpu_unsupported_by_rocm(
            &Examination {
                gpus: vec![Gpu {
                    gfx_target: "gfx90c".to_owned(),
                    is_amd: true,
                    ..Gpu::default()
                }],
                ..Examination::default()
            },
            "",
        );
        // fix-13 and fix-14 print a Windows verify that the Windows-only tests
        // in `fix.rs` run from the catalog copy, so this copy has to match it.
        let hip_sdk_missing = check_13_hip_sdk_missing(
            &Examination {
                os_family: "windows".to_owned(),
                ..Examination::default()
            },
            "ImportError: DLL load failed: amdhip64_6.dll was not found",
        );
        let adrenalin_old = check_14_adrenalin_too_old(
            &Examination {
                os_family: "windows".to_owned(),
                hip_sdk_path: r"C:\Program Files\AMD\ROCm\6.4".to_owned(),
                hipinfo_present: true,
                hipinfo_status: "error".to_owned(),
                ..Examination::default()
            },
            "driver too old",
        );
        // fix-22 is advice only: it scores below the match threshold on purpose,
        // yet a user meets both of its copies, so it is held to the same guard.
        let display_driver_old =
            check_22_display_driver_old(&windows_driver_exam("30.0.13014.8", "2021-08-04"), "");
        for diag in [
            vllm_oom,
            msvc,
            msvc_outdated,
            unsupported_gpu,
            hip_sdk_missing,
            adrenalin_old,
            display_driver_old,
        ] {
            if diag.id == crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID {
                assert_eq!(diag.score, DISPLAY_DRIVER_OLD_SCORE);
            } else {
                assert!(
                    diag.score >= MIN_SCORE_FOR_MATCH,
                    "test setup must yield a real match for {}",
                    diag.id
                );
            }
            let fix = diag.fix.expect("a matched diagnosis must carry a Fix");
            let (verify, notes, run_commands) =
                crate::fix::recipe_verify_notes_and_run_commands(&fix.fix_id)
                    .expect("a shared fix_id must exist in the recipe catalog");
            assert_eq!(fix.verify, verify, "verify diverged for {}", fix.fix_id);
            assert_eq!(
                fix.notes,
                notes.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>(),
                "notes diverged for {}",
                fix.fix_id
            );
            let emitted_run = fix
                .commands
                .iter()
                .map(String::as_str)
                .filter(|line| !line.trim_start().starts_with('#'))
                .collect::<Vec<_>>();
            assert_eq!(
                emitted_run, run_commands,
                "executable command lines diverged for {}",
                fix.fix_id
            );
        }
    }

    /// The `Fix` that each checker builds for itself is shown to users of
    /// Windows PowerShell 5.1 as well as cmd, just like the `rocm fix` catalog.
    /// This runs every checker that answers on Windows and applies the same
    /// rule as the catalog test in `fix.rs` (`fails_in_windows_powershell`), so a
    /// hand-built verify string or command that breaks one of the rule's known
    /// constructs fails here. A construct the rule does not list passes.
    ///
    /// Like that test it pins the text only. Nothing here runs a command in a
    /// shell. The Windows-only tests named on `fails_in_windows_powershell` in
    /// `fix.rs` run the fix-2, fix-13, fix-14, fix-15, fix-21 and fix-22 lines,
    /// and `the_igpu_verify_pins_the_variable_it_names` runs fix-9's, all on
    /// Windows.
    #[test]
    fn windows_diagnosis_remediation_passes_the_powershell_text_rule() {
        // Host state that wakes the checkers which answer from the machine and
        // not from the symptom: an override set, a HIP SDK with no `hipInfo`,
        // an APU beside a discrete GPU, and a runtime that is missing.
        let e = Examination {
            os_family: "windows".to_owned(),
            msvc_redist_present: Some(false),
            hip_sdk_path: r"C:\Program Files\AMD\ROCm\6.4".to_owned(),
            hipinfo_present: false,
            has_apu: true,
            has_discrete_amd: true,
            env: [
                ("HSA_OVERRIDE_GFX_VERSION".to_owned(), "11.0.0".to_owned()),
                ("PATH".to_owned(), r"C:\Windows".to_owned()),
            ]
            .into_iter()
            .collect(),
            ..Examination::default()
        };
        // Wide enough to wake the keyword-driven checkers, so the test sees
        // the Fix they build rather than an empty answer. fix-13 only scores
        // when the paste names HIP failing to load, hence `was not found`.
        let symptom = "hipErrorNoBinaryForGpu HSA_STATUS_ERROR_INVALID_ISA invalid device function \
                       HSA_OVERRIDE_GFX_VERSION vcruntime140_1.dll amdhip64_6.dll was not found \
                       hipblas.dll driver too old";
        let mut seen: Vec<String> = Vec::new();
        for (check, families) in CHECKERS {
            if !families.contains(&"windows") {
                continue;
            }
            let Some(fix) = check(&e, symptom).fix else {
                continue;
            };
            let printed = fix
                .commands
                .iter()
                .map(String::as_str)
                .filter(|c| !c.trim_start().starts_with('#'))
                .chain(std::iter::once(fix.verify.as_str()));
            for line in printed {
                if let Some(why) = crate::fix::fails_in_windows_powershell(line) {
                    panic!("{}: `{line}` uses {why}", fix.fix_id);
                }
            }
            seen.push(fix.fix_id);
        }
        // Guard the test itself: it must have looked at the Windows checkers.
        for expected in [
            "fix-2-unset-override",
            "fix-6-path",
            "fix-9-igpu-dgpu",
            "fix-13-hip-sdk-missing",
            "fix-14-adrenalin-too-old",
            "fix-15-msvc-redist",
        ] {
            assert!(
                seen.iter().any(|id| id == expected),
                "the sweep never saw {expected}; it saw {seen:?}"
            );
        }
    }

    #[test]
    fn report_serializes_expected_shape() {
        let report = diagnose(&linux_base(), "");
        let v = serde_json::to_value(&report).unwrap();
        for key in [
            "matched",
            "has_match",
            "min_score_for_match",
            "high_confidence_threshold",
            "route_when_no_match",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["min_score_for_match"], 50);
        assert_eq!(v["high_confidence_threshold"], 75);
    }
}
