// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The checks on the engine runtime that run on Linux and WSL2: the vLLM
//! out-of-memory and import failures, a second copy of the code object manager,
//! and a `/dev/shm` too small for a serving workload, with the two canonical
//! symptom strings the vLLM engine falls back to. The classifiers the engine
//! calls (`vllm_oom_symptom_is_diagnosable` and its import sibling) run the whole
//! catalog, so they live in the module root rather than here.

use super::scoring::{
    KEYWORDS_COMGR_CONFLICT, KEYWORDS_SHM_TOO_SMALL, KEYWORDS_TORCH_DLPACK_CUDA_VARIANT,
    KEYWORDS_VLLM_OOM, finalize, keyword_score, keyword_score_collapsing_overlaps, zero,
};
use super::{Diagnosis, Fix, MIN_SCORE_FOR_MATCH};
use crate::examine::Examination;
use regex::Regex;

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

/// vLLM ran the GPU out of memory. Keyword-only: an [`Examination`] carries no
/// per-GPU VRAM or tenancy fields, so nothing structural can corroborate this —
/// the user must pass the error via `--symptom`, or arrive from the serve
/// failure note that points here. `e` is therefore unused.
pub(super) fn check_16_vllm_oom(_e: &Examination, symptom: &str) -> Diagnosis {
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
pub(super) fn check_18_comgr_conflict(e: &Examination, symptom: &str) -> Diagnosis {
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

/// Whether the entry `id` clears [`MIN_SCORE_FOR_MATCH`] for `symptom` on every
/// host family the vLLM engine runs on (Linux and WSL2). `run` is the registry
/// run for one host. It is a parameter so a test can hand in a registry where
/// the entry is missing on one family.
pub(super) fn scores_on_every_engine_host(
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
pub(super) fn check_17_torch_dlpack_cuda_variant(_e: &Examination, symptom: &str) -> Diagnosis {
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
pub(super) fn check_19_shm_too_small(e: &Examination, symptom: &str) -> Diagnosis {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::test_support::{linux_base, wsl_base};
    use crate::diagnose::{DiagnoseReport, HIGH_CONFIDENCE, diagnose};

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
    fn vllm_import_keyword_diagnosis_is_available_on_wsl() {
        // The engine's import hint fires on WSL2 too, so `rocm diagnose
        // --symptom` has to answer there. Check 17 used to be registered for
        // Linux only, and this report had no match.
        let report = diagnose(&wsl_base(), VLLM_IMPORT_CANONICAL_SYMPTOM);
        assert!(report.out_of_scope.is_none());
        assert!(report.has_match());
        assert_eq!(report.matched[0].id, "fix-17-torch-dlpack");
    }
}
