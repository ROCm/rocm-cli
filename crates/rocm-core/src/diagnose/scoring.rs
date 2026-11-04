// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The catalog's data and the scoring that reads it.
//!
//! Every keyword table is a list of `(regex, weight, label)` rows matched against
//! the lowercased symptom text, and [`keyword_score`] turns a table into a score
//! plus the evidence labels behind it. The tables are shared by checks on
//! different platforms, which is why they sit together here instead of beside
//! each check. The helpers at the end read an [`Examination`] or build the
//! [`Diagnosis`] a check returns.

use super::{Diagnosis, Fix};
use crate::examine::Examination;
use regex::Regex;

// ---------------------------------------------------------------------------
// Symptom keyword tables: (regex, weight, evidence-label). Patterns are
// lowercase and matched against the lowercased symptom.
// ---------------------------------------------------------------------------

type KeywordTable = &'static [(&'static str, i32, &'static str)];

pub(super) const KEYWORDS_INVALID_ISA: KeywordTable = &[
    (
        "hiperrornobinaryforgpu",
        45,
        "error mentions hipErrorNoBinaryForGpu",
    ),
    (
        "hsa_status_error_invalid_isa",
        50,
        "error mentions HSA_STATUS_ERROR_INVALID_ISA",
    ),
    (
        "invalid device function",
        40,
        "error mentions 'invalid device function'",
    ),
    (
        "no kernel image is available",
        35,
        "error mentions 'no kernel image is available'",
    ),
    (
        r"gfx\d{3,4}.* not (?:in|on) .*arch",
        35,
        "error names a missing gfx in arch list",
    ),
];

pub(super) const KEYWORDS_KFD_PERMISSION: KeywordTable = &[
    (
        "unable to open /dev/kfd",
        50,
        "error mentions /dev/kfd open failure",
    ),
    (
        r"/dev/kfd.*permission denied",
        45,
        "error mentions /dev/kfd permission denied",
    ),
    (
        "hsa_status_error_out_of_resources",
        25,
        "HSA out-of-resources (often perms)",
    ),
    ("failed to open kfd", 35, "error mentions kfd open failure"),
];

pub(super) const KEYWORDS_MODULE_NOT_LOADED: KeywordTable = &[
    (
        "rock module is not loaded",
        50,
        "rocminfo says ROCk module is NOT loaded",
    ),
    ("no devices? found", 20, "vague 'no devices found'"),
    ("hsa_status_error", 10, "HSA error (broad)"),
];

/// What a shared-memory shortage leaves in the error text.
///
/// Weak on purpose, and none of them reaches the match threshold alone. The
/// defining property of this failure is that the crash **does not** name shared
/// memory — a bus error in a data-loader worker, or a write failure against a
/// temporary file. An entry that needed the right words would never fire for the
/// user who needs it, so the machine state has to carry the finding and these
/// only raise it.
pub(super) const KEYWORDS_SHM_TOO_SMALL: KeywordTable = &[
    (r"/dev/shm", 40, "error mentions /dev/shm"),
    ("shared memory", 35, "error mentions shared memory"),
    (
        "bus error",
        25,
        "bus error -- what a data-loader worker reports when the allowance runs out",
    ),
    (
        "dataloader worker.*killed",
        25,
        "a data-loader worker was killed",
    ),
    ("no space left on device", 20, "the device reported full"),
];

/// What a shadowed code object manager leaves in the error text.
///
/// Every one of these is a *weak* signal on its own — a failed device-code
/// compilation has many causes, and this entry is established by the state of
/// the machine rather than by the words. The keywords only raise an already
/// structural finding; none of them reaches the match threshold alone.
pub(super) const KEYWORDS_COMGR_CONFLICT: KeywordTable = &[
    (r"libamd_comgr", 40, "error mentions libamd_comgr"),
    ("comgr", 30, "error mentions comgr"),
    (
        "code object",
        25,
        "error mentions a code object (what comgr produces)",
    ),
    (
        "hiperrornobinaryforgpu",
        25,
        "HIP found no binary for the GPU",
    ),
    ("device code", 15, "error mentions device code (broad)"),
];

pub(super) const KEYWORDS_PATH_MISSING: KeywordTable = &[
    ("rocminfo: command not found", 50, "rocminfo not on PATH"),
    ("command not found.*hipcc", 40, "hipcc not on PATH"),
    ("/opt/rocm/bin", 15, "user mentions /opt/rocm/bin"),
];

pub(super) const KEYWORDS_LIB_MISMATCH: KeywordTable = &[
    (r"libamdhip64\.so", 50, "error mentions libamdhip64.so"),
    ("libhsa-runtime", 45, "error mentions libhsa-runtime"),
    ("libhipblas", 40, "error mentions libhipblas"),
    (
        r"amdhip64_\d+\.dll",
        50,
        "error mentions amdhip64_X.dll (Windows)",
    ),
    (r"hipblas\.dll", 40, "error mentions hipblas.dll (Windows)"),
    ("cannot open shared object file", 25, "ldopen failure"),
    ("dll load failed", 25, "Windows DLL load failure"),
    ("version `?glibc", 5, "tangential glibc version error"),
];

pub(super) const KEYWORDS_HIP_SDK_MISSING: KeywordTable = &[
    ("amdhip64.*not found", 50, "error names amdhip64 missing"),
    ("could not find hip", 40, "error mentions HIP not found"),
    ("hip_path.*not set", 35, "user mentions HIP_PATH unset"),
    (
        "hipinfo.*not recognized",
        45,
        "Windows says hipInfo is not a command",
    ),
];

pub(super) const KEYWORDS_MSVC_REDIST: KeywordTable = &[
    (
        r"vcruntime140(?:_1)?\.dll",
        50,
        "error mentions vcruntime140 / vcruntime140_1",
    ),
    (
        r"api-ms-win-crt-.*\.dll",
        35,
        "error mentions api-ms-win-crt-* DLL",
    ),
    (
        "the (program|application) can't start because",
        25,
        "Windows missing-DLL dialog text",
    ),
    (r"msvcp140\.dll", 30, "error mentions msvcp140.dll"),
];

/// Symptom text for [`check_21_msvc_redist_outdated`].
///
/// An access violation is generic: any native crash prints one. So the code
/// alone stays under [`MIN_SCORE_FOR_MATCH`] (25), the DLL name alone does too
/// (30), and the two together (55) reach "likely". A crash report that names
/// `msvcp140` as the faulting module is specific (55 alone), and with the
/// exception code it reaches high confidence (80). This table is scored with
/// overlap collapsing, so the faulting-module line and the DLL name inside it
/// are one signal, not two.
pub(super) const KEYWORDS_MSVC_REDIST_OUTDATED: KeywordTable = &[
    (
        r"faulting module (?:name|path)?\s*:?\s*(?:[^\s,]*[\\/])?msvcp140",
        55,
        "crash report names msvcp140 as the faulting module",
    ),
    (
        r"0xc0000005|-1073741819|3221225477|access violation",
        25,
        "access violation exit or exception code",
    ),
    (r"msvcp140\.dll", 30, "error mentions msvcp140.dll"),
];

pub(super) const KEYWORDS_REPO_BROKEN: KeywordTable = &[
    (r"404.*repo\.radeon\.com", 50, "404 against repo.radeon.com"),
    (
        "release file (is )?not (yet )?valid",
        30,
        "apt 'release file not valid'",
    ),
    (
        "the following packages have unmet dependencies",
        25,
        "apt unmet dependencies",
    ),
    (
        "unable to locate package rocm",
        35,
        "apt cannot find ROCm package",
    ),
];

pub(super) const KEYWORDS_CONTAINER: KeywordTable = &[
    (
        "hsa_status_error.*permission",
        20,
        "HSA permission error (often container)",
    ),
    (r"/dev/dri.*permission", 30, "/dev/dri permission failure"),
    ("failed to open device", 25, "device open failure"),
];

pub(super) const KEYWORDS_IOMMU_HANG: KeywordTable = &[
    ("hang", 20, "user mentions 'hang'"),
    ("deadlock", 20, "user mentions deadlock"),
    ("timed out waiting", 25, "ring/queue timeout"),
    ("iommu", 30, "user mentions iommu"),
];

pub(super) const KEYWORDS_DPKG_BROKEN: KeywordTable = &[
    ("half[- ]configured", 50, "dpkg 'half-configured'"),
    ("dkms .*failed", 45, "DKMS build failure"),
    ("dpkg: error", 25, "generic dpkg error"),
    (
        "sub-process /usr/bin/dpkg returned",
        25,
        "apt mentions dpkg failure",
    ),
];

pub(super) const KEYWORDS_PAGE_FAULT: KeywordTable = &[
    ("page fault", 40, "user mentions page fault"),
    ("vm_fault", 35, "kernel vm_fault"),
    ("hw_fault", 30, "amdgpu HW fault"),
    ("out_of_registers", 30, "compiler OUT_OF_REGISTERS"),
];

// vLLM's startup OOM. The distinctive signature is a torch/HIP allocation
// failure while the engine is reserving KV-cache VRAM; a bare "out of memory"
// on its own is deliberately sub-threshold so this only claims the failure mode
// when the vLLM/HIP shape of the error is present.
//
// None of these tokens are vLLM-specific on their own: `torch.OutOfMemoryError:
// CUDA out of memory` is the identical shape any ROCm PyTorch job emits (ROCm's
// PyTorch build reports the CUDA-compat name), so this table alone cannot tell
// vLLM's OOM apart from an arbitrary training script hitting the same
// allocator error. `check_16_vllm_oom` therefore scores *only* the lines
// carrying `VLLM_ANCHOR_PATTERN` (see `vllm_anchored_lines`), never the whole
// pasted log: an OOM on an unanchored line contributes nothing at all.
//
// Weighting splits the *allocator message* from the *exception class name*:
//   - The HIP allocator's own messages (`HIP out of memory`, the
//     `hipErrorOutOfMemory` status) and the CUDA-compat message ROCm's PyTorch
//     prints clear `MIN_SCORE_FOR_MATCH` on their own — with the required
//     anchor already establishing this is vLLM's line, one of those messages is
//     a genuine OOM signal.
//   - The bare exception *class* (`torch.OutOfMemoryError`, with or without the
//     `.cuda.` module path) and a bare `out of memory` stay sub-threshold: any
//     PyTorch job emits the class name, so it must corroborate rather than
//     carry the verdict alone (`torch_oom_class_alone_is_not_a_vllm_match`).
// This table alone is scored through `keyword_score_collapsing_overlaps`, which
// de-duplicates overlapping spans, so `HIP out of memory` counts once (as the
// 50-point message), not also as the nested 25-point `out of memory` — one
// phrase yields one evidence bullet. Every other table keeps the catalog's
// default "each entry is an independent signal" scoring.
pub(super) const KEYWORDS_VLLM_OOM: KeywordTable = &[
    (
        "hip out of memory",
        50,
        "error mentions 'HIP out of memory'",
    ),
    (
        // ROCm's HIP runtime OOM status code, e.g. `RuntimeError:
        // hipErrorOutOfMemory`. `\boutofmemory\b` does not match inside it (no
        // word boundary after `hipError`), so it needs its own entry.
        r"hiperroroutofmemory",
        50,
        "error mentions hipErrorOutOfMemory",
    ),
    (
        r"cuda out of memory",
        50,
        "error mentions 'CUDA out of memory' (ROCm reports the CUDA-compat name)",
    ),
    (
        // The exception class name, under either `torch.OutOfMemoryError` or the
        // `.cuda.` module path ROCm's PyTorch reports it under. Sub-threshold
        // alone: it is the class every PyTorch OOM raises, not vLLM's signature.
        r"torch\.(?:cuda\.)?outofmemoryerror",
        45,
        "error mentions torch.OutOfMemoryError",
    ),
    (
        r"tried to allocate .*(?:gib|mib)",
        30,
        "error names an allocation it could not satisfy",
    ),
    (
        r"\boutofmemory\b",
        30,
        "error mentions standalone OutOfMemory",
    ),
    ("out of memory", 25, "error mentions 'out of memory'"),
    (
        r"gpu[-_]memory[-_]utilization",
        20,
        "log mentions gpu_memory_utilization (vLLM VRAM reservation)",
    ),
];

/// The vLLM engine-startup import failure: `torch-c-dlpack-ext` picks its CUDA
/// prebuilt on a ROCm build of torch, and `ctypes.CDLL` aborts the import.
///
/// `libtorch_cuda.so` alone is worth exactly [`MIN_SCORE_FOR_MATCH`], on
/// purpose. It is the one token in that traceback that cannot mean anything
/// else — a ROCm build of torch ships `libtorch_hip.so` and never that file —
/// and a user who pastes only the `OSError` line still has to clear the bar, or
/// the entry loses to the sub-threshold noise it exists to outrank. Everything
/// less specific stays below it: `torch_c_dlpack_ext` on its own says the
/// extension is in the picture, not that it chose the wrong variant.
pub(super) const KEYWORDS_TORCH_DLPACK_CUDA_VARIANT: KeywordTable = &[
    (
        r"libtorch_cuda\.so",
        50,
        "error names libtorch_cuda.so, which a ROCm build of torch does not ship",
    ),
    (
        "torch_c_dlpack_ext",
        45,
        "error names the torch_c_dlpack_ext extension",
    ),
    (
        "_optional_torch_c_dlpack",
        35,
        "error names tvm_ffi's _optional_torch_c_dlpack shim",
    ),
];

/// Score the strongest (top-2) keyword matches in `table` against `symptom`.
///
/// Every entry that matches counts as an independent signal. This is the
/// behaviour every table in the catalog except [`KEYWORDS_VLLM_OOM`] is tuned
/// for -- see [`keyword_score_collapsing_overlaps`] for why that one differs and
/// why the difference must not be generalized.
pub(super) fn keyword_score(symptom: &str, table: KeywordTable) -> (i32, Vec<String>) {
    keyword_score_impl(symptom, table, false)
}

/// Like [`keyword_score`], but collapses matches whose spans overlap to the
/// strongest one, so a single phrase cannot masquerade as two independent
/// signals (e.g. `HIP out of memory` must not also count the `out of memory`
/// nested inside it).
///
/// This is **deliberately not** applied to the rest of the catalog. Several
/// older tables pair a greedy `.*` pattern with a second, genuinely independent
/// keyword, and the greedy span swallows it: `api-ms-win-crt-.*\.dll` runs to
/// the last `.dll` on the line and covers an independent `msvcp140.dll`,
/// `dkms .*failed` covers an independent `dpkg: error`, and so on. Since
/// collapsing can only lower a score, applying it there would push real
/// diagnoses below [`MIN_SCORE_FOR_MATCH`] and make them vanish
/// (`overlap_dedup_does_not_demote_other_catalog_keyword_tables` pins this).
/// [`KEYWORDS_VLLM_OOM`] is safe because its overlaps are true nestings of
/// literal phrases, not artifacts of a greedy wildcard.
pub(super) fn keyword_score_collapsing_overlaps(
    symptom: &str,
    table: KeywordTable,
) -> (i32, Vec<String>) {
    keyword_score_impl(symptom, table, true)
}

fn keyword_score_impl(
    symptom: &str,
    table: KeywordTable,
    collapse_overlaps: bool,
) -> (i32, Vec<String>) {
    if symptom.is_empty() {
        return (0, Vec::new());
    }
    let sym = symptom.to_lowercase();
    let mut hits: Vec<(i32, &'static str, usize, usize)> = Vec::new();
    for (pattern, weight, label) in table {
        if let Ok(re) = Regex::new(pattern)
            && let Some(m) = re.find(&sym)
        {
            hits.push((*weight, label, m.start(), m.end()));
        }
    }
    if hits.is_empty() {
        return (0, Vec::new());
    }
    // Mirror diagnose.py's `hits.sort(reverse=True)`: weight desc, then label desc.
    hits.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(a.1)));
    // Keep the strongest hit for any overlapping span: the same substring must
    // not contribute two evidence bullets (nor two weights toward the score).
    let mut kept: Vec<(i32, &'static str, usize, usize)> = Vec::new();
    for hit in hits {
        let overlaps = collapse_overlaps && kept.iter().any(|k| hit.2 < k.3 && k.2 < hit.3);
        if !overlaps {
            kept.push(hit);
        }
    }
    kept.truncate(2);
    let score = kept.iter().map(|(w, ..)| *w).sum();
    let labels = kept.iter().map(|(_, l, ..)| (*l).to_owned()).collect();
    (score, labels)
}

/// Whether `symptom` (lowercased) matches `pattern`.
pub(super) fn symptom_matches(symptom: &str, pattern: &str) -> bool {
    !symptom.is_empty() && Regex::new(pattern).is_ok_and(|re| re.is_match(&symptom.to_lowercase()))
}

// ---------------------------------------------------------------------------
// Examination accessors
// ---------------------------------------------------------------------------

pub(super) fn amd_gfx_targets(e: &Examination) -> Vec<String> {
    e.gpus
        .iter()
        .filter(|g| g.is_amd && !g.gfx_target.is_empty())
        .map(|g| g.gfx_target.clone())
        .collect()
}

pub(super) fn amd_gpu_count(e: &Examination) -> usize {
    e.gpus.iter().filter(|g| g.is_amd).count()
}

pub(super) fn zero(id: &str, title: &str) -> Diagnosis {
    Diagnosis {
        id: id.to_owned(),
        title: title.to_owned(),
        ..Diagnosis::default()
    }
}

pub(super) fn finalize(
    id: &str,
    title: &str,
    score: i32,
    evidence: Vec<String>,
    fix: Fix,
) -> Diagnosis {
    Diagnosis {
        id: id.to_owned(),
        title: title.to_owned(),
        score: score.min(100),
        evidence,
        fix: Some(fix),
    }
}

pub(super) const KEYWORDS_WSL_NO_DEVICE: KeywordTable = &[
    (
        "no rocm-capable device",
        40,
        "error mentions no ROCm-capable device",
    ),
    (
        "no hip-capable device",
        40,
        "error mentions no HIP-capable device",
    ),
    (r"/dev/dxg", 45, "error mentions /dev/dxg"),
    (
        "hsa_status_error_out_of_resources",
        25,
        "error mentions HSA_STATUS_ERROR_OUT_OF_RESOURCES",
    ),
    (
        "no amd gpus? (?:were )?(?:found|detected)",
        35,
        "error mentions no AMD GPU found",
    ),
];

pub(super) const KEYWORDS_WSL_LOADER: KeywordTable = &[
    (r"librocdxg\.so", 50, "error names librocdxg.so"),
    (r"libdxcore\.so", 50, "error names libdxcore.so"),
    (
        "cannot open shared object file",
        30,
        "error mentions a shared object that could not be opened",
    ),
    (
        "error while loading shared libraries",
        35,
        "error mentions a shared library load failure",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::{HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH};

    #[test]
    fn overlap_dedup_does_not_demote_other_catalog_keyword_tables() {
        // The vLLM-OOM table needs overlapping spans collapsed ("out of memory"
        // nested inside "HIP out of memory"). Other tables must NOT get that
        // treatment: several of them pair a greedy `.*` pattern with a second,
        // genuinely independent keyword, and the greedy span swallows the
        // independent one. Collapsing there silently loses real diagnoses,
        // because a keyword score can only fall.
        //
        // Each case below pins the score these tables produced before the
        // de-duplication was introduced.

        // `api-ms-win-crt-.*\.dll` (35) runs to the *last* `.dll` on the line,
        // covering the independent `msvcp140.dll` (30). Two distinct missing
        // DLLs, two signals -- 65 clears MIN_SCORE_FOR_MATCH, 35 does not.
        let (score, ev) = keyword_score(
            "api-ms-win-crt-runtime-l1-1-0.dll missing and msvcp140.dll not found",
            KEYWORDS_MSVC_REDIST,
        );
        assert_eq!(
            score, 65,
            "two distinct missing DLLs are two signals: {ev:?}"
        );
        assert!(score >= MIN_SCORE_FOR_MATCH);

        // `dkms .*failed` (45) spans the whole line, covering `dpkg: error` (25).
        let (score, ev) = keyword_score(
            "dkms status: dpkg: error processing amdgpu-dkms, build failed",
            KEYWORDS_DPKG_BROKEN,
        );
        assert_eq!(
            score, 70,
            "a DKMS failure and a dpkg error are two signals: {ev:?}"
        );
        assert!(score >= MIN_SCORE_FOR_MATCH);

        // `amdhip64.*not found` (50) spans past `could not find hip` (40):
        // 90 is HIGH confidence, 50 is only a bare match.
        let (score, ev) = keyword_score(
            "amdhip64.dll: could not find hip runtime, not found",
            KEYWORDS_HIP_SDK_MISSING,
        );
        assert_eq!(score, 90, "both HIP-SDK signals must count: {ev:?}");
        assert!(score >= HIGH_CONFIDENCE);

        // `404.*repo\.radeon\.com` (50) runs to the *last* repo.radeon.com on
        // the line, covering `unable to locate package rocm` (35).
        let (score, ev) = keyword_score(
            "e: failed to fetch https://repo.radeon.com/rocm/apt/jammy/release 404 not found, \
             unable to locate package rocm, retrying https://repo.radeon.com/rocm/apt",
            KEYWORDS_REPO_BROKEN,
        );
        assert_eq!(score, 85, "both apt-repo signals must count: {ev:?}");
        assert!(score >= HIGH_CONFIDENCE);
    }

    #[test]
    fn outdated_runtime_symptom_keywords_score_honestly() {
        let score =
            |text: &str| keyword_score_collapsing_overlaps(text, KEYWORDS_MSVC_REDIST_OUTDATED).0;
        // A bare access violation code is generic and stays below the threshold,
        // whichever way it is spelled.
        for bare in [
            "0xC0000005",
            "exit code -1073741819",
            "exit code 3221225477",
        ] {
            assert!(score(bare) > 0, "{bare} must be recognised");
            assert!(score(bare) < MIN_SCORE_FOR_MATCH, "{bare} alone is generic");
        }
        // So does the DLL name alone.
        assert!(score("msvcp140.dll") < MIN_SCORE_FOR_MATCH);
        // The two together are a likely match.
        let pair = score("crashed with 0xC0000005 in msvcp140.dll");
        assert!(
            (MIN_SCORE_FOR_MATCH..HIGH_CONFIDENCE).contains(&pair),
            "{pair}"
        );
        // A faulting-module line is specific on its own, whatever the case, and
        // with the exception code it is high confidence.
        for line in [
            "Faulting module name: MSVCP140.dll",
            "FAULTING MODULE NAME: msvcp140.dll, version: 14.29.30133.0",
            r"Faulting module path: C:\Windows\System32\MSVCP140.dll",
        ] {
            let s = score(line);
            assert!(s >= MIN_SCORE_FOR_MATCH, "{line}: {s}");
            assert!(s < HIGH_CONFIDENCE, "{line}: {s}");
            assert!(score(&format!("{line} Exception code: 0xc0000005")) >= HIGH_CONFIDENCE);
        }
        // Another module faulting is not this failure.
        assert!(
            score("Faulting module name: ucrtbase.dll, Exception code: 0xc0000005")
                < MIN_SCORE_FOR_MATCH
        );
    }

    #[test]
    fn keyword_score_takes_top_two() {
        // INVALID_ISA: two hits (50 + 40) -> 90, not the sum of all.
        let (score, labels) = keyword_score(
            "HSA_STATUS_ERROR_INVALID_ISA and invalid device function and no kernel image is available",
            KEYWORDS_INVALID_ISA,
        );
        assert_eq!(score, 90);
        assert_eq!(labels.len(), 2);
    }
}
