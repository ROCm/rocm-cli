// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The binary's side of `rocm diagnose --model`'s GGUF context-size estimate.
//!
//! The estimate itself lives in `rocm-core` ([`rocm_core::model_readiness`] and
//! the `gguf*` modules). What is here is what only the binary knows: the flags
//! as clap parsed them ([`KvCacheTypeArg`], [`ModelQuery`]), what `--report`
//! says when it is handed a model question it will not answer, and the mask
//! caveat: whether a device-visibility variable is set (read here), over the
//! `amd-smi` rows the host's memory was classified from (passed in).
//!
//! The `Diagnose` command variant, its dispatch arm and `diagnose()` itself stay
//! in `main.rs` with the rest of the command tree.

use crate::gpu_policy::GpuVramUsage;
use clap::ValueEnum;
use rocm_core::gguf_fit::KvCacheType;
use rocm_core::gguf_source::GgufSource;
use rocm_core::model_readiness::{
    ContextRequest, HostFacts, ModelReadiness, attach_context_estimate, note_unverified_mask,
};

/// KV cache element type for `rocm diagnose --model`'s GGUF estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum KvCacheTypeArg {
    /// 16-bit floats (llama.cpp's default).
    #[value(name = "f16")]
    F16,
    /// 8-bit quantized: about half the size of f16.
    #[value(name = "q8_0")]
    Q8_0,
    /// 4-bit quantized: about a quarter the size of f16.
    #[value(name = "q4_0")]
    Q4_0,
}

impl From<KvCacheTypeArg> for KvCacheType {
    fn from(arg: KvCacheTypeArg) -> Self {
        match arg {
            KvCacheTypeArg::F16 => Self::F16,
            KvCacheTypeArg::Q8_0 => Self::Q8_0,
            KvCacheTypeArg::Q4_0 => Self::Q4_0,
        }
    }
}

/// What `rocm diagnose --model` was asked about.
pub(crate) struct ModelQuery {
    pub(crate) model_ref: String,
    pub(crate) context: ContextRequest,
}

impl ModelQuery {
    /// The question `--model` asks, with `--context-size` and `--kv-cache-type`
    /// as given (clap refuses either without `--model`).
    pub(crate) fn new(
        model_ref: String,
        context_size: Option<u64>,
        kv_cache_type: Option<KvCacheTypeArg>,
    ) -> Self {
        Self {
            model_ref,
            context: ContextRequest {
                context_size,
                kv_cache_type: kv_cache_type.map(Into::into),
            },
        }
    }
}

/// What `--report` says to stderr when it was given a model question: the report
/// is about this machine, so the model flags did nothing, and nothing about the
/// model was fetched. A flag that is accepted and silently has no effect reads as
/// an answer that was never asked for.
pub(crate) fn report_ignores_model_note(query: &ModelQuery) -> String {
    let mut ignored = vec!["--model"];
    if query.context.context_size.is_some() {
        ignored.push("--context-size");
    }
    if query.context.kv_cache_type.is_some() {
        ignored.push("--kv-cache-type");
    }
    let verb = if ignored.len() == 1 { "was" } else { "were" };
    format!(
        "note: --report describes this machine, not a model, so {} {verb} ignored and nothing \
         about the model was fetched",
        ignored.join(", ")
    )
}

/// Add the GGUF estimate to a readiness answer, and the mask caveat that goes
/// with it.
///
/// `vram` is the `amd-smi` reading the host's memory was classified from, passed
/// in rather than read again: each read is a subprocess.
pub(crate) fn attach_gguf_estimate(
    readiness: &mut ModelReadiness,
    context: &ContextRequest,
    host: &HostFacts,
    gguf_source: &dyn GgufSource,
    vram: Option<&[GpuVramUsage]>,
) {
    // The one step that can touch a file or the network, and only for a GGUF
    // reference the catalog does not carry; it reads the header, not the weights.
    attach_context_estimate(readiness, context, host, gguf_source);
    // Which card a device-visibility mask selects, in `amd-smi`'s numbering, has
    // not been verified on a multi-GPU machine, so an estimate that compared
    // memory says so when it could have compared the wrong card.
    note_mask_for_rows(readiness, vram, device_visibility_mask_is_set());
}

/// [`note_unverified_mask`] over the rows `amd-smi` reported.
fn note_mask_for_rows(
    readiness: &mut ModelReadiness,
    vram: Option<&[GpuVramUsage]>,
    mask_set: bool,
) {
    let totals: Vec<u64> = vram
        .iter()
        .flat_map(|rows| rows.iter())
        .map(|usage| usage.total_mb)
        .collect();
    note_unverified_mask(readiness, mask_set, &totals);
}

/// Whether any of the variables that hide GPUs from the runtime is set, even to
/// nothing (which hides every one).
///
/// Wider than the mask [`rocm_core::usable_amd_gpu_indices`] applies, which
/// reads `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` only: the HIP runtime
/// also honours `CUDA_VISIBLE_DEVICES`, which the memory classification does
/// not, so a figure compared under it is just as open to doubt.
fn device_visibility_mask_is_set() -> bool {
    [
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ]
    .iter()
    .any(|name| std::env::var_os(name).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::gguf_fit::{ContextEstimate, FitRequest};
    use rocm_core::model_readiness::{AcceleratorMemory, ModelCatalogSource};

    /// A readiness whose estimate compared memory, built without a network or a file.
    fn readiness_with_a_compared_estimate() -> ModelReadiness {
        let mut readiness = crate::model_fit::assess_model_for_host(
            "some-lab/never-curated-7b:Q4_K_M",
            &ModelCatalogSource::Unreachable {
                detail: "not needed here".to_owned(),
            },
            &HostFacts {
                accelerator_memory: AcceleratorMemory::Dedicated(24.0),
                system_ram_gib: None,
            },
            None,
            None,
        );
        let mut estimate = ContextEstimate::unreadable(
            "x.gguf",
            &FitRequest {
                context_size: 4096,
                context_size_defaulted: true,
                kv_cache_type: KvCacheType::F16,
            },
            &anyhow::anyhow!("not read"),
        );
        estimate.available_gib = Some(24.0);
        readiness.estimate = Some(estimate);
        readiness
    }

    fn gpu_row(index: u32, total_gib: u64) -> GpuVramUsage {
        GpuVramUsage {
            index,
            used_mb: 0,
            total_mb: total_gib * 1024,
        }
    }

    /// `--report` ignores the model flags it was given, and says which, so a flag
    /// that did nothing is not taken for one that was answered.
    #[test]
    fn the_report_note_names_exactly_the_model_flags_that_were_ignored() {
        let note = |context_size: Option<u64>, kv: Option<KvCacheTypeArg>| {
            report_ignores_model_note(&ModelQuery::new("o/r:Q4_K_M".to_owned(), context_size, kv))
        };
        let alone = note(None, None);
        assert!(alone.contains("--model was ignored"), "{alone}");
        assert!(!alone.contains("--context-size"), "{alone}");
        assert!(!alone.contains("--kv-cache-type"), "{alone}");
        let both = note(Some(4096), Some(KvCacheTypeArg::Q8_0));
        assert!(
            both.contains("--model, --context-size, --kv-cache-type were ignored"),
            "{both}"
        );
        let size_only = note(Some(4096), None);
        assert!(
            size_only.contains("--model, --context-size were ignored")
                && !size_only.contains("--kv-cache-type"),
            "{size_only}"
        );
        assert!(
            both.contains("nothing about the model was fetched"),
            "{both}"
        );
    }

    /// The flags reach the request as given.
    #[test]
    fn the_query_carries_the_flags_as_given() {
        let query = ModelQuery::new(
            "o/r:Q4_K_M".to_owned(),
            Some(8192),
            Some(KvCacheTypeArg::Q4_0),
        );
        assert_eq!(query.model_ref, "o/r:Q4_K_M");
        assert_eq!(
            query.context,
            ContextRequest {
                context_size: Some(8192),
                kv_cache_type: Some(KvCacheType::Q4_0),
            }
        );
        let unasked = ModelQuery::new("o/r:Q4_K_M".to_owned(), None, None);
        assert_eq!(unasked.context, ContextRequest::default());
    }

    /// The glue between `amd-smi`'s rows and the caveat: cards of two sizes and a
    /// mask say it, and every other host does not.
    #[test]
    fn the_unverified_mask_note_reaches_an_estimate_when_the_gpus_differ() {
        let noted = |rows: Option<&[GpuVramUsage]>, mask_set: bool| {
            let mut readiness = readiness_with_a_compared_estimate();
            note_mask_for_rows(&mut readiness, rows, mask_set);
            readiness
                .estimate
                .unwrap()
                .notes
                .iter()
                .any(|note| note.contains("has not been verified"))
        };
        let mixed = [gpu_row(0, 24), gpu_row(1, 8)];
        let same = [gpu_row(0, 24), gpu_row(1, 24)];
        assert!(noted(Some(&mixed), true));
        assert!(!noted(Some(&mixed), false), "no mask, nothing to doubt");
        assert!(!noted(Some(&same), true), "the same card either way");
        assert!(!noted(None, true), "no rows, no figure compared");
    }

    // Every spelling the flag takes maps to the type the estimate uses, so a
    // mismatch between the two enums cannot silently estimate another cache.
    #[test]
    fn every_kv_cache_flag_value_names_the_matching_estimate_type() {
        for arg in KvCacheTypeArg::value_variants() {
            let name = arg
                .to_possible_value()
                .expect("named")
                .get_name()
                .to_owned();
            let kind = KvCacheType::from(*arg);
            assert_eq!(kind.name(), name);
        }
    }
}
