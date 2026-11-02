// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Will this GGUF model fit in GPU memory at this context size?
//!
//! Weights plus KV cache plus a compute buffer plus a runtime reserve, compared
//! against the dedicated VRAM the readiness check already reads. Pure
//! arithmetic over a [`GgufHeader`]: nothing here reads a file or the network.
//!
//! The KV-cache formula is exact for a plain full-attention model (it is how
//! llama.cpp lays the cache out), and the weights are the file's own size from
//! its tensor offsets, less the input embedding, which llama.cpp keeps in host
//! memory. Only two allowances are **calibrated against real loads on one GPU and
//! one engine build**: the compute buffer and the runtime overhead. The loads were
//! on an MI300X (gfx942) with llama.cpp b9752 on ROCm 7.2, flash attention on, two
//! dense models (a 32-layer Llama-shaped Q4_K_M and a 28-layer Qwen2-shaped Q8_0),
//! each at several context sizes and KV-cache types (fifteen loads), and one more
//! load of the Llama-shaped model made with the card almost full. In fifteen of
//! the sixteen the total came out *above* what the GPU's used memory grew by: by
//! 0.05% to 2.9% with an f16 KV cache and up to 4.5% with a quantized one, where
//! the dequantization allowance is deliberately generous. In the sixteenth it was
//! 5.5% above, because the process overhead measured there was about 0.4 GiB
//! against 1.05-1.28 GiB in the others, for a reason that is not known. Across
//! all sixteen it was 0.05% to 5.5% above and never below. That is the same data
//! the allowances were fitted to, so it measures the fit, not a guarantee:
//! another engine build, another GPU, a mixture-of-experts model or a different
//! batch size can miss by more -- in either direction for a state-space model,
//! whose recurrent state is not counted. That is why the answer is labelled an
//! *estimate* everywhere it is shown, and why layouts the formula does not
//! describe (latent attention, split files, models with no attention layers) are
//! reported as [`EstimateOutcome::CannotEstimate`] rather than guessed at.

use crate::gguf::{GgufHeader, MAX_BLOCK_COUNT};
use crate::model_readiness::{AcceleratorMemory, format_gib};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Memory the engine and the GPU runtime hold beyond the weights, the KV cache
/// and the compute buffer (context objects, loaded kernels, graph metadata).
/// Measured at 1050-1101 MiB for the Llama-shaped model and 1228-1277 MiB for the
/// Qwen2-shaped one on the build named in the module docs; 1.25 GiB covers both.
const RUNTIME_RESERVE_BYTES: f64 = 1280.0 * 1024.0 * 1024.0;

/// llama.cpp's default physical batch size, which sizes its compute buffers.
const COMPUTE_BATCH_TOKENS: f64 = 512.0;

/// Bytes of the attention mask per token of context and batch token: a 16-bit
/// float each.
const MASK_BYTES: f64 = 2.0;

/// The largest context size an estimate is made for: 2^24 tokens.
///
/// Far beyond any model's, and small enough that no figure derived from it can
/// overflow. The CLI refuses a larger request outright, so the report never shows
/// a number a context that size would produce.
pub const MAX_CONTEXT_SIZE: u64 = 1 << 24;

/// No host has this many bytes of memory, so an estimate that comes to more is
/// not a model any host could load: a corrupt header's head counts or widths.
const MAX_PLAUSIBLE_BYTES: f64 = (1_u64 << 60) as f64;

const MIB: u64 = 1024 * 1024;

/// Reported context sizes are rounded down to a multiple of this, so a figure
/// is a number a user would type rather than the exact byte boundary.
const CONTEXT_STEP: u64 = 256;

/// The range of bits per weight a real GGUF falls in: a 1-bit-ish quantization
/// at the bottom, 32-bit floats at the top.
const MIN_PLAUSIBLE_BITS_PER_WEIGHT: f64 = 1.0;
const MAX_PLAUSIBLE_BITS_PER_WEIGHT: f64 = 33.0;

/// A smaller quantization is only suggested when it is meaningfully smaller than
/// the current one; two files within this factor of each other are the same
/// quantization for this purpose.
const QUANT_SMALLER_FACTOR: f64 = 0.95;

/// Typical bits per weight of common llama.cpp quantizations, averaged over a
/// whole model. Approximate: the real figure depends on the model's tensor mix.
const QUANT_BITS_PER_WEIGHT: [(&str, f64); 7] = [
    ("Q2_K", 2.63),
    ("Q3_K_M", 3.91),
    ("IQ4_XS", 4.25),
    ("Q4_K_M", 4.85),
    ("Q5_K_M", 5.69),
    ("Q6_K", 6.56),
    ("Q8_0", 8.5),
];

/// The element type of the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheType {
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "q8_0")]
    Q8_0,
    #[serde(rename = "q4_0")]
    Q4_0,
}

impl KvCacheType {
    /// Every type, widest first.
    pub const ALL: [Self; 3] = [Self::F16, Self::Q8_0, Self::Q4_0];

    /// Bytes per cached element. The quantized types store a 16-bit scale per
    /// block of 32 values: 34 and 18 bytes per block.
    #[must_use]
    pub const fn bytes_per_element(self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8_0 => 34.0 / 32.0,
            Self::Q4_0 => 18.0 / 32.0,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }
}

impl fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for KvCacheType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| format!("`{value}` is not a KV cache type (f16, q8_0 or q4_0)"))
    }
}

/// What the estimate concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateOutcome {
    /// The estimated total is within the GPU's dedicated memory. The total
    /// compared is the sum of the parts as the report shows them, each in whole
    /// MiB rounded up, so it can be a few MiB above `total_bytes`.
    Fits,
    /// The estimated total exceeds the GPU's dedicated memory.
    DoesNotFit,
    /// No comparison could be made. See [`ContextEstimate::cannot_estimate_reason`].
    CannotEstimate,
}

/// Why no comparison could be made.
///
/// The first group is about the file, the second about the host; the figures
/// the file allows are still reported in the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CannotEstimateReason {
    /// The GGUF header could not be fetched or read.
    HeaderUnreadable,
    /// The file is one shard of a split model; the metadata and the weights are
    /// spread over several files.
    SplitFile,
    /// Latent attention (the KV cache is not heads times head width).
    LatentAttention,
    /// No layer keeps a KV cache.
    NoAttentionLayers,
    /// A figure the formula needs is not in the header.
    MissingMetadata,
    /// No GPU is visible to ROCm.
    NoGpu,
    /// The GPU's memory could not be measured.
    AcceleratorMemoryUnknown,
    /// An APU: the pool the engine allocates from is not readable.
    UnifiedMemoryUnreadable,
}

/// What kind of change a [`FitSuggestion`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionKind {
    SmallerContext,
    KvCacheType,
    SmallerQuant,
    CpuOffload,
}

/// One thing that would make a model that does not fit, fit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitSuggestion {
    pub kind: SuggestionKind,
    pub text: String,
}

/// The estimate for one model at one context size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextEstimate {
    /// What the header was read from: a path, or `owner/repo` and file name.
    pub source: String,
    pub outcome: EstimateOutcome,
    pub cannot_estimate_reason: Option<CannotEstimateReason>,
    /// The specific cause, in words, when `cannot_estimate_reason` is set.
    pub detail: Option<String>,
    pub architecture: Option<String>,
    pub layers: Option<u64>,
    /// The context size the figures below are for.
    pub context_size: u64,
    /// Set when the user did not ask for a context size and this is the default.
    pub context_size_defaulted: bool,
    pub kv_cache_type: KvCacheType,
    pub weights_gib: Option<f64>,
    pub kv_cache_gib: Option<f64>,
    pub compute_buffer_gib: Option<f64>,
    pub runtime_reserve_gib: Option<f64>,
    pub total_gib: Option<f64>,
    /// The same figures in bytes (rounded to the nearest byte), for comparing each
    /// component against a measured load without unit conversion. The `*_gib`
    /// fields above are these divided by 2^30.
    pub weights_bytes: Option<u64>,
    pub kv_cache_bytes: Option<u64>,
    pub compute_buffer_bytes: Option<u64>,
    pub runtime_reserve_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    /// KV cache bytes added by each token of context at `kv_cache_type`.
    pub kv_cache_bytes_per_token: Option<f64>,
    /// Dedicated VRAM, when it was measured.
    pub available_gib: Option<f64>,
    /// The largest context that fits at `kv_cache_type`: a multiple of 256, or
    /// the context the model was trained for when that fits and is the limit.
    /// `None` when not even one 256-token step fits, or when nothing was
    /// compared (the GPU's memory was not measured). It can sit below
    /// `context_size` for an outcome of `fits`: a request smaller than one step,
    /// or beyond the trained context, fits without being a step the ceiling
    /// counts.
    pub max_context: Option<u64>,
    /// The context length the model was trained for.
    pub training_context: Option<u64>,
    /// Set for a sliding-window model, whose KV cache is sized as if every layer
    /// attended over the whole context: `max_context` is then a conservative
    /// figure, not the exact boundary.
    #[serde(default)]
    pub conservative_ceiling: bool,
    /// What would make it fit; empty unless the outcome is `does_not_fit`.
    pub suggestions: Vec<FitSuggestion>,
    pub notes: Vec<String>,
}

/// What was asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitRequest {
    pub context_size: u64,
    pub context_size_defaulted: bool,
    pub kv_cache_type: KvCacheType,
}

impl ContextEstimate {
    /// An estimate that could not be made because the header could not be
    /// reached or read. Carries no figure: nothing was learnt about the model.
    #[must_use]
    pub fn unreadable(source: &str, request: &FitRequest, error: &anyhow::Error) -> Self {
        Self {
            source: source.to_owned(),
            outcome: EstimateOutcome::CannotEstimate,
            cannot_estimate_reason: Some(CannotEstimateReason::HeaderUnreadable),
            detail: Some(format!("the GGUF header could not be read: {error:#}")),
            architecture: None,
            layers: None,
            context_size: request.context_size,
            context_size_defaulted: request.context_size_defaulted,
            kv_cache_type: request.kv_cache_type,
            weights_gib: None,
            kv_cache_gib: None,
            compute_buffer_gib: None,
            runtime_reserve_gib: None,
            total_gib: None,
            weights_bytes: None,
            kv_cache_bytes: None,
            compute_buffer_bytes: None,
            runtime_reserve_bytes: None,
            total_bytes: None,
            kv_cache_bytes_per_token: None,
            available_gib: None,
            max_context: None,
            training_context: None,
            conservative_ceiling: false,
            suggestions: Vec::new(),
            notes: Vec::new(),
        }
    }
}

/// The pieces of the total that do not depend on the context size, plus the
/// KV-cache cost per token.
struct Sizing {
    /// Everything after the header: what a bits-per-weight figure is over.
    data: f64,
    /// The part of `data` that lands in VRAM. llama.cpp keeps the input
    /// embedding (`token_embd.weight`) in host memory, so it is left out.
    weights: f64,
    /// The compute buffer's size before anything that grows with the context.
    compute_base: f64,
    /// KV elements cached per token across all layers.
    kv_elements_per_token: f64,
    /// KV elements per token in the widest single layer.
    widest_layer_kv_elements: f64,
    layers: u64,
}

impl Sizing {
    fn kv_bytes(&self, context: u64, kv: KvCacheType) -> f64 {
        self.kv_elements_per_token * context as f64 * kv.bytes_per_element()
    }

    /// What one more token of context costs in the compute buffer. The attention
    /// mask grows with it. A quantized KV cache is also dequantized to 16-bit
    /// floats a layer at a time, which the measured loads showed adding up to
    /// that layer's K and V width per token; where the fixed part is larger than
    /// that it hides it, so this errs high for a small model.
    fn compute_bytes_per_token(&self, kv: KvCacheType) -> f64 {
        let mask = COMPUTE_BATCH_TOKENS * MASK_BYTES;
        if kv == KvCacheType::F16 {
            mask
        } else {
            self.widest_layer_kv_elements.mul_add(2.0, mask)
        }
    }

    fn compute_bytes(&self, context: u64, kv: KvCacheType) -> f64 {
        self.compute_bytes_per_token(kv)
            .mul_add(context as f64, self.compute_base)
    }

    /// Bytes that do not depend on the context size.
    fn fixed(&self) -> f64 {
        self.weights + self.compute_base + RUNTIME_RESERVE_BYTES
    }

    /// Bytes each token of context adds: its KV cache and its compute buffer.
    fn bytes_per_token(&self, kv: KvCacheType) -> f64 {
        self.kv_elements_per_token
            .mul_add(kv.bytes_per_element(), self.compute_bytes_per_token(kv))
    }

    fn total(&self, context: u64, kv: KvCacheType) -> f64 {
        self.bytes_per_token(kv)
            .mul_add(context as f64, self.fixed())
    }

    /// The total the report shows for `weights` of weights at `context` and
    /// `kv`: its four parts, each as printed (whole MiB, rounded up), summed.
    ///
    /// This is the one figure a fit is decided on. The verdict, the largest
    /// context and every "which fits" suggestion go through [`Self::fits`], so
    /// none of them can disagree about whether a given context fits.
    fn shown_total_with(&self, weights: f64, context: u64, kv: KvCacheType) -> f64 {
        shown_sum([
            weights,
            self.kv_bytes(context, kv),
            self.compute_bytes(context, kv),
            RUNTIME_RESERVE_BYTES,
        ])
    }

    fn shown_total(&self, context: u64, kv: KvCacheType) -> f64 {
        self.shown_total_with(self.weights, context, kv)
    }

    fn fits(&self, context: u64, kv: KvCacheType, available: f64) -> bool {
        self.shown_total(context, kv) <= available
    }

    /// Whether the GPU holds `on_gpu` of the layers at `context` and `kv`, by the
    /// same measure as [`Self::fits`]: the layers' share of the weights and the
    /// cache, plus the compute buffer and the reserve, as the report would show
    /// them. At every layer this is `fits`.
    fn layers_fit(&self, on_gpu: u64, context: u64, kv: KvCacheType, available: f64) -> bool {
        let share = on_gpu as f64 / self.layers as f64;
        shown_sum([
            self.weights * share,
            self.kv_bytes(context, kv) * share,
            self.compute_bytes(context, kv),
            RUNTIME_RESERVE_BYTES,
        ]) <= available
    }

    /// The part of a `bits_per_weight` quantization of `params` weights that
    /// lands in VRAM: the same share of the file as now, because the input
    /// embedding stays in host memory at any quantization.
    fn quant_weights(&self, params: f64, bits_per_weight: f64) -> f64 {
        params * bits_per_weight / 8.0 * (self.weights / self.data)
    }

    /// The largest context that fits in `available` bytes at `kv`, in steps of
    /// [`CONTEXT_STEP`] and no larger than `training` (the trained context, as
    /// [`trained_context`] gives it, so never zero): the trained context itself
    /// where it fits, else the largest step for which [`Self::fits`] holds.
    fn max_context(&self, available: f64, kv: KvCacheType, training: Option<u64>) -> Option<u64> {
        let room = available - self.fixed();
        if room <= 0.0 {
            return None;
        }
        let per_token = self.bytes_per_token(kv);
        let raw = (room / per_token).floor();
        // `as` saturates; a context beyond u64 is a model with no KV cost, which
        // `sizing` has already refused.
        let cap = training;
        // Where the estimate reaches the cap, the search starts at the cap
        // itself, whole step or not: rounding first would start a cap of 1500
        // at 1280, and the climb below never passes the cap to reach it.
        let mut context = cap
            .filter(|cap| raw as u64 >= *cap)
            .unwrap_or((raw as u64) / CONTEXT_STEP * CONTEXT_STEP);
        // The estimate above is on the exact total; what is decided on is the
        // shown one, a few MiB higher. Step down until it fits, then up while the
        // next step still does, so the answer is exactly the predicate's.
        // A cap that is not a whole number of steps (a trained context of 250, or
        // of 600) is where this starts, so the first step down lands on a whole
        // step and a start below one step goes to zero rather than below it.
        while context > 0 && !self.fits(context, kv, available) {
            context = if context.is_multiple_of(CONTEXT_STEP) {
                context - CONTEXT_STEP
            } else {
                context / CONTEXT_STEP * CONTEXT_STEP
            };
        }
        while cap.is_none_or(|cap| context + CONTEXT_STEP <= cap)
            && self.fits(context + CONTEXT_STEP, kv, available)
        {
            context += CONTEXT_STEP;
        }
        (context > 0).then_some(context)
    }
}

/// The context length the model was trained for, when the header says one. A
/// header that declares zero declares none: no model was trained for no tokens,
/// and a cap of zero would leave no context that fits and a note that every
/// request is "beyond the 0 tokens the model was trained for".
fn trained_context(header: &GgufHeader) -> Option<u64> {
    header.context_length.filter(|trained| *trained > 0)
}

fn cannot(
    reason: CannotEstimateReason,
    detail: impl Into<String>,
) -> (CannotEstimateReason, String) {
    (reason, detail.into())
}

/// Work out the sizing, or why it cannot be worked out.
fn sizing(header: &GgufHeader, file_size: u64) -> Result<Sizing, (CannotEstimateReason, String)> {
    if header.split_count.is_some_and(|count| count > 1) {
        return Err(cannot(
            CannotEstimateReason::SplitFile,
            "this file is one shard of a split model; its weights and metadata are spread over \
             several files, so one file's size is not the model's",
        ));
    }
    if header.latent_attention {
        return Err(cannot(
            CannotEstimateReason::LatentAttention,
            "this model uses latent attention, whose KV cache is not laid out as heads times \
             head width, so the estimate's formula does not describe it",
        ));
    }
    let missing = |what: &str| {
        cannot(
            CannotEstimateReason::MissingMetadata,
            format!("the GGUF header does not carry {what}"),
        )
    };
    let layers = header
        .block_count
        .filter(|layers| *layers > 0)
        .ok_or_else(|| missing("a layer count"))?;
    // The sizing below loops once per layer. The parser refuses an implausible
    // count; a header built some other way gets the same bound here, so no
    // caller can turn a corrupt figure into a loop that runs until memory ends.
    if layers > MAX_BLOCK_COUNT {
        return Err(cannot(
            CannotEstimateReason::MissingMetadata,
            format!("the header's layer count ({layers}) is not a plausible model"),
        ));
    }
    let data = file_size
        .checked_sub(header.data_offset)
        .filter(|w| *w > 0)
        .ok_or_else(|| {
            cannot(
                CannotEstimateReason::MissingMetadata,
                "the file is no larger than its header, so there are no weights to size",
            )
        })?;
    // The input embedding runs from its offset to the next tensor's, or to the
    // end of the data when it is the last. Clamped so a header that disagrees
    // with the file's size cannot make the weights negative.
    let embedding = header.token_embd.map_or(0, |span| {
        span.next_offset
            .unwrap_or(data)
            .min(data)
            .saturating_sub(span.offset)
    });
    let weights = data.saturating_sub(embedding);

    let mut kv_elements = 0_u64;
    let mut widest_layer = 0_u64;
    for layer in 0..usize::try_from(layers).unwrap_or(usize::MAX) {
        let heads = header.head_count.as_ref().and_then(|p| p.at(layer));
        // llama.cpp reads an absent KV head count as "as many as query heads".
        let kv_heads = match &header.head_count_kv {
            Some(per_layer) => per_layer.at(layer),
            None => heads,
        }
        .ok_or_else(|| missing("a KV head count for every layer"))?;
        if kv_heads == 0 {
            continue;
        }
        let width = |explicit: Option<u64>| -> Result<u64, (CannotEstimateReason, String)> {
            if let Some(width) = explicit {
                return Ok(width);
            }
            match (header.embedding_length, heads) {
                (Some(embedding), Some(heads)) if heads > 0 => Ok(embedding / heads),
                _ => Err(missing("the attention head width")),
            }
        };
        let per_head = width(header.key_length)?.saturating_add(width(header.value_length)?);
        let layer_elements = kv_heads.saturating_mul(per_head);
        widest_layer = widest_layer.max(layer_elements);
        kv_elements = kv_elements.saturating_add(layer_elements);
    }
    if kv_elements == 0 {
        return Err(cannot(
            CannotEstimateReason::NoAttentionLayers,
            "no layer of this model keeps a KV cache, so a context-size estimate has nothing to \
             size",
        ));
    }

    // The part of llama.cpp's compute buffer that does not grow with the
    // context: three activations of the widest intermediate width and three of
    // the embedding width per batch token, in 32-bit floats. Flash attention is
    // assumed, so no attention-score matrix appears. This reproduced the measured
    // buffer exactly for both calibration models.
    let activations = 3.0_f64.mul_add(
        header.feed_forward_length.unwrap_or(0) as f64,
        3.0 * header.embedding_length.unwrap_or(0) as f64,
    );
    let compute_base = COMPUTE_BATCH_TOKENS * activations * 4.0;
    Ok(Sizing {
        data: data as f64,
        weights: weights as f64,
        compute_base,
        kv_elements_per_token: kv_elements as f64,
        widest_layer_kv_elements: widest_layer as f64,
        layers,
    })
}

/// A figure that has to cover a need, so it rounds up.
fn need(bytes: f64) -> String {
    format!("{:.1} GiB", (bytes / GIB * 10.0).ceil() / 10.0)
}

fn gib(bytes: f64) -> f64 {
    bytes / GIB
}

/// "1 token", "4096 tokens".
pub(crate) fn tokens(count: u64) -> String {
    if count == 1 {
        "1 token".to_owned()
    } else {
        format!("{count} tokens")
    }
}

/// A byte count in whole MiB, rounded up: a figure that has to cover a need.
const fn mib_up(bytes: u64) -> u64 {
    bytes.div_ceil(MIB)
}

/// The sum of four byte figures as the report shows them: each rounded to a
/// whole byte count as the byte fields are, then up to a whole MiB, and added.
fn shown_sum(parts: [f64; 4]) -> f64 {
    let mib: u64 = parts.iter().map(|part| mib_up(part.round() as u64)).sum();
    (mib * MIB) as f64
}

/// The parts of an estimate as the report shows them -- each in whole MiB,
/// rounded up -- and their sum. The verdict is decided on that sum, so the
/// figures printed always add up and never contradict it.
fn shown_parts(estimate: &ContextEstimate) -> Option<([u64; 4], u64)> {
    let parts = [
        mib_up(estimate.weights_bytes?),
        mib_up(estimate.kv_cache_bytes?),
        mib_up(estimate.compute_buffer_bytes?),
        mib_up(estimate.runtime_reserve_bytes?),
    ];
    Some((parts, parts.iter().sum()))
}

/// Estimate whether the model described by `header` fits at `request`.
///
/// `file_size` is the size of the whole file in bytes; the weights are what
/// follows the header. `memory` and `system_ram_gib` are the host's, as the
/// readiness check measured them.
#[must_use]
pub fn estimate_fit(
    header: &GgufHeader,
    file_size: u64,
    source: &str,
    request: &FitRequest,
    memory: AcceleratorMemory,
    system_ram_gib: Option<f64>,
) -> ContextEstimate {
    let mut estimate = ContextEstimate {
        source: source.to_owned(),
        outcome: EstimateOutcome::CannotEstimate,
        cannot_estimate_reason: None,
        detail: None,
        architecture: header.architecture.clone(),
        layers: header.block_count,
        context_size: request.context_size,
        context_size_defaulted: request.context_size_defaulted,
        kv_cache_type: request.kv_cache_type,
        weights_gib: None,
        kv_cache_gib: None,
        compute_buffer_gib: None,
        runtime_reserve_gib: None,
        total_gib: None,
        weights_bytes: None,
        kv_cache_bytes: None,
        compute_buffer_bytes: None,
        runtime_reserve_bytes: None,
        total_bytes: None,
        kv_cache_bytes_per_token: None,
        available_gib: memory.measured_gib(),
        max_context: None,
        training_context: trained_context(header),
        conservative_ceiling: header.sliding_window,
        suggestions: Vec::new(),
        notes: Vec::new(),
    };
    let sizing = match sizing(header, file_size) {
        Ok(sizing) => sizing,
        Err((reason, detail)) => {
            estimate.cannot_estimate_reason = Some(reason);
            estimate.detail = Some(detail);
            return estimate;
        }
    };

    let kv = request.kv_cache_type;
    let total = sizing.total(request.context_size, kv);
    if !total.is_finite() || total >= MAX_PLAUSIBLE_BYTES {
        estimate.cannot_estimate_reason = Some(CannotEstimateReason::MissingMetadata);
        estimate.detail = Some(format!(
            "at {} the figures this header implies come to more than an exabyte, which no model \
             or host has: the header's head counts or widths look corrupt, or the context size \
             is not a real one (the most an estimate is made for is {MAX_CONTEXT_SIZE})",
            tokens(request.context_size)
        ));
        return estimate;
    }
    estimate.weights_gib = Some(gib(sizing.weights));
    estimate.kv_cache_gib = Some(gib(sizing.kv_bytes(request.context_size, kv)));
    let compute = sizing.compute_bytes(request.context_size, kv);
    estimate.compute_buffer_gib = Some(gib(compute));
    estimate.runtime_reserve_gib = Some(gib(RUNTIME_RESERVE_BYTES));
    estimate.total_gib = Some(gib(total));
    let bytes = |value: f64| Some(value.round() as u64);
    estimate.weights_bytes = bytes(sizing.weights);
    estimate.kv_cache_bytes = bytes(sizing.kv_bytes(request.context_size, kv));
    estimate.compute_buffer_bytes = bytes(compute);
    estimate.runtime_reserve_bytes = bytes(RUNTIME_RESERVE_BYTES);
    estimate.total_bytes = bytes(total);
    estimate.kv_cache_bytes_per_token = Some(sizing.kv_elements_per_token * kv.bytes_per_element());

    if header.sliding_window {
        estimate.notes.push(
            "this model uses sliding-window attention; its KV cache is sized as if every layer \
             attended over the full context, so the real figure can be lower and the largest \
             context below is conservative"
                .to_owned(),
        );
    }
    if header.ssm {
        estimate.notes.push(
            "this model has state-space layers: the recurrent state they keep is not counted, so \
             the real figure can be higher; its KV cache is sized for every layer unless the \
             header lists head counts per layer, so that part can be lower"
                .to_owned(),
        );
    }
    if header.moe {
        estimate.notes.push(
            "this is a mixture-of-experts model: the weights are the file's exact size, but the \
             compute buffer is sized from one feed-forward width and was not measured for \
             experts, so the real figure can be higher"
                .to_owned(),
        );
    }
    if header.feed_forward_length.is_none() || header.embedding_length.is_none() {
        estimate.notes.push(
            "the header does not carry both the feed-forward and the embedding width, so the \
             compute buffer's fixed part is understated and the real figure can be higher"
                .to_owned(),
        );
    }
    if let Some(trained) = trained_context(header)
        && request.context_size > trained
    {
        estimate.notes.push(format!(
            "{} is beyond the {} the model was trained for",
            tokens(request.context_size),
            tokens(trained)
        ));
    }

    let Some(available_gib) = memory.measured_gib() else {
        let (reason, detail) = match memory {
            AcceleratorMemory::None => (
                CannotEstimateReason::NoGpu,
                "no GPU is visible to ROCm, so there is no memory to compare against",
            ),
            AcceleratorMemory::UnifiedMemoryUnreadable => (
                CannotEstimateReason::UnifiedMemoryUnreadable,
                "this host has no dedicated VRAM and the CLI cannot yet read the memory pool its \
                 engine allocates from, so there is nothing to compare the figures against",
            ),
            AcceleratorMemory::Unknown | AcceleratorMemory::Dedicated(_) => (
                CannotEstimateReason::AcceleratorMemoryUnknown,
                "the GPU memory on this host could not be read, so there is nothing to compare \
                 the figures against",
            ),
        };
        estimate.cannot_estimate_reason = Some(reason);
        estimate.detail = Some(detail.to_owned());
        return estimate;
    };
    let available = available_gib * GIB;
    estimate.max_context = sizing.max_context(available, kv, trained_context(header));

    // Decided on the sum of the parts as the report shows them (each rounded up
    // to a whole MiB), which is at most a few MiB above the exact total: what is
    // printed then always adds up, and can never sit next to a contradicting
    // verdict. `max_context` and the suggestions use the same predicate.
    if sizing.fits(request.context_size, kv, available) {
        estimate.outcome = EstimateOutcome::Fits;
        return estimate;
    }
    estimate.outcome = EstimateOutcome::DoesNotFit;
    estimate.suggestions = suggestions(
        header,
        &sizing,
        request,
        available,
        estimate.max_context,
        system_ram_gib,
    );
    estimate
}

fn suggestions(
    header: &GgufHeader,
    sizing: &Sizing,
    request: &FitRequest,
    available: f64,
    max_context: Option<u64>,
    system_ram_gib: Option<f64>,
) -> Vec<FitSuggestion> {
    let kv = request.kv_cache_type;
    let context = request.context_size;
    let mut out = Vec::new();

    if let Some(max) = max_context {
        let qualifier = if header.sliding_window {
            " (a conservative figure for this model: its sliding-window layers are sized at the \
             full context)"
        } else {
            ""
        };
        out.push(FitSuggestion {
            kind: SuggestionKind::SmallerContext,
            text: format!(
                "use a context of {} or less (llama-server --ctx-size {max}); that is the largest \
                 that fits with the KV cache at {kv}{qualifier}",
                tokens(max)
            ),
        });
    }

    // A narrower cache is only worth naming when it lowers the need. A quantized
    // cache is also dequantized a layer at a time, which adds to the compute
    // buffer; in a model with one or two cached layers that costs more than the
    // cache saves, and "a q8_0 cache would cut the need to ..." would be false.
    let now = sizing.shown_total(context, kv);
    for smaller in KvCacheType::ALL.into_iter().filter(|other| {
        other.bytes_per_element() < kv.bytes_per_element()
            && sizing.shown_total(context, *other) < now
    }) {
        let total_mib = sizing.shown_total(context, smaller) as u64 / MIB;
        let text = if sizing.fits(context, smaller, available) {
            let largest = sizing
                .max_context(available, smaller, trained_context(header))
                .map_or(String::new(), |max| {
                    format!("; the largest context then is {}", tokens(max))
                });
            format!(
                "use a {smaller} KV cache (llama-server --cache-type-k {smaller} --cache-type-v \
                 {smaller}): about {total_mib} MiB at {}, which fits{largest}",
                tokens(context)
            )
        } else {
            format!(
                "a {smaller} KV cache would cut the need to about {total_mib} MiB, which is still \
                 more than this GPU offers"
            )
        };
        out.push(FitSuggestion {
            kind: SuggestionKind::KvCacheType,
            text,
        });
    }
    if out
        .iter()
        .any(|suggestion| suggestion.kind == SuggestionKind::KvCacheType)
    {
        out.push(FitSuggestion {
            kind: SuggestionKind::KvCacheType,
            text: "a quantized V cache needs flash attention enabled in llama.cpp".to_owned(),
        });
    }

    let params = header.param_count as f64;
    let current_bpw = if header.param_count > 0 {
        sizing.data * 8.0 / params
    } else {
        0.0
    };
    // Bits per weight outside what any real quantization reaches means the
    // header's tensor shapes and the file's size do not describe the same
    // model (a truncated or padded file, or a fixture); a "smaller quant"
    // computed from that would be noise.
    if (MIN_PLAUSIBLE_BITS_PER_WEIGHT..=MAX_PLAUSIBLE_BITS_PER_WEIGHT).contains(&current_bpw) {
        let candidates: Vec<(&str, f64)> = QUANT_BITS_PER_WEIGHT
            .iter()
            .rev()
            .filter(|(_, bpw)| *bpw < current_bpw * QUANT_SMALLER_FACTOR)
            .map(|(label, bpw)| (*label, sizing.quant_weights(params, *bpw)))
            .collect();
        let fitting = candidates
            .iter()
            .find(|(_, weights)| sizing.shown_total_with(*weights, context, kv) <= available);
        let text = match (fitting, candidates.last()) {
            (Some((label, weights)), _) => Some(format!(
                "a smaller quantization of the same model, about {label} size (roughly {} of \
                 weights against {} now), would fit at {} if the repository \
                 publishes one",
                need(*weights),
                need(sizing.weights),
                tokens(context)
            )),
            (None, Some((label, _))) => Some(format!(
                "no smaller quantization in the usual range, down to about {label}, fits at \
                 {} with the KV cache at {kv}",
                tokens(context)
            )),
            (None, None) => None,
        };
        if let Some(text) = text {
            out.push(FitSuggestion {
                kind: SuggestionKind::SmallerQuant,
                text,
            });
        }
    }

    out.extend(offload_suggestion(
        sizing,
        request,
        available,
        system_ram_gib,
    ));
    out
}

/// Partial offload: keep as many layers as fit on the GPU and run the rest from
/// system RAM. Weights are spread evenly over the layers, which ignores the
/// embedding and output tensors; the figure is a rough guide.
fn offload_suggestion(
    sizing: &Sizing,
    request: &FitRequest,
    available: f64,
    system_ram_gib: Option<f64>,
) -> Option<FitSuggestion> {
    let layers = sizing.layers;
    let context = request.context_size;
    let kv = request.kv_cache_type;
    let per_layer = (sizing.weights + sizing.kv_bytes(context, kv)) / layers as f64;
    let compute = sizing.compute_bytes(context, kv);
    let gpu_fits = |n: u64| sizing.layers_fit(n, context, kv, available);
    let room = available - compute - RUNTIME_RESERVE_BYTES;
    let mut on_gpu = if room > 0.0 {
        ((room / per_layer).floor() as u64).min(layers)
    } else {
        0
    };
    while on_gpu > 0 && !gpu_fits(on_gpu) {
        on_gpu -= 1;
    }
    while on_gpu < layers && gpu_fits(on_gpu + 1) {
        on_gpu += 1;
    }
    if on_gpu == 0 || on_gpu >= layers {
        return None;
    }
    let host_bytes = per_layer * (layers - on_gpu) as f64;
    if system_ram_gib.is_some_and(|ram| host_bytes > ram * GIB) {
        return None;
    }
    let ram_note = if system_ram_gib.is_some() {
        ""
    } else {
        " (system RAM could not be read, so that is unchecked)"
    };
    Some(FitSuggestion {
        kind: SuggestionKind::CpuOffload,
        text: format!(
            "keep about {on_gpu} of {layers} layers on the GPU (llama-server --n-gpu-layers \
             {on_gpu}); the other {} of weights and cache stay in system RAM{ram_note}, and \
             generation is slower",
            need(host_bytes)
        ),
    })
}

/// The evidence lines for an estimate, in the report's own voice.
#[must_use]
pub fn evidence_lines(estimate: &ContextEstimate) -> Vec<String> {
    let mut lines = Vec::new();
    // Where the figures came from is only worth saying when there was a header
    // to read them from; for an unreadable one the detail says what happened.
    if estimate.cannot_estimate_reason != Some(CannotEstimateReason::HeaderUnreadable) {
        lines.push(format!(
            "estimated from the GGUF header of {} ({} architecture, {} layers); the weights were \
             not downloaded",
            estimate.source,
            estimate.architecture.as_deref().unwrap_or("unknown"),
            estimate
                .layers
                .map_or_else(|| "unknown".to_owned(), |layers| layers.to_string()),
        ));
    }
    if let Some(detail) = &estimate.detail {
        lines.push(detail.clone());
    }
    if let Some(([weights, kv, compute, reserve], total)) = shown_parts(estimate) {
        lines.push(format!(
            "at {}{}: weights {weights} MiB + KV cache ({}) {kv} MiB + compute buffer {compute} \
             MiB + runtime reserve {reserve} MiB = about {total} MiB ({})",
            tokens(estimate.context_size),
            if estimate.context_size_defaulted {
                " (the default; pass --context-size to change it)"
            } else {
                ""
            },
            estimate.kv_cache_type,
            need((total * MIB) as f64),
        ));
    }
    if let Some(available) = estimate.available_gib {
        // Whole MiB, rounded down: a host never reads as offering more than it
        // does, and an integer figure sits beside the integer figures above.
        let offered = available.mul_add(1024.0, 1e-6).floor();
        lines.push(format!(
            "the GPU with the most dedicated VRAM that ROCm can see has {offered} MiB ({}) in \
             total; memory other processes already hold is not subtracted",
            format_gib(available)
        ));
        let conservative = if estimate.conservative_ceiling {
            " (conservative for this model: its sliding-window layers are sized at the full \
             context)"
        } else {
            ""
        };
        match (estimate.outcome, estimate.max_context) {
            (EstimateOutcome::CannotEstimate, _) => {}
            // A request that fits is never followed by a ceiling below it: that
            // would read as the request not fitting. When the request is smaller
            // than one step or beyond the trained context, the ceiling is left
            // out and the trained-context note says what applies.
            (EstimateOutcome::Fits, Some(max)) if max < estimate.context_size => {}
            (EstimateOutcome::Fits, None) => {}
            (_, Some(max)) => lines.push(format!(
                "largest context that fits, with the KV cache at {}: {}{conservative}",
                estimate.kv_cache_type,
                tokens(max)
            )),
            (_, None) => lines.push(format!(
                "not even a {CONTEXT_STEP}-token context fits with the KV cache at {}",
                estimate.kv_cache_type
            )),
        }
    }
    lines
}

/// The standing caveat that goes with every estimate that has figures.
pub const ESTIMATE_CAVEAT: &str = "these are estimates from the file's metadata, with the compute buffer and runtime \
     overhead calibrated on one GPU and engine build (an MI300X running llama.cpp b9752) with \
     flash attention on, which is assumed here (without it llama.cpp also allocates an \
     attention-score matrix that is not counted, and the figure is too low); another build or GPU can differ, so leave some headroom \
     and check with a real run";

/// The standing note about `rocm serve`: the options an estimate talks about are
/// the engine's, and `rocm serve` does not pass them through today.
pub const SERVE_OPTIONS_NOTE: &str = "`rocm serve` does not take a context size, KV cache type or \
     GPU layer count yet; the options named above are llama.cpp's own llama-server flags";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_builder::{Builder, llama_like};
    use crate::gguf::{GgufHeader, parse_gguf_header};
    use std::io::Cursor;

    const MIB: u64 = 1024 * 1024;
    const GIB_U: u64 = 1024 * MIB;

    fn header_of(builder: &Builder, data: usize) -> (GgufHeader, u64) {
        let file = builder.file(data);
        let header = parse_gguf_header(&mut Cursor::new(&file)).unwrap();
        (header, file.len() as u64)
    }

    /// 32 layers, 8 KV heads, key and value width 128 each: 32 * 8 * 256 = 65536
    /// elements per token, so 128 KiB per token at f16 and exactly 1 GiB at 8192.
    ///
    /// `data` is the size of the weights. The file is never built: the header is
    /// parsed from a header-only file and the size is declared, so a test of a
    /// multi-gigabyte model allocates nothing.
    fn llama(data: usize) -> (GgufHeader, u64) {
        let header = parse_gguf_header(&mut Cursor::new(llama_like(10).file(0))).unwrap();
        let size = header.data_offset + data as u64;
        (header, size)
    }

    fn request(context_size: u64, kv: KvCacheType) -> FitRequest {
        FitRequest {
            context_size,
            context_size_defaulted: false,
            kv_cache_type: kv,
        }
    }

    fn estimate(
        header: &GgufHeader,
        size: u64,
        context: u64,
        kv: KvCacheType,
        memory: AcceleratorMemory,
    ) -> ContextEstimate {
        estimate_fit(
            header,
            size,
            "test.gguf",
            &request(context, kv),
            memory,
            Some(64.0),
        )
    }

    #[test]
    fn the_kv_cache_is_layers_times_kv_heads_times_head_width_times_context() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        let kv = result.kv_cache_gib.unwrap();
        assert!(
            (kv - 1.0).abs() < 1e-9,
            "32 layers x 8 kv heads x 256 x 8192 x 2 B = 1 GiB, got {kv}"
        );
        // The quantized types are the block sizes, not a round factor.
        let q8 = estimate(
            &header,
            size,
            8192,
            KvCacheType::Q8_0,
            AcceleratorMemory::Dedicated(64.0),
        );
        assert!((q8.kv_cache_gib.unwrap() - 34.0 / 64.0).abs() < 1e-9);
        let q4 = estimate(
            &header,
            size,
            8192,
            KvCacheType::Q4_0,
            AcceleratorMemory::Dedicated(64.0),
        );
        assert!((q4.kv_cache_gib.unwrap() - 18.0 / 64.0).abs() < 1e-9);
    }

    /// The vocabulary `llama_like(10)` declares puts its `token_embd.weight` at
    /// the start of the data and the next tensor 10 * 4096 bytes later.
    const EMBD_BYTES: u64 = 10 * 4096;
    /// 512 batch tokens x (3 x ffn 14336 + 3 x embedding 4096) x 4 bytes: the
    /// 108 MiB the Llama-shaped calibration model's compute buffer measured.
    const LLAMA_COMPUTE_BASE: u64 = 512 * (3 * 14336 + 3 * 4096) * 4;

    #[test]
    fn the_total_is_the_sum_of_its_parts() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        // The input embedding stays in host memory, so it is not VRAM.
        let weights = (size - header.data_offset - EMBD_BYTES) as f64 / GIB;
        // The attention mask adds 512 x 2 bytes per token of context.
        let compute = (LLAMA_COMPUTE_BASE + 512 * 2 * 8192) as f64 / GIB;
        assert!((result.weights_gib.unwrap() - weights).abs() < 1e-12);
        assert!((result.compute_buffer_gib.unwrap() - compute).abs() < 1e-12);
        assert!((result.runtime_reserve_gib.unwrap() - 1.25).abs() < 1e-12);
        let expected = weights + 1.0 + compute + 1.25;
        assert!((result.total_gib.unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn the_byte_fields_are_exact_and_add_up_to_the_total() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        // 32 layers x 8 kv heads x (128 + 128) x 2 bytes = 131072 per token.
        assert_eq!(result.kv_cache_bytes_per_token, Some(131_072.0));
        assert_eq!(result.kv_cache_bytes, Some(131_072 * 8192));
        assert_eq!(
            result.weights_bytes,
            Some(size - header.data_offset - EMBD_BYTES)
        );
        assert_eq!(result.runtime_reserve_bytes, Some(1280 * MIB));
        assert_eq!(
            result.compute_buffer_bytes,
            Some(LLAMA_COMPUTE_BASE + 512 * 2 * 8192)
        );
        let parts = result.weights_bytes.unwrap()
            + result.kv_cache_bytes.unwrap()
            + result.compute_buffer_bytes.unwrap()
            + result.runtime_reserve_bytes.unwrap();
        assert_eq!(result.total_bytes, Some(parts));
    }

    #[test]
    fn a_per_layer_head_array_skips_layers_without_a_cache() {
        let builder = Builder::new()
            .string("general.architecture", "hybrid")
            .u32("hybrid.block_count", 4)
            .u32("hybrid.embedding_length", 1024)
            .u32("hybrid.attention.head_count", 8)
            .u32_array("hybrid.attention.head_count_kv", &[0, 2, 0, 2])
            .tensor("t", &[1000]);
        let (header, size) = header_of(&builder, 4096);
        let result = estimate(
            &header,
            size,
            1000,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        // 2 layers x 2 kv heads x (128 + 128) x 1000 tokens x 2 bytes
        let expected = 2.0 * 2.0 * 256.0 * 1000.0 * 2.0 / GIB;
        assert!((result.kv_cache_gib.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn an_absent_kv_head_count_means_as_many_as_query_heads() {
        let builder = Builder::new()
            .string("general.architecture", "mha")
            .u32("mha.block_count", 2)
            .u32("mha.embedding_length", 512)
            .u32("mha.attention.head_count", 4)
            .tensor("t", &[10]);
        let (header, size) = header_of(&builder, 64);
        let result = estimate(
            &header,
            size,
            100,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        // 2 layers x 4 heads x (128 + 128) x 100 x 2 bytes
        let expected = 2.0 * 4.0 * 256.0 * 100.0 * 2.0 / GIB;
        assert!((result.kv_cache_gib.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn explicit_key_and_value_lengths_override_the_derived_head_width() {
        let builder = Builder::new()
            .string("general.architecture", "wide")
            .u32("wide.block_count", 1)
            .u32("wide.embedding_length", 512)
            .u32("wide.attention.head_count", 4)
            .u32("wide.attention.head_count_kv", 1)
            .u32("wide.attention.key_length", 192)
            .u32("wide.attention.value_length", 128)
            .tensor("t", &[10]);
        let (header, size) = header_of(&builder, 64);
        let result = estimate(
            &header,
            size,
            100,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        let expected = (192.0 + 128.0) * 100.0 * 2.0 / GIB;
        assert!((result.kv_cache_gib.unwrap() - expected).abs() < 1e-12);
    }

    /// The verdict is decided on the total the report shows -- the sum of its
    /// parts, each rounded up to a whole MiB -- so a GPU of exactly that many MiB
    /// fits and one MiB less does not, and what is printed always agrees.
    #[test]
    fn fit_is_decided_at_the_total_the_report_shows() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let reference = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        let (_, shown_mib) = shown_parts(&reference).unwrap();
        let mib_gib = |mib: u64| mib as f64 / 1024.0;
        let exact = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(mib_gib(shown_mib)),
        );
        assert_eq!(exact.outcome, EstimateOutcome::Fits);
        let under = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(mib_gib(shown_mib - 1)),
        );
        assert_eq!(under.outcome, EstimateOutcome::DoesNotFit);
        // The exact total is never above what is shown, and never more than a few
        // MiB below it (one rounding per part).
        let exact_bytes = reference.total_bytes.unwrap();
        assert!(exact_bytes <= shown_mib * MIB, "{exact_bytes} {shown_mib}");
        assert!(shown_mib * MIB - exact_bytes < 4 * MIB);
        // Memory that is above the exact total and below the shown one: the
        // verdict follows the figure that is printed, never the unrounded one.
        assert!(
            shown_mib * MIB - exact_bytes >= 2,
            "the total is a whole number of MiB"
        );
        let between = (exact_bytes + shown_mib * MIB) as f64 / 2.0 / GIB;
        let tight = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(between),
        );
        assert_eq!(tight.outcome, EstimateOutcome::DoesNotFit);
    }

    /// The figures printed are the figures decided on: whole MiB, summing to the
    /// total, with the host's memory in the same unit -- so no verdict can sit
    /// beside a figure that contradicts it.
    #[test]
    fn the_printed_parts_add_up_and_never_contradict_the_verdict() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let figures = |lines: &str| -> Vec<u64> {
            let line = lines.lines().find(|line| line.starts_with("at ")).unwrap();
            regex::Regex::new(r"(\d+) MiB")
                .unwrap()
                .captures_iter(line)
                .map(|found| found[1].parse().unwrap())
                .collect()
        };
        // Small and large, fitting and not, f16 and quantized.
        for (context, kv, gpu) in [
            (128, KvCacheType::F16, 24.0),
            (8192, KvCacheType::Q8_0, 24.0),
            (131_072, KvCacheType::F16, 24.0),
            (4096, KvCacheType::F16, 1.0),
        ] {
            let result = estimate(
                &header,
                size,
                context,
                kv,
                AcceleratorMemory::Dedicated(gpu),
            );
            let lines = evidence_lines(&result).join("\n");
            let numbers = figures(&lines);
            assert_eq!(numbers.len(), 5, "{lines}");
            assert_eq!(numbers[..4].iter().sum::<u64>(), numbers[4], "{lines}");
            // The GiB figure beside the MiB total is that same total, rounded up.
            let in_gib = need((numbers[4] * MIB) as f64);
            assert!(
                lines.contains(&format!("= about {} MiB ({in_gib})", numbers[4])),
                "{lines}"
            );
            let offered = (gpu * 1024.0) as u64;
            assert!(lines.contains(&format!("has {offered} MiB")), "{lines}");
            // The verdict is the comparison of the two printed figures.
            assert_eq!(
                result.outcome == EstimateOutcome::Fits,
                numbers[4] <= offered,
                "{lines}"
            );
        }
        // No part of a small model reads as a rounded-away zero.
        let tiny = Builder::new()
            .string("general.architecture", "t")
            .u32("t.block_count", 2)
            .u32("t.embedding_length", 128)
            .u32("t.feed_forward_length", 256)
            .u32("t.attention.head_count", 2)
            .u32("t.attention.head_count_kv", 1)
            .tensor("w", &[1024]);
        let (header, size) = header_of(&tiny, 4096);
        let result = estimate(
            &header,
            size,
            256,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        let lines = evidence_lines(&result).join("\n");
        assert!(
            !lines.contains("0.0 GiB") && lines.contains("weights 1 MiB"),
            "{lines}"
        );
    }

    /// A GPU of exactly `mib` MiB.
    fn gpu_mib(mib: u64) -> AcceleratorMemory {
        AcceleratorMemory::Dedicated(mib as f64 / 1024.0)
    }

    /// The total the report shows for this model at `context`, in whole MiB.
    fn shown_mib(header: &GgufHeader, size: u64, context: u64, kv: KvCacheType) -> u64 {
        shown_parts(&estimate(
            header,
            size,
            context,
            kv,
            AcceleratorMemory::Dedicated(1024.0),
        ))
        .unwrap()
        .1
    }

    #[test]
    fn the_largest_context_is_rounded_down_to_a_step_and_capped_at_training() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let kv = KvCacheType::F16;
        let shown = |context| shown_mib(&header, size, context, kv);
        let max_with = |mib: u64| estimate(&header, size, 100_000, kv, gpu_mib(mib)).max_context;
        // A card with exactly the room the report shows for 24576 tokens.
        assert_eq!(max_with(shown(24576)), Some(24576));
        // A step is 256 tokens, 32 MiB here: room for 100 more tokens (12 MiB)
        // rounds down to the same step.
        assert_eq!(max_with(shown(24576) + 12), Some(24576));
        // One MiB short of the step drops to the one below.
        assert_eq!(max_with(shown(24576) - 1), Some(24320));
        // A huge GPU is capped at the context the model was trained for.
        let result = estimate(
            &header,
            size,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(10_000.0),
        );
        assert_eq!(result.max_context, Some(131_072));
    }

    #[test]
    fn a_gpu_too_small_for_the_weights_has_no_context_that_fits() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(0.5),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        assert_eq!(result.max_context, None);
        assert!(
            !result
                .suggestions
                .iter()
                .any(|s| s.kind == SuggestionKind::SmallerContext),
            "a smaller context was suggested where none can fit: {:?}",
            result.suggestions
        );
    }

    #[test]
    fn a_model_that_fits_has_no_suggestions_and_still_reports_its_ceiling() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(24.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::Fits);
        assert!(result.suggestions.is_empty());
        assert_eq!(result.max_context, Some(131_072));
    }

    // The three memory states the readiness check distinguishes. In each the
    // figures the file allows are still reported, and in none is anything
    // compared.
    #[test]
    fn memory_that_cannot_be_compared_still_reports_what_the_model_needs() {
        let (header, size) = llama(GIB_U as usize / 1024);
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
            let result = estimate(&header, size, 4096, KvCacheType::F16, memory);
            assert_eq!(
                result.outcome,
                EstimateOutcome::CannotEstimate,
                "{memory:?}"
            );
            assert_eq!(result.cannot_estimate_reason, Some(reason));
            assert!(
                result.total_gib.is_some(),
                "{memory:?}: the need was not reported"
            );
            assert_eq!(result.available_gib, None);
            assert_eq!(result.max_context, None);
            assert!(result.suggestions.is_empty());
        }
    }

    #[test]
    fn suggestions_name_a_quantized_kv_cache_only_when_it_would_fit() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let fixed = estimate(
            &header,
            size,
            0,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        )
        .total_gib
        .unwrap();
        // 8192 tokens: f16 needs 1 GiB, q8_0 0.53, q4_0 0.28. Room for 0.6.
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(fixed + 0.6),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        let texts: Vec<&str> = result.suggestions.iter().map(|s| s.text.as_str()).collect();
        assert!(
            texts
                .iter()
                .any(|t| t.contains("q8_0 KV cache") && t.contains("which fits")),
            "{texts:?}"
        );
        assert!(
            texts
                .iter()
                .any(|t| t.contains("q4_0 KV cache") && t.contains("which fits")),
            "{texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("flash attention")),
            "{texts:?}"
        );
        // Room for 0.4: q8_0 no longer fits and must not be offered as if it did.
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(fixed + 0.4),
        );
        let q8 = result
            .suggestions
            .iter()
            .find(|s| s.text.contains("q8_0 KV cache"))
            .unwrap();
        assert!(q8.text.contains("still more than"), "{}", q8.text);
        let q4 = result
            .suggestions
            .iter()
            .find(|s| s.text.contains("q4_0 KV cache"))
            .unwrap();
        assert!(q4.text.contains("which fits"), "{}", q4.text);
    }

    #[test]
    fn an_already_quantized_cache_is_not_offered_itself() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            131_072,
            KvCacheType::Q8_0,
            AcceleratorMemory::Dedicated(4.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        // Only the KV-type suggestions are about the cache's type: the
        // context-size one legitimately names the current type.
        for suggestion in result
            .suggestions
            .iter()
            .filter(|s| s.kind == SuggestionKind::KvCacheType)
        {
            assert!(
                !suggestion.text.contains("q8_0 KV cache")
                    && !suggestion.text.contains("f16 KV cache"),
                "{}",
                suggestion.text
            );
        }
        let none_wider = estimate(
            &header,
            size,
            131_072,
            KvCacheType::Q4_0,
            AcceleratorMemory::Dedicated(4.0),
        );
        assert!(
            !none_wider
                .suggestions
                .iter()
                .any(|s| s.kind == SuggestionKind::KvCacheType),
            "the narrowest cache has no narrower one to suggest"
        );
    }

    /// Weights sized like a 7B-parameter model at ~8.5 bits per weight: the
    /// header declares one large tensor, and the file carries that many bytes.
    fn eight_bit_seven_b() -> (GgufHeader, u64) {
        let params: u64 = 7_000_000_000;
        let builder = llama_like(10).tensor("big", &[params]);
        // A real 7.4 GB buffer is not worth allocating: parse a header-only
        // file and size the "file" from the declared data length.
        let file = builder.file(0);
        let header = parse_gguf_header(&mut Cursor::new(&file)).unwrap();
        let size = header.data_offset + (params as f64 * 8.5 / 8.0) as u64;
        (header, size)
    }

    #[test]
    fn a_smaller_quant_is_suggested_only_when_the_estimate_says_it_fits() {
        let (header, size) = eight_bit_seven_b();
        let quant_of = |gpu_gib: f64| {
            let result = estimate(
                &header,
                size,
                2048,
                KvCacheType::F16,
                AcceleratorMemory::Dedicated(gpu_gib),
            );
            assert_eq!(result.outcome, EstimateOutcome::DoesNotFit, "{gpu_gib} GiB");
            result
                .suggestions
                .into_iter()
                .find(|s| s.kind == SuggestionKind::SmallerQuant)
                .unwrap()
                .text
        };
        // The 8.5 bpw weights are ~6.9 GiB. The suggestion is the LARGEST quant
        // whose weights plus the other ~1.6 GiB (cache, compute buffer, reserve)
        // fit, not the smallest that exists: Q5_K_M (~4.6 GiB) is what 6.5 GiB
        // leaves room for, Q4_K_M (~3.9 GiB) what 6, and Q3_K_M (~3.2 GiB) what 5.
        assert!(quant_of(6.5).contains("Q5_K_M size"), "{}", quant_of(6.5));
        assert!(quant_of(6.0).contains("Q4_K_M size"), "{}", quant_of(6.0));
        assert!(quant_of(5.0).contains("Q3_K_M size"), "{}", quant_of(5.0));
        // A GPU too small for any of them says so instead.
        let text = quant_of(1.0);
        assert!(text.contains("no smaller quantization"), "{text}");
    }

    #[test]
    fn no_quant_is_suggested_when_the_tensor_shapes_and_the_file_size_disagree() {
        // 2 GiB of "weights" for ~17 million declared parameters is about a
        // thousand bits per weight: not a quantization, a mismatched file.
        let (header, size) = llama(2 * GIB_U as usize);
        let result = estimate(
            &header,
            size,
            2048,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(1.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        assert!(
            !result
                .suggestions
                .iter()
                .any(|s| s.kind == SuggestionKind::SmallerQuant),
            "{:?}",
            result.suggestions
        );
    }

    #[test]
    fn no_smaller_quant_is_suggested_for_a_model_already_at_the_bottom_of_the_table() {
        let params: u64 = 7_000_000_000;
        let builder = llama_like(10).tensor("big", &[params]);
        let file = builder.file(0);
        let header = parse_gguf_header(&mut Cursor::new(&file)).unwrap();
        let size = header.data_offset + (params as f64 * 2.5 / 8.0) as u64;
        let result = estimate(
            &header,
            size,
            2048,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(1.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        assert!(
            !result
                .suggestions
                .iter()
                .any(|s| s.kind == SuggestionKind::SmallerQuant)
        );
    }

    #[test]
    fn partial_offload_counts_the_layers_that_fit_and_checks_system_ram() {
        let (header, size) = llama(GIB_U as usize);
        // ~1 GiB weights over 32 layers + 1 GiB KV at 8192: ~64 MiB per layer.
        let fixed = estimate(
            &header,
            size,
            0,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        )
        .total_gib
        .unwrap();
        // Fixed allowances plus room for half the layers (1 GiB per ... 2 GiB total / 2).
        let weights = (size - header.data_offset) as f64 / GIB;
        let per_layer = (weights + 1.0) / 32.0;
        let memory = AcceleratorMemory::Dedicated(fixed - weights + per_layer * 16.5);
        let result = estimate(&header, size, 8192, KvCacheType::F16, memory);
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        let offload = result
            .suggestions
            .iter()
            .find(|s| s.kind == SuggestionKind::CpuOffload)
            .unwrap();
        assert!(offload.text.contains("16 of 32 layers"), "{}", offload.text);
        assert!(
            offload.text.contains("--n-gpu-layers 16"),
            "{}",
            offload.text
        );
        // With almost no system RAM the offloaded half has nowhere to go.
        let starved = estimate_fit(
            &header,
            size,
            "t",
            &request(8192, KvCacheType::F16),
            memory,
            Some(0.01),
        );
        assert!(
            !starved
                .suggestions
                .iter()
                .any(|s| s.kind == SuggestionKind::CpuOffload)
        );
        // Unknown RAM: still offered, and the report says it is unchecked.
        let unknown = estimate_fit(
            &header,
            size,
            "t",
            &request(8192, KvCacheType::F16),
            memory,
            None,
        );
        let offload = unknown
            .suggestions
            .iter()
            .find(|s| s.kind == SuggestionKind::CpuOffload)
            .unwrap();
        assert!(offload.text.contains("unchecked"), "{}", offload.text);
    }

    #[test]
    fn no_offload_is_suggested_when_not_even_one_layer_fits() {
        let (header, size) = llama(GIB_U as usize);
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(0.55),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        assert!(
            !result
                .suggestions
                .iter()
                .any(|s| s.kind == SuggestionKind::CpuOffload)
        );
    }

    fn cannot_estimate(builder: &Builder) -> ContextEstimate {
        let (header, size) = header_of(builder, 4096);
        estimate(
            &header,
            size,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        )
    }

    #[test]
    fn layouts_the_formula_does_not_describe_are_not_guessed_at() {
        let base = || {
            Builder::new()
                .string("general.architecture", "x")
                .u32("x.block_count", 4)
                .u32("x.embedding_length", 512)
                .u32("x.attention.head_count", 8)
                .tensor("t", &[10])
        };
        let split = cannot_estimate(&base().u32("split.count", 3));
        assert_eq!(
            split.cannot_estimate_reason,
            Some(CannotEstimateReason::SplitFile)
        );
        let mla = cannot_estimate(&base().u32("x.attention.kv_lora_rank", 512));
        assert_eq!(
            mla.cannot_estimate_reason,
            Some(CannotEstimateReason::LatentAttention)
        );
        let ssm = cannot_estimate(
            &Builder::new()
                .string("general.architecture", "mamba")
                .u32("mamba.block_count", 4)
                .u32("mamba.ssm.state_size", 16)
                .u32_array("mamba.attention.head_count_kv", &[0, 0, 0, 0])
                .tensor("t", &[10]),
        );
        assert_eq!(
            ssm.cannot_estimate_reason,
            Some(CannotEstimateReason::NoAttentionLayers)
        );
        let no_layers = cannot_estimate(
            &Builder::new()
                .string("general.architecture", "x")
                .tensor("t", &[10]),
        );
        assert_eq!(
            no_layers.cannot_estimate_reason,
            Some(CannotEstimateReason::MissingMetadata)
        );
        let no_heads = cannot_estimate(
            &Builder::new()
                .string("general.architecture", "x")
                .u32("x.block_count", 4)
                .tensor("t", &[10]),
        );
        assert_eq!(
            no_heads.cannot_estimate_reason,
            Some(CannotEstimateReason::MissingMetadata)
        );
        let short_array = cannot_estimate(&base().u32_array("x.attention.head_count_kv", &[1, 1]));
        assert_eq!(
            short_array.cannot_estimate_reason,
            Some(CannotEstimateReason::MissingMetadata)
        );
        for result in [&split, &mla, &ssm, &no_layers, &no_heads, &short_array] {
            assert_eq!(result.outcome, EstimateOutcome::CannotEstimate);
            assert_eq!(
                result.total_gib, None,
                "a figure was invented for an unsupported layout"
            );
            assert!(result.detail.is_some());
        }
    }

    #[test]
    fn a_file_with_no_weights_after_the_header_is_not_sized() {
        let (header, _) = llama(0);
        let result = estimate(
            &header,
            header.data_offset,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        assert_eq!(
            result.cannot_estimate_reason,
            Some(CannotEstimateReason::MissingMetadata)
        );
    }

    #[test]
    fn mixed_attention_and_an_oversized_context_are_noted_not_hidden() {
        let builder = llama_like(10).u32("llama.attention.sliding_window", 4096);
        let (header, size) = header_of(&builder, 4096);
        let result = estimate(
            &header,
            size,
            200_000,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(1000.0),
        );
        assert!(
            result.notes.iter().any(|n| n.contains("full context")),
            "{:?}",
            result.notes
        );
        assert!(
            result.notes.iter().any(|n| n.contains("beyond the 131072")),
            "{:?}",
            result.notes
        );
    }

    /// The notes say which way each attention kind can move the figure: a
    /// sliding-window cache is over-counted, a state-space model's recurrent state
    /// is not counted at all, and neither is hidden behind the other's wording.
    #[test]
    fn each_layout_the_formula_only_approximates_says_which_way_it_can_be_off() {
        let notes_of = |builder: &Builder| {
            let (header, size) = header_of(builder, 4096);
            estimate(
                &header,
                size,
                4096,
                KvCacheType::F16,
                AcceleratorMemory::Dedicated(1000.0),
            )
            .notes
            .join("\n")
        };
        let sliding = notes_of(&llama_like(10).u32("llama.attention.sliding_window", 4096));
        assert!(sliding.contains("can be lower"), "{sliding}");
        assert!(!sliding.contains("can be higher"), "{sliding}");
        let ssm = notes_of(&llama_like(10).u32("llama.ssm.state_size", 16));
        assert!(ssm.contains("recurrent state"), "{ssm}");
        assert!(ssm.contains("can be higher"), "{ssm}");
        let moe = notes_of(&llama_like(10).u32("llama.expert_count", 8));
        assert!(
            moe.contains("mixture-of-experts") && moe.contains("can be higher"),
            "{moe}"
        );
        // A dense model with every width present carries none of them.
        let plain = notes_of(&llama_like(10));
        assert!(plain.is_empty(), "{plain}");
    }

    #[test]
    fn a_header_without_the_widths_the_compute_buffer_needs_says_it_is_understated() {
        let builder = Builder::new()
            .string("general.architecture", "x")
            .u32("x.block_count", 2)
            .u32("x.attention.head_count", 4)
            .u32("x.attention.key_length", 64)
            .u32("x.attention.value_length", 64)
            .tensor("w", &[1024]);
        let (header, size) = header_of(&builder, 4096);
        let result = estimate(
            &header,
            size,
            1024,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::Fits);
        assert!(
            result.notes.iter().any(|note| note.contains("understated")),
            "{:?}",
            result.notes
        );
    }

    #[test]
    fn a_sliding_window_model_is_told_its_largest_context_is_conservative() {
        let builder = llama_like(10).u32("llama.attention.sliding_window", 4096);
        let (header, size) = header_of(&builder, 4096);
        let result = estimate(
            &header,
            size,
            131_072,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        assert!(result.conservative_ceiling);
        let lines = evidence_lines(&result).join("\n");
        assert!(
            lines.contains("largest context that fits")
                && lines.contains("conservative for this model"),
            "{lines}"
        );
        assert!(
            result.suggestions.iter().any(|suggestion| suggestion
                .text
                .contains("conservative figure for this model")),
            "{:?}",
            result.suggestions
        );
        // A dense model's ceiling is exact, and says nothing of the kind.
        let (header, size) = llama(GIB_U as usize);
        let dense = estimate(
            &header,
            size,
            131_072,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert!(!dense.conservative_ceiling);
        assert!(!evidence_lines(&dense).join("\n").contains("conservative"));
    }

    #[test]
    fn a_context_size_too_large_to_be_real_is_refused_not_printed_as_an_absurd_figure() {
        let (header, size) = llama(GIB_U as usize / 1024);
        for context in [u64::MAX, MAX_CONTEXT_SIZE * 1_000_000_000_000] {
            let result = estimate(
                &header,
                size,
                context,
                KvCacheType::F16,
                AcceleratorMemory::Dedicated(64.0),
            );
            assert_eq!(result.outcome, EstimateOutcome::CannotEstimate, "{context}");
            assert_eq!(
                result.cannot_estimate_reason,
                Some(CannotEstimateReason::MissingMetadata)
            );
            // No figure at all, in any unit: text and JSON cannot disagree about a
            // number that is not there.
            assert_eq!(result.total_bytes, None);
            assert_eq!(result.total_gib, None);
            assert_eq!(result.kv_cache_bytes, None);
            assert!(result.detail.unwrap().contains("more than an exabyte"));
        }
        // The cap itself is a real request.
        let at_cap = estimate(
            &header,
            size,
            MAX_CONTEXT_SIZE,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        assert!(at_cap.total_bytes.is_some());
        // Head counts that make the cache absurd are refused as well, whatever the
        // context size: no figure is derived from a corrupt shape.
        let builder = llama_like(10).u32("llama.attention.head_count_kv", u32::MAX);
        let (header, size) = header_of(&builder, 4096);
        let hostile = estimate(
            &header,
            size,
            MAX_CONTEXT_SIZE,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        assert_eq!(hostile.outcome, EstimateOutcome::CannotEstimate);
        assert_eq!(hostile.total_bytes, None);
    }

    #[test]
    fn signed_integer_headers_are_sized_like_unsigned_ones() {
        // gguf-py writes Python integers as signed 32-bit values, so this is the
        // shape of most real files: every figure the estimate reads, as type 5.
        let signed = Builder::new()
            .string("general.architecture", "x")
            .i32("x.block_count", 4)
            .i32("x.embedding_length", 512)
            .i32("x.feed_forward_length", 1024)
            .i32_array("x.attention.head_count", &[8, 8, 8, 8])
            .i32_array("x.attention.head_count_kv", &[0, 2, 0, 2])
            .i32("x.context_length", 8192)
            .tensor("w", &[4096]);
        let unsigned = Builder::new()
            .string("general.architecture", "x")
            .u32("x.block_count", 4)
            .u32("x.embedding_length", 512)
            .u32("x.feed_forward_length", 1024)
            .u32_array("x.attention.head_count", &[8, 8, 8, 8])
            .u32_array("x.attention.head_count_kv", &[0, 2, 0, 2])
            .u32("x.context_length", 8192)
            .tensor("w", &[4096]);
        let size_up = |builder: &Builder| {
            let (header, size) = header_of(builder, 8192);
            estimate(
                &header,
                size,
                2048,
                KvCacheType::F16,
                AcceleratorMemory::Dedicated(8.0),
            )
        };
        let (a, b) = (size_up(&signed), size_up(&unsigned));
        assert_eq!(a.outcome, EstimateOutcome::Fits);
        assert_eq!(a, b);
        // Two layers keep a cache of 2 heads x (64 + 64): 256 elements each.
        assert_eq!(a.kv_cache_bytes_per_token, Some(2.0 * 256.0 * 2.0));
    }

    #[test]
    fn a_single_token_is_not_a_plural() {
        assert_eq!(tokens(1), "1 token");
        assert_eq!(tokens(0), "0 tokens");
        assert_eq!(tokens(4096), "4096 tokens");
        let (header, size) = llama(GIB_U as usize / 1024);
        let one = estimate(
            &header,
            size,
            1,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(24.0),
        );
        let lines = evidence_lines(&one).join("\n");
        assert!(lines.contains("at 1 token:"), "{lines}");
        assert!(!lines.contains("1 tokens"), "{lines}");
    }

    #[test]
    fn the_kv_cache_type_round_trips_through_its_name_and_serde() {
        for kind in KvCacheType::ALL {
            assert_eq!(kind.name().parse::<KvCacheType>(), Ok(kind));
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{}\"", kind.name())
            );
        }
        assert!("q5_0".parse::<KvCacheType>().is_err());
    }

    #[test]
    fn the_evidence_states_where_the_figures_came_from_and_how_big_they_are() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            8192,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(24.0),
        );
        let lines = evidence_lines(&result).join("\n");
        assert!(lines.contains("GGUF header of test.gguf"), "{lines}");
        assert!(lines.contains("llama architecture, 32 layers"), "{lines}");
        assert!(lines.contains("KV cache (f16) 1024 MiB"), "{lines}");
        assert!(
            lines.contains(
                "has 24576 MiB (24 GiB) in total; memory other processes already hold is not \
                 subtracted"
            ),
            "{lines}"
        );
        assert!(
            lines.contains("largest context that fits, with the KV cache at f16: 131072 tokens"),
            "{lines}"
        );
    }

    #[test]
    fn a_request_that_fits_is_never_followed_by_a_ceiling_that_says_otherwise() {
        // Room for a 128-token request and not for a whole 256-token step: it
        // fits, and the report must not go on to say that not even 256 tokens do.
        let (header, size) = llama(GIB_U as usize / 1024);
        let at_128 = estimate(
            &header,
            size,
            128,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        let available = (at_128.total_bytes.unwrap() + 4 * MIB) as f64 / GIB;
        let tight = estimate(
            &header,
            size,
            128,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(available),
        );
        assert_eq!(tight.outcome, EstimateOutcome::Fits);
        assert_eq!(tight.max_context, None, "less than one step of room");
        let lines = evidence_lines(&tight).join("\n");
        assert!(
            !lines.contains("not even") && !lines.contains("largest context"),
            "{lines}"
        );
        // And the same request on a card with no room at all is the opposite.
        let none = estimate(
            &header,
            size,
            128,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(1.0),
        );
        assert_eq!(none.outcome, EstimateOutcome::DoesNotFit);
        assert!(
            evidence_lines(&none)
                .join("\n")
                .contains("not even a 256-token context fits"),
        );
    }

    #[test]
    fn a_request_beyond_the_trained_context_that_fits_is_not_told_a_lower_largest_context() {
        // 262144 tokens fits in 192 GiB; the model was trained for 131072, which
        // caps the ceiling. Printing "largest context that fits: 131072" right
        // after showing 262144 fits would contradict itself.
        let (header, size) = llama(GIB_U as usize / 1024);
        let result = estimate(
            &header,
            size,
            262_144,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(192.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::Fits);
        assert_eq!(result.max_context, Some(131_072));
        let lines = evidence_lines(&result).join("\n");
        assert!(!lines.contains("largest context that fits"), "{lines}");
        assert!(
            result.notes.iter().any(|note| note
                .contains("262144 tokens is beyond the 131072 tokens the model was trained for")),
            "{:?}",
            result.notes
        );
    }

    #[test]
    fn a_layer_count_beyond_the_bound_is_refused_not_looped_over() {
        let (mut header, size) = llama(GIB_U as usize / 1024);
        header.block_count = Some(crate::gguf::MAX_BLOCK_COUNT + 1);
        let result = estimate(
            &header,
            size,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        assert_eq!(result.outcome, EstimateOutcome::CannotEstimate);
        assert_eq!(
            result.cannot_estimate_reason,
            Some(CannotEstimateReason::MissingMetadata)
        );
        assert!(
            result
                .detail
                .as_deref()
                .unwrap_or("")
                .contains("not a plausible model"),
            "{:?}",
            result.detail
        );
        assert_eq!(result.total_gib, None);
    }

    // ---- Calibration against measured loads ----------------------------------
    //
    // The figures below were measured on an MI300X (gfx942) with llama.cpp b9752
    // on ROCm 7.2: `llama-server -fa on -np 1`, the default physical batch of
    // 512, every layer on the GPU. The buffer sizes are what llama.cpp logged;
    // the "used" figure is how much the GPU's used VRAM grew across the load.
    // The headers are synthetic but carry the real shapes, tensor offsets and
    // file sizes of the two models.

    /// Llama-3.1-8B Q4_K_M: 32 layers, 8 KV heads, head width 128, embedding 4096,
    /// feed-forward 14336. 4_912_898_304 bytes of tensor data, of which the
    /// 128256 x 4096 Q4_K `token_embd.weight` is 295_501_824 (llama.cpp logged
    /// 281.81 MiB of it host-mapped and 4403.49 MiB on the GPU).
    fn measured_llama() -> (GgufHeader, u64) {
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32("llama.embedding_length", 4096)
            .u32("llama.feed_forward_length", 14336)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.context_length", 131_072)
            .tensor_at("token_embd.weight", &[4096, 128_256], 0)
            .tensor_at("blk.0.attn_q.weight", &[4096, 4096], 295_501_824)
            .tensor_at("output.weight", &[4096, 128_256], 4_617_396_480);
        let (header, size) = header_of(&builder, 0);
        (header, size + 4_912_898_304)
    }

    /// Qwen2.5-1.5B Q8_0: 28 layers, 2 KV heads, head width 128, embedding 1536,
    /// feed-forward 8960, tied embeddings (no `output.weight`). 1_888_581_632
    /// bytes of tensor data, of which the 151936 x 1536 Q8_0 `token_embd.weight`
    /// is 247_959_552 (236.47 MiB logged host-mapped, 1564.63 MiB on the GPU).
    fn measured_qwen() -> (GgufHeader, u64) {
        let builder = Builder::new()
            .string("general.architecture", "qwen2")
            .u32("qwen2.block_count", 28)
            .u32("qwen2.embedding_length", 1536)
            .u32("qwen2.feed_forward_length", 8960)
            .u32("qwen2.attention.head_count", 12)
            .u32("qwen2.attention.head_count_kv", 2)
            .u32("qwen2.context_length", 32_768)
            .tensor_at("token_embd.weight", &[1536, 151_936], 0)
            .tensor_at("blk.0.attn_q.weight", &[1536, 1536], 247_959_552);
        let (header, size) = header_of(&builder, 0);
        (header, size + 1_888_581_632)
    }

    struct Load {
        context: u64,
        kv: KvCacheType,
        /// llama.cpp's logged GPU KV buffer, MiB.
        kv_mib: f64,
        /// llama.cpp's logged GPU compute buffer, MiB.
        compute_mib: f64,
        /// Growth of the GPU's used VRAM across the load, MiB.
        used_mib: f64,
    }

    const fn load(
        context: u64,
        kv: KvCacheType,
        kv_mib: f64,
        compute_mib: f64,
        used_mib: f64,
    ) -> Load {
        Load {
            context,
            kv,
            kv_mib,
            compute_mib,
            used_mib,
        }
    }

    const LLAMA_LOADS: [Load; 11] = [
        load(4096, KvCacheType::F16, 512.0, 112.01, 6128.0),
        load(32768, KvCacheType::F16, 4096.0, 140.01, 9734.0),
        load(65536, KvCacheType::F16, 8192.0, 172.01, 13869.0),
        load(131_072, KvCacheType::F16, 16384.0, 236.01, 22125.0),
        load(4096, KvCacheType::Q8_0, 272.0, 112.09, 5839.0),
        load(32768, KvCacheType::Q8_0, 2176.0, 192.09, 7819.0),
        load(65536, KvCacheType::Q8_0, 4352.0, 352.09, 10152.0),
        load(131_072, KvCacheType::Q8_0, 8704.0, 672.09, 14830.0),
        load(4096, KvCacheType::Q4_0, 144.0, 112.09, 5711.0),
        load(32768, KvCacheType::Q4_0, 1152.0, 192.09, 6797.0),
        load(131_072, KvCacheType::Q4_0, 4608.0, 672.09, 10734.0),
    ];

    const QWEN_LOADS: [Load; 4] = [
        load(4096, KvCacheType::F16, 112.0, 65.51, 3019.0),
        load(32768, KvCacheType::F16, 896.0, 93.51, 3829.0),
        load(32768, KvCacheType::Q8_0, 476.02, 93.59, 3362.0),
        load(131_072, KvCacheType::F16, 3584.0, 189.51, 6615.0),
    ];

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / MIB as f64
    }

    fn check_against_measured(header: &GgufHeader, size: u64, loads: &[Load]) {
        for measured in loads {
            let result = estimate(
                header,
                size,
                measured.context,
                measured.kv,
                AcceleratorMemory::Dedicated(192.0),
            );
            let label = format!("{} tokens, {}", measured.context, measured.kv);
            // The KV cache is exact in every case, quantized ones included.
            let kv = mib(result.kv_cache_bytes.unwrap());
            assert!((kv - measured.kv_mib).abs() < 0.03, "{label}: KV {kv}");
            let compute = mib(result.compute_buffer_bytes.unwrap());
            if measured.kv == KvCacheType::F16 {
                // The fixed part plus the attention mask reproduces the buffer.
                assert!(
                    (compute - measured.compute_mib).abs() < 0.02,
                    "{label}: compute {compute} vs {}",
                    measured.compute_mib
                );
            } else {
                // The dequantization allowance is an upper bound, never below.
                assert!(
                    compute >= measured.compute_mib,
                    "{label}: compute {compute}"
                );
                assert!(
                    compute <= measured.compute_mib + 80.0,
                    "{label}: compute {compute}"
                );
            }
            // The total is above what the load really used, and by under 5%.
            let total = mib(result.total_bytes.unwrap());
            assert!(total >= measured.used_mib, "{label}: total {total}");
            assert!(
                total <= measured.used_mib * 1.05,
                "{label}: total {total} vs {}",
                measured.used_mib
            );
        }
    }

    #[test]
    fn the_llama_shaped_model_reproduces_the_measured_loads() {
        let (header, size) = measured_llama();
        check_against_measured(&header, size, &LLAMA_LOADS);
    }

    #[test]
    fn the_qwen_shaped_tied_embedding_model_reproduces_the_measured_loads() {
        let (header, size) = measured_qwen();
        check_against_measured(&header, size, &QWEN_LOADS);
    }

    /// The sixteenth load, made with the card almost full: the process overhead
    /// measured only about 0.4 GiB there against 1.05-1.28 GiB elsewhere, so the
    /// estimate was further above what was used than in the other fifteen. It is
    /// still above, and the documented range (0.05% to 5.5%) is the range over
    /// all sixteen, so the 5% bound the other fifteen meet is not asserted here.
    #[test]
    fn the_load_made_with_the_card_almost_full_is_estimated_above_by_about_five_and_a_half_percent()
    {
        let (header, size) = measured_llama();
        let result = estimate(
            &header,
            size,
            86_016,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(192.0),
        );
        // llama.cpp logged a 10752 MiB cache and a 192.01 MiB compute buffer.
        assert!((mib(result.kv_cache_bytes.unwrap()) - 10752.0).abs() < 0.03);
        assert!((mib(result.compute_buffer_bytes.unwrap()) - 192.0).abs() < 0.02);
        let total = mib(result.total_bytes.unwrap());
        let used = 15767.0;
        assert!(total >= used, "total {total}");
        assert!(total > used * 1.05 && total < used * 1.06, "total {total}");
    }

    #[test]
    fn the_components_of_two_measured_loads_are_pinned_to_the_byte() {
        let (header, size) = measured_llama();
        let result = estimate(
            &header,
            size,
            32768,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(192.0),
        );
        // 4403.49 MiB on the GPU, the embedding's 281.81 MiB left on the host.
        assert_eq!(result.weights_bytes, Some(4_617_396_480));
        assert_eq!(result.kv_cache_bytes, Some(4096 * MIB));
        // 108 MiB fixed + 32 MiB of mask.
        assert_eq!(result.compute_buffer_bytes, Some(140 * MIB));
        assert_eq!(result.runtime_reserve_bytes, Some(1280 * MIB));
        assert_eq!(result.total_bytes, Some(10_401_341_696));

        let (header, size) = measured_qwen();
        let result = estimate(
            &header,
            size,
            32768,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(192.0),
        );
        assert_eq!(result.weights_bytes, Some(1_640_622_080));
        assert_eq!(result.kv_cache_bytes, Some(896 * MIB));
        assert_eq!(result.compute_buffer_bytes, Some(64_487_424 + 32 * MIB));
        assert_eq!(result.total_bytes, Some(4_020_365_312));
    }

    #[test]
    fn a_quantized_cache_adds_the_dequantization_buffer_and_f16_does_not() {
        let (header, size) = measured_llama();
        let compute = |kv| {
            estimate(
                &header,
                size,
                131_072,
                kv,
                AcceleratorMemory::Dedicated(192.0),
            )
            .compute_buffer_bytes
            .unwrap()
        };
        let base_and_mask = LLAMA_COMPUTE_BASE + 512 * 2 * 131_072;
        assert_eq!(compute(KvCacheType::F16), base_and_mask);
        // One layer's K and V as 16-bit floats per token: 8 x 256 x 2 bytes.
        let extra = 131_072 * 8 * 256 * 2;
        assert_eq!(compute(KvCacheType::Q8_0), base_and_mask + extra);
        assert_eq!(compute(KvCacheType::Q4_0), base_and_mask + extra);
    }

    #[test]
    fn the_largest_context_accounts_for_what_each_token_adds_to_the_compute_buffer() {
        let (header, size) = measured_llama();
        for kv in KvCacheType::ALL {
            // A card with exactly what the report shows for 32768 tokens holds
            // 32768 and not a step more; one MiB less holds a step fewer.
            let card = shown_mib(&header, size, 32768, kv);
            let result = estimate(&header, size, 1, kv, gpu_mib(card));
            assert_eq!(result.max_context, Some(32768), "{kv}");
            let result = estimate(&header, size, 1, kv, gpu_mib(card - 1));
            assert_eq!(result.max_context, Some(32512), "{kv}");
        }
    }

    /// `count` model shapes with awkward sizes: parts that are not whole MiB, so
    /// the shown (rounded-up) total differs from the exact one by up to a few MiB
    /// -- the case a rounding mismatch needs. Deterministic.
    fn odd_shapes(count: usize) -> Vec<(GgufHeader, u64)> {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        (0..count)
            .map(|_| {
                let layers = 4 + next(40) as u32;
                let kv_heads = 1 + next(16) as u32;
                let head = 32 + next(224) as u32;
                let embd = 256 + next(7936) as u32;
                let ffn = 512 + next(20000) as u32;
                let builder = Builder::new()
                    .string("general.architecture", "odd")
                    .u32("odd.block_count", layers)
                    .u32("odd.embedding_length", embd)
                    .u32("odd.feed_forward_length", ffn)
                    .u32("odd.attention.head_count", kv_heads * 4)
                    .u32("odd.attention.head_count_kv", kv_heads)
                    .u32("odd.attention.key_length", head)
                    .u32("odd.attention.value_length", head)
                    .u32("odd.context_length", 8192 * (1 + next(16) as u32))
                    .tensor_at("token_embd.weight", &[u64::from(embd), 1000], 0)
                    .tensor_at("blk.0.attn_q.weight", &[64, 64], 1_000_003);
                let data = 50_000_000 + next(900_000_000);
                // One tensor holds the rest of the parameters, at a bits-per-weight
                // a real quantization has, so the smaller-quant suggestion is
                // reachable (it is withheld when shape and size disagree).
                let bits = 4.0 + next(50) as f64 / 10.0;
                let params = (data as f64 * 8.0 / bits) as u64;
                let builder = builder.tensor_at("blk.1.ffn_up.weight", &[params], 2_000_000);
                let (header, size) = header_of(&builder, 0);
                (header, size + data)
            })
            .collect()
    }

    /// The class behind the rounding defect: the verdict, the largest context
    /// and every suggestion are one decision. Swept over memory sizes at and
    /// within a few MiB of every boundary, for every cache type and several
    /// model shapes.
    #[test]
    fn the_verdict_the_largest_context_and_the_suggestions_are_one_decision() {
        let mut shapes = vec![measured_llama(), measured_qwen(), llama(GIB_U as usize / 2)];
        shapes.extend(odd_shapes(24));
        let quant_bits = |label: &str| {
            QUANT_BITS_PER_WEIGHT
                .iter()
                .find(|(name, _)| *name == label)
                .map(|(_, bits)| *bits)
                .unwrap()
        };
        let label_re = regex::Regex::new(r"about (\S+) size").unwrap();
        let layers_re = regex::Regex::new(r"keep about (\d+) of (\d+) layers").unwrap();
        let kv_re = regex::Regex::new(r"use a (q8_0|q4_0) KV cache .*about (\d+) MiB").unwrap();
        let mut checked = 0;
        for (header, size) in &shapes {
            let sizing = sizing(header, *size).unwrap();
            for kv in KvCacheType::ALL {
                // The request every card is judged against, and the boundaries of
                // every way to make room for it: each cache type's own total at a
                // spread of contexts, and the GPU's share of the layers.
                let request_context = 8192;
                let mut bases: Vec<u64> = Vec::new();
                for other in KvCacheType::ALL {
                    for context in [256_u64, 4096, 8192, 32768, 65536] {
                        bases.push(shown_mib(header, *size, context, other));
                    }
                    for on_gpu in [1, sizing.layers / 2, sizing.layers - 1] {
                        bases.push(
                            shown_sum([
                                sizing.weights * on_gpu as f64 / sizing.layers as f64,
                                sizing.kv_bytes(request_context, other) * on_gpu as f64
                                    / sizing.layers as f64,
                                sizing.compute_bytes(request_context, other),
                                RUNTIME_RESERVE_BYTES,
                            ]) as u64
                                / MIB,
                        );
                    }
                }
                {
                    // Around each boundary, and well away from it.
                    let mut cards: Vec<u64> = Vec::new();
                    for base in bases {
                        cards.extend((-3..=3).map(|d| base.saturating_add_signed(d)));
                        cards.extend([base / 2 + 1, base + 40]);
                    }
                    for card in cards {
                        let memory = gpu_mib(card);
                        let result = estimate(header, *size, request_context, kv, memory);
                        let available = (card * MIB) as f64;
                        // The verdict is the predicate.
                        assert_eq!(
                            result.outcome == EstimateOutcome::Fits,
                            sizing.fits(request_context, kv, available)
                        );
                        // The largest context fits, and no larger step does.
                        if let Some(max) = result.max_context {
                            let at = estimate(header, *size, max, kv, memory);
                            assert_eq!(at.outcome, EstimateOutcome::Fits, "{kv} {card} MiB: {max}");
                            let cap = header.context_length.unwrap_or(u64::MAX);
                            if max + CONTEXT_STEP <= cap {
                                let above = estimate(header, *size, max + CONTEXT_STEP, kv, memory);
                                assert_ne!(
                                    above.outcome,
                                    EstimateOutcome::Fits,
                                    "{kv} {card} MiB: {max} + a step also fits"
                                );
                            }
                        } else {
                            let at = estimate(header, *size, CONTEXT_STEP, kv, memory);
                            assert_ne!(at.outcome, EstimateOutcome::Fits, "{kv} {card} MiB");
                        }
                        if result.outcome == EstimateOutcome::Fits {
                            assert!(result.suggestions.is_empty());
                            continue;
                        }
                        for suggestion in &result.suggestions {
                            checked += 1;
                            match suggestion.kind {
                                SuggestionKind::SmallerContext => {
                                    let max = result.max_context.unwrap();
                                    assert!(
                                        suggestion.text.contains(&format!("--ctx-size {max}")),
                                        "{}",
                                        suggestion.text
                                    );
                                }
                                SuggestionKind::KvCacheType => {
                                    if let Some(found) = kv_re.captures(&suggestion.text) {
                                        let smaller: KvCacheType = found[1].parse().unwrap();
                                        assert!(sizing.fits(request_context, smaller, available));
                                        assert_eq!(
                                            found[2].parse::<u64>().unwrap(),
                                            shown_mib(header, *size, request_context, smaller)
                                        );
                                    } else if suggestion.text.contains("is still more than") {
                                        let smaller = KvCacheType::ALL
                                            .into_iter()
                                            .filter(|o| {
                                                o.bytes_per_element() < kv.bytes_per_element()
                                            })
                                            .any(|o| !sizing.fits(request_context, o, available));
                                        assert!(smaller, "{}", suggestion.text);
                                    }
                                }
                                SuggestionKind::SmallerQuant => {
                                    if suggestion.text.contains("would fit at") {
                                        let label =
                                            &label_re.captures(&suggestion.text).unwrap()[1];
                                        let weights = sizing.quant_weights(
                                            header.param_count as f64,
                                            quant_bits(label),
                                        );
                                        assert!(
                                            sizing.shown_total_with(weights, request_context, kv)
                                                <= available,
                                            "{}",
                                            suggestion.text
                                        );
                                    }
                                }
                                SuggestionKind::CpuOffload => {
                                    let found = layers_re.captures(&suggestion.text).unwrap();
                                    let on_gpu: u64 = found[1].parse().unwrap();
                                    assert!(sizing.layers_fit(
                                        on_gpu,
                                        request_context,
                                        kv,
                                        available
                                    ));
                                    assert!(
                                        !sizing.layers_fit(
                                            on_gpu + 1,
                                            request_context,
                                            kv,
                                            available
                                        ),
                                        "one more layer also fits: {}",
                                        suggestion.text
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            checked > 50,
            "the sweep reached too few suggestions: {checked}"
        );
    }

    /// The shown total can sit a fraction of a byte *below* the exact one (each
    /// part rounds to a byte, then up to a MiB), so the analytic estimate can
    /// fall one step short of the largest context the verdict accepts. The
    /// largest context is then the verdict's, not the analytic one.
    #[test]
    fn the_largest_context_is_the_largest_one_the_verdict_accepts() {
        let sizing = Sizing {
            data: 0.0,
            // 0.4 of a byte over a whole MiB: it rounds to the MiB below.
            weights: 1000.0_f64.mul_add(MIB as f64, 0.4),
            compute_base: 0.0,
            kv_elements_per_token: 0.0,
            widest_layer_kv_elements: 0.0,
            layers: 1,
        };
        let kv = KvCacheType::F16;
        // Only the attention mask grows: 512 x 2 bytes a token, 4 MiB at 4096.
        let available = (2284 * MIB) as f64;
        assert!(
            sizing.total(4096, kv) > available,
            "the exact total is above"
        );
        assert!(sizing.fits(4096, kv, available), "the shown total is not");
        assert_eq!(sizing.max_context(available, kv, None), Some(4096));
        assert!(!sizing.fits(4352, kv, available));
    }

    /// The estimate assumes flash attention is on, and says so where the user reads
    /// it: with it off the compute buffer grows with the context and the figure is
    /// too low, which only the caveat can tell them.
    #[test]
    fn the_standing_caveat_says_flash_attention_is_assumed() {
        assert!(
            ESTIMATE_CAVEAT.contains("flash attention"),
            "{ESTIMATE_CAVEAT}"
        );
        assert!(ESTIMATE_CAVEAT.contains("assumed"), "{ESTIMATE_CAVEAT}");
        assert!(ESTIMATE_CAVEAT.contains("too low"), "{ESTIMATE_CAVEAT}");
    }

    /// A trained context below one step, on a card whose memory falls between the
    /// exact total and the shown one, used to step below zero: a panic in a debug
    /// build and a wrap to a huge context, then a loop, in a release one. The
    /// answer is that no step fits.
    #[test]
    fn a_trained_context_below_one_step_that_does_not_fit_gives_no_context() {
        let sizing = Sizing {
            data: 0.0,
            weights: 1000.0_f64.mul_add(MIB as f64, 0.4),
            compute_base: 0.0,
            kv_elements_per_token: 0.0,
            widest_layer_kv_elements: 0.0,
            layers: 1,
        };
        let kv = KvCacheType::F16;
        // 250 tokens of mask is under a MiB and shows as one: 2281 MiB in all.
        let available = 2280.5 * MIB as f64;
        assert!(!sizing.fits(250, kv, available));
        assert_eq!(sizing.max_context(available, kv, Some(250)), None);
    }

    /// A cap that is not a whole number of steps steps down onto whole steps:
    /// 1500 does not fit, so the answer is 1024, not 1244 or 988.
    #[test]
    fn a_trained_context_that_is_not_a_whole_step_steps_down_to_a_whole_one() {
        let sizing = Sizing {
            data: 0.0,
            weights: 1000.0_f64.mul_add(MIB as f64, 0.4),
            compute_base: 0.0,
            kv_elements_per_token: 0.0,
            widest_layer_kv_elements: 0.0,
            layers: 1,
        };
        let kv = KvCacheType::F16;
        // The mask at 1500 tokens is 1.46 MiB and shows as 2; at 1024 it is 1.
        // The exact total fits 1500 tokens here, so the search starts at the cap
        // itself rather than at a whole step below it.
        let available = 2281.9 * MIB as f64;
        assert!(sizing.total(1500, kv) <= available);
        assert!(!sizing.fits(1500, kv, available));
        assert!(sizing.fits(1024, kv, available));
        assert_eq!(sizing.max_context(available, kv, Some(1500)), Some(1024));
    }

    /// A trained context that is not a whole step and fits is the answer, even
    /// where the estimate does not reach the next whole step: rounding to a step
    /// before capping gave 1280 here.
    #[test]
    fn a_trained_context_that_fits_is_the_largest_even_when_not_a_whole_step() {
        let sizing = Sizing {
            data: 0.0,
            weights: 1000.0 * MIB as f64,
            compute_base: 0.0,
            // One MiB of f16 cache per token, so a MiB of rounding is a token.
            kv_elements_per_token: 512.0 * 1024.0,
            widest_layer_kv_elements: 512.0 * 1024.0,
            layers: 1,
        };
        let kv = KvCacheType::F16;
        // 1500 tokens show as 1000 + 1500 + 2 + 1280 = 3782 MiB; the exact
        // estimate reaches about 1502 tokens, short of the step at 1536.
        let available = 3784.0 * MIB as f64;
        assert!(sizing.fits(1500, kv, available));
        assert!(!sizing.fits(1536, kv, available));
        assert_eq!(sizing.max_context(available, kv, Some(1500)), Some(1500));
        // Without the cap the answer is a whole step.
        assert_eq!(sizing.max_context(available, kv, None), Some(1280));
    }

    /// The same for partial offload: a layer count that the verdict's own
    /// measure accepts is offered, even where the analytic share says none.
    #[test]
    fn partial_offload_offers_the_layers_the_verdict_accepts() {
        let sizing = Sizing {
            data: 0.0,
            // A quarter of it, 1000 MiB and 0.4 of a byte, rounds to 1000 MiB.
            weights: 4000.0_f64.mul_add(MIB as f64, 1.6),
            compute_base: 0.0,
            kv_elements_per_token: 0.0,
            widest_layer_kv_elements: 0.0,
            layers: 4,
        };
        let request = request(4096, KvCacheType::F16);
        // 1000 MiB of weights, 4 MiB of mask, 1280 MiB of reserve.
        let available = (2284 * MIB) as f64;
        assert!(sizing.layers_fit(1, 4096, KvCacheType::F16, available));
        assert!(!sizing.layers_fit(2, 4096, KvCacheType::F16, available));
        let suggestion = offload_suggestion(&sizing, &request, available, Some(64.0))
            .expect("one layer fits by the verdict's measure");
        assert!(
            suggestion.text.contains("keep about 1 of 4 layers"),
            "{}",
            suggestion.text
        );
    }

    #[test]
    fn the_input_embedding_is_only_left_out_of_the_weights_when_the_header_has_one() {
        // Last in the data: it runs to the end of the tensor data.
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 2)
            .u32("llama.embedding_length", 64)
            .u32("llama.attention.head_count", 4)
            .tensor_at("blk.0.attn_q.weight", &[64, 64], 0)
            .tensor_at("token_embd.weight", &[64, 100], 1000);
        let (header, size) = header_of(&builder, 0);
        let size = size + 5000;
        let result = estimate(
            &header,
            size,
            512,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert_eq!(result.weights_bytes, Some(1000));
        // No input embedding at all: every byte of the data is weights.
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 2)
            .u32("llama.embedding_length", 64)
            .u32("llama.attention.head_count", 4)
            .tensor_at("blk.0.attn_q.weight", &[64, 64], 0);
        let (header, size) = header_of(&builder, 0);
        let result = estimate(
            &header,
            size + 5000,
            512,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert_eq!(result.weights_bytes, Some(5000));
        // A header that puts the embedding past the end of the data cannot
        // make the weights negative.
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 2)
            .u32("llama.embedding_length", 64)
            .u32("llama.attention.head_count", 4)
            .tensor_at("token_embd.weight", &[64, 100], 9_000);
        let (header, size) = header_of(&builder, 0);
        let result = estimate(
            &header,
            size + 5000,
            512,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert_eq!(result.weights_bytes, Some(5000));
    }

    // The tests below pin one rule each, with figures a mutation of that rule
    // changes: a case where the right and the wrong expression happen to agree
    // proves nothing, so every one is asymmetric.

    /// A model that costs only its weights, the attention mask and the reserve.
    fn weights_only(weights: u64, layers: u64) -> Sizing {
        Sizing {
            data: weights as f64,
            weights: weights as f64,
            compute_base: 0.0,
            kv_elements_per_token: 0.0,
            widest_layer_kv_elements: 0.0,
            layers,
        }
    }

    #[test]
    fn partial_offload_names_the_share_left_in_system_ram_not_the_share_on_the_gpu() {
        // Eight layers of 1 GiB. The card holds two: 2048 MiB of them, the 4 MiB
        // mask at 4096 tokens and the 1280 MiB reserve make 3332 MiB; a third
        // layer would need 4356.
        let sizing = weights_only(8 * GIB_U, 8);
        let request = request(4096, KvCacheType::F16);
        let available = f64::from(3332 + 100) * MIB as f64;
        assert!(sizing.layers_fit(2, 4096, KvCacheType::F16, available));
        assert!(!sizing.layers_fit(3, 4096, KvCacheType::F16, available));
        let text = |ram: Option<f64>| {
            offload_suggestion(&sizing, &request, available, ram).map(|suggestion| suggestion.text)
        };
        let offered = text(Some(64.0)).unwrap();
        assert!(offered.contains("keep about 2 of 8 layers"), "{offered}");
        // The six layers that do not fit stay in RAM: 6 GiB, not the 2 GiB on the GPU.
        assert!(
            offered.contains("the other 6.0 GiB of weights and cache"),
            "{offered}"
        );
        // So 4 GiB of RAM cannot hold them (the 2 GiB on the GPU would fit in
        // it), and 7 GiB can.
        assert!(text(Some(4.0)).is_none());
        assert!(text(Some(7.0)).is_some());
    }

    #[test]
    fn a_smaller_quantization_keeps_the_input_embedding_out_of_vram() {
        let sizing = Sizing {
            data: 1000.0,
            weights: 400.0,
            compute_base: 0.0,
            kv_elements_per_token: 0.0,
            widest_layer_kv_elements: 0.0,
            layers: 1,
        };
        // 1000 weights at 8 bits is 1000 bytes of file, of which 40% is the
        // embedding that is not VRAM.
        assert!((sizing.quant_weights(1000.0, 8.0) - 400.0).abs() < 1e-9);
        assert!((sizing.quant_weights(1000.0, 4.0) - 200.0).abs() < 1e-9);
    }

    /// A table entry within [`QUANT_SMALLER_FACTOR`] of the current figure is the
    /// same quantization for this purpose: at 4.4 bits per weight, IQ4_XS (4.25)
    /// is not "smaller", so the suggestion skips it for Q3_K_M even though
    /// IQ4_XS would fit. At a factor of 1.0 it would be named.
    #[test]
    fn a_quantization_within_the_smaller_factor_of_the_current_one_is_not_offered() {
        let params: u64 = 7_000_000_000;
        let builder = llama_like(10).tensor("big", &[params]);
        let file = builder.file(0);
        let header = parse_gguf_header(&mut Cursor::new(&file)).unwrap();
        let size = header.data_offset + (params as f64 * 4.4 / 8.0) as u64;
        // 7e9 weights at 4.25 bits are about 3547 MiB, at 4.4 bits about 3672;
        // with the cache (256 MiB), compute buffer (about 110) and the 1280 MiB
        // reserve, IQ4_XS needs about 5193 MiB and the file as it is about 5318.
        let result = estimate(&header, size, 2048, KvCacheType::F16, gpu_mib(5250));
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        let text = &result
            .suggestions
            .iter()
            .find(|s| s.kind == SuggestionKind::SmallerQuant)
            .unwrap()
            .text;
        assert!(text.contains("Q3_K_M size"), "{text}");
        assert!(!text.contains("IQ4_XS"), "{text}");
    }

    #[test]
    fn a_large_embedding_changes_which_smaller_quantization_is_named() {
        // 7e9 parameters at 8.5 bits, 40% of the data being the embedding.
        let params: u64 = 7_000_000_000;
        let data = (params as f64 * 8.5 / 8.0) as u64;
        let embedding = data / 10 * 4;
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32("llama.embedding_length", 4096)
            .u32("llama.feed_forward_length", 14336)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.context_length", 131_072)
            .tensor_at("token_embd.weight", &[4096, 10], 0)
            .tensor_at("big", &[params], embedding);
        let (header, size) = header_of(&builder, 0);
        let size = size + data;
        // What Q5_K_M (5.69 bits per weight) needs in VRAM, worked out here: the
        // 60% of it that is not the embedding, then the cache, compute buffer and
        // reserve at 2048 tokens.
        let weights_q5 = params as f64 * 5.69 / 8.0 * 0.6;
        let other = (128.0_f64 * 1024.0)
            .mul_add(2048.0, (LLAMA_COMPUTE_BASE + 512 * 2 * 2048) as f64)
            + (1280 * MIB) as f64;
        let card_mib = ((weights_q5 + other) / MIB as f64).ceil() as u64 + 8;
        let result = estimate(&header, size, 2048, KvCacheType::F16, gpu_mib(card_mib));
        assert_eq!(result.outcome, EstimateOutcome::DoesNotFit);
        let text = &result
            .suggestions
            .iter()
            .find(|s| s.kind == SuggestionKind::SmallerQuant)
            .unwrap()
            .text;
        // Counting the embedding as VRAM would price Q5_K_M at 4.6 GiB, past this
        // card, and the suggestion would be a much smaller quantization.
        assert!(text.contains("Q5_K_M size"), "{text}");
    }

    #[test]
    fn the_largest_context_after_a_smaller_cache_is_the_one_for_that_cache() {
        let (header, size) = llama(GIB_U as usize / 1024);
        let fixed = estimate(
            &header,
            size,
            0,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        )
        .total_gib
        .unwrap();
        let memory = AcceleratorMemory::Dedicated(fixed + 0.6);
        let result = estimate(&header, size, 8192, KvCacheType::F16, memory);
        let for_cache = |kv| {
            estimate(&header, size, 8192, kv, memory)
                .max_context
                .unwrap()
        };
        assert_ne!(
            for_cache(KvCacheType::F16),
            for_cache(KvCacheType::Q8_0),
            "the two caches must give different ceilings for this to mean anything"
        );
        let q8 = result
            .suggestions
            .iter()
            .find(|s| s.text.contains("q8_0 KV cache"))
            .unwrap();
        assert!(
            q8.text.contains(&format!(
                "the largest context then is {}",
                tokens(for_cache(KvCacheType::Q8_0))
            )),
            "{}",
            q8.text
        );
    }

    fn split_header(count: u32) -> Builder {
        Builder::new()
            .string("general.architecture", "x")
            .u32("x.block_count", 4)
            .u32("x.embedding_length", 512)
            .u32("x.attention.head_count", 8)
            .u32("split.count", count)
            .tensor("t", &[10])
    }

    #[test]
    fn a_two_shard_model_is_refused_and_a_one_file_split_is_sized() {
        assert_eq!(
            cannot_estimate(&split_header(2)).cannot_estimate_reason,
            Some(CannotEstimateReason::SplitFile),
            "half the weights is not the model's"
        );
        // `split.count` of 1 is a whole model that happens to carry the key.
        let whole = cannot_estimate(&split_header(1));
        assert_ne!(whole.outcome, EstimateOutcome::CannotEstimate, "{whole:?}");
        assert!(whole.total_gib.is_some());
    }

    /// The measured Llama-shaped header, with `llama.context_length` set to `trained`.
    fn llama_trained_for(trained: u32) -> (GgufHeader, u64) {
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32("llama.embedding_length", 4096)
            .u32("llama.feed_forward_length", 14336)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.context_length", trained)
            .tensor_at("token_embd.weight", &[4096, 10], 0)
            .tensor_at("blk.0.attn_q.weight", &[64, 64], 40_960);
        let (header, size) = header_of(&builder, 0);
        (header, size + GIB_U)
    }

    #[test]
    fn a_trained_context_of_zero_is_no_trained_context() {
        let (header, size) = llama_trained_for(0);
        let result = estimate(
            &header,
            size,
            4096,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(64.0),
        );
        // No cap: 64 GiB holds far more than the 131072 a real model would trap it at.
        assert!(result.max_context.unwrap() > 131_072, "{result:?}");
        assert_eq!(result.training_context, None);
        assert!(
            !result.notes.iter().any(|note| note.contains("trained for")),
            "a request was called beyond zero trained tokens: {:?}",
            result.notes
        );
    }

    #[test]
    fn only_a_request_past_the_trained_context_is_told_so() {
        let (header, size) = llama_trained_for(8192);
        let notes = |context| {
            estimate(
                &header,
                size,
                context,
                KvCacheType::F16,
                AcceleratorMemory::Dedicated(64.0),
            )
            .notes
        };
        let beyond = |notes: &[String]| notes.iter().any(|n| n.contains("beyond the"));
        assert!(!beyond(&notes(8192)), "the trained context itself is fine");
        assert!(beyond(&notes(8193)));
    }

    #[test]
    fn each_part_of_the_shown_total_rounds_to_a_byte_and_up_to_a_whole_mib() {
        let mib = MIB as f64;
        // A part 0.6 of a byte over a whole MiB is a byte over it: it rounds up
        // to the next MiB. A truncating conversion would show the MiB below.
        assert_eq!(
            shown_sum([mib.mul_add(1000.0, 0.6), 0.0, 0.0, 0.0]) as u64,
            1001 * MIB
        );
        // 0.4 of a byte over rounds down to the MiB itself.
        assert_eq!(
            shown_sum([mib.mul_add(1000.0, 0.4), 0.0, 0.0, 0.0]) as u64,
            1000 * MIB
        );
    }

    #[test]
    fn the_byte_fields_round_a_fractional_cache_size_to_the_nearest_byte() {
        // One layer, one KV head, key and value 4 and 6 wide: 10 elements a
        // token, 10.625 bytes at q8_0 for one token.
        let builder = Builder::new()
            .string("general.architecture", "x")
            .u32("x.block_count", 1)
            .u32("x.embedding_length", 64)
            .u32("x.attention.head_count", 1)
            .u32("x.attention.head_count_kv", 1)
            .u32("x.attention.key_length", 4)
            .u32("x.attention.value_length", 6)
            .tensor("t", &[10]);
        let (header, size) = header_of(&builder, 4096);
        let result = estimate(
            &header,
            size,
            1,
            KvCacheType::Q8_0,
            AcceleratorMemory::Dedicated(8.0),
        );
        assert_eq!(result.kv_cache_bytes, Some(11));
        let parts = result.weights_bytes.unwrap()
            + 11
            + result.compute_buffer_bytes.unwrap()
            + result.runtime_reserve_bytes.unwrap();
        assert_eq!(result.total_bytes, Some(parts));
    }

    /// A model with `layers` cached layers of 8 KV heads and 128-wide keys and values.
    fn few_layered(layers: u32) -> (GgufHeader, u64) {
        let builder = Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", layers)
            .u32("llama.embedding_length", 4096)
            .u32("llama.feed_forward_length", 14336)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.context_length", 131_072)
            .tensor_at("token_embd.weight", &[4096, 10], 0)
            .tensor_at("blk.0.attn_q.weight", &[64, 64], 40_960);
        let (header, size) = header_of(&builder, 0);
        (header, size + GIB_U)
    }

    #[test]
    fn a_quantized_cache_is_not_offered_where_it_would_need_more_than_the_wider_one() {
        // With two cached layers a q8_0 cache (0.53 GiB at this context) plus the
        // f16 copy of one layer it is dequantized through (0.5 GiB) comes to more
        // than the f16 cache itself (1 GiB), mask aside: it would not cut the need.
        let (header, size) = few_layered(2);
        let suggestions = |memory_gib| {
            estimate(
                &header,
                size,
                131_072,
                KvCacheType::F16,
                AcceleratorMemory::Dedicated(memory_gib),
            )
            .suggestions
        };
        let texts: Vec<String> = suggestions(2.5).into_iter().map(|s| s.text).collect();
        assert!(
            !texts.iter().any(|t| t.contains("q8_0 KV cache")),
            "{texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("q4_0 KV cache")),
            "{texts:?}"
        );
        // With four layers the same cache does save, and is offered.
        let (header, size) = few_layered(4);
        let texts: Vec<String> = estimate(
            &header,
            size,
            131_072,
            KvCacheType::F16,
            AcceleratorMemory::Dedicated(2.5),
        )
        .suggestions
        .into_iter()
        .map(|s| s.text)
        .collect();
        assert!(
            texts.iter().any(|t| t.contains("q8_0 KV cache")),
            "{texts:?}"
        );
    }
}
