// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The context window for a model on a host whose memory Lemonade cannot read.
//!
//! `lemonade load` sizes the llama.cpp context from the memory it can see and,
//! when it can see none, falls back to 4096 tokens whatever the model is (a Vega
//! iGPU on Windows logged `memory undetectable, returning 4096`). That is less
//! than a chat prompt with tool definitions needs, so the load must be given a
//! size instead.
//!
//! The estimate is `rocm-core`'s [`estimate_fit`], the one `rocm diagnose --model`
//! runs: from the GGUF header, the largest context whose weights, KV cache,
//! compute buffer and runtime reserve fit. What it is compared against is the
//! pool an integrated GPU shares with the operating system (the same figure
//! [`rocm_core::detect_host_unified_memory_gib`] reads for the readiness check),
//! scaled by [`SHARED_POOL_BUDGET`] because that figure is a ceiling the rest of
//! the machine also draws on, not memory that is free.
//!
//! Applied only on a host with such a pool. A discrete card's dedicated VRAM is
//! not read here, and nothing in this module touches a load on one: this exists
//! for the host class where Lemonade's own sizing has no memory to work from, and
//! it is not evidence that Lemonade's sizing is wrong anywhere else. A recipe
//! that already carries `--ctx-size` is left as it is.
//!
//! Both serving routes are sized. A router load gets the window on the recipe it
//! passes to `lemonade load` ([`recipe_for_load`]), reading the checkpoint from
//! the running server's model listing; a direct `llama-server` launch gets it as
//! a `--ctx-size` argument ([`direct_context_size`]), reading the GGUF it is
//! already about to serve. Where the window cannot be chosen, or would come out
//! smaller than the 4096 tokens Lemonade falls back to anyway, the load is left
//! exactly as it would have been and the reason goes to the serve log.

use rocm_core::gguf::parse_gguf_header;
use rocm_core::gguf_fit::{CannotEstimateReason, FitRequest, KvCacheType, estimate_fit};
use rocm_core::gguf_source::{GgufRef, GgufSource, LiveGgufSource, parse_gguf_ref};
use rocm_core::model_readiness::AcceleratorMemory;
use rocm_core::{AppPaths, format_http_base_url, http_get_text_with_auth};
use rocm_engine_protocol::{ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint};
use serde_json::Value;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use crate::ServeHttpRequest;
use crate::process::{model_names_match, split_context_size_flag};

/// The largest context asked for, however much memory there is.
///
/// A chat-sized window: past this the KV cache costs memory the operating system
/// shares for a window the assistant does not use. It is a default for a
/// computed size, not a limit on the model.
const MAX_COMPUTED_CONTEXT: u64 = 32_768;

/// The smallest window worth asking for: below it the load is left alone.
///
/// The estimate has no floor of its own — it answers with the largest multiple
/// of 256 tokens that fits, all the way down to 256 — so on a pool a model very
/// nearly fills it returns windows far below the 4096 Lemonade falls back to
/// when it can see no memory. Applying one of those would replace Lemonade's
/// number with a *smaller* one, which is the opposite of this module's purpose;
/// returning nothing instead leaves the load exactly as it was. (A 4096-token
/// window does not hold the assistant's own ~4,400-token chat prompt either, but
/// that is the window the user already has on such a host, and announcing a
/// smaller one in its place is the only outcome this can actually make worse.)
const MIN_COMPUTED_CONTEXT: u64 = 4_096;

/// The share of an integrated GPU's pool the model, its KV cache and the
/// runtime's own allocations may take.
///
/// The pool is the ceiling on what the allocator may hand out, and the operating
/// system, the compositor and every other process draw on the same memory, so
/// sizing against all of it would leave nothing for them. 0.75 is a margin, not a
/// measurement.
const SHARED_POOL_BUDGET: f64 = 0.75;

/// How long the model listing may take: the server is local and already ready.
const LISTING_TIMEOUT: Duration = Duration::from_secs(5);

/// How long reading a GGUF header off the hub may take, in all.
///
/// [`LiveGgufSource`]'s own budget is two minutes, which is sized for a
/// user-initiated `rocm diagnose --model` where waiting is the whole point of
/// the command. This read is best-effort sizing in front of a serve the user is
/// already waiting on, and its failure costs only the window Lemonade would have
/// picked anyway — so an unreachable hub must not hold the serve for minutes. A
/// listing plus a header read over a working connection is a second or two;
/// twenty leaves room for a slow link without becoming a hang.
const HEADER_READ_BUDGET: Duration = Duration::from_secs(20);

/// A context size chosen for a model, and the line that says where it came from.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ComputedContext {
    pub(crate) tokens: u64,
    pub(crate) notice: String,
}

/// Whether the recipe already names a context size, which then stands.
pub(crate) fn recipe_pins_context(recipe: Option<&EngineRecipeHint>) -> bool {
    recipe.is_some_and(|recipe| split_context_size_flag(&recipe.required_flags).0.is_some())
}

/// `recipe` with `--ctx-size <tokens>` added; a recipe is synthesized when none
/// was given.
pub(crate) fn recipe_with_context(
    recipe: Option<&EngineRecipeHint>,
    tokens: u64,
) -> EngineRecipeHint {
    let mut recipe = recipe.cloned().unwrap_or_else(|| EngineRecipeHint {
        contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
        engine: crate::ENGINE_NAME.to_owned(),
        ..EngineRecipeHint::default()
    });
    recipe
        .required_flags
        .extend(["--ctx-size".to_owned(), tokens.to_string()]);
    recipe
}

/// The pool this host's integrated GPU shares with the operating system, given
/// the host's gfx target and what the pool reader answered. `None` for a host
/// without an integrated GPU or with no readable pool, which leaves the load to
/// Lemonade.
fn shared_pool_for(gfx_target: Option<&str>, pool_gib: Option<f64>) -> Option<f64> {
    gfx_target
        .is_some_and(rocm_core::gfx_is_apu_family)
        .then_some(pool_gib)
        .flatten()
}

/// [`shared_pool_for`] for this host, detected once per process.
///
/// The detection is not free, and on Windows it is expensive: `detect_host_gfx_target`
/// is evaluated before the APU gate in [`shared_pool_for`], and on that platform it
/// assembles the host inventory, which runs a PowerShell CIM query beside `pnputil`
/// — on discrete-GPU hosts too, where the answer is always `None`. One load per serve
/// is enough to notice, and a serve can load more than once, so the answer is cached
/// rather than taken again. It cannot change while the process runs.
fn host_shared_pool_gib() -> Option<f64> {
    static POOL_GIB: OnceLock<Option<f64>> = OnceLock::new();
    *POOL_GIB.get_or_init(|| {
        shared_pool_for(
            rocm_core::detect_host_gfx_target().as_deref(),
            rocm_core::detect_host_unified_memory_gib(),
        )
    })
}

/// The checkpoint (`owner/repo:file.gguf`) `/v1/models` reports for a model.
fn models_payload_checkpoint(models: &Value, model_ref: &str) -> Option<String> {
    models
        .get("data")
        .or_else(|| models.get("models"))
        .and_then(Value::as_array)?
        .iter()
        .find(|model| {
            ["id", "model", "name"]
                .into_iter()
                .filter_map(|field| model.get(field).and_then(Value::as_str))
                .any(|name| model_names_match(name, model_ref))
        })?
        .get("checkpoint")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Ask the running server which checkpoint it serves a model from. Lists models
/// it has not downloaded yet too (`show_all`), because the first serve of a model
/// is the one whose size matters.
fn query_checkpoint(host: &str, port: u16, model_ref: &str) -> Option<String> {
    let body = http_get_text_with_auth(
        &format_http_base_url(host, port),
        "/v1/models?show_all=true",
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
        LISTING_TIMEOUT,
    )
    .ok()?;
    models_payload_checkpoint(&serde_json::from_str(&body).ok()?, model_ref)
}

/// Where to read a checkpoint's GGUF header: the cached file when the cache holds
/// an unambiguous one, else the repository on Hugging Face (a header read, never
/// the weights).
///
/// A split model is one model, and its header is in the first shard, so a complete
/// shard set resolves to that shard exactly as [`crate::direct_llama::direct_llama_model_path`]
/// resolves the file to serve. Reading the length from `matches` instead would send
/// every split model to the hub — a network read for a model already on disk, and no
/// sizing at all offline.
fn reference_for_checkpoint(paths: Option<&AppPaths>, checkpoint: &str) -> Option<GgufRef> {
    let cached = paths.zip(crate::direct_llama::parse_hf_checkpoint(checkpoint));
    if let Some((paths, checkpoint)) = cached {
        let mut matches = crate::direct_llama::hf_checkpoint_gguf_matches(paths, &checkpoint);
        if let Some(first) = crate::direct_llama::first_shard_of_complete_set(&matches) {
            return Some(GgufRef::Local(first));
        }
        // Several unrelated files, or some but not all shards of one model: not
        // this module's to choose between, and the hub answers for the variant.
        if matches.len() == 1 {
            return matches.pop().map(GgufRef::Local);
        }
    }
    parse_gguf_ref(checkpoint)
}

/// The estimate's answer bounded at both ends: capped at [`MAX_COMPUTED_CONTEXT`],
/// and `None` below [`MIN_COMPUTED_CONTEXT`], where applying it would ship a
/// smaller window than the one it is meant to replace.
fn window_within_bounds(max_context: Option<u64>) -> Option<u64> {
    let tokens = max_context?.min(MAX_COMPUTED_CONTEXT);
    (tokens >= MIN_COMPUTED_CONTEXT).then_some(tokens)
}

/// Choose a context for the model at `reference` against a shared pool of
/// `pool_gib`, or say why none was chosen as the `because ...` clause of
/// [`announce_unsized`].
///
/// The three causes are kept apart because they are not the same news to a user:
/// a header that could not be read may be a download or a network to retry, a
/// layout the estimate declines to size is permanent for that model and no
/// reflection on the host's memory, and a pool too small for
/// [`MIN_COMPUTED_CONTEXT`] is about this host. Reporting them as one disjunction
/// tells a reader whose MoE model sized fine on disk that its header is
/// unreadable or that it does not fit, neither of which happened.
fn computed_context(
    pool_gib: f64,
    system_ram_gib: Option<f64>,
    source: &dyn GgufSource,
    reference: &GgufRef,
    described: &str,
) -> Result<ComputedContext, String> {
    let mut opened = source
        .open(reference)
        .map_err(|_| format!("the GGUF for `{described}` could not be read"))?;
    let header = parse_gguf_header(&mut opened.reader)
        .map_err(|_| format!("the GGUF header for `{described}` could not be read"))?;
    let estimate = estimate_fit(
        &header,
        opened.size,
        &opened.description,
        &FitRequest {
            context_size: MAX_COMPUTED_CONTEXT,
            context_size_defaulted: false,
            kv_cache_type: KvCacheType::F16,
        },
        AcceleratorMemory::Unified {
            gtt_total_gib: pool_gib * SHARED_POOL_BUDGET,
        },
        system_ram_gib,
    );
    if let Some(reason) = estimate.cannot_estimate_reason {
        return Err(format!(
            "the memory `{described}` needs could not be estimated: {}",
            unmeasurable(reason)
        ));
    }
    let Some(tokens) = window_within_bounds(estimate.max_context) else {
        let floor = MIN_COMPUTED_CONTEXT;
        return Err(format!(
            "no window of at least {floor} tokens fits in the {pool_gib:.1} GiB this integrated \
             GPU shares with the operating system"
        ));
    };
    Ok(ComputedContext {
        tokens,
        notice: format!(
            "Context window: {tokens} tokens, chosen from the model file and the {pool_gib:.1} GiB \
             of memory this integrated GPU shares with the operating system."
        ),
    })
}

/// Why [`estimate_fit`] gave no figure to size a window from, in the words the
/// serve log uses.
///
/// Matched exhaustively on purpose: a new [`CannotEstimateReason`] must be given
/// its own wording here rather than falling into a catch-all that would report
/// the wrong cause, which is the whole point of the distinction.
const fn unmeasurable(reason: CannotEstimateReason) -> &'static str {
    match reason {
        CannotEstimateReason::HeaderUnreadable => "its GGUF header could not be read",
        CannotEstimateReason::NotAModel => {
            "the file named is a multimodal projector, not a language model"
        }
        CannotEstimateReason::SplitFile => {
            "it is one shard of a split model, whose metadata and weights are spread over \
             several files"
        }
        CannotEstimateReason::LatentAttention => {
            "its KV cache is not heads times head width, which the estimate does not measure"
        }
        CannotEstimateReason::NoAttentionLayers => {
            "no layer in it keeps a KV cache, so a context window sizes nothing"
        }
        CannotEstimateReason::UnmeasuredLayout => {
            "part of its memory is not counted or not measured (state-space or \
             mixture-of-experts layers), so any window chosen from the total could be too large"
        }
        CannotEstimateReason::MissingMetadata => {
            "its header does not carry a figure the estimate needs"
        }
        // The caller states the host's memory itself, so these three cannot be
        // reached from here; they are the discrete-GPU and unreadable-pool cases
        // that never get as far as choosing a window.
        CannotEstimateReason::NoGpu => "no GPU is visible to ROCm",
        CannotEstimateReason::AcceleratorMemoryUnknown => "this GPU's memory could not be measured",
        CannotEstimateReason::UnifiedMemoryUnreadable => {
            "the memory this GPU shares with the operating system could not be read"
        }
    }
}

/// Print the notice and append it to the serve log, where a managed serve (whose
/// stdout and stderr are discarded) lets the user read it.
fn announce(notice: &str, log_path: Option<&Path>) {
    use std::io::Write as _;
    eprintln!("{notice}");
    if let Some(log_path) = log_path
        && let Ok(mut log) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
    {
        let _ = writeln!(log, "{notice}");
    }
}

/// Say why the window was left to the engine to choose.
///
/// Only *operational* failures reach this. A host with no shared pool is the
/// common case — every discrete-GPU machine — and says nothing, or the serve log
/// of a host this module never had anything to do with fills with it.
fn announce_unsized(reason: &str, log_path: Option<&Path>) {
    announce(
        &format!("Context window: left to the engine to choose, because {reason}."),
        log_path,
    );
}

/// The host facts and the header reader the sizing works from.
///
/// Passed in rather than read inside the sizing so that a test can state a host
/// it is not running on — the same seam `simulated_listing_gpu_gib` gives the
/// model listing in `apps/rocm/src/model_fit.rs`. Without it nothing can
/// exercise the path at all off an integrated-GPU machine.
struct SizingHost<'a> {
    /// The pool an integrated GPU shares with the operating system. `None` on
    /// every other host, which is not a failure and is not reported as one.
    pool_gib: Option<f64>,
    system_ram_gib: Option<f64>,
    /// Where a cached GGUF is looked for. `None` leaves every checkpoint to the
    /// hub, and is what the direct route passes: its file is already resolved.
    paths: Option<AppPaths>,
    source: &'a dyn GgufSource,
}

/// Read the header at `reference` and announce what came of it.
fn sized_against(
    host: &SizingHost<'_>,
    pool_gib: f64,
    reference: &GgufRef,
    described: &str,
    log_path: Option<&Path>,
) -> Option<ComputedContext> {
    let computed = match computed_context(
        pool_gib,
        host.system_ram_gib,
        host.source,
        reference,
        described,
    ) {
        Ok(computed) => computed,
        Err(reason) => {
            announce_unsized(&reason, log_path);
            return None;
        }
    };
    announce(&computed.notice, log_path);
    Some(computed)
}

/// The window for the model the listing reports as `checkpoint`, or `None` with
/// the reason announced.
fn context_for_checkpoint(
    host: &SizingHost<'_>,
    checkpoint: Option<String>,
    log_path: Option<&Path>,
) -> Option<ComputedContext> {
    // Not an integrated GPU, so the engine's own sizing has memory to work from
    // and this module has nothing to say. Deliberately silent: see above.
    let pool_gib = host.pool_gib?;
    let Some(checkpoint) = checkpoint else {
        announce_unsized(
            "the Lemonade model listing named no checkpoint for this model",
            log_path,
        );
        return None;
    };
    let Some(reference) = reference_for_checkpoint(host.paths.as_ref(), &checkpoint) else {
        announce_unsized(
            &format!("`{checkpoint}` does not name a GGUF file that can be read"),
            log_path,
        );
        return None;
    };
    sized_against(host, pool_gib, &reference, &checkpoint, log_path)
}

/// The recipe the router's `lemonade load` should carry: the request's own with
/// a computed `--ctx-size` added where this host needs one, else the request's
/// own unchanged (`None` when it has none either).
///
/// The choice *between* the two lives here rather than at the call site so that
/// a test can show the computed window reaching the load command, instead of
/// only that it was computed.
pub(crate) fn recipe_for_load(
    request: &ServeHttpRequest,
    model_ref: &str,
) -> Option<EngineRecipeHint> {
    let source = LiveGgufSource::new().with_budget(HEADER_READ_BUDGET);
    let host = SizingHost {
        pool_gib: host_shared_pool_gib(),
        system_ram_gib: rocm_core::detect_system_ram_gib(),
        paths: AppPaths::discover().ok(),
        source: &source,
    };
    // The listing is only asked for where its answer could be used: on an
    // integrated-GPU host, for a recipe that leaves the window open.
    let checkpoint = (host.pool_gib.is_some()
        && !recipe_pins_context(request.engine_recipe.as_ref()))
    .then(|| query_checkpoint(&request.host, request.port, model_ref))
    .flatten();
    recipe_for_load_from(request, &host, checkpoint)
}

/// [`recipe_for_load`] against stated host facts and an already-read checkpoint.
fn recipe_for_load_from(
    request: &ServeHttpRequest,
    host: &SizingHost<'_>,
    checkpoint: Option<String>,
) -> Option<EngineRecipeHint> {
    sized_recipe(request, host, checkpoint).or_else(|| request.engine_recipe.clone())
}

/// The request's recipe with a computed window added, or `None` to use the
/// request's recipe as it stands.
fn sized_recipe(
    request: &ServeHttpRequest,
    host: &SizingHost<'_>,
    checkpoint: Option<String>,
) -> Option<EngineRecipeHint> {
    if recipe_pins_context(request.engine_recipe.as_ref()) {
        return None;
    }
    let computed = context_for_checkpoint(host, checkpoint, request.log_path.as_deref())?;
    Some(recipe_with_context(
        request.engine_recipe.as_ref(),
        computed.tokens,
    ))
}

/// The `--ctx-size` a direct `llama-server` launch should carry for the GGUF at
/// `model_path`, or `None` to leave the window to llama-server's own default.
///
/// The direct route never reaches the router, so nothing on it goes through
/// [`recipe_for_load`]: its model file is already resolved, so the header is read
/// straight from it and the running server is asked nothing. Both direct launches
/// — the route chosen up front and the fallback after a router load failure — go
/// through `serve_direct_llama_server`, which is where this is called.
pub(crate) fn direct_context_size(request: &ServeHttpRequest, model_path: &Path) -> Option<u64> {
    // A local file: no hub request, so no budget to shorten.
    let source = LiveGgufSource::new();
    let host = SizingHost {
        pool_gib: host_shared_pool_gib(),
        system_ram_gib: rocm_core::detect_system_ram_gib(),
        paths: None,
        source: &source,
    };
    direct_context_size_from(request, &host, model_path)
}

/// [`direct_context_size`] against stated host facts.
fn direct_context_size_from(
    request: &ServeHttpRequest,
    host: &SizingHost<'_>,
    model_path: &Path,
) -> Option<u64> {
    if recipe_pins_context(request.engine_recipe.as_ref()) {
        return None;
    }
    let pool_gib = host.pool_gib?;
    sized_against(
        host,
        pool_gib,
        &GgufRef::Local(model_path.to_path_buf()),
        &model_path.display().to_string(),
        request.log_path.as_deref(),
    )
    .map(|computed| computed.tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::gguf_source::OpenedGguf;
    use serde_json::json;
    use std::io::Cursor;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// What the serve log calls the model under test, so a reason that names it
    /// can be told from one that does not.
    const DESCRIBED: &str = "fake.gguf";

    /// A Qwen3-4B-shaped header: 36 layers of 8 KV heads with key and value
    /// width 128, so 147,456 bytes of f16 KV cache per token, and a 262,144
    /// token training context.
    fn qwen_shaped_header() -> Vec<u8> {
        qwen_shaped_header_with_experts(None)
    }

    /// The same header, optionally carrying `qwen3.expert_count`, which is what
    /// makes the parser call the model mixture-of-experts.
    fn qwen_shaped_header_with_experts(experts: Option<u32>) -> Vec<u8> {
        fn string(out: &mut Vec<u8>, value: &str) {
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        fn uint(out: &mut Vec<u8>, key: &str, value: u32) {
            string(out, key);
            out.extend_from_slice(&4_u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        let mut metadata = Vec::new();
        string(&mut metadata, "general.architecture");
        metadata.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut metadata, "qwen3");
        for (key, value) in [
            ("qwen3.block_count", 36),
            ("qwen3.embedding_length", 2560),
            ("qwen3.feed_forward_length", 9728),
            ("qwen3.attention.head_count", 32),
            ("qwen3.attention.head_count_kv", 8),
            ("qwen3.attention.key_length", 128),
            ("qwen3.attention.value_length", 128),
            ("qwen3.context_length", 262_144),
        ] {
            uint(&mut metadata, key, value);
        }
        if let Some(experts) = experts {
            uint(&mut metadata, "qwen3.expert_count", experts);
        }
        let entries = 9 + u64::from(u8::from(experts.is_some()));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&entries.to_le_bytes()); // metadata entries
        bytes.extend(metadata);
        string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&256_u64.to_le_bytes());
        bytes.extend_from_slice(&256_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // f32
        bytes.extend_from_slice(&0_u64.to_le_bytes()); // offset
        bytes
    }

    /// Serves one header, said to belong to a file of `size` bytes.
    struct FakeSource {
        header: Vec<u8>,
        size: u64,
    }

    impl GgufSource for FakeSource {
        fn open(&self, _reference: &GgufRef) -> anyhow::Result<OpenedGguf> {
            Ok(OpenedGguf {
                reader: Box::new(Cursor::new(self.header.clone())),
                size: self.size,
                description: "fake.gguf".to_owned(),
            })
        }
    }

    /// Refuses every read, as a file that is missing and a hub that is offline.
    struct UnreadableSource;

    impl GgufSource for UnreadableSource {
        fn open(&self, _reference: &GgufRef) -> anyhow::Result<OpenedGguf> {
            anyhow::bail!("offline")
        }
    }

    fn reference() -> GgufRef {
        GgufRef::Local("fake.gguf".into())
    }

    /// A 2.5 GiB file: about what a Q4_K_M of a 4B model weighs.
    fn four_b_source() -> FakeSource {
        FakeSource {
            header: qwen_shaped_header(),
            size: 5 * GIB / 2,
        }
    }

    /// The chat prompt the assistant sends before the user types anything is
    /// about 4,400 tokens, so a context below it cannot answer a greeting.
    const CHAT_PROMPT_TOKENS: u64 = 4_411;

    #[test]
    fn a_4b_model_on_the_4700u_pool_gets_a_window_the_chat_prompt_fits_in() {
        // The pool `unified_memory.rs` records for the Ryzen 7 4700U.
        let computed = computed_context(7.7, Some(15.0), &four_b_source(), &reference(), DESCRIBED)
            .expect("a 4B model fits in a 7.7 GiB pool");
        assert!(
            computed.tokens > CHAT_PROMPT_TOKENS && computed.tokens < MAX_COMPUTED_CONTEXT,
            "{computed:?}"
        );
        assert_eq!(computed.tokens % 256, 0, "{computed:?}");
        assert!(computed.notice.contains(&computed.tokens.to_string()));
    }

    #[test]
    fn the_window_leaves_the_rest_of_the_pool_to_the_operating_system() {
        let pool_gib = 7.7;
        let computed =
            computed_context(pool_gib, None, &four_b_source(), &reference(), DESCRIBED).unwrap();
        // The weights and the KV cache alone, with none of the allowances the
        // estimate adds on top, stay within three quarters of the pool. The
        // fraction is written out here, not read from `SHARED_POOL_BUDGET`, so a
        // change to the constant cannot move the test with it.
        let weights = (5 * GIB / 2) as f64;
        let kv = computed.tokens as f64 * 147_456.0;
        assert!(weights + kv <= pool_gib * 0.75 * GIB as f64, "{computed:?}");
    }

    #[test]
    fn a_large_pool_is_capped_at_a_chat_sized_window() {
        let computed =
            computed_context(96.0, Some(128.0), &four_b_source(), &reference(), DESCRIBED).unwrap();
        assert_eq!(computed.tokens, MAX_COMPUTED_CONTEXT);
    }

    /// The three ways no window is chosen must not be reported as each other.
    /// One disjunction ("could not be read, or does not fit") told a user whose
    /// mixture-of-experts model sized fine on disk that its header was
    /// unreadable or that it did not fit, and neither had happened.
    #[test]
    fn each_reason_for_choosing_no_window_is_reported_as_itself() {
        let unsized_because = |pool_gib: f64, ram: Option<f64>, source: &dyn GgufSource| {
            computed_context(pool_gib, ram, source, &reference(), DESCRIBED).expect_err("no window")
        };

        // Too small a pool: about this host, and only this host.
        let small = unsized_because(2.0, Some(4.0), &four_b_source());
        assert!(
            small.contains("no window of at least") && small.contains("2.0 GiB"),
            "{small}"
        );

        // Nothing to read: a download or a network to retry.
        let not_gguf = FakeSource {
            header: b"not a gguf file".to_vec(),
            size: GIB,
        };
        for source in [&UnreadableSource as &dyn GgufSource, &not_gguf] {
            let unreadable = unsized_because(7.7, Some(15.0), source);
            assert!(
                unreadable.contains(DESCRIBED) && unreadable.contains("could not be"),
                "{unreadable}"
            );
            assert!(
                !unreadable.contains("no window of at least"),
                "a header that was never read says nothing about what fits: {unreadable}"
            );
        }

        // A layout the estimate declines to measure: permanent for that model,
        // and no reflection on the pool.
        let moe = unsized_because(
            7.7,
            Some(15.0),
            &FakeSource {
                header: qwen_shaped_header_with_experts(Some(8)),
                size: 5 * GIB / 2,
            },
        );
        assert!(moe.contains("mixture-of-experts"), "{moe}");
        assert!(
            !moe.contains("could not be read") && !moe.contains("no window of at least"),
            "the header read fine and no fit verdict was reached: {moe}"
        );
    }

    #[test]
    fn only_a_host_with_an_integrated_gpu_and_a_readable_pool_has_a_shared_pool() {
        assert_eq!(shared_pool_for(Some("gfx90c"), Some(7.7)), Some(7.7));
        assert_eq!(shared_pool_for(Some("gfx1151"), Some(64.0)), Some(64.0));
        // The pool is readable on a discrete card too, and is not its memory.
        assert_eq!(shared_pool_for(Some("gfx1100"), Some(64.0)), None);
        assert_eq!(shared_pool_for(Some("gfx942"), Some(64.0)), None);
        assert_eq!(shared_pool_for(None, Some(64.0)), None);
        assert_eq!(shared_pool_for(Some("gfx90c"), None), None);
    }

    #[test]
    fn a_recipe_that_names_a_context_size_keeps_it() {
        let named = |flags: &[&str]| EngineRecipeHint {
            required_flags: flags.iter().map(|flag| (*flag).to_owned()).collect(),
            ..EngineRecipeHint::default()
        };
        assert!(recipe_pins_context(Some(&named(&["--ctx-size", "8192"]))));
        assert!(recipe_pins_context(Some(&named(&[
            "--temperature",
            "0.5",
            "--ctx-size=4096"
        ]))));
        assert!(!recipe_pins_context(Some(&named(&[
            "--temperature",
            "0.5"
        ]))));
        assert!(!recipe_pins_context(None));
        // A `--ctx-size` that names no value pins nothing: the recipe did not
        // choose a window, so this host's sizing still runs. See
        // `process::split_context_size_flag`.
        assert!(!recipe_pins_context(Some(&named(&["--ctx-size"]))));
        assert!(!recipe_pins_context(Some(&named(&[
            "--ctx-size",
            "--temperature",
            "0.5"
        ]))));
        assert!(!recipe_pins_context(Some(&named(&["--ctx-size="]))));
    }

    #[test]
    fn the_computed_window_is_added_to_the_recipe_without_dropping_its_flags() {
        let authored = EngineRecipeHint {
            engine: "lemonade".to_owned(),
            required_flags: vec!["--temperature".to_owned(), "0.5".to_owned()],
            ..EngineRecipeHint::default()
        };
        let with = recipe_with_context(Some(&authored), 15_360);
        assert_eq!(
            with.required_flags,
            ["--temperature", "0.5", "--ctx-size", "15360"]
        );
        assert_eq!(with.engine, "lemonade");

        let synthesized = recipe_with_context(None, 8192);
        assert_eq!(synthesized.required_flags, ["--ctx-size", "8192"]);
        assert_eq!(synthesized.engine, crate::ENGINE_NAME);
        assert_eq!(synthesized.contract_version, ENGINE_RECIPE_CONTRACT_VERSION);
    }

    #[test]
    fn the_checkpoint_is_read_from_the_entry_for_the_model_being_served() {
        let models = json!({"data": [
            {"id": "Other-GGUF", "checkpoint": "o/other:Q4_K_M.gguf"},
            {
                "id": "Qwen3-4B-Instruct-2507-GGUF",
                "checkpoint": "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "downloaded": false
            },
        ]});
        let expected = Some(
            "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf".to_owned(),
        );
        assert_eq!(
            models_payload_checkpoint(&models, "Qwen3-4B-Instruct-2507-GGUF"),
            expected
        );
        // The shorthand `rocm serve` accepts names the same entry.
        assert_eq!(models_payload_checkpoint(&models, "qwen3-4b"), expected);
        assert_eq!(models_payload_checkpoint(&models, "Unlisted-GGUF"), None);
        // An entry with no checkpoint (a model Lemonade does not source from
        // Hugging Face) is not guessed at.
        let bare = json!({"data": [{"id": "Local-GGUF"}]});
        assert_eq!(models_payload_checkpoint(&bare, "Local-GGUF"), None);
    }

    #[test]
    fn a_checkpoint_with_no_cached_file_is_read_from_the_hub() {
        let reference = reference_for_checkpoint(None, "unsloth/Some-GGUF:Some-Q4_K_M.gguf")
            .expect("a checkpoint names a hub file");
        assert_eq!(
            reference,
            GgufRef::Hub {
                owner: "unsloth".to_owned(),
                repo: "Some-GGUF".to_owned(),
                variant: Some("Some-Q4_K_M.gguf".to_owned()),
            }
        );
        assert_eq!(reference_for_checkpoint(None, "not a checkpoint"), None);
    }

    /// The cached-file branch, which `a_checkpoint_with_no_cached_file_is_read_from_the_hub`
    /// does not reach. A complete split model is one model and its header is in
    /// the first shard: reading the match count instead would send every split
    /// model to the hub, downloading a header for a file already on disk and
    /// sizing nothing at all offline.
    #[test]
    fn a_complete_cached_split_model_is_read_from_its_first_shard() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let paths = AppPaths {
            config_dir: temporary.path().join("config"),
            data_dir: temporary.path().join("data"),
            cache_dir: temporary.path().join("cache"),
        };
        // The repo names are unique to this test so that a real Hugging Face
        // cache on the host (also searched) cannot match them.
        let snapshot = |repo: &str| {
            let dir = paths
                .cache_dir
                .join("huggingface")
                .join("hub")
                .join(format!("models--unsloth--{repo}"))
                .join("snapshots")
                .join("0000000000000000000000000000000000000000");
            std::fs::create_dir_all(&dir).expect("the snapshot directory");
            dir
        };

        let split = snapshot("Ctxsize-Split-Fixture-GGUF");
        let shard = |index: u32| split.join(format!("Split-Q4_K_M-{index:05}-of-00002.gguf"));
        for index in 1..=2 {
            std::fs::write(shard(index), b"gguf").expect("a shard file");
        }
        assert_eq!(
            reference_for_checkpoint(Some(&paths), "unsloth/Ctxsize-Split-Fixture-GGUF:Q4_K_M"),
            Some(GgufRef::Local(shard(1))),
            "a complete split set already on disk must not be read from the hub"
        );

        // One cached file is read from disk as before.
        let single = snapshot("Ctxsize-Single-Fixture-GGUF");
        let only = single.join("Single-Q4_K_M.gguf");
        std::fs::write(&only, b"gguf").expect("a model file");
        assert_eq!(
            reference_for_checkpoint(Some(&paths), "unsloth/Ctxsize-Single-Fixture-GGUF:Q4_K_M"),
            Some(GgufRef::Local(only))
        );

        // Two unrelated files that both match the variant are still ambiguous,
        // and are left to the hub rather than picked between here.
        let ambiguous = snapshot("Ctxsize-Ambiguous-Fixture-GGUF");
        for name in ["Ambiguous-Q4_K_M.gguf", "Ambiguous-Q4_K_M-tuned.gguf"] {
            std::fs::write(ambiguous.join(name), b"gguf").expect("a model file");
        }
        assert_eq!(
            reference_for_checkpoint(
                Some(&paths),
                "unsloth/Ctxsize-Ambiguous-Fixture-GGUF:Q4_K_M"
            ),
            Some(GgufRef::Hub {
                owner: "unsloth".to_owned(),
                repo: "Ctxsize-Ambiguous-Fixture-GGUF".to_owned(),
                variant: Some("Q4_K_M".to_owned()),
            })
        );
    }

    /// The exact edges of the bounds, which `computed_context` cannot show on
    /// its own: `estimate_fit` only answers in steps of 256 tokens.
    #[test]
    fn the_window_is_bounded_at_both_ends() {
        assert_eq!(window_within_bounds(None), None);
        assert_eq!(window_within_bounds(Some(MIN_COMPUTED_CONTEXT - 1)), None);
        assert_eq!(
            window_within_bounds(Some(MIN_COMPUTED_CONTEXT)),
            Some(MIN_COMPUTED_CONTEXT)
        );
        assert_eq!(
            window_within_bounds(Some(MAX_COMPUTED_CONTEXT)),
            Some(MAX_COMPUTED_CONTEXT)
        );
        assert_eq!(
            window_within_bounds(Some(MAX_COMPUTED_CONTEXT * 4)),
            Some(MAX_COMPUTED_CONTEXT)
        );
    }

    /// Measured against the real estimate, not asserted from the rule: on the
    /// 7.7 GiB pool a 3.88 GiB Qwen3-4B still reaches 4096 tokens, 3.89 GiB only
    /// 3840, and 4.41 GiB only 256. Without the floor the last two would be
    /// announced and loaded, replacing Lemonade's own 4096 fallback with less
    /// than it — the opposite of what this module exists to do.
    #[test]
    fn a_window_smaller_than_the_fallback_it_replaces_is_never_used() {
        let sized = |hundredths_of_a_gib: u64| {
            computed_context(
                7.7,
                Some(15.0),
                &FakeSource {
                    header: qwen_shaped_header(),
                    size: hundredths_of_a_gib * GIB / 100,
                },
                &reference(),
                DESCRIBED,
            )
            .ok()
            .map(|computed| computed.tokens)
        };
        assert_eq!(sized(388), Some(MIN_COMPUTED_CONTEXT));
        assert_eq!(sized(389), None, "3840 tokens is below the fallback");
        assert_eq!(sized(441), None, "256 tokens is far below the fallback");
        assert_eq!(sized(442), None, "nothing fits at all");
    }

    const CHECKPOINT: &str =
        "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf";

    fn request_in(dir: &Path) -> ServeHttpRequest {
        ServeHttpRequest {
            service_id: "test".to_owned(),
            model_ref: "qwen-smoke".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: vec![0],
            runtime_id: None,
            env_id: None,
            state_path: dir.join("state.json"),
            log_path: None,
            engine_recipe: None,
        }
    }

    /// A host whose integrated GPU shares the 7.7 GiB pool `unified_memory.rs`
    /// records for the Ryzen 7 4700U, stated rather than detected.
    fn integrated_gpu_host(source: &dyn GgufSource) -> SizingHost<'_> {
        SizingHost {
            pool_gib: Some(7.7),
            system_ram_gib: Some(15.0),
            paths: None,
            source,
        }
    }

    /// Every host without one, which is every discrete-GPU machine.
    fn discrete_gpu_host(source: &dyn GgufSource) -> SizingHost<'_> {
        SizingHost {
            pool_gib: None,
            system_ram_gib: Some(15.0),
            paths: None,
            source,
        }
    }

    /// The router route end to end: the window is computed *and* reaches the
    /// `lemonade load` argument list, which is the thing the serve actually runs.
    #[test]
    fn the_computed_window_reaches_the_lemonade_load_command() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let mut request = request_in(temporary.path());
        request.engine_recipe = Some(EngineRecipeHint {
            required_flags: vec!["--temperature".to_owned(), "0.5".to_owned()],
            ..EngineRecipeHint::default()
        });
        let source = four_b_source();
        let recipe = recipe_for_load_from(
            &request,
            &integrated_gpu_host(&source),
            Some(CHECKPOINT.to_owned()),
        )
        .expect("a readable header on an integrated-GPU host is sized");
        let args = crate::process::lemonade_model_load_args(
            &request.host,
            request.port,
            "Qwen3-4B-Instruct-2507-GGUF",
            "vulkan",
            Some(&recipe),
        );
        let at = args
            .iter()
            .position(|arg| arg == "--ctx-size")
            .unwrap_or_else(|| panic!("the load must carry a context size: {args:?}"));
        let tokens: u64 = args[at + 1]
            .parse()
            .unwrap_or_else(|_| panic!("a numeric window: {args:?}"));
        assert!(
            (MIN_COMPUTED_CONTEXT..=MAX_COMPUTED_CONTEXT).contains(&tokens),
            "{args:?}"
        );
        // The request's own flags are still handed to llama-server.
        assert!(
            args.contains(&"--llamacpp-args".to_owned())
                && args.contains(&"--temperature 0.5".to_owned()),
            "{args:?}"
        );
    }

    /// The other half of that decision: where nothing is computed, the load
    /// carries the request's own recipe exactly as it did before this module.
    #[test]
    fn a_host_with_no_shared_pool_loads_with_the_requests_own_recipe() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let mut request = request_in(temporary.path());
        let authored = EngineRecipeHint {
            required_flags: vec!["--temperature".to_owned(), "0.5".to_owned()],
            ..EngineRecipeHint::default()
        };
        request.engine_recipe = Some(authored.clone());
        let source = four_b_source();
        assert_eq!(
            recipe_for_load_from(
                &request,
                &discrete_gpu_host(&source),
                Some(CHECKPOINT.to_owned())
            ),
            Some(authored)
        );
        // And a recipe that names a window keeps it, pool or no pool.
        let mut pinned = request_in(temporary.path());
        pinned.engine_recipe = Some(EngineRecipeHint {
            required_flags: vec!["--ctx-size".to_owned(), "8192".to_owned()],
            ..EngineRecipeHint::default()
        });
        assert_eq!(
            recipe_for_load_from(
                &pinned,
                &integrated_gpu_host(&source),
                Some(CHECKPOINT.to_owned())
            ),
            pinned.engine_recipe
        );
    }

    /// The direct route never reaches the router, so it is sized separately and
    /// the window has to arrive on llama-server's own command line.
    #[test]
    fn a_direct_serve_passes_the_computed_window_to_llama_server() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let request = request_in(temporary.path());
        let server = temporary.path().join("llama-server");
        let model_path = temporary.path().join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        let source = four_b_source();

        let tokens = direct_context_size_from(&request, &integrated_gpu_host(&source), &model_path)
            .expect("a readable GGUF on an integrated-GPU host is sized");
        assert!((MIN_COMPUTED_CONTEXT..=MAX_COMPUTED_CONTEXT).contains(&tokens));
        let args = |context_size| {
            crate::direct_llama::direct_llama_server_command(
                &request,
                &server,
                &model_path,
                context_size,
            )
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
        };
        let sized = args(Some(tokens));
        let at = sized
            .iter()
            .position(|arg| arg == "--ctx-size")
            .unwrap_or_else(|| panic!("the launch must carry a context size: {sized:?}"));
        assert_eq!(sized[at + 1], tokens.to_string());

        // A discrete-GPU host is untouched: nothing is computed, and the launch
        // carries no window at all.
        assert_eq!(
            direct_context_size_from(&request, &discrete_gpu_host(&source), &model_path),
            None
        );
        assert!(!args(None).iter().any(|arg| arg == "--ctx-size"));

        // A recipe that names a window keeps it; nothing is added beside it.
        let mut pinned = request_in(temporary.path());
        pinned.engine_recipe = Some(EngineRecipeHint {
            required_flags: vec!["--ctx-size".to_owned(), "8192".to_owned()],
            ..EngineRecipeHint::default()
        });
        assert_eq!(
            direct_context_size_from(&pinned, &integrated_gpu_host(&source), &model_path),
            None
        );
    }

    /// The notes `rocm diagnose --model` prints are a claim about what
    /// `rocm serve` hands the engine, and they used to say it "passes none to
    /// the engine" -- which this module made false on both routes. Nothing in
    /// `rocm-core` can see this crate, and `serve_takes_no_estimate_options_yet`
    /// in `apps/rocm/src/cli.rs` only parses an argv, so neither side could
    /// notice: the tie has to be made here, where the window is computed.
    ///
    /// The premise is re-established rather than assumed, so a change that
    /// genuinely stopped sizing the load fails on the sizing rather than on the
    /// wording, and the notes would be free to say "none" again.
    #[test]
    fn the_serve_notes_do_not_deny_the_window_this_module_passes() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let request = request_in(temporary.path());
        let source = four_b_source();
        let host = integrated_gpu_host(&source);
        assert!(
            recipe_for_load_from(&request, &host, Some(CHECKPOINT.to_owned()))
                .is_some_and(|recipe| recipe_pins_context(Some(&recipe))),
            "the premise: the router route passes a context size on this host class"
        );
        assert!(
            direct_context_size_from(&request, &host, Path::new("Qwen3-4B-Q4_K_M.gguf")).is_some(),
            "the premise: the direct route passes one too"
        );
        for note in [
            rocm_core::gguf_fit::SERVE_OPTIONS_NOTE,
            rocm_core::gguf_fit::SERVE_CONTEXT_NOTE,
        ] {
            assert!(
                !note.contains("passes none"),
                "the note denies the window this module just computed: {note}"
            );
            assert!(
                note.contains("integrated GPU"),
                "the note must name the host class whose window is chosen for it: {note}"
            );
        }
    }

    /// A managed serve discards stdout and stderr, so a sizing that silently did
    /// not happen leaves the original symptom with no trace anywhere. Every
    /// operational failure says why — and the one non-failure says nothing.
    #[test]
    fn an_operational_failure_says_why_the_window_was_left_alone() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let mut request = request_in(temporary.path());
        let log_path = temporary.path().join("serve.log");
        request.log_path = Some(log_path.clone());
        let readable = four_b_source();
        let log = || std::fs::read_to_string(&log_path).unwrap_or_default();

        let cases: [(Option<String>, &dyn GgufSource, &str); 3] = [
            (None, &readable, "named no checkpoint for this model"),
            (
                Some("not a checkpoint".to_owned()),
                &readable,
                "`not a checkpoint` does not name a GGUF file",
            ),
            (
                Some(CHECKPOINT.to_owned()),
                &UnreadableSource,
                "could not be read",
            ),
        ];
        for (checkpoint, source, expected) in cases {
            assert_eq!(
                recipe_for_load_from(&request, &integrated_gpu_host(source), checkpoint),
                None
            );
            assert!(log().contains(expected), "{expected}: {}", log());
        }

        // A host with no shared pool is every discrete-GPU machine: this module
        // has nothing to do there and must not say so on every load.
        let quiet_path = temporary.path().join("quiet.log");
        let mut quiet_request = request_in(temporary.path());
        quiet_request.log_path = Some(quiet_path.clone());
        assert_eq!(
            recipe_for_load_from(
                &quiet_request,
                &discrete_gpu_host(&readable),
                Some(CHECKPOINT.to_owned())
            ),
            None
        );
        assert!(
            !quiet_path.exists(),
            "a host with no shared pool must write nothing"
        );
    }
}
