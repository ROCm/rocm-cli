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

use rocm_core::gguf::parse_gguf_header;
use rocm_core::gguf_fit::{FitRequest, KvCacheType, estimate_fit};
use rocm_core::gguf_source::{GgufRef, GgufSource, LiveGgufSource, parse_gguf_ref};
use rocm_core::model_readiness::AcceleratorMemory;
use rocm_core::{AppPaths, format_http_base_url, http_get_text_with_auth};
use rocm_engine_protocol::{ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;

use crate::ServeHttpRequest;
use crate::process::{model_names_match, split_context_size_flag};

/// The largest context asked for, however much memory there is.
///
/// A chat-sized window: past this the KV cache costs memory the operating system
/// shares for a window the assistant does not use. It is a default for a
/// computed size, not a limit on the model.
const MAX_COMPUTED_CONTEXT: u64 = 32_768;

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

/// [`shared_pool_for`] for this host.
fn host_shared_pool_gib() -> Option<f64> {
    shared_pool_for(
        rocm_core::detect_host_gfx_target().as_deref(),
        rocm_core::detect_host_unified_memory_gib(),
    )
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

/// Where to read a checkpoint's GGUF header: the cached file when exactly one
/// matches, else the repository on Hugging Face (a header read, never the
/// weights).
fn reference_for_checkpoint(paths: Option<&AppPaths>, checkpoint: &str) -> Option<GgufRef> {
    let cached = paths.zip(crate::direct_llama::parse_hf_checkpoint(checkpoint));
    if let Some((paths, checkpoint)) = cached {
        let mut matches = crate::direct_llama::hf_checkpoint_gguf_matches(paths, &checkpoint);
        if matches.len() == 1 {
            return matches.pop().map(GgufRef::Local);
        }
    }
    parse_gguf_ref(checkpoint)
}

/// Choose a context for the model at `reference` against a shared pool of
/// `pool_gib`, or `None` when the header cannot be read or sized, or not even a
/// small window fits.
fn computed_context(
    pool_gib: f64,
    system_ram_gib: Option<f64>,
    source: &dyn GgufSource,
    reference: &GgufRef,
) -> Option<ComputedContext> {
    let mut opened = source.open(reference).ok()?;
    let header = parse_gguf_header(&mut opened.reader).ok()?;
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
    let tokens = estimate.max_context?.min(MAX_COMPUTED_CONTEXT);
    Some(ComputedContext {
        tokens,
        notice: format!(
            "Context window: {tokens} tokens, chosen from the model file and the {pool_gib:.1} GiB \
             of memory this integrated GPU shares with the operating system."
        ),
    })
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

/// The recipe a model load should use: the request's own with a context size
/// added when this host has a shared GPU pool and the model's size is known, or
/// `None` to load with the request's recipe unchanged.
pub(crate) fn recipe_for_load(
    request: &ServeHttpRequest,
    model_ref: &str,
) -> Option<EngineRecipeHint> {
    if recipe_pins_context(request.engine_recipe.as_ref()) {
        return None;
    }
    let pool_gib = host_shared_pool_gib()?;
    let checkpoint = query_checkpoint(&request.host, request.port, model_ref)?;
    let reference = reference_for_checkpoint(AppPaths::discover().ok().as_ref(), &checkpoint)?;
    let computed = computed_context(
        pool_gib,
        rocm_core::detect_system_ram_gib(),
        &LiveGgufSource::new(),
        &reference,
    )?;
    announce(&computed.notice, request.log_path.as_deref());
    Some(recipe_with_context(
        request.engine_recipe.as_ref(),
        computed.tokens,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::gguf_source::OpenedGguf;
    use serde_json::json;
    use std::io::Cursor;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// A Qwen3-4B-shaped header: 36 layers of 8 KV heads with key and value
    /// width 128, so 147,456 bytes of f16 KV cache per token, and a 262,144
    /// token training context.
    fn qwen_shaped_header() -> Vec<u8> {
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
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&9_u64.to_le_bytes()); // metadata entries
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
        let computed = computed_context(7.7, Some(15.0), &four_b_source(), &reference())
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
        let computed = computed_context(pool_gib, None, &four_b_source(), &reference()).unwrap();
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
        let computed = computed_context(96.0, Some(128.0), &four_b_source(), &reference()).unwrap();
        assert_eq!(computed.tokens, MAX_COMPUTED_CONTEXT);
    }

    #[test]
    fn a_pool_the_model_does_not_fit_in_leaves_the_load_to_lemonade() {
        assert_eq!(
            computed_context(2.0, Some(4.0), &four_b_source(), &reference()),
            None
        );
    }

    #[test]
    fn an_unreadable_header_leaves_the_load_to_lemonade() {
        assert_eq!(
            computed_context(7.7, Some(15.0), &UnreadableSource, &reference()),
            None
        );
        let not_gguf = FakeSource {
            header: b"not a gguf file".to_vec(),
            size: GIB,
        };
        assert_eq!(
            computed_context(7.7, Some(15.0), &not_gguf, &reference()),
            None
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
}
