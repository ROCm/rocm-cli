// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat-template defaults a locally served model needs on an
//! OpenAI-compatible chat request. Pure: a decision over a model id, no I/O.
//!
//! Two front ends send that request — the `rocm` CLI's local chat provider and
//! the dashboard's chat client — and both must decide this the same way. They
//! did not: the CLI turned hybrid thinking off while the dashboard left it on,
//! so the same locally served model answered in one and reasoned past the
//! request deadline in the other. The rule therefore lives here, in the leaf
//! crate both already depend on, as one copy with one set of tests.

/// Whether a locally served `model` should have its chat template's hybrid
/// "thinking" mode turned off — i.e. whether the request body needs
/// `chat_template_kwargs.enable_thinking = false`.
///
/// Qwen3 chat templates enable thinking by default with an effectively
/// unbounded reasoning budget, so a small quantized build on modest hardware
/// can spend an entire request generating reasoning tokens and return no
/// visible answer at all. Matching is case-insensitive and by substring, so a
/// repository-qualified and quantization-suffixed id such as
/// `Qwen/Qwen3-1.7B-GGUF:Q4_K_M` matches too.
///
/// The decision is made on the name alone and is therefore approximate: every
/// id that spells `qwen3` anywhere matches, whether or not what is actually
/// served behind that name has a hybrid-thinking chat template.
pub fn local_model_should_disable_thinking(model: &str) -> bool {
    model.to_ascii_lowercase().contains("qwen3")
}

#[cfg(test)]
mod tests {
    use super::local_model_should_disable_thinking;

    #[test]
    fn qwen3_ids_match_in_every_spelling_a_serving_endpoint_reports() {
        // A served model id reaches this helper in whatever shape the engine
        // reports it: bare, repository-qualified, quantization-suffixed, and
        // in any case. All of them name the same chat template. `qwen32b` is
        // the awkward one: a catalog alias where the parameter count runs
        // straight on from `qwen3` with no separator at all.
        for model in [
            "qwen3",
            "Qwen3-0.6B-GGUF",
            "Qwen/Qwen3-8B",
            "QWEN3-32B-AWQ",
            "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
            "qwen32b",
        ] {
            assert!(
                local_model_should_disable_thinking(model),
                "`{model}` names a hybrid-thinking chat template and must match"
            );
        }
    }

    #[test]
    fn later_qwen3_families_match_on_the_strength_of_the_name_alone() {
        // Pins the approximation the doc comment above describes, rather than
        // endorsing it. `qwen3.5` and `qwen3.6-27b` are real catalog aliases
        // (`crates/rocm-core/src/model_catalog.json`) that spell `qwen3` as a
        // prefix, so they match on their names alone — nothing here has
        // looked at the chat template actually served behind them. Narrowing
        // the rule to exclude them would change behaviour for a model that
        // matches today, so it has to be a deliberate decision taken against
        // this test and that doc comment together.
        for model in ["qwen3.5", "qwen3.6-27b"] {
            assert!(
                local_model_should_disable_thinking(model),
                "`{model}` matches by name today; excluding it is a behaviour \
                 change, not a refactor"
            );
        }
    }

    #[test]
    fn other_models_are_left_alone() {
        // Everything else keeps the server's own chat-template defaults: the
        // caller sends no `chat_template_kwargs` at all, so a backend that
        // rejects or misreads the key is never handed one. `qwen2.5` is the
        // near miss that matters — same family, no hybrid thinking mode.
        for model in [
            "local-model",
            "qwen2.5-7b-instruct",
            "Qwen/Qwen2.5-Coder-32B",
            "llama-3.1-8b-instruct",
            "gpt-4o-mini",
            "",
        ] {
            assert!(
                !local_model_should_disable_thinking(model),
                "`{model}` must keep the server's own chat-template defaults"
            );
        }
    }
}
