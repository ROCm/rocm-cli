// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which Lemonade recipe serves a model: llama.cpp, or Gufo for the models
//! Lemonade's own registry assigns to it.
//!
//! Lemonade ships its model registry as `resources/server_models.json` inside
//! the runtime tree, keyed by model name, and each entry names the recipe that
//! loads it. Gufo is a separate recipe there, not a llama.cpp backend: it runs
//! only the handful of models it ships kernels for, so the registry is the one
//! place that says a model is a Gufo model. Everything else (a name the
//! registry does not know, a Hugging Face checkpoint, a local GGUF) keeps the
//! llama.cpp path it always had.

use std::path::Path;

use serde_json::Value;

use crate::install::LemonadeInstallManifest;
use crate::state::resolve_lemonade_model_ref;

/// Lemonade's name for the Gufo recipe. Here rather than in `gufo.rs` so
/// `process.rs` can name it without importing `gufo.rs`, which imports it.
pub const GUFO_RECIPE: &str = "gufo";

/// The registry file, relative to the runtime directory.
pub(crate) const SERVER_MODELS_JSON: &str = "resources/server_models.json";

/// A Lemonade recipe the engine knows how to serve a model with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelRecipe {
    /// Lemonade's llama.cpp recipe on a GPU backend (ROCm, else Vulkan).
    LlamaCpp,
    /// Lemonade's Gufo recipe: one `rocm` backend, run from a container image,
    /// on AMD Strix Halo (gfx1151) under Linux only.
    Gufo,
}

impl ModelRecipe {
    /// The recipe's name as Lemonade spells it (`lemonade backends`, `/health`).
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCpp => crate::LLAMACPP_RECIPE,
            Self::Gufo => GUFO_RECIPE,
        }
    }

    /// The recipe's name in a sentence ("a supported Lemonade llama.cpp backend").
    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Gufo => "Gufo",
        }
    }

    /// Parse the `recipe` a service state records. A state written before the
    /// field existed was always a llama.cpp serve, so absence means llama.cpp.
    pub(crate) fn from_state(state: Option<&Value>) -> Self {
        match state
            .and_then(|state| state.get("recipe"))
            .and_then(Value::as_str)
        {
            Some(recipe) if recipe == GUFO_RECIPE => Self::Gufo,
            _ => Self::LlamaCpp,
        }
    }

    /// The recipe that serves `model_ref` on this runtime.
    pub(crate) fn for_model(manifest: &LemonadeInstallManifest, model_ref: &str) -> Self {
        Self::for_model_in(&manifest.runtime_dir, model_ref)
    }

    /// [`Self::for_model`] against a runtime directory, for tests.
    pub(crate) fn for_model_in(runtime_dir: &Path, model_ref: &str) -> Self {
        match registered_recipe(runtime_dir, model_ref).as_deref() {
            Some(recipe) if recipe == GUFO_RECIPE => Self::Gufo,
            _ => Self::LlamaCpp,
        }
    }
}

/// The recipe Lemonade's registry names for `model_ref`, after the engine's
/// own shorthand normalization. `None` when the registry is missing, is not
/// JSON, or has no entry for the name: the caller treats all three as "not a
/// registered model", which is what they meant before this lookup existed.
pub(crate) fn registered_recipe(runtime_dir: &Path, model_ref: &str) -> Option<String> {
    let registry = read_registry(runtime_dir)?;
    registry
        .get(resolve_lemonade_model_ref(model_ref))?
        .get("recipe")?
        .as_str()
        .map(str::to_owned)
}

/// The names Lemonade's registry assigns to `recipe`, sorted.
pub(crate) fn registered_models_for(runtime_dir: &Path, recipe: ModelRecipe) -> Vec<String> {
    let Some(registry) = read_registry(runtime_dir) else {
        return Vec::new();
    };
    let Some(entries) = registry.as_object() else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .iter()
        .filter(|(_, entry)| entry.get("recipe").and_then(Value::as_str) == Some(recipe.as_str()))
        .map(|(name, _)| name.clone())
        .collect();
    names.sort();
    names
}

fn read_registry(runtime_dir: &Path) -> Option<Value> {
    let bytes = std::fs::read(runtime_dir.join(SERVER_MODELS_JSON)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn runtime_with_registry(registry: &Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SERVER_MODELS_JSON);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_vec(registry).unwrap()).unwrap();
        dir
    }

    /// Entries shaped like the Lemonade 2026.47.1 registry's.
    fn registry() -> Value {
        json!({
            "Qwen3.8-27B-Gufo": {
                "checkpoints": {"main": "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_XL.gguf"},
                "recipe": "gufo",
                "suggested": true,
                "size": 19.6
            },
            "DeepSeek-V4-Flash-IQ2XXS-Gufo": {"recipe": "gufo"},
            "Qwen3-0.6B-GGUF": {"checkpoint": "unsloth/Qwen3-0.6B-GGUF:Q4_0", "recipe": "llamacpp"},
            "Whisper-Tiny": {"recipe": "whispercpp"}
        })
    }

    #[test]
    fn registry_recipe_decides_gufo_and_everything_else_stays_llamacpp() {
        let runtime = runtime_with_registry(&registry());
        let cases = [
            ("Qwen3.8-27B-Gufo", ModelRecipe::Gufo),
            ("DeepSeek-V4-Flash-IQ2XXS-Gufo", ModelRecipe::Gufo),
            ("Qwen3-0.6B-GGUF", ModelRecipe::LlamaCpp),
            // A name the registry does not know, and an HF checkpoint, keep the
            // llama.cpp path: nothing says they are Gufo models.
            ("Not-A-Registered-Model", ModelRecipe::LlamaCpp),
            ("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL", ModelRecipe::LlamaCpp),
            // Another recipe the engine does not serve is not mistaken for Gufo.
            ("Whisper-Tiny", ModelRecipe::LlamaCpp),
        ];
        for (model, expected) in cases {
            assert_eq!(
                ModelRecipe::for_model_in(runtime.path(), model),
                expected,
                "{model}"
            );
        }
    }

    #[test]
    fn a_missing_or_unreadable_registry_is_not_a_gufo_model() {
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            ModelRecipe::for_model_in(empty.path(), "Qwen3.8-27B-Gufo"),
            ModelRecipe::LlamaCpp
        );
        let torn = tempfile::tempdir().unwrap();
        let path = torn.path().join(SERVER_MODELS_JSON);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{\"Qwen3.8-27B-Gufo\": {\"reci").unwrap();
        assert_eq!(
            ModelRecipe::for_model_in(torn.path(), "Qwen3.8-27B-Gufo"),
            ModelRecipe::LlamaCpp
        );
    }

    // Sorted as well, but this cannot show it: without serde_json's
    // `preserve_order` a registry is already read in key order. `sort` is there
    // for a build that enables it.
    #[test]
    fn registered_models_lists_only_that_recipe() {
        let runtime = runtime_with_registry(&registry());
        assert_eq!(
            registered_models_for(runtime.path(), ModelRecipe::Gufo),
            ["DeepSeek-V4-Flash-IQ2XXS-Gufo", "Qwen3.8-27B-Gufo"]
        );
    }

    #[test]
    fn a_state_without_a_recipe_reads_as_llamacpp() {
        assert_eq!(ModelRecipe::from_state(None), ModelRecipe::LlamaCpp);
        assert_eq!(
            ModelRecipe::from_state(Some(&json!({"backend_requested": "rocm"}))),
            ModelRecipe::LlamaCpp
        );
        assert_eq!(
            ModelRecipe::from_state(Some(&json!({"recipe": "gufo"}))),
            ModelRecipe::Gufo
        );
    }
}
