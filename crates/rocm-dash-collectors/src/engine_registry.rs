//! Engine registry seam — one place that maps a discovered serving engine to its per-engine sample parser.
//!
//! The daemon can pick a backend per service instead
//! of hard-coding vLLM. The existing vLLM Prometheus parser
//! (`vllm_prom::parse`) is reachable through this seam **unchanged**; Lemonade
//! adds a second parser. This is a focused seam, not a rewrite.

use rocm_dash_core::traits::InstanceSample;

/// A known inference-serving engine kind, keyed by how its live stats are scraped
/// and parsed. New engines slot in here without touching the daemon loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// vLLM — Prometheus `/metrics` text exposition.
    Vllm,
    /// Lemonade Server — JSON `/api/v1/stats`.
    Lemonade,
    /// llama.cpp `llama-server` — `/slots` (parser not yet wired).
    LlamaCpp,
}

impl EngineKind {
    /// Stable lowercase label (matches `rocm engines` / config engine names).
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Lemonade => "lemonade",
            Self::LlamaCpp => "llama.cpp",
        }
    }

    /// The engine's conventional local port. **Fallback only** — for managed
    /// services the bound port comes from the registry
    /// (`ManagedServiceRecord.port`); this default is used only for
    /// unmanaged/external discovery where no registry record exists.
    pub const fn default_port(self) -> u16 {
        match self {
            Self::Vllm => 8000,
            Self::Lemonade => crate::lemonade::LEMONADE_PORT, // 13305
            Self::LlamaCpp => 8080,
        }
    }

    /// Map an engine label (config / discovery) to a kind. Case-insensitive;
    /// accepts the `llama.cpp` / `llamacpp` / `llama_cpp` spellings.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "vllm" => Some(Self::Vllm),
            "lemonade" => Some(Self::Lemonade),
            "llama.cpp" | "llamacpp" | "llama_cpp" => Some(Self::LlamaCpp),
            _ => None,
        }
    }

    /// Parse this engine's raw scrape body into an [`InstanceSample`] using the
    /// engine-appropriate parser. vLLM dispatches to the **unchanged**
    /// `vllm_prom::parse`; Lemonade to `lemonade::parse_stats`. Unwired engines
    /// return an empty sample (never panic).
    pub fn parse_sample(self, body: &str) -> InstanceSample {
        match self {
            Self::Vllm => crate::vllm_prom::parse(body),
            Self::Lemonade => crate::lemonade::parse_stats(body),
            Self::LlamaCpp => InstanceSample::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_kind_labels_and_ports_and_from_label_roundtrip() {
        for kind in [EngineKind::Vllm, EngineKind::Lemonade, EngineKind::LlamaCpp] {
            assert_eq!(EngineKind::from_label(kind.label()), Some(kind));
        }
        assert_eq!(EngineKind::Lemonade.default_port(), 13305);
        assert_eq!(EngineKind::Vllm.default_port(), 8000);
        assert_eq!(
            EngineKind::from_label("LLAMACPP"),
            Some(EngineKind::LlamaCpp)
        );
        assert_eq!(EngineKind::from_label("nope"), None);
    }

    #[test]
    fn parse_sample_dispatches_to_the_engine_parser() {
        // vLLM Prometheus text → the unchanged vLLM parser (kv-cache 0..1 → 0..100).
        let vllm_body = "vllm:gpu_cache_usage_perc 0.5\nvllm:num_requests_running 3\n";
        let vllm_sample = EngineKind::Vllm.parse_sample(vllm_body);
        assert_eq!(vllm_sample.kv_cache_usage_pct, Some(50.0));
        assert_eq!(vllm_sample.running_reqs, Some(3));
        assert_eq!(vllm_sample.gen_tps, None); // counter path, no direct rate

        // Lemonade JSON → the Lemonade parser (rate → gen_tps).
        let lemo_body = r#"{"tokens_per_second": 42.0}"#;
        let lemo_sample = EngineKind::Lemonade.parse_sample(lemo_body);
        assert_eq!(lemo_sample.gen_tps, Some(42.0));
        assert_eq!(lemo_sample.kv_cache_usage_pct, None);
    }
}
