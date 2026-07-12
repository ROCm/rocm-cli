// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host capability probe for the E2E suite.
//!
//! The suite is black-box (it never imports rocm-cli crates), but its
//! per-scenario expectations must still follow the product's real behaviour —
//! chiefly *which serve engine the CLI would pick on this host*. We learn that
//! by spawning the real binary (`rocm examine --json` + `rocm engines list`)
//! once at startup and caching the result.
//!
//! IMPORTANT — the effective serve engine is currently RE-IMPLEMENTED here (see
//! [`effective_serve_engine`]) because no `rocm` command exposes it directly
//! (`examine`'s `default_engine` is a constant `"lemonade"` decoy, not what
//! `serve` selects). This duplicates the product's `select_serve_engine` /
//! `preferred_serve_engine_for_therock_family` logic and WILL drift if the
//! product changes engine support. Task #16 tracks adding a product probe
//! (`examine --json` → `effective_serve_engine`) so the harness can read the
//! product's own decision instead. Until then, the unit tests below guard drift.

use std::sync::OnceLock;

/// Name of the `rocm` binary under test (mirrors the test binary's
/// `rocm_binary()`): `ROCM_CLI_BINARY` or plain `rocm`.
fn rocm_binary() -> String {
    std::env::var("ROCM_CLI_BINARY").unwrap_or_else(|_| "rocm".to_string())
}

/// What `rocm serve <model>` would pick when the user does NOT pass `--engine`,
/// derived from the host's GPU family + OS. Mirrors the product precedence in
/// `preferred_serve_engine_for_host_gpu_summary` (rocm-core): vLLM on
/// data-center families (`*-dcgpu`) and gfx906/908/90a, never on native Windows;
/// otherwise the platform default, lemonade.
///
/// This is the single re-implemented rule (decision #1). When the product grows
/// an `effective_serve_engine` probe field, replace the callers with the parsed
/// field and keep this only as the drift-check reference.
pub fn effective_serve_engine(gfx_target: Option<&str>, os_family: &str) -> String {
    // The vLLM adapter bails on native Windows (WSL builds as Linux, so it
    // reports os_family "linux" and stays eligible).
    if os_family.eq_ignore_ascii_case("windows") {
        return "lemonade".to_owned();
    }
    if family_prefers_vllm(gfx_target) {
        "vllm".to_owned()
    } else {
        "lemonade".to_owned()
    }
}

/// True when a gfx target's TheRock family is vLLM-preferred: any `*-dcgpu`
/// data-center family, or the explicit gfx906/908/90a set. Mirrors
/// `preferred_serve_engine_for_therock_family` + `normalize_therock_family`.
fn family_prefers_vllm(gfx_target: Option<&str>) -> bool {
    let Some(raw) = gfx_target else {
        return false;
    };
    let family = normalize_family(raw);
    family.ends_with("-dcgpu") || matches!(family.as_str(), "gfx906" | "gfx908" | "gfx90a")
}

/// Coarse gfx-target → TheRock family normalization, matching the subset of
/// `rocm_core::normalize_therock_family` that affects engine preference. We only
/// need enough fidelity to decide vLLM-preference: the data-center families
/// (gfx94x/gfx950 → `*-dcgpu`) and the gfx906/908/90a set. Everything else
/// (e.g. gfx1151 Strix) falls through as itself → not vLLM-preferred.
fn normalize_family(raw: &str) -> String {
    let v = raw.trim().to_ascii_lowercase();
    if v.ends_with("-dcgpu") {
        return v;
    }
    if v.starts_with("gfx90a") {
        return "gfx90a".to_owned();
    }
    if v.starts_with("gfx906") {
        return "gfx906".to_owned();
    }
    if v.starts_with("gfx908") {
        return "gfx908".to_owned();
    }
    // MI300-class (gfx942/gfx94x) and gfx950 are data-center parts.
    if v.starts_with("gfx94") {
        return "gfx94X-dcgpu".to_owned();
    }
    if v.starts_with("gfx950") {
        return "gfx950-dcgpu".to_owned();
    }
    v
}

/// The probed host capability, learned once from the real `rocm` binary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HostCapability {
    /// examine --json `os_family`, lowercased (e.g. "linux", "windows", "other").
    pub os_family: String,
    /// examine --json `is_wsl`.
    pub is_wsl: bool,
    /// First AMD GPU's gfx target (e.g. "gfx942", "gfx1151"), if any.
    pub gfx_target: Option<String>,
    /// examine --json `has_amd_gpu`.
    pub has_amd_gpu: bool,
    /// Engine adapters the binary reports as present. Both builtins are always
    /// "built-in", so this is NOT the same as "can start here" — use
    /// [`HostCapability::engine_available`] for that.
    pub available_engines: Vec<String>,
    /// What `serve` picks with no `--engine` on this host (re-implemented rule).
    pub effective_serve_engine: String,
    /// Stable platform identity derived from hardware, not from an artifact name:
    /// "mock" (no AMD GPU), else the family/target (e.g. "mi300x", "strix-halo").
    pub platform_slug: String,
}

impl HostCapability {
    /// Whether a given engine can actually START on this host. Distinct from
    /// "adapter present": vLLM's adapter is built-in everywhere but cannot run on
    /// native Windows or on non-vLLM-preferred families; lemonade runs anywhere.
    pub fn engine_available(&self, engine: &str) -> bool {
        match engine {
            "lemonade" => true,
            "vllm" => {
                !self.os_family.eq_ignore_ascii_case("windows")
                    && family_prefers_vllm(self.gfx_target.as_deref())
            }
            _ => self.available_engines.iter().any(|e| e == engine),
        }
    }
}

/// Cached, probed once per process.
pub fn host_capability() -> &'static HostCapability {
    static CAP: OnceLock<HostCapability> = OnceLock::new();
    CAP.get_or_init(probe_host_capability)
}

/// Spawn the real binary in an isolated env and build a [`HostCapability`].
/// Deliberately does NOT reuse a scenario's isolated root — the probe runs
/// before any scenario, in its own throwaway temp dir.
fn probe_host_capability() -> HostCapability {
    let tmp =
        tempfile::TempDir::with_prefix("rocm-e2e-probe-").expect("failed to create probe temp dir");
    let root = tmp.path();

    let examine = run_probe(root, &["examine", "--json"]);
    let engines = run_probe(root, &["engines", "list"]);

    let (os_family, is_wsl, has_amd_gpu, gfx_target) = parse_examine_json(&examine);
    let available_engines = parse_engines_list(&engines);
    let effective_serve_engine = effective_serve_engine(gfx_target.as_deref(), &os_family);
    let platform_slug = derive_platform_slug(has_amd_gpu, gfx_target.as_deref());

    HostCapability {
        os_family,
        is_wsl,
        gfx_target,
        has_amd_gpu,
        available_engines,
        effective_serve_engine,
        platform_slug,
    }
}

/// Run `rocm <args>` with an isolated config/data/cache root, returning stdout
/// (empty string on any failure — the probe must never panic the suite).
fn run_probe(root: &std::path::Path, args: &[&str]) -> String {
    let mut cmd = std::process::Command::new(rocm_binary());
    cmd.args(args);
    cmd.env("ROCM_CLI_CONFIG_DIR", root.join("config"));
    cmd.env("ROCM_CLI_DATA_DIR", root.join("data"));
    cmd.env("ROCM_CLI_CACHE_DIR", root.join("cache"));
    match cmd.output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
        Err(_) => String::new(),
    }
}

/// Extract `(os_family, is_wsl, has_amd_gpu, first_gfx_target)` from
/// `examine --json`. Tolerant: unknown/missing fields degrade to a mock-like
/// host (os_family "other", no GPU).
fn parse_examine_json(json: &str) -> (String, bool, bool, Option<String>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return ("other".to_owned(), false, false, None);
    };
    let os_family = v
        .get("os_family")
        .and_then(|x| x.as_str())
        .unwrap_or("other")
        .to_ascii_lowercase();
    let is_wsl = v
        .get("is_wsl")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let has_amd_gpu = v
        .get("has_amd_gpu")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let gfx_target = v.get("gpus").and_then(|g| g.as_array()).and_then(|arr| {
        arr.iter().find_map(|gpu| {
            gpu.get("gfx_target")
                .and_then(|t| t.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(str::to_owned)
        })
    });
    (os_family, is_wsl, has_amd_gpu, gfx_target)
}

/// Parse engine names from `rocm engines list`. Engine rows are the lines whose
/// first non-space token is a known engine name (optionally prefixed by the `*`
/// default marker), before the indented `adapter:`/`runtime:` detail lines.
fn parse_engines_list(text: &str) -> Vec<String> {
    let mut engines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start_matches(['*', ' ']);
        if trimmed.is_empty() {
            continue;
        }
        let first = trimmed.split_whitespace().next().unwrap_or("");
        if matches!(first, "lemonade" | "vllm") && !engines.iter().any(|e| e == first) {
            engines.push(first.to_owned());
        }
    }
    engines
}

/// Stable platform identity from hardware. No AMD GPU → "mock". Otherwise a
/// coarse slug from the gfx family (data-center → "mi300x"; Strix gfx115x →
/// "strix-halo"), falling back to the normalized family.
fn derive_platform_slug(has_amd_gpu: bool, gfx_target: Option<&str>) -> String {
    if !has_amd_gpu {
        return "mock".to_owned();
    }
    match gfx_target {
        Some(t) => {
            let f = normalize_family(t);
            if f.ends_with("-dcgpu") {
                "mi300x".to_owned()
            } else if f.starts_with("gfx115") {
                "strix-halo".to_owned()
            } else {
                f
            }
        }
        None => "mock".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Drift guard (decision #1): these pin the re-implemented rule to the
    // product's known behaviour. When task #16 lands a product probe field,
    // this same table becomes the consistency check (harness rule == probe).

    #[test]
    fn mi300x_dcgpu_prefers_vllm() {
        assert_eq!(effective_serve_engine(Some("gfx942"), "linux"), "vllm");
        assert_eq!(
            effective_serve_engine(Some("gfx94X-dcgpu"), "linux"),
            "vllm"
        );
        assert_eq!(effective_serve_engine(Some("gfx950"), "linux"), "vllm");
    }

    #[test]
    fn legacy_dcgpu_set_prefers_vllm() {
        for t in ["gfx906", "gfx908", "gfx90a"] {
            assert_eq!(effective_serve_engine(Some(t), "linux"), "vllm", "{t}");
        }
    }

    #[test]
    fn strix_halo_defaults_to_lemonade() {
        // gfx1151 is NOT a vLLM-preferred family → default engine (lemonade),
        // on Linux AND Windows.
        assert_eq!(effective_serve_engine(Some("gfx1151"), "linux"), "lemonade");
        assert_eq!(
            effective_serve_engine(Some("gfx1151"), "windows"),
            "lemonade"
        );
    }

    #[test]
    fn native_windows_never_prefers_vllm() {
        // Even a data-center family cannot use vLLM on native Windows.
        assert_eq!(
            effective_serve_engine(Some("gfx942"), "windows"),
            "lemonade"
        );
    }

    #[test]
    fn no_gpu_defaults_to_lemonade() {
        assert_eq!(effective_serve_engine(None, "other"), "lemonade");
    }

    #[test]
    fn engine_available_respects_platform() {
        let strix = HostCapability {
            os_family: "windows".to_owned(),
            is_wsl: false,
            gfx_target: Some("gfx1151".to_owned()),
            has_amd_gpu: true,
            available_engines: vec!["lemonade".to_owned(), "vllm".to_owned()],
            effective_serve_engine: "lemonade".to_owned(),
            platform_slug: "strix-halo".to_owned(),
        };
        assert!(strix.engine_available("lemonade"));
        // vLLM adapter is "built-in" but cannot start on Windows / non-dcgpu.
        assert!(!strix.engine_available("vllm"));

        let mi300x = HostCapability {
            os_family: "linux".to_owned(),
            is_wsl: false,
            gfx_target: Some("gfx942".to_owned()),
            has_amd_gpu: true,
            available_engines: vec!["lemonade".to_owned(), "vllm".to_owned()],
            effective_serve_engine: "vllm".to_owned(),
            platform_slug: "mi300x".to_owned(),
        };
        assert!(mi300x.engine_available("vllm"));
        assert!(mi300x.engine_available("lemonade"));
    }

    #[test]
    fn parses_examine_json() {
        let json = r#"{"os_family":"linux","is_wsl":false,"has_amd_gpu":true,
            "gpus":[{"name":"AMD Instinct MI300X","gfx_target":"gfx942"}]}"#;
        let (os, wsl, gpu, gfx) = parse_examine_json(json);
        assert_eq!(os, "linux");
        assert!(!wsl);
        assert!(gpu);
        assert_eq!(gfx.as_deref(), Some("gfx942"));
    }

    #[test]
    fn parses_examine_json_mock_host() {
        let json = r#"{"os_family":"other","is_wsl":false,"has_amd_gpu":false,"gpus":[]}"#;
        let (os, _, gpu, gfx) = parse_examine_json(json);
        assert_eq!(os, "other");
        assert!(!gpu);
        assert_eq!(gfx, None);
    }

    #[test]
    fn parses_engines_list() {
        let text = "\
Local model engines
  Built-in engines are included with rocm-cli.
* lemonade   default embedded Lemonade server with ROCm llama.cpp backend
    adapter: built-in
    runtime: not found
  vllm       Linux/WSL ROCm GPU serving engine through external vLLM
    adapter: built-in
    runtime: not found
  protocol: 0.1.0
";
        assert_eq!(parse_engines_list(text), vec!["lemonade", "vllm"]);
    }

    #[test]
    fn platform_slug_derivation() {
        assert_eq!(derive_platform_slug(false, None), "mock");
        assert_eq!(derive_platform_slug(true, Some("gfx942")), "mi300x");
        assert_eq!(derive_platform_slug(true, Some("gfx1151")), "strix-halo");
    }
}
