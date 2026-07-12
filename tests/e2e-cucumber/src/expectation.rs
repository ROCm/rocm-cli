// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Per-(scenario × host) expectation resolution.
//!
//! Replaces the old global `@expected-failure` / `E2E_EXPECT_FAILURES` model.
//! Each scenario resolves — from its tags + the host [`capability`](crate::capability)
//! probe + the `expectations.toml` xfail matrix — to exactly one of:
//! expected-pass, expected-fail (xfail), or not-applicable (skip).
//!
//! Skip is for "this behaviour can't exist here" (a required engine can't start
//! on this host); xfail is for a declared known bug that reproduces here.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::capability::HostCapability;

/// Tag prefixes (gherkin strips the leading `@`, so we match the bare form).
const ID_PREFIX: &str = "id:";
const REQUIRES_ENGINE_PREFIX: &str = "requires-engine:";
const REQUIRES_GPU_TAG: &str = "requires-gpu";

/// The resolved expectation for one scenario on one host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expectation {
    /// Must pass; a failure is a real regression.
    ExpectPass,
    /// Declared known bug that reproduces on this host; failing is the expected
    /// outcome, passing is an XPASS (stale entry).
    ExpectXfail { bug: String, reason: String },
    /// Not applicable on this host (e.g. required engine can't start); skipped.
    Skip { reason: String },
}

/// Facts extracted from a scenario's tags (all `@`-stripped).
#[derive(Debug, Clone)]
pub struct ScenarioDecl {
    pub id: Option<String>,
    pub requires_gpu: bool,
    /// Engine the scenario pins via `@requires-engine:<e>` (if any).
    pub requires_engine: Option<String>,
}

impl ScenarioDecl {
    /// Parse a scenario's declaration from its gherkin tags (no `@` prefix).
    pub fn from_tags<S: AsRef<str>>(tags: &[S]) -> Self {
        let mut id = None;
        let mut requires_gpu = false;
        let mut requires_engine = None;
        for tag in tags {
            let tag = tag.as_ref();
            if let Some(rest) = tag.strip_prefix(ID_PREFIX) {
                id = Some(rest.to_owned());
            } else if let Some(rest) = tag.strip_prefix(REQUIRES_ENGINE_PREFIX) {
                requires_engine = Some(rest.to_owned());
            } else if tag == REQUIRES_GPU_TAG {
                requires_gpu = true;
            }
        }
        Self {
            id,
            requires_gpu,
            requires_engine,
        }
    }

    /// The engine this scenario effectively serves with: an explicit
    /// `@requires-engine` pin, else the host's default serve engine. Mirrors the
    /// product precedence (explicit `--engine` first).
    pub fn effective_engine<'a>(&'a self, cap: &'a HostCapability) -> &'a str {
        self.requires_engine
            .as_deref()
            .unwrap_or(&cap.effective_serve_engine)
    }
}

/// One xfail condition: all present keys must match (AND).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Condition {
    #[serde(default)]
    pub effective_engine: Option<String>,
    #[serde(default)]
    pub os: Option<String>,
    #[serde(default)]
    pub therock_family: Option<String>,
    #[serde(default)]
    pub is_wsl: Option<bool>,
}

impl Condition {
    /// Does this condition hold for the given host + resolved engine?
    fn matches(&self, cap: &HostCapability, effective_engine: &str) -> bool {
        if let Some(e) = &self.effective_engine
            && !e.eq_ignore_ascii_case(effective_engine)
        {
            return false;
        }
        if let Some(os) = &self.os
            && !os.eq_ignore_ascii_case(&cap.os_family)
        {
            return false;
        }
        if let Some(fam) = &self.therock_family {
            let host_fam = cap.gfx_target.as_deref().unwrap_or("");
            if !glob_match(fam, host_fam) {
                return false;
            }
        }
        if let Some(w) = self.is_wsl
            && w != cap.is_wsl
        {
            return false;
        }
        true
    }
}

/// One xfail matrix entry: a condition plus its bug/reason metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct XfailEntry {
    pub when: Condition,
    pub bug: String,
    pub reason: String,
    #[serde(default)]
    pub serve_timeout_secs: Option<u64>,
}

/// The parsed `expectations.toml`: scenario-id → list of xfail conditions (OR).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct Expectations {
    by_id: BTreeMap<String, Vec<XfailEntry>>,
}

impl Expectations {
    /// Parse from TOML text. The file is a table of `id → [conditions]`.
    pub fn parse(toml_text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_text)
    }

    /// Entries declared for a scenario id (empty if none).
    pub fn entries_for(&self, id: &str) -> &[XfailEntry] {
        self.by_id.get(id).map_or(&[], Vec::as_slice)
    }
}

/// Serializable outcome label for a resolved scenario (for `platform.json`).
impl Expectation {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::ExpectPass => "pass",
            Self::ExpectXfail { .. } => "xfail",
            Self::Skip { .. } => "skip",
        }
    }
}

/// One scenario's resolution on this host, recorded for `platform.json`.
///
/// Lets the central report reconcile expected-vs-actual by id — including
/// scenarios that were skipped, which never appear in cucumber's `report.json`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResolvedScenario {
    pub id: String,
    pub effective_engine: String,
    /// "pass" | "xfail" | "skip".
    pub expected: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ResolvedScenario {
    pub fn new(id: &str, effective_engine: &str, expectation: &Expectation) -> Self {
        let (bug, reason) = match expectation {
            Expectation::ExpectXfail { bug, reason } => (Some(bug.clone()), Some(reason.clone())),
            Expectation::Skip { reason } => (None, Some(reason.clone())),
            Expectation::ExpectPass => (None, None),
        };
        Self {
            id: id.to_owned(),
            effective_engine: effective_engine.to_owned(),
            expected: expectation.label().to_owned(),
            bug,
            reason,
        }
    }
}

/// The `platform.json` sidecar: probed capability + every scenario's resolution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlatformManifest<'a> {
    pub platform_slug: &'a str,
    pub capability: &'a HostCapability,
    pub expectations: Vec<ResolvedScenario>,
}

/// Resolve a scenario's expectation on this host.
///
/// 1. Not-applicable → `Skip`: a `@requires-gpu` scenario on a host with no AMD
///    GPU, or a scenario whose effective engine can't start here.
/// 2. First matching `expectations.toml` condition → `ExpectXfail`.
/// 3. Otherwise → `ExpectPass`.
pub fn resolve(decl: &ScenarioDecl, cap: &HostCapability, matrix: &Expectations) -> Expectation {
    // (1) Applicability / skip.
    if decl.requires_gpu && !cap.has_amd_gpu {
        return Expectation::Skip {
            reason: "requires an AMD GPU; none detected on this host".to_owned(),
        };
    }
    let engine = decl.effective_engine(cap);
    // A scenario that pins or defaults to a real engine and would actually serve
    // needs that engine to be startable here. We treat any GPU scenario with a
    // resolved engine as engine-gated; non-GPU scenarios never gate on engine.
    if decl.requires_gpu && !cap.engine_available(engine) {
        return Expectation::Skip {
            reason: format!(
                "engine '{engine}' cannot start on this host ({})",
                cap.platform_slug
            ),
        };
    }

    // (2) xfail override.
    if let Some(id) = &decl.id {
        for entry in matrix.entries_for(id) {
            if entry.when.matches(cap, engine) {
                return Expectation::ExpectXfail {
                    bug: entry.bug.clone(),
                    reason: entry.reason.clone(),
                };
            }
        }
    }

    // (3) default.
    Expectation::ExpectPass
}

/// Tiny glob: `*` matches any run of chars. Used only for `therock_family`.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern = pattern.to_ascii_lowercase();
    let text = text.to_ascii_lowercase();
    if !pattern.contains('*') {
        return pattern == text;
    }
    let mut pos = 0usize;
    let parts: Vec<&str> = pattern.split('*').collect();
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match text[pos..].find(part) {
            Some(idx) => {
                // First part must anchor at the start unless the pattern led with '*'.
                if i == 0 && idx != 0 {
                    return false;
                }
                pos += idx + part.len();
            }
            None => return false,
        }
    }
    // Last part must anchor at the end unless the pattern trailed with '*'.
    if let Some(last) = parts.last()
        && !last.is_empty()
        && !text.ends_with(last)
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(platform: &str) -> HostCapability {
        match platform {
            "mi300x" => HostCapability {
                os_family: "linux".into(),
                is_wsl: false,
                gfx_target: Some("gfx942".into()),
                has_amd_gpu: true,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "vllm".into(),
                platform_slug: "mi300x".into(),
            },
            "strix-ubuntu" => HostCapability {
                os_family: "linux".into(),
                is_wsl: false,
                gfx_target: Some("gfx1151".into()),
                has_amd_gpu: true,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "strix-halo".into(),
            },
            "strix-windows" => HostCapability {
                os_family: "windows".into(),
                is_wsl: false,
                gfx_target: Some("gfx1151".into()),
                has_amd_gpu: true,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "strix-halo".into(),
            },
            _ => HostCapability {
                os_family: "other".into(),
                is_wsl: false,
                gfx_target: None,
                has_amd_gpu: false,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "mock".into(),
            },
        }
    }

    fn decl(tags: &[&str]) -> ScenarioDecl {
        ScenarioDecl::from_tags(tags)
    }

    fn eai7333_matrix() -> Expectations {
        Expectations::parse(
            r#"
[["serve-default-engine-inference"]]
when = { effective_engine = "vllm" }
bug = "EAI-7333"
reason = "vLLM readiness gap"
serve_timeout_secs = 90
"#,
        )
        .unwrap()
    }

    #[test]
    fn tags_parse_without_at_prefix() {
        let d = decl(&[
            "id:serve-inference-response",
            "requires-gpu",
            "requires-engine:vllm",
        ]);
        assert_eq!(d.id.as_deref(), Some("serve-inference-response"));
        assert!(d.requires_gpu);
        assert_eq!(d.requires_engine.as_deref(), Some("vllm"));
    }

    #[test]
    fn effective_engine_prefers_explicit_pin() {
        let d = decl(&["id:x", "requires-gpu", "requires-engine:vllm"]);
        // Even on a lemonade-default host, an explicit vllm pin wins.
        assert_eq!(d.effective_engine(&cap("strix-windows")), "vllm");
        let d2 = decl(&["id:y", "requires-gpu"]);
        // No pin → host default.
        assert_eq!(d2.effective_engine(&cap("mi300x")), "vllm");
        assert_eq!(d2.effective_engine(&cap("strix-windows")), "lemonade");
    }

    // The core regression test for the run #543 XPASS: default-engine EAI-7333
    // scenarios must xfail on vLLM hosts but expect-pass on lemonade hosts.
    #[test]
    fn eai7333_default_engine_xfails_only_on_vllm() {
        let m = eai7333_matrix();
        let d = decl(&["id:serve-default-engine-inference", "requires-gpu"]);

        // MI300X: default engine vLLM → xfail.
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m),
            Expectation::ExpectXfail { .. }
        ));
        // Strix Ubuntu: gfx1151 → lemonade default → NOT vLLM → expect-pass.
        assert_eq!(
            resolve(&d, &cap("strix-ubuntu"), &m),
            Expectation::ExpectPass
        );
        // Strix Windows: lemonade default → expect-pass (this is the XPASS fix).
        assert_eq!(
            resolve(&d, &cap("strix-windows"), &m),
            Expectation::ExpectPass
        );
    }

    #[test]
    fn requires_gpu_skips_on_mock() {
        let m = eai7333_matrix();
        let d = decl(&["id:serve-default-engine-inference", "requires-gpu"]);
        assert!(matches!(
            resolve(&d, &cap("mock"), &m),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn vllm_pinned_scenario_skips_where_vllm_cannot_start() {
        let m = Expectations::default();
        // Scenario 5-style: pins vLLM.
        let d = decl(&[
            "id:serve-inference-response",
            "requires-gpu",
            "requires-engine:vllm",
        ]);
        // MI300X: vLLM available → not skipped (expect-pass here, no matrix entry).
        assert_eq!(resolve(&d, &cap("mi300x"), &m), Expectation::ExpectPass);
        // Strix Windows: vLLM can't start → skip (N/A).
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn non_gpu_scenario_always_runs() {
        let m = Expectations::default();
        let d = decl(&["id:examine-version"]);
        assert_eq!(resolve(&d, &cap("mock"), &m), Expectation::ExpectPass);
        assert_eq!(resolve(&d, &cap("mi300x"), &m), Expectation::ExpectPass);
    }

    #[test]
    fn unconditional_xfail_applies_everywhere() {
        let m = Expectations::parse(
            r#"
[["serve-short-name-expansion"]]
when = {}
bug = "EAI-7219"
reason = "short-name not surfaced"
"#,
        )
        .unwrap();
        let d = decl(&["id:serve-short-name-expansion"]);
        // No requires-gpu → runs everywhere, always xfail.
        assert!(matches!(
            resolve(&d, &cap("mock"), &m),
            Expectation::ExpectXfail { .. }
        ));
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m),
            Expectation::ExpectXfail { .. }
        ));
    }

    #[test]
    fn os_condition_matches() {
        let m = Expectations::parse(
            r#"
[["serve-lemonade-inference"]]
when = { effective_engine = "lemonade", os = "linux" }
bug = "EAI-7052"
reason = "lemonade vulkan fallback"
"#,
        )
        .unwrap();
        let d = decl(&[
            "id:serve-lemonade-inference",
            "requires-gpu",
            "requires-engine:lemonade",
        ]);
        // Strix Ubuntu (linux, lemonade) → xfail.
        assert!(matches!(
            resolve(&d, &cap("strix-ubuntu"), &m),
            Expectation::ExpectXfail { .. }
        ));
        // Strix Windows (windows, lemonade) → os mismatch → expect-pass.
        assert_eq!(
            resolve(&d, &cap("strix-windows"), &m),
            Expectation::ExpectPass
        );
    }

    #[test]
    fn glob_matches_family() {
        assert!(glob_match("gfx94*", "gfx942"));
        assert!(glob_match("*dcgpu", "gfx94X-dcgpu"));
        assert!(!glob_match("gfx94*", "gfx1151"));
        assert!(glob_match("gfx1151", "gfx1151"));
    }
}
