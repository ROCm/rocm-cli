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
const REQUIRES_OS_PREFIX: &str = "requires-os:";
const REQUIRES_GPU_TAG: &str = "requires-gpu";
const SERVE_TIMEOUT_PREFIX: &str = "serve-timeout:";
const NIGHTLY_TAG: &str = "nightly";

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
    /// OS the scenario requires via `@requires-os:<os>` (e.g. "linux"), if any —
    /// for scenarios whose premise is OS-specific (e.g. adopting a `/opt/rocm`
    /// install only exists on Linux). Runs everywhere when absent.
    pub requires_os: Option<String>,
    /// Serve-readiness timeout (seconds) declared via `@serve-timeout:<secs>`, for
    /// a *passing* scenario that legitimately needs longer than the default — e.g.
    /// a large model whose cold load exceeds the 600s default. Distinct from the
    /// xfail `serve_timeout_secs` in expectations.toml (which shortens a known-bug
    /// serve to fail fast); this lengthens a genuinely-slow expected-pass serve.
    pub serve_timeout_secs: Option<u64>,
    /// `@nightly`: an expensive scenario (e.g. a large-model serve) that is skipped
    /// on ordinary per-PR / on-demand runs to keep them fast, and only runs when
    /// the nightly workflow opts in via `E2E_INCLUDE_NIGHTLY`.
    pub nightly: bool,
}

impl ScenarioDecl {
    /// Parse a scenario's declaration from its gherkin tags (no `@` prefix).
    pub fn from_tags<S: AsRef<str>>(tags: &[S]) -> Self {
        let mut id = None;
        let mut requires_gpu = false;
        let mut requires_engine = None;
        let mut requires_os = None;
        let mut serve_timeout_secs = None;
        let mut nightly = false;
        for tag in tags {
            let tag = tag.as_ref();
            if let Some(rest) = tag.strip_prefix(ID_PREFIX) {
                id = Some(rest.to_owned());
            } else if let Some(rest) = tag.strip_prefix(REQUIRES_ENGINE_PREFIX) {
                requires_engine = Some(rest.to_owned());
            } else if let Some(rest) = tag.strip_prefix(REQUIRES_OS_PREFIX) {
                requires_os = Some(rest.to_ascii_lowercase());
            } else if let Some(rest) = tag.strip_prefix(SERVE_TIMEOUT_PREFIX) {
                serve_timeout_secs = rest.parse::<u64>().ok();
            } else if tag == REQUIRES_GPU_TAG {
                requires_gpu = true;
            } else if tag == NIGHTLY_TAG {
                nightly = true;
            }
        }
        Self {
            id,
            requires_gpu,
            requires_engine,
            requires_os,
            serve_timeout_secs,
            nightly,
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
///
/// `deny_unknown_fields`: every field is an optional `#[serde(default)]`, and
/// `matches` returns `true` when they are all `None` (an empty condition matches
/// unconditionally). Without this, a typo'd key (e.g. `engine` for
/// `effective_engine`) would parse to an all-`None` condition = an always-xfail
/// that silently masks real regressions on every platform. Reject unknown keys.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

    /// The shortest `serve_timeout_secs` among the conditions matching this host
    /// for a scenario, if any. A known bug that manifests as a serve which never
    /// becomes ready should fail fast rather than burn the full cold-start window
    /// — so the harness applies this per scenario instead of the global default.
    pub fn serve_timeout_for(
        &self,
        id: &str,
        cap: &HostCapability,
        effective_engine: &str,
    ) -> Option<u64> {
        self.entries_for(id)
            .iter()
            .filter(|e| e.when.matches(cap, effective_engine))
            .filter_map(|e| e.serve_timeout_secs)
            .min()
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
    /// Component versions (OS/ROCm/vLLM/lemonade) for the report heading.
    pub versions: crate::capability::PlatformVersions,
    pub expectations: Vec<ResolvedScenario>,
}

/// Resolve a scenario's expectation on this host.
///
/// 1. Not-applicable → `Skip`: a `@nightly` scenario when nightly isn't included,
///    a `@requires-gpu` scenario on a host with no AMD GPU, a `@requires-os:<os>`
///    scenario on a different OS, or a scenario whose effective engine can't start.
/// 2. First matching `expectations.toml` condition → `ExpectXfail`.
/// 3. Otherwise → `ExpectPass`.
///
/// `include_nightly` is set by the nightly workflow (via `E2E_INCLUDE_NIGHTLY`);
/// ordinary per-PR / on-demand runs pass `false` so expensive `@nightly`
/// scenarios stay out of the fast path.
pub fn resolve(
    decl: &ScenarioDecl,
    cap: &HostCapability,
    matrix: &Expectations,
    include_nightly: bool,
) -> Expectation {
    // (1) Applicability / skip.
    if decl.nightly && !include_nightly {
        return Expectation::Skip {
            reason: "nightly-only scenario; set E2E_INCLUDE_NIGHTLY to run".to_owned(),
        };
    }
    if decl.requires_gpu && !cap.has_amd_gpu {
        return Expectation::Skip {
            reason: "requires an AMD GPU; none detected on this host".to_owned(),
        };
    }
    if let Some(os) = &decl.requires_os
        && !os.eq_ignore_ascii_case(&cap.os_family)
    {
        return Expectation::Skip {
            reason: format!("requires os '{os}'; this host is '{}'", cap.os_family),
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
            "id:serve-vllm-inference",
            "requires-gpu",
            "requires-engine:vllm",
        ]);
        assert_eq!(d.id.as_deref(), Some("serve-vllm-inference"));
        assert!(d.requires_gpu);
        assert_eq!(d.requires_engine.as_deref(), Some("vllm"));
        assert_eq!(d.serve_timeout_secs, None);
    }

    #[test]
    fn serve_timeout_tag_parses_seconds() {
        let d = decl(&["id:serve-large-model-inference", "serve-timeout:2400"]);
        assert_eq!(d.serve_timeout_secs, Some(2400));
        // A malformed value is ignored rather than panicking.
        let bad = decl(&["id:x", "serve-timeout:soon"]);
        assert_eq!(bad.serve_timeout_secs, None);
    }

    #[test]
    fn nightly_scenario_skips_unless_included() {
        let m = Expectations::default();
        // A @nightly GPU scenario on an applicable host: skipped when nightly is
        // NOT included (the per-PR fast path), runs when it is.
        let d = decl(&["id:big", "requires-gpu", "nightly"]);
        assert!(d.nightly);
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m, false),
            Expectation::Skip { .. }
        ));
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, true),
            Expectation::ExpectPass
        );
        // The nightly gate is cheapest-first: a @nightly scenario that ALSO can't
        // run here (no GPU) still skips regardless of the include flag.
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, true),
            Expectation::Skip { .. }
        ));
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
            resolve(&d, &cap("mi300x"), &m, false),
            Expectation::ExpectXfail { .. }
        ));
        // Strix Ubuntu: gfx1151 → lemonade default → NOT vLLM → expect-pass.
        assert_eq!(
            resolve(&d, &cap("strix-ubuntu"), &m, false),
            Expectation::ExpectPass
        );
        // Strix Windows: lemonade default → expect-pass (this is the XPASS fix).
        assert_eq!(
            resolve(&d, &cap("strix-windows"), &m, false),
            Expectation::ExpectPass
        );
    }

    #[test]
    fn requires_gpu_skips_on_mock() {
        let m = eai7333_matrix();
        let d = decl(&["id:serve-default-engine-inference", "requires-gpu"]);
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn vllm_pinned_scenario_skips_where_vllm_cannot_start() {
        let m = Expectations::default();
        // Scenario 5-style: pins vLLM.
        let d = decl(&[
            "id:serve-vllm-inference",
            "requires-gpu",
            "requires-engine:vllm",
        ]);
        // MI300X: vLLM available → not skipped (expect-pass here, no matrix entry).
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, false),
            Expectation::ExpectPass
        );
        // Strix Windows: vLLM can't start → skip (N/A).
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn requires_os_skips_on_other_os() {
        let m = Expectations::default();
        // Linux-only scenario (e.g. adopt /opt/rocm): runs on Linux hosts, skips
        // on Windows.
        let d = decl(&["id:runtime-adopt-preexisting-rejected", "requires-os:linux"]);
        // Runs on a Linux GPU host; skips where os_family != linux (windows, and
        // the "other" fixture host).
        assert_eq!(
            resolve(&d, &cap("strix-ubuntu"), &m, false),
            Expectation::ExpectPass
        );
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m, false),
            Expectation::Skip { .. }
        ));
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn non_gpu_scenario_always_runs() {
        let m = Expectations::default();
        let d = decl(&["id:examine-version"]);
        assert_eq!(
            resolve(&d, &cap("mock"), &m, false),
            Expectation::ExpectPass
        );
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, false),
            Expectation::ExpectPass
        );
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
            resolve(&d, &cap("mock"), &m, false),
            Expectation::ExpectXfail { .. }
        ));
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m, false),
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
            resolve(&d, &cap("strix-ubuntu"), &m, false),
            Expectation::ExpectXfail { .. }
        ));
        // Strix Windows (windows, lemonade) → os mismatch → expect-pass.
        assert_eq!(
            resolve(&d, &cap("strix-windows"), &m, false),
            Expectation::ExpectPass
        );
    }

    #[test]
    fn serve_timeout_applies_only_when_condition_matches() {
        let m = Expectations::parse(
            r#"
[["serve-default-engine-inference"]]
when = { effective_engine = "vllm" }
bug = "EAI-7333"
reason = "vLLM readiness gap"
serve_timeout_secs = 90
"#,
        )
        .unwrap();
        // MI300X: effective engine vllm → override applies (fail fast).
        assert_eq!(
            m.serve_timeout_for("serve-default-engine-inference", &cap("mi300x"), "vllm"),
            Some(90)
        );
        // Strix Windows: lemonade default → condition doesn't match → no override
        // (full window; the scenario is expected to pass there).
        assert_eq!(
            m.serve_timeout_for(
                "serve-default-engine-inference",
                &cap("strix-windows"),
                "lemonade"
            ),
            None
        );
        // Unknown id → no override.
        assert_eq!(m.serve_timeout_for("nope", &cap("mi300x"), "vllm"), None);
    }

    #[test]
    fn glob_matches_family() {
        assert!(glob_match("gfx94*", "gfx942"));
        assert!(glob_match("*dcgpu", "gfx94X-dcgpu"));
        assert!(!glob_match("gfx94*", "gfx1151"));
        assert!(glob_match("gfx1151", "gfx1151"));
    }
}
