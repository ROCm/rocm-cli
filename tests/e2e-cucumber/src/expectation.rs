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

/// Tag prefixes in their normalized bare form. cucumber-rs currently supplies
/// scenario tags with a leading `@`; [`ScenarioDecl::from_tags`] accepts either
/// shape so direct unit fixtures and runtime parser output cannot drift.
const ID_PREFIX: &str = "id:";
const REQUIRES_ENGINE_PREFIX: &str = "requires-engine:";
const REQUIRES_OS_PREFIX: &str = "requires-os:";
const REQUIRES_GPU_TAG: &str = "requires-gpu";
const REQUIRES_MULTI_GPU_TAG: &str = "requires-multi-gpu";
const REQUIRES_GFX_TARGET_TAG: &str = "requires-gfx-target";
const REQUIRES_NO_GPU_TAG: &str = "requires-no-gpu";
const REQUIRES_BARE_METAL_TAG: &str = "requires-bare-metal";
const REQUIRES_WSL_TAG: &str = "requires-wsl";
const SERVE_TIMEOUT_PREFIX: &str = "serve-timeout:";
const NIGHTLY_TAG: &str = "nightly";
const LIFECYCLE_TAG: &str = "lifecycle";
const MERGE_QUEUE_TAG: &str = "merge-queue";

/// The resolved expectation for one scenario on one host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expectation {
    /// Must pass; a failure is a real regression.
    ExpectPass,
    /// Declared known bug that reproduces on this host; failing is the expected
    /// outcome. A deterministic pass is a stale XPASS; a `flaky` expectation
    /// tolerates either result while the intermittent bug remains open.
    ExpectXfail {
        bug: String,
        reason: String,
        flaky: bool,
    },
    /// Not applicable on this host (e.g. required engine can't start); skipped.
    Skip { reason: String },
}

/// Facts extracted from a scenario's tags.
#[derive(Debug, Clone)]
pub struct ScenarioDecl {
    pub id: Option<String>,
    pub requires_gpu: bool,
    /// `@requires-multi-gpu`: the scenario's premise is a host with MORE THAN ONE
    /// AMD GPU present, so a single-GPU host is not a weaker version of it — it
    /// is a different situation in which the assertion does not hold. Skipped
    /// wherever the probed device count is not known to exceed one.
    ///
    /// `@requires-gpu` cannot express this: it asks only whether *a* device is
    /// usable, and Strix Halo answers yes with exactly one. The distinction is
    /// load-bearing for the visibility-mask ordinal scenarios, where a mask token
    /// that names a device the host does not have makes the product's ordinal
    /// probe report "unknown" rather than a visible set, and `--gpu` validation
    /// then takes its permissive fallback instead of refusing.
    pub requires_multi_gpu: bool,
    /// `@requires-gfx-target`: the scenario needs a detected chip name but does
    /// not access the GPU. This permits resolver dry-runs on WSL before GPU
    /// passthrough is ready without weakening `@requires-gpu` serve scenarios.
    pub requires_gfx_target: bool,
    /// `@requires-no-gpu`: the scenario's premise is a host with NO usable AMD GPU
    /// (e.g. a GPU-required serve must fail fast). Skipped on any host that has a
    /// GPU — the inverse of `requires_gpu`. This is how the no-GPU fail-fast path
    /// gets live coverage: it runs on the GitHub-hosted mock job.
    pub requires_no_gpu: bool,
    /// `@requires-bare-metal`: the scenario's premise is a host running the
    /// in-tree amdgpu driver, so it does not hold under WSL2 — which uses
    /// `/dev/dxg` and the Windows host driver instead.
    ///
    /// Two things stop short there. `rocm diagnose` skips its bare-metal catalog
    /// *by design* rather than emitting Linux diagnoses that cannot apply, so a
    /// scenario needing a catalog match has no premise. And `examine`'s probe
    /// returns as soon as it recognises WSL2, before most of its steps run, so a
    /// scenario asserting anything those steps populate has none either.
    ///
    /// This is a SKIP and not an xfail row: the diagnose half is deliberate and
    /// separately unit-tested, and recording it as a known bug would tell every
    /// later reader the opposite. (How much the probe gives up on WSL2 is a
    /// separate question, and a separate ticket.)
    ///
    /// `@requires-os:linux` cannot express this — WSL2 reports an os_family of
    /// `linux`, so it matches.
    pub requires_bare_metal: bool,
    /// `@requires-wsl`: the exact inverse — the scenario's premise IS a WSL2
    /// host, so it is skipped on native Linux, native Windows and everything
    /// else. Same reason `@requires-os:linux` cannot stand in for it.
    pub requires_wsl: bool,
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
    /// `@lifecycle`: an expensive, OS-mutating release-lifecycle scenario
    /// (packaging, real installer, install/uninstall) that is skipped by default
    /// so the fast E2E suite stays fast, and only runs when the caller opts in via
    /// `E2E_INCLUDE_LIFECYCLE`. Runs explicitly in heavy CI and on demand.
    pub lifecycle: bool,
    /// `@merge-queue`: a heavy real-GPU serve that is redundant with a cheaper
    /// per-engine canary on the fast path, so it is skipped on ordinary per-PR /
    /// on-demand runs and only runs in the merge queue, which opts in via
    /// `E2E_MERGE_QUEUE`. Keeps the PR feedback loop short while still exercising
    /// the full serve matrix before a change lands.
    pub merge_queue: bool,
}

impl ScenarioDecl {
    /// Parse a scenario's declaration from its gherkin tags, accepting the
    /// leading `@` shape cucumber-rs supplies at runtime and bare unit fixtures.
    pub fn from_tags<S: AsRef<str>>(tags: &[S]) -> Self {
        let mut id = None;
        let mut requires_gpu = false;
        let mut requires_multi_gpu = false;
        let mut requires_gfx_target = false;
        let mut requires_no_gpu = false;
        let mut requires_bare_metal = false;
        let mut requires_wsl = false;
        let mut requires_engine = None;
        let mut requires_os = None;
        let mut serve_timeout_secs = None;
        let mut nightly = false;
        let mut lifecycle = false;
        let mut merge_queue = false;
        for tag in tags {
            let tag = tag
                .as_ref()
                .strip_prefix('@')
                .unwrap_or_else(|| tag.as_ref());
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
            } else if tag == REQUIRES_MULTI_GPU_TAG {
                requires_multi_gpu = true;
            } else if tag == REQUIRES_GFX_TARGET_TAG {
                requires_gfx_target = true;
            } else if tag == REQUIRES_NO_GPU_TAG {
                requires_no_gpu = true;
            } else if tag == REQUIRES_BARE_METAL_TAG {
                requires_bare_metal = true;
            } else if tag == REQUIRES_WSL_TAG {
                requires_wsl = true;
            } else if tag == NIGHTLY_TAG {
                nightly = true;
            } else if tag == LIFECYCLE_TAG {
                lifecycle = true;
            } else if tag == MERGE_QUEUE_TAG {
                merge_queue = true;
            }
        }
        Self {
            id,
            requires_gpu,
            requires_multi_gpu,
            requires_gfx_target,
            requires_no_gpu,
            requires_bare_metal,
            requires_wsl,
            requires_engine,
            requires_os,
            serve_timeout_secs,
            nightly,
            lifecycle,
            merge_queue,
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
    pub has_amd_gpu: Option<bool>,
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
        if let Some(has_gpu) = self.has_amd_gpu
            && has_gpu != cap.has_amd_gpu
        {
            return false;
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
    /// Non-deterministic known bug: an XPASS is expected and non-fatal.
    #[serde(default)]
    pub flaky: bool,
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

    /// Every scenario id this file declares a condition for.
    pub fn declared_ids(&self) -> impl Iterator<Item = &str> {
        self.by_id.keys().map(String::as_str)
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

    /// Whether any declared condition for this scenario matches this host — the
    /// scenario is a known bug here and failing is its expected outcome.
    ///
    /// This is the same test [`resolve`] applies in its xfail step, exposed on
    /// its own so a step can ask "is this run expected to fail?" without
    /// re-deriving applicability. Steps use it to spend effort where it changes
    /// a result: a serve backing an expect-pass scenario is worth relaunching
    /// after a stall, one backing a known bug is not.
    pub fn is_xfail(&self, id: &str, cap: &HostCapability, effective_engine: &str) -> bool {
        self.entries_for(id)
            .iter()
            .any(|e| e.when.matches(cap, effective_engine))
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
    /// The `Feature:` this scenario belongs to. Recorded here because a SKIPPED
    /// scenario never reaches `report.json`, so the report has no other way to
    /// place it under its feature in the grouped grid.
    ///
    /// No `#[serde(default)]` here: this struct only derives `Serialize`, so a
    /// deserialization attribute would be dead. Backward compatibility for
    /// artifacts written before these fields existed lives entirely on the
    /// consuming side — `ManifestExpectation` in the `e2e-report` crate.
    pub feature: String,
    /// The scenario's own name (`<key>-<NN> - <description>`). Carries the
    /// per-feature index the report sorts rows by, and gives skipped scenarios a
    /// human label they'd otherwise lack.
    pub scenario: String,
    pub effective_engine: String,
    /// "pass" | "xfail" | "skip".
    pub expected: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub flaky: bool,
}

impl ResolvedScenario {
    pub fn new(
        id: &str,
        feature: &str,
        scenario: &str,
        effective_engine: &str,
        expectation: &Expectation,
    ) -> Self {
        let (bug, reason, flaky) = match expectation {
            Expectation::ExpectXfail { bug, reason, flaky } => {
                (Some(bug.clone()), Some(reason.clone()), *flaky)
            }
            Expectation::Skip { reason } => (None, Some(reason.clone()), false),
            Expectation::ExpectPass => (None, None, false),
        };
        Self {
            id: id.to_owned(),
            feature: feature.to_owned(),
            scenario: scenario.to_owned(),
            effective_engine: effective_engine.to_owned(),
            expected: expectation.label().to_owned(),
            bug,
            reason,
            flaky,
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
///    a `@merge-queue` scenario outside the merge queue, a `@requires-gpu`
///    scenario on a host with no AMD GPU, a `@requires-multi-gpu` scenario on a
///    host that does not have more than one, a `@requires-bare-metal` scenario on
///    WSL2, a `@requires-os:<os>` scenario on a different OS, or a scenario whose
///    effective engine can't start.
/// 2. First matching `expectations.toml` condition → `ExpectXfail`.
/// 3. Otherwise → `ExpectPass`.
///
/// `include_nightly` is set by the nightly workflow (via `E2E_INCLUDE_NIGHTLY`);
/// ordinary per-PR / on-demand runs pass `false` so expensive `@nightly`
/// scenarios stay out of the fast path. `include_lifecycle` is set (via
/// `E2E_INCLUDE_LIFECYCLE`) only when the caller opts into the expensive,
/// OS-mutating release-lifecycle scenarios; the default fast suite keeps them out.
/// `include_merge_queue` is set only in the merge queue (via `E2E_MERGE_QUEUE`);
/// per-PR runs pass `false` so heavy `@merge-queue` serves stay off the PR path (a
/// cheaper per-engine canary covers them) and run once before the change lands.
pub fn resolve(
    decl: &ScenarioDecl,
    cap: &HostCapability,
    matrix: &Expectations,
    include_nightly: bool,
    include_lifecycle: bool,
    include_merge_queue: bool,
) -> Expectation {
    // (1) Applicability / skip.
    if decl.nightly && !include_nightly {
        return Expectation::Skip {
            reason: "nightly-only scenario; set E2E_INCLUDE_NIGHTLY to run".to_owned(),
        };
    }
    if decl.lifecycle && !include_lifecycle {
        return Expectation::Skip {
            reason: "lifecycle-only scenario; set E2E_INCLUDE_LIFECYCLE=1 to run".to_owned(),
        };
    }
    if decl.merge_queue && !include_merge_queue {
        return Expectation::Skip {
            reason: "merge-queue-only scenario; set E2E_MERGE_QUEUE to run".to_owned(),
        };
    }
    if decl.requires_gpu && !cap.has_amd_gpu {
        return Expectation::Skip {
            reason: "requires an AMD GPU; none detected on this host".to_owned(),
        };
    }
    if decl.requires_multi_gpu && cap.amd_gpu_count.is_none_or(|count| count <= 1) {
        // An unknown count skips as well: the premise is "more than one device
        // is present", and a host that cannot be counted has not shown that.
        // Running there is what turned a real multi-GPU assertion into a
        // single-GPU failure in the first place.
        let detected = cap.amd_gpu_count.map_or_else(
            || "the count could not be probed".to_owned(),
            |count| format!("{count} detected"),
        );
        return Expectation::Skip {
            reason: format!("requires more than one AMD GPU; {detected}"),
        };
    }
    if decl.requires_gfx_target && cap.gfx_target.is_none() {
        return Expectation::Skip {
            reason: "requires a detected AMD GFX target; none detected on this host".to_owned(),
        };
    }
    if decl.requires_no_gpu && cap.has_amd_gpu {
        return Expectation::Skip {
            reason: "requires a host with no AMD GPU; this host has one".to_owned(),
        };
    }
    if decl.requires_bare_metal && cap.is_wsl {
        return Expectation::Skip {
            reason: "requires a bare-metal host; this one is WSL2".to_owned(),
        };
    }
    if decl.requires_wsl && !cap.is_wsl {
        return Expectation::Skip {
            reason: "requires WSL; this host is not running under WSL".to_owned(),
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
                    flaky: entry.flaky,
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
                amd_gpu_count: Some(8),
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "vllm".into(),
                platform_slug: "mi300x".into(),
            },
            "strix-ubuntu" => HostCapability {
                os_family: "linux".into(),
                is_wsl: false,
                gfx_target: Some("gfx1151".into()),
                has_amd_gpu: true,
                amd_gpu_count: Some(1),
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "strix-halo".into(),
            },
            "strix-windows" => HostCapability {
                os_family: "windows".into(),
                is_wsl: false,
                gfx_target: Some("gfx1151".into()),
                has_amd_gpu: true,
                amd_gpu_count: Some(1),
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "strix-halo".into(),
            },
            // A WSL2 dev box with the ROCm passthrough in place: `os_family` is
            // `linux` and a GPU is usable, so nothing but `is_wsl` distinguishes
            // it from bare metal. That is precisely why `@requires-os:linux`
            // cannot stand in for `@requires-bare-metal`.
            //
            // The device COUNT is unknown on every WSL host: that platform
            // reaches the GPU through `/dev/dxg`, exposing neither a KFD
            // topology nor an amdgpu DRM card for the count probe to read. So
            // `@requires-multi-gpu` skips there whatever the hardware is.
            "wsl2" => HostCapability {
                os_family: "linux".into(),
                is_wsl: true,
                gfx_target: Some("gfx1151".into()),
                has_amd_gpu: true,
                amd_gpu_count: None,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "strix-halo-wsl".into(),
            },
            // The same box without the passthrough: the gfx target is reported
            // by the Windows-side driver but ROCm cannot reach it, so the probe
            // resolves no usable GPU (see `capability::host_has_usable_gpu`).
            "wsl" => HostCapability {
                os_family: "linux".into(),
                is_wsl: true,
                gfx_target: None,
                has_amd_gpu: false,
                amd_gpu_count: None,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "wsl".into(),
            },
            // The hosted WSL runner before the ROCm passthrough is complete:
            // Windows still reports the gfx target, but the CLI cannot use it.
            "wsl-no-passthrough" => HostCapability {
                os_family: "linux".into(),
                is_wsl: true,
                gfx_target: Some("gfx1151".into()),
                has_amd_gpu: false,
                amd_gpu_count: None,
                available_engines: vec!["lemonade".into(), "vllm".into()],
                effective_serve_engine: "lemonade".into(),
                platform_slug: "strix-halo-wsl".into(),
            },
            _ => HostCapability {
                os_family: "other".into(),
                is_wsl: false,
                gfx_target: None,
                has_amd_gpu: false,
                amd_gpu_count: Some(0),
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
    fn tags_parse_with_runtime_at_prefix() {
        let d = decl(&[
            "@id:lifecycle-linux-install",
            "@requires-os:linux",
            "@lifecycle",
        ]);
        assert_eq!(d.id.as_deref(), Some("lifecycle-linux-install"));
        assert_eq!(d.requires_os.as_deref(), Some("linux"));
        assert!(d.lifecycle);
    }

    #[test]
    fn bare_metal_tag_parses_in_both_shapes() {
        assert!(decl(&["id:x", "requires-bare-metal"]).requires_bare_metal);
        assert!(decl(&["@id:x", "@requires-bare-metal"]).requires_bare_metal);
        // Absent by default, so no existing scenario changes meaning.
        assert!(!decl(&["id:x", "requires-gpu"]).requires_bare_metal);
    }

    #[test]
    fn detected_target_requirement_does_not_require_gpu_passthrough() {
        let matrix = Expectations::default();
        let scenario = decl(&["id:resolver-preview", "requires-gfx-target"]);

        assert_eq!(
            resolve(
                &scenario,
                &cap("wsl-no-passthrough"),
                &matrix,
                false,
                false,
                false,
            ),
            Expectation::ExpectPass
        );
        assert!(matches!(
            resolve(&scenario, &cap("mock"), &matrix, false, false, false,),
            Expectation::Skip { .. }
        ));
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
            resolve(&d, &cap("mi300x"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, true, false, false),
            Expectation::ExpectPass
        );
        // The nightly gate is cheapest-first: a @nightly scenario that ALSO can't
        // run here (no GPU) still skips regardless of the include flag.
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, true, false, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn lifecycle_scenario_skips_unless_included() {
        let m = Expectations::default();
        // A @lifecycle scenario is skipped on the default fast path and runs only
        // when the caller opts in via E2E_INCLUDE_LIFECYCLE.
        let d = decl(&[
            "id:lifecycle-linux-install",
            "requires-os:linux",
            "lifecycle",
        ]);
        assert!(d.lifecycle);
        assert!(matches!(
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        assert_eq!(
            resolve(&d, &cap("strix-ubuntu"), &m, false, true, false),
            Expectation::ExpectPass
        );
        // Even when included, an inapplicable OS still skips (os gate is checked).
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m, false, true, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn merge_queue_scenario_skips_unless_included() {
        let m = Expectations::default();
        // A @merge-queue GPU serve on an applicable host: skipped on the per-PR
        // fast path (a cheaper canary covers it), runs in the merge queue.
        let d = decl(&[
            "id:serve-default-engine-inference",
            "requires-gpu",
            "merge-queue",
        ]);
        assert!(d.merge_queue);
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, false, false, true),
            Expectation::ExpectPass
        );
        // Independent of the nightly axis: a merge-queue scenario is not opted in
        // by E2E_INCLUDE_NIGHTLY.
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m, true, false, false),
            Expectation::Skip { .. }
        ));
        // Cheapest-first: still skips where it can't run at all (no GPU).
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, false, false, true),
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
            resolve(&d, &cap("mi300x"), &m, false, false, false),
            Expectation::ExpectXfail { .. }
        ));
        // Strix Ubuntu: gfx1151 → lemonade default → NOT vLLM → expect-pass.
        assert_eq!(
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::ExpectPass
        );
        // Strix Windows: lemonade default → expect-pass (this is the XPASS fix).
        assert_eq!(
            resolve(&d, &cap("strix-windows"), &m, false, false, false),
            Expectation::ExpectPass
        );
    }

    #[test]
    fn requires_gpu_skips_on_mock() {
        let m = eai7333_matrix();
        let d = decl(&["id:serve-default-engine-inference", "requires-gpu"]);
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn multi_gpu_tag_parses_in_both_shapes() {
        assert!(decl(&["id:x", "requires-multi-gpu"]).requires_multi_gpu);
        assert!(decl(&["@id:x", "@requires-multi-gpu"]).requires_multi_gpu);
        // Absent by default, and `@requires-gpu` does not imply it — otherwise
        // every existing GPU scenario would silently stop running on Strix.
        let gpu_only = decl(&["id:x", "requires-gpu"]);
        assert!(!gpu_only.requires_multi_gpu);
        assert!(gpu_only.requires_gpu);
        // Nor the reverse: the tags are independent flags on the same scenario.
        assert!(!decl(&["id:x", "requires-multi-gpu"]).requires_gpu);
    }

    /// Both directions of the gate. A scenario whose premise is a second device
    /// must RUN where one exists and SKIP where it does not — a gate that only
    /// ever skips would silently retire the assertion everywhere.
    #[test]
    fn requires_multi_gpu_runs_only_where_a_second_device_is_present() {
        let m = Expectations::default();
        let d = decl(&[
            "id:serve-rocr-reindexed-gpu-index-rejected",
            "requires-gpu",
            "requires-multi-gpu",
            "requires-os:linux",
        ]);
        // MI300X has eight devices → the premise holds → the scenario runs.
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, false, false, false),
            Expectation::ExpectPass
        );
        // Strix Halo has exactly one. `@requires-gpu` is satisfied there, which
        // is why the scenario used to run and fail on its premise.
        assert!(matches!(
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        // No GPU at all, and a host whose count could not be probed (WSL), skip
        // too: neither has SHOWN a second device.
        for host in ["mock", "wsl2"] {
            assert!(
                matches!(
                    resolve(&d, &cap(host), &m, false, false, false),
                    Expectation::Skip { .. }
                ),
                "{host} must not run a multi-GPU scenario"
            );
        }
    }

    /// The tag line of a scenario in a real `.feature` file, by `@id:` slug.
    /// Tags share one line with the id in this suite, which is what makes the
    /// lookup exact rather than a guess about declaration order.
    fn feature_tags(feature_text: &'static str, id: &str) -> Vec<&'static str> {
        feature_text
            .lines()
            .map(str::trim)
            .find(|line| {
                line.starts_with('@')
                    && line
                        .split_whitespace()
                        .any(|tag| tag == format!("@id:{id}").as_str())
            })
            .unwrap_or_else(|| panic!("no tag line for @id:{id}"))
            .split_whitespace()
            .collect()
    }

    /// The gate is only worth anything if the scenario in the file actually
    /// carries it, so resolve the REAL tag line rather than a hand-written copy:
    /// dropping `@requires-multi-gpu` from the feature file has to fail a test,
    /// not silently restore the single-GPU failure this was added to fix.
    ///
    /// The sibling `serve-19` is asserted in the same breath to stay UNGATED: its
    /// `HIP_VISIBLE_DEVICES=0` mask names a device every GPU host has, so its
    /// visible set resolves on one GPU and the refusal holds there. Gating it too
    /// would retire live coverage on the Strix lanes for no reason.
    #[test]
    fn the_rocr_reindexed_scenario_is_gated_on_multi_gpu_and_its_sibling_is_not() {
        let feature = include_str!("../features/model_serving.feature");
        let m = Expectations::default();

        let rocr = ScenarioDecl::from_tags(&feature_tags(
            feature,
            "serve-rocr-reindexed-gpu-index-rejected",
        ));
        assert!(rocr.requires_multi_gpu, "serve-20 must carry the gate");
        assert!(matches!(
            resolve(&rocr, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        assert_eq!(
            resolve(&rocr, &cap("mi300x"), &m, false, false, false),
            Expectation::ExpectPass
        );

        let masked =
            ScenarioDecl::from_tags(&feature_tags(feature, "serve-masked-gpu-index-rejected"));
        assert!(
            !masked.requires_multi_gpu,
            "serve-19 holds on a single GPU and must keep running there"
        );
        for host in ["strix-ubuntu", "mi300x"] {
            assert_eq!(
                resolve(&masked, &cap(host), &m, false, false, false),
                Expectation::ExpectPass,
                "{host} must still run serve-19"
            );
        }
    }

    /// The single-GPU skip must name the count, not just repeat the `@requires-gpu`
    /// reason: on Strix Halo a GPU *is* present, so "none detected" would send a
    /// reader of the report looking for a missing device that is right there.
    #[test]
    fn multi_gpu_skip_reason_distinguishes_itself_from_the_no_gpu_one() {
        let m = Expectations::default();
        let d = decl(&["id:x", "requires-gpu", "requires-multi-gpu"]);
        let Expectation::Skip { reason } =
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false)
        else {
            panic!("a single-GPU host must skip a multi-GPU scenario");
        };
        assert!(
            reason.contains("more than one") && reason.contains('1'),
            "reason should name the shortfall, got: {reason}"
        );
    }

    #[test]
    fn requires_no_gpu_runs_only_on_hosts_without_a_gpu() {
        let m = Expectations::default();
        let d = decl(&["id:serve-no-gpu-fails-fast", "requires-no-gpu"]);
        // The mock host has no AMD GPU → the no-GPU premise applies → runs.
        assert_eq!(
            resolve(&d, &cap("mock"), &m, false, false, false),
            Expectation::ExpectPass
        );
        // Every GPU host skips it — the premise can't hold there.
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        assert!(matches!(
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn requires_bare_metal_skips_on_wsl_and_runs_everywhere_else() {
        let m = Expectations::default();
        let d = decl(&["id:diagnose-matches-known-symptom", "requires-bare-metal"]);
        for wsl in ["wsl2", "wsl"] {
            assert!(matches!(
                resolve(&d, &cap(wsl), &m, false, false, false),
                Expectation::Skip { .. }
            ));
        }
        // Bare-metal hosts of every shape still run it — including the mock lane,
        // which is where these scenarios earn their keep as a required check.
        for host in ["mock", "mi300x", "strix-ubuntu", "strix-windows"] {
            assert_eq!(
                resolve(&d, &cap(host), &m, false, false, false),
                Expectation::ExpectPass,
                "{host} is bare metal and must still run the scenario"
            );
        }
    }

    #[test]
    fn combining_requires_os_and_requires_bare_metal_narrows_to_native_linux() {
        // diagnose-08's exact tag set. It needs a Linux host (the assertion reads
        // back a shell rc file) that is ALSO not WSL (the only auto-applicable
        // fix reaching the consent gate does not apply there). Neither tag says
        // that alone, so the pair has to compose — and a `-n` filtered local run
        // bypasses this resolution entirely, which makes it worth pinning here
        // rather than trusting a hand-run of the suite.
        let m = Expectations::default();
        let d = decl(&[
            "id:diagnose-fix-requires-agreement-before-changing-anything",
            "requires-os:linux",
            "requires-bare-metal",
        ]);
        for wsl in ["wsl", "wsl2"] {
            assert!(
                matches!(
                    resolve(&d, &cap(wsl), &m, false, false, false),
                    Expectation::Skip { .. }
                ),
                "{wsl} is linux but not bare metal, so the scenario has no premise"
            );
        }
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        // `mock` is deliberately absent: the fixture models it as os_family
        // "other", so it cannot stand for the real mock lane here.
        for host in ["mi300x", "strix-ubuntu"] {
            assert_eq!(
                resolve(&d, &cap(host), &m, false, false, false),
                Expectation::ExpectPass,
                "{host} is native Linux and must still run the scenario"
            );
        }
    }

    /// The exact inverse of the tag above: proving both directions keeps the
    /// pair from silently collapsing into one predicate.
    #[test]
    fn requires_wsl_runs_only_on_wsl_hosts() {
        let m = Expectations::default();
        let d = decl(&["id:examine-detects-wsl", "requires-wsl"]);
        assert!(d.requires_wsl);
        for wsl in ["wsl", "wsl2"] {
            assert_eq!(
                resolve(&d, &cap(wsl), &m, false, false, false),
                Expectation::ExpectPass
            );
        }
        for platform in ["mock", "mi300x", "strix-ubuntu", "strix-windows"] {
            assert!(matches!(
                resolve(&d, &cap(platform), &m, false, false, false),
                Expectation::Skip { .. }
            ));
        }
    }

    #[test]
    fn requires_os_linux_does_not_stand_in_for_requires_bare_metal() {
        // The reason the tag has to exist: WSL2 reports os_family "linux", so an
        // `@requires-os:linux` scenario runs there. If this ever starts skipping,
        // the two tags have collapsed into one and the new one is redundant.
        let m = Expectations::default();
        let d = decl(&["id:some-linux-scenario", "requires-os:linux"]);
        assert_eq!(
            resolve(&d, &cap("wsl2"), &m, false, false, false),
            Expectation::ExpectPass
        );
    }

    #[test]
    fn bare_metal_skip_is_not_recorded_as_a_known_bug() {
        // WSL2 out-of-scope is deliberate product behaviour, so it must resolve to
        // Skip even when the scenario also carries an xfail row — otherwise the
        // report would call a designed-in limitation a defect.
        let m = Expectations::parse(
            r#"
[["diagnose-matches-known-symptom"]]
when = {}
bug = "EAI-0000"
reason = "unrelated open bug"
"#,
        )
        .unwrap();
        let d = decl(&["id:diagnose-matches-known-symptom", "requires-bare-metal"]);
        assert!(matches!(
            resolve(&d, &cap("wsl2"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn vllm_pinned_scenario_skips_where_vllm_cannot_start() {
        let m = Expectations::default();
        // The feature-qualified `serve-vllm-inference` scenario pins vLLM.
        let d = decl(&[
            "id:serve-vllm-inference",
            "requires-gpu",
            "requires-engine:vllm",
        ]);
        // MI300X: vLLM available → not skipped (expect-pass here, no matrix entry).
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, false, false, false),
            Expectation::ExpectPass
        );
        // Strix Windows: vLLM can't start → skip (N/A).
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m, false, false, false),
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
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::ExpectPass
        );
        assert!(matches!(
            resolve(&d, &cap("strix-windows"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
        assert!(matches!(
            resolve(&d, &cap("mock"), &m, false, false, false),
            Expectation::Skip { .. }
        ));
    }

    #[test]
    fn non_gpu_scenario_always_runs() {
        let m = Expectations::default();
        let d = decl(&["id:examine-version"]);
        assert_eq!(
            resolve(&d, &cap("mock"), &m, false, false, false),
            Expectation::ExpectPass
        );
        assert_eq!(
            resolve(&d, &cap("mi300x"), &m, false, false, false),
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
            resolve(&d, &cap("mock"), &m, false, false, false),
            Expectation::ExpectXfail { .. }
        ));
        assert!(matches!(
            resolve(&d, &cap("mi300x"), &m, false, false, false),
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
            resolve(&d, &cap("strix-ubuntu"), &m, false, false, false),
            Expectation::ExpectXfail { .. }
        ));
        // Strix Windows (windows, lemonade) → os mismatch → expect-pass.
        assert_eq!(
            resolve(&d, &cap("strix-windows"), &m, false, false, false),
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
    fn is_xfail_follows_the_matching_condition() {
        let m = Expectations::parse(
            r#"
[["serve-default-engine-inference"]]
when = { effective_engine = "vllm" }
bug = "EAI-7333"
reason = "vLLM readiness gap"
"#,
        )
        .unwrap();
        assert!(m.is_xfail("serve-default-engine-inference", &cap("mi300x"), "vllm"));
        // A condition that doesn't match this host leaves the scenario expect-pass.
        assert!(!m.is_xfail(
            "serve-default-engine-inference",
            &cap("strix-windows"),
            "lemonade"
        ));
        // No entry at all → expect-pass.
        assert!(!m.is_xfail("nope", &cap("mi300x"), "vllm"));
    }

    #[test]
    fn gpu_bug_condition_does_not_match_an_unusable_wsl_gpu() {
        let m = Expectations::parse(
            r#"
[["chat-end-to-end-local-model"]]
when = { effective_engine = "lemonade", os = "linux", therock_family = "gfx*", has_amd_gpu = true }
bug = "EAI-7423"
reason = "Applies only when the scenario uses a real Lemonade GPU serve."
flaky = true
"#,
        )
        .unwrap();

        assert!(m.is_xfail(
            "chat-end-to-end-local-model",
            &cap("strix-ubuntu"),
            "lemonade"
        ));
        assert!(!m.is_xfail(
            "chat-end-to-end-local-model",
            &cap("wsl-no-passthrough"),
            "lemonade"
        ));
    }

    #[test]
    fn gpu_report_disagreement_xfail_requires_a_reported_gfx_target() {
        let m = Expectations::parse(include_str!("../expectations.toml")).unwrap();
        assert!(m.is_xfail(
            "examine-both-forms-agree-on-gpu",
            &cap("wsl-no-passthrough"),
            "lemonade"
        ));
        assert!(!m.is_xfail("examine-both-forms-agree-on-gpu", &cap("wsl"), "lemonade"));
    }

    /// EAI-8031's row (Windows lane cold-downloading its HF checkpoint every
    /// scenario, racing serve readiness) is resolved by wiring `HF_HOME` into
    /// the Windows lane the same way every Linux lane already has it — not by
    /// this test, which only guards against a silent regression. Deleting the
    /// row's own scoping tests alongside the row left nothing to fail if
    /// someone re-added it without re-diagnosing the underlying cause; this
    /// makes reintroducing it a deliberate decision instead of a copy-paste.
    #[test]
    fn hf_checkpoint_serve_has_no_xfail_on_windows_gpu_hosts() {
        let m = Expectations::parse(include_str!("../expectations.toml")).unwrap();
        assert!(!m.is_xfail(
            "serve-hf-checkpoint-inference",
            &cap("strix-windows"),
            "lemonade"
        ));
    }

    /// A row whose scenario has been renamed or deleted never matches anything,
    /// so it can neither xfail nor go stale: it just sits there stating a bug
    /// that nothing measures. Nothing at runtime notices — resolution is keyed
    /// FROM the scenario TO this file, never the other way — so the orphan is
    /// caught here instead.
    #[test]
    fn every_expectation_row_names_a_scenario_that_exists() {
        let mut scenario_ids = std::collections::BTreeSet::new();
        for path in feature_files() {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
            // Read tags off TAG LINES only, the way `scenarios_of` in
            // tests/feature_naming.rs does. Scanning the whole file would also
            // pick up an `@id:` written inside a `#` comment — and since this
            // set is what decides whether a row is an orphan, a commented-out
            // id would vouch for a row that measures nothing, which is the one
            // thing this test exists to catch.
            for line in text.lines() {
                let line = line.trim();
                if !line.starts_with('@') {
                    continue;
                }
                for tag in line.split_whitespace() {
                    if let Some(id) = tag.strip_prefix("@id:") {
                        scenario_ids.insert(id.to_owned());
                    }
                }
            }
        }
        assert!(
            !scenario_ids.is_empty(),
            "read no `@id:` tags at all, so this check would pass vacuously"
        );

        let m = Expectations::parse(include_str!("../expectations.toml")).unwrap();
        // Both sides, not just the scenarios. An `expectations.toml` that
        // declared nothing would make `orphans` empty and this check green —
        // it would be reporting "no stale rows" about a file it never read.
        // A sibling test would catch the empty parse, but this one should not
        // depend on that to mean what it says.
        let declared: Vec<&str> = m.declared_ids().collect();
        assert!(
            !declared.is_empty(),
            "expectations.toml declared no rows at all, so this check would pass vacuously"
        );
        let orphans: Vec<&str> = declared
            .iter()
            .copied()
            .filter(|id| !scenario_ids.contains(*id))
            .collect();
        assert!(
            orphans.is_empty(),
            "expectations.toml declares rows for scenarios that no longer exist: {orphans:?}\n\
             Delete the row, or fix the id if the scenario was renamed."
        );
    }

    /// Every `.feature` file the suite runs.
    ///
    /// `features/` is flat, as `scenarios_of` in tests/feature_naming.rs also
    /// assumes, so this does not recurse. It refuses to run rather than quietly
    /// covering less if that ever stops being true: a subdirectory would simply
    /// vanish from the checks below, and a check that silently stops looking at
    /// something is worse than no check at all.
    fn feature_files() -> Vec<std::path::PathBuf> {
        feature_files_in(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("features"))
    }

    /// The directory scan itself, split out so the subdirectory refusal can be
    /// exercised against a temporary tree — against the real `features/` it
    /// could only ever fire by someone breaking the repository.
    fn feature_files_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(dir).expect("no features directory") {
            let path = entry.expect("unreadable features directory entry").path();
            assert!(
                !path.is_dir(),
                "features/ has grown a subdirectory ({}), which this scan does not descend \
                 into — make it recursive, here and in tests/feature_naming.rs, before moving \
                 any .feature file into one",
                path.display()
            );
            if path.extension().is_some_and(|ext| ext == "feature") {
                paths.push(path);
            }
        }
        assert!(
            !paths.is_empty(),
            "found no .feature files in {}",
            dir.display()
        );
        paths
    }

    #[test]
    fn the_feature_scan_takes_the_flat_files_it_finds() {
        let dir = tempfile::tempdir().expect("no temp dir");
        std::fs::write(dir.path().join("a.feature"), "Feature: a\n").unwrap();
        std::fs::write(dir.path().join("notes.md"), "ignored\n").unwrap();
        // A file named exactly `.feature` has no extension, so this scan skips
        // it. Pinned on both sides — `tests/feature_naming.rs` once matched on a
        // `.feature` string suffix instead, which would have taken this one and
        // left the two scans covering different sets.
        std::fs::write(dir.path().join(".feature"), "not a feature file\n").unwrap();
        let found = feature_files_in(dir.path());
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].ends_with("a.feature"), "{found:?}");
    }

    #[test]
    #[should_panic(expected = "found no .feature files")]
    fn the_feature_scan_refuses_a_directory_with_no_feature_files() {
        // The assertion this pins reports an emptied `features/` once, at the
        // scan, naming the directory. It is not what stops the two callers
        // agreeing with everything — both already guard themselves ("read no
        // `@id:` tags at all, so this check would pass vacuously" and its
        // counterpart over claimed ids), so they fail either way; this just
        // fails first and says what actually went wrong.
        //
        // Unreachable against the real `features/`, so pinned here — and
        // pinned the same way in `tests/feature_naming.rs`, whose scan carries
        // the same assertion.
        //
        // The fixture writes a non-`.feature` file on purpose: what the scan
        // refuses is an empty RESULT, not an empty directory.
        let dir = tempfile::tempdir().expect("no temp dir");
        std::fs::write(dir.path().join("notes.md"), "ignored\n").unwrap();
        let _ = feature_files_in(dir.path());
    }

    #[test]
    #[should_panic(expected = "has grown a subdirectory")]
    fn the_feature_scan_refuses_a_subdirectory_rather_than_skipping_it() {
        // The branch that stops this scan quietly covering less than it claims.
        // Unreachable against the real `features/`, so it is pinned here.
        let dir = tempfile::tempdir().expect("no temp dir");
        std::fs::write(dir.path().join("a.feature"), "Feature: a\n").unwrap();
        std::fs::create_dir(dir.path().join("nested")).unwrap();
        let _ = feature_files_in(dir.path());
    }

    /// What one feature file claims about expected failures.
    struct FailureClaims {
        /// Scenario ids introduced by a comment block claiming "Expected to
        /// FAIL".
        ids: Vec<String>,
        /// 1-indexed lines whose claim reached no scenario id.
        unbound: Vec<usize>,
    }

    /// Scan one feature file for "Expected to FAIL" claims.
    ///
    /// A scenario is written comment block → tag line(s) → `Scenario:`, and the
    /// block is contiguous with the tags it introduces, so a blank line ends a
    /// claim's reach: a file header or a note left in a previous scenario's
    /// body describes something other than whatever comes next.
    ///
    /// A claim that reaches nothing is REPORTED, not dropped. Dropping it is
    /// how a scenario leaves this check's coverage without anyone noticing —
    /// one blank line between a comment and its tag line would be enough, and
    /// a blank line is the same character the file already uses to separate
    /// scenarios. Reporting it also keeps the failure honest about where the
    /// problem is: the comment, not the scenario that happens to follow it.
    ///
    /// This is line-shaped, not a Gherkin parse. It assumes the phrase appears
    /// only in real `#` comments (not inside a docstring or table cell) and
    /// that `Scenario:`/`Scenario Outline:` are the only block terminators —
    /// `Background:` is not treated as one. Neither construct exists anywhere in
    /// `features/` today, and both would fail loudly (an unbound claim) rather
    /// than silently if one appeared, but a reader adding either should revisit
    /// this scan first.
    fn claimed_failure_ids(text: &str) -> FailureClaims {
        let (mut ids, mut unbound) = (Vec::new(), Vec::new());
        let (mut claim_lines, mut pending_ids): (Vec<usize>, Vec<String>) =
            (Vec::new(), Vec::new());
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                // Clearing `pending_ids` here means a blank line between a tag
                // line and its `Scenario:` would drop the ids too. Deliberate,
                // and no longer silent: the claim above them then reaches
                // nothing and is reported as unbound.
                unbound.append(&mut claim_lines);
                pending_ids.clear();
            } else if line.starts_with('#') {
                if line.contains("Expected to FAIL") {
                    claim_lines.push(index + 1);
                }
            } else if line.starts_with('@') {
                for tag in line.split_whitespace() {
                    if let Some(id) = tag.strip_prefix("@id:") {
                        pending_ids.push(id.to_owned());
                    }
                }
            } else if line.starts_with("Scenario:") || line.starts_with("Scenario Outline:") {
                if !claim_lines.is_empty() {
                    if pending_ids.is_empty() {
                        unbound.append(&mut claim_lines);
                    } else {
                        ids.extend(pending_ids.iter().cloned());
                    }
                }
                claim_lines.clear();
                pending_ids.clear();
            }
        }
        unbound.append(&mut claim_lines);
        FailureClaims { ids, unbound }
    }

    #[test]
    fn a_contiguous_comment_block_binds_to_its_scenario() {
        let claims = claimed_failure_ids(
            "Feature: f\n\
             \n\
             \x20 # Expected to FAIL. The thing is broken.\n\
             \x20 @id:thing-is-broken\n\
             \x20 Scenario: f-01 - The thing works\n\
             \x20   Given a thing\n",
        );
        assert_eq!(claims.ids, ["thing-is-broken"]);
        assert!(claims.unbound.is_empty(), "{:?}", claims.unbound);
    }

    #[test]
    fn a_file_header_mentioning_the_phrase_condemns_no_scenario() {
        let claims = claimed_failure_ids(
            "# Historical note: f-01 was once Expected to FAIL.\n\
             \n\
             Feature: f\n\
             \n\
             \x20 @id:thing-works\n\
             \x20 Scenario: f-01 - The thing works\n\
             \x20   Given a thing\n",
        );
        // Not attributed to `thing-works` — that was the round-9 false positive.
        assert!(claims.ids.is_empty(), "{:?}", claims.ids);
        // But not swallowed either: it is reported against its own line.
        assert_eq!(claims.unbound, [1]);
    }

    #[test]
    fn a_comment_a_blank_line_from_its_tag_is_reported_not_dropped() {
        let claims = claimed_failure_ids(
            "Feature: f\n\
             \n\
             \x20 # Expected to FAIL. The thing is broken.\n\
             \n\
             \x20 @id:thing-is-broken\n\
             \x20 Scenario: f-01 - The thing works\n\
             \x20   Given a thing\n",
        );
        // The scenario has silently left the guard's coverage in every earlier
        // version of this scan. It must not do so silently.
        assert!(claims.ids.is_empty(), "{:?}", claims.ids);
        assert_eq!(claims.unbound, [3]);
    }

    #[test]
    fn a_claim_does_not_inherit_a_stale_id_from_across_a_blank_line() {
        // Pins the `pending_ids.clear()` half of the blank-line arm, which the
        // other scan tests leave dead: without it `stale-id` would still be
        // pending when the claim below is bound, and the claim would be
        // attributed to a scenario it says nothing about — silently, and with
        // the row check then vouching for the wrong id.
        let claims = claimed_failure_ids(
            "Feature: f\n\
             \n\
             \x20 @id:stale-id\n\
             \n\
             \x20 # Expected to FAIL. The thing is broken.\n\
             \x20 Scenario: f-01 - The thing works\n",
        );
        assert!(claims.ids.is_empty(), "{:?}", claims.ids);
        assert_eq!(claims.unbound, [5]);
    }

    #[test]
    fn a_claim_dangling_at_end_of_file_is_reported_not_dropped() {
        // The `Scenario:` that would have bound this claim was deleted, leaving
        // the comment as the last thing in the file. Covers the flush after the
        // loop, which the blank-line and `Scenario:` cases never reach.
        let claims = claimed_failure_ids(
            "Feature: f\n\
             \n\
             \x20 # Expected to FAIL. The thing is broken.\n\
             \x20 @id:thing-is-broken\n",
        );
        assert!(claims.ids.is_empty(), "{:?}", claims.ids);
        assert_eq!(claims.unbound, [3]);
    }

    /// The inverse orphan: a scenario whose comment still says "Expected to
    /// FAIL" after its row has been deleted.
    ///
    /// `expectations.toml` owns that state, but the feature files restate it in
    /// prose for the reader, and prose does not move when a row does. When a bug
    /// is fixed the harness flips correctly and silently — leaving a comment that
    /// tells the next reader the opposite of what the suite now enforces. Four
    /// rows were deleted while this file was being written, so the drift is not
    /// hypothetical.
    #[test]
    fn every_expected_to_fail_comment_still_has_a_row() {
        let mut claimed = Vec::new();
        let mut unbound = Vec::new();
        for path in feature_files() {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
            let scan = claimed_failure_ids(&text);
            claimed.extend(scan.ids);
            unbound.extend(
                scan.unbound
                    .into_iter()
                    .map(|number| format!("{}:{number}", path.display())),
            );
        }
        // Before asking whether the claims still have rows, insist that every
        // claim reached a scenario at all. A claim that reaches none is not
        // harmless: it means that scenario is no longer covered by this check,
        // and the whole point of the check is that nothing drops out of it
        // quietly. Naming the file and line sends the reader to the comment
        // itself rather than to whichever scenario happened to follow it.
        assert!(
            unbound.is_empty(),
            "these lines say 'Expected to FAIL' but bind to no scenario id, so the scenario they \
             describe is not covered by this check: {unbound:?}\n\
             A claim reaches only the tag line directly beneath it — no blank line in between. \
             Move the comment against its tag line, or reword it if it is not a claim about a \
             specific scenario."
        );
        assert!(
            !claimed.is_empty(),
            "found no scenario claiming 'Expected to FAIL', so this check would pass vacuously"
        );

        let m = Expectations::parse(include_str!("../expectations.toml")).unwrap();
        let declared: std::collections::BTreeSet<&str> = m.declared_ids().collect();
        let lying: Vec<&String> = claimed
            .iter()
            .filter(|id| !declared.contains(id.as_str()))
            .collect();
        assert!(
            lying.is_empty(),
            "these scenarios still say 'Expected to FAIL' but have no expectations row, so the \
             comment asserts the opposite of what the suite enforces: {lying:?}\n\
             Delete the comment along with the row — the bug it described is fixed, and the \
             scenario now guards the fix."
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
