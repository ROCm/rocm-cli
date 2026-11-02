// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Sidecar schema and parsing: the `platform.json` manifest (host capability,
//! component versions, per-scenario expectations) and the `commands.jsonl`
//! record of `rocm` invocations, both read from beside a platform's
//! `report.json`.

use std::path::Path;

use serde::Deserialize;

/// One recorded `rocm` invocation from a platform's `commands.jsonl` sidecar.
#[derive(Deserialize)]
pub(crate) struct CommandRecord {
    pub(crate) scenario: Option<String>,
    pub(crate) subcommand: String,
    /// Full command as executed (e.g. "rocm serve Qwen/... --engine vllm").
    /// Falls back to `subcommand` for older artifacts that predate this field.
    #[serde(default)]
    pub(crate) command: Option<String>,
    #[serde(default)]
    pub(crate) engine: Option<String>,
    /// True when `engine` was the CLI's own default choice (no `--engine` flag),
    /// so the report can show it as "<engine> (default)".
    #[serde(default)]
    pub(crate) engine_is_default: bool,
}

/// Read a platform's `commands.jsonl` (sibling of `report.json`). Missing file =
/// no records (older artifacts, or a platform that recorded none).
pub(super) fn parse_commands(json_path: &Path) -> Vec<CommandRecord> {
    let path = json_path.with_file_name("commands.jsonl");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// The `platform.json` sidecar written by the harness: the probed host
/// capability plus every scenario's resolved expectation (including skips, which
/// never appear in `report.json`). This is the source of truth for a platform's
/// identity and for the expected-vs-actual reconciliation.
#[derive(Deserialize)]
pub(super) struct PlatformManifest {
    #[serde(default)]
    pub(super) platform_slug: String,
    #[serde(default)]
    pub(super) capability: Option<ManifestCapability>,
    /// Component versions (OS/ROCm/vLLM/lemonade); absent in older artifacts.
    #[serde(default)]
    pub(super) versions: PlatformVersions,
    #[serde(default)]
    pub(super) expectations: Vec<ManifestExpectation>,
}

#[derive(Deserialize)]
pub(super) struct ManifestCapability {
    #[serde(default)]
    pub(super) effective_serve_engine: String,
}

/// Per-platform component versions, mirrored from the harness `platform.json`.
/// All optional — a source not present on a platform is simply omitted.
#[derive(Deserialize, Default, Clone)]
pub(super) struct PlatformVersions {
    #[serde(default)]
    pub(super) os: Option<String>,
    #[serde(default)]
    pub(super) rocm: Option<String>,
    #[serde(default)]
    pub(super) vllm: Option<String>,
    #[serde(default)]
    pub(super) lemonade: Option<String>,
    /// Release channel ("release" | "nightly") the platform's active runtime was
    /// installed under. `None` on artifacts predating the field, or when no
    /// managed runtime was active — those render as a single unlabelled column.
    #[serde(default)]
    pub(super) channel: Option<String>,
}

impl PlatformVersions {
    /// Rendered "os X · ROCm Y · vLLM Z · lemonade W" line for a column heading,
    /// skipping absent components. Empty when nothing is known.
    pub(super) fn summary(&self) -> String {
        [
            self.os.as_deref().map(|v| format!("OS {v}")),
            self.rocm.as_deref().map(|v| format!("ROCm {v}")),
            self.vllm.as_deref().map(|v| format!("vLLM {v}")),
            self.lemonade.as_deref().map(|v| format!("lemonade {v}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ")
    }

    /// The software-stack versions (ROCm/vLLM/lemonade, WITHOUT the OS — that goes
    /// in its own cell), for the summary-matrix Platform cell. Empty when none are
    /// known (e.g. mock, which has no installed runtime).
    pub(super) fn platform_stack(&self) -> String {
        [
            self.rocm.as_deref().map(|v| format!("ROCm {v}")),
            self.vllm.as_deref().map(|v| format!("vLLM {v}")),
            self.lemonade.as_deref().map(|v| format!("lemonade {v}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ")
    }
}

/// One scenario's resolved expectation, keyed by its stable `@id`.
#[derive(Deserialize, Clone)]
pub(super) struct ManifestExpectation {
    pub(super) id: String,
    /// The `Feature:` this scenario belongs to. Absent in artifacts predating
    /// the grouped grid — see [`feature_of`](super::feature_of) for the fallback.
    #[serde(default)]
    pub(super) feature: String,
    /// The scenario's own name (`<key>-<NN> - <description>`), carrying the
    /// per-feature index rows are sorted by. Absent in older artifacts.
    #[serde(default)]
    pub(super) scenario: String,
    #[serde(default)]
    pub(super) effective_engine: String,
    /// "pass" | "xfail" | "skip".
    pub(super) expected: String,
    #[serde(default)]
    pub(super) bug: Option<String>,
    #[serde(default)]
    pub(super) reason: Option<String>,
    /// Non-deterministic known bug: an XPASS is expected and non-fatal.
    #[serde(default)]
    pub(super) flaky: bool,
}

/// Read a platform's `platform.json` (sibling of `report.json`). Missing =
/// `None` (older artifacts predating the expectation system).
pub(super) fn parse_platform_manifest(json_path: &Path) -> Option<PlatformManifest> {
    let path = json_path.with_file_name("platform.json");
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}
