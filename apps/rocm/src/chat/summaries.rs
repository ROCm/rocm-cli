// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Deterministic summaries of the read-only tool output the chat assistant
//! reads back: `examine`, the model recipes, ComfyUI status, the managed
//! services, a port check and the engine inventory. Each
//! `deterministic_*_tool_summary` function recognises one tool's text, reads
//! the values it needs out of it and returns a short plain-language summary,
//! or `None` when the text is not that tool's. `deterministic_chat_tool_summary`
//! is the entry point the turn loop calls; when it returns a summary the chat
//! shows it instead of asking the local model to restate the output.
//!
//! The model summary is the exception to "reads the values it needs out of
//! it": it recognises the `rocm model` listing from the text
//! (`model_catalog_text::text_is_model_catalog_listing`) but takes every value
//! it prints from `load_model_recipe_registry` and
//! `model_catalog_text::model_recipe_engine_availability`. It used to parse the
//! `--verbose` listing's `aliases=[…] min_gpu_mem=…` rows instead, so pointing
//! the chat tool at the curated listing in #1041 silently removed the summary.
//!
//! The parsers that turn tool text back into rows (`DeterministicServiceRow`,
//! `DeterministicEngineRow`) and `managed_service_running_state`, the mapping
//! from a managed service's status to `running`, `starting`, `not_running` or
//! `unknown`, live here too. The services tool text itself is rendered by
//! `chat/mcp_bridge.rs` (`render_services_tool_result_text`), which reads
//! `managed_service_running_state` from this module.
//!
//! A leaf of the `chat` module: nothing here calls back into `chat.rs`. The
//! only edges out are `providers` (the assistant model ids),
//! `managed_service_status::friendly_engine_label` and the two model-catalog
//! helpers above.

use crate::managed_service_status::friendly_engine_label;
use crate::model_catalog_text::{
    ModelEngineAvailability, model_recipe_engine_availability, text_is_model_catalog_listing,
};
use crate::providers;
use rocm_core::placeholder::empty_as_unknown;
use rocm_core::{AppPaths, ModelRecipeRecord, load_model_recipe_registry};
use std::fmt::Write as _;

#[derive(Debug, Default)]
struct DeterministicServiceRow {
    pub(crate) service_id: String,
    pub(crate) engine: String,
    pub(crate) model: String,
    pub(crate) status: String,
    pub(crate) running_state: String,
    pub(crate) endpoint: String,
}

fn parse_deterministic_service_rows(tool_text: &str) -> Vec<DeterministicServiceRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicServiceRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("- ") {
            if rest.starts_with("service_id=") {
                if let Some(row) = current.take() {
                    rows.push(row);
                }
                let mut row = DeterministicServiceRow::default();
                for token in rest.split_whitespace() {
                    if let Some((key, value)) = token.split_once('=') {
                        match key {
                            "service_id" => row.service_id = value.to_owned(),
                            "engine" => row.engine = value.to_owned(),
                            "model" => row.model = value.to_owned(),
                            "status" => row.status = value.to_owned(),
                            "running_state" => row.running_state = value.to_owned(),
                            "endpoint" => row.endpoint = value.to_owned(),
                            _ => {}
                        }
                    }
                }
                if row.running_state.is_empty() {
                    row.running_state = managed_service_running_state(&row.status).to_owned();
                }
                current = Some(row);
                continue;
            }
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicServiceRow {
                service_id: rest.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(value) = trimmed.strip_prefix("status:") {
            row.status = value.trim().to_owned();
            row.running_state = managed_service_running_state(&row.status).to_owned();
        } else if let Some(value) = trimmed.strip_prefix("engine:") {
            row.engine = value.trim().to_owned();
        } else if let Some(value) = trimmed.strip_prefix("model:") {
            row.model = value.trim().to_owned();
        } else if let Some(value) = trimmed.strip_prefix("endpoint:") {
            row.endpoint = value.trim().to_owned();
        }
    }
    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

#[derive(Debug, Default)]
struct DeterministicEngineRow {
    pub(crate) engine: String,
    pub(crate) runtime: String,
    pub(crate) note: String,
}

fn parse_deterministic_engine_rows(tool_text: &str) -> Vec<DeterministicEngineRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicEngineRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim().trim_start_matches("* ").trim_start();
        if let Some(engine) = engine_name_from_inventory_line(trimmed) {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicEngineRow {
                engine: engine.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(runtime) = trimmed.strip_prefix("runtime:") {
            row.runtime = runtime.trim().to_owned();
        } else if let Some(note) = trimmed.strip_prefix("note:") {
            row.note = note.trim().to_owned();
        }
    }
    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

fn engine_name_from_inventory_line(line: &str) -> Option<&'static str> {
    ["lemonade", "vllm", "aim"].into_iter().find(|engine| {
        line == *engine
            || line
                .strip_prefix(*engine)
                .is_some_and(|rest| rest.starts_with(char::is_whitespace))
    })
}

pub(super) fn managed_service_running_state(status: &str) -> &'static str {
    match status {
        "ready" | "running" => "running",
        "starting" | "recovering" => "starting",
        "failed" | "stopped" => "not_running",
        _ => "unknown",
    }
}

pub(super) fn deterministic_rocm_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("examine:") {
        return None;
    }
    let mut lines = Vec::new();
    let gpu = chat_tool_value(tool_text, "driver_detail")
        .filter(|value| value != "<unknown>")
        .or_else(|| {
            let target = chat_tool_value(tool_text, "detected_gfx_target")?;
            Some(format!("AMD GPU target {target}"))
        });
    if let Some(gpu) = gpu {
        lines.push(format!("  GPU: {gpu}"));
    }

    let active_status = chat_tool_value(tool_text, "active_runtime_status");
    if active_status.as_deref() == Some("ready") {
        let family = chat_tool_value(tool_text, "active_runtime_family")
            .filter(|value| value != "<unset>" && value != "<unknown>");
        let version = chat_tool_value(tool_text, "active_runtime_version")
            .filter(|value| value != "<unset>" && value != "<unknown>");
        let mut detail = "installed and active for ROCm CLI".to_owned();
        if let Some(family) = family {
            let _ = write!(detail, " ({family})");
        }
        if let Some(version) = version {
            let _ = write!(detail, ", {version}");
        }
        lines.push(format!("  ROCm/TheRock: {detail}"));
        if let Some(root) = chat_tool_value(tool_text, "active_runtime_root")
            .or_else(|| chat_tool_value(tool_text, "setup_runtime_root"))
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Install folder: {root}"));
        }
        if let Some(cache) = chat_tool_value(tool_text, "active_runtime_pip_cache_dir")
            .or_else(|| chat_tool_value(tool_text, "setup_runtime_pip_cache_dir"))
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Downloads/cache: {cache}"));
        }
    } else if let Some(status) = active_status.as_deref() {
        lines.push(format!("  ROCm/TheRock: active runtime status is {status}"));
        if let Some(root) = chat_tool_value(tool_text, "setup_runtime_root")
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Selected setup folder: {root}"));
        }
        if let Some(cache) = chat_tool_value(tool_text, "setup_runtime_pip_cache_dir")
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Downloads/cache: {cache}"));
        }
    }

    if chat_tool_value(tool_text, "legacy_rocm_status").as_deref() == Some("not_detected")
        && active_status.as_deref() == Some("ready")
    {
        lines.push(
            "  Note: no global legacy ROCm install was found; ROCm CLI is using its managed TheRock runtime."
                .to_owned(),
        );
    }

    (!lines.is_empty()).then(|| lines.join("\n"))
}

pub(super) fn deterministic_chat_tool_summary(paths: &AppPaths, tool_text: &str) -> Option<String> {
    deterministic_rocm_tool_summary(tool_text)
        .or_else(|| deterministic_model_tool_summary(paths, tool_text))
        .or_else(|| deterministic_combined_status_tool_summary(tool_text))
}

fn deterministic_combined_status_tool_summary(tool_text: &str) -> Option<String> {
    let summaries = [
        deterministic_comfyui_tool_summary(tool_text),
        deterministic_port_status_tool_summary(tool_text),
        deterministic_engine_inventory_tool_summary(tool_text),
        deterministic_services_tool_summary(tool_text),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    (!summaries.is_empty()).then(|| summaries.join("\n"))
}

/// Whether a deterministic summary of `tool_text` answers the turn by itself.
///
/// The model summary does not: it names three recipes out of a catalog of
/// dozens, so whatever the user actually asked about the catalog still goes to
/// the local model. Keyed on the same listing check the summary itself uses, so
/// the two cannot disagree about what a model listing is.
pub(super) fn deterministic_summary_can_stand_alone(tool_text: &str) -> bool {
    !text_is_model_catalog_listing(tool_text)
}

/// The canonical id of the tiny GGUF model the catalog carries for checking
/// that GPU serving starts at all.
const SMOKE_TEST_MODEL_ID: &str = "Qwen3-0.6B-GGUF";

/// The canonical id of the catalog's 8 GiB-class chat recipe.
const EIGHT_GIB_CLASS_MODEL_ID: &str = "meta-llama/Llama-3.2-3B-Instruct";

/// The chat's standing answer to "what can this machine run": the low-VRAM
/// assistant, the smoke-test model and the 8 GiB-class option, by name.
///
/// Worth printing because the `rocm model` listing the chat tool renders is
/// grouped by hardware target and, on a small GPU, names nothing that fits.
/// `tool_text` only identifies the tool; every value comes from the registry,
/// so a change to either listing's wording cannot quietly empty this out.
fn deterministic_model_tool_summary(paths: &AppPaths, tool_text: &str) -> Option<String> {
    if !text_is_model_catalog_listing(tool_text) {
        return None;
    }
    let registry = load_model_recipe_registry().ok()?;
    let recipe_for = |id: &str| {
        registry
            .recipes
            .iter()
            .find(|recipe| recipe.canonical_model_id == id)
    };

    let mut lines = Vec::new();
    if let Some(recipe) = recipe_for(providers::LEMONADE_ASSISTANT_MODEL_ID) {
        lines.push(format!(
            "  Recommended local assistant: qwen ({})",
            recipe.canonical_model_id
        ));
        lines.push(format!(
            "    Fit: needs about {} GPU memory.",
            model_recipe_gpu_memory(recipe, "unknown")
        ));
        lines.push(format!(
            "    Engine: {}",
            engine_fit_labels(&model_recipe_engine_availability(recipe, Some(paths)))
        ));
    }

    if let Some(recipe) = recipe_for(SMOKE_TEST_MODEL_ID) {
        lines.push(format!(
            "  Tiny smoke test: qwen-smoke ({})",
            recipe.canonical_model_id
        ));
        lines.push(
            "    Use this only to check that GPU serving starts; it is not the default assistant."
                .to_owned(),
        );
    }

    if let Some(recipe) = recipe_for(EIGHT_GIB_CLASS_MODEL_ID) {
        lines.push(format!(
            "  8 GiB-class option: llama ({})",
            recipe.canonical_model_id
        ));
        lines.push(format!(
            "    Fit: asks for {}; this can be tight on APUs and depends on available shared GPU memory.",
            model_recipe_gpu_memory(recipe, "unknown GPU memory")
        ));
        lines.push(format!(
            "    Engines: {}",
            engine_fit_labels(&model_recipe_engine_availability(recipe, Some(paths)))
        ));
    }

    if lines.is_empty() {
        return None;
    }
    lines.push(
        "  Run `rocm examine` to refresh GPU memory details before starting anything large."
            .to_owned(),
    );
    Some(lines.join("\n"))
}

fn model_recipe_gpu_memory(recipe: &ModelRecipeRecord, unknown: &str) -> String {
    recipe
        .min_gpu_mem_gb
        .map_or_else(|| unknown.to_owned(), |value| format!("{value} GiB"))
}

/// Each preferred engine with what this host can do with it, as
/// `lemonade: built_in` or `vllm: WSL/Linux only on Windows`.
///
/// A recipe with no preferred engine reads `<none listed>` rather than an empty
/// tail, so the line never ends in a dangling colon.
fn engine_fit_labels(engines: &[ModelEngineAvailability]) -> String {
    if engines.is_empty() {
        return "<none listed>".to_owned();
    }
    engines
        .iter()
        .map(engine_fit_label)
        .collect::<Vec<_>>()
        .join(", ")
}

fn engine_fit_label(engine: &ModelEngineAvailability) -> String {
    use crate::PlatformRuleOut;
    use crate::model_catalog_text::ModelEngineLocation;

    let name = &engine.engine;
    match engine.platform_rule_out {
        PlatformRuleOut::NativeWindowsVllm | PlatformRuleOut::NativeWindowsAim => {
            format!("{name}: WSL/Linux only on Windows")
        }
        PlatformRuleOut::AimDockerUnavailable => format!("{name}: needs a reachable Docker daemon"),
        PlatformRuleOut::Allowed => match engine.location {
            ModelEngineLocation::BuiltIn => format!("{name}: built_in"),
            ModelEngineLocation::Plugin(_) => format!("{name}: available"),
            ModelEngineLocation::Unavailable(_) => format!("{name}: not installed"),
            ModelEngineLocation::NotChecked => format!("{name}: not checked"),
        },
    }
}

pub(crate) fn deterministic_comfyui_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("ComfyUI") || !tool_text.contains("Running") {
        return None;
    }
    let mut lines = Vec::new();
    match chat_tool_value(tool_text, "installed").as_deref() {
        Some("yes") => lines.push("  ComfyUI: installed.".to_owned()),
        Some("no") => lines.push("  ComfyUI: not installed.".to_owned()),
        Some(value) => lines.push(format!("  ComfyUI: installed status is {value}.")),
        None => {}
    }
    let status = chat_tool_value(tool_text, "status");
    // One test of "unknown" for both the `Running:` line and the notes below,
    // so the label and the reason behind it cannot disagree.
    let status_unknown = status
        .as_deref()
        .is_some_and(|status| status.eq_ignore_ascii_case("unknown"));
    if let Some(status) = status {
        let running_text = if status.eq_ignore_ascii_case("running") {
            "running".to_owned()
        } else if status.eq_ignore_ascii_case("starting") {
            "starting".to_owned()
        } else if status_unknown {
            "unknown".to_owned()
        } else if status.eq_ignore_ascii_case(crate::comfyui::RUNNING_OUTSIDE_ROCM_CLI_STATUS) {
            // Something answers that rocm-cli did not start; the chat's own
            // status answer calls this running too.
            "running (not started by rocm-cli)".to_owned()
        } else {
            format!("not running ({status})")
        };
        lines.push(format!("  Running: {running_text}."));
    }
    if let Some(models_path) = chat_tool_value(tool_text, "models path") {
        lines.push(format!("  Models folder: {models_path}."));
    }
    // A note about the install itself (an interrupted reinstall, which names
    // the command that finishes it) sits between `status`'s `ComfyUI` heading
    // and its `Running` section, whose own notes the `Running:` line sums up.
    for line in tool_text
        .lines()
        .skip_while(|line| line.trim() != "ComfyUI")
        .take_while(|line| line.trim() != "Running")
    {
        if let Some(note) = line.trim().strip_prefix("note:").map(str::trim)
            && !note.is_empty()
        {
            lines.push(format!("  Note: {note}."));
        }
    }
    // An unknown running state is not summed up by the `Running:` line: its
    // notes say why it is unknown and what answers, and its next step how to
    // clear it, so they are passed on.
    if status_unknown {
        for line in tool_text
            .lines()
            .skip_while(|line| line.trim() != "Running")
            .skip(1)
            .take_while(|line| line.starts_with("  "))
        {
            let line = line.trim();
            if let Some(note) = line.strip_prefix("note:").map(str::trim) {
                lines.push(format!("  Note: {note}."));
            } else if let Some(step) = line.strip_prefix("next step:").map(str::trim) {
                lines.push(format!("  Next step: {step}."));
            }
        }
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_services_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("Local Servers")
        && !tool_text.contains("managed_services:")
        && !tool_text.contains("services:")
    {
        return None;
    }
    let rows = parse_deterministic_service_rows(tool_text);
    let mut lines = Vec::new();
    let live = rows
        .iter()
        .filter(|row| matches!(row.running_state.as_str(), "running" | "starting"))
        .collect::<Vec<_>>();
    if live.is_empty() {
        lines.push("  Local model servers: none running under ROCm CLI.".to_owned());
    } else {
        lines.push(format!(
            "  Local model servers running/starting: {}.",
            live.len()
        ));
        for row in live.iter().take(4) {
            lines.push(format!(
                "    {} {} ({}) at {}.",
                row.engine,
                row.model,
                row.running_state,
                empty_as_unknown(&row.endpoint)
            ));
        }
    }
    let past = rows
        .iter()
        .filter(|row| !matches!(row.running_state.as_str(), "running" | "starting"))
        .take(3)
        .map(|row| {
            format!(
                "{} {} {}",
                empty_as_unknown(&row.engine),
                empty_as_unknown(&row.model),
                empty_as_unknown(&row.status)
            )
        })
        .collect::<Vec<_>>();
    if !past.is_empty() {
        lines.push(format!("  Past/non-running records: {}.", past.join("; ")));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_port_status_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("port_status:") && !tool_text.contains("listening:") {
        return None;
    }
    let port = chat_tool_value(tool_text, "port")?;
    let listening = chat_tool_value(tool_text, "listening").unwrap_or_else(|| "unknown".to_owned());
    let mut lines = Vec::new();
    let state = match listening.as_str() {
        "true" => "listening",
        "false" => "not listening",
        _ => "unknown",
    };
    lines.push(format!("  Port {port}: {state}."));
    if let Some(hint) = chat_tool_value(tool_text, "hint") {
        lines.push(format!("  Hint: {hint}."));
    }
    if chat_tool_value(tool_text, "managed_service").as_deref() == Some("none") {
        lines.push("  Managed service: none on that endpoint.".to_owned());
    }
    Some(lines.join("\n"))
}

fn deterministic_engine_inventory_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("Local model engines") {
        return None;
    }
    let rows = parse_deterministic_engine_rows(tool_text);
    if rows.is_empty() {
        return None;
    }
    let mut lines = vec!["  Engine runtimes:".to_owned()];
    for row in rows {
        let mut line = format!(
            "    {}: {}",
            friendly_engine_label(&row.engine),
            empty_as_unknown(&row.runtime)
        );
        if !row.note.trim().is_empty() {
            let _ = write!(line, " ({})", row.note.trim());
        }
        line.push('.');
        lines.push(line);
    }
    Some(lines.join("\n"))
}

fn chat_tool_value(tool_text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    tool_text.lines().find_map(|line| {
        line.trim()
            .strip_prefix(&prefix)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;

    /// A ComfyUI that answers but that rocm-cli did not start is running, as
    /// the chat's own status answer says (`Running: yes`), not "not running".
    #[test]
    fn a_comfyui_running_outside_rocm_cli_is_summed_up_as_running() {
        let status = format!(
            "ComfyUI\n\n  installed: yes\n\nRunning\n  status: {}\n  url: \
             http://127.0.0.1:8188\n  note: ROCm CLI did not start this ComfyUI process\n",
            crate::comfyui::RUNNING_OUTSIDE_ROCM_CLI_STATUS.to_uppercase()
        );

        let summary = deterministic_comfyui_tool_summary(&status).expect("a summary");

        assert!(
            summary.contains("  Running: running (not started by rocm-cli)."),
            "{summary}"
        );
        assert!(!summary.contains("not running"), "{summary}");
    }

    /// The `Running:` label and the notes behind an unknown state come from
    /// one case-insensitive test, so a differently cased status keeps both.
    #[test]
    fn an_unknown_comfyui_state_keeps_its_notes_whatever_its_case() {
        let status = "ComfyUI\n\n  installed: yes\n\nRunning\n  status: UNKNOWN\n  note: the \
                      record cannot be parsed\n  next step: run `rocm comfyui stop`\n";

        let summary = deterministic_comfyui_tool_summary(status).expect("a summary");

        assert!(summary.contains("  Running: unknown."), "{summary}");
        assert!(
            summary.contains("  Note: the record cannot be parsed."),
            "{summary}"
        );
        assert!(
            summary.contains("  Next step: run `rocm comfyui stop`."),
            "{summary}"
        );
    }

    #[test]
    fn parse_deterministic_service_rows_reads_inline_key_value_form() {
        let rows = parse_deterministic_service_rows(
            "  - service_id=svc-1 engine=vllm model=qwen status=ready running_state=running endpoint=127.0.0.1:8000",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].service_id, "svc-1");
        assert_eq!(rows[0].engine, "vllm");
        assert_eq!(rows[0].status, "ready");
        assert_eq!(rows[0].running_state, "running");
        assert_eq!(rows[0].endpoint, "127.0.0.1:8000");
    }

    #[test]
    fn parse_deterministic_service_rows_derives_running_state_when_absent() {
        let rows = parse_deterministic_service_rows(
            "\
- svc-2
status: starting
engine: lemonade
model: qwen
endpoint: 127.0.0.1:8001
",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].service_id, "svc-2");
        assert_eq!(rows[0].running_state, "starting");
    }

    #[test]
    fn parse_deterministic_service_rows_derives_running_state_for_inline_form_when_absent() {
        let rows = parse_deterministic_service_rows(
            "  - service_id=svc-3 engine=vllm model=qwen status=starting endpoint=127.0.0.1:8002",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].running_state, "starting");
    }

    #[test]
    fn parse_deterministic_engine_rows_reads_runtime_and_note() {
        let rows = parse_deterministic_engine_rows(
            "\
* lemonade
  runtime: gfx120X-all
  note: default engine
* vllm
  runtime: gfx110X-all
",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].engine, "lemonade");
        assert_eq!(rows[0].runtime, "gfx120X-all");
        assert_eq!(rows[0].note, "default engine");
        assert_eq!(rows[1].engine, "vllm");
        assert_eq!(rows[1].note, "");
    }

    #[test]
    fn managed_service_running_state_maps_every_known_status() {
        assert_eq!(managed_service_running_state("ready"), "running");
        assert_eq!(managed_service_running_state("running"), "running");
        assert_eq!(managed_service_running_state("starting"), "starting");
        assert_eq!(managed_service_running_state("recovering"), "starting");
        assert_eq!(managed_service_running_state("failed"), "not_running");
        assert_eq!(managed_service_running_state("stopped"), "not_running");
        assert_eq!(managed_service_running_state("anything-else"), "unknown");
    }

    #[test]
    fn deterministic_rocm_tool_summary_interprets_managed_runtime_as_installed() {
        let summary = deterministic_rocm_tool_summary(
            "\
examine:
  driver_detail: AMD Radeon RX 9070 XT driver 32.0.23033.1002
  legacy_rocm_status: not_detected
runtime_state:
  active_runtime_status: ready
  active_runtime_root: D:\\ROCm\\therock_venvs
  active_runtime_pip_cache_dir: D:\\ROCm\\therock_venvs\\pip-cache
  active_runtime_version: 7.13.0a20260511 (build 2026-05-11)
  active_runtime_family: gfx120X-all
",
        )
        .expect("examine output should summarize");

        assert!(summary.contains("GPU: AMD Radeon RX 9070 XT driver 32.0.23033.1002"));
        assert!(summary.contains("ROCm/TheRock: installed and active for ROCm CLI"));
        assert!(summary.contains("gfx120X-all"));
        assert!(summary.contains(r"Install folder: D:\ROCm\therock_venvs"));
        assert!(summary.contains(r"Downloads/cache: D:\ROCm\therock_venvs\pip-cache"));
        assert!(summary.contains("no global legacy ROCm install was found"));
    }

    /// Both `rocm model` renderers are run for real and their output fed to the
    /// summary, so the summary cannot keep passing against a listing format
    /// nothing renders any more. Pointing the chat tool at the curated listing
    /// in #1041 silently emptied this block, and the fixture test that stood
    /// here — hand-written in the `--verbose` format — stayed green through it.
    #[test]
    fn both_model_listings_summarize_the_low_vram_assistant() {
        let (_root, paths) = test_paths("model-summary-from-renderers");
        for listing in [
            crate::render_model_registry_text_with_context_and_host(Some(&paths), None, None),
            crate::render_model_registry_verbose_text_with_context_and_host(
                Some(&paths),
                None,
                None,
            ),
        ] {
            let summary = deterministic_model_tool_summary(&paths, &listing)
                .unwrap_or_else(|| panic!("this `rocm model` listing must summarize:\n{listing}"));

            assert!(summary.contains(&format!(
                "Recommended local assistant: qwen ({})",
                providers::LEMONADE_ASSISTANT_MODEL_ID
            )));
            assert!(summary.contains("Fit: needs about 4 GiB GPU memory."));
            assert!(summary.contains("Engine: lemonade:"));
            assert!(summary.contains(&format!(
                "Tiny smoke test: qwen-smoke ({SMOKE_TEST_MODEL_ID})"
            )));
            assert!(summary.contains(&format!(
                "8 GiB-class option: llama ({EIGHT_GIB_CLASS_MODEL_ID})"
            )));
            assert!(summary.contains("Fit: asks for 8 GiB;"));
            assert!(summary.contains("Engines: vllm:"));
            assert!(summary.contains(
                "Run `rocm examine` to refresh GPU memory details before starting anything large."
            ));
            // Dropped with the run-on list of every larger recipe: it was the
            // bulk of the block and the worst value per token for a small model.
            assert!(!summary.contains("Larger recipes"), "{summary}");
        }
    }

    /// The listing text only says *which tool* produced it; every value in the
    /// summary comes from the recipe registry. A header on its own therefore
    /// summarizes exactly as a full listing does — which is what makes the
    /// summary survive a rewording of either renderer.
    #[test]
    fn model_summary_values_come_from_the_registry_not_the_listing_text() {
        let (_root, paths) = test_paths("model-summary-registry-sourced");
        let from_header_only =
            deterministic_model_tool_summary(&paths, "Recommended models - run one with x")
                .expect("the curated header alone identifies the model tool");
        let from_full_listing = deterministic_model_tool_summary(
            &paths,
            &crate::render_model_registry_text_with_context_and_host(Some(&paths), None, None),
        )
        .expect("the full listing summarizes");
        assert_eq!(from_header_only, from_full_listing);
        assert!(from_header_only.contains("Recommended local assistant: qwen"));
    }

    /// The second place keyed on the listing: a model summary names three
    /// recipes, so the local model still answers what was asked. Both listings
    /// must be recognised, and an unrelated tool's text must not be.
    #[test]
    fn only_a_model_listing_summary_is_barred_from_standing_alone() {
        let (_root, paths) = test_paths("model-summary-stand-alone");
        for listing in [
            crate::render_model_registry_text_with_context_and_host(Some(&paths), None, None),
            crate::render_model_registry_verbose_text_with_context_and_host(
                Some(&paths),
                None,
                None,
            ),
        ] {
            assert!(
                !deterministic_summary_can_stand_alone(&listing),
                "a model listing summary answers only part of the question:\n{listing}"
            );
        }
        assert!(deterministic_summary_can_stand_alone(
            "examine:\n  legacy_rocm_status: not_detected\n"
        ));
    }
}
