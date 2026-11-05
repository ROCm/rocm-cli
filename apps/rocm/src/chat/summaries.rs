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
//! The parsers that turn tool text back into rows (`DeterministicModelRecipe`,
//! `DeterministicServiceRow`, `DeterministicEngineRow`) and
//! `managed_service_running_state`, the mapping from a managed service's status
//! to `running`, `starting`, `not_running` or `unknown`, live here too. The
//! services tool text itself is rendered by `chat.rs`
//! (`render_services_tool_result_text`), which reads
//! `managed_service_running_state` from this module.
//!
//! A leaf of the `chat` module: nothing here calls back into `chat.rs`. The
//! only edges out are `providers` (the assistant model ids) and
//! `managed_service_status::friendly_engine_label`.

use crate::managed_service_status::friendly_engine_label;
use crate::providers;
use rocm_core::placeholder::empty_as_unknown;
use std::fmt::Write as _;

#[derive(Debug, Default)]
struct DeterministicModelRecipe {
    pub(crate) canonical_id: String,
    pub(crate) min_gpu_mem_gib: Option<u32>,
    pub(crate) engines: Vec<String>,
    pub(crate) engine_statuses: Vec<String>,
    pub(crate) warnings: Vec<String>,
}

fn parse_deterministic_model_recipes(tool_text: &str) -> Vec<DeterministicModelRecipe> {
    let mut recipes = Vec::new();
    let mut current: Option<DeterministicModelRecipe> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(recipe) = parse_model_recipe_header(trimmed) {
            if let Some(previous) = current.replace(recipe) {
                recipes.push(previous);
            }
            continue;
        }
        let Some(recipe) = current.as_mut() else {
            continue;
        };
        if is_engine_status_line(trimmed) {
            recipe.engine_statuses.push(trimmed.to_owned());
        } else if let Some(warning) = trimmed.strip_prefix("warning:") {
            recipe.warnings.push(warning.trim().to_owned());
        }
    }
    if let Some(recipe) = current {
        recipes.push(recipe);
    }
    recipes
}

fn parse_model_recipe_header(line: &str) -> Option<DeterministicModelRecipe> {
    if !line.contains(" aliases=[") || !line.contains("min_gpu_mem=") {
        return None;
    }
    let canonical_id = line.split_whitespace().next()?.to_owned();
    let engines = parse_bracketed_csv(line, "engines=[", "]");
    let min_gpu_mem_gib = line
        .split_once("min_gpu_mem=")
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u32>().ok());
    Some(DeterministicModelRecipe {
        canonical_id,
        min_gpu_mem_gib,
        engines,
        engine_statuses: Vec::new(),
        warnings: Vec::new(),
    })
}

fn parse_bracketed_csv(line: &str, start: &str, end: &str) -> Vec<String> {
    line.split_once(start)
        .and_then(|(_, rest)| rest.split_once(end))
        .map(|(values, _)| {
            values
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn is_engine_status_line(line: &str) -> bool {
    matches!(
        line.split_once(':').map(|(engine, _)| engine),
        Some("lemonade" | "vllm" | "aim")
    )
}

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

pub(super) fn deterministic_chat_tool_summary(tool_text: &str) -> Option<String> {
    deterministic_rocm_tool_summary(tool_text)
        .or_else(|| deterministic_model_tool_summary(tool_text))
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

pub(super) fn deterministic_summary_can_stand_alone(tool_text: &str) -> bool {
    !tool_text.contains("model recipes")
}

fn deterministic_model_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("model recipes") {
        return None;
    }
    let recipes = parse_deterministic_model_recipes(tool_text);
    if recipes.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == providers::LEMONADE_ASSISTANT_MODEL_ID)
    {
        lines.push(format!(
            "  Recommended local assistant: qwen ({})",
            recipe.canonical_id
        ));
        lines.push(format!(
            "    Fit: needs about {} GPU memory.",
            recipe
                .min_gpu_mem_gib
                .map_or_else(|| "unknown".to_owned(), |value| format!("{value} GiB"))
        ));
        lines.push(format!(
            "    Engine: {}",
            deterministic_engine_fit_summary(recipe, "lemonade")
        ));
    }

    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == "Qwen3-0.6B-GGUF")
    {
        lines.push(format!(
            "  Tiny smoke test: qwen-smoke ({})",
            recipe.canonical_id
        ));
        lines.push(
            "    Use this only to check that GPU serving starts; it is not the default assistant."
                .to_owned(),
        );
    }

    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == "meta-llama/Llama-3.2-3B-Instruct")
    {
        lines.push(format!(
            "  8 GiB-class option: llama ({})",
            recipe.canonical_id
        ));
        lines.push(format!(
            "    Fit: asks for {}; this can be tight on APUs and depends on available shared GPU memory.",
            recipe
                .min_gpu_mem_gib.map_or_else(|| "unknown GPU memory".to_owned(), |value| format!("{value} GiB"))
        ));
        lines.push(format!(
            "    Engines: {}",
            deterministic_available_engine_names(recipe).join(", ")
        ));
    }

    let larger = recipes
        .iter()
        .filter(|recipe| recipe.min_gpu_mem_gib.is_some_and(|value| value > 8))
        .filter(|recipe| recipe.canonical_id != providers::BUILTIN_ASSISTANT_MODEL_ID)
        .map(|recipe| {
            format!(
                "{} asks for {}",
                recipe.canonical_id,
                recipe.min_gpu_mem_gib.map_or_else(
                    || "more GPU memory".to_owned(),
                    |value| format!("{value} GiB")
                )
            )
        })
        .collect::<Vec<_>>();
    if !larger.is_empty() {
        lines.push(format!(
            "  Larger recipes are not low-VRAM defaults: {}.",
            larger.join("; ")
        ));
    }

    let linux_wsl_only = recipes
        .iter()
        .flat_map(|recipe| {
            recipe
                .engine_statuses
                .iter()
                .filter(|status| {
                    status.contains("unsupported_native_windows")
                        || status.to_ascii_lowercase().contains("wsl/linux")
                        || status.to_ascii_lowercase().contains("linux/wsl")
                })
                .filter_map(|status| {
                    let (engine, _) = status.split_once(':')?;
                    Some(format!(
                        "{} uses {} through WSL/Linux on Windows",
                        recipe.canonical_id, engine
                    ))
                })
        })
        .collect::<Vec<_>>();
    if !linux_wsl_only.is_empty() {
        lines.push(format!(
            "  Native Windows note: {}.",
            linux_wsl_only.join("; ")
        ));
    }

    if !lines.is_empty() {
        lines.push(
            "  Run `rocm examine` to refresh GPU memory details before starting anything large."
                .to_owned(),
        );
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_engine_fit_summary(recipe: &DeterministicModelRecipe, engine: &str) -> String {
    let status = recipe
        .engine_statuses
        .iter()
        .find(|status| status.starts_with(&format!("{engine}:")))
        .map(std::string::String::as_str);
    match status {
        Some(status) if status.contains("unsupported_native_windows") => {
            format!("{engine}: WSL/Linux only on Windows")
        }
        Some(status)
            if status.split_once(':').is_some_and(|(_, rest)| {
                rest.trim_start().starts_with("available")
                    || rest.trim_start().starts_with("adapter_available")
            }) =>
        {
            format!("{engine}: available")
        }
        Some(status) => status.to_owned(),
        None if recipe.engines.iter().any(|candidate| candidate == engine) => {
            format!("{engine}: listed")
        }
        None => format!("{engine}: not listed"),
    }
}

fn deterministic_available_engine_names(recipe: &DeterministicModelRecipe) -> Vec<String> {
    let engines = recipe
        .engine_statuses
        .iter()
        .filter_map(|status| {
            let (engine, rest) = status.split_once(':')?;
            (rest.trim_start().starts_with("available")
                || rest.trim_start().starts_with("adapter_available"))
            .then_some(engine.to_owned())
        })
        .collect::<Vec<_>>();
    if engines.is_empty() {
        recipe.engines.clone()
    } else {
        engines
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
    if let Some(status) = chat_tool_value(tool_text, "status") {
        let running_text = if status.eq_ignore_ascii_case("running") {
            "running".to_owned()
        } else if status.eq_ignore_ascii_case("starting") {
            "starting".to_owned()
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

    #[test]
    fn parse_deterministic_model_recipes_reads_header_fields_and_notes() {
        let recipes = parse_deterministic_model_recipes(
            "\
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen, lemonade-qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
lemonade: ready
warning: slow on this GPU
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
vllm: not_installed
",
        );
        assert_eq!(recipes.len(), 2);
        assert_eq!(recipes[0].canonical_id, "Qwen3-4B-Instruct-2507-GGUF");
        assert_eq!(recipes[0].min_gpu_mem_gib, Some(4));
        assert_eq!(recipes[0].engines, vec!["lemonade".to_owned()]);
        assert_eq!(
            recipes[0].engine_statuses,
            vec!["lemonade: ready".to_owned()]
        );
        assert_eq!(recipes[0].warnings, vec!["slow on this GPU".to_owned()]);
        assert_eq!(recipes[1].canonical_id, "Qwen/Qwen3.5-4B");
        assert_eq!(recipes[1].min_gpu_mem_gib, Some(12));
    }

    #[test]
    fn parse_deterministic_model_recipes_requires_aliases_bracket_in_header() {
        let recipes = parse_deterministic_model_recipes(
            "\
  NotAHeader task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
",
        );
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].canonical_id, "Qwen/Qwen3.5-4B");
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

    #[test]
    fn deterministic_model_tool_summary_identifies_low_vram_assistant() {
        let summary = deterministic_model_tool_summary(
            "\
rocm_command:
model recipes
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen, lemonade-qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
      warning: recommended Lemonade GGUF assistant for ROCm machines
  Qwen3-0.6B-GGUF aliases=[qwen-smoke, lemonade-tiny] task=chat dtype=gguf device=gpu_required min_gpu_mem=2 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
      warning: tiny Lemonade GGUF smoke-test model; not the default assistant
  Qwen/Qwen2.5-0.5B-Instruct aliases=[qwen-tiny] task=chat dtype=float16 device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
      engine_support:
        vllm: adapter_available path=D:\\rocm\\rocm-engine-vllm.exe runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux vLLM ROCm
  meta-llama/Llama-3.2-3B-Instruct aliases=[llama] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=8 GiB engines=[lemonade, vllm]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
        vllm: available path=D:\\rocm\\rocm-engine-vllm.exe
",
        )
        .expect("model output should summarize");

        assert!(summary.contains("Recommended local assistant: qwen"));
        assert!(summary.contains("Qwen3-4B-Instruct-2507-GGUF"));
        assert!(summary.contains("4 GiB"));
        assert!(summary.contains("Tiny smoke test: qwen-smoke"));
        assert!(summary.contains("Qwen3-0.6B-GGUF"));
        assert!(summary.contains("8 GiB-class option: llama"));
        assert!(summary.contains("lemonade, vllm"));
        assert!(summary.contains("Qwen/Qwen3.5-4B asks for 12 GiB"));
        assert!(summary.contains("Native Windows note"));
        assert!(summary.contains("Run `rocm examine`"));
    }
}
