// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The ROCm tool schema the providers advertise to a model: the OpenAI
//! function-calling definitions ([`rocm_openai_tool_definitions`]) and the same
//! catalog reshaped for Anthropic's `tools` field ([`rocm_anthropic_tool_definitions`]).
//!
//! A leaf of the `providers` module: nothing here depends on the rest of it.

pub(super) fn rocm_openai_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        rocm_openai_tool(
            "examine",
            "Read the current ROCm host, GPU, runtime, driver, and engine status.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "bridge_snapshot",
            "Read a full ROCm snapshot including examine data, engines, services, automations, and GPU telemetry.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "gpu_snapshot",
            "Read the current local AMD GPU telemetry snapshot when available.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "engines",
            "List available ROCm serving engines and whether each is installed.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "services",
            "List managed local model servers and their current status.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "service_logs",
            "Read the recent log output for a managed local model server.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "service_id": { "type": "string" },
                    "lines": { "type": "integer", "minimum": 1, "maximum": 500 }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "automations",
            "List watcher and automation status plus recent automation events.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "event_limit": { "type": "integer", "minimum": 1, "maximum": 64 }
                },
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "natural_language_plan",
            "Ask ROCm CLI to translate a natural-language ROCm request into a visible plan without executing it.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "request": { "type": "string" }
                },
                "required": ["request"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "path_exists",
            "Read whether an exact local filesystem path exists and whether its parent folder exists. Use this only to check user-provided install folders before requesting a ROCm install.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "port_status",
            "Read whether a local loopback TCP port is listening and whether a ROCm-managed service owns it. Use this for questions like whether something is running on port 8188 or the local model server port. Do not use it for non-local hosts.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "host": { "type": "string" },
                    "port": { "type": "integer", "minimum": 1, "maximum": 65535 }
                },
                "required": ["port"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "rocm_command",
            "Run or request a supported ROCm CLI command with argv-style arguments. Known read-only inspection commands may run immediately. Commands that install, start, serve, stop, delete, or change ROCm state are paused for user review before they run. Use this tool for mutating TheRock setup, including --prefix PATH when the user gives an install folder, --build-date YYYY-MM-DD or --version VERSION, config changes, ComfyUI install/start, engine install, vLLM management, and local LLM serve actions; do not request shell commands, CPU execution, or public network binds.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 64
                    },
                    "reason": { "type": "string" }
                },
                "required": ["args"],
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "update_check",
            "Check whether ROCm CLI knows about newer TheRock runtime versions.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        rocm_openai_tool(
            "install_sdk_dry_run",
            "Preview a TheRock SDK install without changing the machine.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "channel": { "type": "string", "enum": ["release", "nightly"] },
                    "format": { "type": "string", "enum": ["wheel", "tarball"] },
                    "prefix": { "type": "string" },
                    "version": { "type": "string" },
                    "build_date": { "type": "string" }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

fn rocm_openai_tool(
    name: &str,
    description: &str,
    parameters: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
}

pub(super) fn rocm_anthropic_tool_definitions() -> Vec<serde_json::Value> {
    rocm_openai_tool_definitions()
        .into_iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            Some(serde_json::json!({
                "name": function.get("name")?.clone(),
                "description": function.get("description")?.clone(),
                "input_schema": function.get("parameters")?.clone(),
            }))
        })
        .collect()
}
