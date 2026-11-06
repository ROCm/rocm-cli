// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The chat assistant's prompt-intent heuristics: what a user's latest message
//! asks for, decided from its words alone. `fallback_rocm_tool_call_for_prompt`
//! maps a prompt to the read-only or approval-gated tool call the assistant
//! proposes when it does not need the local model to choose one (with
//! `fallback_config_tool_call_for_prompt`, `fallback_engine_list_tool_call` and
//! `fallback_services_list_tool_call` behind it),
//! `prompt_can_use_read_only_without_local_assistant` says whether a status
//! question can be answered before any model server is up, and `install_sdk_chat_approval_for_prompt` (with
//! `install_sdk_without_prefix_chat_approval`) turns an install or setup
//! request into the `rocm install sdk` approval card. The `prompt_asks_*`,
//! `prompt_mentions_*` and `prompt_requests_*` predicates hold the phrase
//! lists, and `latest_user_chat_message` cuts the newest message out of the
//! conversation text the chat builds around it, so older words in the
//! conversation never steer these heuristics.
//!
//! A leaf of the `chat` module: nothing here calls back into `chat.rs`. The
//! edges out are `prompt_text.rs` (the phrase and install-detail extraction),
//! `rocm_argv.rs`, `tool_policy.rs` (`ChatToolApprovalRequest`),
//! `freeform_planner.rs` (`format_structured_tool_call`) and `providers` (the
//! tool-call type). The turn loop in `chat/turn.rs` reads this module's items
//! through `super::intent`.

use crate::freeform_planner::format_structured_tool_call;
use crate::prompt_text::{
    any_substring, infer_engine_from_request, requested_install_prefix_from_prompt,
    requested_therock_build_date_from_prompt, requested_therock_version_from_prompt,
};
use crate::providers;
use crate::rocm_argv::chat_cli_arg_value;
use crate::tool_policy::ChatToolApprovalRequest;
use rocm_core::DEFAULT_LOCAL_HOST;

pub(super) fn latest_user_chat_message(prompt: &str) -> &str {
    const MARKER: &str = "\nNew message:\n";
    prompt
        .rfind(MARKER)
        .map_or(prompt, |index| &prompt[index + MARKER.len()..])
        .trim()
}

pub(super) fn prompt_can_use_read_only_without_local_assistant(prompt: &str) -> bool {
    let normalized = latest_user_chat_message(prompt).to_ascii_lowercase();
    let mentions_status_subject = any_substring(
        &normalized,
        &[
            "comfyui",
            "comfy ui",
            "comfy",
            "vllm",
            "lemonade",
            "llama.cpp",
            "llama cpp",
            "qwen",
            "model server",
            "local server",
            "local model server",
            "assistant server",
            "port",
            "8188",
            "therock",
            "rocm",
        ],
    );
    mentions_status_subject
        && (prompt_asks_running_or_status(&normalized)
            || any_substring(
                &normalized,
                &["installed", "available", "detected", "engine status"],
            ))
}

pub(super) fn prompt_mentions_serving_engine_or_service(normalized_prompt: &str) -> bool {
    any_substring(
        normalized_prompt,
        &[
            "vllm",
            "lemonade",
            "llama.cpp",
            "llama cpp",
            "qwen",
            "model server",
            "local server",
            "local model server",
            "assistant server",
        ],
    )
}

pub(super) fn prompt_asks_engine_install_state(normalized_prompt: &str) -> bool {
    any_substring(
        normalized_prompt,
        &["installed", "available", "detected", "engine status"],
    )
}

pub(super) fn fallback_engine_list_tool_call() -> providers::ChatToolCall {
    providers::ChatToolCall {
        id: Some("fallback-engine-list".to_owned()),
        name: "rocm_command".to_owned(),
        arguments: serde_json::json!({ "args": ["engines", "list"] }),
    }
}

pub(super) fn fallback_services_list_tool_call() -> providers::ChatToolCall {
    providers::ChatToolCall {
        id: Some("fallback-services-list".to_owned()),
        name: "rocm_command".to_owned(),
        arguments: serde_json::json!({ "args": ["services", "list", "--all"] }),
    }
}

pub(super) fn fallback_rocm_tool_call_for_prompt(prompt: &str) -> Option<providers::ChatToolCall> {
    let prompt = latest_user_chat_message(prompt);
    let normalized = prompt.to_ascii_lowercase();
    let asks_running_or_status = prompt_asks_running_or_status(&normalized);
    let mentions_comfyui = any_substring(&normalized, &["comfyui", "comfy ui", "comfy"]);
    if asks_running_or_status
        && !mentions_comfyui
        && any_substring(&normalized, &["8188", "port 8188"])
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-port-8188-status".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": DEFAULT_LOCAL_HOST, "port": 8188 }),
        });
    }
    let mentions_serving_engine_or_service = prompt_mentions_serving_engine_or_service(&normalized);
    if mentions_serving_engine_or_service && prompt_asks_engine_install_state(&normalized) {
        return Some(fallback_engine_list_tool_call());
    }
    if mentions_serving_engine_or_service && asks_running_or_status {
        return Some(fallback_services_list_tool_call());
    }
    let mentions_llm_or_model = any_substring(
        &normalized,
        &["llm", "llms", "model", "models", "assistant"],
    );
    let asks_support_or_fit = any_substring(
        &normalized,
        &[
            "support",
            "supported",
            "run",
            "runs",
            "fit",
            "fits",
            "can my machine",
            "can this machine",
        ],
    );
    if mentions_llm_or_model && asks_support_or_fit {
        return Some(providers::ChatToolCall {
            id: Some("fallback-rocm-model".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["model"] }),
        });
    }

    if mentions_comfyui {
        if any_substring(&normalized, &["log", "logs"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-logs".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["comfyui", "logs"] }),
            });
        }
        if asks_running_or_status {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-status".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["comfyui", "status"] }),
            });
        }
        if any_substring(&normalized, &["start", "run", "launch", "open"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-start".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["comfyui", "start"],
                    "reason": "Start ComfyUI locally after the user approves it."
                }),
            });
        }
        if any_substring(
            &normalized,
            &[
                "can you setup",
                "can you set up",
                "please setup",
                "please set up",
                "setup comfyui for me",
                "set up comfyui for me",
                "install comfyui",
                "download comfyui",
            ],
        ) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-install".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["comfyui", "install"],
                    "reason": "Install ComfyUI into ROCm CLI's managed app folder after the user approves it."
                }),
            });
        }
        return Some(providers::ChatToolCall {
            id: Some("fallback-comfyui-status".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["comfyui", "status"] }),
        });
    }

    if let Some(call) = fallback_config_tool_call_for_prompt(&normalized) {
        return Some(call);
    }

    if mentions_llm_or_model
        && any_substring(
            &normalized,
            &[
                "serve",
                "server",
                "start",
                "setup and serve",
                "set up and serve",
                "run locally",
                "local model",
                "local assistant",
            ],
        )
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-serve-qwen".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "lemonade", "--device", "gpu_required", "--managed"],
                "reason": "Start the recommended local assistant after the user approves it."
            }),
        });
    }

    let mentions_setup = any_substring(&normalized, &["setup", "set up", "install"]);
    let asks_how = any_substring(&normalized, &["how", "help", "what do i need"]);
    let mentions_rocm_or_therock = any_substring(&normalized, &["therock", "rocm"]);
    let requested_install_prefix = requested_install_prefix_from_prompt(prompt);
    if requested_install_prefix.is_none()
        && let Some(approval) = install_sdk_without_prefix_chat_approval(prompt)
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-install-folder".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": approval.args,
                "reason": "Ask the user to choose the ROCm/TheRock install folder before installing."
            }),
        });
    }
    if mentions_rocm_or_therock
        && prompt_requests_install_action(&normalized)
        && let Some(prefix) = requested_install_prefix
    {
        let mut args = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--channel".to_owned(),
            "release".to_owned(),
            "--format".to_owned(),
            "wheel".to_owned(),
            "--prefix".to_owned(),
            prefix,
        ];
        if let Some(build_date) = requested_therock_build_date_from_prompt(&normalized) {
            args.push("--build-date".to_owned());
            args.push(build_date);
        } else if let Some(version) = requested_therock_version_from_prompt(&normalized) {
            args.push("--version".to_owned());
            args.push(version);
        }
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-install".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": args,
                "reason": "Install TheRock ROCm into the user-selected folder after the user approves it."
            }),
        });
    }
    if asks_how && mentions_setup && any_substring(&normalized, &["therock", "rocm"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-examine".to_owned()),
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        });
    }

    let asks_where_installed = any_substring(
        &normalized,
        &[
            "where is rocm",
            "where's rocm",
            "where is therock",
            "where's therock",
            "where did rocm",
            "where did therock",
            "where rocm is installed",
            "where therock is installed",
            "rocm install folder",
            "therock install folder",
            "rocm installed at",
            "therock installed at",
        ],
    );
    let asks_status = asks_where_installed
        || any_substring(
            &normalized,
            &[
                "is rocm installed",
                "is therock installed",
                "is therock setup",
                "is therock set up",
                "rocm installed",
                "therock installed",
                "check this rocm setup",
                "which gpu",
                "what gpu",
                "gpu is on",
                "gpu do i have",
                "my machine",
            ],
        );
    if asks_status && any_substring(&normalized, &["gpu", "rocm", "therock", "setup"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-examine".to_owned()),
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        });
    }

    None
}

pub(super) fn prompt_asks_running_or_status(normalized: &str) -> bool {
    any_substring(
        normalized,
        &[
            "running",
            "is it up",
            "is this up",
            "is there",
            "are there",
            "status",
            "started",
            "listening",
            "on port",
            "port ",
        ],
    )
}

fn fallback_config_tool_call_for_prompt(normalized: &str) -> Option<providers::ChatToolCall> {
    if !any_substring(
        normalized,
        &[
            "config",
            "setting",
            "settings",
            "default engine",
            "telemetry",
        ],
    ) {
        return None;
    }
    if any_substring(normalized, &["show", "check", "what", "current", "list"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-config-show".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["config", "show"] }),
        });
    }
    if any_substring(normalized, &["default engine", "set engine", "use engine"]) {
        // Word-boundary inference: a bare `contains("aim")` matched "aiming".
        if let Some(engine) = infer_engine_from_request(normalized) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-default-engine".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-default-engine", engine],
                    "reason": "Change ROCm CLI's default engine after the user approves it."
                }),
            });
        }
    }
    if normalized.contains("telemetry") {
        if any_substring(normalized, &["off", "disable", "disabled"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-telemetry-off".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-telemetry", "off"],
                    "reason": "Turn ROCm CLI telemetry off after the user approves it."
                }),
            });
        }
        if any_substring(normalized, &["local", "on", "enable", "enabled"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-telemetry-local".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-telemetry", "local"],
                    "reason": "Enable local-only ROCm inspection after the user approves it."
                }),
            });
        }
    }
    None
}

pub(super) fn install_sdk_without_prefix_chat_approval(
    prompt: &str,
) -> Option<ChatToolApprovalRequest> {
    install_sdk_chat_approval_for_prompt(prompt).and_then(|approval| {
        chat_cli_arg_value(&approval.args, "--prefix")
            .is_none()
            .then_some(approval)
    })
}

pub(super) fn install_sdk_chat_approval_for_prompt(
    prompt: &str,
) -> Option<ChatToolApprovalRequest> {
    let prompt = latest_user_chat_message(prompt);
    let normalized = prompt.to_ascii_lowercase();
    if !prompt_requests_rocm_install_or_setup(&normalized) {
        return None;
    }
    let mut args = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        if normalized.contains("nightly") {
            "nightly".to_owned()
        } else {
            "release".to_owned()
        },
        "--format".to_owned(),
        "wheel".to_owned(),
    ];
    if let Some(prefix) = requested_install_prefix_from_prompt(prompt) {
        args.push("--prefix".to_owned());
        args.push(prefix);
    }
    if let Some(build_date) = requested_therock_build_date_from_prompt(&normalized) {
        args.push("--build-date".to_owned());
        args.push(build_date);
    } else if let Some(version) = requested_therock_version_from_prompt(&normalized) {
        args.push("--version".to_owned());
        args.push(version);
    }
    Some(ChatToolApprovalRequest {
        pending_title: "Install ROCm".to_owned(),
        command_title: "Install".to_owned(),
        display_command: Some(format_structured_tool_call("rocm", &args)),
        args,
        explanation: Some(
            "Install ROCm/TheRock after the user chooses the install folder.".to_owned(),
        ),
    })
}

fn prompt_requests_rocm_install_or_setup(normalized: &str) -> bool {
    let mentions_rocm_stack = any_substring(
        normalized,
        &[
            "rocm",
            "therock",
            "the rock",
            "amd gpu package",
            "amd gpu ready",
            "local ai",
        ],
    );
    if !mentions_rocm_stack {
        return false;
    }
    prompt_requests_install_action(normalized)
        || any_substring(
            normalized,
            &[
                "need rocm",
                "need therock",
                "need the rock",
                "need to get rocm",
                "need to get therock",
                "get rocm installed",
                "get therock installed",
                "get the rock installed",
                "make rocm work",
                "make therock work",
                "make the rock work",
                "prepare rocm",
                "prepare therock",
                "prepare the rock",
                "set up my amd gpu",
                "setup my amd gpu",
                "set up local ai",
                "setup local ai",
                "get local ai working",
                "make local ai work",
                "rocm please",
                "make my amd gpu ready",
                "make my gpu ready",
                "get my amd gpu ready",
                "get my gpu ready",
            ],
        )
}

fn prompt_requests_install_action(normalized: &str) -> bool {
    any_substring(
        normalized,
        &[
            "install this",
            "install specific",
            "install the",
            "install rocm",
            "install therock",
            "install the rock",
            "install amd gpu package",
            "install local ai",
            "can you install",
            "please install",
            "rocm please",
            "therock please",
            "the rock please",
            "need to install",
            "want to install",
            "i need rocm",
            "i need therock",
            "i need the rock",
            "get me rocm",
            "get me therock",
            "get my gpu ready",
            "make my gpu ready",
            "make my amd gpu ready",
            "get installed",
            "setup for me",
            "set up for me",
            "setup rocm",
            "set up rocm",
            "setup therock",
            "set up therock",
            "setup the rock",
            "set up the rock",
            "setup my gpu",
            "set up my gpu",
            "setup amd gpu",
            "set up amd gpu",
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_policy::{
        chat_tool_approval_request, chat_tool_call_is_read_only, normalized_chat_rocm_command_args,
        rocm_chat_tool_requested_command,
    };

    #[test]
    fn fallback_tool_call_routes_where_installed_to_read_only_examine() {
        for prompt in [
            "where is rocm installed?",
            "where is TheRock installed?",
            "what is the ROCm install folder?",
            "where did rocm install to?",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "examine", "{prompt}");
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }
    }

    #[test]
    fn default_engine_fallback_matches_aim_only_as_a_word() {
        let proposed = |prompt: &str| {
            fallback_config_tool_call_for_prompt(prompt).map(|call| {
                call.arguments["args"][2]
                    .as_str()
                    .expect("engine arg")
                    .to_owned()
            })
        };
        assert_eq!(
            proposed("change the default engine to aim").as_deref(),
            Some("aim")
        );
        assert_eq!(
            proposed("change the default engine to vllm, aiming for low latency").as_deref(),
            Some("vllm")
        );
        assert_eq!(
            proposed(
                "change the default engine so I can maintain my claim, aiming for low latency"
            ),
            None
        );
    }

    #[test]
    fn fallback_tool_call_runs_model_support_checks() {
        let call =
            fallback_rocm_tool_call_for_prompt("Which LLMs can this machine support?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec!["model".to_owned()]
        );

        assert!(fallback_rocm_tool_call_for_prompt("Which LLM are you using?").is_none());
    }

    #[test]
    fn fallback_tool_call_routes_running_questions_to_status_tools() {
        let comfy = fallback_rocm_tool_call_for_prompt("Is ComfyUI running?").unwrap();
        assert_eq!(comfy.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&comfy).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&comfy));

        let comfy_port =
            fallback_rocm_tool_call_for_prompt("Is ComfyUI running on port 8188?").unwrap();
        assert_eq!(comfy_port.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&comfy_port).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&comfy_port));

        for prompt in [
            "Is vLLM running?",
            "is lemonade running?",
            "is the model server running?",
            "is qwen running?",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command", "{prompt}");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec!["services".to_owned(), "list".to_owned(), "--all".to_owned(),],
                "{prompt}"
            );
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }

        let port = fallback_rocm_tool_call_for_prompt("what is running on port 8188?").unwrap();
        assert_eq!(port.name, "port_status");
        assert_eq!(
            port.arguments,
            serde_json::json!({ "host": DEFAULT_LOCAL_HOST, "port": 8188 })
        );
        assert!(chat_tool_call_is_read_only(&port));
    }

    #[test]
    fn fallback_tool_call_routes_engine_install_state_to_engines_list() {
        for prompt in ["is vLLM installed?", "is Lemonade available?"] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command", "{prompt}");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec!["engines".to_owned(), "list".to_owned()],
                "{prompt}"
            );
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }
    }

    #[test]
    fn fallback_tool_call_runs_status_checks() {
        for prompt in [
            "Which GPU is on my machine, and is ROCm installed?",
            "Is TheRock setup on my machine?",
            "Check this ROCm setup.",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "examine");
            assert_eq!(call.arguments, serde_json::json!({}));
        }
    }

    #[test]
    fn fallback_tool_call_routes_therock_setup_to_install_folder_flow() {
        let call = fallback_rocm_tool_call_for_prompt("How do I setup TheRock?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn fallback_tool_call_preserves_requested_therock_install_prefix() {
        for (prompt, expected_prefix) in [
            (
                "install TheRock for me in D:\\ROCm\\therock_venvs",
                "D:\\ROCm\\therock_venvs",
            ),
            ("install ROCm to D:\\ROCm\\temp", "D:\\ROCm\\temp"),
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                    "--prefix".to_owned(),
                    expected_prefix.to_owned(),
                ]
            );
            assert!(!chat_tool_call_is_read_only(&call));
        }
    }

    #[test]
    fn fallback_tool_call_routes_requested_therock_build_date_install() {
        for (prompt, expected_prefix) in [
            (
                "Install this specific TheRock wheel from date 06052026 into D:\\ROCm\\therock_venvs",
                "D:\\ROCm\\therock_venvs",
            ),
            (
                "install ROCm at D:\\ROCm\\temp with build date 2026-06-05",
                "D:\\ROCm\\temp",
            ),
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                    "--prefix".to_owned(),
                    expected_prefix.to_owned(),
                    "--build-date".to_owned(),
                    "2026-06-05".to_owned(),
                ],
                "{prompt}"
            );
            assert!(!chat_tool_call_is_read_only(&call));
        }
    }

    #[test]
    fn fallback_tool_call_does_not_install_therock_without_folder() {
        let call = fallback_rocm_tool_call_for_prompt(
            "Install this specific TheRock wheel from date 06052026",
        )
        .unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--build-date".to_owned(),
                "2026-06-05".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn chat_how_to_setup_question_opens_install_folder_flow() {
        assert!(
            install_sdk_without_prefix_chat_approval("How do I setup TheRock?").is_some(),
            "a setup question should ask for the install folder"
        );
        assert!(
            install_sdk_without_prefix_chat_approval("install therock").is_some(),
            "a direct install command should open the folder picker"
        );
    }

    #[test]
    fn chat_install_intent_preserves_bare_folder_path() {
        let approval =
            install_sdk_chat_approval_for_prompt("install therock D:\\ROCm\\therock_venvs")
                .expect("direct install prompt should be recognized");

        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
            ]
        );
    }

    #[test]
    fn fallback_tool_call_routes_requested_therock_exact_version_install() {
        let call = fallback_rocm_tool_call_for_prompt(
            "Install the TheRock ROCm wheel version 7.13.0a20260605 into D:\\ROCm\\therock_venvs",
        )
        .unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
                "--version".to_owned(),
                "7.13.0a20260605".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn fallback_tool_call_routes_simple_config_changes() {
        let show = fallback_rocm_tool_call_for_prompt("Show current ROCm CLI config").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&show).unwrap(),
            vec!["config".to_owned(), "show".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&show));

        let engine = fallback_rocm_tool_call_for_prompt("Set the default engine to vllm").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&engine).unwrap(),
            vec![
                "config".to_owned(),
                "set-default-engine".to_owned(),
                "vllm".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&engine));

        let telemetry =
            fallback_rocm_tool_call_for_prompt("Disable telemetry in settings").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&telemetry).unwrap(),
            vec![
                "config".to_owned(),
                "set-telemetry".to_owned(),
                "off".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&telemetry));
    }

    #[test]
    fn fallback_tool_call_routes_comfyui_support_and_actions() {
        let status = fallback_rocm_tool_call_for_prompt("How do I setup ComfyUI?").unwrap();
        assert_eq!(status.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&status).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&status));

        let install = fallback_rocm_tool_call_for_prompt("Can you setup ComfyUI for me?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&install).unwrap(),
            vec!["comfyui".to_owned(), "install".to_owned()]
        );
        assert!(!chat_tool_call_is_read_only(&install));
        let approval =
            chat_tool_approval_request(&install, Some("Install ComfyUI after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Install ComfyUI");

        let start = fallback_rocm_tool_call_for_prompt("Can you start ComfyUI?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&start).unwrap(),
            vec!["comfyui".to_owned(), "start".to_owned()]
        );
        assert!(!chat_tool_call_is_read_only(&start));
        let approval =
            chat_tool_approval_request(&start, Some("Start ComfyUI after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Start ComfyUI");
    }

    #[test]
    fn fallback_tool_call_routes_local_llm_serve_requests() {
        let call =
            fallback_rocm_tool_call_for_prompt("Can you setup and serve an LLM for me?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "serve".to_owned(),
                "qwen".to_owned(),
                "--engine".to_owned(),
                "lemonade".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
        let approval =
            chat_tool_approval_request(&call, Some("Start qwen after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Start local model server");
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some("rocm serve qwen --engine lemonade --device gpu_required --managed")
        );
    }
}
