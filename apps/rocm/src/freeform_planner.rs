// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The freeform natural-language planner: turns a prose request (e.g. `rocm
//! "serve qwen3.5 with vllm"`) into a [`StructuredRequestPlan`] of concrete
//! `rocm` tool calls.
//!
//! Two paths build a plan, tried in order:
//!
//! - **Deterministic.** [`build_freeform_plan_with_recipes`] pattern-matches the
//!   request against known phrasing (serve/install/inspect/comfyui/etc.) and the
//!   model-recipe registry. This covers the common cases without ever leaving
//!   the machine.
//! - **Provider-assisted.** When the deterministic pass can't resolve enough of
//!   the request and a cloud provider is configured
//!   ([`configured_planner_provider`]), [`resolve_freeform_plan_with_provider`]
//!   sends a structured prompt ([`build_provider_planner_prompt`]) and parses
//!   the provider's tool-call response
//!   ([`provider_planner_response_to_plan`]), validating it the same way a
//!   deterministic plan is validated — it can never bypass the GPU-required or
//!   local-bind-only rules the deterministic path enforces.
//!
//! [`render_freeform_plan`]/[`render_structured_request_plan`] render a plan for
//! the no-`--yes` review path; [`freeform_plan_next_action_with_context`]
//! reduces a plan to the single next [`FreeformPlanAction`] the dispatch flow
//! executes. Everything downstream of "is this request freeform at all"
//! (`parse_freeform_invocation`/`should_treat_as_freeform`) and "is this action
//! safe to execute" (`validate_freeform_execution_action`) stays in `main.rs` —
//! this module only builds and renders the plan.

use anyhow::{Context, Result, bail};
use clap::Parser as _;
use rocm_core::{
    AppPaths, ModelRecipeRecord, RocmCliConfig, load_model_recipe_registry,
    resolve_builtin_model_recipe,
};
use serde::Deserialize;
use std::fmt::Write as _;

use crate::service_launch::is_loopback_host;
use crate::{Cli, chat, providers};

pub(crate) fn render_freeform_plan(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> String {
    let plan = build_freeform_plan_with_context(request, paths, config);
    render_structured_request_plan(&plan, paths)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FreeformPlanAction {
    pub title: String,
    pub args: Vec<String>,
    pub approval_required: bool,
    pub reason: String,
    pub has_placeholders: bool,
    pub provider_assisted: bool,
}

#[cfg(test)]
pub(crate) fn freeform_plan_next_action(
    request: &str,
    config: &RocmCliConfig,
) -> Option<FreeformPlanAction> {
    let plan = build_freeform_plan(request, config);
    plan_next_action(plan)
}

pub(crate) fn freeform_plan_next_action_with_context(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Option<FreeformPlanAction> {
    let plan = build_freeform_plan_with_context(request, paths, config);
    plan_next_action(plan)
}

pub(crate) fn plan_next_action(plan: StructuredRequestPlan) -> Option<FreeformPlanAction> {
    plan.actions.last().map(|action| FreeformPlanAction {
        title: action.title.to_owned(),
        args: action.args.clone(),
        approval_required: action.approval == "required",
        reason: action.reason.to_owned(),
        has_placeholders: action
            .args
            .iter()
            .any(|arg| arg.starts_with('<') && arg.ends_with('>')),
        provider_assisted: plan.provider_assisted,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StructuredRequestPlan {
    request: String,
    planner: String,
    provider_assisted: bool,
    intent: PlannerIntent,
    confidence: &'static str,
    approval: &'static str,
    parsed: Vec<(String, String)>,
    pub(crate) actions: Vec<PlannedToolCall>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlannerIntent {
    Ask,
    Serve,
    InstallSdk,
    InstallDriver,
    Update,
    Uninstall,
    Inspect,
}

impl PlannerIntent {
    const fn label(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Serve => "serve",
            Self::InstallSdk => "install sdk",
            Self::InstallDriver => "install driver",
            Self::Update => "update check",
            Self::Uninstall => "uninstall",
            Self::Inspect => "inspect",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedToolCall {
    title: &'static str,
    tool: &'static str,
    pub(crate) args: Vec<String>,
    pub(crate) approval: &'static str,
    reason: &'static str,
}

impl PlannedToolCall {
    const fn read_only(title: &'static str, args: Vec<String>, reason: &'static str) -> Self {
        Self {
            title,
            tool: "rocm",
            args,
            approval: "not required",
            reason,
        }
    }

    const fn approval_required(
        title: &'static str,
        args: Vec<String>,
        reason: &'static str,
    ) -> Self {
        Self {
            title,
            tool: "rocm",
            args,
            approval: "required",
            reason,
        }
    }
}

#[cfg(test)]
fn build_freeform_plan(request: &str, config: &RocmCliConfig) -> StructuredRequestPlan {
    build_freeform_plan_with_recipes(request, config, None, host_default_engine(None))
}

/// The engine this host serves on, for callers that need it as an owned default.
///
/// Kept separate from the planner so the planner stays a pure function of its
/// inputs: it bakes the engine into a `rocm serve --engine <engine>` command, and
/// an explicit `--engine` outranks every other signal in [`select_serve_engine`],
/// so a probe hidden inside it would make that command silently host-dependent
/// and its tests unreproducible.
pub(crate) fn host_default_engine(paths: Option<&AppPaths>) -> &'static str {
    rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(paths))
}

/// `host_default_engine` is the engine this GPU serves on, resolved by the
/// caller. It is the last resort: an engine named in the request wins, then the
/// configured default, then the matched recipe's preference.
///
/// It must NOT be [`default_engine_for_platform`]. The value chosen here is
/// written into a literal `--engine` argument, which outranks even the
/// configured default when the generated command runs — so a GPU-blind constant
/// here forces Lemonade onto an Instinct host through the strongest override
/// available, which is the defect this whole path is meant to avoid.
fn build_freeform_plan_with_recipes(
    request: &str,
    config: &RocmCliConfig,
    recipes: Option<&[ModelRecipeRecord]>,
    host_default_engine: &str,
) -> StructuredRequestPlan {
    let trimmed = request.trim();
    let lower = trimmed.to_ascii_lowercase();
    let default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(host_default_engine);

    if planner_is_serve_request(&lower) {
        let requested_model = infer_model_from_request(trimmed)
            .filter(|model| !generic_model_phrase(model))
            .or_else(|| infer_small_local_model_from_request(&lower))
            .or_else(|| infer_recommended_assistant_model_from_request(&lower));
        let resolved_recipe =
            requested_model.and_then(|model| planner_resolve_model_recipe(model, recipes));
        let model = resolved_recipe
            .as_ref()
            .map(|recipe| recipe.canonical_model_id.as_str())
            .or(requested_model)
            .unwrap_or("<model>");
        let engine = infer_engine_from_request(&lower)
            .or_else(|| {
                resolved_recipe
                    .as_ref()
                    .and_then(|recipe| recipe.preferred_engines.first().map(String::as_str))
            })
            .unwrap_or(default_engine);
        let device_policy = infer_device_policy_from_request(&lower).or_else(|| {
            resolved_recipe
                .as_ref()
                .map(|recipe| recipe.device_policy.as_str())
        });
        if device_policy.is_some_and(device_policy_is_cpu_mode) {
            let mut parsed = vec![
                ("model".to_owned(), model.to_owned()),
                ("engine".to_owned(), engine.to_owned()),
                ("device_policy".to_owned(), "cpu_not_supported".to_owned()),
                ("mode".to_owned(), "managed".to_owned()),
            ];
            if model == "<model>" {
                parsed.push(("missing".to_owned(), "model".to_owned()));
            }
            if let Some(requested_model) = requested_model
                && requested_model != model
            {
                parsed.push(("model_alias".to_owned(), requested_model.to_owned()));
            }
            if let Some(recipe) = resolved_recipe.as_ref() {
                parsed.push(("recipe_source".to_owned(), recipe.source.clone()));
                parsed.push(("recipe_dtype".to_owned(), recipe.dtype.clone()));
            }
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Serve,
                confidence: if model == "<model>" { "medium" } else { "high" },
                approval: "not available; ROCm CLI requires ROCm GPU execution",
                parsed,
                actions: Vec::new(),
                notes: vec![
                    "CPU mode is not offered by rocm-cli.".to_owned(),
                    "Use a ROCm-capable AMD GPU and request GPU execution.".to_owned(),
                ],
            };
        }
        let device_policy = device_policy.map(planned_device_policy_without_fallback);
        let mut args = vec![
            "serve".to_owned(),
            model.to_owned(),
            "--engine".to_owned(),
            engine.to_owned(),
        ];
        if let Some(device_policy) = device_policy {
            args.push("--device".to_owned());
            args.push(device_policy.to_owned());
        }
        args.push("--managed".to_owned());
        let mut parsed = vec![
            ("model".to_owned(), model.to_owned()),
            ("engine".to_owned(), engine.to_owned()),
            (
                "device_policy".to_owned(),
                device_policy.unwrap_or("gpu_required").to_owned(),
            ),
            ("mode".to_owned(), "managed".to_owned()),
        ];
        if model == "<model>" {
            parsed.push(("missing".to_owned(), "model".to_owned()));
        }
        if let Some(requested_model) = requested_model
            && requested_model != model
        {
            parsed.push(("model_alias".to_owned(), requested_model.to_owned()));
        }
        if let Some(recipe) = resolved_recipe.as_ref() {
            parsed.push(("recipe_source".to_owned(), recipe.source.clone()));
            parsed.push(("recipe_dtype".to_owned(), recipe.dtype.clone()));
        }
        let mut notes = vec![
            "Final execution must come from the structured tool call shown above.".to_owned(),
            "GPU-preferred recipe policies are treated as GPU required; no CPU fallback is implied."
                .to_owned(),
        ];
        if let Some(recipe) = resolved_recipe.as_ref() {
            notes.extend(recipe.warnings.iter().cloned());
        }
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Serve,
            confidence: if model == "<model>" { "medium" } else { "high" },
            approval: "required before launch; plan rendering is read-only",
            parsed,
            actions: vec![
                PlannedToolCall::read_only(
                    "Inspect host/runtime state",
                    vec!["examine".to_owned()],
                    "read-only inspection",
                ),
                PlannedToolCall::read_only(
                    "Show local model engines",
                    vec!["engines".to_owned(), "list".to_owned()],
                    "read-only engine list",
                ),
                PlannedToolCall::approval_required(
                    "Launch local endpoint",
                    args,
                    "starts or changes a local serving process",
                ),
            ],
            notes,
        };
    }

    if lower.contains("driver") {
        let mut args = vec!["install".to_owned(), "driver".to_owned()];
        if lower.contains("dkms") {
            args.push("--dkms".to_owned());
        }
        args.push("--yes".to_owned());
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::InstallDriver,
            confidence: "high",
            approval: "required before driver changes",
            parsed: vec![(
                "driver_flow".to_owned(),
                if lower.contains("dkms") {
                    "dkms"
                } else {
                    "platform default"
                }
                .to_owned(),
            )],
            actions: vec![
                PlannedToolCall::read_only(
                    "Inspect host/driver state",
                    vec!["examine".to_owned()],
                    "read-only inspection",
                ),
                PlannedToolCall::approval_required(
                    "Install driver",
                    args,
                    "driver changes are privileged or disruptive",
                ),
            ],
            notes: vec!["Driver changes are always explicit and never silent.".to_owned()],
        };
    }

    if planner_mentions_comfyui(&lower) {
        if chat::any_substring(&lower, &["log", "logs"]) {
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Inspect,
                confidence: "high",
                approval: "not required for inspection",
                parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
                actions: vec![PlannedToolCall::read_only(
                    "Read ComfyUI logs",
                    vec!["comfyui".to_owned(), "logs".to_owned()],
                    "read-only app log check",
                )],
                notes: Vec::new(),
            };
        }
        if chat::any_substring(&lower, &["start", "run", "launch", "open"]) {
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Serve,
                confidence: "high",
                approval: "required before launch",
                parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
                actions: vec![PlannedToolCall::approval_required(
                    "Start ComfyUI",
                    vec!["comfyui".to_owned(), "start".to_owned()],
                    "starts a local ComfyUI process",
                )],
                notes: Vec::new(),
            };
        }
        if planner_requests_comfyui_install(&lower) {
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::InstallSdk,
                confidence: "high",
                approval: "required before installing ComfyUI",
                parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
                actions: vec![PlannedToolCall::approval_required(
                    "Install ComfyUI",
                    vec!["comfyui".to_owned(), "install".to_owned()],
                    "installs ComfyUI into ROCm CLI's managed app folder",
                )],
                notes: vec!["ComfyUI uses the active ROCm CLI managed TheRock runtime.".to_owned()],
            };
        }
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Inspect,
            confidence: "high",
            approval: "not required for inspection",
            parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
            actions: vec![PlannedToolCall::read_only(
                "Check ComfyUI",
                vec!["comfyui".to_owned(), "status".to_owned()],
                "read-only app status check",
            )],
            notes: Vec::new(),
        };
    }

    if planner_is_install_sdk_request(&lower) {
        let channel = if lower.contains("nightly") {
            "nightly"
        } else {
            "release"
        };
        let build_date = chat::requested_therock_build_date_from_prompt(&lower);
        let Some(prefix) = chat::requested_install_prefix_from_prompt(trimmed) else {
            let mut parsed = vec![
                ("channel".to_owned(), channel.to_owned()),
                ("format".to_owned(), "wheel".to_owned()),
            ];
            if let Some(build_date) = build_date.as_deref() {
                parsed.push(("build_date".to_owned(), build_date.to_owned()));
            }
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Ask,
                confidence: "medium",
                approval: "not applicable until an install folder is chosen",
                parsed,
                actions: Vec::new(),
                notes: vec![
                    "Choose a ROCm/TheRock install folder before approving an install.".to_owned(),
                    "Say something like: install TheRock into D:\\ROCm\\therock_venvs.".to_owned(),
                ],
            };
        };
        let mut args = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--channel".to_owned(),
            channel.to_owned(),
            "--format".to_owned(),
            "wheel".to_owned(),
            "--prefix".to_owned(),
            prefix.clone(),
        ];
        if let Some(build_date) = build_date.as_deref() {
            args.push("--build-date".to_owned());
            args.push(build_date.to_owned());
        }
        let mut parsed = vec![
            ("channel".to_owned(), channel.to_owned()),
            ("format".to_owned(), "wheel".to_owned()),
            ("prefix".to_owned(), prefix),
        ];
        if let Some(build_date) = build_date.as_deref() {
            parsed.push(("build_date".to_owned(), build_date.to_owned()));
        }
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::InstallSdk,
            confidence: "high",
            approval: "required before installing or switching runtimes",
            parsed,
            actions: vec![
                PlannedToolCall::read_only(
                    "Inspect current runtime",
                    vec!["examine".to_owned()],
                    "read-only inspection",
                ),
                PlannedToolCall::approval_required(
                    "Install TheRock SDK",
                    args,
                    "changes managed runtime state",
                ),
            ],
            notes: vec!["TheRock pip venv install is the default managed runtime path.".to_owned()],
        };
    }

    if lower.contains("update") {
        let apply = lower.contains("apply") || lower.contains("upgrade");
        let args = if apply {
            vec!["update".to_owned(), "--apply".to_owned()]
        } else {
            vec!["update".to_owned()]
        };
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Update,
            confidence: "high",
            approval: if apply {
                "required before applying updates"
            } else {
                "not required for update inspection"
            },
            parsed: vec![(
                "mode".to_owned(),
                if apply { "apply" } else { "check" }.to_owned(),
            )],
            actions: vec![if apply {
                PlannedToolCall::approval_required(
                    "Apply update",
                    args,
                    "installs or switches managed runtime state",
                )
            } else {
                PlannedToolCall::read_only("Check updates", args, "read-only update inspection")
            }],
            notes: vec!["Update checks compare against the selected runtime channel.".to_owned()],
        };
    }

    if lower.contains("uninstall") || lower.contains("remove rocm") {
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Uninstall,
            confidence: "high",
            approval: "required before deleting installed files",
            parsed: Vec::new(),
            actions: vec![
                PlannedToolCall::read_only(
                    "Preview uninstall",
                    vec!["uninstall".to_owned(), "--dry-run".to_owned()],
                    "dry-run planning",
                ),
                PlannedToolCall::approval_required(
                    "Apply uninstall",
                    vec!["uninstall".to_owned(), "--yes".to_owned()],
                    "deletes installed files and state",
                ),
            ],
            notes: Vec::new(),
        };
    }

    if planner_is_inspect_request(&lower) {
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Inspect,
            confidence: "high",
            approval: "not required for inspection",
            parsed: vec![("engine".to_owned(), default_engine.to_owned())],
            actions: vec![PlannedToolCall::read_only(
                "Inspect local ROCm state",
                vec!["examine".to_owned()],
                "read-only inspection",
            )],
            notes: vec![
                "Use /chat <prompt> in the TUI for provider-backed answers.".to_owned(),
                "Use /plan with an explicit install, update, serve, or uninstall request for action planning.".to_owned(),
            ],
        };
    }

    StructuredRequestPlan {
        request: trimmed.to_owned(),
        planner: "hybrid-parser-v1".to_owned(),
        provider_assisted: false,
        intent: PlannerIntent::Ask,
        confidence: "low",
        approval: "not applicable",
        parsed: Vec::new(),
        actions: Vec::new(),
        notes: vec![
            "No ROCm action matched this request.".to_owned(),
            "Run `rocm --help` to see available commands, or rephrase to include an action such as install, update, serve, uninstall, check, or inspect.".to_owned(),
        ],
    }
}

fn planned_device_policy_without_fallback(policy: &str) -> &str {
    match policy {
        "gpu_preferred" => "gpu_required",
        other => other,
    }
}

fn device_policy_is_cpu_mode(policy: &str) -> bool {
    matches!(
        policy.trim().to_ascii_lowercase().as_str(),
        "cpu" | "cpu_only"
    )
}

pub(crate) fn serve_args_request_cpu_device(args: &[String]) -> bool {
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--device=")
            && device_policy_is_cpu_mode(value)
        {
            return true;
        }
        if arg == "--device"
            && let Some(value) = iter.peek()
            && device_policy_is_cpu_mode(value)
        {
            return true;
        }
    }
    false
}

fn planner_resolve_model_recipe(
    model_ref: &str,
    recipes: Option<&[ModelRecipeRecord]>,
) -> Option<ModelRecipeRecord> {
    if let Some(recipes) = recipes {
        return recipes
            .iter()
            .find(|recipe| recipe.matches_ref(model_ref))
            .cloned();
    }
    resolve_builtin_model_recipe(model_ref)
}

pub(crate) fn build_freeform_plan_with_context(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> StructuredRequestPlan {
    // Resolved once here, where `paths` is in scope, so the generated
    // `rocm serve --engine <engine>` names the engine this GPU actually serves on.
    let host_engine = host_default_engine(Some(paths));
    let registry = match load_model_recipe_registry() {
        Ok(registry) => Some(registry),
        Err(error) => {
            let mut plan =
                build_freeform_plan_with_recipes(request, config, Some(&[]), host_engine);
            plan.confidence = "medium";
            plan.notes.push(format!(
                "Model recipe registry could not be loaded: {error}. Fix the recipe index before using registry aliases."
            ));
            return plan;
        }
    };
    let mut deterministic = build_freeform_plan_with_recipes(
        request,
        config,
        registry
            .as_ref()
            .map(|registry| registry.recipes.as_slice()),
        host_engine,
    );
    if !freeform_plan_needs_ambiguity_resolution(&deterministic) {
        return deterministic;
    }
    let Some(provider) = configured_planner_provider(config) else {
        return deterministic;
    };

    match resolve_freeform_plan_with_provider(request, paths, config, provider, &deterministic) {
        Ok(plan) => plan,
        Err(error) => {
            deterministic.notes.push(format!(
                "Provider-assisted planner `{provider}` could not resolve this request: {error}"
            ));
            deterministic.notes.push(
                "No provider-produced tool call was used. Fill the placeholder values or run /chat for help."
                    .to_owned(),
            );
            deterministic
        }
    }
}

fn configured_planner_provider(config: &RocmCliConfig) -> Option<&str> {
    config
        .planner_provider
        .as_deref()
        .map(str::trim)
        .filter(|provider| !provider.is_empty())
        .filter(|provider| matches!(*provider, "local" | "openai" | "anthropic"))
}

fn freeform_plan_needs_ambiguity_resolution(plan: &StructuredRequestPlan) -> bool {
    plan.confidence != "high"
        || plan.actions.iter().any(|action| {
            action
                .args
                .iter()
                .any(|arg| arg.starts_with('<') && arg.ends_with('>'))
        })
}

fn resolve_freeform_plan_with_provider(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
    provider: &str,
    deterministic: &StructuredRequestPlan,
) -> Result<StructuredRequestPlan> {
    if provider != "local" && !config.provider_enabled(provider) {
        bail!(
            "cloud provider is disabled; run `rocm config enable-provider {provider}` before sending planner prompts"
        );
    }
    let prompt = build_provider_planner_prompt(request, deterministic);
    let response = providers::provider_chat(
        paths,
        provider,
        &providers::ChatRequest {
            model: None,
            messages: vec![providers::ChatMessage {
                role: "user".to_owned(),
                content: prompt,
            }],
            max_tokens: Some(512),
            temperature: None,
            top_p: None,
            rocm_tools: false,
        },
    )?;
    provider_planner_response_to_plan(request, provider, &response.content)
}

fn build_provider_planner_prompt(request: &str, deterministic: &StructuredRequestPlan) -> String {
    let next_tool_call = deterministic.actions.last().map_or_else(
        || "rocm examine".to_owned(),
        |action| format_structured_tool_call(action.tool, &action.args),
    );
    format!(
        "You are resolving an ambiguous rocm-cli request. Return only JSON with this shape: \
{{\"intent\":\"serve|install_sdk|install_driver|update|uninstall|inspect\",\
\"confidence\":\"high|medium|low\",\
\"tool_call\":{{\"tool\":\"rocm\",\"args\":[\"...\"]}},\
\"notes\":[\"short note\"]}}.\n\
Allowed rocm actions: examine; engines list; install sdk; install driver; update; serve; uninstall. Install sdk must include --prefix PATH chosen by the user, and may include --build-date YYYY-MM-DD or --version VERSION.\n\
Do not invent CPU fallback. Do not include shell commands. Do not include markdown.\n\
User request: {request}\n\
Deterministic planner intent: {}\n\
Deterministic next tool call: {next_tool_call}",
        deterministic.intent.label()
    )
}

#[derive(Debug, Deserialize)]
struct ProviderPlannerResponse {
    intent: String,
    #[serde(default)]
    confidence: Option<String>,
    tool_call: ProviderPlannerToolCall,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProviderPlannerToolCall {
    tool: String,
    args: Vec<String>,
}

pub(crate) fn provider_planner_response_to_plan(
    request: &str,
    provider: &str,
    content: &str,
) -> Result<StructuredRequestPlan> {
    let response = parse_provider_planner_response(content)?;
    let args = validate_provider_planner_tool_call(&response.tool_call)?;
    let intent = planner_intent_from_provider_response(&response.intent, &args)?;
    let approval_required = provider_planner_args_require_approval(&args);
    let action = if approval_required {
        PlannedToolCall::approval_required(
            "Provider-resolved action",
            args,
            "provider-assisted ambiguity resolution; review before running",
        )
    } else {
        PlannedToolCall::read_only(
            "Provider-resolved action",
            args,
            "provider-assisted read-only plan",
        )
    };
    let mut notes = vec![
        "Provider-assisted planning is optional and only uses the configured planner provider."
            .to_owned(),
        "The provider output was reduced to a validated rocm tool call; no shell command is executed."
            .to_owned(),
    ];
    notes.extend(
        response
            .notes
            .into_iter()
            .filter(|note| !note.trim().is_empty()),
    );
    Ok(StructuredRequestPlan {
        request: request.trim().to_owned(),
        planner: format!("hybrid-parser-v1 + provider:{provider}"),
        provider_assisted: true,
        intent,
        confidence: sanitize_provider_confidence(response.confidence.as_deref()),
        approval: if approval_required {
            "required before execution"
        } else {
            "not required for inspection"
        },
        parsed: vec![("provider".to_owned(), provider.to_owned())],
        actions: vec![action],
        notes,
    })
}

fn parse_provider_planner_response(content: &str) -> Result<ProviderPlannerResponse> {
    let json = strip_json_fence(content.trim());
    serde_json::from_str::<ProviderPlannerResponse>(json)
        .context("provider planner response was not valid JSON")
}

fn strip_json_fence(content: &str) -> &str {
    let trimmed = content.trim();
    if let Some(rest) = trimmed.strip_prefix("```json")
        && let Some((json, _)) = rest.trim_start().split_once("```")
    {
        return json.trim();
    }
    if let Some(rest) = trimmed.strip_prefix("```")
        && let Some((json, _)) = rest.trim_start().split_once("```")
    {
        return json.trim();
    }
    trimmed
}

fn validate_provider_planner_tool_call(call: &ProviderPlannerToolCall) -> Result<Vec<String>> {
    if call.tool != "rocm" {
        bail!("provider planner returned unsupported tool `{}`", call.tool);
    }
    if call.args.is_empty() {
        bail!("provider planner returned an empty tool call");
    }
    if call
        .args
        .iter()
        .any(|arg| arg.trim().is_empty() || arg.contains('\0'))
    {
        bail!("provider planner returned an invalid empty or NUL argument");
    }
    let argv = std::iter::once("rocm".to_owned())
        .chain(call.args.iter().cloned())
        .collect::<Vec<_>>();
    Cli::try_parse_from(argv)
        .context("provider planner returned a rocm command that is not valid")?;

    match call.args.as_slice() {
        [command] if command == "examine" => {}
        [command, subcommand] if command == "engines" && subcommand == "list" => {}
        [command, subcommand, ..] if command == "install" && subcommand == "sdk" => {
            chat::validate_chat_rocm_command_safety(&call.args)?;
        }
        [command, subcommand, ..] if command == "install" && subcommand == "driver" => {}
        [command, ..] if command == "update" => {}
        [command, ..] if command == "serve" => {
            if chat::chat_cli_has_flag(&call.args, "--allow-public-bind") {
                bail!("provider planner cannot request public network binding");
            }
            if serve_args_request_cpu_device(&call.args) {
                bail!(
                    "provider planner cannot request CPU execution; rocm-cli requires ROCm GPU execution"
                );
            }
            if let Some(host) = chat::chat_cli_arg_value(&call.args, "--host")
                && !is_loopback_host(host)
            {
                bail!("provider planner cannot request non-local host `{host}`");
            }
            if chat::chat_cli_has_flag(&call.args, "--foreground")
                || !chat::chat_cli_has_flag(&call.args, "--managed")
            {
                bail!("provider planner must request managed serving with --managed");
            }
        }
        [command, ..] if command == "uninstall" => {}
        _ => bail!(
            "provider planner returned an unsupported rocm action: {}",
            format_structured_tool_call("rocm", &call.args)
        ),
    }
    Ok(call.args.clone())
}

fn planner_intent_from_provider_response(intent: &str, args: &[String]) -> Result<PlannerIntent> {
    let args_intent = planner_intent_from_args(args)?;
    let declared = match intent.trim().to_ascii_lowercase().as_str() {
        "serve" => PlannerIntent::Serve,
        "install_sdk" | "install sdk" => PlannerIntent::InstallSdk,
        "install_driver" | "install driver" => PlannerIntent::InstallDriver,
        "update" => PlannerIntent::Update,
        "uninstall" => PlannerIntent::Uninstall,
        "inspect" | "examine" => PlannerIntent::Inspect,
        _ => bail!("provider planner returned unsupported intent `{intent}`"),
    };
    if declared != args_intent {
        bail!(
            "provider planner intent `{}` did not match tool call intent `{}`",
            declared.label(),
            args_intent.label()
        );
    }
    Ok(args_intent)
}

fn planner_intent_from_args(args: &[String]) -> Result<PlannerIntent> {
    match args.first().map(String::as_str) {
        Some("serve") => Ok(PlannerIntent::Serve),
        Some("install") if args.get(1).is_some_and(|arg| arg == "sdk") => {
            Ok(PlannerIntent::InstallSdk)
        }
        Some("install") if args.get(1).is_some_and(|arg| arg == "driver") => {
            Ok(PlannerIntent::InstallDriver)
        }
        Some("update") => Ok(PlannerIntent::Update),
        Some("uninstall") => Ok(PlannerIntent::Uninstall),
        Some("examine" | "engines") => Ok(PlannerIntent::Inspect),
        _ => bail!("unsupported provider planner args"),
    }
}

fn provider_planner_args_require_approval(args: &[String]) -> bool {
    matches!(
        args.first().map(String::as_str),
        Some("install" | "serve" | "uninstall")
    ) || args.iter().any(|arg| arg == "--apply")
}

fn sanitize_provider_confidence(confidence: Option<&str>) -> &'static str {
    match confidence
        .unwrap_or("medium")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "high" => "high",
        "low" => "low",
        _ => "medium",
    }
}

pub(crate) fn render_structured_request_plan(
    plan: &StructuredRequestPlan,
    paths: &AppPaths,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "request plan");
    let _ = writeln!(output, "  request: {}", plan.request);
    let _ = writeln!(output, "  planner: {}", plan.planner);
    let _ = writeln!(output, "  tool_schema: {}", providers::ROCM_TOOL_SCHEMA_ID);
    let _ = writeln!(output, "  intent: {}", plan.intent.label());
    let _ = writeln!(output, "  confidence: {}", plan.confidence);
    let _ = writeln!(output, "  approval: {}", plan.approval);
    if !plan.parsed.is_empty() {
        let _ = writeln!(output, "  parsed:");
        for (key, value) in &plan.parsed {
            let _ = writeln!(output, "    {key}: {value}");
        }
    }
    let _ = writeln!(output, "  plan:");
    if plan.actions.is_empty() {
        let _ = writeln!(output, "    No ROCm action selected.");
    } else {
        for (index, action) in plan.actions.iter().enumerate() {
            let _ = writeln!(output, "    {}. {}", index + 1, action.title);
            let _ = writeln!(
                output,
                "       tool_call: {}",
                format_structured_tool_call(action.tool, &action.args)
            );
            let _ = writeln!(output, "       approval: {}", action.approval);
            let _ = writeln!(output, "       reason: {}", action.reason);
        }
    }
    if let Some(next) = plan.actions.last() {
        let _ = writeln!(
            output,
            "  next_tool_call: {}",
            format_structured_tool_call(next.tool, &next.args)
        );
        let _ = writeln!(output, "  next_tool_approval: {}", next.approval);
    }
    if plan.intent == PlannerIntent::Uninstall {
        let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    }
    for note in &plan.notes {
        let _ = writeln!(output, "  note: {note}");
    }
    output
}

fn planner_is_serve_request(lower: &str) -> bool {
    lower.contains("serve")
        || lower.contains("run a small local model")
        || lower.contains("local model on cpu")
        || lower.contains("start a local model")
}

fn planner_is_install_sdk_request(lower: &str) -> bool {
    let installish = contains_planner_word(lower, "install")
        || contains_planner_word(lower, "setup")
        || lower.contains("set up");
    let target = lower.contains("therock") || contains_planner_word(lower, "sdk");
    installish && target
}

fn planner_is_inspect_request(lower: &str) -> bool {
    let inspectish = contains_planner_word(lower, "inspect")
        || contains_planner_word(lower, "check")
        || contains_planner_word(lower, "status")
        || contains_planner_word(lower, "examine")
        || contains_planner_word(lower, "which")
        || contains_planner_word(lower, "where")
        || lower.contains("what is installed")
        || lower.contains("what's installed")
        || lower.contains("is installed")
        || lower.contains("is rocm installed")
        || lower.contains("is therock installed")
        || lower.contains("is the rock installed");
    let target = lower.contains("rocm")
        || lower.contains("therock")
        || lower.contains("the rock")
        || lower.contains("gpu")
        || lower.contains("driver")
        || lower.contains("setup")
        || lower.contains("installed")
        || lower.contains("this computer")
        || lower.contains("this machine");
    inspectish && target
}

fn planner_mentions_comfyui(lower: &str) -> bool {
    chat::any_substring(lower, &["comfyui", "comfy ui", "comfy"])
}

fn planner_requests_comfyui_install(lower: &str) -> bool {
    chat::any_substring(
        lower,
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
    )
}

fn contains_planner_word(text: &str, expected: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '.')
        .any(|word| word == expected)
}

fn infer_engine_from_request(lower: &str) -> Option<&'static str> {
    ["lemonade", "vllm"]
        .into_iter()
        .find(|engine| lower.contains(*engine))
}

fn infer_model_from_request(request: &str) -> Option<&str> {
    let trimmed = request.trim();
    let lower = trimmed.to_ascii_lowercase();
    let serve_index = lower.find("serve")?;
    let after = trimmed.get(serve_index + "serve".len()..)?.trim();
    if after.is_empty() {
        return None;
    }
    let end = after
        .find(" with ")
        .or_else(|| after.find(" using "))
        .unwrap_or(after.len());
    let model = after[..end].trim();
    (!model.is_empty()).then_some(model)
}

fn infer_small_local_model_from_request(lower: &str) -> Option<&'static str> {
    (lower.contains("small local model") || lower.contains("tiny local model"))
        .then_some("sshleifer/tiny-gpt2")
}

fn infer_recommended_assistant_model_from_request(lower: &str) -> Option<&'static str> {
    (lower.contains("start a local model")
        || lower.contains("local assistant")
        || lower.contains("recommended model")
        || lower.contains("serve an llm")
        || lower.contains("serve a local llm"))
    .then_some(providers::LEMONADE_ASSISTANT_MODEL_ID)
}

fn generic_model_phrase(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().as_str(),
        "a model" | "an llm" | "a local llm" | "local assistant" | "the assistant"
    )
}

fn infer_device_policy_from_request(lower: &str) -> Option<&'static str> {
    if lower.contains("cpu") {
        Some("cpu")
    } else if lower.contains("gpu preferred") || lower.contains("prefer gpu") {
        Some("gpu_preferred")
    } else if lower.contains("gpu") {
        Some("gpu")
    } else {
        None
    }
}

pub(crate) fn format_structured_tool_call(tool: &str, args: &[String]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(tool.to_owned());
    parts.extend(args.iter().map(|arg| quote_tool_arg(arg)));
    parts.join(" ")
}

fn quote_tool_arg(value: &str) -> String {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        value.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn hybrid_planner_normalizes_model_alias_and_structured_serve_call() {
        let plan = build_freeform_plan("serve qwen3.5 with vllm", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert!(
            plan.parsed
                .contains(&("model".to_owned(), "Qwen/Qwen3.5-4B".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("model_alias".to_owned(), "qwen3.5".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("engine".to_owned(), "vllm".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("mode".to_owned(), "managed".to_owned()))
        );
        assert!(plan.actions.iter().any(|action| {
            action.approval == "required"
                && action.args
                    == vec![
                        "serve".to_owned(),
                        "Qwen/Qwen3.5-4B".to_owned(),
                        "--engine".to_owned(),
                        "vllm".to_owned(),
                        "--device".to_owned(),
                        "gpu_required".to_owned(),
                        "--managed".to_owned(),
                    ]
        }));
    }

    #[test]
    fn hybrid_planner_can_use_active_recipe_registry_aliases() {
        let mut recipe = resolve_builtin_model_recipe("tiny-gpt2").expect("tiny recipe");
        recipe.canonical_model_id = "Acme/SignedTiny".to_owned();
        recipe.aliases = vec!["signedtiny".to_owned()];
        recipe.source = "signed_recipe_index".to_owned();
        recipe.preferred_engines = vec!["vllm".to_owned()];
        recipe.device_policy = "cpu_only".to_owned();
        recipe.dtype = "float16".to_owned();

        let plan = build_freeform_plan_with_recipes(
            "serve signedtiny",
            &RocmCliConfig::default(),
            Some(&[recipe]),
            "lemonade",
        );

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert!(
            plan.parsed
                .contains(&("model".to_owned(), "Acme/SignedTiny".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("model_alias".to_owned(), "signedtiny".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("recipe_source".to_owned(), "signed_recipe_index".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("recipe_dtype".to_owned(), "float16".to_owned()))
        );
        assert!(plan.actions.is_empty());
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("CPU mode is not offered"))
        );
    }

    #[test]
    fn hybrid_planner_builds_nightly_therock_install_call() {
        let plan = build_freeform_plan(
            "install the latest TheRock nightly for this GPU into D:\\ROCm\\therock_venvs",
            &RocmCliConfig::default(),
        );

        assert_eq!(plan.intent, PlannerIntent::InstallSdk);
        assert!(
            plan.parsed
                .contains(&("channel".to_owned(), "nightly".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("prefix".to_owned(), "D:\\ROCm\\therock_venvs".to_owned()))
        );
        assert!(plan.actions.iter().any(|action| {
            action.title == "Install TheRock SDK"
                && action.approval == "required"
                && action.args
                    == vec![
                        "install".to_owned(),
                        "sdk".to_owned(),
                        "--channel".to_owned(),
                        "nightly".to_owned(),
                        "--format".to_owned(),
                        "wheel".to_owned(),
                        "--prefix".to_owned(),
                        "D:\\ROCm\\therock_venvs".to_owned(),
                    ]
        }));
    }

    #[test]
    fn hybrid_planner_builds_requested_therock_build_date_install_call() {
        let plan = build_freeform_plan(
            "install the TheRock wheel from date 06052026 into D:\\ROCm\\therock_venvs",
            &RocmCliConfig::default(),
        );

        assert_eq!(plan.intent, PlannerIntent::InstallSdk);
        assert!(
            plan.parsed
                .contains(&("build_date".to_owned(), "2026-06-05".to_owned()))
        );
        assert!(plan.actions.iter().any(|action| {
            action.title == "Install TheRock SDK"
                && action.approval == "required"
                && action.args
                    == vec![
                        "install".to_owned(),
                        "sdk".to_owned(),
                        "--channel".to_owned(),
                        "release".to_owned(),
                        "--format".to_owned(),
                        "wheel".to_owned(),
                        "--prefix".to_owned(),
                        "D:\\ROCm\\therock_venvs".to_owned(),
                        "--build-date".to_owned(),
                        "2026-06-05".to_owned(),
                    ]
        }));
    }

    #[test]
    fn hybrid_planner_asks_for_folder_before_therock_install() {
        let plan = build_freeform_plan(
            "install the TheRock wheel from date 06052026",
            &RocmCliConfig::default(),
        );

        assert_eq!(plan.intent, PlannerIntent::Ask);
        assert!(plan.actions.is_empty());
        assert!(plan.approval.contains("install folder"));
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("install folder"))
        );
        assert!(
            plan.parsed
                .contains(&("build_date".to_owned(), "2026-06-05".to_owned()))
        );
    }

    #[test]
    fn hybrid_planner_handles_small_cpu_model_without_gpu_fallback() {
        let plan = build_freeform_plan("run a small local model on cpu", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert!(
            plan.parsed
                .contains(&("model".to_owned(), "sshleifer/tiny-gpt2".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("device_policy".to_owned(), "cpu_not_supported".to_owned()))
        );
        assert!(plan.actions.is_empty());
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("CPU mode is not offered"))
        );
    }

    #[test]
    fn hybrid_planner_lets_a_configured_engine_outrank_the_host_default() {
        let config = RocmCliConfig {
            default_engine: Some("lemonade".to_owned()),
            ..RocmCliConfig::default()
        };
        let plan = build_freeform_plan_with_recipes(
            "serve some/unmatched-model",
            &config,
            Some(&[]),
            "vllm",
        );

        let engine_arg = plan
            .actions
            .iter()
            .find_map(|action| {
                let index = action.args.iter().position(|arg| arg == "--engine")?;
                action.args.get(index + 1).cloned()
            })
            .expect("the generated serve command must name an engine");
        assert_eq!(
            engine_arg, "lemonade",
            "an engine the user configured must still win:\n{:?}",
            plan.actions
        );
    }

    #[test]
    fn hybrid_planner_bakes_the_host_engine_into_the_generated_serve_command() {
        // The generated command carries an explicit `--engine`, which outranks
        // every other signal in `select_serve_engine` -- including the configured
        // default. So whatever this planner picks IS what runs, and on an Instinct
        // host that must be vLLM. A GPU-blind constant here reintroduced the very
        // bug this PR fixes, through the strongest override available.
        //
        // An empty recipe set is what reaches the host default: a request naming
        // an engine, or a matched recipe that prefers one, is answered before the
        // fallback -- correctly, since a GGUF model only Lemonade can serve must
        // not be forced onto vLLM by the host.
        let plan = build_freeform_plan_with_recipes(
            "serve some/unmatched-model",
            &RocmCliConfig::default(),
            Some(&[]),
            "vllm",
        );

        let engine_arg = plan
            .actions
            .iter()
            .find_map(|action| {
                let index = action.args.iter().position(|arg| arg == "--engine")?;
                action.args.get(index + 1).cloned()
            })
            .expect("the generated serve command must name an engine");
        assert_eq!(
            engine_arg, "vllm",
            "the host's engine must reach the generated command:\n{:?}",
            plan.actions
        );
    }

    #[test]
    fn hybrid_planner_defaults_generic_local_assistant_to_validated_qwen() {
        let plan = build_freeform_plan_with_recipes(
            "start a local model",
            &RocmCliConfig::default(),
            None,
            "lemonade",
        );

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert!(plan.parsed.contains(&(
            "model".to_owned(),
            providers::BUILTIN_ASSISTANT_MODEL_ID.to_owned()
        )));
        assert!(plan.actions.iter().any(|action| {
            action.approval == "required"
                && action.args
                    == vec![
                        "serve".to_owned(),
                        providers::BUILTIN_ASSISTANT_MODEL_ID.to_owned(),
                        "--engine".to_owned(),
                        "lemonade".to_owned(),
                        "--device".to_owned(),
                        "gpu_required".to_owned(),
                        "--managed".to_owned(),
                    ]
        }));
    }

    #[test]
    fn freeform_plan_next_action_rejects_cpu_mode_request() {
        assert!(
            freeform_plan_next_action("run a small local model on cpu", &RocmCliConfig::default())
                .is_none()
        );
    }

    #[test]
    fn freeform_plan_next_action_surfaces_approval_action() {
        let action =
            freeform_plan_next_action("serve qwen3.5 with vllm", &RocmCliConfig::default())
                .expect("serve request should have next action");

        assert_eq!(action.title, "Launch local endpoint");
        assert!(action.approval_required);
        assert!(!action.has_placeholders);
        assert_eq!(
            action.args,
            vec![
                "serve".to_owned(),
                "Qwen/Qwen3.5-4B".to_owned(),
                "--engine".to_owned(),
                "vllm".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
    }

    #[test]
    fn hybrid_planner_driver_action_includes_yes_for_approved_execution() {
        let plan = build_freeform_plan(
            "install the linux driver with dkms",
            &RocmCliConfig::default(),
        );
        let action = plan
            .actions
            .iter()
            .find(|action| action.title == "Install driver")
            .expect("driver plan should include install action");

        assert_eq!(plan.intent, PlannerIntent::InstallDriver);
        assert_eq!(
            action.args,
            vec![
                "install".to_owned(),
                "driver".to_owned(),
                "--dkms".to_owned(),
                "--yes".to_owned(),
            ]
        );
    }

    #[test]
    fn hybrid_planner_unknown_request_is_read_only_inspection() {
        let plan = build_freeform_plan("what is installed here", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Inspect);
        assert!(
            plan.actions
                .iter()
                .all(|action| action.approval == "not required")
        );
        assert!(
            plan.actions
                .iter()
                .all(|action| action.args == vec!["examine".to_owned()])
        );
    }

    #[test]
    fn hybrid_planner_routes_common_status_questions_to_read_only_inspection() {
        for prompt in [
            "is rocm installed?",
            "which gpu is on my machine?",
            "where is therock installed?",
        ] {
            let plan = build_freeform_plan(prompt, &RocmCliConfig::default());

            assert_eq!(plan.intent, PlannerIntent::Inspect, "{prompt}");
            assert_eq!(plan.approval, "not required for inspection", "{prompt}");
            assert_eq!(plan.actions.len(), 1, "{prompt}");
            assert_eq!(plan.actions[0].approval, "not required", "{prompt}");
            assert_eq!(plan.actions[0].args, vec!["examine".to_owned()], "{prompt}");
        }
    }

    #[test]
    fn hybrid_planner_routes_comfyui_help_and_actions() {
        let status = build_freeform_plan("how do i setup comfyui", &RocmCliConfig::default());
        assert_eq!(status.intent, PlannerIntent::Inspect);
        assert_eq!(status.approval, "not required for inspection");
        assert_eq!(
            status.actions[0].args,
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert_eq!(status.actions[0].approval, "not required");

        let install =
            build_freeform_plan("can you setup comfyui for me", &RocmCliConfig::default());
        assert_eq!(install.approval, "required before installing ComfyUI");
        assert_eq!(
            install.actions[0].args,
            vec!["comfyui".to_owned(), "install".to_owned()]
        );
        assert_eq!(install.actions[0].approval, "required");

        let start = build_freeform_plan("can you start comfyui", &RocmCliConfig::default());
        assert_eq!(start.approval, "required before launch");
        assert_eq!(
            start.actions[0].args,
            vec!["comfyui".to_owned(), "start".to_owned()]
        );
        assert_eq!(start.actions[0].approval, "required");
    }

    #[test]
    fn hybrid_planner_casual_request_has_no_rocm_action() {
        let plan = build_freeform_plan("hi", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Ask);
        assert!(plan.actions.is_empty());
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("No ROCm action"))
        );
    }

    #[test]
    fn render_freeform_plan_exposes_structured_tool_calls() {
        let (_root, paths) = test_paths("hybrid-render");
        let rendered =
            render_freeform_plan("serve qwen3.5 with vllm", &paths, &RocmCliConfig::default());

        assert!(rendered.contains("planner: hybrid-parser-v1"));
        assert!(rendered.contains("tool_schema: rocm-tools-v0"));
        assert!(rendered.contains(
            "tool_call: rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        ));
        assert!(rendered.contains(
            "next_tool_call: rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        ));
        assert!(rendered.contains("next_tool_approval: required"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn provider_planner_response_reduces_to_validated_rocm_tool_call() -> Result<()> {
        let content = r#"{
            "intent": "serve",
            "confidence": "high",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "sshleifer/tiny-gpt2", "--engine", "vllm", "--device", "gpu_required", "--managed"]
            },
            "notes": ["resolved the missing model to a tiny test model"]
        }"#;

        let plan = provider_planner_response_to_plan("start a local model", "local", content)?;

        assert!(plan.provider_assisted);
        assert!(plan.planner.contains("provider:local"));
        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert_eq!(plan.actions[0].approval, "required");
        assert_eq!(
            plan.actions[0].args,
            vec![
                "serve".to_owned(),
                "sshleifer/tiny-gpt2".to_owned(),
                "--engine".to_owned(),
                "vllm".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("validated rocm tool call"))
        );
        Ok(())
    }

    #[test]
    fn provider_planner_rejects_public_bind_requests() {
        for content in [
            r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "tiny.gguf", "--engine", "vllm", "--allow-public-bind", "--managed"]
            }
        }"#,
            r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "tiny.gguf", "--engine", "vllm", "--host", "0.0.0.0", "--managed"]
            }
        }"#,
        ] {
            let error = provider_planner_response_to_plan("serve publicly", "local", content)
                .unwrap_err()
                .to_string();

            assert!(
                error.contains("public network binding") || error.contains("non-local host"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn provider_planner_requires_managed_serve_requests() {
        for args in [
            vec!["serve", "qwen", "--engine", "vllm"],
            vec!["serve", "qwen", "--engine", "vllm", "--foreground"],
        ] {
            let call = ProviderPlannerToolCall {
                tool: "rocm".to_owned(),
                args: args.into_iter().map(str::to_owned).collect(),
            };
            let error = validate_provider_planner_tool_call(&call)
                .unwrap_err()
                .to_string();

            assert!(error.contains("--managed"), "unexpected error: {error}");
        }
    }

    #[test]
    fn provider_planner_requires_user_folder_for_therock_install() {
        let call = ProviderPlannerToolCall {
            tool: "rocm".to_owned(),
            args: vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
            ],
        };
        let error = validate_provider_planner_tool_call(&call)
            .unwrap_err()
            .to_string();

        assert!(error.contains("ask the user"), "unexpected error: {error}");
    }

    #[test]
    fn provider_planner_response_rejects_cpu_serve_device() {
        for args in [
            vec![
                "serve",
                "sshleifer/tiny-gpt2",
                "--engine",
                "vllm",
                "--device",
                "cpu",
                "--managed",
            ],
            vec![
                "serve",
                "sshleifer/tiny-gpt2",
                "--engine",
                "vllm",
                "--device=cpu",
                "--managed",
            ],
            vec![
                "serve",
                "sshleifer/tiny-gpt2",
                "--engine",
                "vllm",
                "--device",
                "cpu_only",
                "--managed",
            ],
        ] {
            let call = ProviderPlannerToolCall {
                tool: "rocm".to_owned(),
                args: args.into_iter().map(str::to_owned).collect(),
            };
            let error = validate_provider_planner_tool_call(&call)
                .unwrap_err()
                .to_string();

            assert!(error.contains("CPU execution"));
            assert!(error.contains("ROCm GPU execution"));
        }
    }
}
