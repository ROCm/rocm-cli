// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Answers a freeform request can get without running a command:
//! `render_freeform_read_only_answer` answers a plan whose single action needs
//! no approval and is a read-only `examine`, `comfyui status` or `comfyui logs`,
//! and `render_freeform_examine_answer` renders the plain-language ROCm status
//! (or install location) that the `examine` case prints.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `run_freeform` in `main.rs` is the only caller of
//! `render_freeform_read_only_answer`, which is therefore `pub(crate)` and
//! re-imported there; `render_freeform_examine_answer` has no caller outside
//! this module and stays private. It uses `chat::any_substring` and, besides
//! `chat.rs`, depends on `comfyui.rs`, `runtimes_cmd.rs` and `therock.rs`;
//! none of those calls back into this module. No unit test in this crate
//! calls either function directly.

use std::fmt::Write as _;

use anyhow::Result;
use rocm_core::{AppPaths, ExamineSummary, RocmCliConfig};
use rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES;

use crate::runtimes_cmd::{
    current_runtime_manifest, recover_setup_runtime_registration, runtime_usability_status,
};
use crate::{chat, comfyui, freeform_planner, therock};

pub(crate) fn render_freeform_read_only_answer(
    request: &str,
    plan: &freeform_planner::StructuredRequestPlan,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    if plan.actions.len() != 1 || plan.actions[0].approval != "not required" {
        return Ok(None);
    }
    match plan.actions[0].args.as_slice() {
        [command] if command == "examine" => {
            render_freeform_examine_answer(request, paths, config).map(Some)
        }
        [command, subcommand] if command == "comfyui" && subcommand == "status" => {
            comfyui::render_chat_status_answer(paths).map(Some)
        }
        [command, subcommand] if command == "comfyui" && subcommand == "logs" => {
            let logs = comfyui::render_logs(paths, DEFAULT_LOG_TAIL_LINES)?;
            Ok(Some(format!("ComfyUI logs\n\n{logs}")))
        }
        _ => Ok(None),
    }
}

fn render_freeform_examine_answer(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<String> {
    recover_setup_runtime_registration(paths, config)?;
    let examine = ExamineSummary::gather()?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let active = current_runtime_manifest(config, &manifests);
    let lower = request.to_ascii_lowercase();
    let asks_where = chat::any_substring(&lower, &["where", "folder", "path"]);
    let mut output = String::new();
    let _ = writeln!(
        output,
        "{}",
        if asks_where {
            "ROCm install location"
        } else {
            "ROCm status"
        }
    );
    let _ = writeln!(output);

    if let Some(detail) = examine.driver.detail.as_deref() {
        let _ = writeln!(output, "GPU: {detail}");
    } else if let Some(target) = examine.detected_gfx_target.as_deref() {
        let _ = writeln!(output, "GPU: AMD GPU target {target}");
    } else {
        let _ = writeln!(output, "GPU: I could not identify an AMD GPU yet.");
    }
    if let Some(target) = examine.detected_gfx_target.as_deref() {
        let _ = writeln!(output, "Target: {target}");
    }

    if let Some(manifest) = active {
        let status = runtime_usability_status(manifest);
        if status == "ready" {
            let _ = writeln!(output, "ROCm/TheRock: installed and active for ROCm CLI");
        } else {
            let _ = writeln!(output, "ROCm/TheRock: found, but status is {status}");
        }
        let _ = writeln!(output, "Folder: {}", manifest.install_root.display());
        let _ = writeln!(
            output,
            "Version: {}",
            therock::runtime_version_display(&manifest.version)
        );
        let _ = writeln!(output, "GPU package: {}", manifest.family);
    } else {
        let setup_root = config.setup.therock_venv.as_deref();
        if let Some(root) = setup_root {
            let _ = writeln!(
                output,
                "ROCm/TheRock: setup folder saved, but no active runtime is selected"
            );
            let _ = writeln!(output, "Folder: {}", root.display());
        } else if manifests.len() == 1 {
            let manifest = &manifests[0];
            let _ = writeln!(
                output,
                "ROCm/TheRock: installed, but not selected as the active runtime"
            );
            let _ = writeln!(output, "Folder: {}", manifest.install_root.display());
            let _ = writeln!(output, "Runtime: {}", manifest.runtime_key);
            let _ = writeln!(
                output,
                "Next step: rocm runtimes activate {}",
                manifest.runtime_key
            );
        } else if manifests.is_empty() {
            let _ = writeln!(output, "ROCm/TheRock: not installed for ROCm CLI yet");
            let _ = writeln!(
                output,
                "Next step: run `rocm` and choose Install ROCm, or ask to install TheRock into a folder you choose."
            );
        } else {
            let _ = writeln!(
                output,
                "ROCm/TheRock: multiple installs found, but none is active"
            );
            let _ = writeln!(output, "Run `rocm runtimes list` to choose one.");
        }
    }

    if examine.legacy_rocm.status == "not_detected" && active.is_some() {
        let _ = writeln!(
            output,
            "Note: ROCm CLI is using its managed TheRock runtime, not a global ROCm install."
        );
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Nothing was changed.");
    Ok(output)
}
