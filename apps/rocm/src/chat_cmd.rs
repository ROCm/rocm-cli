// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm chat`: the command handler (`run_chat_command`), the body of the
//! `Command::Chat` arm of `dispatch()` relocated here verbatim. In a terminal
//! with no `--prompt` it hands off to the dashboard chat (`dash::run_chat`),
//! noting that `--provider` is dropped there; otherwise it takes the prompt
//! from `--prompt` or piped standard input and prints the reply through
//! `chat::render_chat_prompt_text`, or the status screen through
//! `chat::render_chat_text` when there is no prompt at all.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. This is a module of its own, not part of `chat.rs`,
//! because the handler calls into both `chat.rs` and `dash.rs` while `dash.rs`
//! already calls into `chat.rs`: placing it in `chat.rs` would make the two
//! depend on each other. This module depends on both, and neither depends on
//! it. `Provider`, the clap enum behind `--provider`, and `ChatInferenceParams`
//! stay in `main.rs`; `run_chat_command` is `pub(crate)` for `dispatch()`,
//! which calls it through the re-import `use crate::chat_cmd::run_chat_command;`.
//!
//! Four source-text tests in `main.rs` (`command_chat_interactive_routes_to_dash_run_chat_not_tui`,
//! `command_chat_interactive_notes_dropped_provider_flag`,
//! `command_chat_honors_chat_mock_and_keeps_prompt_passthrough` and
//! `command_chat_reads_prompt_from_piped_stdin`) read this file at test time and
//! assert on the text of `run_chat_command`, so a rename of the function or of
//! the calls they look for has to be made there too.

use anyhow::Result;
use rocm_core::{AppPaths, interactive_terminal};

use crate::{ChatInferenceParams, Provider, chat, dash, provider_name};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_chat_command(
    provider: Option<Provider>,
    model: Option<String>,
    prompt: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    tools: bool,
    chat_mock: bool,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    if interactive_terminal() && prompt.is_none() {
        // Interactive `rocm chat` routes to the unified dash chat
        // (parity reached, Phases 3-8). The dash auto-detects the
        // provider and supports `/provider` switching; --provider is
        // honored only on the non-interactive render path below. The
        // legacy TUI assistant has been retired.
        if provider.is_some() {
            // --provider isn't threaded into the interactive dash; say so
            // instead of dropping it silently — the user can switch live.
            eprintln!("note: launching the dash chat; switch providers with /provider <name>");
        }
        return dash::run_chat(
            chat_mock,
            ChatInferenceParams {
                temperature,
                top_p,
                max_tokens,
            },
            false,
        );
    }
    // Resolve the prompt from `--prompt` or, when it is omitted and
    // stdin is not a terminal, from piped standard input — the
    // documented `echo "…" | rocm chat` path. Only when neither
    // supplies a prompt do we fall back to the status screen, which
    // two kinds of invocation reach: stdin is a TTY that the
    // interactive branch above declined (it also requires stdout to be
    // one), or stdin was redirected and carried nothing but whitespace
    // — an empty pipe or `< /dev/null`.
    let prompt = match prompt {
        Some(prompt) => Some(prompt),
        None => chat::read_piped_prompt()?,
    };
    match prompt {
        Some(prompt) => print!(
            "{}",
            chat::render_chat_prompt_text(
                &paths,
                provider.map_or("local", provider_name),
                model.as_deref(),
                &prompt,
                tools,
                ChatInferenceParams {
                    temperature,
                    top_p,
                    max_tokens,
                },
            )?
        ),
        None => print!(
            "{}",
            chat::render_chat_text(&paths, provider.map_or("local", provider_name))?
        ),
    }
    Ok(())
}
