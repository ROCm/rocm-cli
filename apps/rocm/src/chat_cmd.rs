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
//! Four source-text tests in this module's own `mod tests`
//! (`command_chat_interactive_routes_to_dash_run_chat_not_tui`,
//! `command_chat_interactive_notes_dropped_provider_flag`,
//! `command_chat_honors_chat_mock_and_keeps_prompt_passthrough` and
//! `command_chat_reads_prompt_from_piped_stdin`) read this file at test time and
//! assert on the text of `run_chat_command`, so a rename of the function or of
//! the calls they look for has to be made there too. They share
//! `slice_braced_block` and `strip_line_comments` with `main.rs`'s tests through
//! `crate::tests::`.

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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::tests::{slice_braced_block, strip_line_comments};

    /// The source of `chat_cmd.rs`, where the `Command::Chat` handler lives
    /// (it was the arm's own body in `dispatch()` in `main.rs`).
    fn chat_cmd_source() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("chat_cmd.rs");
        fs::read_to_string(path).expect("read apps/rocm/src/chat_cmd.rs")
    }

    /// Extract the `Command::Chat` handler, `run_chat_command`: its parameter
    /// list (where the `--chat-mock` and sampling fields are visible, as the
    /// arm's destructure pattern once showed them) followed by its brace-balanced
    /// body, so both the received fields and the dispatch code are visible.
    fn command_chat_handler_body(src: &str) -> String {
        let name = "pub(crate) fn run_chat_command(";
        let fn_at = src.find(name).expect("Command::Chat handler present");
        let from_params = &src[fn_at + name.len()..];
        // The parameter list ends at the signature's `) -> Result<()> {`, which
        // is also the opener of the body block.
        let opener = ") -> Result<()> {";
        let params_end = from_params
            .find(opener)
            .expect("handler signature ends with `) -> Result<()> {`");
        let body = slice_braced_block(&from_params[params_end..], opener);
        // Return the parameter list + body together.
        format!("{}{}", &from_params[..params_end], body)
    }

    #[test]
    fn command_chat_interactive_routes_to_dash_run_chat_not_tui() {
        let src = chat_cmd_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        assert!(
            body.contains("dash::run_chat("),
            "interactive `rocm chat` must route to dash::run_chat(...); body:\n{body}"
        );
        assert!(
            !body.contains("tui::run"),
            "Command::Chat handler must NOT call tui::run after the reroute; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_interactive_notes_dropped_provider_flag() {
        // Phase-9 polish: --provider on interactive `rocm chat` is no longer
        // silently ignored — the handler emits a one-line note when provider is
        // set before rerouting to the dash. Proven by reading the handler body
        // (the interactive branch requires a TTY, unavailable in CI).
        let src = chat_cmd_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        assert!(
            body.contains("provider.is_some()"),
            "handler must gate the note on a set --provider; body:\n{body}"
        );
        assert!(
            body.contains("/provider"),
            "the note must point the user at /provider for live switching; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_honors_chat_mock_and_keeps_prompt_passthrough() {
        let src = chat_cmd_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        // --chat-mock is forwarded into the dash reroute.
        assert!(
            body.contains("chat_mock,") || body.contains("chat_mock\n"),
            "Command::Chat must destructure the --chat-mock field; body:\n{body}"
        );
        assert!(
            body.contains("dash::run_chat(") && body.contains("chat_mock,"),
            "--chat-mock must be forwarded to run_chat; body:\n{body}"
        );
        assert!(
            body.contains("ChatInferenceParams")
                && body.contains("temperature,")
                && body.contains("top_p,")
                && body.contains("max_tokens,"),
            "sampling controls must be forwarded to interactive dash chat; body:\n{body}"
        );
        // The scriptable prompt passthrough stays on the text render path,
        // NOT the dash/TUI.
        assert!(
            body.contains("chat::render_chat_prompt_text("),
            "prompt passthrough must still call render_chat_prompt_text; body:\n{body}"
        );
        assert!(
            body.contains("chat::render_chat_text("),
            "non-interactive no-prompt path must still call render_chat_text; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_reads_prompt_from_piped_stdin() {
        // A structural guard on the wiring, not a behavioral test: the handler
        // reads the real fd 0, which an in-process test cannot pipe. The
        // behavior — piped text reaching the model unaltered — is covered end
        // to end by the `chat-09` scenario (`@id:chat-cli-stdin-prompt`); what
        // is checked here is that the `--prompt`-less arm still resolves the
        // prompt from `read_piped_prompt`, with the result feeding the dispatch
        // rather than being discarded or read after the send decision is
        // already made.
        let src = chat_cmd_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        assert!(
            body.contains("chat::read_piped_prompt("),
            "no-prompt path must read piped stdin via read_piped_prompt; body:\n{body}"
        );
        assert!(
            body.contains("None => chat::read_piped_prompt()?"),
            "the piped read must supply the prompt that is dispatched (and \
             propagate its error), not be a discarded call; body:\n{body}"
        );
        let read_at = body
            .find("chat::read_piped_prompt(")
            .expect("asserted above");
        let send_at = body
            .find("chat::render_chat_prompt_text(")
            .unwrap_or_else(|| panic!("chat handler no longer sends a prompt; body:\n{body}"));
        assert!(
            read_at < send_at,
            "stdin must be read before the send/status-screen decision, or a \
             piped prompt cannot influence it; body:\n{body}"
        );
    }
}
