// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Text extraction from a natural-language request, shared by `chat.rs`'s
//! prompt-intent heuristics (`fallback_rocm_tool_call_for_prompt`,
//! `fallback_config_tool_call_for_prompt`, `install_sdk_chat_approval_for_prompt`
//! and the `prompt_*` predicates) and the freeform planner: which of a list of
//! phrases a prompt
//! contains (`any_substring`), which install folder, TheRock build date or
//! TheRock version it names (`requested_install_prefix_from_prompt`,
//! `requested_therock_build_date_from_prompt`,
//! `requested_therock_version_from_prompt` and their private helpers), and
//! which serving engine it asks for (`infer_engine_from_request`).
//!
//! A leaf module: it depends only on `therock::RuntimeVersionSelector`, and
//! neither `chat.rs` nor `freeform_planner.rs` has to import the other for
//! these helpers. `tool_policy.rs` does not use it. They were relocated verbatim from `chat.rs` and
//! `freeform_planner.rs`; `chat.rs` re-exports `any_substring`, which
//! `freeform_answer.rs` still reaches as `chat::any_substring`.

use crate::therock;

pub(crate) fn any_substring(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

pub(crate) fn requested_install_prefix_from_prompt(prompt: &str) -> Option<String> {
    let lower = prompt.to_ascii_lowercase();
    for phrase in [
        "--prefix=",
        "--prefix ",
        " into folder ",
        " into ",
        " in folder ",
        " in ",
        " to folder ",
        " to ",
        " under ",
        " at ",
        " use folder ",
        " use ",
        "use ",
        " folder is ",
        " folder: ",
        "folder:",
    ] {
        let mut search_start = 0;
        while let Some(relative_index) = lower[search_start..].find(phrase) {
            let value_start = search_start + relative_index + phrase.len();
            if let Some(prefix) = clean_requested_install_prefix(&prompt[value_start..]) {
                return Some(prefix);
            }
            search_start = value_start;
        }
    }
    requested_bare_install_prefix_from_prompt(prompt)
}

fn requested_bare_install_prefix_from_prompt(prompt: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let mut remaining = prompt;
        while let Some(start) = remaining.find(quote) {
            let after_start = &remaining[start + quote.len_utf8()..];
            let Some(end) = after_start.find(quote) else {
                break;
            };
            if let Some(prefix) = clean_requested_install_prefix(&after_start[..end]) {
                return Some(prefix);
            }
            remaining = &after_start[end + quote.len_utf8()..];
        }
    }
    prompt.split_whitespace().find_map(|token| {
        clean_requested_install_prefix(
            token.trim_matches(|ch: char| {
                matches!(ch, '"' | '\'' | '`' | ',' | ';' | '.' | ')' | ']')
            }),
        )
    })
}

fn clean_requested_install_prefix(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    let mut chars = trimmed.chars();
    let first = chars.next()?;
    let extracted = if first == '"' || first == '\'' {
        let close = trimmed[first.len_utf8()..].find(first)?;
        &trimmed[first.len_utf8()..first.len_utf8().saturating_add(close)]
    } else {
        trimmed
            .split(['\r', '\n', ',', ';'])
            .next()
            .unwrap_or_default()
            .split(" and ")
            .next()
            .unwrap_or_default()
            .split(" with ")
            .next()
            .unwrap_or_default()
            .split(" then ")
            .next()
            .unwrap_or_default()
            .trim_end_matches(['.', ')', ']'])
    };
    let prefix = extracted.trim();
    if prefix.is_empty() || !looks_like_user_path(prefix) {
        return None;
    }
    Some(prefix.to_owned())
}

fn looks_like_user_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with('~')
        || value.starts_with("\\\\")
        || value.contains(":\\")
        || value.contains(":/")
        || value.contains('\\')
}

pub(crate) fn requested_therock_build_date_from_prompt(normalized: &str) -> Option<String> {
    normalized
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| token.len() == 8 && token.chars().all(|ch| ch.is_ascii_digit()))
        .find_map(
            |token| match therock::RuntimeVersionSelector::build_date(token).ok()? {
                therock::RuntimeVersionSelector::BuildDate(date) => Some(date),
                therock::RuntimeVersionSelector::Version(_) => None,
            },
        )
        .or_else(|| {
            normalized
                .split_whitespace()
                .map(|token| {
                    token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
                })
                .find_map(|token| {
                    match therock::RuntimeVersionSelector::build_date(token).ok()? {
                        therock::RuntimeVersionSelector::BuildDate(date) => Some(date),
                        therock::RuntimeVersionSelector::Version(_) => None,
                    }
                })
        })
}

pub(crate) fn requested_therock_version_from_prompt(normalized: &str) -> Option<String> {
    normalized
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '+' | '-' | '_')))
        .filter(|token| looks_like_therock_runtime_version(token))
        .find_map(|token| {
            therock::RuntimeVersionSelector::version(token)
                .ok()
                .map(|_| token.to_owned())
        })
}

fn looks_like_therock_runtime_version(token: &str) -> bool {
    let Some((prefix, date)) = token.rsplit_once('a') else {
        return false;
    };
    !prefix.is_empty()
        && prefix.chars().next().is_some_and(|ch| ch.is_ascii_digit())
        && prefix.contains('.')
        && date.len() == 8
        && date.chars().all(|ch| ch.is_ascii_digit())
        && therock::RuntimeVersionSelector::build_date(date).is_ok()
}

pub(crate) fn infer_engine_from_request(lower: &str) -> Option<&'static str> {
    // Word-boundary match, not substring: "aim" as a bare `.contains` also
    // matches "aiming"/"claim"/"maintain", silently steering an unrelated
    // chat request onto the Docker-backed AIM engine.
    let words: std::collections::HashSet<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    ["lemonade", "vllm", "aim"]
        .into_iter()
        .find(|engine| words.contains(engine))
}

#[cfg(test)]
mod tests {
    use super::infer_engine_from_request;

    /// "aim" must match as its own word, not as a substring of an unrelated
    /// word like "aiming"/"claim"/"maintain" in a free-text chat request.
    #[test]
    fn infer_engine_from_request_does_not_match_aim_as_a_substring() {
        assert_eq!(
            infer_engine_from_request("serve a small model, aiming for low latency"),
            None
        );
        assert_eq!(
            infer_engine_from_request("please explain how to maintain a claim"),
            None
        );
        assert_eq!(
            infer_engine_from_request("serve qwen3.5-9b with aim"),
            Some("aim")
        );
    }
}
