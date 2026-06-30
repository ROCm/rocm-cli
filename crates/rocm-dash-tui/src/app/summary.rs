// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure display utilities for slash-tool outcomes and natural-language plans.
//!
//! Concise, length-bounded summaries of read-only slash-tool results (never a
//! raw JSON transcript dump, per docs/ux-guidelines.md) plus the parser that
//! maps the `natural_language_plan` tool result into a [`PlannedAction`]. All
//! free functions — no `AppState` access — split out of `app/mod.rs` to keep the
//! core reducer + event loop focused.

use super::PlannedAction;

/// Max object fields a slash-tool summary surfaces before truncating with an
/// ellipsis line. Keeps `summarize_json_value` output terse and length-bounded.
pub(super) const SUMMARY_MAX_FIELDS: usize = 8;

/// Parse the `natural_language_plan` tool result into the rendered plan text and
/// the structured next action. The bin returns
/// `structuredContent: { request, text, action: {...}|null }`; we surface the
/// concise rendered `text` (the review) and map `action` to [`PlannedAction`].
/// Returns `None` only when the structured payload is missing/unusable.
pub(super) fn parse_plan_result(v: &serde_json::Value) -> Option<(String, Option<PlannedAction>)> {
    let text = v
        .pointer("/structuredContent/text")
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let action = v
        .pointer("/structuredContent/action")
        .filter(|a| a.is_object())
        .map(|a| PlannedAction {
            args: a
                .get("args")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            approval_required: a
                .get("approval_required")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            has_placeholders: a
                .get("has_placeholders")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            provider_assisted: a
                .get("provider_assisted")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        });
    Some((text, action))
}

/// Render a CONCISE, human summary of a read-only slash-tool outcome — never a
/// raw JSON transcript dump (per docs/ux-guidelines.md). Errors and approval
/// notes are surfaced plainly; a success object is reduced to a short headline
/// plus a few key/value lines, with arrays/objects collapsed to counts.
pub(super) fn summarize_slash_tool(
    label: &str,
    outcome: &crate::tool_exec::RocmToolOutcome,
) -> String {
    use crate::tool_exec::RocmToolOutcome;
    match outcome {
        RocmToolOutcome::Error(e) => format!("/{label} failed: {e}"),
        RocmToolOutcome::ApprovalRequired(_) => {
            format!("/{label}: this action needs approval (read-only command did not expect it)")
        }
        RocmToolOutcome::Result(v) => {
            let body = summarize_json_value(v);
            if body.is_empty() {
                format!("/{label}: done")
            } else {
                format!("/{label}:\n{body}")
            }
        }
    }
}

/// Collapse a JSON value into at most a handful of readable lines. Scalars print
/// inline; objects list their top-level fields (nested containers shown as
/// counts); arrays show a length. Keeps slash-tool output terse and scannable.
pub(super) fn summarize_json_value(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Object(map) => {
            let mut lines = Vec::new();
            for (k, val) in map.iter().take(SUMMARY_MAX_FIELDS) {
                lines.push(format!("  {k}: {}", scalar_or_shape(val)));
            }
            if map.len() > SUMMARY_MAX_FIELDS {
                lines.push(format!(
                    "  … ({} more fields)",
                    map.len() - SUMMARY_MAX_FIELDS
                ));
            }
            lines.join("\n")
        }
        Value::Array(arr) => format!("  {} item(s)", arr.len()),
        other => format!("  {}", scalar_or_shape(other)),
    }
}

/// A scalar's plain text, or a shape hint (`{N fields}` / `[N items]`) for a
/// nested container — used so summaries never inline a whole subtree.
pub(super) fn scalar_or_shape(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(a) => format!("[{} items]", a.len()),
        Value::Object(o) => format!("{{{} fields}}", o.len()),
    }
}
