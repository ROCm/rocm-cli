// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The CLI's audit and lifecycle log: `record_cli_audit_event` writes one
//! audit event through `rocm_core::append_audit_event`, then appends the same
//! event as a text line to `logs/cli-lifecycle.log` and to a per-action file
//! under `logs/cli/` (`append_cli_lifecycle_logs`, `render_cli_lifecycle_log_line`,
//! `append_text_log_line`, `cli_lifecycle_log_path`, `cli_action_log_path`),
//! with `sanitize_log_component`/`sanitize_log_value` keeping file names and
//! log lines to one safe line each.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. This module depends only on `rocm_core`. Callers in
//! other modules keep reaching these items as `crate::<name>` through the
//! `use crate::cli_audit::{...};` re-import in `main.rs`; `git grep` for the
//! name finds them, so they are not listed here. `record_cli_audit_event`,
//! `cli_lifecycle_log_path` and `sanitize_log_value` are `pub(crate)`; every
//! other item has no caller outside this module and stays private.
//!
//! The two regression tests for this log live in this module's own
//! `mod tests`, reaching the shared `test_paths` fixture through
//! `crate::tests::`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rocm_core::{AppPaths, AuditEventRecord, append_audit_event};

pub(crate) fn record_cli_audit_event(
    paths: &AppPaths,
    category: &str,
    action: &str,
    level: &str,
    message: impl Into<String>,
    service_id: Option<&str>,
) {
    let event = AuditEventRecord {
        at_unix_ms: rocm_core::unix_time_millis(),
        source: "rocm".to_owned(),
        category: category.to_owned(),
        actor: "cli".to_owned(),
        level: level.to_owned(),
        action: action.to_owned(),
        message: message.into(),
        watcher_id: None,
        service_id: service_id.map(str::to_owned),
    };
    if let Err(error) = append_audit_event(paths, &event) {
        eprintln!("warning: failed to write audit event: {error}");
    }
    if let Err(error) = append_cli_lifecycle_logs(paths, &event) {
        eprintln!("warning: failed to write CLI lifecycle log: {error}");
    }
}

fn append_cli_lifecycle_logs(paths: &AppPaths, event: &AuditEventRecord) -> Result<()> {
    paths.ensure()?;
    let line = render_cli_lifecycle_log_line(event);
    append_text_log_line(&cli_lifecycle_log_path(paths), &line)?;
    append_text_log_line(
        &cli_action_log_path(paths, &event.category, &event.action),
        &line,
    )?;
    Ok(())
}

fn render_cli_lifecycle_log_line(event: &AuditEventRecord) -> String {
    format!(
        "{} level={} category={} action={} service_id={} message={}\n",
        event.at_unix_ms,
        sanitize_log_value(&event.level),
        sanitize_log_value(&event.category),
        sanitize_log_value(&event.action),
        event.service_id.as_deref().unwrap_or("<none>"),
        sanitize_log_value(&event.message)
    )
}

fn append_text_log_line(path: &Path, line: &str) -> Result<()> {
    let parent = path.parent().context("log path has no parent directory")?;
    fs::create_dir_all(parent)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(line.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub(crate) fn cli_lifecycle_log_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("logs").join("cli-lifecycle.log")
}

fn cli_action_log_path(paths: &AppPaths, category: &str, action: &str) -> PathBuf {
    paths.data_dir.join("logs").join("cli").join(format!(
        "{}-{}.log",
        sanitize_log_component(category),
        sanitize_log_component(action)
    ))
}

fn sanitize_log_component(value: &str) -> String {
    let component = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    if component.is_empty() {
        "unknown".to_owned()
    } else {
        component
    }
}

pub(crate) fn sanitize_log_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '\r' | '\n' | '\t' => ' ',
            _ => ch,
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn record_cli_audit_event_writes_cli_lifecycle_record() -> Result<()> {
        let (root, paths) = test_paths("cli-audit");

        record_cli_audit_event(
            &paths,
            "runtime",
            "runtime_activate",
            "info",
            "activated test runtime",
            None,
        );

        let events = rocm_core::load_recent_audit_events(&paths, 1)?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, "rocm");
        assert_eq!(events[0].actor, "cli");
        assert_eq!(events[0].category, "runtime");
        assert_eq!(events[0].action, "runtime_activate");
        assert_eq!(events[0].message, "activated test runtime");
        let lifecycle_log = fs::read_to_string(cli_lifecycle_log_path(&paths))?;
        assert!(lifecycle_log.contains("category=runtime"));
        assert!(lifecycle_log.contains("action=runtime_activate"));
        assert!(lifecycle_log.contains("message=activated test runtime"));
        let action_log =
            fs::read_to_string(cli_action_log_path(&paths, "runtime", "runtime_activate"))?;
        assert_eq!(lifecycle_log, action_log);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn cli_lifecycle_log_sanitizes_components_and_newlines() -> Result<()> {
        let (root, paths) = test_paths("cli-lifecycle-sanitize");
        let event = AuditEventRecord {
            at_unix_ms: 42,
            source: "rocm".to_owned(),
            category: "Run Time".to_owned(),
            actor: "cli".to_owned(),
            level: "info".to_owned(),
            action: "install/sdk".to_owned(),
            message: "line one\nline two".to_owned(),
            watcher_id: None,
            service_id: Some("svc-1".to_owned()),
        };

        append_cli_lifecycle_logs(&paths, &event)?;

        let line = fs::read_to_string(cli_lifecycle_log_path(&paths))?;
        assert!(line.contains("service_id=svc-1"));
        assert!(line.contains("message=line one line two"));
        assert!(
            cli_action_log_path(&paths, "Run Time", "install/sdk")
                .ends_with("run-time-install-sdk.log")
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
