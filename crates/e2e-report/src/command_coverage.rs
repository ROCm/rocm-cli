// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! CLI command-coverage audit: which `rocm` commands were exercised by at
//! least one platform's E2E run, and whether they passed where they ran.
//!
//! Consumes [`crate::consolidated::PlatformReport`] but shares no type with
//! the `Grid`/reconciliation model itself — this is a separate concern
//! bolted onto the same consolidated markdown output.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::consolidated::PlatformReport;

/// A command signature: what we group invocations by in the coverage table.
///
/// `command` is the full invocation as executed; `engine` is the engine that
/// actually ran, with a "(default)" suffix when the CLI chose it itself. Grouping
/// on both keeps an explicit `--engine vllm` distinct from a default that
/// resolved to vLLM, and distinct from the same command resolving to lemonade on
/// another platform.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
struct CommandKey {
    command: String,
    engine: String,
}

/// The `rocm` command surface we measure coverage against — the denominator.
///
/// Curated from the CLI's own `--help` tree (top-level subcommands and their
/// meaningful second-level subcommands), normalized to the `rocm <base>` shape
/// that `record_command`'s signature produces (see `derive_subcommand`). Pure
/// `help`/`completions` plumbing is intentionally excluded — they aren't product
/// behaviour worth an E2E. When the CLI gains a command, add it here so the
/// coverage % reflects the real surface (a deliberate, reviewable denominator
/// beats silently drifting).
pub(crate) const KNOWN_COMMAND_SURFACE: &[&str] = &[
    "rocm remote targets",
    "rocm remote serve",
    "rocm remote doctor",
    "rocm remote status",
    "rocm remote attach",
    "rocm remote stop",
    "rocm examine",
    "rocm diagnose",
    "rocm fix",
    "rocm version",
    "rocm setup status",
    "rocm setup reset",
    "rocm chat",
    "rocm install sdk",
    "rocm install driver",
    "rocm update",
    "rocm runtimes list",
    "rocm runtimes activate",
    "rocm runtimes rollback",
    "rocm runtimes uninstall",
    "rocm runtimes import",
    "rocm runtimes adopt",
    "rocm engines list",
    "rocm engines install",
    "rocm engines shell",
    "rocm model",
    "rocm serve",
    "rocm comfyui status",
    "rocm comfyui install",
    "rocm comfyui start",
    "rocm comfyui stop",
    "rocm comfyui logs",
    "rocm comfyui models-path",
    "rocm services list",
    "rocm services logs",
    "rocm services stop",
    "rocm services restart",
    "rocm automations list",
    "rocm automations enable",
    "rocm automations disable",
    "rocm config show",
    "rocm config set-engine",
    "rocm config set-default-engine",
    "rocm config set-default-runtime",
    "rocm config set-telemetry",
    "rocm config set-permissions",
    "rocm logs",
    "rocm dash",
    "rocm uninstall",
];

/// Normalize a recorded command signature to its base `rocm <base>` form for
/// matching against `KNOWN_COMMAND_SURFACE` — drops the behaviour-shaping
/// suffixes `record_command` appends (` --engine`, ` (default engine)`).
fn command_base(sig: &str) -> &str {
    sig.split(" --engine")
        .next()
        .unwrap_or(sig)
        .split(" (default engine)")
        .next()
        .unwrap_or(sig)
        .trim()
}

/// The `KNOWN_COMMAND_SURFACE` entry a recorded command exercises, if any.
///
/// A recorded base carries positionals the surface entry does not — e.g.
/// `rocm serve Qwen/Qwen2.5-1.5B-Instruct` exercises the surface command
/// `rocm serve`. Match by the LONGEST surface entry that is a word-boundary
/// prefix of the base, so a two-word command (`rocm install sdk`) wins over any
/// shorter prefix and `rocm serve <model>` maps to `rocm serve`.
fn matched_surface_command(base: &str) -> Option<&'static str> {
    KNOWN_COMMAND_SURFACE
        .iter()
        .copied()
        .filter(|cmd| base == *cmd || base.starts_with(&format!("{cmd} ")))
        .max_by_key(|cmd| cmd.len())
}

/// Coverage of the known command surface: `(covered, total, uncovered_sorted)`.
/// A command counts as covered if any platform ran a matching invocation.
pub(crate) fn command_coverage_summary(
    reports: &[PlatformReport],
) -> (usize, usize, Vec<&'static str>) {
    use std::collections::BTreeSet;
    let mut exercised: BTreeSet<&'static str> = BTreeSet::new();
    for r in reports {
        for c in &r.commands {
            if let Some(cmd) = matched_surface_command(command_base(&c.subcommand)) {
                exercised.insert(cmd);
            }
        }
    }
    let uncovered: Vec<&'static str> = KNOWN_COMMAND_SURFACE
        .iter()
        .copied()
        .filter(|cmd| !exercised.contains(*cmd))
        .collect();
    let total = KNOWN_COMMAND_SURFACE.len();
    (total - uncovered.len(), total, uncovered)
}

/// Build the "which rocm commands are exercised, with which models/engines, on
/// which platform, and do they work" coverage table.
///
/// For each (command, model, engine) × platform cell: ✅ if every scenario that
/// ran that command on that platform passed, ❌ if any failed, `n/a` if the
/// command was never run there. "Passed" follows the scenario's own result, so a
/// command that is *supposed* to be rejected (its scenario asserts the failure)
/// still reads as ✅ — the tested behaviour held.
pub(crate) fn command_coverage_markdown(reports: &[PlatformReport]) -> String {
    // Platform columns in matrix order (platform+os), de-duplicated across tiers.
    let mut columns: Vec<String> = Vec::new();
    for r in reports {
        let col = r.label.clone();
        if !columns.contains(&col) {
            columns.push(col);
        }
    }

    // key → (column → all-passed-so-far). Absent column = not run there.
    let mut cells: BTreeMap<CommandKey, BTreeMap<String, bool>> = BTreeMap::new();
    for r in reports {
        let col = r.label.clone();
        let passed = r.scenario_pass_map();
        for c in &r.commands {
            // Full command as executed; fall back to the stripped signature for
            // older artifacts that predate the `command` field.
            let command = c.command.clone().unwrap_or_else(|| c.subcommand.clone());
            // Engine actually used, with a "(default)" marker when the CLI chose
            // it (no explicit --engine flag).
            let engine = match c.engine.as_deref() {
                Some(e) if c.engine_is_default => format!("{e} (default)"),
                Some(e) => e.to_string(),
                None => String::new(),
            };
            let key = CommandKey { command, engine };
            // A command's cell is ✅ only if EVERY scenario that ran it on this
            // platform passed; an unknown scenario is treated as passed (the
            // command ran and we have no failing evidence). ❌ here means the
            // command did NOT work on this platform — whether or not the failure
            // is a known/expected bug (that nuance lives in the expectation grid;
            // this coverage table only cares whether it worked here).
            let ok = c
                .scenario
                .as_deref()
                .and_then(|s| passed.get(s).copied())
                .unwrap_or(true);
            let entry = cells
                .entry(key)
                .or_default()
                .entry(col.clone())
                .or_insert(true);
            *entry = *entry && ok;
        }
    }

    if cells.is_empty() {
        return String::new();
    }

    let (covered, total, uncovered) = command_coverage_summary(reports);
    let pct = (covered * 100).checked_div(total).unwrap_or(0);

    let mut out = String::from("\n### Command coverage\n\n");
    let _ = writeln!(
        out,
        "**CLI surface coverage: {covered}/{total} commands ({pct}%)** exercised by \
         at least one platform.\n"
    );
    out.push_str("_Which `rocm` commands are exercised, with which engine, per platform. ");
    out.push_str(
        "✅ ran and worked here · ❌ ran but did not work here · `n/a` not run on this \
         platform — this row is a specific model/engine invocation and this platform \
         serves a different one, or the command is not applicable to its GPU/OS/engine._\n\n",
    );

    out.push_str("| Command | Engine |");
    for col in &columns {
        let _ = write!(out, " {col} |");
    }
    out.push('\n');
    out.push_str("|---|---|");
    for _ in &columns {
        out.push_str(":--:|");
    }
    out.push('\n');

    for (key, per_col) in &cells {
        let engine = if key.engine.is_empty() {
            "n/a"
        } else {
            &key.engine
        };
        let _ = write!(out, "| `{}` | {} |", key.command, engine);
        for col in &columns {
            // Not-run cells render as a grayed `n/a`, not blank, so an empty cell
            // clearly means "not applicable here" rather than looking broken.
            let mark = match per_col.get(col) {
                Some(true) => " ✅ |",
                Some(false) => " ❌ |",
                None => " `n/a` |",
            };
            out.push_str(mark);
        }
        out.push('\n');
    }

    // Fold-out list of the command surface NOT yet exercised by any platform, so
    // the coverage % is actionable rather than just a number.
    if !uncovered.is_empty() {
        let _ = write!(
            out,
            "\n<details><summary>Uncovered commands ({})</summary>\n\n",
            uncovered.len()
        );
        for cmd in &uncovered {
            let _ = writeln!(out, "- `{cmd}`");
        }
        out.push_str("\n</details>\n");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_base_strips_suffixes() {
        assert_eq!(command_base("rocm serve --engine"), "rocm serve");
        assert_eq!(command_base("rocm serve (default engine)"), "rocm serve");
        assert_eq!(command_base("rocm install sdk"), "rocm install sdk");
    }

    #[test]
    fn matched_surface_command_maps_positionals_and_prefers_longest() {
        // Regression: a serve command embeds the model in its base, so it must
        // still map to the surface entry `rocm serve` (was counted uncovered).
        assert_eq!(
            matched_surface_command("rocm serve Qwen/Qwen2.5-1.5B-Instruct"),
            Some("rocm serve")
        );
        assert_eq!(
            matched_surface_command("rocm serve Qwen3-0.6B-GGUF"),
            Some("rocm serve")
        );
        // Longest-prefix wins: a two-word surface command is not shadowed by a
        // shorter one, and `rocm install sdk` maps to itself, not `rocm install`.
        assert_eq!(
            matched_surface_command("rocm install sdk"),
            Some("rocm install sdk")
        );
        // A bare exact match still works; an unknown command matches nothing.
        assert_eq!(
            matched_surface_command("rocm version"),
            Some("rocm version")
        );
        assert_eq!(matched_surface_command("rocm bogus"), None);
    }
}
