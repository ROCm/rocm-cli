// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared helpers for launching `rocm` sub-commands from operational screens (Phase 3 Wave 1).
//!
//! Every screen that routes a mutating action through the
//! approval gate + job-bridge resolves the binary the same way, so the logic
//! lives here once instead of being re-implemented per screen.

/// The `rocm` binary to invoke: this process's own path (so an in-tree dev
/// build calls itself), or the bare name `rocm` (PATH lookup) when
/// `current_exe()` is unavailable — never a silent no-op.
pub fn resolve_exe() -> String {
    std::env::current_exe()
        .ok()
        .map_or_else(|| "rocm".to_string(), |p| p.to_string_lossy().into_owned())
}

/// Short, human-readable basename of a resolved command, for approval previews.
pub fn exe_label(cmd: &str) -> &str {
    cmd.rsplit(['/', '\\']).next().unwrap_or(cmd)
}

/// Quote `value` for display if it's empty or contains whitespace or a
/// shell-meaningful character, so a command preview can't misrepresent where
/// one argument ends and the next begins. Mirrors the display-only quoting
/// already used for the same purpose in `apps/rocm` (e.g.
/// `therock.rs::quote_display_arg`).
fn quote_display_arg(value: &str) -> String {
    if value.is_empty()
        || value
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '[' | ']' | '(' | ')' | '&' | ';' | '|'))
    {
        format!("\"{}\"", value.replace('"', "\\\""))
    } else {
        value.to_owned()
    }
}

/// Join `args` into a single display string, quoting any argument that needs
/// it so an approval preview or job-console title unambiguously shows where
/// each argument begins and ends.
pub fn display_args(args: &[String]) -> String {
    args.iter()
        .map(|a| quote_display_arg(a))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exe_label_strips_unix_and_windows_paths() {
        assert_eq!(exe_label("/usr/local/bin/rocm"), "rocm");
        assert_eq!(exe_label("C:\\tools\\rocm.exe"), "rocm.exe");
        assert_eq!(exe_label("rocm"), "rocm");
    }

    #[test]
    fn resolve_exe_is_never_empty() {
        assert!(!resolve_exe().is_empty());
    }

    #[test]
    fn display_args_passes_through_plain_values() {
        let args = vec!["--channel".to_string(), "release".to_string()];
        assert_eq!(display_args(&args), "--channel release");
    }

    #[test]
    fn display_args_quotes_values_with_spaces() {
        let args = vec!["--prefix".to_string(), "/mnt/my folder".to_string()];
        assert_eq!(display_args(&args), "--prefix \"/mnt/my folder\"");
    }

    #[test]
    fn display_args_escapes_embedded_quotes() {
        let args = vec!["say \"hi\"".to_string()];
        assert_eq!(display_args(&args), "\"say \\\"hi\\\"\"");
    }

    #[test]
    fn display_args_quotes_empty_value() {
        let args = vec!["--tag".to_string(), String::new()];
        assert_eq!(display_args(&args), "--tag \"\"");
    }

    #[test]
    fn display_args_quotes_shell_metacharacters() {
        let args = vec!["a&b".to_string(), "c;d".to_string(), "e|f".to_string()];
        assert_eq!(display_args(&args), "\"a&b\" \"c;d\" \"e|f\"");
    }
}
