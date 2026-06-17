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
        .ok().map_or_else(|| "rocm".to_string(), |p| p.to_string_lossy().into_owned())
}

/// Short, human-readable basename of a resolved command, for approval previews.
pub fn exe_label(cmd: &str) -> &str {
    cmd.rsplit(['/', '\\']).next().unwrap_or(cmd)
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
}
