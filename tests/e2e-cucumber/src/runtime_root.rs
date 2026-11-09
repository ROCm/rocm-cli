// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/// The field `rocm examine` prints for the managed runtime's install folder.
const ACTIVE_RUNTIME_ROOT_FIELD: &str = "active_runtime_root";

/// How many times the `runtimes/wheel` pair of path segments appears in `root`.
///
/// Compared segment by segment, not as a substring, so a folder that merely ends
/// in `runtimes` or starts with `wheel` is not counted. Both separators are
/// accepted because the root is printed in the host's own form.
fn runtimes_wheel_count(root: &str) -> usize {
    let segments: Vec<&str> = root
        .split(['/', '\\'])
        .filter(|segment| !segment.is_empty())
        .collect();
    segments
        .windows(2)
        .filter(|pair| pair[0] == "runtimes" && pair[1] == "wheel")
        .count()
}

/// The active runtime's folder as `rocm examine` states it, if it states one.
///
/// Only a line whose name is exactly `active_runtime_root` counts. `examine`
/// prints it only while a managed runtime is active, so `None` means there is
/// nothing to judge. A sibling such as `active_runtime_pip_cache_dir` shares the
/// prefix but is another fact and is not read here.
fn active_runtime_root(examine_output: &str) -> Option<&str> {
    examine_output.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(':')?;
        (name == ACTIVE_RUNTIME_ROOT_FIELD).then(|| value.trim())
    })
}

/// Judge the active managed runtime's folder from `rocm examine` output.
///
/// Re-provisioning inside an existing runtime once produced a folder holding
/// `runtimes/wheel` twice (`runtimes/wheel/<key>/runtimes/wheel/<key>`). A healthy
/// folder holds it at most once. Returns the folder when it is fine, and the
/// reason when it is not, including when the output states no folder at all,
/// which a scenario that made a runtime active must never see.
///
/// # Errors
///
/// When no `active_runtime_root` line is present, its value is empty, or the
/// folder is nested.
pub fn check_runtime_root_not_nested(examine_output: &str) -> Result<&str, String> {
    let Some(root) = active_runtime_root(examine_output) else {
        return Err(format!(
            "no '{ACTIVE_RUNTIME_ROOT_FIELD}:' line in the examine output, so there is no \
             managed runtime folder to judge"
        ));
    };
    if root.is_empty() {
        return Err(format!(
            "the examine output's '{ACTIVE_RUNTIME_ROOT_FIELD}:' line names no folder"
        ));
    }
    let count = runtimes_wheel_count(root);
    if count > 1 {
        return Err(format!(
            "the managed runtime folder is recursively nested: `runtimes/wheel` appears \
             {count} times in {root}"
        ));
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::check_runtime_root_not_nested;

    // The shape of an `active_runtime_root` a healthy managed runtime printed on a
    // Linux and a Windows GPU lane: `<data dir>/runtimes/wheel/<runtime key>`.
    const LINUX_ROOT: &str = "/home/runner/_work/rocm-cli/e2e-prewarm-multi-arch-v2/data/runtimes/wheel/release-wheel-multi-arch-7-14-1-0b5cf9d493993efb";
    const WINDOWS_ROOT: &str = r"C:\runner\_work\rocm-cli\e2e-prewarm-multi-arch-v2\data\runtimes\wheel\release-wheel-multi-arch-7-14-1-f1ec7c6b4e815e9f";

    fn examine_output(root_line: &str) -> String {
        format!(
            "ROCm setup check\n  AMD GPU detected\n\nDetails\nrocm examine\n  os: linux\n  \
             managed_runtimes: 1\n  active_runtime_status: ready\n{root_line}\n  \
             active_runtime_version: 7.14.1\n"
        )
    }

    #[test]
    fn a_flat_linux_root_passes_and_is_returned() {
        let output = examine_output(&format!("  active_runtime_root: {LINUX_ROOT}"));
        assert_eq!(check_runtime_root_not_nested(&output), Ok(LINUX_ROOT));
    }

    #[test]
    fn a_flat_windows_root_passes_with_its_drive_colon_intact() {
        let output = examine_output(&format!("  active_runtime_root: {WINDOWS_ROOT}"));
        assert_eq!(check_runtime_root_not_nested(&output), Ok(WINDOWS_ROOT));
    }

    #[test]
    fn a_root_holding_runtimes_wheel_twice_fails_naming_it() {
        let nested = format!("{LINUX_ROOT}/runtimes/wheel/release-wheel-other");
        let output = examine_output(&format!("  active_runtime_root: {nested}"));
        let error = check_runtime_root_not_nested(&output).unwrap_err();
        assert!(error.contains("recursively nested"), "{error}");
        assert!(error.contains("2 times"), "{error}");
        assert!(error.contains(&nested), "{error}");
    }

    #[test]
    fn a_nested_windows_root_fails_in_backslash_form() {
        let nested = format!(r"{WINDOWS_ROOT}\runtimes\wheel\release-wheel-other");
        let output = examine_output(&format!("  active_runtime_root: {nested}"));
        assert!(check_runtime_root_not_nested(&output).is_err());
    }

    #[test]
    fn a_nested_root_fails_when_the_separators_are_mixed() {
        let nested = r"C:\data\runtimes\wheel\key/runtimes/wheel/key2";
        let output = examine_output(&format!("  active_runtime_root: {nested}"));
        assert!(check_runtime_root_not_nested(&output).is_err());
    }

    #[test]
    fn folders_that_only_resemble_the_pair_are_not_counted() {
        let root = "/data/old-runtimes/wheel-cache/runtimes/wheel/key";
        let output = examine_output(&format!("  active_runtime_root: {root}"));
        assert_eq!(check_runtime_root_not_nested(&output), Ok(root));
    }

    #[test]
    fn a_missing_field_fails_loudly() {
        let output = "ROCm setup check\n  AMD GPU detected\n  managed_runtimes: 0\n";
        let error = check_runtime_root_not_nested(output).unwrap_err();
        assert!(error.contains("active_runtime_root"), "{error}");
    }

    #[test]
    fn an_empty_value_fails_loudly() {
        let output = examine_output("  active_runtime_root:");
        let error = check_runtime_root_not_nested(&output).unwrap_err();
        assert!(error.contains("names no folder"), "{error}");
    }

    #[test]
    fn only_the_exact_field_is_read() {
        // A nested path in a sibling field, or in text that mentions the old label,
        // is neither read as the root nor allowed to stand in for a missing one.
        let nested = format!("{LINUX_ROOT}/runtimes/wheel/x");
        let output = format!(
            "  active_runtime_pip_cache_dir: {nested}\n  Folder: {nested}\n  active_runtime_root: {LINUX_ROOT}\n"
        );
        assert_eq!(check_runtime_root_not_nested(&output), Ok(LINUX_ROOT));

        let only_siblings =
            format!("  active_runtime_pip_cache_dir: {LINUX_ROOT}\n  Folder: {LINUX_ROOT}\n");
        assert!(check_runtime_root_not_nested(&only_siblings).is_err());
    }
}
