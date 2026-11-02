// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Windows install scenarios must refuse a Linux/WSL host before any CLI
//! call, even when an explicit cucumber filter bypasses the platform tags.
//! Check the actual step wiring without ever invoking a driver installer.

#[path = "e2e/host_guard.rs"]
mod host_guard;

#[test]
fn windows_driver_install_steps_check_the_native_host_before_running_the_cli() {
    // A Windows checkout may use CRLF line endings, so match on LF text.
    let source = include_str!("e2e/driver_steps.rs").replace("\r\n", "\n");
    for name in [
        "ask_windows_driver_install",
        "approve_windows_driver_install",
    ] {
        let start = source
            .find(&format!("async fn {name}("))
            .expect("the Windows install step exists");
        let body = source[start..]
            .split_once("\n}")
            .expect("the step has a closing brace")
            .0;
        let command = body.find("crate::run_rocm").expect("the step runs the CLI");
        // The helper panics unless the build is native Windows (WSL included
        // in the refusal). Pin the call before the CLI runs in the real step.
        // Comment lines are dropped first, so a comment naming the call cannot
        // satisfy the check.
        let code_before_cli: String = body[..command]
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code_before_cli.contains("require_native_windows();"),
            "{name} must require native Windows before invoking the CLI:\n{body}"
        );
    }
}

#[test]
fn the_host_check_accepts_only_windows_and_refuses_linux_and_wsl() {
    host_guard::require_os_is_windows("windows");
    for os in ["linux", "macos", ""] {
        assert!(
            std::panic::catch_unwind(|| host_guard::require_os_is_windows(os)).is_err(),
            "{os:?} must be refused"
        );
    }
}
