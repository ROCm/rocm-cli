// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host checks steps run before relying on an OS-specific premise: the
//! Windows driver-install steps before any CLI call, and the KFD topology step
//! before reading sysfs.
//!
//! Kept free of the cucumber world so `driver_host_guard.rs` can include it and
//! test each check with the host facts as arguments. The `e2e` target has
//! `harness = false`, so a test inside it would never run.

/// Panic unless `os` names native Windows. WSL reports `linux`.
pub fn require_os_is_windows(os: &str) {
    assert_eq!(
        os, "windows",
        "this driver-install step requires native Windows"
    );
}

/// Proof that this host was checked and is bare-metal Linux. Only
/// [`this_host_is_bare_metal_linux`] makes one, so a function that takes one
/// (`kfd_gpu_node_versions` in `examine_steps.rs`) cannot be called without
/// the check having passed. That parameter is the whole enforcement: no test
/// checks that a step calls the check, and dropping the parameter is a
/// signature change a reviewer sees.
pub struct BareMetalLinux(());

/// The check behind [`this_host_is_bare_metal_linux`], taking the host facts
/// as arguments so it can be tested on any host. Private, so no step can hand
/// it facts of its own.
fn bare_metal_linux(os: &str, is_wsl: bool) -> BareMetalLinux {
    assert!(
        os == "linux" && !is_wsl,
        "this step is tagged @requires-os:linux @requires-bare-metal, but runs on \
         os={os:?} wsl={is_wsl}: its premise exists only on bare-metal Linux"
    );
    BareMetalLinux(())
}

/// Panic unless this host is Linux and not WSL2, as the harness's capability
/// probe reports it. For a step whose scenario is tagged
/// `@requires-os:linux @requires-bare-metal`: reaching it anywhere else means
/// the tags were bypassed (an explicit filter) or dropped, and the step would
/// otherwise find no premise and pass without asserting.
pub fn this_host_is_bare_metal_linux() -> BareMetalLinux {
    let cap = e2e_cucumber::capability::host_capability();
    bare_metal_linux(&cap.os_family, cap.is_wsl)
}

// Run by `driver_host_guard.rs`, which includes this file and has the test
// harness; the `e2e` target has none, so they never run there.
#[cfg(test)]
mod tests {
    #[test]
    fn the_bare_metal_linux_check_accepts_only_bare_metal_linux() {
        super::bare_metal_linux("linux", false);
        for (os, is_wsl) in [("linux", true), ("windows", false), ("macos", false)] {
            assert!(
                std::panic::catch_unwind(|| super::bare_metal_linux(os, is_wsl)).is_err(),
                "{os:?} wsl={is_wsl} must be refused"
            );
        }
        // Not called: it probes the real binary. This line only keeps the
        // function referenced, so this target compiles it without a dead-code
        // warning; it tests nothing about who calls it.
        let _: fn() -> super::BareMetalLinux = super::this_host_is_bare_metal_linux;
    }
}
