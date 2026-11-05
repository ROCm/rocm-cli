// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The closed catalog of `rocm fix` remediations.
//!
//! `FixRecipe`, the per-platform class table each recipe carries (`Platform`
//! and the `PRINT_ON_*`/`AUTO_ON_*` shorthands), `RECIPES` itself, the lookups
//! by fix-id (`find_recipe`, `is_catalog_id`, `class_on`, `needs_on`), and the
//! test-only oracles that `diagnose.rs` and the e2e crate use to hold their
//! own copies of a recipe to this one. `RECIPES` together with `diagnose.rs`'s
//! checkers is the authoritative failure catalog; rendering lives in
//! `render.rs`, the published JSON contract in `manifest.rs`.

use super::consent::current_os;
use super::options::{FixClass, FixOptions};
use super::runners;

/// Shared verbatim by this recipe's own note and `diagnose.rs`'s
/// `check_18_comgr_conflict` evidence, which states the same claim in its own
/// words at the point the real paths are known. A second statement of one
/// claim is a second thing to keep correct, and the two had already drifted
/// apart in wording before they shared this constant.
pub(crate) const COMGR_CONFLICT_NEITHER_OPTION_RECOMMENDED: &str = "Neither option is recommended over the other: which is right depends on which stack you mean to keep, and removing the wrong one breaks a working environment.";

/// One operating system a recipe applies on, and what it does there.
///
/// Scope and class are held together rather than in two parallel lists because
/// they are one fact: a fix applies *on Linux as print-only* and *on Windows as
/// auto*. Splitting them is what let `fix-2`'s flat flag contradict its own
/// platform-split runner.
pub(super) struct Platform {
    pub(super) os: &'static str,
    pub(super) class: FixClass,
    /// The argument that moves [`FixClass::NeedsArgument`] to acting. `None`
    /// for every other class; the pairing is held by
    /// `a_needs_argument_platform_names_the_argument`.
    pub(super) needs: Option<&'static str>,
}

/// Shorthand for the common case: the same class on a platform, needing nothing.
pub(super) const fn on(os: &'static str, class: FixClass) -> Platform {
    Platform {
        os,
        class,
        needs: None,
    }
}

const PRINT_ON_LINUX_WINDOWS_AND_WSL: &[Platform] = &[
    on("linux", FixClass::PrintOnly),
    on("windows", FixClass::PrintOnly),
    on("wsl", FixClass::PrintOnly),
];
const PRINT_ON_LINUX: &[Platform] = &[on("linux", FixClass::PrintOnly)];
const PRINT_ON_WINDOWS: &[Platform] = &[on("windows", FixClass::PrintOnly)];
/// Every WSL recipe. All print-only: each one installs a package with sudo,
/// edits loader configuration, or belongs to the Windows host, and none of that
/// meets the bar an auto-applied fix has to clear.
const PRINT_ON_WSL: &[Platform] = &[on("wsl", FixClass::PrintOnly)];
/// Both Linux families, for a problem that is about neither the `amdgpu` module
/// nor the Windows host driver and so is real on either.
const PRINT_ON_LINUX_AND_WSL: &[Platform] = &[
    on("linux", FixClass::PrintOnly),
    on("wsl", FixClass::PrintOnly),
];
const AUTO_ON_LINUX: &[Platform] = &[on("linux", FixClass::Auto)];
const AUTO_ON_LINUX_WINDOWS_AND_WSL: &[Platform] = &[
    on("linux", FixClass::Auto),
    on("windows", FixClass::Auto),
    on("wsl", FixClass::Auto),
];

/// `fix-2-unset-override`. `run_unset_override` splits by platform: the Windows
/// arm takes `FixOptions` and does the dry-run/confirm dance, while
/// `run_unset_override_linux` takes no options, reports where the override is
/// set, and returns without touching a dotfile. WSL runs that same Linux arm.
const PRINT_ON_LINUX_AND_WSL_AUTO_ON_WINDOWS: &[Platform] = &[
    on("linux", FixClass::PrintOnly),
    on("windows", FixClass::Auto),
    on("wsl", FixClass::PrintOnly),
];

/// `fix-9-igpu-dgpu`. Both arms short-circuit when no device index is given:
/// they print the query that identifies the discrete GPU and return, so there
/// is nothing for `--dry-run` to preview and no prompt to refuse.
const NEEDS_DEVICE_INDEX: &[Platform] = &[
    Platform {
        os: "linux",
        class: FixClass::NeedsArgument,
        needs: Some("--device-index"),
    },
    Platform {
        os: "windows",
        class: FixClass::NeedsArgument,
        needs: Some("--device-index"),
    },
];

/// `info-20-gpu-no-rocm-support`. Not "wsl": WSL2 reports no GPU topology, so
/// the checker that produces this id never fires there and `rocm fix` would be
/// claiming an answer for a host it was never asked about.
const DIAGNOSE_ONLY_ON_LINUX_AND_WINDOWS: &[Platform] = &[
    on("linux", FixClass::DiagnoseOnly),
    on("windows", FixClass::DiagnoseOnly),
];

/// A remediation recipe keyed by the stable `fix-id`.
pub(super) struct FixRecipe {
    pub(super) fix_id: &'static str,
    pub(super) title: &'static str,
    pub(super) rationale: &'static str,
    pub(super) commands: &'static [&'static str],
    pub(super) needs_sudo: bool,
    pub(super) needs_reboot: bool,
    pub(super) needs_relogin: bool,
    pub(super) verify: &'static str,
    /// A second verify line for Linux and WSL, printed as `Verify (Linux, WSL):`
    /// after `verify`. `None` when `verify` runs the same on every OS the
    /// recipe applies on. `verify` is shown to Windows users, so it cannot
    /// name a tool that only exists on Linux (and cannot chain with `&&`);
    /// this field carries the Linux check without that limit. It is still
    /// printed on Windows, labelled, as the `# Linux:` command lines are, so
    /// the text rule for Windows lines applies to it too.
    pub(super) verify_linux: Option<&'static str>,
    pub(super) notes: &'static [&'static str],
    /// Where to read more about this failure. `None` for most entries today --
    /// an empty answer beats a guessed one, which would send a user somewhere
    /// wrong with an air of authority.
    pub(super) doc_url: Option<&'static str>,
    /// Every operating system this applies on, and what it does on each.
    pub(super) applies_on: &'static [Platform],
    pub(super) runner: Option<fn(&FixOptions) -> i32>,
}

impl FixRecipe {
    /// What this recipe does on `os`, or `None` when it does not apply there.
    pub(super) fn platform(&self, os: &str) -> Option<&Platform> {
        self.applies_on.iter().find(|p| p.os == os)
    }

    /// What this recipe does on the running host. [`FixClass::PrintOnly`] when
    /// it does not apply here at all — callers reach the OS gate in [`apply`]
    /// before the class is ever acted on, and understating is the safe default.
    pub(super) fn class_here(&self) -> FixClass {
        self.platform(current_os())
            .map_or(FixClass::PrintOnly, |p| p.class)
    }

    /// The operating systems this applies on, in catalog order.
    pub(super) fn os_scope(&self) -> Vec<&'static str> {
        self.applies_on.iter().map(|p| p.os).collect()
    }
}

/// The recipe registry. Mirrors the diagnosis catalog; only the small, safe
/// fixes carry a `runner`, and what each one does can differ by platform.
pub(super) const RECIPES: &[FixRecipe] = &[
    FixRecipe {
        fix_id: "fix-1-arch",
        title: "GPU gfx target not in framework arch list",
        rationale: "Your GPU's gfx target is not in the framework wheel's compiled kernel list. Re-install the framework from an index that includes this gfx, OR rebuild llama.cpp with AMDGPU_TARGETS=<gfx>.",
        commands: &[
            "# PyTorch (Linux): a nightly often carries kernels a release has not shipped yet.",
            "# Pick the nightly for the ROCm major you have, not an older one.",
            "pip uninstall -y torch torchvision torchaudio",
            "pip install --pre torch torchvision torchaudio \\",
            "  --index-url https://download.pytorch.org/whl/nightly/rocm7.14",
            "# PyTorch (Windows): use TheRock's per-gfx wheels (https://github.com/ROCm/TheRock).",
            "# llama.cpp:",
            "# cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<your_gfx_target>",
            "# cmake --build build -j",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())\"",
        verify_linux: None,
        notes: &[
            "TheRock per-gfx wheels are the recommended fallback when the official pytorch index does not yet cover your gfx (and the only first-party option on Windows AMD).",
            "HSA_OVERRIDE_GFX_VERSION is NOT the right fix here -- it papers over the mismatch and risks page faults at runtime.",
        ],
        doc_url: Some(
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
        ),
        applies_on: PRINT_ON_LINUX_WINDOWS_AND_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-2-unset-override",
        title: "Unset HSA_OVERRIDE_GFX_VERSION",
        rationale: "HSA_OVERRIDE_GFX_VERSION is set, but your GPU now has a native wheel. The override hides the real gfx and causes page faults / OUT_OF_REGISTERS at runtime.",
        commands: &[
            "# Linux:",
            "unset HSA_OVERRIDE_GFX_VERSION",
            "# Then remove the line from ~/.bashrc / ~/.zshrc / ~/.profile.",
            "# Windows (PowerShell):",
            "[Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION',$null,'User')",
            "# Or remove via System Properties -> Environment Variables.",
            "# This only affects NEW shells; close and reopen this terminal before running the verify command.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        // One line for cmd and Windows PowerShell (bash runs it too): it
        // prints the override's value, or OK_UNSET. It needs `python`, which
        // a Windows Python install and a virtual environment provide but a
        // stock Ubuntu does not (that has `python3` only), so Linux and WSL get
        // their own line below.
        verify: "python -c \"import os; print(os.environ.get('HSA_OVERRIDE_GFX_VERSION') or 'OK_UNSET')\"",
        // The same answer from the shell's own variable, with no Python and
        // no external command, so it runs on a bare system. It has no `||`:
        // this line is printed on Windows too, where the text rule refuses it.
        verify_linux: Some("echo \"${HSA_OVERRIDE_GFX_VERSION:-OK_UNSET}\""),
        notes: &[
            "Linux, WSL: run the `Verify (Linux, WSL)` line. It reads the shell's own variable and needs no Python. The plain `Verify` line runs `python`, which a Windows Python install and a virtual environment provide, but a stock Ubuntu install has only `python3`, so the plain line fails there with `command not found` even after the override is cleared.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_LINUX_AND_WSL_AUTO_ON_WINDOWS,
        runner: Some(runners::run_unset_override),
    },
    FixRecipe {
        fix_id: "fix-3-rocm-kernel",
        title: "ROCm/distro/kernel triple unsupported",
        rationale: "ROCm is installed but your kernel/distro combination is outside the supported matrix. Match the kernel to the matrix first, then rerun `rocm install driver --dkms --yes` (documented distros only, which includes Ubuntu 26.04).",
        commands: &[
            "# Cross-check the live AMD matrix before changing anything:",
            "#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
            "# Common fix on Ubuntu: install the HWE kernel that matches your ROCm release, then reboot.",
        ],
        needs_sudo: false,
        needs_reboot: true,
        needs_relogin: false,
        verify: "lsmod | grep amdgpu && rocminfo >/dev/null 2>&1",
        verify_linux: None,
        notes: &[],
        doc_url: Some(
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
        ),
        applies_on: PRINT_ON_LINUX,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-4-render-group",
        title: "Add user to render/video groups",
        rationale: "The current user can't open /dev/kfd because they aren't in the render group. Adding the user is the safe, standard fix.",
        commands: &["sudo usermod -a -G render,video \"$USER\""],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: true,
        verify: "groups | tr ' ' '\\n' | grep -E '^(render|video)$' && rocminfo | head -n 5",
        verify_linux: None,
        notes: &[],
        doc_url: None,
        applies_on: AUTO_ON_LINUX,
        runner: Some(runners::run_render_group),
    },
    FixRecipe {
        fix_id: "fix-5-amdgpu-load",
        title: "Load amdgpu (and clear any blacklist)",
        rationale: "The amdgpu kernel module is not loaded. Check /etc/modprobe.d for a blacklist entry, regenerate the initramfs, and modprobe.",
        commands: &[
            "grep -RIl 'blacklist amdgpu' /etc/modprobe.d /usr/lib/modprobe.d 2>/dev/null || true",
            "sudo $EDITOR <file shown above>     # remove the blacklist line",
            "sudo update-initramfs -u            # Debian/Ubuntu",
            "sudo dracut -f                      # Fedora/RHEL",
            "sudo modprobe amdgpu",
        ],
        needs_sudo: true,
        needs_reboot: true,
        needs_relogin: false,
        verify: "lsmod | grep amdgpu && rocminfo | head -n 5",
        verify_linux: None,
        notes: &[
            "If Secure Boot is enabled and amdgpu still won't load, the DKMS module isn't signed. Either sign it with mokutil or disable Secure Boot in firmware.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_LINUX,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-6-path",
        title: "Add the ROCm/HIP bin directory to PATH",
        rationale: "Linux: ROCm is installed but its bin directory isn't on PATH, so `rocminfo` / `hipcc` aren't visible to the shell. Windows: the HIP SDK is installed but its bin directory isn't on the User PATH, so `hipInfo.exe` and the runtime DLLs can't be found.",
        commands: &[
            "# Linux:",
            "echo 'export PATH=\"/opt/rocm/bin:$PATH\"' >> ~/.bashrc",
            "# Windows (PowerShell):",
            "[Environment]::SetEnvironmentVariable('PATH', 'C:\\Program Files\\AMD\\ROCm\\<version>\\bin;' + [Environment]::GetEnvironmentVariable('PATH', 'User'), 'User')",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        // The line every OS can run, Windows included: `hipcc` ships with the
        // HIP SDK there. Linux gets its own line below.
        verify: "hipcc --version",
        // `rocminfo` prints the GPU agents and needs no compiler, so it also
        // passes on a runtime-only Linux install, which has no `hipcc`.
        verify_linux: Some("rocminfo | head -n 5"),
        // The Windows command above is the manual-paste twin of what
        // `path_export_windows` actually runs -- same scope, same prepend
        // ordering -- so a user who pastes it and a user who lets `rocm fix`
        // do it end up with the same User PATH. These notes say why it is not
        // the `setx` one-liner a Windows user would otherwise reach for.
        notes: &[
            "Windows: the PowerShell line reads and rewrites the User scope only, and puts the SDK's bin dir first -- the same write `rocm fix fix-6-path` performs. Do not substitute `setx PATH \"%PATH%;...\"`: setx silently truncates the value at 1024 characters while still exiting 0, and %PATH% expands to the combined user+system PATH, so that form would copy every system entry into your User PATH.",
            "Windows: a PATH change only takes effect in NEW shells -- close this terminal and reopen it before running the verify command.",
            "Linux, WSL: run the `Verify (Linux, WSL)` line. `rocminfo` prints the GPU and needs no compiler, so it passes on a runtime-only install. The plain `Verify` line runs `hipcc --version`, which exists on Windows (HIP SDK) and on a Linux install that has the compiler, and fails on a runtime-only Linux install even when PATH is fixed, because that install has no `hipcc`.",
        ],
        doc_url: None,
        applies_on: AUTO_ON_LINUX_WINDOWS_AND_WSL,
        runner: Some(runners::run_path_export),
    },
    FixRecipe {
        fix_id: "fix-7-stale-repos",
        title: "Quarantine duplicate AMD repos",
        rationale: "More than one ROCm/AMDGPU repo file exists. The package manager is mixing versions; quarantine the extras before reinstalling.",
        commands: &[
            "# Run only the line matching your package manager:",
            "ls /etc/apt/sources.list.d/ | grep -iE 'rocm|amdgpu|radeon' || true   # Debian/Ubuntu",
            "ls /etc/yum.repos.d/ | grep -iE 'rocm|amdgpu|radeon' || true          # RHEL/Fedora/Rocky/Alma/Oracle",
            "ls /etc/zypp/repos.d/ | grep -iE 'rocm|amdgpu|radeon' || true         # SLES/openSUSE",
            "# For each duplicate file:",
            "sudo mv /etc/apt/sources.list.d/<file>.list /etc/apt/sources.list.d/<file>.list.bak   # or the yum/zypp equivalent",
            "# Run only the line matching your package manager:",
            "sudo apt update      # Debian/Ubuntu",
            "sudo dnf makecache   # RHEL/Fedora/Rocky/Alma/Oracle",
            "sudo zypper refresh  # SLES/openSUSE",
        ],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: false,
        verify: "# Run only the line matching your package manager:\n\
                 bash -c 'set -o pipefail; sudo apt update 2>&1 | tail -n 20'      # Debian/Ubuntu\n\
                 bash -c 'set -o pipefail; sudo dnf makecache 2>&1 | tail -n 20'   # RHEL/Fedora/Rocky/Alma/Oracle\n\
                 bash -c 'set -o pipefail; sudo zypper refresh 2>&1 | tail -n 20'  # SLES/openSUSE",
        verify_linux: None,
        notes: &[],
        doc_url: None,
        applies_on: PRINT_ON_LINUX,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-8-wheel-rocm",
        title: "Reinstall the framework against the system ROCm/HIP major",
        rationale: "The framework's bundled HIP version doesn't match the system ROCm (Linux) or HIP SDK (Windows). libamdhip64.so.X / amdhip64_X.dll load failures are the usual signal.",
        commands: &[
            "pip uninstall -y torch torchvision torchaudio",
            "# Linux: install the index for the ROCm major `rocm examine` reports:",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.14",
            "# ROCm 10 has no released PyTorch index yet; it is on the nightly channel:",
            "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm10.0",
            "# Windows: use TheRock's wheels matching your HIP SDK major:",
            "#   https://github.com/ROCm/TheRock",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "python -c \"import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())\"",
        verify_linux: None,
        notes: &[],
        doc_url: Some(
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
        ),
        applies_on: PRINT_ON_LINUX_WINDOWS_AND_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-9-igpu-dgpu",
        title: "Hide the iGPU with HIP_VISIBLE_DEVICES",
        rationale: "Both an APU iGPU and a discrete AMD GPU are visible. Pin the runtime to the dGPU so the iGPU doesn't destabilise it.",
        commands: &[
            "# Linux:",
            "rocminfo | grep -E 'Agent |Marketing|gfx'   # find the dGPU index",
            "export HIP_VISIBLE_DEVICES=<dGPU-index>",
            "# Windows:",
            "& \"$env:HIP_PATH\\bin\\hipInfo.exe\" | Select-String \"device#|Name\"",
            "setx HIP_VISIBLE_DEVICES <dGPU-index>",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "python -c \"import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))\"",
        verify_linux: None,
        notes: &[
            "Pass --device-index N to persist the env var; without it, this fix only prints the rocminfo / hipInfo query so you can identify N.",
        ],
        doc_url: None,
        applies_on: NEEDS_DEVICE_INDEX,
        runner: Some(runners::run_hip_visible_devices),
    },
    FixRecipe {
        fix_id: "fix-10-container",
        title: "Re-launch the container with AMD devices passed through",
        rationale: "The container can't see /dev/kfd or /dev/dri/renderD*. Pass the devices and the host's render group via the runtime flags.",
        commands: &[
            "docker run --rm -it \\",
            "  --device=/dev/kfd \\",
            "  --device=/dev/dri \\",
            "  --group-add render \\",
            "  --security-opt seccomp=unconfined \\",
            "  --shm-size=8g \\",
            "  rocm/pytorch:latest",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "rocminfo | head -n 5",
        verify_linux: None,
        notes: &[
            "Rootless podman additionally needs `--userns=keep-id` and a host user that is in the render group; podman maps it through.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_LINUX,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-11-iommu",
        title: "Add iommu=pt to the kernel command line",
        rationale: "Multi-GPU jobs hang when the IOMMU is in the default 'on' mode with translation; pass-through mode fixes the hang. This requires editing GRUB and rebooting; we will not do that for you.",
        commands: &[
            "cat /proc/cmdline",
            "sudo $EDITOR /etc/default/grub        # add iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT",
            "sudo update-grub                       # Debian/Ubuntu",
            "sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora/RHEL",
            "# Reboot, then retry the multi-GPU workload.",
        ],
        needs_sudo: true,
        needs_reboot: true,
        needs_relogin: false,
        verify: "cat /proc/cmdline | grep -o 'iommu=\\w*'",
        verify_linux: None,
        notes: &[],
        doc_url: None,
        applies_on: PRINT_ON_LINUX,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-12-installer",
        title: "Reset repo-native install state and reinstall",
        rationale: "The repo-native package-manager install left a half-configured DKMS / repo state. Clear the broken package state, quarantine the ROCm/AMDGPU repo files found on this host, let the package manager forget the broken state, then reinstall via rocm-cli's own repo-native flow.",
        commands: &[
            "# Applies only where `rocm install driver --dkms` plans a reinstall (AMD-documented",
            "# distros, Ubuntu 26.04 included). On any other distro it plans nothing: skip the",
            "# purge/remove and reinstall per AMD's install docs.",
            "# Run only the block matching your package manager. On Debian/Ubuntu, run all",
            "# three lines even if an earlier one reports an error -- the purge must not",
            "# be skipped just because the half-configured state made `install -f` fail:",
            "sudo dpkg --configure -a      # Debian/Ubuntu",
            "sudo apt-get install -f       # Debian/Ubuntu",
            "sudo apt-get purge -y amdgpu-dkms   # Debian/Ubuntu",
            "sudo dnf remove -y amdgpu-dkms                                                             # RHEL/Fedora/Rocky/Alma/Oracle",
            "sudo zypper remove -y amdgpu-dkms                                                          # SLES/openSUSE",
            "# sudo mv /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/amdgpu.list.bak       # or the yum/zypp equivalent; quarantine, do not delete yet",
            "# Run only the line matching your package manager:",
            "sudo apt update      # Debian/Ubuntu",
            "sudo dnf makecache   # RHEL/Fedora/Rocky/Alma/Oracle",
            "sudo zypper refresh  # SLES/openSUSE",
            "rocm install driver --dkms --yes",
        ],
        needs_sudo: true,
        needs_reboot: true,
        needs_relogin: false,
        // See the matching comment on this fix's `verify` in diagnose.rs:
        // must query the targeted amdgpu-dkms package's installed status
        // (not a bare rocm/amdgpu name grep), confirm dkms reports the
        // module built, and check rocminfo's own exit code directly.
        verify: "(dpkg-query -W -f='${Status}' amdgpu-dkms 2>/dev/null | grep -q 'install ok installed' \
                 || rpm -q amdgpu-dkms >/dev/null 2>&1) \
                 && dkms status amdgpu 2>/dev/null | grep -q installed \
                 && rocminfo >/dev/null 2>&1",
        verify_linux: None,
        notes: &[
            "If the purge/remove step warns it will take unrelated packages with it, stop and resolve those by hand before continuing.",
            "The `rocm install driver --dkms --yes` line and the verify below apply only on AMD-documented distros; `rocm install driver --dkms --yes` exits non-zero where it plans nothing (undocumented distros). There, verify with `rocminfo` alone.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_LINUX,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-13-hip-sdk-missing",
        title: "Install the AMD HIP SDK for Windows",
        rationale: "Your framework links against HIP but the HIP SDK isn't installed on this host. The runtime DLLs (amdhip64_X.dll, hipblas.dll, hsa-runtime64.dll) and hipInfo.exe ship inside the SDK installer.",
        commands: &[
            "# Download and install the HIP SDK (matched to your framework's HIP major):",
            "#   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html",
            "# After install, reopen the shell so HIP_PATH and PATH pick up the new install.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "powershell -NoProfile -Command \"& (Join-Path ([Environment]::GetEnvironmentVariable('HIP_PATH')) 'bin\\hipInfo.exe') | Select-Object -First 5\"",
        verify_linux: None,
        notes: &[
            "If you only need PyTorch on Windows AMD and don't need the C/C++ HIP toolchain, the TheRock wheels bundle their own HIP runtime and may not require a system HIP SDK install.",
        ],
        doc_url: Some("https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"),
        applies_on: PRINT_ON_WINDOWS,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-14-adrenalin-too-old",
        title: "Update the Adrenalin / kernel-mode driver",
        rationale: "The HIP SDK is installed but the AMD kernel-mode driver (Adrenalin / Adrenalin Pro) is older than the SDK release notes call out. The user-space SDK and the driver have to match.",
        commands: &[
            "# Cross-check the HIP SDK release notes for the exact driver pairing:",
            "#   https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html",
            "# Then download the matching driver from:",
            "#   https://www.amd.com/en/support",
            "# Reboot after the install for the kernel-mode driver to take effect.",
        ],
        needs_sudo: false,
        needs_reboot: true,
        needs_relogin: false,
        verify: "powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | Where-Object Name -match 'AMD|Radeon' | Select-Object -First 1).DriverVersion\"",
        verify_linux: None,
        notes: &[],
        doc_url: Some(
            "https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html",
        ),
        applies_on: PRINT_ON_WINDOWS,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-15-msvc-redist",
        title: "Install the MSVC 2015-2022 runtime redistributable",
        rationale: "Programs built with Visual C++ need the MSVC 2015-2022 runtime. That includes the HIP SDK's amdhip64_X.dll and the llama.cpp server that Lemonade installs. When vcruntime140.dll / vcruntime140_1.dll aren't on PATH, `import torch` fails with a missing-DLL error that points at vcruntime140_1.dll, not at the HIP runtime itself.",
        commands: &[
            "# Download and install (x64):",
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe",
            "# After the install, reopen the shell and re-run your import / hipInfo check.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "powershell -NoProfile -Command \"(Get-Item ([Environment]::GetFolderPath('System') + '\\vcruntime140.dll'),([Environment]::GetFolderPath('System') + '\\vcruntime140_1.dll') -ErrorAction Stop).VersionInfo | Select-Object FileName,FileVersion\"",
        verify_linux: None,
        notes: &[
            "If installing the redistributable still leaves a missing-DLL error, the failing DLL is probably amdhip64_X.dll itself; that points at fix-13-hip-sdk-missing rather than this fix. If the runtime is present but the program crashes with an access violation, see fix-21-msvc-redist-outdated.",
        ],
        doc_url: Some("https://aka.ms/vs/17/release/vc_redist.x64.exe"),
        applies_on: PRINT_ON_WINDOWS,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-16-vllm-oom",
        title: "vLLM ran the GPU out of memory at startup",
        rationale: "vLLM reserves a fixed fraction (~90%) of each GPU's TOTAL VRAM for its KV cache by default, regardless of the model size or how much is currently free. Two different faults surface the same OOM, and they need opposite responses. If the GPU is shared or already busy, that reservation collides with memory in use and lowering it (or moving to a less-busy GPU) is the fix. If the model genuinely does not fit in this GPU's VRAM, lowering the reservation only trades an earlier OOM for a later one -- use a smaller or quantized model instead (rocm-cli serves one model on a single GPU; it does not shard a model across GPUs).",
        commands: &[
            "# Case 1 -- shared/busy GPU (tenancy collision): lower the reservation,",
            "# or steer vLLM onto a less-busy device:",
            "rocm serve <model> --gpu-memory-utilization 0.5",
            "rocm serve <model> --gpu <index>",
            "# Case 2 -- the model genuinely does not fit: the reservation is not the",
            "# problem; pick a smaller or quantized model (single-GPU serving only).",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "rocm serve <model> <case-appropriate options above>   # re-run and watch for a clean startup",
        verify_linux: None,
        notes: &[
            "Only lower --gpu-memory-utilization when the GPU is shared or already busy; on a GPU dedicated to this server it cannot create the room a too-large model needs.",
            "This entry is keyword-matched from the error text: `rocm diagnose` cannot see per-GPU VRAM or tenancy, so pass the failure with --symptom (or arrive from the `rocm serve` failure note).",
        ],
        // Matches `check_16_vllm_oom`'s `["linux", "wsl"]` registration: WSL2 is
        // the supported way to serve from a Windows host and the failure mode is
        // just as real there, so the fix has to be runnable where the diagnosis
        // is reachable. Pinned by
        // `every_checker_platform_is_covered_by_its_recipe`.
        doc_url: None,
        applies_on: PRINT_ON_LINUX_AND_WSL,
        runner: None,
    },
    // The number is a stable handle, not a position: `fix-16` is held by the
    // vLLM out-of-memory entry above, so this one takes 17 rather than
    // colliding and forcing whichever lands second to rename a published id.
    FixRecipe {
        fix_id: "fix-17-torch-dlpack",
        title: "Restore the engine's pinned torch (torch-c-dlpack-ext loads the CUDA variant)",
        rationale: "vLLM's engine start aborts at import time when torch-c-dlpack-ext loads its CUDA prebuilt on a ROCm torch: it picks the variant from torch.cuda.is_available(), which is True on ROCm because PyTorch reuses the torch.cuda namespace for HIP, and it ships no ROCm variant. tvm_ffi imports it as OPTIONAL but guards only ImportError/AttributeError, while ctypes.CDLL raises OSError -- so the optional import kills the process. Both defects are upstream; nothing here is misconfigured. What you can change locally is the torch version: outside the 2.4-2.9 range there is no prebuilt to load, the extension raises the handled ImportError, and tvm_ffi falls back to its JIT path with a warning.",
        // Three labelled groups, because the steps run in three different places
        // and `print_recipe` renders them as one undifferentiated `$`-prefixed
        // list. Unlabelled, a user pasting the block wholesale is relying on
        // terminal stdin buffering to land the probes in the subshell -- and on
        // the reinstall NOT landing there, since it replaces the very
        // environment that shell is standing in.
        commands: &[
            "# --- step 1 of 3, in YOUR shell ---",
            "# Opens an INTERACTIVE subshell with the engine's environment active,",
            "# and does not return until you leave it. Run this line on its own.",
            "rocm engines shell vllm",
            "# --- step 2 of 3, INSIDE the subshell step 1 opened ---",
            "# Confirm the trigger before changing anything. It has to be the",
            "# ENGINE's interpreter, not the one on your PATH -- they are different",
            "# interpreters, and only the engine's decides this failure.",
            "python -c \"import torch; print(torch.__version__, torch.version.hip)\"",
            "python -c \"import importlib.metadata as m; print(m.version('torch-c-dlpack-ext'))\"",
            "# This entry applies ONLY when torch.version.hip is set, torch.__version__",
            "# is in the 2.4-2.9 range, and torch-c-dlpack-ext is installed. Outside",
            "# that range the extension raises a handled ImportError and this is not",
            "# the failure you are looking at. Then leave the subshell:",
            "exit",
            "# --- step 3 of 3, back in YOUR OWN shell ---",
            "# If all three held, put the engine's pinned torch back. Do NOT run",
            "# this from inside the subshell: it replaces the environment that",
            "# shell is standing in.",
            "rocm engines install vllm --reinstall",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "rocm serve <model> --engine vllm   # then `rocm services list --all` and `rocm services logs <service-id>` to confirm the import no longer aborts",
        verify_linux: None,
        notes: &[
            "Running vLLM on ROCm is not by itself a reason to apply this. torch-c-dlpack-ext arrives as a transitive dependency of tilelang, which vLLM pins, and it only misbehaves on the torch versions it ships prebuilts for.",
            "The usual way a runtime lands in the failing range is `rocm install sdk` being re-run after the engine was installed, which overwrites the engine's pinned torch. Reinstalling the engine puts the pin back.",
            "A service that failed at startup is hidden from a plain `rocm services list`; pass --all to recover its id.",
        ],
        doc_url: None,
        // Matches `check_17_torch_dlpack_cuda_variant`'s `["linux", "wsl"]`
        // registration. vLLM is supported on WSL2 and its engine runs the same
        // ELF import there, so the diagnosis the engine hint points at has to
        // be reachable there too. Pinned by
        // `every_checker_platform_is_covered_by_its_recipe`.
        applies_on: PRINT_ON_LINUX_AND_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-1-gpu-not-exposed",
        title: "Expose the GPU to the WSL distro (/dev/dxg)",
        rationale: "WSL reaches the GPU through /dev/dxg, provided by the Windows host driver via GPU-PV. Without that device nothing else in the ROCm stack can work, so this comes before any package or loader question. In a container the device has to be passed in explicitly; on a host it means the Windows driver or the WSL kernel needs attention.",
        commands: &[
            "# In a container, pass the device and the WSL libraries in:",
            "#   --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl",
            "# On a WSL host, update WSL and the Windows AMD driver, then:",
            "#   wsl --update",
            "#   wsl --shutdown",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "ls -l /dev/dxg",
        verify_linux: None,
        notes: &[
            "A container running on WSL2 reports itself as WSL but sees /dev/dxg only when it was started with the device. Check that before touching the Windows driver.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-2-dxcore-missing",
        title: "Restore the WSL DXCore libraries",
        rationale: "/usr/lib/wsl/lib holds the DXCore shims the ROCm runtime uses to talk to the Windows host driver. WSL mounts that directory itself, so a distro package manager can neither install nor repair it -- the fix is on the Windows side, plus a loader-path entry inside the distro.",
        commands: &[
            "# From Windows, refresh the WSL runtime that provides these libraries:",
            "#   wsl --update",
            "#   wsl --shutdown",
            "# Inside the distro, put them on the loader path:",
            "echo /usr/lib/wsl/lib | sudo tee /etc/ld.so.conf.d/wsl.conf",
            "sudo ldconfig",
        ],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: false,
        verify: "ls -l /usr/lib/wsl/lib/libdxcore.so && ldconfig -p | grep libdxcore",
        verify_linux: None,
        notes: &["apt cannot repair /usr/lib/wsl: it is a mount supplied by WSL, not a package."],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-3-rocdxg-missing",
        title: "Install ROCDXG in the WSL distro",
        rationale: "ROCDXG (librocdxg) is the ROCm-to-DXCore shim the WSL path runs on. It is a distro-side package, so unlike the driver and DXCore pieces this one is entirely in the user's hands.",
        commands: &[
            "rocm install driver",
            "# Then, once the plan looks right:",
            "#   rocm install driver --yes",
        ],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: false,
        verify: "ldconfig -p | grep librocdxg",
        verify_linux: None,
        notes: &[
            "Print-only on purpose: this downloads a .deb from a release page and installs it with sudo. `rocm install driver` prints the plan first so the URL and the package are reviewable before anything runs.",
            "The download is checked against a digest pinned for that ROCDXG release. To install a release rocm-cli has no digest for, set ROCM_CLI_ROCDXG_SHA256 to the one published with it.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-4-rocdxg-not-linked",
        title: "Refresh the linker cache so ROCDXG is loadable",
        rationale: "librocdxg is installed but absent from the linker cache, so the runtime will not find it at load time. Usually a missed `ldconfig` after a manual install.",
        commands: &["sudo ldconfig"],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: false,
        verify: "ldconfig -p | grep librocdxg",
        verify_linux: None,
        notes: &[
            "If ldconfig alone does not do it, the library landed outside the linker's search path: add that directory under /etc/ld.so.conf.d/ and re-run.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-5-distro-too-old",
        title: "Move to a distro release the WSL path supports",
        rationale: "Ubuntu 22.04 ships glibc 2.35, below the glibc 2.38 / GLIBCXX_3.4.32 floor every published Lemonade embeddable is linked against, so the engine cannot start there at all. This is a hard floor, not a recommendation.",
        commands: &[
            "# From Windows, install a supported distro alongside the current one:",
            "#   wsl --install -d Ubuntu-24.04",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "f=/etc/os-release; [ -e \"$f\" ] || f=/usr/lib/os-release; grep VERSION_ID \"$f\" 2>/dev/null | head -1",
        verify_linux: None,
        notes: &[
            "Distros install side by side, so the current one can stay until the new one is set up.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-6-host-driver-too-old",
        title: "Update the AMD driver on the Windows host",
        rationale: "Under WSL the GPU kernel-mode driver lives on the Windows host, not in the distro. When the distro-side plumbing is complete and ROCm still sees no GPU, the host driver is the remaining variable.",
        commands: &[
            "# On the Windows host, not in this distro:",
            "#   install a WSL-capable AMD Adrenalin driver, then `wsl --shutdown`.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "rocminfo | head -n 20",
        verify_linux: None,
        notes: &[
            "Nothing inside the distro can carry this out, which is why it prints rather than runs.",
            "The ROCm release and the Adrenalin release are paired; check the WSL install guide for the version that matches your ROCm.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-wsl-7-wsl1",
        title: "Convert the distro from WSL 1 to WSL 2",
        rationale: "WSL 1 translates syscalls rather than running a kernel, and exposes no GPU device at all. No driver or package work can give it ROCm support; the distro has to be converted.",
        commands: &[
            "# From Windows PowerShell:",
            "#   wsl --set-version <distro> 2",
            "#   wsl --set-default-version 2",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "uname -r",
        verify_linux: None,
        notes: &[
            "Converting rewrites the distro's filesystem and can take a long time on a large install. Back up anything you cannot lose first.",
        ],
        doc_url: None,
        applies_on: PRINT_ON_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-18-comgr-conflict",
        title: "Code object manager library does not belong to the active HIP runtime",
        rationale: "HIP compiles device code at run time through libamd_comgr, and this machine holds more than one copy of it. The copy the loader picks belongs to a different installation than the HIP runtime that loads, so compilation fails with an error that names neither the library nor the second copy. A second copy is not itself a fault -- many correct installations hold one -- so what is reported here is specifically the mismatch.",
        // No repair, and no recommendation between the two. Removing a stack or
        // reordering the search path can each break a working Python
        // environment, and which is right depends on which stack the user means
        // to keep -- a question only they can answer. `rocm diagnose` fills in
        // the real paths for this machine; these are the shapes.
        commands: &[
            "# Find every copy and which one loads:",
            "rocm examine --json    # read comgr_paths, comgr_selected, hip_selected",
            "# Then pick ONE of the following. They are alternatives, not steps.",
            "# (a) Keep the system installation: remove or uninstall the wheel that",
            "#     supplies the second copy.",
            "# (b) Keep the wheel: order the search path so the wheel's own copy of",
            "#     both libraries is found first, making the wheel the active runtime.",
            "export LD_LIBRARY_PATH=\"<directory of the wheel's own copy>:$LD_LIBRARY_PATH\"",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "python -c \"import torch; torch.zeros(1, device='cuda')\"",
        verify_linux: None,
        notes: &[
            COMGR_CONFLICT_NEITHER_OPTION_RECOMMENDED,
            "This describes the environment outside the CLI's managed runtimes. `rocm serve` puts a managed runtime's libraries first on purpose, so inside one the wheel copy wins by design and that is correct.",
        ],
        doc_url: None,
        // Not `PRINT_ON_LINUX`: which copy the loader picks has nothing to do
        // with the amdgpu module, and the two copies collide on WSL2 just the
        // same. Print-only on both: neither way out can be chosen for the user.
        applies_on: PRINT_ON_LINUX_AND_WSL,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-19-shm-too-small",
        title: "Raise the shared memory allowance",
        rationale: "A serving workload needs gigabytes of /dev/shm; a container gives it 64 MB by default, and WSL2 ships the same default. When the allowance runs out the workload crashes without the message ever naming shared memory -- a data-loader worker killed by a bus error, or a failed write to a temporary file -- so there is no route from what the user sees back to the cause.",
        // Two situations, one cause. A running container cannot be resized, so
        // the container case is a restart rather than a command that changes
        // this machine; the host case is a remount plus the fstab line that
        // makes it survive a reboot.
        commands: &[
            "# Check what you have:",
            "df -h /dev/shm",
            "# In a container: start it again with a larger allowance.",
            "#   docker run --shm-size=8g ...        # as fix-10-container shows",
            "# On a host: remount, then make it stick across a reboot.",
            "sudo mount -o remount,size=8g /dev/shm",
            "# /etc/fstab:  tmpfs  /dev/shm  tmpfs  defaults,size=8g  0 0",
        ],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: false,
        verify: "df -h /dev/shm",
        verify_linux: None,
        notes: &[
            "A running container cannot have its allowance changed. It has to be started again with the larger value.",
            "This is reported below 1 GiB. Silence is not proof of enough: a container given 2 GiB clears that bar and can still be too small for a large model.",
            "8g matches what fix-10-container already tells you to pass, so the two stay consistent.",
        ],
        doc_url: None,
        // Not `PRINT_ON_LINUX`: the size of a tmpfs has nothing to do with the
        // amdgpu module, and WSL2 ships the same 64 MB default a container does.
        // Print-only on both because every remedy above needs sudo.
        applies_on: PRINT_ON_LINUX_AND_WSL,
        runner: None,
    },
    // A new id rather than a rename of `fix-15`: that entry means "the runtime is
    // missing", and a stable id keeps its meaning. This one is "present but too
    // old", which has a different cause, a different symptom and a different
    // remedy (update, not install).
    FixRecipe {
        fix_id: "fix-21-msvc-redist-outdated",
        title: "Update the MSVC 2015-2022 runtime redistributable",
        rationale: "The Visual C++ runtime in System32 is older than 14.40. Programs built with a recent Visual C++ toolset, such as the llama.cpp server that Lemonade installs, need 14.40 or newer. On an older runtime they crash with an access violation (0xC0000005) inside MSVCP140.dll, and nothing says the runtime is the cause. The runtime is present, so there is no missing-DLL error. A copy bundled with another app, such as Python, can still make a presence check pass.",
        commands: &[
            "# Update the runtime (x64). This needs the Windows administrator prompt, so",
            "# the command is shown here and not run for you:",
            "winget install --id Microsoft.VCRedist.2015+.x64 --exact",
            "# Or download and run the installer:",
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe",
            "# Then close and restart the program (or Lemonade) so it loads the new runtime.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "powershell -NoProfile -Command \"(Get-Item ([Environment]::GetFolderPath('System') + '\\msvcp140.dll') -ErrorAction Stop).VersionInfo.FileVersion\"",
        verify_linux: None,
        notes: &[
            "The verify command prints the runtime version. It should read 14.40 or higher.",
            "`rocm fix` only prints this plan. It does not run the install: the install needs the Windows administrator prompt and your consent.",
            "14.40 is the first runtime that matches the toolset where the standard library changed. A newer program can still ask for a newer runtime, so install the latest one.",
        ],
        doc_url: Some("https://aka.ms/vs/17/release/vc_redist.x64.exe"),
        applies_on: PRINT_ON_WINDOWS,
        runner: None,
    },
    FixRecipe {
        fix_id: crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID,
        title: "Look for a newer AMD display driver (advice only)",
        rationale: "The AMD display driver on this Windows machine is dated before the first HIP SDK for Windows (27 July 2023). AMD publishes no minimum driver version for the HIP SDK or for the Lemonade Vulkan backend, so the minimum is not established and this is advice, not a finding that anything is broken. rocm-cli never installs drivers on Windows.",
        commands: &[
            "# Find the current display driver for this GPU on AMD's support page:",
            // A literal because a `static` slice cannot format a constant in. The
            // URL must equal `windows_driver::AMD_SUPPORT_URL`; the display-driver
            // test below pins it, and the diagnose copy builds the line from the constant.
            "#   https://www.amd.com/en/support",
            "# Restart the machine after the driver install.",
        ],
        needs_sudo: false,
        needs_reboot: true,
        needs_relogin: false,
        verify: "powershell -NoProfile -Command \"Get-CimInstance Win32_VideoController | Where-Object Name -match 'AMD|Radeon' | Select-Object Name,DriverVersion,DriverDate\"",
        verify_linux: None,
        notes: &[
            crate::windows_driver::NO_MINIMUM_EXPLANATION,
            crate::windows_driver::FLOOR_NOTE,
        ],
        doc_url: Some(crate::windows_driver::AMD_SUPPORT_URL),
        applies_on: PRINT_ON_WINDOWS,
        runner: None,
    },
    // The first [`FixClass::DiagnoseOnly`] member, and an `info-` id rather than
    // a `fix-` one: the `fix-N` series names problems that have a remedy, and
    // this entry's whole content is that there is none. The id still belongs in
    // this catalog, because `rocm diagnose` prints `apply with: rocm fix <id>`
    // for every cause it reports, and `rocm fix` has to answer for every id it
    // prints -- here by naming the problem and changing nothing, which is what
    // `Plan::Unfixable` does.
    FixRecipe {
        fix_id: "info-20-gpu-no-rocm-support",
        title: "GPU has no ROCm/HIP support (Vulkan backend still works)",
        rationale: "Every AMD GPU on this machine is a part ROCm has never shipped a runtime build for -- not one that is merely undetected. No ROCm or HIP install, driver version or environment variable changes that, so there is nothing to apply. Vulkan-based engines, which is what Lemonade runs by default, do not use ROCm and work on it.",
        commands: &[],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "",
        verify_linux: None,
        notes: &[
            "HSA_OVERRIDE_GFX_VERSION is NOT a workaround here: it makes the runtime load kernels built for another architecture, which faults at runtime rather than adding support.",
            // Not "`rocm serve` already picks a Vulkan engine on this host":
            // that is a claim about the CLI's own runtime behaviour which this
            // catalog does not decide. A configured default engine outranks
            // `default_engine_for_host`, and Lemonade picks its llama.cpp
            // backend itself. The matching note in `diagnose`'s `check_20` is
            // worded the same way.
            "To use this GPU for inference, serve with a Vulkan-capable engine. If you have set a default engine that needs ROCm, that one will not work here; `rocm serve --engine lemonade` does not need it.",
        ],
        doc_url: Some(
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
        ),
        applies_on: DIAGNOSE_ONLY_ON_LINUX_AND_WINDOWS,
        runner: None,
    },
];

/// Assert that a recipe whose steps span more than one shell says which shell
/// each group runs in.
///
/// Shared by the two copies of the plan — the catalog recipe here and the
/// `Fix` [`crate::diagnose`] attaches to its finding — because a divergence
/// between them is exactly the kind of thing a reader would meet and not the
/// author.
///
/// Scope, so the guarantee is not read wider than it is: this holds the
/// *command block* only, and only its shell boundary. It says nothing about
/// `summary`, `notes` or `verify`. Byte-equality of the two blocks is a
/// separate check — [`assert_plan_matches_the_catalog_copy`] — and the prose
/// around them is deliberately not identical, because [`FixRecipe`] has a
/// `rationale` field that `Fix` has no counterpart for: the upstream-defect
/// explanation that `diagnose` carries as a note is printed by `rocm fix` out
/// of `rationale` instead, and duplicating it into `notes` would print it
/// twice.
///
/// Both renderers print every line of the block with the same `$ ` prefix, so
/// ordering alone tells a reader nothing: it does not say that
/// `rocm engines shell vllm` opened an interactive subshell the probes belong
/// inside, nor that the reinstall must run outside it because it replaces the
/// very environment that subshell is standing in. Rendered flat, a user pasting
/// the block wholesale was left depending on terminal stdin buffering to land
/// each line in the right shell. Hold the block to marking the boundary in both
/// directions — ordered (open it, leave it, only then act) and labelled.
#[cfg(test)]
pub(crate) fn assert_engine_shell_boundary_is_labelled(fix_id: &str, commands: &[&str]) {
    let position = |needle: &str| {
        commands
            .iter()
            .position(|c| c.trim() == needle)
            .unwrap_or_else(|| {
                panic!("{fix_id}: the plan no longer runs `{needle}`:\n{commands:#?}")
            })
    };
    let open = position("rocm engines shell vllm");
    let leave = position("exit");
    let act = position("rocm engines install vllm --reinstall");
    assert!(
        open < leave && leave < act,
        "{fix_id}: the subshell has to be opened, then left, and only then may the \
         reinstall run -- it replaces the environment that subshell stands in:\n{commands:#?}"
    );
    let is_comment = |c: &&str| c.trim_start().starts_with('#');
    assert!(
        commands[open + 1..leave].iter().any(|c| !is_comment(c)),
        "{fix_id}: nothing actually runs inside the subshell, so opening one is \
         unexplained:\n{commands:#?}"
    );
    let labelled = |from: usize, to: usize, want: &str| {
        commands[from..to]
            .iter()
            .any(|c| is_comment(c) && c.contains(want))
    };
    assert!(
        labelled(open, leave, "INSIDE"),
        "{fix_id}: the block has to say the probes run INSIDE the subshell rather \
         than merely list them after it:\n{commands:#?}"
    );
    assert!(
        labelled(leave, act, "YOUR OWN shell"),
        "{fix_id}: the block has to say the reinstall runs back in the user's own \
         shell:\n{commands:#?}"
    );
}

/// Assert that the command block a [`crate::diagnose::Fix`] carries is
/// byte-identical to the catalog recipe's, line for line.
///
/// The two are hand-maintained copies of one plan in two modules, and the
/// block is the part a user pastes into a shell, so a silent divergence is a
/// user pasting steps that no longer match the ones `rocm fix` prints. Holding
/// the shell boundary in both (see
/// [`assert_engine_shell_boundary_is_labelled`]) leaves the wording free to
/// drift; this closes that.
#[cfg(test)]
pub(crate) fn assert_plan_matches_the_catalog_copy(fix_id: &str, commands: &[&str]) {
    let recipe = find_recipe(fix_id)
        .unwrap_or_else(|| panic!("{fix_id}: no catalog recipe to compare the plan against"));
    assert_eq!(
        recipe.commands, commands,
        "{fix_id}: the plan `diagnose` attaches has drifted from the catalog recipe \
         `rocm fix` prints; they are one plan and a user may meet either copy"
    );
}

/// Assert that the `needs_reboot` a [`crate::diagnose::Fix`] carries matches
/// the catalog recipe's.
///
/// Same hand-maintained-copies problem as [`assert_plan_matches_the_catalog_copy`],
/// for a single flag instead of the command block: `FixRecipe` and `Fix` set
/// `needs_reboot` independently, so a silent divergence means `rocm diagnose`
/// and `rocm fix <id>` tell a user different things about the same fix-id
/// (this is what happened with `fix-5-amdgpu-load` before it was closed here).
#[cfg(test)]
pub(crate) fn assert_needs_reboot_matches_the_catalog(fix_id: &str, needs_reboot: bool) {
    let recipe = find_recipe(fix_id)
        .unwrap_or_else(|| panic!("{fix_id}: no catalog recipe to compare against"));
    assert_eq!(
        recipe.needs_reboot, needs_reboot,
        "{fix_id}: diagnose's needs_reboot has drifted from the catalog recipe \
         `rocm fix` reports; a user may see either surface for the same fix-id"
    );
}

pub(super) fn find_recipe(fix_id: &str) -> Option<&'static FixRecipe> {
    RECIPES.iter().find(|r| r.fix_id == fix_id)
}

/// The oldest ROCm major this CLI will leave on a machine.
///
/// A statement about the product, not a preference: the installer ships ROCm 7
/// series wheels and supports the ROCm 10 layout, so 7 is the floor a user can
/// actually end up on. Remediations are checked against it, because a step
/// naming a wheel index older than this cannot help anyone the catalog can
/// reach — and in `fix-8-wheel-rocm`'s case re-creates the very mismatch it
/// reports.
///
/// Raise it when the installer stops producing ROCm 7.
///
/// Test-only: it exists to be asserted against, not to steer runtime behaviour.
#[cfg(test)]
pub(crate) const OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS: u32 = 7;

/// Every PyTorch ROCm wheel index named in `commands`, as `(major, command)`.
///
/// Shared by the catalog's guard and `diagnose`'s, so the two cannot disagree
/// about what counts as naming an index.
#[cfg(test)]
pub(crate) fn torch_rocm_indexes_named_in<'a>(
    commands: impl IntoIterator<Item = &'a str>,
) -> Vec<(u32, String)> {
    commands
        .into_iter()
        .filter_map(|c| {
            let tail = &c[c.find("download.pytorch.org/whl/")?..];
            let digits: String = tail[tail.find("rocm")? + 4..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().ok().map(|major| (major, c.to_owned()))
        })
        .collect()
}

/// Test-only view of a recipe's actionable remediation: its `verify` command,
/// its `notes`, and its *executable* command lines (comment/prose lines
/// stripped). Lets the diagnosis catalog cross-check that the Fix it emits for a
/// shared `fix_id` has not silently diverged from this recipe.
#[cfg(test)]
pub(crate) fn recipe_verify_notes_and_run_commands(
    fix_id: &str,
) -> Option<(&'static str, &'static [&'static str], Vec<&'static str>)> {
    find_recipe(fix_id).map(|recipe| {
        let run_commands = recipe
            .commands
            .iter()
            .copied()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>();
        (recipe.verify, recipe.notes, run_commands)
    })
}

/// Test-only: the Linux and WSL verify line of a recipe, when it has one.
#[cfg(test)]
pub(crate) fn recipe_linux_verify(fix_id: &str) -> Option<&'static str> {
    find_recipe(fix_id).and_then(|recipe| recipe.verify_linux)
}

/// What one Windows shell did with a line typed into it.
#[cfg(all(test, windows))]
pub(crate) struct ShellRun {
    pub shell: &'static str,
    pub success: bool,
    pub status: String,
    pub stdout: String,
    pub stderr: String,
}

/// Runs `text` the way a user would type it into Windows PowerShell 5.1.
///
/// The text goes on stdin (`-Command -`), which parses it as typed. Passing it
/// as an argument of an outer `-Command` is not the same thing: PowerShell
/// joins its arguments after removing their quotes, so the inner command text
/// would be cut at the first space and the pipe would run in the outer shell.
///
/// `env` sets (`Some`) or removes (`None`) variables for the child only.
///
/// `PSModulePath` is removed first, so Windows PowerShell 5.1 starts with its own
/// default module path. A test run from a PowerShell 7 session inherits that
/// session's path, which lists the PowerShell 7 modules first. 5.1 then picks
/// the PSReadLine built for 7, cannot load it, and writes "Cannot load PSReadline
/// module" to stderr, which the tests below treat as a failure of the line.
#[cfg(all(test, windows))]
pub(crate) fn run_typed_in_powershell(text: &str, env: &[(&str, Option<&str>)]) -> ShellRun {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut command = Command::new("powershell");
    command.args(["-NoProfile", "-Command", "-"]);
    command.env_remove("PSModulePath");
    for (name, value) in env {
        match value {
            Some(v) => command.env(name, v),
            None => command.env_remove(name),
        };
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("powershell starts");
    let mut stdin = child.stdin.take().expect("stdin is piped");
    writeln!(stdin, "{text}").expect("the line is written");
    drop(stdin);
    shell_run(
        "powershell",
        child.wait_with_output().expect("powershell finishes"),
    )
}

/// Runs `line` the way a user would type it into cmd (`cmd /C`, with the line
/// passed unquoted and unescaped). `env` is as for [`run_typed_in_powershell`].
#[cfg(all(test, windows))]
pub(crate) fn run_typed_in_cmd(line: &str, env: &[(&str, Option<&str>)]) -> ShellRun {
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    let mut command = Command::new("cmd");
    command.arg("/C").raw_arg(line);
    command.env_remove("PSModulePath");
    for (name, value) in env {
        match value {
            Some(v) => command.env(name, v),
            None => command.env_remove(name),
        };
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = command
        .spawn()
        .expect("cmd starts")
        .wait_with_output()
        .expect("cmd finishes");
    shell_run("cmd", out)
}

/// Runs one printed line in cmd and in Windows PowerShell, in that order.
#[cfg(all(test, windows))]
pub(crate) fn run_typed_in_cmd_and_powershell(
    line: &str,
    env: &[(&str, Option<&str>)],
) -> Vec<ShellRun> {
    vec![
        run_typed_in_cmd(line, env),
        run_typed_in_powershell(line, env),
    ]
}

#[cfg(all(test, windows))]
fn shell_run(shell: &'static str, out: std::process::Output) -> ShellRun {
    ShellRun {
        shell,
        success: out.status.success(),
        status: format!("{:?}", out.status),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Where a `powershell ... -Command "..."` argument starts in `lowered` (the
/// line, lowercased): the byte offset of the flag, the quote that opens the
/// argument, and the text after that quote.
///
/// PowerShell accepts any prefix of `-Command` down to `-c`, so `-c` and
/// `-comm` are found as well.
#[cfg(test)]
fn command_argument(lowered: &str) -> Option<(usize, char, &str)> {
    lowered.match_indices('-').find_map(|(at, _)| {
        let head = &lowered[..at];
        // Only a PowerShell line: `python -c "..."` has the same flag.
        if head.chars().next_back().is_some_and(|c| !c.is_whitespace())
            || !(head.contains("powershell") || head.contains("pwsh"))
        {
            return None;
        }
        let rest = &lowered[at..];
        let flag = rest.split_whitespace().next()?;
        if flag.len() < 2 || !"-command".starts_with(flag) {
            return None;
        }
        let after = rest[flag.len()..].trim_start();
        let quote = after.chars().next().filter(|c| matches!(c, '"' | '\''))?;
        Some((at, quote, &after[1..]))
    })
}

/// Test-only rule for what a printed command must not contain when it is shown
/// to a user of Windows PowerShell 5.1 as well as cmd:
///
/// - `&&` and `||`, which are syntax errors in Windows PowerShell 5.1;
/// - a bare `where`, which starts a command and is the `Where-Object` alias in
///   PowerShell, not `where.exe`. A command starts after `|`, `;`, `(`, `{` and
///   a lone `&` (cmd's unconditional separator), at the start of a
///   `-Command "..."` body, and after `cmd /c`, so `echo x & where foo` is
///   caught. The `&` of a call operator (`& 'path\exe'`) is split the same way
///   and stays legal, because the word after it is not `where`;
/// - a `%NAME%` reference, which PowerShell passes on as literal text. For
///   `setx PATH "%PATH%;..."` that writes `%PATH%` into the user's Path;
/// - a `-Command` (or `-c`, or any longer prefix of it) argument in single
///   quotes. cmd does not treat `'` as a quote, so it splits the argument at
///   its spaces;
/// - inside the double-quoted body of a `powershell ... -Command "..."` line, a
///   backslash-escaped quote (`\"`) or any `$`. PowerShell does not take `\` as
///   an escape, so typed into PowerShell the `\"` ends the string early; and
///   an outer PowerShell expands `$_`, `$env:NAME` and the like before the inner
///   one starts, so they would be empty or hold the outer value, and a value
///   with a space in it would split the inner command. Use single quotes inside
///   the body, `[Environment]::GetEnvironmentVariable('NAME')` for a variable,
///   `[Environment]::GetFolderPath('System')` for the system directory, and a
///   cmdlet that needs no `$_`. There is no exemption. The `-Command` match
///   ignores case;
/// - outside such a body, an empty-string argument (`""` or `''`). Windows
///   PowerShell 5.1 drops it when it calls a native program, so
///   `setx NAME ""` reaches setx as `setx NAME`, which fails with "Invalid
///   syntax" and clears nothing. Use a .NET call such as
///   `[Environment]::SetEnvironmentVariable('NAME',$null,'User')`;
/// - inside a `-Command` body, a `Get-Item` that does not carry
///   `-ErrorAction Stop`. `Get-Item` reports a missing path as a
///   non-terminating error and PowerShell exits 0 after one, so a verify that
///   checks a file this way would pass with the file missing. Other cmdlets
///   that behave this way are not listed.
///
/// It is a text rule over those known-bad constructs. It is not a parser: it
/// does not prove a line is valid or that it runs, and a construct it does not
/// list passes. On Windows, the verify lines of fix-2, fix-13, fix-14, fix-15,
/// fix-21 and fix-22 are run by `the_override_verify_runs_in_cmd_and_powershell`,
/// `the_hip_sdk_verify_runs_the_exe_under_hip_path`,
/// `the_adrenalin_verify_runs_and_its_filter_keeps_only_amd`,
/// `the_msvc_verify_runs_in_cmd_and_powershell` (fix-15 again in
/// `the_msvc_verify_exits_nonzero_when_a_dll_is_missing`),
/// `the_outdated_msvc_verify_prints_a_version_and_fails_when_the_dll_is_missing`
/// and `the_display_driver_verify_runs_in_cmd_and_powershell`; fix-9's Windows
/// line, which only `diagnose.rs` prints, by
/// `the_igpu_verify_pins_the_variable_it_names`.
/// Some lines `diagnose.rs` prints are PowerShell-only by design, and are
/// checked as such where they are built.
///
/// Shared by the checks on the `RECIPES` catalog here and on the `Fix` values
/// that `diagnose.rs` builds. A copy of this rule lives in the e2e crate
/// (`powershell_text_rule::refusal`), which cannot depend on rocm-core;
/// `the_powershell_text_rule_agrees_with_the_shared_case_table` and the test
/// next to that copy both read `tests/e2e-cucumber/fixtures/powershell_text_rule_cases.txt`,
/// so a case one rule gets wrong fails on that side.
#[cfg(test)]
pub(crate) fn fails_in_windows_powershell(line: &str) -> Option<&'static str> {
    if line.contains("&&") || line.contains("||") {
        return Some("`&&` or `||`, a syntax error in Windows PowerShell 5.1");
    }
    // `-Command` is not case-sensitive, and neither is this match.
    let lowered = line.to_ascii_lowercase();
    let command = command_argument(&lowered);
    // An opening brace is written as an escape on purpose: the advised-commands
    // scanner skips a cfg(test) item by counting braces, and a lone one throws it off.
    let starts_bare_where = |text: &str| {
        text.split(['|', ';', '&', '(', '\u{7b}']).any(|part| {
            let mut words = part.split_whitespace().map(|w| w.trim_matches('"'));
            let mut first = words.next();
            if first == Some("cmd") {
                first = words.find(|w| !w.starts_with('/'));
            }
            first == Some("where")
        })
    };
    if starts_bare_where(&lowered)
        || command.is_some_and(|(_, _, body)| starts_bare_where(body.trim_start()))
    {
        return Some("a bare `where`, which is `Where-Object` in PowerShell");
    }
    // The text outside a `-Command "..."` body, which is typed at the outer
    // shell and so is where an argument can be dropped.
    let outer = command.map_or(lowered.as_str(), |(at, _, _)| &lowered[..at]);
    if outer
        .split_whitespace()
        .any(|word| word == "\"\"" || word == "''")
    {
        return Some(
            "an empty-string argument (`\"\"` or `''`), which Windows PowerShell 5.1 drops when it calls a native program",
        );
    }
    if let Some((_, quote, body)) = command {
        if quote == '\'' {
            return Some(
                "a single-quoted `-Command` argument, which cmd does not take as one argument",
            );
        }
        if body.contains("\\\"") {
            return Some(
                "a backslash-escaped quote inside a `powershell -Command \"...\"` body, which ends the string early when typed into PowerShell",
            );
        }
        if body.contains('$') {
            return Some(
                "a `$` inside a `powershell -Command \"...\"` body (`$_`, `$env:NAME`), which an outer PowerShell expands before the inner one starts",
            );
        }
        if body.contains("get-item") && !body.contains("-erroraction stop") {
            return Some(
                "a `Get-Item` without `-ErrorAction Stop`: a missing path is a non-terminating error, so the line prints it and still exits 0",
            );
        }
    }
    // The text between a `%` and the next one. The last piece has no closing
    // `%`, so it is never a name.
    let pieces: Vec<&str> = line.split('%').collect();
    let names_a_cmd_variable = pieces
        .get(1..pieces.len().saturating_sub(1))
        .unwrap_or_default()
        .iter()
        .any(|name| {
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        });
    names_a_cmd_variable.then_some("a `%NAME%` reference, which PowerShell does not expand")
}

/// Test-only view of the platform families a recipe is runnable on.
///
/// `None` when no recipe carries `fix_id`. Lets the diagnosis catalog assert
/// that every platform a checker answers on is a platform `rocm fix` will
/// actually act on, which `recipe_verify_notes_and_run_commands` cannot see.
#[cfg(test)]
pub(crate) fn recipe_applies_on(fix_id: &str) -> Option<Vec<&'static str>> {
    find_recipe(fix_id).map(|recipe| recipe.applies_on.iter().map(|p| p.os).collect())
}

/// Whether `fix_id` names an entry in the catalog.
///
/// A predicate rather than the lookup itself, because the caller that needs it
/// is deciding whether a caller-supplied string may be published, not reading a
/// recipe. Handing back the recipe would also make a private type reachable
/// from outside this module.
#[must_use]
pub(crate) fn is_catalog_id(fix_id: &str) -> bool {
    find_recipe(fix_id).is_some()
}

/// What the catalog says `rocm fix <fix_id>` does on `os`.
///
/// `None` when the id is not in the catalog, or when it is but does not apply
/// on that operating system at all.
///
/// This exists so `diagnose` can read the class off the catalog instead of
/// restating it. The two were hand-maintained copies, and they had already
/// drifted: `fix-9-igpu-dgpu` was `auto_applicable: true` here and `false` in
/// the Linux arm of its checker, with no test comparing them.
#[must_use]
pub fn class_on(fix_id: &str, os: &str) -> Option<FixClass> {
    find_recipe(fix_id)
        .and_then(|r| r.platform(os))
        .map(|p| p.class)
}

/// The argument a [`FixClass::NeedsArgument`] entry is waiting for on `os`.
///
/// Read alongside [`class_on`] for the same reason: `format_flags` names the
/// argument, and `diagnose` renders those flags too. Without this, the same
/// entry would read "needs --device-index" from `rocm fix` and "needs an
/// argument" from `rocm diagnose`, which is the divergence `format_flags`
/// exists to prevent.
#[must_use]
pub fn needs_on(fix_id: &str, os: &str) -> Option<&'static str> {
    find_recipe(fix_id)
        .and_then(|r| r.platform(os))
        .and_then(|p| p.needs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every platform class a recipe carries, as `(fix-id, os, class)`.
    fn classes() -> Vec<(&'static str, &'static str, FixClass)> {
        RECIPES
            .iter()
            .flat_map(|r| r.applies_on.iter().map(|p| (r.fix_id, p.os, p.class)))
            .collect()
    }

    // Ubuntu 26.04 gets the DKMS driver plan (`dkms_driver_plan_is_supported`),
    // so no recipe may still describe it as in-tree or unplanned.
    #[test]
    fn recipes_do_not_describe_ubuntu_26_04_as_unplanned() {
        for r in RECIPES {
            let text = std::iter::once(r.rationale)
                .chain(r.commands.iter().copied())
                .chain(r.notes.iter().copied())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                !text.contains("in-tree") && !text.contains("apt install rocm"),
                "{} still describes Ubuntu 26.04 as in-tree",
                r.fix_id
            );
        }
    }

    /// No entry sends a user to a wheel index older than any ROCm this CLI
    /// installs.
    ///
    /// The decay this catches is invisible to everything else. The advice
    /// compiles, the tests pass, and `assert_plan_matches_the_catalog_copy`
    /// confirms the catalog and `diagnose` copies agree — which they did, while
    /// both went stale together. A guard checking that two copies match each
    /// other cannot notice that both have drifted from the world outside.
    #[test]
    fn no_remediation_names_a_wheel_index_older_than_the_rocm_we_install() {
        let named: Vec<(&str, u32, String)> = RECIPES
            .iter()
            .flat_map(|r| {
                torch_rocm_indexes_named_in(r.commands.iter().copied())
                    .into_iter()
                    .map(move |(major, command)| (r.fix_id, major, command))
            })
            .collect();

        // Non-vacuity: entries do name wheel indexes, and a parser that matched
        // none would leave the assertion below checking an empty list forever.
        assert!(
            !named.is_empty(),
            "no entry was seen to name a wheel index, so this guard is checking nothing"
        );

        let stale: Vec<_> = named
            .iter()
            .filter(|(_, major, _)| *major < OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS)
            .collect();
        assert!(
            stale.is_empty(),
            "these steps name a ROCm older than {OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS}, the oldest \
             this CLI installs, so a user following them installs a framework for a major they \
             do not have: {stale:#?}"
        );
    }

    /// Exactly which entries are `DIAGNOSE-ONLY`, and on which platforms.
    ///
    /// The class shipped with no member on purpose: it is what lets a
    /// detect-but-cannot-repair failure be accepted as an entry at all. It
    /// shipped before the first member arrived because `rocm diagnose --json`
    /// already serialises `class`, and a consumer that deserialised that output
    /// into a fixed enum would reject a payload carrying a variant added later
    /// -- shipping the full set from the start means that variant is already
    /// something callers have to tolerate.
    ///
    /// `info-20-gpu-no-rocm-support` is the first member, and this is now the
    /// pin rather than the tripwire: the list is exact, so promoting an entry
    /// to "there is no fix" -- the strongest thing this catalog can say about a
    /// problem, and the one a user can do least with -- fails here until it is
    /// written down.
    #[test]
    fn diagnose_only_has_exactly_the_members_the_catalog_names() {
        let members: Vec<(&str, Vec<&str>)> = RECIPES
            .iter()
            .filter(|r| {
                r.applies_on
                    .iter()
                    .any(|p| p.class == FixClass::DiagnoseOnly)
            })
            .map(|r| {
                (
                    r.fix_id,
                    r.applies_on
                        .iter()
                        .filter(|p| p.class == FixClass::DiagnoseOnly)
                        .map(|p| p.os)
                        .collect(),
                )
            })
            .collect();
        assert_eq!(
            members,
            vec![("info-20-gpu-no-rocm-support", vec!["linux", "windows"])],
            "DIAGNOSE-ONLY membership changed. That is the class working as intended -- update \
             this test, and say in the commit why the new entry has no fix."
        );
    }

    /// The first `DIAGNOSE-ONLY` member reaches [`Plan::Unfixable`] from the
    /// real catalog, not only from the synthetic recipe above.
    ///
    /// `the_class_gate_is_reached_only_after_the_platform_gate` and its sibling
    /// both build a recipe by hand, so between them they prove the branch works
    /// without proving anything ships through it. This runs `plan_of` on the
    /// catalog entry itself, on each OS it is scoped to.
    #[test]
    fn the_catalogs_diagnose_only_entry_plans_as_unfixable() {
        let recipe = find_recipe("info-20-gpu-no-rocm-support")
            .expect("the informational entry is in the catalog");
        assert_eq!(
            recipe.os_scope(),
            vec!["linux", "windows"],
            "the checker is registered for linux and windows, so the recipe has to answer there"
        );
        for os in recipe.os_scope() {
            assert_eq!(
                recipe.platform(os).map(|p| p.class),
                Some(FixClass::DiagnoseOnly),
                "{os}: `rocm fix info-20-gpu-no-rocm-support` must name the problem and change \
                 nothing"
            );
        }
        assert!(
            recipe.runner.is_none() && recipe.commands.is_empty() && recipe.verify.is_empty(),
            "an entry with no fix has nothing to run, nothing to copy and nothing to verify"
        );
    }

    #[test]
    fn every_recipe_id_is_unique_and_covers_the_catalog() {
        let mut ids: Vec<&str> = RECIPES.iter().map(|r| r.fix_id).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate fix-id in RECIPES");
        // 22 bare-metal/Windows entries (fix-16-vllm-oom, fix-17-torch-dlpack,
        // fix-18-comgr-conflict, fix-19-shm-too-small,
        // fix-21-msvc-redist-outdated, fix-22-display-driver-old and the
        // informational info-20-gpu-no-rocm-support among them) plus the 7 WSL
        // ones.
        assert_eq!(count, 29, "expected 29 catalog entries");
    }

    /// `rocm fix fix-21-msvc-redist-outdated` prints the plan and runs nothing.
    ///
    /// The plan includes a `winget install` line, and says the CLI does not run
    /// it. This is the test that makes that sentence true: no runner, and the
    /// only platform class is print-only. Letting the CLI run the install would
    /// need the maintainers' agreement, because it takes the Windows
    /// administrator prompt.
    #[test]
    fn the_outdated_runtime_recipe_is_print_only_on_windows() {
        let recipe = RECIPES
            .iter()
            .find(|r| r.fix_id == "fix-21-msvc-redist-outdated")
            .expect("the entry exists");
        assert!(
            recipe.runner.is_none(),
            "an install must not be run for the user"
        );
        assert_eq!(recipe.applies_on.len(), 1);
        assert_eq!(recipe.applies_on[0].os, "windows");
        assert_eq!(recipe.applies_on[0].class, FixClass::PrintOnly);
        assert!(
            recipe
                .commands
                .contains(&"winget install --id Microsoft.VCRedist.2015+.x64 --exact")
        );
        assert!(
            recipe
                .notes
                .iter()
                .any(|n| n.contains("only prints this plan"))
        );
    }

    #[test]
    fn the_dlpack_recipe_says_which_shell_each_step_runs_in() {
        let recipe =
            find_recipe("fix-17-torch-dlpack").expect("fix-17-torch-dlpack must be in the catalog");
        assert_engine_shell_boundary_is_labelled(recipe.fix_id, recipe.commands);
    }

    /// One direction only, on purpose. A recipe the catalog says the CLI
    /// carries out somewhere must have something to carry it out with, or
    /// `apply` reaches its internal-error path. The converse is deliberately
    /// not asserted: a print-only recipe may hold a runner that reports without
    /// writing, which is exactly what `fix-2-unset-override` does on Linux.
    #[test]
    fn the_utilization_hint_example_matches_the_recipe_command() {
        // The shared hint carries a worked value, and `fix-16-vllm-oom` hands
        // the user a command carrying another. They drifted once already: the
        // docs and the recipe moved to 0.5 while the const kept recommending
        // 0.1, so one `rocm diagnose` printed "e.g. 0.1 for a small model"
        // directly above `rocm serve <model> --gpu-memory-utilization 0.5`.
        // Nothing caught it -- the cross-check in `diagnose` exempts the summary
        // as prose framing, and a line-based grep misses the const because the
        // flag and the value sit on different continuation lines.
        const FLAG: &str = "--gpu-memory-utilization";
        let hint = crate::VLLM_GPU_MEMORY_UTILIZATION_HINT;
        let example = hint
            .split_once("e.g. ")
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .unwrap_or_else(|| {
                panic!("the shared hint no longer carries a worked `e.g. <value>`:\n{hint}")
            });
        let recipe = find_recipe("fix-16-vllm-oom").expect("fix-16-vllm-oom is in the catalog");
        let commanded = recipe
            .commands
            .iter()
            .find_map(|line| line.split_once(FLAG))
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .unwrap_or_else(|| {
                panic!(
                    "fix-16-vllm-oom no longer commands `{FLAG}`:\n{:#?}",
                    recipe.commands
                )
            });
        assert_eq!(
            example, commanded,
            "the shared hint recommends `{FLAG} {example}` while `rocm fix fix-16-vllm-oom` \
             commands `{FLAG} {commanded}`; both can print in a single `rocm diagnose` report"
        );
    }

    #[test]
    fn the_utilization_hint_states_the_bound_the_parser_accepts() {
        // `rocm serve`'s `parse_gpu_memory_utilization` rejects `<= 0` and
        // accepts `1`, so the domain is (0, 1]. The behaviour itself is proved
        // by `gpu_memory_utilization_rejects_out_of_range_and_unparsable_values`
        // in `apps/rocm/src/serve_recipe.rs`, which rejects `0`/`0.0` and accepts
        // `1`; this assertion pins only the wording that advertises it, which
        // has no behaviour of its own. `<0-1>` advertises `0`, and this
        // hint prints in three places (pre-launch note, engine OOM hint, and the
        // `fix-16-vllm-oom` summary), so a user who OOMs, reads the tool's own
        // advice and passes `0` is rejected by the same tool.
        //
        // Pinned because the wording was silently reverted once: it was fixed on
        // this branch, then a merge resolved the same line from a pre-fix tree.
        // Nothing went red, because the only pin on this const checked the
        // worked example (`0.5`) and not the range text.
        const FLAG: &str = "--gpu-memory-utilization";
        let hint = crate::VLLM_GPU_MEMORY_UTILIZATION_HINT;
        assert!(
            hint.contains(&format!("{FLAG} <fraction greater than 0 and at most 1>")),
            "the hint must state the bound `{FLAG}` actually accepts:\n{hint}"
        );
        assert!(
            !hint.contains("<0-1>"),
            "`<0-1>` wrongly advertises 0, which `{FLAG}` rejects:\n{hint}"
        );
    }

    #[test]
    fn a_recipe_that_acts_has_a_runner() {
        for r in RECIPES {
            let acts_somewhere = r
                .applies_on
                .iter()
                .any(|p| matches!(p.class, FixClass::Auto | FixClass::NeedsArgument));
            assert!(
                !acts_somewhere || r.runner.is_some(),
                "{}: the catalog says the CLI carries this out on some platform, \
                 but there is no runner to do it",
                r.fix_id
            );
        }
    }

    /// The commands and the verify line of a recipe that applies on Windows
    /// are shown to users of Windows PowerShell and of cmd, so none may use
    /// what `fails_in_windows_powershell` lists. The set of recipes is derived
    /// from `applies_on`, so a new Windows recipe is covered without editing
    /// this test.
    ///
    /// This pins the text only. The Windows-only tests named on
    /// `fails_in_windows_powershell` run the verify lines of fix-2, fix-13,
    /// fix-14, fix-15, fix-21 and fix-22 in a shell; the other catalog lines
    /// are not run. The same
    /// rule is applied to the `Fix` values that `diagnose.rs` builds, in
    /// `windows_diagnosis_remediation_passes_the_powershell_text_rule`.
    #[test]
    fn windows_recipes_pass_the_powershell_text_rule() {
        let windows: Vec<&FixRecipe> = RECIPES
            .iter()
            .filter(|r| r.platform("windows").is_some())
            .collect();
        assert!(
            windows.iter().any(|r| r.fix_id == "fix-15-msvc-redist"),
            "the derivation found no Windows recipes: {windows:?}",
            windows = windows.iter().map(|r| r.fix_id).collect::<Vec<_>>()
        );
        for r in windows {
            let printed = r
                .commands
                .iter()
                .copied()
                .filter(|c| !c.trim_start().starts_with('#'))
                .chain(std::iter::once(r.verify))
                .chain(r.verify_linux);
            for line in printed {
                if let Some(why) = fails_in_windows_powershell(line) {
                    panic!("{}: `{line}` uses {why}", r.fix_id);
                }
            }
        }
    }

    /// The fix-2 Linux line, run in `/bin/sh` with a `PATH` that holds nothing
    /// (no `python`, no `env`, no `grep`): it must still report the override's
    /// value, or OK_UNSET once it is cleared, and exit 0. The plain `python`
    /// line fails on this PATH.
    #[cfg(unix)]
    #[test]
    fn the_override_linux_verify_runs_with_nothing_on_path() {
        let linux = find_recipe("fix-2-unset-override")
            .and_then(|r| r.verify_linux)
            .expect("fix-2 has a Linux verify line");
        let empty = tempfile::tempdir().expect("an empty PATH directory is created");
        let run = |line: &str, value: Option<&str>| {
            let mut command = std::process::Command::new("/bin/sh");
            command
                .arg("-c")
                .arg(line)
                .env_clear()
                .env("PATH", empty.path());
            if let Some(value) = value {
                command.env("HSA_OVERRIDE_GFX_VERSION", value);
            }
            let out = command.output().expect("/bin/sh runs");
            (
                out.status.success(),
                String::from_utf8_lossy(&out.stdout).trim().to_owned(),
            )
        };
        assert_eq!(run(linux, Some("11.0.0")), (true, "11.0.0".to_owned()));
        assert_eq!(run(linux, None), (true, "OK_UNSET".to_owned()));
        let plain = find_recipe("fix-2-unset-override").unwrap().verify;
        assert!(!run(plain, None).0, "the plain line needs python on PATH");
    }

    /// The fix-15 verify line, taken apart. This is a structural check on the
    /// text, so it runs on every host: a misspelled cmdlet or environment
    /// variable fails here even where no Windows shell is available. It does not
    /// replace running the line. `the_msvc_verify_runs_in_cmd_and_powershell`
    /// does that, and only on Windows.
    #[test]
    fn the_msvc_verify_is_a_wellformed_powershell_command() {
        let verify = find_recipe("fix-15-msvc-redist")
            .expect("fix-15 is in the catalog")
            .verify;
        // Starts with `powershell` so cmd runs it too. The body is one
        // double-quoted argument with no `$`, so an outer PowerShell has
        // nothing to expand, whatever the system directory's path holds.
        let body = verify
            .strip_prefix("powershell -NoProfile -Command \"")
            .and_then(|rest| rest.strip_suffix('"'))
            .unwrap_or_else(|| panic!("not a quoted `powershell -Command` argument: {verify}"));
        assert!(
            !body.contains('"'),
            "an inner quote would end the argument: {body}"
        );
        // Every command word is a real cmdlet, and the Windows directory is
        // read inside the body by .NET rather than from a variable.
        let cmdlets: Vec<&str> = body
            .split('|')
            .map(|stage| {
                stage
                    .trim()
                    .trim_start_matches('(')
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
            })
            .collect();
        assert_eq!(cmdlets, ["Get-Item", "Select-Object"], "{body}");
        assert!(
            !body.contains('$'),
            "an outer PowerShell would expand it: {body}"
        );
        assert_eq!(
            body.matches("[Environment]::GetFolderPath('System')")
                .count(),
            2,
            "{body}"
        );
        for dll in ["vcruntime140.dll", "vcruntime140_1.dll"] {
            assert!(body.contains(dll), "{dll} is not checked: {body}");
        }
        assert!(
            body.contains(".VersionInfo") && body.ends_with("Select-Object FileName,FileVersion"),
            "{body}"
        );
    }

    /// Runs the fix-15 verify line the way a user would: typed into cmd, and
    /// typed into Windows PowerShell. Both must exit 0 and print each DLL on its own
    /// row with a dotted version. The runtime is installed on the Windows CI hosts, so a
    /// failure here means the line is broken.
    ///
    /// The PowerShell leg feeds the line on stdin (`-Command -`), which parses
    /// it as typed text. Passing it as an argument of an outer `-Command` is
    /// not the same thing: PowerShell joins its arguments after removing their
    /// quotes, so the inner command text would be cut at the first space and
    /// the pipe would run in the outer shell.
    ///
    /// Only compiled on Windows. Linux and macOS lanes do not run it, which is
    /// why `the_msvc_verify_is_a_wellformed_powershell_command` also checks the
    /// text on every host.
    #[cfg(windows)]
    #[test]
    fn the_msvc_verify_runs_in_cmd_and_powershell() {
        let verify = find_recipe("fix-15-msvc-redist")
            .expect("fix-15 is in the catalog")
            .verify;
        for run in run_typed_in_cmd_and_powershell(verify, &[]) {
            let ShellRun {
                shell,
                success,
                status,
                stdout: text,
                stderr: err,
            } = run;
            assert!(success, "{shell}: {status}\n{text}\n{err}");
            assert!(err.trim().is_empty(), "{shell} wrote to stderr: {err}");
            // `Select-Object FileName,FileVersion` prints a table with one row
            // per DLL: the path, then the version, on one line. Check each DLL
            // on its own row, so a pairing that lost or crossed a version (a
            // row with a name and no version, or both names on one row) fails.
            // `vcruntime140.dll` is not a substring of `vcruntime140_1.dll`,
            // so each name picks out exactly one row. This proves each listed
            // DLL has a dotted version beside it; it cannot prove the version
            // is the right one for that file. A path wrapped onto a second
            // line would fail here, and then the test is wrong, not the line.
            let is_version = |w: &str| {
                let parts: Vec<&str> = w.split('.').collect();
                parts.len() >= 3
                    && parts
                        .iter()
                        .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
            };
            for dll in ["vcruntime140.dll", "vcruntime140_1.dll"] {
                let rows: Vec<&str> = text.lines().filter(|l| l.contains(dll)).collect();
                assert_eq!(
                    rows.len(),
                    1,
                    "{shell}: expected one row for {dll}:\n{text}"
                );
                let row = rows[0];
                assert!(
                    row.split_whitespace().any(is_version),
                    "{shell}: the {dll} row has no dotted version:\n{text}"
                );
                let other = if dll == "vcruntime140.dll" {
                    "vcruntime140_1.dll"
                } else {
                    "vcruntime140.dll"
                };
                assert!(
                    !row.contains(other),
                    "{shell}: {dll} and {other} share a row:\n{text}"
                );
            }
        }
    }

    /// A DLL that is missing must fail the fix-15 verify. `Get-Item` reports a
    /// missing path as a non-terminating error, and PowerShell exits 0 after one,
    /// so without `-ErrorAction Stop` the line printed an error for the missing
    /// DLL and still exited 0. The line is run with the name of the second DLL
    /// changed to one that does not exist; the system directory cannot be
    /// redirected to one without it, because PowerShell does not start under a
    /// `SystemRoot` that is not the real one. Only compiled on Windows.
    #[cfg(windows)]
    #[test]
    fn the_msvc_verify_exits_nonzero_when_a_dll_is_missing() {
        let verify = find_recipe("fix-15-msvc-redist")
            .expect("fix-15 is in the catalog")
            .verify;
        let broken = verify.replace("vcruntime140_1.dll", "vcruntime140_absent.dll");
        assert_ne!(
            broken, verify,
            "the recipe no longer names vcruntime140_1.dll"
        );
        for run in run_typed_in_cmd_and_powershell(&broken, &[]) {
            let ShellRun {
                shell,
                success,
                stdout,
                stderr,
                ..
            } = run;
            assert!(
                !success,
                "{shell} exited 0 with a DLL missing:\n{stdout}\n{stderr}"
            );
            assert!(
                stderr.contains("vcruntime140_absent.dll"),
                "{shell} did not name the missing DLL on stderr:\n{stderr}"
            );
        }
    }

    /// fix-21's verify line, run typed into cmd and into Windows PowerShell:
    /// it prints the dotted version of `msvcp140.dll`. With the DLL's name
    /// changed to one that does not exist, both shells must exit non-zero and
    /// name the file: `Get-Item` reports a missing path as a non-terminating
    /// error, so without `-ErrorAction Stop` the line exits 0 with no output
    /// (the case `the_msvc_verify_exits_nonzero_when_a_dll_is_missing` covers
    /// for fix-15). `diagnose.rs` holds its copy of the line equal to this one.
    /// Only compiled on Windows.
    #[cfg(windows)]
    #[test]
    fn the_outdated_msvc_verify_prints_a_version_and_fails_when_the_dll_is_missing() {
        let verify = find_recipe("fix-21-msvc-redist-outdated")
            .expect("fix-21 is in the catalog")
            .verify;
        for run in run_typed_in_cmd_and_powershell(verify, &[]) {
            let ShellRun {
                shell,
                success,
                status,
                stdout,
                stderr,
            } = run;
            assert!(success, "{shell}: {status}\n{stdout}\n{stderr}");
            assert!(
                stderr.trim().is_empty(),
                "{shell} wrote to stderr: {stderr}"
            );
            let version = stdout.trim();
            let parts: Vec<&str> = version.split('.').collect();
            assert!(
                parts.len() >= 3
                    && parts
                        .iter()
                        .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())),
                "{shell}: not a dotted version: {stdout}"
            );
        }
        let broken = verify.replace("msvcp140.dll", "msvcp140_absent.dll");
        assert_ne!(broken, verify, "the recipe no longer names msvcp140.dll");
        for run in run_typed_in_cmd_and_powershell(&broken, &[]) {
            let ShellRun {
                shell,
                success,
                stdout,
                stderr,
                ..
            } = run;
            assert!(
                !success,
                "{shell} exited 0 with the DLL missing:\n{stdout}\n{stderr}"
            );
            assert!(
                stderr.contains("msvcp140_absent.dll"),
                "{shell} did not name the missing DLL on stderr:\n{stderr}"
            );
        }
    }

    /// Runs the fix-2 verify line typed into cmd and into Windows PowerShell,
    /// once with the override unset and once with it set. Python is on PATH on
    /// the Windows CI hosts. Only compiled on Windows; the line's shape is
    /// checked on every host by `the_override_verify_is_one_line_for_every_shell`.
    #[cfg(windows)]
    #[test]
    fn the_override_verify_runs_in_cmd_and_powershell() {
        let verify = find_recipe("fix-2-unset-override").unwrap().verify;
        for (value, expected) in [(None, "OK_UNSET"), (Some("11.0.0"), "11.0.0")] {
            for run in
                run_typed_in_cmd_and_powershell(verify, &[("HSA_OVERRIDE_GFX_VERSION", value)])
            {
                let ShellRun {
                    shell,
                    success,
                    status,
                    stdout: text,
                    stderr: err,
                } = run;
                assert!(success, "{shell}: {status}\n{text}\n{err}");
                assert_eq!(text.trim(), expected, "{shell}, stderr: {err}");
            }
        }
    }

    /// The text of fix-13's verify, taken apart. Runs on every host. It reads
    /// `HIP_PATH`, joins `bin\hipInfo.exe` onto it as a single-quoted string,
    /// and keeps the first lines; `the_hip_sdk_verify_runs_the_exe_under_hip_path`
    /// runs it, on Windows.
    #[test]
    fn the_hip_sdk_verify_reads_hip_path_and_runs_hipinfo() {
        let verify = find_recipe("fix-13-hip-sdk-missing").unwrap().verify;
        let body = verify
            .strip_prefix("powershell -NoProfile -Command \"")
            .and_then(|rest| rest.strip_suffix('"'))
            .unwrap_or_else(|| panic!("not a quoted `powershell -Command` argument: {verify}"));
        assert!(
            !body.contains('"'),
            "an inner quote ends the argument: {body}"
        );
        assert!(
            body.starts_with("& (Join-Path ([Environment]::GetEnvironmentVariable('HIP_PATH')) 'bin\\hipInfo.exe')")
                && body.ends_with("| Select-Object -First 5"),
            "{body}"
        );
    }

    /// The Where-Object stage of fix-14's verify, as written in the recipe.
    /// The Windows test and the shape test both take the filter from here, so
    /// neither keeps a copy of it.
    fn adrenalin_filter(verify: &str) -> &str {
        verify
            .split(" | ")
            .find(|stage| stage.starts_with("Where-Object"))
            .unwrap_or_else(|| panic!("fix-14's verify has no Where-Object stage: {verify}"))
    }

    /// Runs on every host: the filter that picks the AMD adapter is the one
    /// the Windows test exercises, so it has to be found and be the AMD match.
    #[test]
    fn the_adrenalin_verify_filters_on_amd_or_radeon() {
        let verify = find_recipe("fix-14-adrenalin-too-old").unwrap().verify;
        assert_eq!(
            adrenalin_filter(verify),
            "Where-Object Name -match 'AMD|Radeon'"
        );
    }

    /// Runs the fix-13 verify line in cmd and in Windows PowerShell against a
    /// throwaway SDK directory whose name contains a space: the single-quoted
    /// Join-Path form exists to survive one. `whoami.exe` stands in for
    /// `hipInfo.exe`; nothing else is reachable through `HIP_PATH`, so output
    /// means the line ran the file it built from `HIP_PATH`. With `HIP_PATH`
    /// pointing at a directory that has no `hipInfo.exe`, the line must fail
    /// and say so, which shows the variable is the one it reads. Only compiled
    /// on Windows.
    #[cfg(windows)]
    #[test]
    fn the_hip_sdk_verify_runs_the_exe_under_hip_path() {
        let verify = find_recipe("fix-13-hip-sdk-missing").unwrap().verify;
        let root = std::env::temp_dir().join(format!("rocm hip sdk {}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("bin")).expect("the SDK directory is created");
        let system_root = std::env::var("SystemRoot").expect("SystemRoot is set on Windows");
        std::fs::copy(
            std::path::Path::new(&system_root).join("System32\\whoami.exe"),
            root.join("bin").join("hipInfo.exe"),
        )
        .expect("a stand-in hipInfo.exe is copied");
        let empty = root.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let hip_path = root.to_string_lossy().into_owned();
        let empty_path = empty.to_string_lossy().into_owned();

        let present =
            run_typed_in_cmd_and_powershell(verify, &[("HIP_PATH", Some(hip_path.as_str()))]);
        let missing =
            run_typed_in_cmd_and_powershell(verify, &[("HIP_PATH", Some(empty_path.as_str()))]);
        let _ = std::fs::remove_dir_all(&root);

        for run in present {
            let ShellRun {
                shell,
                success,
                status,
                stdout,
                stderr,
            } = run;
            assert!(success, "{shell}: {status}\n{stdout}\n{stderr}");
            assert!(
                stderr.trim().is_empty(),
                "{shell} wrote to stderr: {stderr}"
            );
            assert!(
                !stdout.trim().is_empty(),
                "{shell} printed nothing from the stand-in hipInfo.exe"
            );
        }
        for run in missing {
            let ShellRun {
                shell,
                success,
                stdout,
                stderr,
                ..
            } = run;
            assert!(
                !(success && stderr.trim().is_empty()) && stdout.trim().is_empty(),
                "{shell} ran something with no hipInfo.exe under HIP_PATH:\n{stdout}\n{stderr}"
            );
        }
    }

    /// Runs the fix-22 verify line typed into cmd and into Windows PowerShell:
    /// both must parse it and exit 0 with nothing on stderr. The line used a
    /// `{ $_.Name ... }` filter, and `$_` inside the double-quoted `-Command`
    /// body is expanded by an outer PowerShell to nothing, so it failed with
    /// "The term '.Name' is not recognized" on every host. Only compiled on
    /// Windows.
    #[cfg(windows)]
    #[test]
    fn the_display_driver_verify_runs_in_cmd_and_powershell() {
        let verify = find_recipe("fix-22-display-driver-old").unwrap().verify;
        for run in run_typed_in_cmd_and_powershell(verify, &[]) {
            let ShellRun {
                shell,
                success,
                status,
                stdout,
                stderr,
            } = run;
            assert!(success, "{shell}: {status}\n{stdout}\n{stderr}");
            assert!(
                stderr.trim().is_empty(),
                "{shell} wrote to stderr: {stderr}"
            );
        }
    }

    /// Runs the fix-14 verify line in cmd and in Windows PowerShell: both must
    /// parse it and exit 0 with nothing on stderr. Its output is empty on a
    /// host with no AMD adapter, so that alone would pass for a filter that
    /// matches nothing. The filter text is therefore taken from the recipe and
    /// run over made-up adapters; only the AMD ones may come out. Only compiled
    /// on Windows.
    #[cfg(windows)]
    #[test]
    fn the_adrenalin_verify_runs_and_its_filter_keeps_only_amd() {
        let verify = find_recipe("fix-14-adrenalin-too-old").unwrap().verify;
        for run in run_typed_in_cmd_and_powershell(verify, &[]) {
            let ShellRun {
                shell,
                success,
                status,
                stdout,
                stderr,
            } = run;
            assert!(success, "{shell}: {status}\n{stdout}\n{stderr}");
            assert!(
                stderr.trim().is_empty(),
                "{shell} wrote to stderr: {stderr}"
            );
        }
        let filter = adrenalin_filter(verify);
        let synthetic = format!(
            "@([pscustomobject]@{{Name='AMD Radeon X'}},[pscustomobject]@{{Name='Other'}},\
             [pscustomobject]@{{Name='Intel UHD'}},[pscustomobject]@{{Name='Radeon Pro W'}}) \
             | {filter} | ForEach-Object {{ $_.Name }}"
        );
        let run = run_typed_in_powershell(&synthetic, &[]);
        assert!(
            run.success,
            "{}\n{}\n{}",
            run.status, run.stdout, run.stderr
        );
        let names: Vec<&str> = run.stdout.lines().map(str::trim).collect();
        assert_eq!(names, ["AMD Radeon X", "Radeon Pro W"], "{}", run.stderr);
    }

    /// The cases the rule is held to live in one file that the e2e crate's copy
    /// of the rule reads too, so the two cannot disagree on a listed line.
    const RULE_CASES: &str =
        include_str!("../../../../tests/e2e-cucumber/fixtures/powershell_text_rule_cases.txt");

    #[test]
    fn the_powershell_text_rule_agrees_with_the_shared_case_table() {
        let mut flagged = 0;
        let mut passed = 0;
        for (number, raw) in RULE_CASES.lines().enumerate() {
            let raw = raw.trim_end();
            if raw.is_empty() || raw.starts_with('#') {
                continue;
            }
            if let Some(line) = raw.strip_prefix("flag ") {
                assert!(
                    fails_in_windows_powershell(line).is_some(),
                    "line {}: `{line}`",
                    number + 1
                );
                flagged += 1;
            } else if let Some(line) = raw.strip_prefix("pass ") {
                assert_eq!(
                    fails_in_windows_powershell(line),
                    None,
                    "line {}: `{line}`",
                    number + 1
                );
                passed += 1;
            } else {
                panic!("line {}: neither `flag` nor `pass`: {raw}", number + 1);
            }
        }
        // The table was read, not skipped.
        assert!(
            flagged >= 20 && passed >= 15,
            "{flagged} flag, {passed} pass"
        );
        // The catalog's own fix-2 line is one of the lines it must let through.
        let fix_2 = find_recipe("fix-2-unset-override").unwrap().verify;
        assert_eq!(fails_in_windows_powershell(fix_2), None, "{fix_2}");
    }

    /// The fix-2 verify line is shown on Linux, WSL and Windows, so it must be
    /// one command that bash, cmd and PowerShell all run the same way: a
    /// `python -c` whose program is one double-quoted argument with no inner
    /// double quote and no `$`, so no shell expands anything in it.
    #[test]
    fn the_override_verify_is_one_line_for_every_shell() {
        let verify = find_recipe("fix-2-unset-override").unwrap().verify;
        let program = verify
            .strip_prefix("python -c \"")
            .and_then(|rest| rest.strip_suffix('"'))
            .unwrap_or_else(|| panic!("not a quoted `python -c` argument: {verify}"));
        assert!(!program.contains(['"', '$', '`', '%']), "{program}");
        assert!(
            program.contains("os.environ.get('HSA_OVERRIDE_GFX_VERSION')")
                && program.contains("'OK_UNSET'"),
            "{program}"
        );
    }

    #[test]
    fn a_needs_argument_platform_names_the_argument() {
        for (fix_id, os, class) in classes() {
            let needs = RECIPES
                .iter()
                .find(|r| r.fix_id == fix_id)
                .and_then(|r| r.platform(os))
                .and_then(|p| p.needs);
            assert_eq!(
                class == FixClass::NeedsArgument,
                needs.is_some(),
                "{fix_id} on {os}: the argument is the whole difference between this \
                 class and AUTO, so it has to be named -- and nothing else may name one"
            );
        }
    }

    #[test]
    fn only_these_entries_act_on_the_machine_and_only_on_these_platforms() {
        let acts: Vec<(&str, &str, FixClass)> = classes()
            .into_iter()
            .filter(|(_, _, c)| matches!(c, FixClass::Auto | FixClass::NeedsArgument))
            .collect();
        assert_eq!(
            acts,
            vec![
                // Linux reports and returns; only the Windows arm mutates.
                ("fix-2-unset-override", "windows", FixClass::Auto),
                ("fix-4-render-group", "linux", FixClass::Auto),
                ("fix-6-path", "linux", FixClass::Auto),
                ("fix-6-path", "windows", FixClass::Auto),
                ("fix-6-path", "wsl", FixClass::Auto),
                // Both arms short-circuit until `--device-index` names a target.
                ("fix-9-igpu-dgpu", "linux", FixClass::NeedsArgument),
                ("fix-9-igpu-dgpu", "windows", FixClass::NeedsArgument),
            ]
        );
        // fix-16-vllm-oom carries a workaround the user must weigh (lowering the
        // VRAM reservation is wrong for a model that genuinely does not fit), so
        // it is deliberately PRINT-ONLY and must never join the AUTO set.
        let oom = find_recipe("fix-16-vllm-oom").expect("OOM recipe is in the catalog");
        assert!(
            oom.applies_on
                .iter()
                .all(|p| p.class == FixClass::PrintOnly),
            "fix-16-vllm-oom must stay PRINT-ONLY on every platform it applies to"
        );
        assert!(oom.runner.is_none(), "a PRINT-ONLY recipe has no runner");
    }

    #[test]
    fn no_entry_claims_a_platform_twice() {
        for r in RECIPES {
            let mut seen: Vec<&str> = r.applies_on.iter().map(|p| p.os).collect();
            let count = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                count,
                "{}: a platform listed twice makes `platform()` order-dependent",
                r.fix_id
            );
        }
    }

    #[test]
    fn the_old_display_driver_recipe_is_print_only_advice_on_windows() {
        let recipe = RECIPES
            .iter()
            .find(|r| r.fix_id == crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID)
            .expect("the recipe is registered");
        // rocm-cli never installs drivers on Windows: every printed line is a
        // comment, and the only platform is Windows, print-only.
        assert!(recipe.commands.iter().all(|c| c.starts_with('#')));
        assert!(recipe.commands.iter().any(|c| c.contains("Restart")));
        assert!(recipe.runner.is_none());
        assert_eq!(recipe.applies_on.len(), 1);
        assert_eq!(recipe.applies_on[0].os, "windows");
        assert_eq!(recipe.applies_on[0].class, FixClass::PrintOnly);
        assert!(recipe.rationale.contains("not established"));
        // The rationale spells the cutoff out in prose; it must name the same
        // day as the constant the comparison uses.
        let cutoff = crate::windows_driver::FIRST_HIP_SDK_FOR_WINDOWS;
        assert_eq!(
            (cutoff.year, cutoff.month, cutoff.day),
            (2023, 7, 27),
            "the cutoff moved: update the \"27 July 2023\" prose in this recipe, README.md,              skills/rocm-doctor/reference.md, docs/testing.md and docs/manual-testing.md"
        );
        assert!(recipe.rationale.contains("27 July 2023"));
        // The literal URL in the commands is the shared constant.
        let url_line = format!("#   {}", crate::windows_driver::AMD_SUPPORT_URL);
        assert!(recipe.commands.contains(&url_line.as_str()));
        assert_eq!(recipe.doc_url, Some(crate::windows_driver::AMD_SUPPORT_URL));
    }

    #[test]
    fn needs_reboot_true_fix_ids_match_the_known_set() {
        // assert_needs_reboot_matches_the_catalog only ever runs for
        // fix-5-amdgpu-load, so every other fix-id's hardcoded needs_reboot
        // literals in diagnose.rs have no guard against drifting from the
        // catalog. This doesn't reach into diagnose.rs, but it does catch an
        // accidental catalog edit and pins the expected set by name so a
        // deliberate catalog change forces a look at diagnose.rs's matching
        // literals too.
        let expected: std::collections::BTreeSet<&str> = [
            "fix-3-rocm-kernel",
            "fix-5-amdgpu-load",
            "fix-11-iommu",
            "fix-12-installer",
            "fix-14-adrenalin-too-old",
            "fix-22-display-driver-old",
        ]
        .into_iter()
        .collect();
        let actual: std::collections::BTreeSet<&str> = RECIPES
            .iter()
            .filter(|r| r.needs_reboot)
            .map(|r| r.fix_id)
            .collect();
        assert_eq!(
            actual, expected,
            "the catalog's needs_reboot:true set has changed -- update diagnose.rs's \
             matching literals and this test's expected set together"
        );
    }

    #[test]
    fn exit_code_1_is_unreachable_by_construction() {
        // apply() returns 1 only when a platform's class is Auto and the recipe
        // has no runner (`Plan::MissingRunner`). `a_recipe_that_acts_has_a_runner`
        // keeps that impossible; assert the reachability argument here so
        // reference.md's row 1 is accounted for rather than silently untested.
        assert!(
            RECIPES.iter().all(|r| {
                !r.applies_on.iter().any(|p| p.class.applies_itself()) || r.runner.is_some()
            }),
            "an auto-applicable recipe without a runner would make exit 1 reachable"
        );
    }

    #[test]
    fn fix_7_apt_commands_match_the_diagnose_copy_verbatim() {
        // Same drift guard as fix_12_apt_commands_match_the_diagnose_copy_verbatim:
        // diagnose.rs builds its own per-family command list for fix-7 rather than
        // reusing this recipe, so drive diagnose() itself instead of comparing
        // against a hardcoded literal that can't detect diagnose.rs drifting.
        let e = crate::examine::Examination {
            os_family: "linux".to_owned(),
            distro_id: "ubuntu".to_owned(),
            rocm_repos_seen: vec![
                "/etc/apt/sources.list.d/rocm.list".to_owned(),
                "/etc/apt/sources.list.d/amdgpu.list".to_owned(),
            ],
            ..crate::examine::Examination::default()
        };
        let report = crate::diagnose::diagnose(&e, "stale repo conflict");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-7-stale-repos")
            .expect("duplicate repo files should be diagnosed");
        let diagnose_apt_list_cmd = hit
            .fix
            .as_ref()
            .unwrap()
            .commands
            .first()
            .expect("fix-7 always lists the repo dir first on a recognized family")
            .as_str();

        let recipe = RECIPES
            .iter()
            .find(|r| r.fix_id == "fix-7-stale-repos")
            .expect("fix-7-stale-repos recipe must exist");
        let recipe_apt_list_cmd = recipe
            .commands
            .iter()
            .find(|c| c.trim_end().ends_with("# Debian/Ubuntu") && c.contains("ls /etc/apt"))
            .map(|c| c.split('#').next().unwrap().trim_end())
            .expect("recipe must have a commented apt list line");
        assert_eq!(
            recipe_apt_list_cmd, diagnose_apt_list_cmd,
            "this recipe's apt list command must match what diagnose.rs's apt family actually emits for fix-7-stale-repos"
        );

        // `verify` must also track diagnose.rs's per-family text, not just
        // `commands`: diagnose.rs varies `verify` by family too, so compare
        // each family's line against what diagnose() actually emits for it.
        // The dnf/zypper `ls` lines get the same treatment as apt above --
        // only asserting the apt one left those two families' list commands
        // able to drift from the recipe undetected.
        for (family_repo_dir, marker) in [
            ("/etc/apt/sources.list.d/", "# Debian/Ubuntu"),
            ("/etc/yum.repos.d/", "# RHEL/Fedora/Rocky/Alma/Oracle"),
            ("/etc/zypp/repos.d/", "# SLES/openSUSE"),
        ] {
            let e = crate::examine::Examination {
                os_family: "linux".to_owned(),
                distro_id: "ubuntu".to_owned(),
                rocm_repos_seen: vec![
                    format!("{family_repo_dir}rocm.repo"),
                    format!("{family_repo_dir}amdgpu.repo"),
                ],
                ..crate::examine::Examination::default()
            };
            let report = crate::diagnose::diagnose(&e, "stale repo conflict");
            let hit = report
                .matched
                .iter()
                .find(|d| d.id == "fix-7-stale-repos")
                .expect("duplicate repo files should be diagnosed");
            let fix = hit.fix.as_ref().unwrap();
            let diagnose_list_cmd = fix
                .commands
                .first()
                .expect("fix-7 always lists the repo dir first on a recognized family")
                .as_str();
            let diagnose_verify = fix.verify.as_str();

            let recipe_list_cmd = recipe
                .commands
                .iter()
                .find(|c| c.trim_end().ends_with(marker) && c.contains("ls /"))
                .map_or_else(
                    || panic!("recipe must have a commented list line for {marker}"),
                    |c| c.split('#').next().unwrap().trim_end(),
                );
            assert_eq!(
                recipe_list_cmd, diagnose_list_cmd,
                "this recipe's {marker} list command must match what diagnose.rs emits for fix-7-stale-repos"
            );

            let recipe_verify_line = recipe
                .verify
                .lines()
                .find(|l| l.trim_end().ends_with(marker))
                .map_or_else(
                    || panic!("recipe verify must have a line for {marker}"),
                    |l| l.split('#').next().unwrap().trim(),
                );
            assert_eq!(
                recipe_verify_line, diagnose_verify,
                "this recipe's {marker} verify line must match what diagnose.rs emits for fix-7-stale-repos"
            );
        }
    }

    #[test]
    fn fix_12_recipe_clears_state_before_reinstall_and_keeps_mv_commented() {
        let recipe = RECIPES
            .iter()
            .find(|r| r.fix_id == "fix-12-installer")
            .expect("fix-12-installer recipe must exist");
        let commands = recipe.commands.join("\n");
        assert!(
            commands.contains("dpkg --configure -a"),
            "must clear the half-configured dpkg state before reinstalling: {commands}"
        );
        assert!(
            commands.contains("purge -y amdgpu-dkms"),
            "must purge the broken driver package: {commands}"
        );
        assert!(
            commands.contains("# sudo mv /etc/apt/sources.list.d/amdgpu.list"),
            "repo quarantine command must be commented out, not ready-to-run: {commands}"
        );
        assert!(
            commands.contains("rocm install driver --dkms --yes"),
            "reinstall command must actually install (bare 'rocm install driver' is a non-mutating preflight no-op): {commands}"
        );
        assert!(
            recipe.verify.contains("dpkg") && recipe.verify.contains("rpm"),
            "verify must query real dpkg/rpm package state, not the self-reported \
             rocm_install_method marker (that marker is set from repo-file presence \
             before the package actually installs): {}",
            recipe.verify
        );
        assert!(
            !recipe
                .commands
                .iter()
                .any(|c| c.contains("configure -a &&") || c.contains("install -f &&")),
            "the apt block must not chain configure/install -f into the purge with `&&`: \
             a failure there must not skip the purge: {commands}"
        );
    }

    #[test]
    fn fix_3_verify_matches_the_diagnose_copy_verbatim() {
        // No drift guard previously existed between this recipe and
        // diagnose.rs's `check_3_rocm_kernel_unsupported` for fix-3 (unlike
        // fix-7/fix-12's `commands`, fix-3's `commands` legitimately differ --
        // diagnose.rs injects the host's live kernel/distro/rocm values into a
        // comment this host-unaware recipe can't know -- but `verify` has no
        // such excuse to diverge).
        let e = crate::examine::Examination {
            os_family: "linux".to_owned(),
            rocm_version: "6.4.1".to_owned(),
            amdgpu_loaded: Some(false),
            ..crate::examine::Examination::default()
        };
        let report = crate::diagnose::diagnose(&e, "");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-3-rocm-kernel")
            .expect("rocm installed with amdgpu not loaded should be diagnosed");
        let diagnose_verify = hit.fix.as_ref().unwrap().verify.as_str();

        let recipe = RECIPES
            .iter()
            .find(|r| r.fix_id == "fix-3-rocm-kernel")
            .expect("fix-3-rocm-kernel recipe must exist");
        assert_eq!(
            recipe.verify, diagnose_verify,
            "this recipe's verify step must match what diagnose.rs emits for fix-3-rocm-kernel"
        );
    }

    #[test]
    fn fix_12_apt_commands_match_the_diagnose_copy_verbatim() {
        // `crate::diagnose` builds its own apt command list for fix-12 rather
        // than reusing this recipe (it branches per detected package-manager
        // family, which this catalog entry cannot do -- it prints guidance for
        // every family at once). Drive `diagnose()` itself so a drift between
        // the two copies actually fails this test, rather than comparing
        // against a hardcoded literal that can't detect diagnose.rs changing
        // out from under it.
        let e = crate::examine::Examination {
            os_family: "linux".to_owned(),
            distro_id: "ubuntu".to_owned(),
            distro_version: "24.04".to_owned(),
            rocm_install_method: "repo-native".to_owned(),
            ..crate::examine::Examination::default()
        };
        let report = crate::diagnose::diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let diagnose_apt_commands: Vec<&str> = hit
            .fix
            .as_ref()
            .unwrap()
            .commands
            .iter()
            .map(String::as_str)
            .filter(|c| !c.trim_start().starts_with('#'))
            .take(3)
            .collect();

        let recipe = RECIPES
            .iter()
            .find(|r| r.fix_id == "fix-12-installer")
            .expect("fix-12-installer recipe must exist");
        let apt_commands_without_trailing_comment: Vec<&str> = recipe
            .commands
            .iter()
            .filter(|c| c.trim_end().ends_with("# Debian/Ubuntu"))
            .map(|c| c.split('#').next().unwrap().trim_end())
            .take(3)
            .collect();
        assert_eq!(
            apt_commands_without_trailing_comment, diagnose_apt_commands,
            "this recipe's apt lines must match what diagnose.rs's apt family actually emits for fix-12-installer"
        );
    }
}
