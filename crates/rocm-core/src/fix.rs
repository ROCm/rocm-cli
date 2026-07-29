// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Apply remediations for diagnosed ROCm failure modes.
//!
//! Rust port of the `rocm-doctor` skill's `apply_fix.py`. Only small, safe,
//! well-bounded fixes are auto-applicable (the runners below); everything else
//! is advisory and only prints its plan. The consent model mirrors the Python:
//! print the exact change, honor `--dry-run`, refuse on a non-interactive shell
//! without `--yes`, and otherwise confirm before mutating anything.
//!
//! Exit codes match `apply_fix.py`: `0` ok/dry-run/print-only, `2` unknown id,
//! `3` environment/OS not right, `4` a command failed, `5` user declined.

use crate::examine::{run, which};
use crate::{runtime_is_linux, runtime_is_windows};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const RUN_TIMEOUT: Duration = Duration::from_mins(1);
const QUERY_TIMEOUT: Duration = Duration::from_secs(8);

/// Options controlling how a fix is applied.
#[derive(Debug, Clone, Default)]
pub struct FixOptions {
    /// Skip the interactive confirmation (the user already approved the plan).
    pub yes: bool,
    /// Show the plan without changing anything.
    pub dry_run: bool,
    /// For `fix-9-igpu-dgpu`: the discrete GPU index to pin.
    pub device_index: Option<i64>,
}

/// A remediation recipe keyed by the stable `fix-id`.
struct FixRecipe {
    fix_id: &'static str,
    title: &'static str,
    rationale: &'static str,
    auto_applicable: bool,
    commands: &'static [&'static str],
    needs_sudo: bool,
    needs_reboot: bool,
    needs_relogin: bool,
    verify: &'static str,
    notes: &'static [&'static str],
    applies_on: &'static [&'static str],
    runner: Option<fn(&FixOptions) -> i32>,
}

const LINUX_AND_WINDOWS: &[&str] = &["linux", "windows"];
const LINUX_ONLY: &[&str] = &["linux"];
const WINDOWS_ONLY: &[&str] = &["windows"];

/// The recipe registry. Mirrors the diagnosis catalog; only the four small,
/// safe fixes carry a `runner` and are auto-applicable.
const RECIPES: &[FixRecipe] = &[
    FixRecipe {
        fix_id: "fix-1-arch",
        title: "GPU gfx target not in framework arch list",
        rationale: "Your GPU's gfx target is not in the framework wheel's compiled kernel list. Re-install the framework from an index that includes this gfx, OR rebuild llama.cpp with AMDGPU_TARGETS=<gfx>.",
        auto_applicable: false,
        commands: &[
            "# PyTorch (Linux): switch to the ROCm nightly that ships the gfx115x kernels.",
            "pip uninstall -y torch torchvision torchaudio",
            "pip install --pre torch torchvision torchaudio \\",
            "  --index-url https://download.pytorch.org/whl/nightly/rocm6.4",
            "# PyTorch (Windows): use TheRock's per-gfx wheels (https://github.com/ROCm/TheRock).",
            "# llama.cpp:",
            "# cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<your_gfx_target>",
            "# cmake --build build -j",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())\"",
        notes: &[
            "TheRock per-gfx wheels are the recommended fallback when the official pytorch index does not yet cover your gfx (and the only first-party option on Windows AMD).",
            "HSA_OVERRIDE_GFX_VERSION is NOT the right fix here -- it papers over the mismatch and risks page faults at runtime.",
        ],
        applies_on: LINUX_AND_WINDOWS,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-2-unset-override",
        title: "Unset HSA_OVERRIDE_GFX_VERSION",
        rationale: "HSA_OVERRIDE_GFX_VERSION is set, but your GPU now has a native wheel. The override hides the real gfx and causes page faults / OUT_OF_REGISTERS at runtime.",
        auto_applicable: true,
        commands: &[
            "# Linux:",
            "unset HSA_OVERRIDE_GFX_VERSION",
            "# Then remove the line from ~/.bashrc / ~/.zshrc / ~/.profile.",
            "# Windows:",
            "setx HSA_OVERRIDE_GFX_VERSION \"\"",
            "# Or remove via System Properties -> Environment Variables.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "env | grep HSA_OVERRIDE_GFX_VERSION || echo OK_UNSET",
        notes: &[],
        applies_on: LINUX_AND_WINDOWS,
        runner: Some(run_unset_override),
    },
    FixRecipe {
        fix_id: "fix-3-rocm-kernel",
        title: "ROCm/distro/kernel triple unsupported",
        rationale: "ROCm is installed but your kernel/distro combination is outside the supported matrix. Match the kernel to the matrix before reinstalling, or rerun with --no-dkms and accept the risk.",
        auto_applicable: false,
        commands: &[
            "# Cross-check the live AMD matrix before changing anything:",
            "#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
            "# Common fix on Ubuntu: install the HWE kernel that matches your ROCm release, then reboot.",
        ],
        needs_sudo: false,
        needs_reboot: true,
        needs_relogin: false,
        verify: "lsmod | grep amdgpu && rocminfo | head -n 5",
        notes: &[],
        applies_on: LINUX_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-4-render-group",
        title: "Add user to render/video groups",
        rationale: "The current user can't open /dev/kfd because they aren't in the render group. Adding the user is the safe, standard fix.",
        auto_applicable: true,
        commands: &["sudo usermod -a -G render,video \"$USER\""],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: true,
        verify: "groups | tr ' ' '\\n' | grep -E '^(render|video)$' && rocminfo | head -n 5",
        notes: &[],
        applies_on: LINUX_ONLY,
        runner: Some(run_render_group),
    },
    FixRecipe {
        fix_id: "fix-5-amdgpu-load",
        title: "Load amdgpu (and clear any blacklist)",
        rationale: "The amdgpu kernel module is not loaded. Check /etc/modprobe.d for a blacklist entry, regenerate the initramfs, and modprobe.",
        auto_applicable: false,
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
        notes: &[
            "If Secure Boot is enabled and amdgpu still won't load, the DKMS module isn't signed. Either sign it with mokutil or disable Secure Boot in firmware.",
        ],
        applies_on: LINUX_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-6-path",
        title: "Add the ROCm/HIP bin directory to PATH",
        rationale: "Linux: ROCm is installed but its bin directory isn't on PATH, so `rocminfo` / `hipcc` aren't visible to the shell. Windows: the HIP SDK is installed but its bin directory isn't on the User PATH, so `hipInfo.exe` and the runtime DLLs can't be found.",
        auto_applicable: true,
        commands: &[
            "# Linux:",
            "echo 'export PATH=\"/opt/rocm/bin:$PATH\"' >> ~/.bashrc",
            "# Windows:",
            "setx PATH \"%PATH%;C:\\Program Files\\AMD\\ROCm\\<version>\\bin\"",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "rocminfo | head -n 5 && hipcc --version",
        notes: &[],
        applies_on: LINUX_AND_WINDOWS,
        runner: Some(run_path_export),
    },
    FixRecipe {
        fix_id: "fix-7-stale-repos",
        title: "Quarantine duplicate AMD repos",
        rationale: "More than one ROCm/AMDGPU repo file exists. The package manager is mixing versions; quarantine the extras before reinstalling.",
        auto_applicable: false,
        commands: &[
            "ls /etc/apt/sources.list.d/ | grep -iE 'rocm|amdgpu|radeon'",
            "# For each duplicate file:",
            "sudo mv /etc/apt/sources.list.d/<file>.list /etc/apt/sources.list.d/<file>.list.bak",
            "sudo apt update",
        ],
        needs_sudo: true,
        needs_reboot: false,
        needs_relogin: false,
        verify: "sudo apt update 2>&1 | tail -n 20",
        notes: &[],
        applies_on: LINUX_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-8-wheel-rocm",
        title: "Reinstall the framework against the system ROCm/HIP major",
        rationale: "The framework's bundled HIP version doesn't match the system ROCm (Linux) or HIP SDK (Windows). libamdhip64.so.X / amdhip64_X.dll load failures are the usual signal.",
        auto_applicable: false,
        commands: &[
            "pip uninstall -y torch torchvision torchaudio",
            "# Linux: pick the index that matches your system ROCm major:",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3",
            "# Windows: use TheRock's wheels matching your HIP SDK major:",
            "#   https://github.com/ROCm/TheRock",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "python -c \"import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())\"",
        notes: &[],
        applies_on: LINUX_AND_WINDOWS,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-9-igpu-dgpu",
        title: "Hide the iGPU with HIP_VISIBLE_DEVICES",
        rationale: "Both an APU iGPU and a discrete AMD GPU are visible. Pin the runtime to the dGPU so the iGPU doesn't destabilise it.",
        auto_applicable: true,
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
        notes: &[
            "Pass --device-index N to persist the env var; without it, this fix only prints the rocminfo / hipInfo query so you can identify N.",
        ],
        applies_on: LINUX_AND_WINDOWS,
        runner: Some(run_hip_visible_devices),
    },
    FixRecipe {
        fix_id: "fix-10-container",
        title: "Re-launch the container with AMD devices passed through",
        rationale: "The container can't see /dev/kfd or /dev/dri/renderD*. Pass the devices and the host's render group via the runtime flags.",
        auto_applicable: false,
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
        notes: &[
            "Rootless podman additionally needs `--userns=keep-id` and a host user that is in the render group; podman maps it through.",
        ],
        applies_on: LINUX_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-11-iommu",
        title: "Add iommu=pt to the kernel command line",
        rationale: "Multi-GPU jobs hang when the IOMMU is in the default 'on' mode with translation; pass-through mode fixes the hang. This requires editing GRUB and rebooting; we will not do that for you.",
        auto_applicable: false,
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
        notes: &[],
        applies_on: LINUX_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-12-installer",
        title: "Reset amdgpu-install state and reinstall",
        rationale: "amdgpu-install left a half-configured DKMS / repo state. Run the documented uninstall, clean up, and reinstall without the flag that broke things (commonly --accept-eula on newer installers).",
        auto_applicable: false,
        commands: &[
            "sudo amdgpu-install --uninstall",
            "sudo apt autoremove --purge -y",
            "sudo apt update",
            "sudo amdgpu-install --usecase=rocm,hip",
        ],
        needs_sudo: true,
        needs_reboot: true,
        needs_relogin: false,
        verify: "dpkg -l | grep -E 'rocm|amdgpu' | head -n 20 && rocminfo | head -n 5",
        notes: &[
            "If `apt autoremove --purge` warns it will remove unrelated packages, stop and resolve those by hand before continuing.",
        ],
        applies_on: LINUX_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-13-hip-sdk-missing",
        title: "Install the AMD HIP SDK for Windows",
        rationale: "Your framework links against HIP but the HIP SDK isn't installed on this host. The runtime DLLs (amdhip64_X.dll, hipblas.dll, hsa-runtime64.dll) and hipInfo.exe ship inside the SDK installer.",
        auto_applicable: false,
        commands: &[
            "# Download and install the HIP SDK (matched to your framework's HIP major):",
            "#   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html",
            "# After install, reopen the shell so HIP_PATH and PATH pick up the new install.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "powershell -NoProfile -Command \"& \\\"$env:HIP_PATH\\bin\\hipInfo.exe\\\" | Select-Object -First 5\"",
        notes: &[
            "If you only need PyTorch on Windows AMD and don't need the C/C++ HIP toolchain, the TheRock wheels bundle their own HIP runtime and may not require a system HIP SDK install.",
        ],
        applies_on: WINDOWS_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-14-adrenalin-too-old",
        title: "Update the Adrenalin / kernel-mode driver",
        rationale: "The HIP SDK is installed but the AMD kernel-mode driver (Adrenalin / Adrenalin Pro) is older than the SDK release notes call out. The user-space SDK and the driver have to match.",
        auto_applicable: false,
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
        verify: "powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | Select-Object -First 1).DriverVersion\"",
        notes: &[],
        applies_on: WINDOWS_ONLY,
        runner: None,
    },
    FixRecipe {
        fix_id: "fix-15-msvc-redist",
        title: "Install the MSVC 2015-2022 runtime redistributable",
        rationale: "The HIP SDK's amdhip64_X.dll links against the MSVC 2015-2022 runtime. When vcruntime140.dll / vcruntime140_1.dll aren't on PATH, `import torch` fails with a missing-DLL error that points at vcruntime140_1.dll, not at the HIP runtime itself.",
        auto_applicable: false,
        commands: &[
            "# Download and install (x64):",
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe",
            "# After the install, reopen the shell and re-run your import / hipInfo check.",
        ],
        needs_sudo: false,
        needs_reboot: false,
        needs_relogin: false,
        verify: "where vcruntime140.dll && where vcruntime140_1.dll",
        notes: &[
            "If installing the redistributable still leaves a missing-DLL error, the failing DLL is probably amdhip64_X.dll itself; that points at fix-13-hip-sdk-missing rather than this fix.",
        ],
        applies_on: WINDOWS_ONLY,
        runner: None,
    },
];

fn find_recipe(fix_id: &str) -> Option<&'static FixRecipe> {
    RECIPES.iter().find(|r| r.fix_id == fix_id)
}

const fn current_os() -> &'static str {
    if runtime_is_windows() {
        "windows"
    } else if runtime_is_linux() {
        "linux"
    } else {
        "other"
    }
}

/// Whether `value` is a `rocm diagnose` ranking position (`#2`, or a bare `2`)
/// rather than a fix-id.
///
/// Used only to turn an unknown-id refusal into a corrective one; it never
/// selects a fix, because a position is meaningful only within the report that
/// produced it and `fix` has no memory of that report.
fn looks_like_a_diagnosis_position(value: &str) -> bool {
    let trimmed = value.trim();
    let digits = trimmed.strip_prefix('#').unwrap_or(trimmed);
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

/// Whether the CLI will apply `fix_id` itself, or `None` if it isn't a known
/// fix. `RECIPES` is the authority: `apply()` dispatches on it, so this is the
/// value any other surface describing a fix has to agree with.
///
/// Test-only: the one production consumer is `apply()`, which reads the recipe
/// directly. This exists so `diagnose`'s tests can assert the two surfaces
/// agree without exposing `RECIPES`.
#[cfg(test)]
pub(crate) fn auto_applicable_for(fix_id: &str) -> Option<bool> {
    find_recipe(fix_id).map(|r| r.auto_applicable)
}

/// List every fix-id (id, kind, OS scope, title).
#[must_use]
pub fn list_recipes() -> String {
    use std::fmt::Write as _;
    let mut out = String::from("Available fix-ids (mirror the diagnosis catalog):\n");
    // The markers were printed with nothing saying what they mean.
    out.push_str(
        "  AUTO = `rocm fix <id>` can carry it out; PRINT-ONLY = it prints the steps for you to run.\n",
    );
    for r in RECIPES {
        let kind = if r.auto_applicable {
            "AUTO"
        } else {
            "PRINT-ONLY"
        };
        let scope = r.applies_on.join("/");
        let _ = writeln!(
            out,
            "  [{kind:>10}] [{scope:>14}] {}  -- {}",
            r.fix_id, r.title
        );
    }
    out
}

fn print_recipe(r: &FixRecipe) {
    println!("Fix:        {}  -- {}", r.fix_id, r.title);
    println!("OS scope:   {}", r.applies_on.join(", "));
    println!("Rationale:  {}", r.rationale);
    if !r.commands.is_empty() {
        println!("Commands:");
        for c in r.commands {
            println!("  $ {c}");
        }
    }
    let mut flags = Vec::new();
    if r.needs_sudo {
        flags.push("requires sudo");
    }
    if r.needs_reboot {
        flags.push("requires reboot");
    }
    if r.needs_relogin {
        flags.push("requires re-login");
    }
    if !r.auto_applicable {
        flags.push("manual only (this command will NOT run it)");
    }
    if !flags.is_empty() {
        println!("Flags:      {}", flags.join(", "));
    }
    for n in r.notes {
        println!("Note:       {n}");
    }
    if !r.verify.is_empty() {
        println!("Verify:     {}", r.verify);
    }
}

/// Apply (or print) the fix identified by `fix_id`. Returns the process exit code.
#[must_use]
pub fn apply(fix_id: &str, opts: &FixOptions) -> i32 {
    let Some(recipe) = find_recipe(fix_id) else {
        eprintln!("Unknown fix-id: {fix_id}");
        if looks_like_a_diagnosis_position(fix_id) {
            // `rocm diagnose` ranks findings `#1`, `#2`, and users reach for that
            // number here. It is a position in one report, not a name -- and it
            // does not line up with the catalog's `fix-1 … fix-15` either, so a
            // bare "unknown id" left them with nothing to correct.
            eprintln!(
                "`{fix_id}` looks like a position in a `rocm diagnose` report, not a fix-id."
            );
            eprintln!(
                "Use the `id:` shown against that cause — `rocm diagnose` prints an `apply with:` line you can copy."
            );
        } else {
            eprintln!("Run `rocm diagnose` to see which fix-id applies.");
        }
        return 2;
    };
    print_recipe(recipe);
    println!();

    let os = current_os();
    if !recipe.applies_on.contains(&os) {
        println!(
            "This fix only applies on: {}. Running OS is: {os}.",
            recipe.applies_on.join(", ")
        );
        return 3;
    }
    if !recipe.auto_applicable {
        println!("This fix is print-only (manual change required).");
        println!("Copy the commands above, run them yourself, then verify with:");
        if !recipe.verify.is_empty() {
            println!("  $ {}", recipe.verify);
        }
        return 0;
    }
    if let Some(runner) = recipe.runner {
        runner(opts)
    } else {
        // Internal error (auto-applicable recipe with no runner) -> 1, not 4
        // (4 is reserved for "attempted but the command failed").
        eprintln!("Internal error: auto-applicable recipe has no runner.");
        1
    }
}

// ---------------------------------------------------------------------------
// Consent / environment helpers
// ---------------------------------------------------------------------------

/// The half of the consent decision that needs no I/O: `Some(verdict)` when the
/// answer is already determined, `None` when the user must actually be asked.
///
/// Split out from [`confirm`] so the two rules that matter — `--yes` approves,
/// and a non-interactive shell *refuses* rather than silently proceeding — are
/// testable without a controlled stdin. A test that drove `confirm` directly
/// would block on `read_line` whenever the suite happened to run with a terminal
/// attached.
const fn consent_without_prompt(assume_yes: bool, stdin_is_terminal: bool) -> Option<bool> {
    if assume_yes {
        return Some(true);
    }
    if !stdin_is_terminal {
        return Some(false);
    }
    None
}

fn confirm(prompt: &str, assume_yes: bool) -> bool {
    if let Some(verdict) = consent_without_prompt(assume_yes, std::io::stdin().is_terminal()) {
        if !verdict {
            println!("Non-interactive shell and --yes not passed; refusing to apply.");
        }
        return verdict;
    }
    print!("{prompt} [y/N]: ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn is_root() -> bool {
    run("id", &["-u"], QUERY_TIMEOUT).1.trim() == "0"
}

/// Pick the shell rc file to append to (.zshrc for zsh, else .bashrc).
fn shell_rc_file() -> Option<PathBuf> {
    let home = home_dir()?;
    let shell = std::env::var("SHELL").unwrap_or_default();
    let primary = if shell.contains("zsh") {
        home.join(".zshrc")
    } else {
        home.join(".bashrc")
    };
    if !primary.exists() && home.join(".bashrc").exists() {
        Some(home.join(".bashrc"))
    } else {
        Some(primary)
    }
}

fn append_line(path: &Path, header: &str, line: &str) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "\n{header}")?;
    writeln!(file, "{line}")
}

// ---------------------------------------------------------------------------
// Runners (one per auto-applicable fix)
// ---------------------------------------------------------------------------

/// fix-4: add the current user to the render group (and 'video' for safety).
fn run_render_group(opts: &FixOptions) -> i32 {
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_default();
    if user.is_empty() {
        println!("Could not determine current user from $USER/$LOGNAME.");
        return 3;
    }
    if !which("usermod") {
        println!("`usermod` not on PATH; cannot add groups.");
        return 3;
    }
    let root = is_root();
    if !which("sudo") && !root {
        println!("`sudo` is not on PATH and we are not root; cannot add groups.");
        return 3;
    }
    let (program, args): (&str, Vec<String>) = if root {
        (
            "usermod",
            vec![
                "-a".into(),
                "-G".into(),
                "render,video".into(),
                user.clone(),
            ],
        )
    } else {
        (
            "sudo",
            vec![
                "usermod".into(),
                "-a".into(),
                "-G".into(),
                "render,video".into(),
                user.clone(),
            ],
        )
    };
    println!("Will run: {program} {}", args.join(" "));
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return 0;
    }
    if !confirm("Add user to render,video groups?", opts.yes) {
        return 5;
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let (rc, out, err) = run(program, &arg_refs, RUN_TIMEOUT);
    print!("{out}");
    eprint!("{err}");
    if rc != 0 {
        println!("usermod exited {rc}; group membership NOT changed.");
        return 4;
    }
    println!("Added {user} to render,video.");
    println!(
        "IMPORTANT: log out and back in (or reboot) for the membership to take effect in new shells and services. `newgrp render` patches the current shell only."
    );
    0
}

/// fix-2: help the user clear HSA_OVERRIDE_GFX_VERSION for future shells.
fn run_unset_override(opts: &FixOptions) -> i32 {
    if runtime_is_windows() {
        run_unset_override_windows(opts)
    } else {
        run_unset_override_linux()
    }
}

fn run_unset_override_linux() -> i32 {
    let current = std::env::var("HSA_OVERRIDE_GFX_VERSION").unwrap_or_default();
    if current.is_empty() {
        println!("HSA_OVERRIDE_GFX_VERSION is already unset in this shell.");
    } else {
        println!("HSA_OVERRIDE_GFX_VERSION={current} is set in this shell.");
        println!("In your current shell, run:");
        println!("  unset HSA_OVERRIDE_GFX_VERSION");
        println!("(This command can't unset it in your parent shell; it only sees a copy.)");
    }
    let Some(home) = home_dir() else {
        return 0;
    };
    let candidates = [
        home.join(".bashrc"),
        home.join(".bash_profile"),
        home.join(".zshrc"),
        home.join(".profile"),
        home.join(".config").join("fish").join("config.fish"),
    ];
    let hits: Vec<PathBuf> = candidates
        .into_iter()
        .filter(|f| {
            std::fs::read_to_string(f).is_ok_and(|b| b.contains("HSA_OVERRIDE_GFX_VERSION"))
        })
        .collect();
    if hits.is_empty() {
        println!("\nNo persistent HSA_OVERRIDE_GFX_VERSION found in your shell rc files.");
        return 0;
    }
    println!("\nPersistent HSA_OVERRIDE_GFX_VERSION found in:");
    for f in &hits {
        println!("  - {}", f.display());
    }
    println!(
        "\nRemove or comment those lines manually. This command does NOT edit your shell rc files for you; that's your dotfiles. Suggested:"
    );
    for f in &hits {
        println!(
            "  $ $EDITOR {}   # delete or comment the HSA_OVERRIDE_GFX_VERSION line",
            f.display()
        );
    }
    0
}

fn run_unset_override_windows(opts: &FixOptions) -> i32 {
    let current = std::env::var("HSA_OVERRIDE_GFX_VERSION").unwrap_or_default();
    if current.is_empty() {
        println!("HSA_OVERRIDE_GFX_VERSION is not set in this shell.");
    } else {
        println!("HSA_OVERRIDE_GFX_VERSION={current} is set in this shell.");
        println!("Note: clearing it in your Windows env scope does NOT affect this");
        println!("already-open shell -- close and reopen your terminal afterwards.");
    }
    let user_val = ps_env_scope("HSA_OVERRIDE_GFX_VERSION", "User");
    let machine_val = ps_env_scope("HSA_OVERRIDE_GFX_VERSION", "Machine");
    if user_val.is_empty() && machine_val.is_empty() {
        println!("\nNo persistent HSA_OVERRIDE_GFX_VERSION found in either the User");
        println!("or Machine env scope. You're done after closing/reopening shells.");
        return 0;
    }
    println!("\nPersistent HSA_OVERRIDE_GFX_VERSION found in:");
    if !user_val.is_empty() {
        println!("  User scope:    {user_val}");
    }
    if !machine_val.is_empty() {
        println!("  Machine scope: {machine_val}");
    }
    if !user_val.is_empty() {
        println!("\nClear from the User scope (no admin needed):");
        println!("  Will run: setx HSA_OVERRIDE_GFX_VERSION \"\"");
        if opts.dry_run {
            println!("  (dry-run; not executed)");
        } else if confirm("Clear HSA_OVERRIDE_GFX_VERSION from User scope?", opts.yes) {
            let (rc, out, err) = run("setx", &["HSA_OVERRIDE_GFX_VERSION", ""], RUN_TIMEOUT);
            print!("{out}");
            eprint!("{err}");
            if rc != 0 {
                println!("setx exited {rc}; User scope NOT changed.");
                return 4;
            }
            println!("Cleared from User scope. Reopen your terminal for it to take effect.");
        }
    }
    if !machine_val.is_empty() {
        println!(
            "\nThe Machine scope value cannot be cleared without an Admin shell. Either run an elevated PowerShell and execute:"
        );
        println!(
            "  [Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION', $null, 'Machine')"
        );
        println!(
            "or remove it through System Properties -> Environment Variables -> System variables. This command does NOT elevate itself."
        );
    }
    0
}

/// fix-6: persist the ROCm/HIP bin directory on PATH (with consent).
fn run_path_export(opts: &FixOptions) -> i32 {
    if runtime_is_windows() {
        run_path_export_windows(opts)
    } else {
        run_path_export_linux(opts)
    }
}

fn run_path_export_linux(opts: &FixOptions) -> i32 {
    // Same resolver `examine` uses, so the line we append names the install the
    // report pointed at -- including a versioned root like /opt/rocm-6.4.1.
    let Some(install) = crate::discover_rocm_installs().into_iter().next() else {
        println!("No ROCm install found; nothing to add to PATH.");
        return 3;
    };
    let bin_path = install.path.join("bin");
    if !bin_path.is_dir() {
        println!(
            "{} does not exist; nothing to add to PATH.",
            bin_path.display()
        );
        return 3;
    }
    let bin_dir_owned = bin_path.to_string_lossy().into_owned();
    let bin_dir = bin_dir_owned.as_str();
    let Some(rc_file) = shell_rc_file() else {
        println!("Could not determine your home directory.");
        return 3;
    };
    let export_line = format!("export PATH=\"{bin_dir}:$PATH\"");
    if let Ok(existing) = std::fs::read_to_string(&rc_file)
        && existing
            .lines()
            .any(|l| l.contains("PATH=") && l.contains(bin_dir))
    {
        println!(
            "{} already adds {bin_dir} to PATH; no change.",
            rc_file.display()
        );
        return 0;
    }
    println!("Plan: append the following line to {}:", rc_file.display());
    println!("  {export_line}");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return 0;
    }
    if !confirm(&format!("Append to {}?", rc_file.display()), opts.yes) {
        return 5;
    }
    if let Err(exc) = append_line(
        &rc_file,
        "# Added by rocm examine (fix-6-path)",
        &export_line,
    ) {
        println!("Failed to write {}: {exc}", rc_file.display());
        return 4;
    }
    println!(
        "Appended to {}. Open a new shell or run `source {}` for the change to take effect.",
        rc_file.display(),
        rc_file.display()
    );
    0
}

fn run_path_export_windows(opts: &FixOptions) -> i32 {
    let mut sdk_path = std::env::var("HIP_PATH").unwrap_or_default();
    if sdk_path.is_empty() {
        sdk_path = newest_rocm_install_dir();
    }
    if sdk_path.is_empty() {
        println!("No HIP SDK install found. Run fix-13-hip-sdk-missing first.");
        return 3;
    }
    let bin_dir = Path::new(&sdk_path).join("bin");
    if !bin_dir.is_dir() {
        println!(
            "{} does not exist on disk; HIP SDK install looks incomplete.",
            bin_dir.display()
        );
        return 3;
    }
    let bin_dir = bin_dir.to_string_lossy().into_owned();
    let user_path = ps_env_scope("PATH", "User");
    if !user_path.is_empty() && user_path.to_lowercase().contains(&bin_dir.to_lowercase()) {
        println!("User PATH already contains {bin_dir}; no change.");
        return 0;
    }
    let new_path = if user_path.is_empty() {
        bin_dir.clone()
    } else {
        format!("{user_path};{bin_dir}")
    };
    println!("Plan: prepend {bin_dir} to your User PATH:");
    println!("  setx PATH \"{new_path}\"");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return 0;
    }
    if !confirm("Update User PATH?", opts.yes) {
        return 5;
    }
    let (rc, out, err) = run("setx", &["PATH", &new_path], RUN_TIMEOUT);
    print!("{out}");
    eprint!("{err}");
    if rc != 0 {
        println!("setx exited {rc}; User PATH NOT changed.");
        return 4;
    }
    println!(
        "Added {bin_dir} to your User PATH. setx only takes effect in NEW shells -- close this terminal and reopen it before re-running hipInfo."
    );
    0
}

/// fix-9: persist HIP_VISIBLE_DEVICES so the iGPU is hidden.
fn run_hip_visible_devices(opts: &FixOptions) -> i32 {
    if let Some(idx) = opts.device_index.filter(|&i| i < 0) {
        println!("--device-index must be >= 0 (got {idx}).");
        return 3;
    }
    if runtime_is_windows() {
        run_hip_visible_devices_windows(opts)
    } else {
        run_hip_visible_devices_linux(opts)
    }
}

fn run_hip_visible_devices_linux(opts: &FixOptions) -> i32 {
    let Some(idx) = opts.device_index else {
        println!(
            "Run `rocminfo | grep -E 'Agent |Marketing|gfx'` and identify the row of your DISCRETE GPU (the iGPU is typically Agent 1). Then re-run with --device-index N."
        );
        return 3;
    };
    let Some(rc_file) = shell_rc_file() else {
        println!("Could not determine your home directory.");
        return 3;
    };
    pin_device_in_rc_file(&rc_file, idx, opts, || {
        confirm(&format!("Append to {}?", rc_file.display()), opts.yes)
    })
}

/// The part of fix-9 that actually touches the user's machine: decide whether
/// the rc file already pins a device, honour `--dry-run`, ask for consent, and
/// append.
///
/// Taking the rc path and the consent decision as parameters keeps this — the
/// one auto-fix path on Linux that really mutates a file — testable end to end
/// against a temp file, with no `$HOME` mutation and no dependency on whether
/// stdin happens to be a terminal. Exit codes are the contract documented in
/// `skills/rocm-doctor/reference.md`: 0 applied/dry-run/already-set, 4 write
/// failed, 5 declined.
fn pin_device_in_rc_file(
    rc_file: &Path,
    idx: i64,
    opts: &FixOptions,
    consent: impl FnOnce() -> bool,
) -> i32 {
    let export_line = format!("export HIP_VISIBLE_DEVICES={idx}");
    if let Ok(existing) = std::fs::read_to_string(rc_file)
        && existing.contains("HIP_VISIBLE_DEVICES=")
    {
        println!(
            "{} already sets HIP_VISIBLE_DEVICES; edit by hand rather than appending a second copy.",
            rc_file.display()
        );
        return 0;
    }
    println!("Plan: append the following line to {}:", rc_file.display());
    println!("  {export_line}");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return 0;
    }
    if !consent() {
        return 5;
    }
    if let Err(exc) = append_line(
        rc_file,
        "# Added by rocm examine (fix-9-igpu-dgpu)",
        &export_line,
    ) {
        println!("Failed to write {}: {exc}", rc_file.display());
        return 4;
    }
    println!(
        "Appended to {}. Open a new shell for the change to take effect, then re-run your workload.",
        rc_file.display()
    );
    0
}

fn run_hip_visible_devices_windows(opts: &FixOptions) -> i32 {
    let Some(idx) = opts.device_index else {
        println!("Run the following to identify the discrete GPU's index:");
        println!(
            "  & \"$env:HIP_PATH\\bin\\hipInfo.exe\" | Select-String \"device#|Name|gcnArchName\""
        );
        println!(
            "Then re-run with --device-index N (the iGPU is typically device# 0; the dGPU is usually device# 1)."
        );
        return 3;
    };
    let existing = ps_env_scope("HIP_VISIBLE_DEVICES", "User");
    if !existing.is_empty() {
        println!(
            "User scope already sets HIP_VISIBLE_DEVICES={existing:?}; remove or update it manually rather than overwriting from this command."
        );
        return 0;
    }
    println!("Plan: persist HIP_VISIBLE_DEVICES in the User env scope:");
    println!("  setx HIP_VISIBLE_DEVICES {idx}");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return 0;
    }
    if !confirm("Set HIP_VISIBLE_DEVICES in the User scope?", opts.yes) {
        return 5;
    }
    let (rc, out, err) = run(
        "setx",
        &["HIP_VISIBLE_DEVICES", &idx.to_string()],
        RUN_TIMEOUT,
    );
    print!("{out}");
    eprint!("{err}");
    if rc != 0 {
        println!("setx exited {rc}; HIP_VISIBLE_DEVICES NOT changed.");
        return 4;
    }
    println!(
        "setx only takes effect in NEW shells -- close this terminal and reopen it before re-running your workload."
    );
    0
}

/// Read a Windows environment variable from a given scope via PowerShell.
fn ps_env_scope(var: &str, scope: &str) -> String {
    let script = format!("[Environment]::GetEnvironmentVariable('{var}','{scope}')");
    let (rc, out, _) = run(
        "powershell",
        &["-NoProfile", "-Command", &script],
        QUERY_TIMEOUT,
    );
    if rc == 0 {
        out.trim().to_owned()
    } else {
        String::new()
    }
}

/// Newest ROCm/HIP SDK install root on this host, or empty if there is none.
///
/// This was the third independent Windows install scan in the codebase, and it
/// disagreed with the other two on every axis that matters: it searched
/// `C:\Program Files (x86)\AMD\ROCm` where the resolver searches
/// `C:\Program Files\ROCm`, it sorted with a plain `versions.sort()` so `6.2`
/// outranked `6.10`, and it accepted any directory whose name began with a
/// digit without checking for an install marker. So `fix-6-path` could put an
/// older SDK — or an empty directory — on the user's PATH.
///
/// It now asks the same resolver as `examine`, which also means `$ROCM_PATH` is
/// honoured here for the first time.
fn newest_rocm_install_dir() -> String {
    crate::discover_rocm_installs()
        .into_iter()
        .next()
        .map(|install| install.path.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plant a directory the shared resolver will accept as a ROCm install.
    /// `bin/rocminfo` is one of the markers it gates on; a bare directory is
    /// deliberately not enough.
    fn plant_install(root: &std::path::Path) {
        std::fs::create_dir_all(root.join("bin")).expect("create planted install");
        std::fs::write(root.join("bin").join("rocminfo"), "").expect("write marker");
    }

    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn the_path_fix_finds_the_install_the_rest_of_the_cli_found() {
        // Discriminating on purpose: the scanner this replaces looked only in
        // two hardcoded `C:\Program Files` directories and ignored $ROCM_PATH
        // outright, so it returned nothing here no matter what was planted.
        // Going through the shared resolver is what makes this pass -- and is
        // what stops fix-6-path putting 6.2 on PATH when 6.10 is installed.
        let root = std::env::temp_dir().join(format!(
            "rocm-fix-path-resolver-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let install = root.join("rocm-6.10.0");
        plant_install(&install);

        let previous = std::env::var_os("ROCM_PATH");
        unsafe {
            std::env::set_var("ROCM_PATH", &install);
        }
        let found = newest_rocm_install_dir();
        unsafe {
            match previous {
                Some(value) => std::env::set_var("ROCM_PATH", value),
                None => std::env::remove_var("ROCM_PATH"),
            }
        }
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            found,
            install.to_string_lossy(),
            "fix-6-path must resolve installs the same way examine does"
        );
    }

    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn the_path_fix_reports_nothing_rather_than_a_directory_with_no_install_in_it() {
        // The old scan accepted any directory whose name started with a digit,
        // so an empty leftover could be put on PATH. The resolver requires a
        // marker.
        let root = std::env::temp_dir().join(format!(
            "rocm-fix-path-empty-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let empty = root.join("6.10");
        std::fs::create_dir_all(&empty).expect("create empty dir");

        let previous = std::env::var_os("ROCM_PATH");
        unsafe {
            std::env::set_var("ROCM_PATH", &empty);
        }
        let found = newest_rocm_install_dir();
        unsafe {
            match previous {
                Some(value) => std::env::set_var("ROCM_PATH", value),
                None => std::env::remove_var("ROCM_PATH"),
            }
        }
        std::fs::remove_dir_all(&root).ok();

        assert!(
            found.is_empty(),
            "an empty directory is not an install, but {found:?} was returned"
        );
    }

    #[test]
    fn every_recipe_id_is_unique_and_covers_the_catalog() {
        let mut ids: Vec<&str> = RECIPES.iter().map(|r| r.fix_id).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate fix-id in RECIPES");
        assert_eq!(count, 15, "expected 15 catalog entries");
    }

    #[test]
    fn auto_applicable_recipes_have_a_runner() {
        for r in RECIPES {
            assert_eq!(
                r.auto_applicable,
                r.runner.is_some(),
                "{}: auto_applicable must match presence of a runner",
                r.fix_id
            );
        }
    }

    #[test]
    fn exactly_the_four_known_fixes_are_auto() {
        let auto: Vec<&str> = RECIPES
            .iter()
            .filter(|r| r.auto_applicable)
            .map(|r| r.fix_id)
            .collect();
        assert_eq!(
            auto,
            vec![
                "fix-2-unset-override",
                "fix-4-render-group",
                "fix-6-path",
                "fix-9-igpu-dgpu"
            ]
        );
    }

    #[test]
    fn unknown_fix_id_returns_2() {
        let code = apply("fix-does-not-exist", &FixOptions::default());
        assert_eq!(code, 2);
    }

    #[test]
    fn dry_run_never_mutates_and_returns_zero_for_auto_linux_fix() {
        if !runtime_is_linux() {
            return;
        }
        // fix-2 unset-override is print-only on linux (no mutation regardless);
        // a dry-run must report success without changing anything.
        let opts = FixOptions {
            dry_run: true,
            ..FixOptions::default()
        };
        let code = apply("fix-2-unset-override", &opts);
        assert_eq!(code, 0);
    }

    #[test]
    fn print_only_fix_returns_zero() {
        if !runtime_is_linux() {
            return;
        }
        let code = apply("fix-5-amdgpu-load", &FixOptions::default());
        assert_eq!(code, 0);
    }

    #[test]
    fn windows_only_fix_refused_on_linux() {
        if !runtime_is_linux() {
            return;
        }
        let code = apply("fix-13-hip-sdk-missing", &FixOptions::default());
        assert_eq!(code, 3);
    }

    #[test]
    fn list_includes_all_ids() {
        let listing = list_recipes();
        for r in RECIPES {
            assert!(listing.contains(r.fix_id), "listing missing {}", r.fix_id);
        }
    }

    #[test]
    fn listing_explains_its_applicability_markers() {
        // The markers were emitted with nothing saying what they mean, leaving
        // the reader to guess whether PRINT-ONLY was a debug flag.
        let listing = list_recipes();
        assert!(
            listing.contains("AUTO =") && listing.contains("PRINT-ONLY ="),
            "the listing must explain its markers:\n{listing}"
        );
    }

    #[test]
    fn diagnosis_positions_are_recognised_as_positions() {
        // What `rocm diagnose` shows as `#1`/`#2`, plus the bare number a user
        // might type instead.
        for value in ["#1", "#2", "1", "12", " #3 "] {
            assert!(
                looks_like_a_diagnosis_position(value),
                "{value} should read as a ranking position"
            );
        }
        // Real ids and other typos must keep the generic refusal — claiming a
        // fix-id is a "position" would be worse than saying nothing.
        for value in [
            "fix-4-render-group",
            "fix-1-arch",
            "bogus",
            "#",
            "",
            "fix-#1",
        ] {
            assert!(
                !looks_like_a_diagnosis_position(value),
                "{value} should NOT read as a ranking position"
            );
        }
    }

    #[test]
    fn a_position_argument_is_refused_with_the_same_exit_code() {
        // Still 2: this is clearer wording on an existing refusal, not a new
        // behaviour that a script could start depending on.
        assert_eq!(apply("#1", &FixOptions::default()), 2);
        assert_eq!(apply("bogus", &FixOptions::default()), 2);
    }

    // ── Consent and mutation contract ──────────────────────────────
    //
    // `skills/rocm-doctor/reference.md` documents six exit codes, and the skill
    // tells an agent it may run an auto-fix because the CLI "refuses on a
    // non-interactive shell without --yes" and "confirms before mutating".
    // Before these tests, codes 1, 4 and 5 were asserted nowhere in the repo and
    // no test ever reached a runner that writes to disk: the two that look like
    // they cover it don't — `dry_run_never_mutates_...` picks fix-2, whose Linux
    // runner takes no FixOptions and never prompts, and diagnose.feature's
    // preview scenario picks print-only fix-1, which returns before any runner.
    //
    // fix-9 is the auto-fix used here because its Linux runner is the one that
    // genuinely appends to a file, and `pin_device_in_rc_file` lets that run
    // against a temp path with an injected consent verdict.

    /// A unique scratch path under the workspace's test-artifact dir, following
    /// the convention in `lib.rs`'s tests (no `tempfile` dev-dependency here).
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("core")
            .join(format!(
                "fix-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
        std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
        dir
    }

    fn pinning_opts() -> FixOptions {
        FixOptions {
            device_index: Some(1),
            ..FixOptions::default()
        }
    }

    #[test]
    fn yes_flag_approves_without_prompting() {
        assert_eq!(consent_without_prompt(true, false), Some(true));
        assert_eq!(consent_without_prompt(true, true), Some(true));
    }

    #[test]
    fn non_interactive_shell_refuses_instead_of_proceeding() {
        // The rule the skill relies on: without --yes and with nothing to prompt,
        // the answer is a definite NO, never an implicit yes.
        assert_eq!(consent_without_prompt(false, false), Some(false));
    }

    #[test]
    fn interactive_shell_without_yes_must_actually_ask() {
        assert_eq!(consent_without_prompt(false, true), None);
    }

    #[test]
    fn declining_consent_returns_5_and_writes_nothing() {
        let rc = scratch_dir("declined").join(".bashrc");
        std::fs::write(&rc, "# existing\n").expect("seed rc file");

        let code = pin_device_in_rc_file(&rc, 1, &pinning_opts(), || false);

        assert_eq!(code, 5, "a declined fix must exit 5");
        assert_eq!(
            std::fs::read_to_string(&rc).expect("read rc"),
            "# existing\n",
            "a declined fix must not touch the file"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn dry_run_returns_0_and_writes_nothing_even_with_consent() {
        let rc = scratch_dir("dryrun").join(".bashrc");
        std::fs::write(&rc, "# existing\n").expect("seed rc file");
        let opts = FixOptions {
            dry_run: true,
            ..pinning_opts()
        };

        // Consent would be granted; --dry-run must short-circuit before it.
        let code = pin_device_in_rc_file(&rc, 1, &opts, || {
            panic!("dry-run must not reach the consent gate")
        });

        assert_eq!(code, 0);
        assert_eq!(
            std::fs::read_to_string(&rc).expect("read rc"),
            "# existing\n",
            "a dry-run must not touch the file"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn granting_consent_appends_the_export_and_returns_0() {
        let rc = scratch_dir("granted").join(".bashrc");
        std::fs::write(&rc, "# existing\n").expect("seed rc file");

        let code = pin_device_in_rc_file(&rc, 3, &pinning_opts(), || true);

        assert_eq!(code, 0);
        let body = std::fs::read_to_string(&rc).expect("read rc");
        assert!(
            body.starts_with("# existing\n"),
            "existing rc content must be preserved:\n{body}"
        );
        assert!(
            body.contains("export HIP_VISIBLE_DEVICES=3"),
            "expected the export line to be appended:\n{body}"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn an_rc_file_that_already_pins_a_device_is_left_alone() {
        let rc = scratch_dir("already-pinned").join(".bashrc");
        let original = "export HIP_VISIBLE_DEVICES=0\n";
        std::fs::write(&rc, original).expect("seed rc file");

        let code = pin_device_in_rc_file(&rc, 1, &pinning_opts(), || {
            panic!("an already-pinned rc file must not reach the consent gate")
        });

        assert_eq!(code, 0, "already-pinned is success, not failure");
        assert_eq!(
            std::fs::read_to_string(&rc).expect("read rc"),
            original,
            "must not append a second, conflicting pin"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn a_failed_write_returns_4_rather_than_reporting_success() {
        // A directory where the rc file should be: the append fails at open().
        // This is the only way the documented "attempted but the command
        // failed" code is reachable for this fix.
        let dir = scratch_dir("write-fails");
        let rc = dir.join(".bashrc");
        std::fs::create_dir_all(&rc).expect("create dir in place of rc file");

        let code = pin_device_in_rc_file(&rc, 1, &pinning_opts(), || true);

        assert_eq!(code, 4, "a failed write must exit 4, not 0");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pinning_without_a_device_index_is_not_applicable() {
        if !runtime_is_linux() {
            return;
        }
        // Reached through the real runner: the missing --device-index guard sits
        // ahead of any home-directory lookup, so this needs no scratch state.
        let code = run_hip_visible_devices_linux(&FixOptions::default());
        assert_eq!(code, 3, "a missing --device-index is 'not applicable' (3)");
    }

    #[test]
    fn exit_code_1_is_unreachable_by_construction() {
        // apply() returns 1 only for an auto_applicable recipe with no runner.
        // `auto_applicable_recipes_have_a_runner` keeps that impossible; assert
        // the reachability argument here so reference.md's row 1 is accounted
        // for rather than silently untested.
        assert!(
            RECIPES
                .iter()
                .all(|r| !r.auto_applicable || r.runner.is_some()),
            "an auto-applicable recipe without a runner would make exit 1 reachable"
        );
    }
}
