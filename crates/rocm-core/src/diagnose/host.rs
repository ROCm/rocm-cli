// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The checks on the host's configuration that run on more than one platform:
//! wheel architecture, the HSA override, `PATH`, the wheel's ROCm against the
//! system's, an integrated GPU beside a discrete one, and GPUs ROCm does not
//! support.

use super::scoring::{
    KEYWORDS_INVALID_ISA, KEYWORDS_LIB_MISMATCH, KEYWORDS_PAGE_FAULT, KEYWORDS_PATH_MISSING,
    amd_gfx_targets, amd_gpu_count, finalize, keyword_score, symptom_matches, zero,
};
use super::{Diagnosis, Fix};
use crate::examine::Examination;
use regex::Regex;

pub(super) fn check_1_arch_not_in_wheel(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_INVALID_ISA);
    score += kw_score;
    evidence.extend(kw_ev);

    let framework_arch = &e.framework_arch_list;
    let gfx_targets = amd_gfx_targets(e);
    if !framework_arch.is_empty() && !gfx_targets.is_empty() {
        let missing: Vec<String> = gfx_targets
            .iter()
            .filter(|t| !framework_arch.contains(t))
            .cloned()
            .collect();
        if missing.is_empty() {
            score -= 30;
            evidence.push(format!(
                "framework arch list {framework_arch:?} already includes GPU target(s) {gfx_targets:?}"
            ));
        } else {
            score += 55;
            evidence.push(format!(
                "GPU gfx target(s) {missing:?} not in framework arch list {framework_arch:?}"
            ));
        }
    }

    if matches!(e.framework.as_str(), "pytorch" | "llama-cpp")
        && framework_arch.is_empty()
        && !gfx_targets.is_empty()
    {
        evidence.push(
            "Framework arch list unknown -- cannot confirm without `python -c 'import torch; print(torch.cuda.get_arch_list())'`."
                .to_owned(),
        );
    }

    if score <= 0 {
        return zero("fix-1-arch", "GPU gfx not in framework arch list");
    }
    let fix = Fix {
        summary: "Reinstall the framework from a wheel index that includes this GPU's gfx target. Use HSA_OVERRIDE_GFX_VERSION ONLY as a temporary workaround when no native wheel exists.".to_owned(),
        commands: vec![
            "# Recommended: a PyTorch ROCm nightly, which often carries kernels a".to_owned(),
            "# release has not shipped yet. Pick the nightly for the ROCm major you".to_owned(),
            "# have, not an older one.".to_owned(),
            "pip uninstall -y torch torchvision torchaudio".to_owned(),
            "pip install --pre torch torchvision torchaudio \\\n  --index-url https://download.pytorch.org/whl/nightly/rocm7.14".to_owned(),
            "# llama.cpp: rebuild with AMDGPU_TARGETS set to this GPU's gfx.".to_owned(),
            "# cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx_target>".to_owned(),
        ],
        fix_id: "fix-1-arch".to_owned(),
        verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())\"".to_owned(),
        notes: notes_1_arch(e),
        ..Fix::default()
    };
    finalize(
        "fix-1-arch",
        "GPU gfx target not in framework's build arch list",
        score,
        evidence,
        fix,
    )
}

/// The arch-list evidence this checker reads can now come from a managed
/// runtime's torch, which the bare `pip` commands above would not touch — they
/// resolve against whatever interpreter is on `PATH`, a different environment.
/// Say which one the evidence describes rather than letting the commands imply
/// it.
fn notes_1_arch(e: &Examination) -> Vec<String> {
    let mut notes = vec![
        "TheRock (rocm/TheRock) ships nightly per-gfx wheels and is the preferred fallback when the official pytorch wheel index does not yet cover your gfx target.".to_owned(),
    ];
    if e.framework_source == "managed-runtime" {
        notes.push(
            "This host's torch was read from the active managed runtime, not from `PATH`. Run the commands above against that runtime's own interpreter -- `rocm examine --json` names it under framework_notes -- or a bare `pip` will change a different environment and leave this unfixed."
                .to_owned(),
        );
    }
    notes
}

pub(super) fn check_2_hsa_override_unneeded(e: &Examination, symptom: &str) -> Diagnosis {
    let override_val = e
        .env
        .get("HSA_OVERRIDE_GFX_VERSION")
        .cloned()
        .unwrap_or_default();
    if override_val.is_empty() {
        return zero(
            "fix-2-unset-override",
            "HSA_OVERRIDE_GFX_VERSION set unnecessarily",
        );
    }
    let mut score = 30;
    let mut evidence = vec![format!(
        "HSA_OVERRIDE_GFX_VERSION={override_val} is set in the current shell"
    )];

    let (pf_score, pf_ev) = keyword_score(symptom, KEYWORDS_PAGE_FAULT);
    score += pf_score;
    evidence.extend(pf_ev);
    if e.dmesg_amdgpu_tail
        .iter()
        .any(|l| l.to_lowercase().contains("page fault"))
    {
        score += 20;
        evidence.push("kernel ring shows amdgpu page faults".to_owned());
    }

    let framework_arch = &e.framework_arch_list;
    let gfx_targets = amd_gfx_targets(e);
    if !framework_arch.is_empty()
        && !gfx_targets.is_empty()
        && gfx_targets.iter().all(|t| framework_arch.contains(t))
    {
        score += 25;
        evidence.push(format!(
            "every detected GPU target ({gfx_targets:?}) is in the framework arch list ({framework_arch:?}); the override is hiding the native gfx."
        ));
    }

    let fix = if e.os_family == "windows" {
        Fix {
            summary: "Clear HSA_OVERRIDE_GFX_VERSION (Windows) and use the native HIP SDK / wheel.".to_owned(),
            commands: vec![
                "# Inspect the User and Machine env scopes:".to_owned(),
                "[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','User')".to_owned(),
                "[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','Machine')".to_owned(),
                "# Clear from the User scope (does NOT affect already-open shells):".to_owned(),
                "[Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION',$null,'User')".to_owned(),
                "# Or remove via System Properties -> Environment Variables.".to_owned(),
            ],
            fix_id: "fix-2-unset-override".to_owned(),
            verify: "powershell -NoProfile -Command \"[Environment]::GetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION','User')\"".to_owned(),
            ..Fix::default()
        }
    } else {
        Fix {
            summary: "Unset HSA_OVERRIDE_GFX_VERSION and use the native wheel.".to_owned(),
            commands: vec![
                "unset HSA_OVERRIDE_GFX_VERSION".to_owned(),
                "# Also remove it from ~/.bashrc / ~/.zshrc / ~/.profile if persisted.".to_owned(),
            ],
            fix_id: "fix-2-unset-override".to_owned(),
            // The `rocm fix` catalog's Linux line (needs no Python, which a
            // stock Ubuntu lacks), then the check that clearing the override
            // did not break the GPU. The catalog lines are shared with Windows
            // and cannot assume torch; this arm only builds for a bash host,
            // so it can run a second command after the first. `python3`, not
            // `python`: whatever interpreter has torch, a stock `python3`, a
            // virtual environment and a conda environment all provide it.
            verify: "echo \"${HSA_OVERRIDE_GFX_VERSION:-OK_UNSET}\"; python3 -c \"import torch; print(torch.cuda.is_available())\"".to_owned(),
            ..Fix::default()
        }
    };
    finalize(
        "fix-2-unset-override",
        "HSA_OVERRIDE_GFX_VERSION set on a GPU that has a native wheel",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_6_path_missing(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let env_path = e.env.get("PATH").cloned().unwrap_or_default();
    let windows = e.os_family == "windows";
    let bin_dir;

    if windows {
        let sdk_path = &e.hip_sdk_path;
        bin_dir = if sdk_path.is_empty() {
            r"C:\Program Files\AMD\ROCm\<version>\bin".to_owned()
        } else {
            format!("{sdk_path}\\bin")
        };
        if !sdk_path.is_empty() && !e.hipinfo_present {
            score += 50;
            evidence.push(format!(
                "{sdk_path} exists but hipInfo.exe wasn't found in its bin directory"
            ));
        }
        if !sdk_path.is_empty()
            && !env_path.is_empty()
            && !env_path.to_lowercase().contains(&bin_dir.to_lowercase())
        {
            score += 20;
            evidence.push(format!("{bin_dir} is not in PATH"));
        }
    } else {
        let rocm_path = &e.rocm_path;
        bin_dir = if rocm_path.is_empty() {
            "/opt/rocm/bin".to_owned()
        } else {
            format!("{rocm_path}/bin")
        };
        if !rocm_path.is_empty() && !e.rocminfo_present {
            score += 50;
            evidence.push(format!("{rocm_path} exists but `rocminfo` is not on PATH"));
        }
        if !rocm_path.is_empty() && !env_path.is_empty() && !env_path.contains(&bin_dir) {
            score += 20;
            evidence.push(format!("{bin_dir} is not in $PATH"));
        }
    }

    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_PATH_MISSING);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-6-path", "ROCm not on PATH");
    }
    let fix = if windows {
        // The command printed here is the manual-paste twin of the write
        // `crate::fix::path_export_windows` performs, and is built the same
        // way: read the *User* scope, put `bin_dir` first, write the User
        // scope back. `setx PATH "%PATH%;{bin_dir}"` -- what this used to
        // advertise -- is not an equivalent: setx truncates the value at 1024
        // characters while still exiting 0, and `%PATH%` expands to the
        // combined user+system PATH, so pasting it would fold every system
        // entry into the user's own PATH. The escape is the same one the real
        // write uses, so an apostrophe in `bin_dir` cannot end the PowerShell
        // string literal early in the printed copy either.
        let escaped_bin_dir = crate::quoting::ps_single_quote_escape(&bin_dir);
        Fix {
            summary: format!("Add {bin_dir} to your User PATH and reopen the shell."),
            commands: vec![
                format!(
                    "[Environment]::SetEnvironmentVariable('PATH', '{escaped_bin_dir};' + [Environment]::GetEnvironmentVariable('PATH', 'User'), 'User')"
                ),
                "# Or: System Properties -> Environment Variables -> Path -> Edit -> New."
                    .to_owned(),
                "# A PATH change only affects NEW shells; close and reopen this terminal afterwards."
                    .to_owned(),
            ],
            fix_id: "fix-6-path".to_owned(),
            verify: format!(
                "powershell -NoProfile -Command \"& '{escaped_bin_dir}\\hipInfo.exe' | Select-Object -First 5\""
            ),
            ..Fix::default()
        }
    } else {
        Fix {
            summary: format!("Add {bin_dir} to PATH for this shell and persist in your shell rc."),
            commands: vec![
                format!("export PATH={bin_dir}:$PATH"),
                format!("echo 'export PATH={bin_dir}:$PATH' >> ~/.bashrc   # or ~/.zshrc"),
            ],
            fix_id: "fix-6-path".to_owned(),
            // The `rocm fix` catalog's Linux line. `rocminfo` lives in the bin
            // directory this fix puts on PATH, so it printing the GPU shows
            // both that PATH is fixed and that the GPU is visible. It does
            // not chain `hipcc --version`: a runtime-only install has no
            // `hipcc`, and the chain would fail there with PATH fixed.
            verify: "rocminfo | head -n 5".to_owned(),
            ..Fix::default()
        }
    };
    finalize(
        "fix-6-path",
        "ROCm/HIP binaries not on PATH after install",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_8_wheel_rocm_mismatch(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let windows = e.os_family == "windows";
    let fw_rocm = &e.framework_rocm_version;
    let sys_rocm = if windows {
        &e.hip_sdk_version
    } else {
        &e.rocm_version
    };

    let fw_major = major_version(fw_rocm);
    let sys_major = major_version(sys_rocm);
    // Only meaningful when the framework resolves its HIP from the system. A
    // managed runtime's torch loads it from a sibling `_rocm_sdk_core` package
    // inside the runtime, so its HIP major is free to differ from the system's
    // on a completely healthy host — and "reinstall torch" would be wrong there.
    let framework_uses_system_rocm = e.framework_source != "managed-runtime";
    if let (Some(fw), Some(sys)) = (&fw_major, &sys_major)
        && fw != sys
        && framework_uses_system_rocm
    {
        score += 50;
        let runtime = if windows { "HIP SDK" } else { "ROCm" };
        evidence.push(format!(
            "Framework links HIP {fw} but system {runtime} is {sys}"
        ));
    }

    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_LIB_MISMATCH);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-8-wheel-rocm", "Wheel/ROCm mismatch");
    }
    let fix = if windows {
        Fix {
            summary: "Reinstall the framework against the HIP SDK major you have installed (or install the HIP SDK major the wheel needs).".to_owned(),
            commands: vec![
                "pip uninstall -y torch torchvision torchaudio".to_owned(),
                "# TheRock publishes Windows ROCm wheels per HIP SDK release:".to_owned(),
                "#   https://github.com/ROCm/TheRock".to_owned(),
                "# Match the wheel index to the HIP SDK major you have on disk.".to_owned(),
                "python -c \"import torch; print(torch.__version__, torch.version.hip)\"".to_owned(),
            ],
            fix_id: "fix-8-wheel-rocm".to_owned(),
            verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\"".to_owned(),
            ..Fix::default()
        }
    } else {
        Fix {
            summary: "Reinstall the framework from the wheel index that matches the system ROCm major (or upgrade the system ROCm to match the wheel).".to_owned(),
            commands: vec![
                "pip uninstall -y torch torchvision torchaudio".to_owned(),
                "# Install the index for the ROCm major `rocm examine` reports:".to_owned(),
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.14".to_owned(),
                "# ROCm 10 has no released PyTorch index yet; it is on the nightly channel:".to_owned(),
                "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm10.0".to_owned(),
                "# Then re-check:".to_owned(),
                "python -c \"import torch; print(torch.__version__, torch.version.hip)\"".to_owned(),
            ],
            fix_id: "fix-8-wheel-rocm".to_owned(),
            verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\"".to_owned(),
            ..Fix::default()
        }
    };
    finalize(
        "fix-8-wheel-rocm",
        "Framework wheel built for a different ROCm major than the system",
        score,
        evidence,
        fix,
    )
}

/// Extract `X.Y` from a version-ish string.
fn major_version(s: &str) -> Option<String> {
    let re = Regex::new(r"(\d+)\.(\d+)").ok()?;
    let caps = re.captures(s)?;
    Some(format!("{}.{}", &caps[1], &caps[2]))
}

pub(super) fn check_9_igpu_dgpu_collision(e: &Examination, symptom: &str) -> Diagnosis {
    if !(e.has_apu && e.has_discrete_amd) {
        return zero("fix-9-igpu-dgpu", "iGPU+dGPU collision");
    }
    let visible = e
        .env
        .get("HIP_VISIBLE_DEVICES")
        .or_else(|| e.env.get("ROCR_VISIBLE_DEVICES"))
        .filter(|v| !v.is_empty());
    let mut score = 40;
    let mut evidence = vec!["machine has both an AMD APU and an AMD discrete GPU".to_owned()];
    if visible.is_none() {
        score += 25;
        evidence.push("HIP_VISIBLE_DEVICES is unset; runtime sees BOTH GPUs".to_owned());
    }
    if symptom_matches(symptom, r"(crash|segfault|signal 11)") {
        score += 15;
        evidence.push("user mentions a crash / segfault".to_owned());
    }

    let gfx_targets = amd_gfx_targets(e);
    // Name the discrete GPU explicitly instead of guessing by gfx number: on
    // RDNA3 the integrated APU (gfx1103 / gfx115x) can outrank its discrete
    // neighbor (gfx1100/1101/1102), so "the higher-numbered target is the dGPU"
    // is actively wrong for exactly the APU+dGPU pairing this check fires on.
    let discrete_targets: Vec<String> = e
        .gpus
        .iter()
        .filter(|g| g.is_amd && g.is_apu == Some(false) && !g.gfx_target.is_empty())
        .map(|g| g.gfx_target.clone())
        .collect();
    let apu_targets: Vec<String> = e
        .gpus
        .iter()
        .filter(|g| g.is_amd && g.is_apu == Some(true) && !g.gfx_target.is_empty())
        .map(|g| g.gfx_target.clone())
        .collect();
    let note = if discrete_targets.is_empty() {
        format!(
            "Detected gfx targets: {gfx_targets:?}. Pin HIP_VISIBLE_DEVICES to the discrete GPU so the integrated APU is hidden."
        )
    } else {
        format!(
            "Detected gfx targets: {gfx_targets:?}. Discrete GPU(s): {discrete_targets:?}; integrated APU(s): {apu_targets:?}. Pin HIP_VISIBLE_DEVICES to the discrete GPU - do not assume the higher-numbered gfx target is the dGPU (on RDNA3 the APU can be higher)."
        )
    };
    // The catalog classes this NEEDS-ARG, which is what the marker now says:
    // without --device-index both the Linux and Windows runners only print the
    // query that finds the index and change nothing. The note used to end
    // "despite being marked AUTO", which was true of the old flat flag and is
    // exactly the overstatement this change removes.
    let note = format!(
        "{note} Without --device-index, `rocm fix` only prints this query and makes no change."
    );
    let fix = if e.os_family == "windows" {
        Fix {
            summary: "Pin the HIP runtime to the discrete GPU with HIP_VISIBLE_DEVICES so the iGPU is hidden.".to_owned(),
            commands: vec![
                "# Confirm which index is the dGPU (hipInfo.exe output order):".to_owned(),
                "& \"$env:HIP_PATH\\bin\\hipInfo.exe\" | Select-String \"device#|Name|gcnArchName\"".to_owned(),
                "# Then persist HIP_VISIBLE_DEVICES in the User environment:".to_owned(),
                "setx HIP_VISIBLE_DEVICES 1".to_owned(),
                "# `setx` only takes effect in NEW shells; reopen the terminal.".to_owned(),
            ],
            fix_id: "fix-9-igpu-dgpu".to_owned(),
            verify: "powershell -NoProfile -Command \"[Environment]::SetEnvironmentVariable('HIP_VISIBLE_DEVICES','1','Process'); python -c 'import torch; print(torch.cuda.device_count())'\"".to_owned(),
            notes: vec![note],
            ..Fix::default()
        }
    } else {
        Fix {
            summary: "Pin the runtime to the discrete GPU with HIP_VISIBLE_DEVICES so the iGPU is hidden.".to_owned(),
            commands: vec![
                "# Confirm which index is the dGPU (`rocminfo` output order):".to_owned(),
                "rocminfo | grep -E 'Agent |gfx|Marketing'".to_owned(),
                "# Then pin HIP to the dGPU (typically index 1 when an APU is index 0):".to_owned(),
                "export HIP_VISIBLE_DEVICES=1".to_owned(),
                "# Persist in your shell rc or your launch script.".to_owned(),
            ],
            fix_id: "fix-9-igpu-dgpu".to_owned(),
            verify: "HIP_VISIBLE_DEVICES=1 python -c \"import torch; print(torch.cuda.device_count())\"".to_owned(),
            notes: vec![note],
            ..Fix::default()
        }
    };
    finalize(
        "fix-9-igpu-dgpu",
        "iGPU enumerated alongside dGPU and destabilising the runtime",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_20_gpu_unsupported_by_rocm(e: &Examination, _symptom: &str) -> Diagnosis {
    const ID: &str = "info-20-gpu-no-rocm-support";
    const TITLE: &str = "GPU has no ROCm/HIP support (Vulkan backend still works)";

    // "This host has no ROCm/HIP support" is a claim about the whole machine,
    // so one unsupported GPU is not enough to make it: an APU+dGPU laptop with
    // a gfx90c iGPU next to a supported discrete card does have ROCm support,
    // via the dGPU. Firing there would top-rank "GPU has no ROCm/HIP support"
    // ahead of `fix-9-igpu-dgpu` (40, below MIN_SCORE_FOR_MATCH) -- which is
    // the entry that actually addresses that pairing -- and tell a user with
    // working hardware the opposite of the truth. That is the same misleading
    // top-ranked diagnosis this ticket exists to remove, on another host shape.
    //
    // An AMD GPU whose gfx target is empty is *undetected*, not supported, and
    // it blocks the claim too: we cannot say the host has no ROCm support while
    // a card we failed to identify is sitting in it.
    //
    // "Unsupported" is `host_gpu::gfx_target_has_no_rocm_build`, not a second
    // list of target names here. That predicate is the one the install path
    // already refuses on, it is token-normalised (so `gfx90c:xnack-` and
    // uppercase answer the same), and it is deliberately narrower than
    // `normalize_therock_family` returning `None` -- which also answers `None`
    // for a target this catalog simply does not recognise yet. Conflating
    // "confirmed unsupported" with "unrecognised" would turn a detection gap
    // into a false claim about the hardware; keeping a bare copy of the target
    // names here would let the two drift into disagreeing about the same card.
    let targets = amd_gfx_targets(e);
    let all_unsupported = !targets.is_empty()
        && targets.len() == amd_gpu_count(e)
        && targets
            .iter()
            .all(|gfx| crate::host_gpu::gfx_target_has_no_rocm_build(gfx));
    if !all_unsupported {
        return zero(ID, TITLE);
    }
    let gfx = targets.join(", ");

    // A real `Fix`, classed `DIAGNOSE-ONLY` in the catalog, rather than
    // `fix: None`. The class exists to say "detected, explained, no remedy",
    // and routing through it keeps one promise true across every surface:
    // `rocm diagnose` prints `apply with: rocm fix <id>` for every cause,
    // README and `--help` say `rocm fix` takes that id, and `rocm fix
    // info-20-gpu-no-rocm-support` answers -- by naming the problem and exiting
    // 0 without touching the machine. A `None` here would have made the printed
    // command a lie for this one id and left `--json` consumers a null to learn
    // about.
    let fix = Fix {
        // Deliberately a claim about the GPU and not about what `rocm serve`
        // will do. "`rocm serve` already picks a Vulkan engine on this host" was
        // neither: `default_engine_for_host` answers lemonade here, but a
        // configured default outranks it, and whether Lemonade then lands on
        // Vulkan is its own runtime choice over
        // `LLAMACPP_BACKEND_PRIORITY ["rocm","vulkan"]`. A note promising an
        // outcome this catalog does not decide would be false for anyone who
        // had set an engine. The wording below matches `therock`'s refusal for
        // the same hardware, which is the other place a user meets this fact.
        summary: "No ROCm/HIP runtime build exists for this architecture, so there is nothing to install or configure. Vulkan-based engines do not use ROCm and work on it; Lemonade, the default engine here, is one.".to_owned(),
        fix_id: ID.to_owned(),
        notes: vec![
            "HSA_OVERRIDE_GFX_VERSION is NOT a workaround here: it makes the runtime load kernels built for another architecture, which faults at runtime rather than adding support.".to_owned(),
            "To use this GPU for inference, serve with a Vulkan-capable engine. If you have set a default engine that needs ROCm, that one will not work here; `rocm serve --engine lemonade` does not need it.".to_owned(),
        ],
        ..Fix::default()
    };

    finalize(
        ID,
        TITLE,
        60,
        vec![format!(
            "Detected GPU architecture {gfx} has no ROCm/HIP runtime build. \
             Vulkan-based engines (the default Lemonade backend) still work on it."
        )],
        fix,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::test_support::{igpu_dgpu_host, linux_base, wsl_base};
    use crate::diagnose::{
        DiagnoseReport, HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH, Route, diagnose, render_report_text,
    };
    use crate::examine::Gpu;
    use crate::fix::{OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS, torch_rocm_indexes_named_in};
    use crate::rocm_install::discover_rocm_installs_in;

    /// A host whose framework HIP major differs from its system ROCm: torch on
    /// HIP 7, a system ROCm 6 beside it.
    fn hip_major_differs_from_system_rocm(framework_source: &str) -> Examination {
        Examination {
            framework: "pytorch".to_owned(),
            framework_rocm_version: "hip=7.14.60850".to_owned(),
            framework_source: framework_source.to_owned(),
            rocm_version: "6.4.1".to_owned(),
            ..linux_base()
        }
    }

    /// A remedy has to be able to resolve the mismatch that produced it.
    ///
    /// `check_8` fires when the framework's HIP major differs from the system
    /// ROCm major, and tells the user to pick the wheel index matching their
    /// system. If every index it names is older than any ROCm this CLI installs,
    /// there is nothing to pick: following the instruction installs a torch for
    /// a major the machine does not have, and this same check fires again on the
    /// result. A remedy that re-creates its own precondition is worse than none,
    /// because the user has spent a reinstall to arrive back where they started.
    #[test]
    fn the_wheel_mismatch_remedy_can_resolve_the_mismatch_it_reports() {
        let machine = Examination {
            framework: "pytorch".to_owned(),
            framework_rocm_version: "hip=6.4.43482".to_owned(),
            framework_source: "path".to_owned(),
            rocm_version: "7.14.60850".to_owned(),
            ..linux_base()
        };

        let hit = check_8_wheel_rocm_mismatch(&machine, "");
        // Non-vacuity: without the finding there is no remedy to judge, and the
        // assertions below would pass against a check that never fires.
        let fix = hit
            .fix
            .as_ref()
            .expect("a HIP 6 torch against a ROCm 7 system is the mismatch this check exists for");

        let named = torch_rocm_indexes_named_in(fix.commands.iter().map(String::as_str));
        assert!(
            !named.is_empty(),
            "the remedy names no wheel index at all, so \"pick the one that matches\" points at \
             nothing: {:?}",
            fix.commands
        );
        // Every index, not merely one of them. "At least one is current" would
        // pass with a stale line sitting beside a good one, and the user picking
        // between them has no way to tell which is which -- that is the same
        // position this defect put them in to begin with.
        let stale: Vec<_> = named
            .iter()
            .filter(|(major, _)| *major < OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS)
            .collect();
        assert!(
            stale.is_empty(),
            "this remedy offers an index older than ROCm {OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS}, \
             which this CLI installs. A user who picks that line installs a torch for a major \
             they do not have, and this same check fires again on the result: {stale:?}"
        );
    }

    #[test]
    fn a_managed_runtimes_hip_is_not_measured_against_the_system_rocm() {
        // A managed runtime's torch loads HIP from a sibling `_rocm_sdk_core`
        // package inside the runtime, never from the system install, so the two
        // majors are free to differ on a perfectly healthy host. Before
        // `examine` probed the runtime this could not fire, because the field it
        // reads was always empty; now that it is populated, the comparison has
        // to be told when it is meaningless -- or fixing the probe would hand
        // every such host a spurious "reinstall torch".
        let managed = diagnose(&hip_major_differs_from_system_rocm("managed-runtime"), "");
        assert!(
            !managed.matched.iter().any(|d| d.id == "fix-8-wheel-rocm"),
            "a managed runtime must not be told to reinstall torch: {:?}",
            managed
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );

        // The same host, same versions, with torch coming from the ambient
        // interpreter: there the comparison is exactly right, and the checker
        // must keep its full strength.
        let ambient = diagnose(&hip_major_differs_from_system_rocm("path"), "");
        let finding = ambient
            .matched
            .iter()
            .find(|d| d.id == "fix-8-wheel-rocm")
            .expect("an ambient torch built against a different ROCm major is a real mismatch");
        assert!(
            finding.score >= MIN_SCORE_FOR_MATCH,
            "the version evidence alone has to establish it: {}",
            finding.score
        );
    }

    #[test]
    fn check_20_is_silent_without_a_confirmed_unsupported_gpu() {
        assert_eq!(
            check_20_gpu_unsupported_by_rocm(&Examination::default(), "").score,
            0,
            "a host with no GPUs at all must not match"
        );

        let supported = Examination {
            gpus: vec![Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                ..Gpu::default()
            }],
            ..Examination::default()
        };
        assert_eq!(
            check_20_gpu_unsupported_by_rocm(&supported, "").score,
            0,
            "a gfx target ROCm supports must not match: {:?}",
            check_20_gpu_unsupported_by_rocm(&supported, "")
        );
    }

    #[test]
    fn check_20_is_silent_when_a_supported_gpu_sits_beside_the_unsupported_one() {
        // An APU+dGPU laptop: a gfx90c iGPU next to a supported discrete card.
        // This host *does* have ROCm support, through the dGPU, so the
        // whole-machine claim must not be made -- and in particular must not
        // out-rank `fix-9-igpu-dgpu`, which is the entry for this pairing.
        let apu_plus_dgpu = Examination {
            os_family: "linux".to_owned(),
            has_amd_gpu: true,
            has_apu: true,
            has_discrete_amd: true,
            gpus: vec![
                Gpu {
                    name: "AMD Radeon Vega 7".to_owned(),
                    gfx_target: "gfx90c".to_owned(),
                    is_amd: true,
                    is_apu: Some(true),
                    ..Gpu::default()
                },
                Gpu {
                    name: "AMD Radeon RX 7900 XTX".to_owned(),
                    gfx_target: "gfx1100".to_owned(),
                    is_amd: true,
                    is_apu: Some(false),
                    ..Gpu::default()
                },
            ],
            env: std::collections::BTreeMap::from([(
                "HIP_VISIBLE_DEVICES".to_owned(),
                "1".to_owned(),
            )]),
            ..Examination::default()
        };
        assert_eq!(
            check_20_gpu_unsupported_by_rocm(&apu_plus_dgpu, "").score,
            0,
            "a supported dGPU beside the unsupported iGPU means the host is not \
             without ROCm support: {:?}",
            check_20_gpu_unsupported_by_rocm(&apu_plus_dgpu, "")
        );

        let report = diagnose(&apu_plus_dgpu, "");
        assert!(
            !report
                .matched
                .iter()
                .any(|d| d.id == "info-20-gpu-no-rocm-support"),
            "and it must not reach the report at all, where its flat 60 would \
             rank above fix-9-igpu-dgpu's 40: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn check_20_is_silent_while_another_amd_gpu_is_still_unidentified() {
        // An AMD GPU with an empty gfx target is undetected, not supported.
        // Claiming the host has no ROCm support while a card we could not
        // identify is in it would turn a detection gap into a false claim --
        // the same trap `host_gpu::gfx_target_has_no_rocm_build`'s doc comment
        // avoids.
        let e = Examination {
            has_amd_gpu: true,
            gpus: vec![
                Gpu {
                    gfx_target: "gfx90c".to_owned(),
                    is_amd: true,
                    ..Gpu::default()
                },
                Gpu {
                    name: "AMD unknown device".to_owned(),
                    gfx_target: String::new(),
                    is_amd: true,
                    ..Gpu::default()
                },
            ],
            ..Examination::default()
        };
        assert_eq!(
            check_20_gpu_unsupported_by_rocm(&e, "").score,
            0,
            "an unidentified AMD GPU blocks the whole-machine claim: {:?}",
            check_20_gpu_unsupported_by_rocm(&e, "")
        );
    }

    #[test]
    fn check_20_names_the_detected_architecture_and_carries_no_commands() {
        let e = Examination {
            gpus: vec![Gpu {
                gfx_target: "gfx90c".to_owned(),
                is_amd: true,
                ..Gpu::default()
            }],
            ..Examination::default()
        };
        let finding = check_20_gpu_unsupported_by_rocm(&e, "");
        assert!(
            finding.score >= MIN_SCORE_FOR_MATCH,
            "a confirmed-unsupported GPU is a real finding: {finding:?}"
        );
        let fix = finding
            .fix
            .as_ref()
            .expect("the catalog entry is carried even though it has no remedy");
        assert_eq!(
            fix.fix_id, "info-20-gpu-no-rocm-support",
            "the embedded fix-id has to be the catalog id `rocm fix` resolves: {finding:?}"
        );
        assert!(
            fix.commands.is_empty() && fix.verify.is_empty(),
            "there is nothing to run and nothing to verify: {finding:?}"
        );
        assert!(
            finding.evidence.iter().any(|ev| ev.contains("gfx90c")),
            "the evidence must name the architecture: {finding:?}"
        );
    }

    #[test]
    fn rendering_keeps_the_apply_with_line_for_a_cause_with_nothing_to_run() {
        let report = DiagnoseReport {
            matched: vec![check_20_gpu_unsupported_by_rocm(
                &Examination {
                    gpus: vec![Gpu {
                        gfx_target: "gfx90c".to_owned(),
                        is_amd: true,
                        ..Gpu::default()
                    }],
                    ..Examination::default()
                },
                "",
            )],
            has_match: true,
            min_score_for_match: MIN_SCORE_FOR_MATCH,
            high_confidence_threshold: HIGH_CONFIDENCE,
            route_when_no_match: Route {
                target: "github".to_owned(),
                url: "https://example.invalid".to_owned(),
            },
            out_of_scope: None,
            model: None,
        };
        let out = render_report_text(&report, 5);
        assert!(
            out.contains("apply with: rocm fix info-20-gpu-no-rocm-support"),
            "the id the report prints has to be one `rocm fix` accepts:\n{out}"
        );
        assert!(
            out.contains("gfx90c"),
            "the evidence must still render:\n{out}"
        );
        // Not rendered from the checker's own `class` -- `check_20` returns the
        // `Fix::default()` PRINT-ONLY until `take_applicability_from_the_catalog`
        // fills it from the catalog, which is what the full `diagnose()` path does and
        // what `fix_13_does_not_fire_on_a_lemonade_vulkan_host_with_no_hip_evidence`
        // asserts. This one pins the lines that do not depend on it.
        assert!(
            !out.contains("   $ "),
            "a cause with no commands must not print an empty command block:\n{out}"
        );
        assert!(
            !out.contains("verify after fix:"),
            "and nothing to verify means no verify line:\n{out}"
        );
    }

    #[test]
    fn arch_not_in_wheel_strong_signal() {
        let mut e = linux_base();
        e.framework = "pytorch".to_owned();
        e.framework_arch_list = vec!["gfx1100".to_owned()];
        e.gpus = vec![Gpu {
            gfx_target: "gfx1151".to_owned(),
            is_amd: true,
            ..Gpu::default()
        }];
        let report = diagnose(&e, "HSA_STATUS_ERROR_INVALID_ISA");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-1-arch");
        // 50 (keyword) + 55 (missing arch), clamped to 100.
        assert_eq!(top.score, 100);
    }

    #[test]
    fn path_missing_names_the_versioned_rocm_root() {
        // A box whose only ROCm is a versioned root used to report an empty
        // rocm_path, so both structural signals here scored zero and the fix
        // could only fire on symptom keywords -- and then named /opt/rocm/bin,
        // which does not exist on that machine.
        //
        // Drive the real resolver rather than hand-setting rocm_path, so this
        // fails if versioned discovery regresses and not just if the check does.
        let root = std::env::temp_dir().join(format!(
            "rocm-core-diagnose-versioned-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        let opt = root.join("opt");
        let install = opt.join("rocm-6.4.1");
        std::fs::create_dir_all(install.join("bin")).expect("plant a fake versioned install");
        std::fs::write(install.join("bin").join("rocminfo"), "").expect("plant the install marker");

        let discovered = discover_rocm_installs_in(std::slice::from_ref(&opt), None);
        let found = discovered
            .first()
            .expect("the resolver must find a versioned-only install");

        let mut e = linux_base();
        e.rocm_path = found.path.to_string_lossy().into_owned();
        e.rocminfo_present = false;
        e.env.insert("PATH".to_owned(), "/usr/bin:/bin".to_owned());

        let report = diagnose(&e, "");
        let top = &report.matched[0];

        assert_eq!(top.id, "fix-6-path");
        assert_eq!(top.score, 70, "50 (rocminfo absent) + 20 (bin not on PATH)");
        let fix = top.fix.as_ref().expect("fix-6 carries a fix");
        assert!(
            fix.summary.contains("rocm-6.4.1/bin"),
            "guidance must name the versioned root the resolver found: {}",
            fix.summary
        );
        assert!(
            !fix.summary.contains("/opt/rocm/bin"),
            "guidance must not name the conventional root that is absent here: {}",
            fix.summary
        );

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn igpu_dgpu_collision_fires_for_rdna3_apu_plus_discrete() {
        // End-to-end guard for EAI-7412: a gfx1103 (Phoenix APU) + gfx1100
        // (Navi 31 dGPU) box must now trigger the iGPU+dGPU collision fix,
        // which was silently suppressed while gfx1100 was misclassified as an
        // APU. The note must name the discrete target rather than telling the
        // user the higher-numbered gfx target is the dGPU (false here).
        let mut e = linux_base();
        e.has_apu = true;
        e.has_discrete_amd = true;
        e.gpus = vec![
            Gpu {
                gfx_target: "gfx1103".to_owned(),
                is_amd: true,
                is_apu: Some(true),
                ..Gpu::default()
            },
            Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                is_apu: Some(false),
                ..Gpu::default()
            },
        ];
        let report = diagnose(&e, "torch crashes with a segfault");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-9-igpu-dgpu")
            .expect("iGPU+dGPU collision should be diagnosed");
        let note = hit.fix.as_ref().unwrap().notes.join(" ");
        assert!(note.contains("gfx1100"), "note must name the dGPU: {note}");
        assert!(
            !note.contains("usually the higher-numbered"),
            "note must not repeat the old wrong gfx-number heuristic: {note}"
        );
        assert!(
            note.contains("Without --device-index") && !note.contains("marked AUTO"),
            "note must warn that fix-9 is a no-op without --device-index: {note}"
        );
    }

    /// The `note:` lines `render_report_text` prints for one diagnosis, in
    /// order, with the `   note: ` prefix stripped.
    fn rendered_notes(report: &DiagnoseReport, id: &str) -> Vec<String> {
        let text = render_report_text(report, report.matched.len());
        let lines: Vec<&str> = text.lines().collect();
        let id_line = lines
            .iter()
            .position(|l| l.trim_start() == format!("id: {id}"))
            .unwrap_or_else(|| panic!("{id} should appear in the rendered report:\n{text}"));
        lines[id_line..]
            .iter()
            .take_while(|l| !l.trim_start().starts_with("apply with:"))
            .filter_map(|l| l.trim_start().strip_prefix("note: "))
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn fix_9_states_the_device_index_caveat_exactly_once() {
        // `render_report_text` prints every note on its own line, so two notes
        // that both say "--device-index is required for the auto-apply to do
        // anything" reach the user as two bullets saying the same thing. The
        // caveat is worth stating -- once. Checked on the rendered lines rather
        // than on `Fix::notes`, because the duplicate is only a defect at the
        // point where it is printed, and checked on both OS branches, which
        // build their `Fix` separately and have drifted apart before.
        for os_family in ["linux", "windows"] {
            let report = diagnose(&igpu_dgpu_host(os_family), "torch crashes with a segfault");
            let notes = rendered_notes(&report, "fix-9-igpu-dgpu");
            let caveats: Vec<&String> = notes
                .iter()
                .filter(|n| n.contains("--device-index"))
                .collect();
            assert_eq!(
                caveats.len(),
                1,
                "{os_family}: the --device-index caveat must be stated once, not {}; notes: {notes:#?}",
                caveats.len()
            );
            let mut seen = std::collections::BTreeSet::new();
            for note in &notes {
                assert!(
                    seen.insert(note.clone()),
                    "{os_family}: fix-9 prints the same note twice: {note}"
                );
            }
        }
    }

    #[test]
    fn fix_9_igpu_dgpu_waits_for_an_argument_rather_than_claiming_it_will_act() {
        // This pinned `auto_applicable: true` until the catalog gained a class.
        // The claim was wrong: asked plainly, both arms of
        // `run_hip_visible_devices` print the query that identifies the discrete
        // GPU and return without touching the machine, so a user -- and an agent
        // reading the same output -- was told a change was coming that never
        // came, and given exit 0 to confirm it. The entry is NEEDS-ARG: it acts
        // only once `--device-index` names a target.
        //
        // Kept pinned for the original reason, with the expectation corrected.
        // It still flips the `Fix` field `rocm diagnose --json` serialises and
        // the `flags:` line `render_report_text` prints, so a regression back to
        // an unconditional auto claim fails here rather than needing someone to
        // eyeball the output.
        let mut e = linux_base();
        e.has_apu = true;
        e.has_discrete_amd = true;
        e.gpus = vec![
            Gpu {
                gfx_target: "gfx1103".to_owned(),
                is_amd: true,
                is_apu: Some(true),
                ..Gpu::default()
            },
            Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                is_apu: Some(false),
                ..Gpu::default()
            },
        ];
        let report = diagnose(&e, "torch crashes with a segfault");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-9-igpu-dgpu")
            .expect("iGPU+dGPU collision should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        assert!(
            !fix.auto_applicable,
            "fix-9-igpu-dgpu does nothing until it is told which device to pin, so the report \
             must not say the CLI will carry it out"
        );
        assert_eq!(
            fix.class,
            crate::fix::FixClass::NeedsArgument,
            "the report's class has to be the catalog's class for this entry on Linux"
        );

        let text = render_report_text(&report, report.matched.len());
        let lines: Vec<&str> = text.lines().collect();
        let id_line = lines
            .iter()
            .position(|l| l.trim_start() == "id: fix-9-igpu-dgpu")
            .expect("fix-9-igpu-dgpu should appear in the rendered report");
        let flags_line = lines[id_line..]
            .iter()
            .find(|l| l.trim_start().starts_with("flags:"))
            .expect("fix-9-igpu-dgpu should have a flags: line");
        // Exact match, not `contains`: fix-9 carries only the class flag, and
        // the whole point of the class is that "will not run it" has more than
        // one reason. A `contains` check would pass against the print-only
        // wording, which would send the user off to run the steps by hand when
        // the CLI will in fact do it for them once given the argument. The
        // argument is named rather than described, which is why this text comes
        // from the catalog and not from a literal here.
        assert_eq!(
            flags_line.trim_start(),
            "flags: needs --device-index before `rocm fix` will run it",
            "rendered flags: line for fix-9-igpu-dgpu: {flags_line}"
        );
    }

    #[test]
    fn arch_covered_is_negative_and_filtered() {
        let mut e = linux_base();
        e.framework = "pytorch".to_owned();
        e.framework_arch_list = vec!["gfx1151".to_owned()];
        e.gpus = vec![Gpu {
            gfx_target: "gfx1151".to_owned(),
            is_amd: true,
            ..Gpu::default()
        }];
        // No symptom: -30 from covered arch => score <= 0 => not reported.
        let report = diagnose(&e, "");
        assert!(report.matched.iter().all(|d| d.id != "fix-1-arch"));
    }

    #[test]
    fn an_installed_rocm_is_not_reported_as_missing_from_path_on_wsl() {
        // `rocminfo_present` is set by the bare-metal GPU probe, which WSL skips.
        // Left at its default it read as "rocminfo is not on PATH", so fix-6 --
        // enabled on WSL because PATH problems are real there -- scored 50 on
        // every WSL host that had ROCm installed.
        let mut e = wsl_base();
        e.rocm_path = "/opt/rocm".to_owned();
        e.rocminfo_present = true;
        e.env
            .insert("PATH".to_owned(), "/opt/rocm/bin:/usr/bin:/bin".to_owned());
        let report = diagnose(&e, "");
        assert!(
            !report.matched.iter().any(|d| d.id == "fix-6-path"),
            "ROCm is installed and on PATH here: {:?}",
            report.matched
        );
    }

    #[test]
    fn an_environment_override_is_still_diagnosed_on_wsl() {
        // The gain from the family split: HSA_OVERRIDE_GFX_VERSION has nothing to
        // do with the kernel module, so it was always a valid question on WSL --
        // but the old wholesale skip meant it went unanswered there.
        let mut e = wsl_base();
        e.env
            .insert("HSA_OVERRIDE_GFX_VERSION".to_owned(), "11.0.0".to_owned());
        let report = diagnose(&e, "memory access fault page fault");
        assert!(
            report
                .matched
                .iter()
                .any(|d| d.id == "fix-2-unset-override"),
            "matched: {:?}",
            report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
        );
    }

    /// The Windows `Fix` for fix-9, which only `rocm diagnose` prints: its verify
    /// differs from the catalog's because it has to run in PowerShell.
    fn windows_igpu_fix() -> Fix {
        check_9_igpu_dgpu_collision(&igpu_dgpu_host("windows"), "torch crashes with a segfault")
            .fix
            .expect("an APU beside a discrete GPU on Windows gets a fix")
    }

    /// Runs on every host. The Windows fix-9 verify sets `HIP_VISIBLE_DEVICES`
    /// to `1` for the process, then runs one single-quoted python program, all
    /// in one double-quoted `-Command` argument.
    /// `the_igpu_verify_pins_the_variable_it_names` runs it, on Windows.
    #[test]
    fn the_windows_igpu_verify_is_one_quoted_powershell_argument() {
        let verify = windows_igpu_fix().verify;
        let body = verify
            .strip_prefix("powershell -NoProfile -Command \"")
            .and_then(|rest| rest.strip_suffix('"'))
            .unwrap_or_else(|| panic!("not a quoted `powershell -Command` argument: {verify}"));
        assert!(
            !body.contains('"'),
            "an inner quote ends the argument: {body}"
        );
        assert!(
            body.starts_with("[Environment]::SetEnvironmentVariable('HIP_VISIBLE_DEVICES','1','Process'); python -c '"),
            "{body}"
        );
    }

    /// Runs the fix-9 Windows verify line in cmd and in Windows PowerShell with
    /// a stand-in `torch` module first on `PYTHONPATH`. The module's
    /// `cuda.device_count()` returns what `HIP_VISIBLE_DEVICES` holds, so the
    /// line, unchanged, prints `1` only if it set that variable, to that value,
    /// before python started. The variable is removed from the child's
    /// environment first, so a value left on the host cannot satisfy it. Python
    /// is on PATH on the Windows CI hosts. Only compiled on Windows.
    #[cfg(windows)]
    #[test]
    fn the_igpu_verify_pins_the_variable_it_names() {
        let verify = windows_igpu_fix().verify;
        let dir = std::env::temp_dir().join(format!("rocm igpu torch {}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the stand-in module directory is created");
        std::fs::write(
            dir.join("torch.py"),
            "import os\n\nclass cuda:\n    @staticmethod\n    def device_count():\n        \
             return os.environ.get('HIP_VISIBLE_DEVICES', 'UNSET')\n",
        )
        .expect("the stand-in torch is written");
        let python_path = dir.to_string_lossy().into_owned();
        let runs = crate::fix::run_typed_in_cmd_and_powershell(
            &verify,
            &[
                ("HIP_VISIBLE_DEVICES", None),
                ("PYTHONPATH", Some(python_path.as_str())),
            ],
        );
        let _ = std::fs::remove_dir_all(&dir);
        for run in runs {
            let crate::fix::ShellRun {
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
            assert_eq!(stdout.trim(), "1", "{shell}, stderr: {stderr}");
        }
    }

    /// fix-6's verify line lives in the `rocm fix` catalog and in the Fix that
    /// `check_6_path_missing` builds. The shared-id guard in `catalog.rs`
    /// (`diagnosis_remediation_matches_the_fix_catalog_for_shared_fix_ids`)
    /// covers fix-13, fix-14, fix-15 and fix-16; this one keeps fix-6's two copies from drifting.
    ///
    /// The Windows arm of the checker is not compared with the catalog on
    /// purpose. It names the SDK's own `hipInfo.exe` in a PowerShell command,
    /// which the single catalog line cannot do, so it is checked for being a
    /// PowerShell command instead. Its text is also covered by the Windows sweep in
    /// `catalog.rs` (`windows_diagnosis_remediation_passes_the_powershell_text_rule`).
    #[test]
    fn the_path_fix_verify_matches_the_catalog_copy() {
        let (verify, _, _) = crate::fix::recipe_verify_notes_and_run_commands("fix-6-path")
            .expect("fix-6-path is in the recipe catalog");
        // A ROCm install whose bin directory has no `rocminfo` or `hipInfo`.
        let host = |os_family: &str| Examination {
            os_family: os_family.to_owned(),
            rocm_path: "/opt/rocm".to_owned(),
            hip_sdk_path: r"C:\Program Files\AMD\ROCm\6.4".to_owned(),
            ..Examination::default()
        };
        let linux = check_6_path_missing(&host("linux"), "hipcc: command not found")
            .fix
            .expect("check 6 builds a Fix for its own symptom");
        // The Linux arm runs the catalog's Linux line and nothing else. The
        // line shared with Windows (`hipcc --version`) is not chained on: a
        // runtime-only install has no `hipcc`, and the catalog note says that
        // line fails there even with PATH fixed.
        let catalog_linux = crate::fix::recipe_linux_verify("fix-6-path")
            .expect("fix-6-path has a Linux verify line in the recipe catalog");
        assert_eq!(
            linux.verify, catalog_linux,
            "the verify line for fix-6-path diverged between `diagnose` and `rocm fix`"
        );
        assert!(
            !linux.verify.contains("hipcc") && !linux.verify.contains(verify),
            "the Linux verify must pass on a runtime-only install, which has no hipcc: {}",
            linux.verify
        );
        let windows = check_6_path_missing(&host("windows"), "hipcc: command not found")
            .fix
            .expect("check 6 builds a Fix for its own symptom");
        assert!(
            windows
                .verify
                .starts_with("powershell -NoProfile -Command "),
            "the Windows verify line must run in PowerShell: {}",
            windows.verify
        );
    }

    /// A folder with an apostrophe in it, such as `C:\Users\O'Neil\ROCm\bin`,
    /// must not end a PowerShell string literal early in either printed
    /// fix-6 line: the PATH write or the verify.
    #[test]
    fn the_windows_path_fix_doubles_an_apostrophe_in_the_sdk_folder() {
        let e = Examination {
            os_family: "windows".to_owned(),
            hip_sdk_path: r"C:\Users\O'Neil\ROCm".to_owned(),
            hipinfo_present: false,
            ..Examination::default()
        };
        let fix = check_6_path_missing(&e, "hipInfo.exe not found")
            .fix
            .expect("check 6 builds a Fix for its own symptom");
        let write = fix
            .commands
            .iter()
            .find(|c| c.contains("SetEnvironmentVariable"))
            .expect("the PATH write is printed");
        assert!(
            write.contains(r"'C:\Users\O''Neil\ROCm\bin;'"),
            "the apostrophe must be doubled: {write}"
        );
        assert!(
            fix.verify
                .contains(r"& 'C:\Users\O''Neil\ROCm\bin\hipInfo.exe'"),
            "the apostrophe must be doubled: {}",
            fix.verify
        );
    }

    /// fix-2's Linux verify line is in the `rocm fix` catalog (`Verify (Linux,
    /// WSL)`) and in the Linux arm of `check_2_hsa_override_unneeded`, which
    /// runs it and then the GPU check the catalog line cannot carry (it is
    /// shared with Windows and cannot assume torch). Neither may need `python`:
    /// a stock Ubuntu has `python3` only. The Windows arm names the User scope
    /// in a PowerShell command and is covered by the Windows sweep in `catalog.rs`
    /// (`windows_diagnosis_remediation_passes_the_powershell_text_rule`).
    #[test]
    fn the_override_fix_verify_matches_the_catalog_copy() {
        let catalog_linux = crate::fix::recipe_linux_verify("fix-2-unset-override")
            .expect("fix-2-unset-override has a Linux verify line in the recipe catalog");
        let e = Examination {
            os_family: "linux".to_owned(),
            env: std::iter::once(("HSA_OVERRIDE_GFX_VERSION".to_owned(), "11.0.0".to_owned()))
                .collect(),
            ..Examination::default()
        };
        let linux = check_2_hsa_override_unneeded(&e, "")
            .fix
            .expect("check 2 builds a Fix when the override is set");
        assert_eq!(
            linux.verify,
            format!(
                "{catalog_linux}; python3 -c \"import torch; print(torch.cuda.is_available())\""
            ),
            "the verify line for fix-2-unset-override diverged between `diagnose` and `rocm fix`"
        );
        // A stock Ubuntu has `python3` and no `python`.
        assert!(
            !linux.verify.split_whitespace().any(|word| word == "python"),
            "{}",
            linux.verify
        );
        // Clearing the override must leave the GPU usable; the printed line is
        // what tells the user whether it did.
        assert!(
            linux.verify.contains("torch.cuda.is_available()"),
            "{}",
            linux.verify
        );
    }

    /// Writes each `(name, script)` as an executable in a new directory, to be
    /// the whole `PATH` of a shell that runs a Linux verify line.
    #[cfg(unix)]
    fn stub_bin_dir(stubs: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("the stub directory is created");
        for (name, script) in stubs {
            let path = dir.path().join(name);
            crate::fake_executable::write_fake_executable(&path, format!("#!/bin/sh\n{script}\n"))
                .expect("a stub is planted");
        }
        dir
    }

    /// Runs one printed line in `/bin/sh` whose `PATH` is only `bin` and whose
    /// environment is only `PATH` and `env`. Returns whether it exited 0, and
    /// its stdout and stderr.
    #[cfg(unix)]
    fn run_in_sh(
        line: &str,
        bin: &std::path::Path,
        env: &[(&str, &str)],
    ) -> (bool, String, String) {
        let out = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(line)
            .env_clear()
            .env("PATH", bin)
            .envs(env.iter().copied())
            .output()
            .expect("/bin/sh runs");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// The fix-6 line `rocm diagnose` prints on Linux, run on a runtime-only
    /// install: `rocminfo` is on PATH once the fix has worked, and there is no
    /// `hipcc`. The line must succeed and show the GPU. The old line, with
    /// `&& hipcc --version` chained on, fails on the same PATH, which is what
    /// the catalog note says about the plain verify line.
    #[cfg(unix)]
    #[test]
    fn the_path_fix_verify_passes_on_a_runtime_only_linux_install() {
        let real_head = ["/usr/bin/head", "/bin/head"]
            .into_iter()
            .find(|p| std::path::Path::new(p).exists())
            .expect("head is installed");
        let bin = stub_bin_dir(&[
            (
                "rocminfo",
                "for n in 1 2 3 4 5 6 7 8; do echo \"agent $n\"; done",
            ),
            ("head", &format!("exec {real_head} \"$@\"")),
        ]);
        let fix = check_6_path_missing(
            &Examination {
                os_family: "linux".to_owned(),
                rocm_path: "/opt/rocm".to_owned(),
                ..Examination::default()
            },
            "rocminfo: command not found",
        )
        .fix
        .expect("check 6 builds a Fix for its own symptom");
        let (ok, stdout, stderr) = run_in_sh(&fix.verify, bin.path(), &[]);
        assert!(ok, "{}: {stdout}\n{stderr}", fix.verify);
        assert_eq!(
            stdout.lines().collect::<Vec<_>>(),
            ["agent 1", "agent 2", "agent 3", "agent 4", "agent 5"]
        );
        // The control: the chain this arm used to print fails on this PATH.
        let (old_ok, _, old_err) =
            run_in_sh("rocminfo | head -n 5 && hipcc --version", bin.path(), &[]);
        assert!(!old_ok && old_err.contains("hipcc"), "{old_err}");
    }

    /// The fix-2 line `rocm diagnose` prints on Linux, run on a system that has
    /// `python3` and no `python`, with the override set and with it cleared. It
    /// must print the override's value (or OK_UNSET) and then the torch check,
    /// and exit 0. The line this arm used to print, which called `python`,
    /// fails on the same PATH.
    #[cfg(unix)]
    #[test]
    fn the_override_fix_verify_runs_where_only_python3_exists() {
        let bin = stub_bin_dir(&[("python3", "echo True")]);
        let fix = check_2_hsa_override_unneeded(
            &Examination {
                os_family: "linux".to_owned(),
                env: std::iter::once(("HSA_OVERRIDE_GFX_VERSION".to_owned(), "11.0.0".to_owned()))
                    .collect(),
                ..Examination::default()
            },
            "",
        )
        .fix
        .expect("check 2 builds a Fix when the override is set");
        for (env, shown) in [
            (&[("HSA_OVERRIDE_GFX_VERSION", "11.0.0")][..], "11.0.0"),
            (&[][..], "OK_UNSET"),
        ] {
            let (ok, stdout, stderr) = run_in_sh(&fix.verify, bin.path(), env);
            assert!(ok, "{}: {stdout}\n{stderr}", fix.verify);
            assert_eq!(
                stdout.lines().collect::<Vec<_>>(),
                [shown, "True"],
                "{stderr}"
            );
        }
        // The control: calling `python` on this PATH is what the reviewer saw.
        let (old_ok, _, old_err) = run_in_sh("python -c 'print(1)'", bin.path(), &[]);
        assert!(!old_ok && old_err.contains("python"), "{old_err}");
    }
}
