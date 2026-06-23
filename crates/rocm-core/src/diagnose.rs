//! ROCm failure-mode diagnosis.
//!
//! Rust port of the `rocm-doctor` skill's `diagnose.py`. It matches an
//! [`Examination`] (plus optional user symptom text) against a **closed list**
//! of known misconfigurations and returns ranked [`Diagnosis`] results, each
//! with the evidence it used and a [`Fix`] (plan + verify step). When nothing
//! matches it routes the user upstream rather than guessing.
//!
//! The catalog is deliberately closed: new failure modes are added here, not
//! invented at runtime. Keyword tables, thresholds, and tracker URLs are the
//! data; the per-check logic mirrors `diagnose.py` field-for-field so the two
//! stay behaviorally identical. See `plans/rocm-doctor-examine-migration-plan.md`.

use crate::examine::Examination;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// At/above this score a diagnosis is treated as a real match.
pub const MIN_SCORE_FOR_MATCH: i32 = 50;
/// At/above this score the agent may propose the fix immediately.
pub const HIGH_CONFIDENCE: i32 = 75;

/// A proposed remediation for a [`Diagnosis`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fix {
    pub summary: String,
    pub commands: Vec<String>,
    pub needs_sudo: bool,
    pub needs_reboot: bool,
    pub needs_relogin: bool,
    pub fix_id: String,
    pub auto_applicable: bool,
    pub notes: Vec<String>,
    pub verify: String,
}

/// A single scored match against the failure-mode catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Diagnosis {
    pub id: String,
    pub title: String,
    pub score: i32,
    pub evidence: Vec<String>,
    pub fix: Option<Fix>,
}

/// Where to send a report when no catalog entry matches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Route {
    pub target: String,
    pub url: String,
}

/// The full diagnosis output (mirrors `diagnose.py --json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnoseReport {
    /// All nonzero-score diagnoses, highest score first.
    pub matched: Vec<Diagnosis>,
    pub min_score_for_match: i32,
    pub high_confidence_threshold: i32,
    pub route_when_no_match: Route,
}

impl DiagnoseReport {
    /// Whether at least one diagnosis cleared [`MIN_SCORE_FOR_MATCH`].
    #[must_use]
    pub fn has_match(&self) -> bool {
        self.matched.iter().any(|d| d.score >= MIN_SCORE_FOR_MATCH)
    }
}

/// Upstream tracker for a framework key.
fn upstream_tracker(target: &str) -> &'static str {
    match target {
        "pytorch" => "https://github.com/pytorch/pytorch/issues  (tag with rocm label)",
        "llama-cpp" => "https://github.com/ggml-org/llama.cpp/issues",
        "lemonade" => "https://github.com/lemonade-sdk/lemonade/issues",
        "ollama" => "https://github.com/ollama/ollama/issues",
        "lm-studio" => "https://lmstudio.ai/docs/app  (use in-app support; no public repo)",
        "amdgpu-install" => "https://repo.radeon.com  (raise via your AMD support contact)",
        _ => "https://github.com/ROCm/ROCm/issues",
    }
}

// ---------------------------------------------------------------------------
// Symptom keyword tables: (regex, weight, evidence-label). Patterns are
// lowercase and matched against the lowercased symptom.
// ---------------------------------------------------------------------------

type KeywordTable = &'static [(&'static str, i32, &'static str)];

const KEYWORDS_INVALID_ISA: KeywordTable = &[
    (
        "hiperrornobinaryforgpu",
        45,
        "error mentions hipErrorNoBinaryForGpu",
    ),
    (
        "hsa_status_error_invalid_isa",
        50,
        "error mentions HSA_STATUS_ERROR_INVALID_ISA",
    ),
    (
        "invalid device function",
        40,
        "error mentions 'invalid device function'",
    ),
    (
        "no kernel image is available",
        35,
        "error mentions 'no kernel image is available'",
    ),
    (
        r"gfx\d{3,4}.* not (?:in|on) .*arch",
        35,
        "error names a missing gfx in arch list",
    ),
];

const KEYWORDS_KFD_PERMISSION: KeywordTable = &[
    (
        "unable to open /dev/kfd",
        50,
        "error mentions /dev/kfd open failure",
    ),
    (
        r"/dev/kfd.*permission denied",
        45,
        "error mentions /dev/kfd permission denied",
    ),
    (
        "hsa_status_error_out_of_resources",
        25,
        "HSA out-of-resources (often perms)",
    ),
    ("failed to open kfd", 35, "error mentions kfd open failure"),
];

const KEYWORDS_MODULE_NOT_LOADED: KeywordTable = &[
    (
        "rock module is not loaded",
        50,
        "rocminfo says ROCk module is NOT loaded",
    ),
    ("no devices? found", 20, "vague 'no devices found'"),
    ("hsa_status_error", 10, "HSA error (broad)"),
];

const KEYWORDS_PATH_MISSING: KeywordTable = &[
    ("rocminfo: command not found", 50, "rocminfo not on PATH"),
    ("command not found.*hipcc", 40, "hipcc not on PATH"),
    ("/opt/rocm/bin", 15, "user mentions /opt/rocm/bin"),
];

const KEYWORDS_LIB_MISMATCH: KeywordTable = &[
    (r"libamdhip64\.so", 50, "error mentions libamdhip64.so"),
    ("libhsa-runtime", 45, "error mentions libhsa-runtime"),
    ("libhipblas", 40, "error mentions libhipblas"),
    (
        r"amdhip64_\d+\.dll",
        50,
        "error mentions amdhip64_X.dll (Windows)",
    ),
    (r"hipblas\.dll", 40, "error mentions hipblas.dll (Windows)"),
    ("cannot open shared object file", 25, "ldopen failure"),
    ("dll load failed", 25, "Windows DLL load failure"),
    ("version `?glibc", 5, "tangential glibc version error"),
];

const KEYWORDS_HIP_SDK_MISSING: KeywordTable = &[
    ("amdhip64.*not found", 50, "error names amdhip64 missing"),
    ("could not find hip", 40, "error mentions HIP not found"),
    ("hip_path.*not set", 35, "user mentions HIP_PATH unset"),
    (
        "hipinfo.*not recognized",
        45,
        "Windows says hipInfo is not a command",
    ),
];

const KEYWORDS_MSVC_REDIST: KeywordTable = &[
    (
        r"vcruntime140(?:_1)?\.dll",
        50,
        "error mentions vcruntime140 / vcruntime140_1",
    ),
    (
        r"api-ms-win-crt-.*\.dll",
        35,
        "error mentions api-ms-win-crt-* DLL",
    ),
    (
        "the (program|application) can't start because",
        25,
        "Windows missing-DLL dialog text",
    ),
    (r"msvcp140\.dll", 30, "error mentions msvcp140.dll"),
];

const KEYWORDS_REPO_BROKEN: KeywordTable = &[
    (r"404.*repo\.radeon\.com", 50, "404 against repo.radeon.com"),
    (
        "release file (is )?not (yet )?valid",
        30,
        "apt 'release file not valid'",
    ),
    (
        "the following packages have unmet dependencies",
        25,
        "apt unmet dependencies",
    ),
    (
        "unable to locate package rocm",
        35,
        "apt cannot find ROCm package",
    ),
];

const KEYWORDS_CONTAINER: KeywordTable = &[
    (
        "hsa_status_error.*permission",
        20,
        "HSA permission error (often container)",
    ),
    (r"/dev/dri.*permission", 30, "/dev/dri permission failure"),
    ("failed to open device", 25, "device open failure"),
];

const KEYWORDS_IOMMU_HANG: KeywordTable = &[
    ("hang", 20, "user mentions 'hang'"),
    ("deadlock", 20, "user mentions deadlock"),
    ("timed out waiting", 25, "ring/queue timeout"),
    ("iommu", 30, "user mentions iommu"),
];

const KEYWORDS_DPKG_BROKEN: KeywordTable = &[
    ("half[- ]configured", 50, "dpkg 'half-configured'"),
    ("dkms .*failed", 45, "DKMS build failure"),
    ("dpkg: error", 25, "generic dpkg error"),
    (
        "sub-process /usr/bin/dpkg returned",
        25,
        "apt mentions dpkg failure",
    ),
    ("--accept-eula", 40, "user mentions --accept-eula"),
];

const KEYWORDS_PAGE_FAULT: KeywordTable = &[
    ("page fault", 40, "user mentions page fault"),
    ("vm_fault", 35, "kernel vm_fault"),
    ("hw_fault", 30, "amdgpu HW fault"),
    ("out_of_registers", 30, "compiler OUT_OF_REGISTERS"),
];

/// Score the strongest (top-2) keyword matches in `table` against `symptom`.
fn keyword_score(symptom: &str, table: KeywordTable) -> (i32, Vec<String>) {
    if symptom.is_empty() {
        return (0, Vec::new());
    }
    let sym = symptom.to_lowercase();
    let mut hits: Vec<(i32, &'static str)> = Vec::new();
    for (pattern, weight, label) in table {
        if Regex::new(pattern).is_ok_and(|re| re.is_match(&sym)) {
            hits.push((*weight, label));
        }
    }
    if hits.is_empty() {
        return (0, Vec::new());
    }
    // Mirror diagnose.py's `hits.sort(reverse=True)`: weight desc, then label desc.
    hits.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(a.1)));
    hits.truncate(2);
    let score = hits.iter().map(|(w, _)| *w).sum();
    let labels = hits.iter().map(|(_, l)| (*l).to_owned()).collect();
    (score, labels)
}

/// Whether `symptom` (lowercased) matches `pattern`.
fn symptom_matches(symptom: &str, pattern: &str) -> bool {
    !symptom.is_empty() && Regex::new(pattern).is_ok_and(|re| re.is_match(&symptom.to_lowercase()))
}

// ---------------------------------------------------------------------------
// Examination accessors
// ---------------------------------------------------------------------------

fn amd_gfx_targets(e: &Examination) -> Vec<String> {
    e.gpus
        .iter()
        .filter(|g| g.is_amd && !g.gfx_target.is_empty())
        .map(|g| g.gfx_target.clone())
        .collect()
}

fn amd_gpu_count(e: &Examination) -> usize {
    e.gpus.iter().filter(|g| g.is_amd).count()
}

fn zero(id: &str, title: &str) -> Diagnosis {
    Diagnosis {
        id: id.to_owned(),
        title: title.to_owned(),
        ..Diagnosis::default()
    }
}

fn finalize(id: &str, title: &str, score: i32, evidence: Vec<String>, fix: Fix) -> Diagnosis {
    Diagnosis {
        id: id.to_owned(),
        title: title.to_owned(),
        score: score.min(100),
        evidence,
        fix: Some(fix),
    }
}

// ---------------------------------------------------------------------------
// Per-misconfiguration checkers (1:1 with diagnose.py)
// ---------------------------------------------------------------------------

fn check_1_arch_not_in_wheel(e: &Examination, symptom: &str) -> Diagnosis {
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
            "# Recommended: PyTorch ROCm nightly that ships the gfx115x kernels.".to_owned(),
            "pip uninstall -y torch torchvision torchaudio".to_owned(),
            "pip install --pre torch torchvision torchaudio \\\n  --index-url https://download.pytorch.org/whl/nightly/rocm6.4".to_owned(),
            "# llama.cpp: rebuild with AMDGPU_TARGETS set to this GPU's gfx.".to_owned(),
            "# cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx_target>".to_owned(),
        ],
        fix_id: "fix-1-arch".to_owned(),
        auto_applicable: false,
        verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())\"".to_owned(),
        notes: vec![
            "TheRock (rocm/TheRock) ships nightly per-gfx wheels and is the preferred fallback when the official pytorch wheel index does not yet cover your gfx target.".to_owned(),
        ],
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

fn check_2_hsa_override_unneeded(e: &Examination, symptom: &str) -> Diagnosis {
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
                "setx HSA_OVERRIDE_GFX_VERSION \"\"".to_owned(),
                "# Or remove via System Properties -> Environment Variables.".to_owned(),
            ],
            fix_id: "fix-2-unset-override".to_owned(),
            auto_applicable: true,
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
            auto_applicable: true,
            verify: "env | grep HSA_OVERRIDE_GFX_VERSION || echo OK_UNSET; python -c \"import torch; print(torch.cuda.is_available())\"".to_owned(),
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

fn check_3_rocm_kernel_unsupported(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let kernel = &e.kernel_release;
    let distro = &e.distro_id;
    let distro_v = &e.distro_version;
    let rocm_version = &e.rocm_version;

    if !rocm_version.is_empty() && e.amdgpu_loaded == Some(false) {
        score += 30;
        evidence.push(format!(
            "ROCm {rocm_version} is installed but the amdgpu kernel module is not loaded; this is typical when DKMS failed against an unsupported kernel."
        ));
    }

    let (_kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_DPKG_BROKEN);
    if kw_ev.iter().any(|l| l.to_lowercase().contains("dkms")) {
        score += 30;
        evidence.extend(kw_ev);
    }

    if score <= 0 {
        return zero("fix-3-rocm-kernel", "ROCm/distro/kernel triple unsupported");
    }
    let fix = Fix {
        summary: "Cross-check your kernel/distro against the live AMD compatibility matrix before reinstalling.".to_owned(),
        commands: vec![
            format!("# Current: kernel={kernel} distro={distro} {distro_v} rocm={rocm_version}"),
            "# Compare to the live AMD matrix:".to_owned(),
            "#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html".to_owned(),
            "# If your kernel is above the supported range, install the HWE".to_owned(),
            "# kernel that matches ROCm, or rerun amdgpu-install with --no-dkms.".to_owned(),
        ],
        fix_id: "fix-3-rocm-kernel".to_owned(),
        auto_applicable: false,
        needs_reboot: true,
        verify: "lsmod | grep amdgpu && rocminfo | head -n 20".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-3-rocm-kernel",
        "ROCm version + distro/kernel form an unsupported triple",
        score,
        evidence,
        fix,
    )
}

fn check_4_render_group(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    if e.in_render_group == Some(false) {
        score += 35;
        evidence.push("user is NOT in the 'render' group".to_owned());
    }
    if e.in_video_group == Some(false) {
        score += 10;
        evidence.push("user is NOT in the 'video' group".to_owned());
    }
    if let Some(kfd) = &e.kfd
        && kfd.exists
        && kfd.user_can_write == Some(false)
    {
        score += 25;
        evidence.push(format!(
            "/dev/kfd exists (mode {}, group {}) but the current user can't write to it",
            kfd.mode, kfd.owner_group
        ));
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_KFD_PERMISSION);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-4-render-group", "User missing render/video group");
    }
    let kfd_group = e
        .kfd
        .as_ref()
        .map(|k| k.owner_group.clone())
        .filter(|g| !g.is_empty())
        .unwrap_or_else(|| "render".to_owned());
    let fix = Fix {
        summary: format!("Add the current user to '{kfd_group}' (and 'video' for safety) and log out/in."),
        commands: vec![format!("sudo usermod -a -G {kfd_group},video \"$USER\"")],
        needs_sudo: true,
        needs_relogin: true,
        fix_id: "fix-4-render-group".to_owned(),
        auto_applicable: true,
        verify: "groups | tr ' ' '\\n' | grep -E '^(render|video)$' && ls -l /dev/kfd && rocminfo | head -n 5".to_owned(),
        notes: vec![
            "Group membership only takes effect after a full re-login (or reboot). `newgrp render` will give the current shell access but not other terminals or services.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-4-render-group",
        "User not in render/video group (or /dev/kfd owned by the other group)",
        score,
        evidence,
        fix,
    )
}

fn check_5_amdgpu_blacklisted(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let blacklisted = &e.amdgpu_blacklisted_in;
    if !blacklisted.is_empty() {
        score += 55;
        evidence.push(format!("amdgpu is blacklisted in: {blacklisted:?}"));
    }
    if e.amdgpu_loaded == Some(false) {
        score += 35;
        evidence.push("amdgpu module is not loaded".to_owned());
    }
    if e.rocminfo_status == "not-loaded" {
        score += 25;
        evidence.push("rocminfo says 'ROCk module is NOT loaded'".to_owned());
    }
    if e.secure_boot == "enabled" && e.amdgpu_loaded == Some(false) {
        score += 10;
        evidence.push("Secure Boot is enabled and amdgpu didn't load -- DKMS modules are often blocked until you sign them or disable Secure Boot.".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_MODULE_NOT_LOADED);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-5-amdgpu-load", "amdgpu not loaded");
    }
    let mut commands = Vec::new();
    if !blacklisted.is_empty() {
        for f in blacklisted {
            commands.push(format!(
                "# Inspect & remove the blacklist line: sudo $EDITOR {f}"
            ));
        }
        commands.push("sudo update-initramfs -u   # Debian/Ubuntu".to_owned());
        commands.push("sudo dracut -f             # Fedora/RHEL".to_owned());
    }
    commands.push("sudo modprobe amdgpu".to_owned());
    if e.secure_boot == "enabled" {
        commands.push("# Secure Boot is on; if amdgpu still won't load, the DKMS module isn't signed. Sign it (mokutil) or disable Secure Boot.".to_owned());
    }
    let fix = Fix {
        summary: "Remove amdgpu from any modprobe blacklist and load it.".to_owned(),
        commands,
        needs_sudo: true,
        needs_reboot: !blacklisted.is_empty(),
        fix_id: "fix-5-amdgpu-load".to_owned(),
        auto_applicable: false,
        verify: "lsmod | grep amdgpu && rocminfo | head -n 5".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-5-amdgpu-load",
        "amdgpu kernel module not loaded (or blacklisted)",
        score,
        evidence,
        fix,
    )
}

fn check_6_path_missing(e: &Examination, symptom: &str) -> Diagnosis {
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
        Fix {
            summary: format!("Add {bin_dir} to your User PATH and reopen the shell."),
            commands: vec![
                format!("setx PATH \"%PATH%;{bin_dir}\""),
                "# Or: System Properties -> Environment Variables -> Path -> Edit -> New."
                    .to_owned(),
                "# `setx` only affects NEW shells; close and reopen this terminal afterwards."
                    .to_owned(),
            ],
            fix_id: "fix-6-path".to_owned(),
            auto_applicable: true,
            verify: format!(
                "powershell -NoProfile -Command \"& \\\"{bin_dir}\\hipInfo.exe\\\" | Select-Object -First 5\""
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
            auto_applicable: true,
            verify: "rocminfo | head -n 5 && hipcc --version".to_owned(),
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

fn check_7_stale_repos(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let repos = &e.rocm_repos_seen;
    if repos.len() >= 2 {
        score += 40;
        evidence.push(format!(
            "{} ROCm/AMDGPU repo files present: {repos:?}",
            repos.len()
        ));
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_REPO_BROKEN);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-7-stale-repos", "Stale ROCm repos");
    }
    let mut commands =
        vec!["ls /etc/apt/sources.list.d/ | grep -iE 'rocm|amdgpu|radeon' || true".to_owned()];
    for r in repos {
        commands.push(format!(
            "# sudo mv {r} {r}.bak     # quarantine, do not delete yet"
        ));
    }
    commands.push("sudo apt update".to_owned());
    commands.push("# If apt now resolves, reinstall via the correct method only:".to_owned());
    commands.push(
        "#   amdgpu-install --usecase=rocm,hip --no-dkms   # if you want amdgpu-install".to_owned(),
    );
    commands.push("#   or use the distro packages exclusively".to_owned());
    let fix = Fix {
        summary: "Quarantine duplicate ROCm/AMDGPU repo files and resolve apt before re-running any installer.".to_owned(),
        commands,
        needs_sudo: true,
        fix_id: "fix-7-stale-repos".to_owned(),
        auto_applicable: false,
        verify: "sudo apt update 2>&1 | tail -n 20".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-7-stale-repos",
        "Stale or conflicting APT/DNF repos from prior installer runs",
        score,
        evidence,
        fix,
    )
}

fn check_8_wheel_rocm_mismatch(e: &Examination, symptom: &str) -> Diagnosis {
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
    if let (Some(fw), Some(sys)) = (&fw_major, &sys_major)
        && fw != sys
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
            auto_applicable: false,
            verify: "python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\"".to_owned(),
            ..Fix::default()
        }
    } else {
        Fix {
            summary: "Reinstall the framework from the wheel index that matches the system ROCm major (or upgrade the system ROCm to match the wheel).".to_owned(),
            commands: vec![
                "pip uninstall -y torch torchvision torchaudio".to_owned(),
                "# Pick the index that matches your system ROCm major. Examples:".to_owned(),
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4".to_owned(),
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3".to_owned(),
                "# Then re-check:".to_owned(),
                "python -c \"import torch; print(torch.__version__, torch.version.hip)\"".to_owned(),
            ],
            fix_id: "fix-8-wheel-rocm".to_owned(),
            auto_applicable: false,
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

fn check_9_igpu_dgpu_collision(e: &Examination, symptom: &str) -> Diagnosis {
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
            auto_applicable: true,
            verify: "powershell -NoProfile -Command \"$env:HIP_VISIBLE_DEVICES=1; python -c \\\"import torch; print(torch.cuda.device_count())\\\"\"".to_owned(),
            notes: vec![format!("Detected gfx targets: {gfx_targets:?}. The dGPU is usually the higher-numbered family (gfx11xx).")],
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
            auto_applicable: false,
            verify: "HIP_VISIBLE_DEVICES=1 python -c \"import torch; print(torch.cuda.device_count())\"".to_owned(),
            notes: vec![format!("Detected gfx targets: {gfx_targets:?}. The dGPU is usually the higher-numbered family (gfx11xx).")],
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

fn check_10_container_devices(e: &Examination, symptom: &str) -> Diagnosis {
    if !e.in_container {
        return zero("fix-10-container", "Container missing devices");
    }
    let kind = if e.container_kind.is_empty() {
        "container".to_owned()
    } else {
        e.container_kind.clone()
    };
    let mut score = 25;
    let mut evidence = vec![format!("running inside a {kind}")];
    if let Some(kfd) = &e.kfd {
        if !kfd.exists {
            score += 40;
            evidence.push("/dev/kfd is not present in the container".to_owned());
        } else if kfd.user_can_write == Some(false) {
            score += 30;
            evidence.push("/dev/kfd is present but not writable by the container user".to_owned());
        }
    } else {
        score += 40;
        evidence.push("/dev/kfd is not present in the container".to_owned());
    }
    if e.render_devices.is_empty() {
        score += 20;
        evidence.push("no /dev/dri/renderD* visible in the container".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_CONTAINER);
    score += kw_score;
    evidence.extend(kw_ev);

    let fix = Fix {
        summary: "Re-launch the container with the AMD devices and the render group passed through.".to_owned(),
        commands: vec![
            "# Docker / Podman flags AMD-recommends:".to_owned(),
            "docker run --rm -it \\".to_owned(),
            "  --device=/dev/kfd \\".to_owned(),
            "  --device=/dev/dri \\".to_owned(),
            "  --group-add render \\".to_owned(),
            "  --security-opt seccomp=unconfined \\".to_owned(),
            "  --shm-size=8g \\".to_owned(),
            "  rocm/pytorch:latest".to_owned(),
            "# Rootless podman: also pass `--userns=keep-id` and ensure the".to_owned(),
            "# host user is in the render group; podman maps it through.".to_owned(),
        ],
        fix_id: "fix-10-container".to_owned(),
        auto_applicable: false,
        verify: "rocminfo | head -n 5".to_owned(),
        notes: vec!["Use rocm/pytorch or rocm/dev-ubuntu-22.04 as a known-good image. Mixing host ROCm + container ROCm versions is a separate footgun.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-10-container",
        "Container can't see /dev/kfd or /dev/dri/renderD*",
        score,
        evidence,
        fix,
    )
}

fn check_11_iommu_hang(e: &Examination, symptom: &str) -> Diagnosis {
    if amd_gpu_count(e) < 2 {
        return zero("fix-11-iommu", "Multi-GPU IOMMU hang");
    }
    let mut score = 0;
    let mut evidence = vec![format!("{} AMD GPUs detected", amd_gpu_count(e))];
    let iommu = &e.iommu_kernel_param;
    if !iommu.is_empty() && iommu != "pt" {
        score += 25;
        evidence.push(format!("kernel cmdline has iommu={iommu} (not 'pt')"));
    }
    if iommu.is_empty() {
        score += 10;
        evidence.push("no iommu= flag on kernel cmdline (default may be 'on')".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_IOMMU_HANG);
    score += kw_score;
    evidence.extend(kw_ev);

    if score < 25 {
        return zero("fix-11-iommu", "Multi-GPU IOMMU hang");
    }
    let fix = Fix {
        summary: "Add `iommu=pt` to the kernel command line so DMA goes through pass-through mode. This requires editing GRUB and rebooting.".to_owned(),
        commands: vec![
            "# Inspect the current cmdline:".to_owned(),
            "cat /proc/cmdline".to_owned(),
            "# Edit /etc/default/grub and add iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT:".to_owned(),
            "sudo $EDITOR /etc/default/grub".to_owned(),
            "sudo update-grub                # Debian/Ubuntu".to_owned(),
            "sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora/RHEL".to_owned(),
            "# Reboot for the change to take effect, then retry the multi-GPU job.".to_owned(),
        ],
        needs_sudo: true,
        needs_reboot: true,
        fix_id: "fix-11-iommu".to_owned(),
        auto_applicable: false,
        verify: "cat /proc/cmdline | grep -o 'iommu=\\w*'".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-11-iommu",
        "Multi-GPU hang on systems with IOMMU enabled",
        score,
        evidence,
        fix,
    )
}

fn check_12_amdgpu_install_broken(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let method = &e.rocm_install_method;
    if method == "amdgpu-install" {
        evidence.push("ROCm was installed via amdgpu-install".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_DPKG_BROKEN);
    score += kw_score;
    evidence.extend(kw_ev);
    if method == "amdgpu-install" && kw_score > 0 {
        score += 20;
    }

    if score <= 0 {
        return zero("fix-12-installer", "amdgpu-install broken state");
    }
    let fix = Fix {
        summary: "Run amdgpu-install's documented uninstall sequence to clear the half-configured state, THEN reinstall without the flag that broke it.".to_owned(),
        commands: vec![
            "sudo amdgpu-install --uninstall".to_owned(),
            "sudo apt autoremove --purge -y".to_owned(),
            "sudo apt update".to_owned(),
            "# Reinstall. Drop --accept-eula if you used it previously; the".to_owned(),
            "# newer installer rejects it and leaves a half-configured repo.".to_owned(),
            "sudo amdgpu-install --usecase=rocm,hip".to_owned(),
        ],
        needs_sudo: true,
        needs_reboot: true,
        fix_id: "fix-12-installer".to_owned(),
        auto_applicable: false,
        verify: "dpkg -l | grep -E 'rocm|amdgpu' | head -n 20 && rocminfo | head -n 5".to_owned(),
        notes: vec!["If `apt autoremove` warns it will remove unrelated packages, stop and resolve those by hand before continuing.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-12-installer",
        "amdgpu-install left a broken state (repo regression / partial DKMS)",
        score,
        evidence,
        fix,
    )
}

fn check_13_hip_sdk_missing(e: &Examination, symptom: &str) -> Diagnosis {
    if e.os_family != "windows" {
        return zero("fix-13-hip-sdk-missing", "HIP SDK not installed");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    let sdk_path = &e.hip_sdk_path;
    if sdk_path.is_empty() {
        score += 35;
        evidence.push("No HIP SDK install found under C:\\Program Files\\AMD\\ROCm".to_owned());
    } else if !e.hipinfo_present {
        score += 30;
        evidence.push(format!(
            "HIP SDK at {sdk_path} but hipInfo.exe is missing from its bin directory"
        ));
    }
    if e.has_amd_gpu && e.framework == "pytorch" && e.framework_rocm_version.starts_with("hip=") {
        score += 25;
        evidence
            .push("PyTorch is a HIP build but the HIP SDK is not present on this host".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_HIP_SDK_MISSING);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-13-hip-sdk-missing", "HIP SDK not installed");
    }
    let fix = Fix {
        summary: "Install the AMD HIP SDK for Windows; the HIP runtime DLLs and hipInfo.exe come from there.".to_owned(),
        commands: vec![
            "# Download and install the HIP SDK (matched to your framework's HIP major):".to_owned(),
            "#   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html".to_owned(),
            "# After install, reopen the shell so HIP_PATH and PATH pick up the new install.".to_owned(),
        ],
        fix_id: "fix-13-hip-sdk-missing".to_owned(),
        auto_applicable: false,
        verify: "powershell -NoProfile -Command \"& \\\"$env:HIP_PATH\\bin\\hipInfo.exe\\\" | Select-Object -First 5\"".to_owned(),
        notes: vec!["If you only need PyTorch on Windows AMD and don't need the C/C++ HIP toolchain, the TheRock wheels bundle their own HIP runtime and may not require a system HIP SDK install.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-13-hip-sdk-missing",
        "HIP SDK not installed (Windows)",
        score,
        evidence,
        fix,
    )
}

fn check_14_adrenalin_too_old(e: &Examination, symptom: &str) -> Diagnosis {
    if e.os_family != "windows" {
        return zero("fix-14-adrenalin-too-old", "Adrenalin driver too old");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    let sdk_path = &e.hip_sdk_path;
    if !sdk_path.is_empty() && e.hipinfo_present && !matches!(e.hipinfo_status.as_str(), "ok" | "")
    {
        score += 35;
        evidence.push(format!(
            "HIP SDK at {sdk_path} is installed but hipInfo.exe reports {:?}; this typically means the kernel-mode driver doesn't match the SDK.",
            e.hipinfo_status
        ));
    }
    if !e.adrenalin_version.is_empty() {
        evidence.push(format!(
            "Adrenalin / kernel-mode driver version: {}",
            e.adrenalin_version
        ));
    }
    if symptom_matches(symptom, r"driver.*(too old|out of date|unsupported)") {
        score += 35;
        evidence.push("error mentions 'driver too old / out of date / unsupported'".to_owned());
    }
    if symptom_matches(symptom, r"hsa.*invalid agent|no agents (were )?found") {
        score += 25;
        evidence.push("HSA error suggests driver/runtime can't enumerate the GPU".to_owned());
    }

    if score <= 0 {
        return zero("fix-14-adrenalin-too-old", "Adrenalin driver too old");
    }
    let fix = Fix {
        summary: "Update the AMD Adrenalin (or PRO) graphics driver to the version the HIP SDK release notes call out as the supported pairing.".to_owned(),
        commands: vec![
            "# Cross-check the HIP SDK release notes for the exact driver pairing:".to_owned(),
            "#   https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html".to_owned(),
            "# Then download the matching driver from:".to_owned(),
            "#   https://www.amd.com/en/support".to_owned(),
            "# Reboot after the install for the kernel-mode driver to take effect.".to_owned(),
        ],
        needs_reboot: true,
        fix_id: "fix-14-adrenalin-too-old".to_owned(),
        auto_applicable: false,
        verify: "powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | Select-Object -First 1).DriverVersion\"".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-14-adrenalin-too-old",
        "Adrenalin / kernel-mode driver too old for the installed HIP SDK",
        score,
        evidence,
        fix,
    )
}

fn check_15_msvc_redist(e: &Examination, symptom: &str) -> Diagnosis {
    if e.os_family != "windows" {
        return zero("fix-15-msvc-redist", "MSVC runtime missing");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    if e.msvc_redist_present == Some(false) {
        score += 45;
        evidence.push("vcruntime140.dll / vcruntime140_1.dll not resolvable on PATH".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_MSVC_REDIST);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-15-msvc-redist", "MSVC runtime missing");
    }
    let fix = Fix {
        summary: "Install the Microsoft Visual C++ 2015-2022 redistributable so the HIP SDK's amdhip64_*.dll can load.".to_owned(),
        commands: vec![
            "# Download & install (x64):".to_owned(),
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe".to_owned(),
            "# After the install, reopen the shell and re-run your import / hipInfo check.".to_owned(),
        ],
        fix_id: "fix-15-msvc-redist".to_owned(),
        auto_applicable: false,
        verify: "where vcruntime140.dll && where vcruntime140_1.dll".to_owned(),
        notes: vec!["If installing the redistributable still leaves a missing-DLL error, the failing DLL is probably amdhip64_X.dll itself; that points at fix-13-hip-sdk-missing (the HIP SDK install) rather than this fix.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-15-msvc-redist",
        "MSVC runtime missing (HIP DLLs cannot load)",
        score,
        evidence,
        fix,
    )
}

/// A checker plus the OS families it applies to.
type Checker = (fn(&Examination, &str) -> Diagnosis, &'static [&'static str]);

const CHECKERS: &[Checker] = &[
    (check_1_arch_not_in_wheel, &["linux", "windows"]),
    (check_2_hsa_override_unneeded, &["linux", "windows"]),
    (check_3_rocm_kernel_unsupported, &["linux"]),
    (check_4_render_group, &["linux"]),
    (check_5_amdgpu_blacklisted, &["linux"]),
    (check_6_path_missing, &["linux", "windows"]),
    (check_7_stale_repos, &["linux"]),
    (check_8_wheel_rocm_mismatch, &["linux", "windows"]),
    (check_9_igpu_dgpu_collision, &["linux", "windows"]),
    (check_10_container_devices, &["linux"]),
    (check_11_iommu_hang, &["linux"]),
    (check_12_amdgpu_install_broken, &["linux"]),
    (check_13_hip_sdk_missing, &["windows"]),
    (check_14_adrenalin_too_old, &["windows"]),
    (check_15_msvc_redist, &["windows"]),
];

/// Run every applicable checker, drop zero-score results, sort by score
/// descending (stable, so ties keep catalog order).
fn run_all_checks(e: &Examination, symptom: &str) -> Vec<Diagnosis> {
    let os_family = if e.os_family.is_empty() {
        "linux"
    } else {
        e.os_family.as_str()
    };
    let mut results: Vec<Diagnosis> = CHECKERS
        .iter()
        .filter(|(_, applicable)| applicable.contains(&os_family))
        .map(|(check, _)| check(e, symptom))
        .filter(|d| d.score > 0)
        .collect();
    // Stable sort by score descending: ties keep catalog order.
    results.sort_by_key(|d| std::cmp::Reverse(d.score));
    results
}

fn route_when_no_match(e: &Examination) -> Route {
    let target = match e.framework.as_str() {
        "pytorch" => "pytorch",
        "llama-cpp" => "llama-cpp",
        "lemonade" => "lemonade",
        "ollama" => "ollama",
        "lm-studio" => "lm-studio",
        _ => "rocm-core",
    };
    Route {
        target: target.to_owned(),
        url: upstream_tracker(target).to_owned(),
    }
}

/// Diagnose an examination against the closed catalog.
#[must_use]
pub fn diagnose(e: &Examination, symptom: &str) -> DiagnoseReport {
    DiagnoseReport {
        matched: run_all_checks(e, symptom),
        min_score_for_match: MIN_SCORE_FOR_MATCH,
        high_confidence_threshold: HIGH_CONFIDENCE,
        route_when_no_match: route_when_no_match(e),
    }
}

/// Render the human-facing diagnosis view (mirrors `diagnose.py`'s text output).
#[must_use]
pub fn render_report_text(report: &DiagnoseReport, top: usize) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if report.matched.is_empty() {
        let route = &report.route_when_no_match;
        out.push_str("rocm examine: no known misconfiguration matched.\n\n");
        out.push_str("This is the explicit 'I don't recognise this failure mode' case. Do not speculate; file the symptom + this examination output upstream:\n");
        let _ = writeln!(out, "  {:>12}: {}", route.target, route.url);
        out.push('\n');
        out.push_str("Include the JSON from `rocm examine --json` in your report.\n");
        return out;
    }
    for (i, d) in report.matched.iter().take(top).enumerate() {
        let tier = if d.score >= HIGH_CONFIDENCE {
            "HIGH"
        } else if d.score >= MIN_SCORE_FOR_MATCH {
            "LIKELY"
        } else {
            "WEAK"
        };
        let _ = writeln!(out, "#{} [{tier} score={}/100] {}", i + 1, d.score, d.title);
        let _ = writeln!(out, "   id: {}", d.id);
        for ev in &d.evidence {
            let _ = writeln!(out, "   - {ev}");
        }
        if let Some(fix) = &d.fix {
            let _ = writeln!(out, "   plan: {}", fix.summary);
            for c in &fix.commands {
                let _ = writeln!(out, "     $ {c}");
            }
            let mut flags = Vec::new();
            if fix.needs_sudo {
                flags.push("sudo");
            }
            if fix.needs_reboot {
                flags.push("reboot required");
            }
            if fix.needs_relogin {
                flags.push("re-login required");
            }
            if fix.auto_applicable {
                flags.push("rocm examine --fix can run it");
            }
            if !flags.is_empty() {
                let _ = writeln!(out, "   flags: {}", flags.join(", "));
            }
            for n in &fix.notes {
                let _ = writeln!(out, "   note: {n}");
            }
            if !fix.verify.is_empty() {
                let _ = writeln!(out, "   verify after fix: {}", fix.verify);
            }
        }
        out.push('\n');
    }
    if let Some(high) = report.matched.iter().find(|d| d.score >= HIGH_CONFIDENCE) {
        let _ = writeln!(out, "Next step: propose `rocm examine --fix {}`.", high.id);
    } else {
        out.push_str("Highest-scoring match is below the HIGH_CONFIDENCE threshold. Confirm one more piece of evidence before applying.\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examine::{Device, Examination, Gpu};

    fn linux_base() -> Examination {
        Examination {
            os_family: "linux".to_owned(),
            ..Examination::default()
        }
    }

    #[test]
    fn render_group_missing_is_diagnosed() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-4-render-group");
        assert_eq!(top.score, 45); // 35 render + 10 video
        assert!(top.fix.as_ref().unwrap().auto_applicable);
    }

    #[test]
    fn render_group_with_symptom_is_high_confidence() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "RuntimeError: unable to open /dev/kfd");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-4-render-group");
        assert!(top.score >= HIGH_CONFIDENCE, "score was {}", top.score);
        assert!(report.has_match());
    }

    #[test]
    fn kfd_not_writable_adds_score() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.kfd = Some(Device {
            path: "/dev/kfd".to_owned(),
            exists: true,
            mode: "crw-rw----".to_owned(),
            owner_group: "render".to_owned(),
            user_can_write: Some(false),
            ..Device::default()
        });
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-4-render-group");
        assert_eq!(top.score, 60); // 35 render + 25 kfd
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
    fn no_match_routes_upstream() {
        let mut e = linux_base();
        e.framework = "pytorch".to_owned();
        let report = diagnose(&e, "");
        assert!(report.matched.is_empty());
        assert!(!report.has_match());
        assert_eq!(report.route_when_no_match.target, "pytorch");
        assert!(report.route_when_no_match.url.contains("pytorch/pytorch"));
    }

    #[test]
    fn no_match_default_route_is_rocm_core() {
        let e = linux_base();
        let report = diagnose(&e, "");
        assert_eq!(report.route_when_no_match.target, "rocm-core");
    }

    #[test]
    fn windows_only_checks_skipped_on_linux() {
        let e = linux_base();
        let report = diagnose(&e, "vcruntime140.dll is missing");
        // fix-15 is windows-only; must not appear on a linux exam.
        assert!(report.matched.iter().all(|d| d.id != "fix-15-msvc-redist"));
    }

    #[test]
    fn msvc_redist_diagnosed_on_windows() {
        let mut e = Examination {
            os_family: "windows".to_owned(),
            ..Examination::default()
        };
        e.msvc_redist_present = Some(false);
        let report = diagnose(
            &e,
            "The program can't start because vcruntime140.dll is missing",
        );
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-15-msvc-redist");
        assert!(top.score >= MIN_SCORE_FOR_MATCH);
    }

    #[test]
    fn iommu_requires_two_gpus_and_min_score() {
        // Single GPU: never fires.
        let mut e = linux_base();
        e.gpus = vec![Gpu {
            is_amd: true,
            ..Gpu::default()
        }];
        e.iommu_kernel_param = "on".to_owned();
        assert!(
            diagnose(&e, "hang")
                .matched
                .iter()
                .all(|d| d.id != "fix-11-iommu")
        );
        // Two GPUs + iommu=on (25) clears the per-rule >=25 gate.
        e.gpus = vec![
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
        ];
        let report = diagnose(&e, "");
        assert!(report.matched.iter().any(|d| d.id == "fix-11-iommu"));
    }

    #[test]
    fn keyword_score_takes_top_two() {
        // INVALID_ISA: two hits (50 + 40) -> 90, not the sum of all.
        let (score, labels) = keyword_score(
            "HSA_STATUS_ERROR_INVALID_ISA and invalid device function and no kernel image is available",
            KEYWORDS_INVALID_ISA,
        );
        assert_eq!(score, 90);
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn report_serializes_expected_shape() {
        let report = diagnose(&linux_base(), "");
        let v = serde_json::to_value(&report).unwrap();
        for key in [
            "matched",
            "min_score_for_match",
            "high_confidence_threshold",
            "route_when_no_match",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["min_score_for_match"], 50);
        assert_eq!(v["high_confidence_threshold"], 75);
    }
}
