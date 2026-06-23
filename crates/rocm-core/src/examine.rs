//! Host examination probe.
//!
//! Rust port of the `rocm-doctor` skill's `examine.py`. It gathers the host
//! signals the diagnosis catalog reasons over and serializes them as the
//! **Examination** JSON document (`rocm examine --json`). The field names and
//! shapes mirror `examine.py` field-for-field so the catalog consumes the CLI's
//! output unchanged. See `plans/rocm-doctor-examine-migration-plan.md`.

use crate::{runtime_is_linux, runtime_is_windows};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Environment variables that commonly steer (or break) ROCm/HIP runtime
/// behavior. Captured verbatim into `Examination::env`.
const TRACKED_ENV_VARS: &[&str] = &[
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HIP_PLATFORM",
    "PYTORCH_ROCM_ARCH",
    "HCC_AMDGPU_TARGET",
    "AMDGPU_TARGETS",
    "LD_LIBRARY_PATH",
    "PATH",
];

/// Repo files dropped by the `amdgpu-install` pipeline; their presence marks an
/// amdgpu-install-managed ROCm.
const AMDGPU_INSTALL_MARKERS: &[&str] = &[
    "/etc/apt/sources.list.d/amdgpu.list",
    "/etc/apt/sources.list.d/rocm.list",
    "/etc/apt/sources.list.d/radeon.list",
    "/etc/yum.repos.d/amdgpu.repo",
    "/etc/yum.repos.d/rocm.repo",
];

/// Marketing-name fragments that identify an AMD APU when `rocminfo` is absent.
const APU_KEYWORDS: &[&str] = &[
    "strix halo",
    "ryzen ai max",
    "phoenix",
    "hawk point",
    "strix point",
    "krackan",
    "rembrandt",
    "raphael",
    "barcelo",
    "lucienne",
    "renoir",
    "cezanne",
];

/// A single GPU as enumerated by `lspci`/`rocminfo` (Linux) or the display
/// inventory (Windows).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gpu {
    pub name: String,
    pub gfx_target: String,
    pub pci_id: String,
    pub is_apu: Option<bool>,
    pub is_amd: bool,
}

/// Stat of a device node such as `/dev/kfd` or `/dev/dri/renderD*`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Device {
    pub path: String,
    pub exists: bool,
    pub mode: String,
    pub owner_user: String,
    pub owner_group: String,
    pub user_can_read: Option<bool>,
    pub user_can_write: Option<bool>,
}

/// Structured machine state consumed by the diagnosis catalog. Field order and
/// names mirror `examine.py`'s `Examination` dataclass so the JSON contract is
/// identical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Examination {
    // platform
    pub os_family: String,
    pub os_version: String,
    pub distro_id: String,
    pub distro_version: String,
    pub kernel_release: String,
    pub kernel_cmdline: String,
    pub is_wsl: bool,

    // hardware
    pub cpu_vendor: String,
    pub cpu_model: String,
    pub gpus: Vec<Gpu>,
    pub has_amd_gpu: bool,
    pub has_nvidia_gpu: bool,
    pub has_apu: bool,
    pub has_discrete_amd: bool,

    // driver / runtime (Linux)
    pub amdgpu_loaded: Option<bool>,
    pub amdgpu_blacklisted_in: Vec<String>,
    pub amdkfd_loaded: Option<bool>,
    pub secure_boot: String,
    pub iommu_kernel_param: String,
    pub kfd: Option<Device>,
    pub render_devices: Vec<Device>,

    // user / groups (Linux)
    pub user_name: String,
    pub user_groups: Vec<String>,
    pub in_render_group: Option<bool>,
    pub in_video_group: Option<bool>,

    // ROCm install (Linux)
    pub rocm_version: String,
    pub rocm_install_method: String,
    pub rocm_path: String,
    pub rocminfo_present: bool,
    pub rocminfo_status: String,
    pub hip_libs_on_ld_path: Option<bool>,
    pub rocm_repos_seen: Vec<String>,

    // HIP SDK install (Windows)
    pub hip_sdk_path: String,
    pub hip_sdk_version: String,
    pub hipinfo_present: bool,
    pub hipinfo_status: String,
    pub adrenalin_version: String,
    pub msvc_redist_present: Option<bool>,

    // framework
    pub framework: String,
    pub framework_version: String,
    pub framework_rocm_version: String,
    pub framework_arch_list: Vec<String>,
    pub framework_notes: Vec<String>,

    // environment
    pub env: BTreeMap<String, String>,

    // container
    pub in_container: bool,
    pub container_kind: String,

    // evidence
    pub dmesg_amdgpu_tail: Vec<String>,
    pub notes: Vec<String>,
    pub probe_failures: Vec<String>,
}

impl Default for Examination {
    fn default() -> Self {
        Self {
            os_family: "unknown".to_owned(),
            os_version: String::new(),
            distro_id: String::new(),
            distro_version: String::new(),
            kernel_release: String::new(),
            kernel_cmdline: String::new(),
            is_wsl: false,
            cpu_vendor: "unknown".to_owned(),
            cpu_model: String::new(),
            gpus: Vec::new(),
            has_amd_gpu: false,
            has_nvidia_gpu: false,
            has_apu: false,
            has_discrete_amd: false,
            amdgpu_loaded: None,
            amdgpu_blacklisted_in: Vec::new(),
            amdkfd_loaded: None,
            secure_boot: "unknown".to_owned(),
            iommu_kernel_param: String::new(),
            kfd: None,
            render_devices: Vec::new(),
            user_name: String::new(),
            user_groups: Vec::new(),
            in_render_group: None,
            in_video_group: None,
            rocm_version: String::new(),
            rocm_install_method: String::new(),
            rocm_path: String::new(),
            rocminfo_present: false,
            rocminfo_status: String::new(),
            hip_libs_on_ld_path: None,
            rocm_repos_seen: Vec::new(),
            hip_sdk_path: String::new(),
            hip_sdk_version: String::new(),
            hipinfo_present: false,
            hipinfo_status: String::new(),
            adrenalin_version: String::new(),
            msvc_redist_present: None,
            framework: "unknown".to_owned(),
            framework_version: String::new(),
            framework_rocm_version: String::new(),
            framework_arch_list: Vec::new(),
            framework_notes: Vec::new(),
            env: BTreeMap::new(),
            in_container: false,
            container_kind: String::new(),
            dmesg_amdgpu_tail: Vec::new(),
            notes: Vec::new(),
            probe_failures: Vec::new(),
        }
    }
}

/// Which framework probe to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameworkProbe {
    Auto,
    PyTorch,
    LlamaCpp,
    Skip,
}

impl Examination {
    /// Probe the host and return the examination. Never fails; probe errors are
    /// recorded in `probe_failures`/`notes` and the relevant fields are left at
    /// their defaults (matching `examine.py`'s degrade-gracefully behavior).
    #[must_use]
    pub fn probe(framework: FrameworkProbe) -> Self {
        let mut e = Self::default();
        probe_os(&mut e);
        if e.os_family == "linux" {
            probe_cpu_linux(&mut e);
            probe_gpus_lspci(&mut e);
            probe_gpus_rocminfo(&mut e);
            summarise_gpu_categories(&mut e);
            probe_modules(&mut e);
            probe_user(&mut e);
            probe_devices(&mut e);
            probe_secure_boot(&mut e);
            probe_rocm_install(&mut e);
            probe_env(&mut e);
            probe_container(&mut e);
            probe_dmesg_amdgpu(&mut e);
            probe_framework(&mut e, framework);
        } else if e.os_family == "windows" {
            probe_cpu_windows(&mut e);
            probe_gpus_windows(&mut e);
            probe_hip_sdk_windows(&mut e);
            probe_adrenalin_windows(&mut e);
            probe_msvc_redist_windows(&mut e);
            summarise_gpu_categories(&mut e);
            probe_env(&mut e);
            probe_framework(&mut e, framework);
        } else {
            e.notes.push(format!(
                "rocm examine supports Linux and Windows; got {}. This skill cannot help on this platform.",
                e.os_family
            ));
        }
        e
    }

    /// Exit code mirroring `examine.py`: `2` = wrong platform / WSL / no AMD GPU
    /// (skill can't help), `3` = a key probe failed (soft warning), `0` = ok.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        if self.is_wsl || !matches!(self.os_family.as_str(), "linux" | "windows") {
            return 2;
        }
        if !self.has_amd_gpu {
            return 2;
        }
        if self.os_family == "linux" {
            if !self.probe_failures.is_empty() && !self.rocminfo_present && self.gpus.is_empty() {
                return 3;
            }
        } else if !self.probe_failures.is_empty() && !self.hipinfo_present && self.gpus.is_empty() {
            return 3;
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Shell / fs helpers (never panic)
// ---------------------------------------------------------------------------

/// Run a command with a timeout. Returns `(rc, stdout, stderr)`. `rc` is `127`
/// when the program can't be spawned and `124` on timeout.
pub(crate) fn run(program: &str, args: &[&str], timeout: Duration) -> (i32, String, String) {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return (127, String::new(), String::new());
    };
    let stdout_handle = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut buf = String::new();
            let _ = stdout.read_to_string(&mut buf);
            buf
        })
    });
    let stderr_handle = child.stderr.take().map(|mut stderr| {
        thread::spawn(move || {
            let mut buf = String::new();
            let _ = stderr.read_to_string(&mut buf);
            buf
        })
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };
    let stdout = stdout_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr = stderr_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let rc = match status {
        Some(status) => status.code().unwrap_or(-1),
        None => 124,
    };
    (rc, stdout, stderr)
}

fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Whether `program` resolves on `PATH` (best-effort, no execution).
pub(crate) fn which(program: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    let (sep, exts): (char, &[&str]) = if runtime_is_windows() {
        (';', &[".exe", ".bat", ".cmd", ""])
    } else {
        (':', &[""])
    };
    for dir in path.split(sep) {
        if dir.is_empty() {
            continue;
        }
        for ext in exts {
            if Path::new(dir).join(format!("{program}{ext}")).is_file() {
                return true;
            }
        }
    }
    false
}

const SHORT: Duration = Duration::from_secs(5);
const MEDIUM: Duration = Duration::from_secs(8);

// ---------------------------------------------------------------------------
// Platform probes
// ---------------------------------------------------------------------------

fn probe_os(e: &mut Examination) {
    e.os_version = std::env::consts::OS.to_owned();
    if runtime_is_linux() {
        e.os_family = "linux".to_owned();
        e.kernel_release = run("uname", &["-r"], SHORT).1.trim().to_owned();
        e.kernel_cmdline = read_text("/proc/cmdline").trim().to_owned();
        let osr = read_text("/etc/os-release");
        for line in osr.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let value = value.trim().trim_matches('"');
            match key {
                "ID" => e.distro_id = value.to_owned(),
                "VERSION_ID" => e.distro_version = value.to_owned(),
                _ => {}
            }
        }
        if let Some(param) = parse_iommu_param(&e.kernel_cmdline) {
            e.iommu_kernel_param = param;
        }
        let proc_version = read_text("/proc/version").to_lowercase();
        if proc_version.contains("microsoft")
            || proc_version.contains("wsl")
            || std::env::var_os("WSL_DISTRO_NAME").is_some()
        {
            e.is_wsl = true;
        }
    } else if runtime_is_windows() {
        e.os_family = "windows".to_owned();
    } else {
        e.os_family = "other".to_owned();
    }
}

/// Extract the value of `iommu=<value>` from a kernel cmdline string.
fn parse_iommu_param(cmdline: &str) -> Option<String> {
    cmdline.split_whitespace().find_map(|token| {
        token
            .strip_prefix("iommu=")
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

fn probe_cpu_linux(e: &mut Examination) {
    let txt = read_text("/proc/cpuinfo");
    for line in txt.lines() {
        if (e.cpu_vendor == "unknown")
            && line.starts_with("vendor_id")
            && let Some((_, value)) = line.split_once(':')
        {
            let value = value.trim();
            e.cpu_vendor = if value.contains("AMD") {
                "amd".to_owned()
            } else if value.contains("Intel") {
                "intel".to_owned()
            } else {
                value.to_lowercase()
            };
        }
        if e.cpu_model.is_empty()
            && line.starts_with("model name")
            && let Some((_, value)) = line.split_once(':')
        {
            e.cpu_model = value.trim().to_owned();
        }
        if e.cpu_vendor != "unknown" && !e.cpu_model.is_empty() {
            break;
        }
    }
}

fn probe_cpu_windows(e: &mut Examination) {
    let (rc, out, _) = run(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
        ],
        MEDIUM,
    );
    if rc == 0 && !out.trim().is_empty() {
        e.cpu_model = out
            .trim()
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();
        let lname = e.cpu_model.to_lowercase();
        e.cpu_vendor = if lname.contains("amd") {
            "amd".to_owned()
        } else if lname.contains("intel") {
            "intel".to_owned()
        } else {
            "unknown".to_owned()
        };
    } else {
        e.probe_failures
            .push("Get-CimInstance Win32_Processor failed; cannot identify CPU.".to_owned());
    }
}

// ---------------------------------------------------------------------------
// GPU probes
// ---------------------------------------------------------------------------

/// Best-effort `(gfx_target, is_apu)` for an AMD marketing name.
fn classify_amd_marketing_name(name: &str) -> (String, bool) {
    let mut n = name.to_lowercase();
    for deco in ["(tm)", "(r)", "(c)", "(\u{2122})"] {
        n = n.replace(deco, " ");
    }
    let n = n.split_whitespace().collect::<Vec<_>>().join(" ");
    let contains = |needle: &str| n.contains(needle);
    if contains("ryzen ai max") || contains("strix halo") {
        return ("gfx1151".to_owned(), true);
    }
    if contains("radeon 8050s") || contains("radeon 8060s") || contains("radeon 8045s") {
        return ("gfx1151".to_owned(), true);
    }
    if contains("radeon 880m")
        || contains("radeon 890m")
        || contains("strix point")
        || contains("krackan")
    {
        return ("gfx1150".to_owned(), true);
    }
    if contains("radeon 780m")
        || contains("radeon 760m")
        || contains("radeon 740m")
        || contains("phoenix")
        || contains("hawk point")
    {
        return ("gfx1103".to_owned(), true);
    }
    (String::new(), APU_KEYWORDS.iter().any(|kw| n.contains(kw)))
}

/// Whether a gfx target belongs to an APU family the doctor cares about
/// (gfx115x / gfx110x / gfx103x).
fn gfx_is_apu_family(gfx: &str) -> bool {
    let g = gfx.to_lowercase();
    (g.starts_with("gfx110") || g.starts_with("gfx115"))
        && g.len() >= 7
        && g.as_bytes()[6].is_ascii_digit()
}

fn probe_gpus_lspci(e: &mut Examination) {
    if !which("lspci") {
        e.probe_failures
            .push("lspci not found; cannot enumerate PCI GPUs".to_owned());
        return;
    }
    let (rc, out, _) = run("lspci", &["-nn", "-D"], MEDIUM);
    if rc != 0 {
        e.probe_failures
            .push("lspci returned non-zero; PCI enumeration incomplete".to_owned());
        return;
    }
    for line in out.lines() {
        let is_controller = line.contains("VGA compatible controller")
            || line.contains("3D controller")
            || line.contains("Display controller");
        if !is_controller {
            continue;
        }
        let pci_id = line
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_owned();
        let is_amd = line.contains("[1002")
            || line.contains("Advanced Micro Devices")
            || line.contains("AMD");
        let is_nvidia = line.contains("[10de") || line.contains("NVIDIA");
        let name = extract_lspci_name(line);
        if is_nvidia {
            e.has_nvidia_gpu = true;
            e.gpus.push(Gpu {
                name,
                pci_id,
                is_amd: false,
                is_apu: Some(false),
                ..Gpu::default()
            });
            continue;
        }
        if !is_amd {
            continue;
        }
        let (gfx_guess, is_apu_guess) = classify_amd_marketing_name(&name);
        e.gpus.push(Gpu {
            name,
            gfx_target: gfx_guess,
            pci_id,
            is_apu: Some(is_apu_guess),
            is_amd: true,
        });
    }
}

/// Pull the marketing name out of an `lspci -nn` line: the text between the
/// controller-kind `]:` and the trailing `[vendor:device]`.
fn extract_lspci_name(line: &str) -> String {
    let after_colon = match line.find("]:") {
        Some(idx) => &line[idx + 2..],
        None => match line.find(':') {
            Some(idx) => &line[idx + 1..],
            None => line,
        },
    };
    let trimmed = match after_colon.rfind('[') {
        Some(idx) => &after_colon[..idx],
        None => after_colon,
    };
    trimmed.trim().to_owned()
}

fn probe_gpus_rocminfo(e: &mut Examination) {
    if !which("rocminfo") {
        e.rocminfo_present = false;
        e.rocminfo_status = "missing".to_owned();
        return;
    }
    e.rocminfo_present = true;
    let (rc, out, err) = run("rocminfo", &[], Duration::from_secs(15));
    if rc != 0 {
        let merged = format!("{out}\n{err}").to_lowercase();
        e.rocminfo_status = if merged.contains("rock module is not loaded") {
            "not-loaded".to_owned()
        } else if merged.contains("permission denied") || merged.contains("operation not permitted")
        {
            "permission-denied".to_owned()
        } else {
            format!("error rc={rc}")
        };
        return;
    }
    e.rocminfo_status = "ok".to_owned();

    let mut gfx_targets: Vec<(String, String)> = Vec::new();
    let mut cur_name = String::new();
    let mut cur_marketing = String::new();
    let mut cur_is_gpu = false;
    for line in out.lines() {
        let s = line.trim();
        if s.starts_with("Agent ") {
            if cur_is_gpu && cur_name.starts_with("gfx") {
                gfx_targets.push((cur_name.clone(), cur_marketing.clone()));
            }
            cur_name.clear();
            cur_marketing.clear();
            cur_is_gpu = false;
        } else if let Some(rest) = s.strip_prefix("Name:") {
            cur_name = rest.trim().to_owned();
        } else if let Some(rest) = s.strip_prefix("Marketing Name:") {
            cur_marketing = rest.trim().to_owned();
        } else if let Some(rest) = s.strip_prefix("Device Type:") {
            cur_is_gpu = rest.contains("GPU");
        }
    }
    if cur_is_gpu && cur_name.starts_with("gfx") {
        gfx_targets.push((cur_name, cur_marketing));
    }
    if gfx_targets.is_empty() {
        return;
    }

    let amd_indices: Vec<usize> = e
        .gpus
        .iter()
        .enumerate()
        .filter(|(_, g)| g.is_amd)
        .map(|(idx, _)| idx)
        .collect();
    for (idx, (gfx, marketing)) in gfx_targets.into_iter().enumerate() {
        if let Some(&gpu_idx) = amd_indices.get(idx) {
            let gpu = &mut e.gpus[gpu_idx];
            gpu.gfx_target = gfx.clone();
            if !marketing.is_empty() && gpu.name.is_empty() {
                gpu.name = marketing;
            }
            gpu.is_apu = Some(gfx_is_apu_family(&gfx));
        } else {
            let is_apu = gfx_is_apu_family(&gfx);
            e.gpus.push(Gpu {
                name: if marketing.is_empty() {
                    "AMD GPU".to_owned()
                } else {
                    marketing
                },
                gfx_target: gfx,
                is_amd: true,
                is_apu: Some(is_apu),
                ..Gpu::default()
            });
        }
    }
}

fn summarise_gpu_categories(e: &mut Examination) {
    e.has_amd_gpu = e.gpus.iter().any(|g| g.is_amd);
    e.has_apu = e.gpus.iter().any(|g| g.is_amd && g.is_apu == Some(true));
    e.has_discrete_amd = e.gpus.iter().any(|g| g.is_amd && g.is_apu == Some(false));
}

// ---------------------------------------------------------------------------
// Kernel module / device probes (Linux)
// ---------------------------------------------------------------------------

fn probe_modules(e: &mut Examination) {
    let (rc, out, _) = run("lsmod", &[], SHORT);
    let module_text = if rc == 0 {
        Some(out.lines().skip(1).collect::<Vec<_>>().join("\n"))
    } else {
        let txt = read_text("/proc/modules");
        if txt.is_empty() { None } else { Some(txt) }
    };
    if let Some(text) = module_text {
        let modules: Vec<&str> = text
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        e.amdgpu_loaded = Some(modules.contains(&"amdgpu"));
        e.amdkfd_loaded = Some(modules.contains(&"amdkfd"));
    }

    for dir in ["/etc/modprobe.d", "/usr/lib/modprobe.d", "/run/modprobe.d"] {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("conf") {
                continue;
            }
            let body = read_text(&path.to_string_lossy());
            if body.lines().any(line_blacklists_amdgpu) {
                e.amdgpu_blacklisted_in
                    .push(path.to_string_lossy().into_owned());
            }
        }
    }
}

/// Matches `^\s*blacklist\s+amdgpu\b`.
fn line_blacklists_amdgpu(line: &str) -> bool {
    let rest = line.trim_start();
    let Some(rest) = rest.strip_prefix("blacklist") else {
        return false;
    };
    let rest = rest.trim_start();
    rest == "amdgpu"
        || rest.strip_prefix("amdgpu").is_some_and(|tail| {
            tail.is_empty() || !tail.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        })
}

fn probe_devices(e: &mut Examination) {
    e.kfd = Some(stat_device("/dev/kfd", &e.user_name, &e.user_groups));
    if let Ok(entries) = std::fs::read_dir("/dev/dri") {
        let mut render: Vec<String> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("renderD")
                    .then(|| entry.path().to_string_lossy().into_owned())
            })
            .collect();
        render.sort();
        for path in render {
            e.render_devices
                .push(stat_device(&path, &e.user_name, &e.user_groups));
        }
    }
}

fn stat_device(path: &str, user_name: &str, user_groups: &[String]) -> Device {
    let mut device = Device {
        path: path.to_owned(),
        exists: Path::new(path).exists(),
        ..Device::default()
    };
    if !device.exists {
        return device;
    }
    let (rc, out, _) = run("stat", &["-c", "%A|%U|%G", path], SHORT);
    if rc == 0 {
        let fields: Vec<&str> = out.trim().split('|').collect();
        if fields.len() == 3 {
            device.mode = fields[0].to_owned();
            device.owner_user = fields[1].to_owned();
            device.owner_group = fields[2].to_owned();
            let (can_read, can_write) = mode_access(
                &device.mode,
                &device.owner_user,
                &device.owner_group,
                user_name,
                user_groups,
            );
            device.user_can_read = can_read;
            device.user_can_write = can_write;
        }
    }
    device
}

/// Derive read/write access from a `stat`-style mode string and group
/// membership, following POSIX precedence (owner, then group, then other).
fn mode_access(
    mode: &str,
    owner_user: &str,
    owner_group: &str,
    user_name: &str,
    user_groups: &[String],
) -> (Option<bool>, Option<bool>) {
    let bytes = mode.as_bytes();
    if bytes.len() < 10 {
        return (None, None);
    }
    let class = if !user_name.is_empty() && user_name == owner_user {
        1
    } else if user_groups.iter().any(|g| g == owner_group) {
        4
    } else {
        7
    };
    let read = bytes[class] == b'r';
    let write = bytes[class + 1] == b'w';
    (Some(read), Some(write))
}

fn probe_user(e: &mut Examination) {
    e.user_name = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_default();
    let (rc, out, _) = run("id", &["-Gn"], Duration::from_secs(3));
    if rc == 0 {
        e.user_groups = out.split_whitespace().map(str::to_owned).collect();
    }
    e.in_render_group = Some(e.user_groups.iter().any(|g| g == "render"));
    e.in_video_group = Some(e.user_groups.iter().any(|g| g == "video"));
}

fn probe_secure_boot(e: &mut Examination) {
    if !which("mokutil") {
        return;
    }
    let (rc, out, _) = run("mokutil", &["--sb-state"], Duration::from_secs(3));
    if rc == 0 {
        let o = out.to_lowercase();
        if o.contains("enabled") {
            e.secure_boot = "enabled".to_owned();
        } else if o.contains("disabled") {
            e.secure_boot = "disabled".to_owned();
        }
    }
}

// ---------------------------------------------------------------------------
// ROCm install probe (Linux)
// ---------------------------------------------------------------------------

fn probe_rocm_install(e: &mut Examination) {
    let mut rocm_dir = String::new();
    let rocm_path_env = std::env::var("ROCM_PATH").unwrap_or_default();
    for candidate in ["/opt/rocm", rocm_path_env.as_str()] {
        if !candidate.is_empty() && Path::new(candidate).is_dir() {
            rocm_dir = candidate.to_owned();
            break;
        }
    }
    e.rocm_path = rocm_dir.clone();

    if !rocm_dir.is_empty() {
        for fname in ["version", "version-utils", "version-libs"] {
            let f = Path::new(&rocm_dir).join(".info").join(fname);
            if f.exists() {
                e.rocm_version = read_text(&f.to_string_lossy()).trim().to_owned();
                break;
            }
        }
        if e.rocm_version.is_empty()
            && let Ok(real) = std::fs::canonicalize(&rocm_dir)
            && let Some(version) = extract_rocm_version(&real.to_string_lossy())
        {
            e.rocm_version = version;
        }
    }

    for marker in AMDGPU_INSTALL_MARKERS {
        if Path::new(marker).exists() {
            e.rocm_install_method = "amdgpu-install".to_owned();
            e.rocm_repos_seen.push((*marker).to_owned());
        }
    }

    if e.rocm_install_method.is_empty() {
        if which("dpkg") {
            let (rc, out, _) = run("dpkg", &["-l", "rocm-hip-runtime"], MEDIUM);
            if rc == 0 && out.contains("rocm-hip-runtime") {
                e.rocm_install_method = "apt".to_owned();
            }
        }
        if e.rocm_install_method.is_empty() && which("rpm") {
            let (rc, out, _) = run("rpm", &["-q", "rocm-hip-runtime"], MEDIUM);
            if rc == 0 && out.contains("rocm-hip-runtime") {
                e.rocm_install_method = "dnf".to_owned();
            }
        }
    }
    if e.rocm_install_method.is_empty() {
        e.rocm_install_method = if rocm_dir.is_empty() {
            "none".to_owned()
        } else {
            "tarball-or-other".to_owned()
        };
    }

    for dir in ["/etc/apt/sources.list.d", "/etc/yum.repos.d"] {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name.contains("rocm") || name.contains("amdgpu") || name.contains("radeon") {
                let full = entry.path().to_string_lossy().into_owned();
                if !e.rocm_repos_seen.contains(&full) {
                    e.rocm_repos_seen.push(full);
                }
            }
        }
    }
}

/// Pull `X.Y[.Z]` out of a `rocm-X.Y.Z` path component.
fn extract_rocm_version(path: &str) -> Option<String> {
    let idx = path.find("rocm-")?;
    let tail = &path[idx + "rocm-".len()..];
    let version: String = tail
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let trimmed = version.trim_matches('.');
    (trimmed.contains('.')).then(|| trimmed.to_owned())
}

// ---------------------------------------------------------------------------
// Framework probes
// ---------------------------------------------------------------------------

const PYTORCH_PROBE: &str = concat!(
    "import json,sys\n",
    "out={'ok':False}\n",
    "try:\n",
    "  import torch\n",
    "  out['ok']=True\n",
    "  out['version']=torch.__version__\n",
    "  out['hip']=getattr(torch.version,'hip',None)\n",
    "  out['cuda']=getattr(torch.version,'cuda',None)\n",
    "  out['is_available']=bool(torch.cuda.is_available())\n",
    "  try: out['device_count']=int(torch.cuda.device_count())\n",
    "  except Exception: out['device_count']=0\n",
    "  try: out['arch_list']=list(torch.cuda.get_arch_list())\n",
    "  except Exception: out['arch_list']=[]\n",
    "except Exception as ex:\n",
    "  out['error']=type(ex).__name__+': '+str(ex)\n",
    "sys.stdout.write(json.dumps(out))\n",
);

fn probe_framework(e: &mut Examination, framework: FrameworkProbe) {
    match framework {
        FrameworkProbe::Skip => e.framework = "skipped".to_owned(),
        FrameworkProbe::PyTorch => probe_pytorch(e),
        FrameworkProbe::LlamaCpp => probe_llama_cpp(e),
        FrameworkProbe::Auto => {
            if which("python") || which("python3") {
                probe_pytorch(e);
                if e.framework == "pytorch" {
                    return;
                }
            }
            probe_llama_cpp(e);
        }
    }
}

fn probe_pytorch(e: &mut Examination) {
    let py = if which("python") {
        "python"
    } else if which("python3") {
        "python3"
    } else {
        e.framework_notes
            .push("No python interpreter found to probe torch.".to_owned());
        return;
    };
    let (rc, out, err) = run(py, &["-c", PYTORCH_PROBE], Duration::from_secs(20));
    let (out, err) = if (rc != 0 || out.trim().is_empty()) && py == "python" && which("python3") {
        let (_, out2, err2) = run("python3", &["-c", PYTORCH_PROBE], Duration::from_secs(20));
        if out2.trim().is_empty() {
            (out, err)
        } else {
            (out2, err2)
        }
    } else {
        (out, err)
    };
    if out.trim().is_empty() {
        e.framework_notes.push(
            "Could not import torch; if PyTorch is in a venv, activate it and re-run inside that venv."
                .to_owned(),
        );
        if let Some(last) = err.trim().lines().last() {
            let snippet: String = last.chars().take(200).collect();
            e.framework_notes.push(format!("python stderr: {snippet}"));
        }
        return;
    }
    let Ok(data) = serde_json::from_str::<serde_json::Value>(out.trim()) else {
        let snippet: String = out.chars().take(200).collect();
        e.framework_notes
            .push(format!("torch probe returned non-JSON: {snippet}"));
        return;
    };
    if data.get("ok").and_then(serde_json::Value::as_bool) != Some(true) {
        let err = data
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        e.framework_notes
            .push(format!("torch import failed: {err}"));
        return;
    }
    e.framework = "pytorch".to_owned();
    e.framework_version = data
        .get("version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let hip = data.get("hip").and_then(serde_json::Value::as_str);
    let cuda = data.get("cuda").and_then(serde_json::Value::as_str);
    if let Some(hip) = hip.filter(|h| !h.is_empty()) {
        e.framework_rocm_version = format!("hip={hip}");
    } else if let Some(cuda) = cuda.filter(|c| !c.is_empty()) {
        e.framework_rocm_version = format!("cuda={cuda}");
        e.framework_notes.push(
            "This torch wheel is a CUDA build, not a ROCm build. Reinstall from the ROCm wheel index."
                .to_owned(),
        );
    }
    if let Some(arch) = data.get("arch_list").and_then(serde_json::Value::as_array) {
        e.framework_arch_list = arch
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
    }
    if data
        .get("is_available")
        .and_then(serde_json::Value::as_bool)
        == Some(false)
    {
        e.framework_notes.push(
            "torch.cuda.is_available() returned False -- runtime can't see a GPU.".to_owned(),
        );
    }
}

fn probe_llama_cpp(e: &mut Examination) {
    let binary = ["llama-cli", "llama-server", "main"]
        .into_iter()
        .find(|name| which(name));
    let Some(binary) = binary else {
        e.framework_notes
            .push("No llama.cpp binary (llama-cli/llama-server/main) on PATH.".to_owned());
        return;
    };
    let (rc, out, err) = run(binary, &["--version"], Duration::from_secs(10));
    let body = format!("{out}{err}");
    if rc != 0 && body.is_empty() {
        e.framework_notes
            .push(format!("{binary} --version exited rc={rc}"));
        return;
    }
    e.framework = "llama-cpp".to_owned();
    e.framework_version = body.trim().lines().next().map_or_else(
        || "unknown".to_owned(),
        |line| line.chars().take(200).collect(),
    );
    if body.contains("HIP") || body.contains("ROCm") || body.contains("hipBLAS") {
        e.framework_rocm_version = "GGML_HIP=ON".to_owned();
    } else {
        e.framework_notes.push(
            "llama.cpp binary doesn't advertise HIP/ROCm support; was it built with `cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx>`?"
                .to_owned(),
        );
    }
}

// ---------------------------------------------------------------------------
// Misc probes
// ---------------------------------------------------------------------------

fn probe_env(e: &mut Examination) {
    for key in TRACKED_ENV_VARS {
        let Ok(value) = std::env::var(key) else {
            continue;
        };
        let value = if matches!(*key, "PATH" | "LD_LIBRARY_PATH") && value.len() > 4000 {
            format!("{}...[truncated]", &value[..4000])
        } else {
            value
        };
        e.env.insert((*key).to_owned(), value);
    }
    let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let mut hit: Option<String> = None;
    for dir in ld.split(':') {
        if dir.is_empty() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("libamdhip64")
                {
                    hit = Some(entry.path().to_string_lossy().into_owned());
                    break;
                }
            }
        }
        if hit.is_some() {
            break;
        }
    }
    if let Some(hit) = hit {
        e.hip_libs_on_ld_path = Some(true);
        e.notes
            .push(format!("libamdhip64 visible via LD_LIBRARY_PATH: {hit}"));
    } else {
        e.hip_libs_on_ld_path = if ld.is_empty() { None } else { Some(false) };
    }
}

fn probe_container(e: &mut Examination) {
    for (marker, kind) in [("/.dockerenv", "docker"), ("/run/.containerenv", "podman")] {
        if Path::new(marker).exists() {
            e.in_container = true;
            e.container_kind = kind.to_owned();
            return;
        }
    }
    let cg = read_text("/proc/1/cgroup");
    if !cg.is_empty()
        && ["docker", "containerd", "lxc", "kubepods", "podman"]
            .iter()
            .any(|x| cg.contains(x))
    {
        e.in_container = true;
        if e.container_kind.is_empty() {
            e.container_kind = "container".to_owned();
        }
    }
}

fn probe_dmesg_amdgpu(e: &mut Examination) {
    let (rc, out, _) = run("journalctl", &["-k", "--no-pager", "-n", "400"], MEDIUM);
    let text = if rc == 0 && !out.is_empty() {
        out
    } else {
        let (rc2, out2, _) = run("dmesg", &[], SHORT);
        if rc2 == 0 { out2 } else { String::new() }
    };
    if text.is_empty() {
        return;
    }
    let interesting = [
        "page fault",
        "ras controller",
        "vm_fault",
        "amdgpu_device_init",
        "out_of_registers",
        "ring",
        "gpu reset",
        "psp",
        "hw_fault",
    ];
    let mut hits: Vec<String> = Vec::new();
    for line in text.lines() {
        if !line.contains("amdgpu") && !line.contains("amdkfd") {
            continue;
        }
        let lower = line.to_lowercase();
        if interesting.iter().any(|s| lower.contains(s)) {
            hits.push(line.trim().chars().take(300).collect());
        }
    }
    let start = hits.len().saturating_sub(15);
    e.dmesg_amdgpu_tail = hits.split_off(start);
}

// ---------------------------------------------------------------------------
// Windows-specific probes (best-effort)
// ---------------------------------------------------------------------------

const WIN_GPU_SCRIPT: &str = "Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -match 'VEN_1002' -or $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' -or $_.PNPDeviceID -match 'VEN_10DE' -or $_.Name -match 'NVIDIA' } | ForEach-Object { \"$($_.Name)`t$($_.DriverVersion)`t$($_.PNPDeviceID)\" }";

fn probe_gpus_windows(e: &mut Examination) {
    let (rc, out, _) = run(
        "powershell",
        &["-NoProfile", "-Command", WIN_GPU_SCRIPT],
        MEDIUM,
    );
    if rc != 0 {
        e.probe_failures
            .push("Win32_VideoController query failed; cannot enumerate GPUs.".to_owned());
        return;
    }
    for line in out.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let name = fields
            .first()
            .copied()
            .unwrap_or_default()
            .trim()
            .to_owned();
        let pnp = fields.get(2).copied().unwrap_or_default().trim().to_owned();
        let lname = name.to_lowercase();
        let is_amd = pnp.to_uppercase().contains("VEN_1002")
            || lname.contains("amd")
            || lname.contains("radeon")
            || lname.contains("instinct");
        let is_nvidia = pnp.to_uppercase().contains("VEN_10DE") || lname.contains("nvidia");
        if is_nvidia && !is_amd {
            e.has_nvidia_gpu = true;
            e.gpus.push(Gpu {
                name,
                pci_id: pnp,
                is_amd: false,
                is_apu: Some(false),
                ..Gpu::default()
            });
            continue;
        }
        if !is_amd {
            continue;
        }
        let (gfx_guess, is_apu_guess) = classify_amd_marketing_name(&name);
        e.gpus.push(Gpu {
            name,
            gfx_target: gfx_guess,
            pci_id: pnp,
            is_apu: Some(is_apu_guess),
            is_amd: true,
        });
    }
}

fn probe_hip_sdk_windows(e: &mut Examination) {
    let mut root = std::env::var("HIP_PATH").unwrap_or_default();
    if root.is_empty() || !Path::new(&root).is_dir() {
        // Scan the conventional install location for the newest ROCm dir.
        let base = Path::new(r"C:\Program Files\AMD\ROCm");
        if let Ok(entries) = std::fs::read_dir(base) {
            let mut versions: Vec<String> = entries
                .flatten()
                .filter(|entry| entry.path().is_dir())
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect();
            versions.sort();
            if let Some(latest) = versions.last() {
                root = base.join(latest).to_string_lossy().into_owned();
            }
        }
    }
    if root.is_empty() || !Path::new(&root).is_dir() {
        return;
    }
    e.hip_sdk_path = root.clone();
    e.hip_sdk_version = Path::new(&root)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let hipinfo = Path::new(&root).join("bin").join("hipInfo.exe");
    if hipinfo.is_file() {
        e.hipinfo_present = true;
        let (rc, out, _) = run(&hipinfo.to_string_lossy(), &[], Duration::from_secs(15));
        if rc == 0 {
            e.hipinfo_status = "ok".to_owned();
            for line in out.lines() {
                if let Some(rest) = line.trim().strip_prefix("gcnArchName:")
                    && let Some(gfx) = crate::extract_first_gfx_token(rest)
                    && let Some(gpu) = e
                        .gpus
                        .iter_mut()
                        .find(|g| g.is_amd && g.gfx_target.is_empty())
                {
                    gpu.gfx_target = gfx;
                    gpu.is_apu = Some(gfx_is_apu_family(&gpu.gfx_target));
                }
            }
        } else {
            e.hipinfo_status = format!("error rc={rc}");
        }
    } else {
        e.hipinfo_present = false;
        e.hipinfo_status = "missing".to_owned();
    }
}

fn probe_adrenalin_windows(e: &mut Examination) {
    let script = "(Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -match 'VEN_1002' -or $_.Name -match 'AMD|Radeon|Instinct' } | Select-Object -First 1).DriverVersion";
    let (rc, out, _) = run("powershell", &["-NoProfile", "-Command", script], MEDIUM);
    if rc == 0 && !out.trim().is_empty() {
        e.adrenalin_version = out
            .trim()
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();
    }
}

fn probe_msvc_redist_windows(e: &mut Examination) {
    let mut search_dirs: Vec<String> = Vec::new();
    if let Ok(path) = std::env::var("PATH") {
        search_dirs.extend(path.split(';').map(str::to_owned));
    }
    for dir in [r"C:\Windows\System32", r"C:\Windows\SysWOW64"] {
        search_dirs.push(dir.to_owned());
    }
    let present = search_dirs.iter().any(|dir| {
        !dir.is_empty()
            && (Path::new(dir).join("vcruntime140.dll").is_file()
                || Path::new(dir).join("vcruntime140_1.dll").is_file())
    });
    e.msvc_redist_present = Some(present);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examination_serializes_expected_keys() {
        let e = Examination::default();
        let value = serde_json::to_value(&e).expect("serialize");
        // A representative slice of the contract diagnose.py depends on.
        for key in [
            "os_family",
            "gpus",
            "has_amd_gpu",
            "in_render_group",
            "amdgpu_loaded",
            "kfd",
            "rocm_install_method",
            "rocminfo_status",
            "framework_arch_list",
            "env",
            "dmesg_amdgpu_tail",
            "probe_failures",
        ] {
            assert!(value.get(key).is_some(), "missing key: {key}");
        }
    }

    #[test]
    fn examination_top_level_keys_match_examine_py_contract() {
        // The exact field set examine.py emits. diagnose.py reads against these
        // names, so this is the frozen wire contract — adding/removing/renaming a
        // top-level field here is a contract change and must be intentional.
        let expected: std::collections::BTreeSet<&str> = [
            "os_family",
            "os_version",
            "distro_id",
            "distro_version",
            "kernel_release",
            "kernel_cmdline",
            "is_wsl",
            "cpu_vendor",
            "cpu_model",
            "gpus",
            "has_amd_gpu",
            "has_nvidia_gpu",
            "has_apu",
            "has_discrete_amd",
            "amdgpu_loaded",
            "amdgpu_blacklisted_in",
            "amdkfd_loaded",
            "secure_boot",
            "iommu_kernel_param",
            "kfd",
            "render_devices",
            "user_name",
            "user_groups",
            "in_render_group",
            "in_video_group",
            "rocm_version",
            "rocm_install_method",
            "rocm_path",
            "rocminfo_present",
            "rocminfo_status",
            "hip_libs_on_ld_path",
            "rocm_repos_seen",
            "hip_sdk_path",
            "hip_sdk_version",
            "hipinfo_present",
            "hipinfo_status",
            "adrenalin_version",
            "msvc_redist_present",
            "framework",
            "framework_version",
            "framework_rocm_version",
            "framework_arch_list",
            "framework_notes",
            "env",
            "in_container",
            "container_kind",
            "dmesg_amdgpu_tail",
            "notes",
            "probe_failures",
        ]
        .into_iter()
        .collect();
        let value = serde_json::to_value(Examination::default()).expect("serialize");
        let actual: std::collections::BTreeSet<&str> = value
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            actual, expected,
            "Examination top-level keys drifted from examine.py"
        );
    }

    #[test]
    fn default_uses_unknown_sentinels() {
        let e = Examination::default();
        assert_eq!(e.os_family, "unknown");
        assert_eq!(e.cpu_vendor, "unknown");
        assert_eq!(e.secure_boot, "unknown");
        assert_eq!(e.framework, "unknown");
    }

    #[test]
    fn examination_round_trips_through_json() {
        let e = Examination::default();
        let json = serde_json::to_string(&e).expect("serialize");
        let back: Examination = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.os_family, e.os_family);
        assert_eq!(back.framework, e.framework);
    }

    #[test]
    fn optional_bool_serializes_as_null_not_omitted() {
        let e = Examination::default();
        let value = serde_json::to_value(&e).expect("serialize");
        assert!(value.get("in_render_group").expect("present").is_null());
        assert!(value.get("amdgpu_loaded").expect("present").is_null());
        assert!(value.get("kfd").expect("present").is_null());
    }

    #[test]
    fn iommu_param_parsed_from_cmdline() {
        assert_eq!(
            parse_iommu_param("BOOT_IMAGE=/vmlinuz iommu=pt amd_iommu=on quiet"),
            Some("pt".to_owned())
        );
        assert_eq!(parse_iommu_param("BOOT_IMAGE=/vmlinuz quiet"), None);
    }

    #[test]
    fn lspci_name_extraction() {
        let line = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]";
        assert_eq!(
            extract_lspci_name(line),
            "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"
        );
    }

    #[test]
    fn blacklist_amdgpu_detection() {
        assert!(line_blacklists_amdgpu("blacklist amdgpu"));
        assert!(line_blacklists_amdgpu("  blacklist   amdgpu"));
        assert!(!line_blacklists_amdgpu("blacklist amdgpufoo"));
        assert!(!line_blacklists_amdgpu("# blacklist amdgpu"));
    }

    #[test]
    fn mode_access_owner_group_other_precedence() {
        // crw-rw---- root render: a render member can write, others cannot.
        let groups = vec!["render".to_owned()];
        let (r, w) = mode_access("crw-rw----", "root", "render", "alice", &groups);
        assert_eq!((r, w), (Some(true), Some(true)));
        let (r, w) = mode_access("crw-rw----", "root", "render", "alice", &[]);
        assert_eq!((r, w), (Some(false), Some(false)));
    }

    #[test]
    fn gfx_apu_family_classification() {
        // Mirrors examine.py's `gfx11[05]\d` regex exactly (faithful parity),
        // including that it matches gfx1100 — diagnose.py is written against
        // this behavior, so we reproduce it rather than "correct" it.
        assert!(gfx_is_apu_family("gfx1151"));
        assert!(gfx_is_apu_family("gfx1103"));
        assert!(gfx_is_apu_family("gfx1100"));
        assert!(!gfx_is_apu_family("gfx1200"));
        assert!(!gfx_is_apu_family("gfx942"));
    }

    #[test]
    fn marketing_name_maps_strix_halo() {
        assert_eq!(
            classify_amd_marketing_name("AMD Radeon(TM) 8060S Graphics"),
            ("gfx1151".to_owned(), true)
        );
        assert_eq!(
            classify_amd_marketing_name("Ryzen AI Max+ 395"),
            ("gfx1151".to_owned(), true)
        );
    }

    #[test]
    fn rocm_version_extracted_from_path() {
        assert_eq!(
            extract_rocm_version("/opt/rocm-6.4.1"),
            Some("6.4.1".to_owned())
        );
        assert_eq!(extract_rocm_version("/opt/rocm"), None);
    }
}
