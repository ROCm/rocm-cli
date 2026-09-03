// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host examination probe.
//!
//! Rust port of the `rocm-doctor` skill's `examine.py`. It gathers the host
//! signals the diagnosis catalog reasons over and serializes them as the
//! **Examination** JSON document (`rocm examine --json`). The field names and
//! shapes mirror `examine.py` field-for-field so the catalog consumes the CLI's
//! output unchanged.

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
/// Cap on captured `PATH` / `LD_LIBRARY_PATH` length, to keep the Examination
/// JSON bounded. Generous on purpose: the stored value is read by the catalog
/// (e.g. the `fix-6-path` PATH check), so the cut must sit well past any
/// realistic ROCm/HIP bin entry to avoid false "not on PATH" diagnoses.
const ENV_VALUE_MAX_CHARS: usize = 16_000;

const TRACKED_ENV_VARS: &[&str] = &[
    "HSA_OVERRIDE_GFX_VERSION",
    // Legacy ROCm releases need this to find the GPU through DXG under WSL.
    "HSA_ENABLE_DXG_DETECTION",
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

/// The oldest distro release the WSL path supports.
///
/// Ubuntu 22.04 ships glibc 2.35, below the glibc 2.38 / `GLIBCXX_3.4.32` floor
/// every published Lemonade embeddable is linked against, so the engine cannot
/// start there. See `docs/wsl.md`.
pub const WSL_MIN_UBUNTU: (u32, u32) = (24, 4);

/// WSL2-specific machine state. `None` on every other platform.
///
/// WSL reaches the GPU through `/dev/dxg` and the Windows host driver rather than
/// the in-tree `amdgpu` module, so none of the bare-metal driver fields describe
/// it. These are the facts the WSL half of the catalog reasons over.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WslFacts {
    /// `1` or `2`. A kernel release that names neither is read as `2`: WSL 2 has
    /// been the default for years, and the cost of the two errors is not
    /// symmetric — calling a WSL 2 host "WSL 1" tells the user to convert a
    /// distro that is already converted. `0` only in the default value, which
    /// stands for "the probe did not run".
    pub version: u8,
    pub dxg_device: bool,
    pub dxcore: bool,
    pub wsl_lib_dir: bool,
    pub librocdxg: bool,
    pub rocdxg_dids: bool,
    /// Whether the linker cache lists ROCDXG.
    ///
    /// `None` when `ldconfig` could not be run at all — on Debian and its
    /// derivatives it lives in `/sbin`, off a non-root user's `PATH`. An
    /// unreadable cache is not an unregistered library, and reporting it as one
    /// told users with a working install to re-run `ldconfig`.
    pub ldconfig_librocdxg: Option<bool>,
    /// Whether `rocminfo` is on PATH.
    pub rocminfo: bool,
    /// Whether ROCm can actually enumerate a GPU here.
    ///
    /// `None` when `rocminfo` is absent, so the question could not be asked. This
    /// is the only WSL-collected evidence that the plumbing is complete yet no
    /// device is reachable, which is what distinguishes an out-of-date Windows
    /// host driver from a distro-side fault. The bare-metal `has_amd_gpu` cannot
    /// stand in: the probes that populate it are skipped here, so it is always
    /// false on WSL and reads as "no GPU" on a perfectly healthy machine.
    pub rocm_sees_gpu: Option<bool>,
    /// `None` when the distro release could not be parsed, which fails closed:
    /// an unreadable release is not evidence of a supported one.
    pub distro_supported: Option<bool>,
    /// `None` when WSL interop could not reach the Windows host — distinct from
    /// a host that answered and reported no AMD adapter, which is `Some("")`.
    pub host_driver_version: Option<String>,
    pub host_reachable: bool,
    /// Whether these facts were gathered from inside the distribution.
    ///
    /// `false` when inspected from the Windows host over `wsl.exe`, which sees
    /// the GPU stack but no environment — so the checks that read one cannot
    /// run, and a caller must say so rather than let "nothing matched" read as
    /// a clean bill of health.
    pub locally_probed: bool,
}

/// Structured machine state consumed by the diagnosis catalog.
///
/// Field order and names mirror `examine.py`'s `Examination` dataclass so the
/// JSON contract is identical, except for the `wsl` section, which has no
/// `examine.py` analogue.
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
    /// Populated only when `is_wsl`; see [`WslFacts`].
    pub wsl: Option<WslFacts>,

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

    // CLI addition (not in examine.py): a coarse machine-readable verdict so
    // callers branch on this field instead of the process exit code.
    // One of: "ok" | "no-amd-gpu" | "wsl" | "unsupported-os" | "degraded".
    pub status: String,
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
            wsl: None,
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
            status: "ok".to_owned(),
        }
    }
}

/// Guidance shown when WSL2 is detected.
///
/// Named for what it does. It used to be a route-out — `rocm examine` did not
/// cover the ROCm-on-WSL flow and sent the user elsewhere — and it kept that
/// name for a while after it stopped routing anyone anywhere. It now explains
/// which checks are skipped on this platform and points at the one that covers
/// it.
pub const WSL_PLATFORM_NOTE: &str = "Detected WSL2. The GPU is reached through /dev/dxg and the Windows host driver, so the bare-metal driver checks do not apply and are skipped. Run `rocm diagnose` for the WSL-specific checks. Setup guide: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html";

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
        if e.is_wsl {
            // WSL2 keeps the *driver* probes skipped: it reaches the GPU through
            // /dev/dxg and the Windows host driver, not the in-tree amdgpu module
            // or /dev/kfd, so asking about modprobe, the render group or
            // /dev/kfd would only mislead.
            //
            // Everything else applies. This used to return here after the
            // framework probe alone, which left `env`, the container fields and
            // the ROCm install at their defaults -- so the WSL half of the
            // catalog had nothing to read and questions with no kernel-module
            // component, like "is HSA_OVERRIDE_GFX_VERSION set", went unanswered
            // on the one platform most likely to need them.
            probe_wsl(&mut e);
            probe_rocm_install(&mut e);
            probe_env(&mut e);
            probe_container(&mut e);
            probe_framework(&mut e, framework);
            e.status = e.compute_status();
            return e;
        }
        if e.os_family == "linux" {
            probe_cpu_linux(&mut e);
            probe_gpus_lspci(&mut e);
            probe_gpus_rocminfo(&mut e);
            probe_gpus_sysfs_fallback(&mut e);
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
        e.status = e.compute_status();
        e
    }

    /// Coarse machine-readable verdict reported via the `status` field. The
    /// process exit code does NOT encode this — `rocm examine` always exits 0
    /// when it ran; callers branch on `status` instead.
    fn compute_status(&self) -> String {
        let value = if self.is_wsl {
            "wsl"
        } else if !matches!(self.os_family.as_str(), "linux" | "windows") {
            "unsupported-os"
        } else if !self.has_amd_gpu {
            "no-amd-gpu"
        } else if !self.probe_failures.is_empty() {
            "degraded"
        } else {
            "ok"
        };
        value.to_owned()
    }
}

// ---------------------------------------------------------------------------
// Shell / fs helpers (never panic)
// ---------------------------------------------------------------------------

/// Run a command with a timeout. Returns `(rc, stdout, stderr)`. `rc` is `127`
/// when the program can't be spawned and `124` on timeout.
pub(crate) fn run(program: &str, args: &[&str], timeout: Duration) -> (i32, String, String) {
    let (rc, stdout, stderr) = run_raw(program, args, timeout);
    (
        rc,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

/// [`run`] without the UTF-8 assumption, for output that is not UTF-8.
fn run_raw(program: &str, args: &[&str], timeout: Duration) -> (i32, Vec<u8>, Vec<u8>) {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return (127, Vec::new(), Vec::new());
    };
    // Bytes, then a lossy conversion at the end. `read_to_string` FAILS on
    // invalid UTF-8 and the error was discarded, so a single stray byte silently
    // emptied the whole capture — which reads downstream as "the command printed
    // nothing", not as "the output could not be decoded".
    let stdout_handle = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        })
    });
    let stderr_handle = child.stderr.take().map(|mut stderr| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
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
    let stdout: Vec<u8> = stdout_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr: Vec<u8> = stderr_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let rc = match status {
        Some(status) => status.code().unwrap_or(-1),
        None => 124,
    };
    (rc, stdout, stderr)
}

/// Run a command whose output is UTF-16LE, as `wsl.exe`'s is.
///
/// Decoding is by declaration, not detection. Sniffing the encoding cannot work
/// here: UTF-16LE text in a Latin or Cyrillic script is made entirely of bytes
/// below 0x80, so it is *valid UTF-8* and decodes without error straight into
/// mojibake — no NUL-density or validity test can tell the two apart. The one
/// reliable fact is which program produced the bytes.
fn run_utf16le(program: &str, args: &[&str], timeout: Duration) -> (i32, String) {
    let (rc, stdout, _) = run_raw(program, args, timeout);
    (rc, decode_utf16le(&stdout))
}

fn decode_utf16le(bytes: &[u8]) -> String {
    let body = bytes.strip_prefix(&[0xFF, 0xFE]).unwrap_or(bytes);
    // `chunks_exact` drops a trailing odd byte rather than panicking on it.
    let units: Vec<u16> = body
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    String::from_utf16_lossy(&units)
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
    // The OS *version*, mirroring examine.py's `platform.version()`. This used
    // to hold `std::env::consts::OS`, which is the OS *name* and so merely
    // repeated `os_family`, telling a reader of the report nothing.
    //
    // Left empty on platforms the CLI does not support rather than guessed at:
    // an empty field reads as "not collected", while a wrong one would be
    // reasoned over by the diagnosis catalog.
    e.os_version = if runtime_is_linux() {
        run("uname", &["-v"], SHORT).1.trim().to_owned()
    } else if runtime_is_windows() {
        run("cmd", &["/C", "ver"], SHORT).1.trim().to_owned()
    } else {
        String::new()
    };
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
        // One shared answer. This used to be its own predicate, and it differed
        // from the install summary's — so `rocm examine` and `rocm examine
        // --json` could disagree about the platform they were describing.
        e.is_wsl = crate::is_wsl_host();
    } else if runtime_is_windows() {
        e.os_family = "windows".to_owned();
    } else {
        e.os_family = "other".to_owned();
    }
}

/// Collect the WSL-specific facts the WSL half of the catalog reasons over.
///
/// Reuses [`crate::detect_wsl_summary`] for the plumbing it already probes rather
/// than restating those paths, and adds the facts no existing caller needed: the
/// WSL major version, whether the distro release clears the supported floor, and
/// the Windows host driver version.
fn probe_wsl(e: &mut Examination) {
    let summary = crate::detect_wsl_summary();
    let (host_reachable, host_driver_version) = host_driver_fields(crate::detect_wsl_host_driver());
    let rocminfo = which("rocminfo");
    e.wsl = Some(WslFacts {
        version: if crate::is_wsl1_kernel(&e.kernel_release) {
            1
        } else {
            2
        },
        dxg_device: summary.as_ref().is_some_and(|s| s.dxg_device),
        dxcore: summary.as_ref().is_some_and(|s| s.dxcore),
        wsl_lib_dir: Path::new("/usr/lib/wsl/lib").is_dir(),
        librocdxg: summary.as_ref().is_some_and(|s| s.librocdxg),
        rocdxg_dids: summary.as_ref().is_some_and(|s| s.rocdxg_dids),
        // `None` when ldconfig itself could not be run, which the summary's
        // bool cannot express -- recover it from the same source the summary used.
        ldconfig_librocdxg: crate::ldconfig_lists_librocdxg(),
        rocminfo,
        rocm_sees_gpu: rocminfo.then(probe_rocminfo_sees_gpu),
        distro_supported: distro_clears_wsl_floor(&e.distro_id, &e.distro_version),
        host_driver_version,
        host_reachable,
        locally_probed: true,
    });
    sync_shared_fields_from_wsl(e);
    e.notes.push(WSL_PLATFORM_NOTE.to_owned());
}

/// Flatten a host-driver probe into the two [`WslFacts`] fields that carry it.
///
/// One place, so the in-guest and host-side probes cannot drift on the point
/// that matters: `None` means the question went unanswered, and `Some("")` means
/// it was answered with "no AMD adapter". Only the second is evidence.
fn host_driver_fields(probe: crate::WslHostDriverProbe) -> (bool, Option<String>) {
    match probe {
        crate::WslHostDriverProbe::Unreachable => (false, None),
        crate::WslHostDriverProbe::NoAmdDisplay => (true, Some(String::new())),
        crate::WslHostDriverProbe::Version(version) => (true, Some(version)),
    }
}

/// Collect the WSL facts from *outside* the distro, over `wsl.exe`.
///
/// Emits `key=value` lines rather than JSON so the guest side needs nothing but
/// a POSIX shell. The Python preflight this replaces injected a Python program
/// and so required `python3` in the distro — on a machine being checked precisely
/// because it is not set up yet.
///
/// `librocdxg` is globbed across `/opt/rocm*` for the same reason the in-guest
/// probe resolves it across installs: a versioned root must not read as missing.
const WSL_REMOTE_PROBE: &str = r#"
echo "kernel=$(uname -r 2>/dev/null)"
if [ -e /dev/dxg ]; then echo dxg=1; else echo dxg=0; fi
if [ -e /usr/lib/wsl/lib/libdxcore.so ]; then echo dxcore=1; else echo dxcore=0; fi
if [ -d /usr/lib/wsl/lib ]; then echo wsllib=1; else echo wsllib=0; fi
if ls /opt/rocm*/lib/librocdxg.so >/dev/null 2>&1 \
  || ls /usr/local/rocm*/lib/librocdxg.so >/dev/null 2>&1; then echo librocdxg=1; else echo librocdxg=0; fi
if ls /opt/rocm*/share/rocdxg/dids.conf >/dev/null 2>&1 \
  || ls /usr/local/rocm*/share/rocdxg/dids.conf >/dev/null 2>&1; then echo dids=1; else echo dids=0; fi
for ldc in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
  if cache=$(command -v "$ldc" >/dev/null 2>&1 && "$ldc" -p 2>/dev/null); then
    case "$cache" in *librocdxg.so*) echo ldconfig=1 ;; *) echo ldconfig=0 ;; esac
    break
  fi
done
if command -v rocminfo >/dev/null 2>&1; then
  echo rocminfo=1
  if rocminfo 2>/dev/null | grep -qi gfx; then echo rocmgfx=1; else echo rocmgfx=0; fi
else
  echo rocminfo=0
fi
. /etc/os-release 2>/dev/null
echo "id=${ID}"
echo "version=${VERSION_ID}"
"#;

/// Parse `wsl.exe -l -q` into distribution names.
///
/// `-q` prints one bare name per line, so the whole line is the name. It is NOT
/// split on whitespace: `wsl --import "My Distro"` is legal, and truncating that
/// to `My` would both fail to match what the user asked for and hand a wrong
/// name to `wsl.exe -d`.
///
/// The header and `*` handling below is for tolerance only — `-q` emits neither,
/// but a caller passing `-l -v` should not silently get its header row back as a
/// distribution.
#[must_use]
pub fn parse_wsl_distro_list(text: &str) -> Vec<String> {
    text.replace('\u{0}', "")
        .lines()
        .filter_map(|raw| {
            let line = raw.trim();
            if line.is_empty() || line.to_uppercase().starts_with("NAME") {
                return None;
            }
            let line = line.strip_prefix('*').map_or(line, str::trim);
            (!line.is_empty()).then(|| line.to_owned())
        })
        .collect()
}

fn parse_remote_flag(fields: &BTreeMap<String, String>, key: &str) -> bool {
    fields.get(key).is_some_and(|value| value == "1")
}

/// A remote flag that can also report that the question went unanswered.
fn parse_remote_tristate(fields: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    match fields.get(key).map(String::as_str) {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// Inspect a WSL distribution from the Windows host.
///
/// Returns an [`Examination`] the ordinary catalog can be run against, so the
/// host-side check and the in-distro one share a single set of rules. Nothing
/// needs to be installed in the target distro.
///
/// # Errors
///
/// When `wsl.exe` is unavailable, no distribution matches, or the probe cannot
/// be run inside the selected distribution.
pub fn probe_wsl_distro_from_host(distro: Option<&str>) -> Result<Examination, String> {
    if !which("wsl.exe") {
        return Err(
            "wsl.exe was not found; inspecting a distribution this way only works from the Windows host"
                .to_owned(),
        );
    }
    // `-q` prints names only. `-l -v` adds a header row that is localised, and a
    // header the parser fails to recognise is not skipped -- it is taken for a
    // distribution name.
    let (rc, listed) = run_utf16le("wsl.exe", &["-l", "-q"], MEDIUM);
    if rc != 0 {
        return Err("could not list WSL distributions".to_owned());
    }
    let distros = parse_wsl_distro_list(&listed);
    let selected = match distro {
        Some(name) => {
            if !distros.iter().any(|d| d.eq_ignore_ascii_case(name)) {
                return Err(format!(
                    "no WSL distribution named '{name}'; found: {}",
                    distros.join(", ")
                ));
            }
            name.to_owned()
        }
        None => match distros.as_slice() {
            [] => return Err("no WSL distributions were found".to_owned()),
            [only] => only.clone(),
            many => {
                return Err(format!(
                    "several WSL distributions are installed; name one with --distro: {}",
                    many.join(", ")
                ));
            }
        },
    };

    let (rc, out, _) = run(
        "wsl.exe",
        &["-d", &selected, "--exec", "/bin/sh", "-c", WSL_REMOTE_PROBE],
        Duration::from_secs(30),
    );
    if rc != 0 {
        return Err(format!(
            "could not inspect '{selected}'; the distribution may be stopped or unreachable"
        ));
    }
    let fields: BTreeMap<String, String> = out
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let (host_reachable, host_driver_version) =
        host_driver_fields(crate::detect_local_windows_host_driver());

    let mut e = Examination {
        os_family: "linux".to_owned(),
        is_wsl: true,
        kernel_release: fields.get("kernel").cloned().unwrap_or_default(),
        distro_id: fields.get("id").cloned().unwrap_or_default(),
        distro_version: fields.get("version").cloned().unwrap_or_default(),
        ..Examination::default()
    };
    let rocminfo = parse_remote_flag(&fields, "rocminfo");
    e.wsl = Some(WslFacts {
        version: if crate::is_wsl1_kernel(&e.kernel_release) {
            1
        } else {
            2
        },
        dxg_device: parse_remote_flag(&fields, "dxg"),
        dxcore: parse_remote_flag(&fields, "dxcore"),
        wsl_lib_dir: parse_remote_flag(&fields, "wsllib"),
        librocdxg: parse_remote_flag(&fields, "librocdxg"),
        rocdxg_dids: parse_remote_flag(&fields, "dids"),
        ldconfig_librocdxg: parse_remote_tristate(&fields, "ldconfig"),
        rocminfo,
        rocm_sees_gpu: rocminfo.then(|| parse_remote_flag(&fields, "rocmgfx")),
        distro_supported: distro_clears_wsl_floor(&e.distro_id, &e.distro_version),
        // Running on the host, the driver is a local question rather than one
        // that has to cross the interop boundary. It can still go unanswered —
        // the inventory query can fail — and that must stay distinguishable from
        // "the host has no AMD adapter", which is a finding.
        host_driver_version,
        host_reachable,
        locally_probed: false,
    });
    sync_shared_fields_from_wsl(&mut e);
    e.status = "wsl".to_owned();
    Ok(e)
}

/// Whether `rocminfo` enumerates a GPU agent.
///
/// Only the yes/no answer is taken. Parsing the agents into `gpus` is the job of
/// the bare-metal probe, which stays skipped here — this exists so the WSL
/// catalog can tell "the plumbing is complete but no device is reachable" from
/// "the plumbing is incomplete", which is the difference between blaming the
/// Windows host driver and blaming the distro.
fn probe_rocminfo_sees_gpu() -> bool {
    let (rc, out, _) = run("rocminfo", &[], MEDIUM);
    rc == 0 && out.to_lowercase().contains("gfx")
}

/// Mirror the WSL facts onto the shared fields the cross-platform checks read.
///
/// Those checks (PATH, the wheel/ROCm pairing) are valid on WSL and enabled
/// there, but they read fields the bare-metal GPU probe populates — and that
/// probe is skipped here. Left at their defaults they do not read as "unknown",
/// they read as "absent": `rocminfo_present: false` made the PATH check score 50
/// on every WSL host that had ROCm installed.
fn sync_shared_fields_from_wsl(e: &mut Examination) {
    let Some(wsl) = e.wsl.as_ref() else {
        return;
    };
    e.rocminfo_present = wsl.rocminfo;
    e.rocminfo_status = match (wsl.rocminfo, wsl.rocm_sees_gpu) {
        (false, _) => "missing".to_owned(),
        (true, Some(true)) => "ok".to_owned(),
        (true, Some(false)) => "no-agents".to_owned(),
        (true, None) => "unknown".to_owned(),
    };
}

/// Whether the distro release clears the WSL floor in [`WSL_MIN_UBUNTU`].
///
/// `None` means the release could not be read as a `major.minor` pair. That is
/// deliberately not "supported": an unparseable release is not evidence of a good
/// one, and reporting a perfect host on a release nobody could identify is how a
/// user ends up chasing a GPU fault that is really a glibc floor.
///
/// Only Ubuntu carries a floor today, because that is the only distro the WSL
/// path documents. Anything else returns `None` rather than a false verdict.
fn distro_clears_wsl_floor(distro_id: &str, distro_version: &str) -> Option<bool> {
    if !distro_id.eq_ignore_ascii_case("ubuntu") {
        return None;
    }
    let (major, minor) = distro_version.split_once('.')?;
    let major: u32 = major.parse().ok()?;
    let minor: u32 = minor.parse().ok()?;
    Some((major, minor) >= WSL_MIN_UBUNTU)
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

/// Whether a gfx target belongs to an AMD APU family.
///
/// APUs: gfx1103 (Phoenix / Hawk Point) and the gfx115x parts (Strix Point /
/// Strix Halo). Their neighbors gfx1100 / gfx1101 / gfx1102 (Navi 31 / 32 / 33)
/// share the gfx110x prefix but are *discrete* RDNA3 GPUs, so they must not
/// match — otherwise they inflate `has_apu` and suppress `has_discrete_amd`,
/// which gates the iGPU+dGPU collision fix. (Target -> product per LLVM
/// AMDGPUUsage.)
///
/// Two consumers rely on this, for different reasons:
///
/// - the doctor, to tell an integrated GPU from a discrete one when diagnosing
///   iGPU+dGPU device-visibility collisions;
/// - serve's VRAM reporting, because an APU has no private VRAM — see
///   `vram_capacity_is_meaningful` in the `rocm` binary.
///
/// This answers a question about the *part*, not about the host: a machine can
/// pair an APU with a discrete card, so a true verdict here does not mean every
/// GPU on the host is integrated.
pub fn gfx_is_apu_family(gfx: &str) -> bool {
    let g = gfx.to_lowercase();
    // gfx115x: every Strix part is an APU.
    if gfx_model_digit(&g, "gfx115").is_some() {
        return true;
    }
    // gfx110x: only gfx1103 and above are APUs; gfx1100/1101/1102 are discrete.
    if let Some(digit) = gfx_model_digit(&g, "gfx110") {
        return digit >= 3;
    }
    false
}

/// The model digit that follows `prefix` in a gfx target, e.g. `3` from
/// `gfx1103` given prefix `gfx110`. Returns `None` when `gfx` does not start
/// with `prefix` or has no digit there. Any trailing feature suffix (such as
/// `:sramecc+:xnack-`) is ignored, matching how gcnArchName can be reported.
fn gfx_model_digit(gfx: &str, prefix: &str) -> Option<u32> {
    gfx.strip_prefix(prefix)?.chars().next()?.to_digit(10)
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

/// Last-resort GPU discovery from sysfs, for hosts where neither `lspci` nor
/// `rocminfo` is reachable.
///
/// Both preceding probes shell out, so on a machine with no `lspci` installed
/// and ROCm's `bin` off PATH they enumerate nothing and `has_amd_gpu` comes back
/// false — on a machine that plainly has a GPU. That is not hypothetical: it is
/// what the MI300X runner reports, and it is why the e2e harness reads the human
/// text form rather than this one (`capability.rs`). The human report never had
/// the problem because it asks the kernel directly, via
/// [`crate::detect_linux_sysfs_gfx_target`].
///
/// So ask the kernel here too, but only as a fallback: `lspci` carries PCI ids,
/// vendor strings and the APU/discrete distinction that sysfs does not, and
/// those are worth keeping whenever they are available.
fn probe_gpus_sysfs_fallback(e: &mut Examination) {
    if e.gpus.iter().any(|gpu| gpu.is_amd) {
        return;
    }
    let Some(gfx_target) = crate::detect_linux_sysfs_gfx_target() else {
        return;
    };
    // `is_apu` is left unset rather than guessed: sysfs gives the target, not
    // the packaging, and `summarise_gpu_categories` reads `Some(true)` /
    // `Some(false)` to populate has_apu / has_discrete_amd. Claiming either
    // would be inventing a fact, so both stay false and only has_amd_gpu moves.
    e.gpus.push(Gpu {
        name: "AMD GPU (from kernel topology)".to_owned(),
        gfx_target,
        is_amd: true,
        is_apu: None,
        ..Gpu::default()
    });
    e.notes.push(
        "GPU discovered from the kernel topology because neither lspci nor rocminfo was \
         available; PCI id and marketing name are unknown."
            .to_owned(),
    );
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
    // Shared with the human `rocm examine` report and the fix-6 PATH runner so
    // all three agree on which install is the active one. Handles versioned
    // roots (`/opt/rocm-6.4.1`) as well as the conventional `/opt/rocm`.
    let install = crate::discover_rocm_installs().into_iter().next();
    let rocm_dir = install
        .as_ref()
        .map(|install| install.path.to_string_lossy().into_owned())
        .unwrap_or_default();
    e.rocm_path = rocm_dir.clone();
    e.rocm_version = install
        .and_then(|install| install.version)
        .unwrap_or_default();

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
pub(crate) fn extract_rocm_version(path: &str) -> Option<String> {
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
        let value = if matches!(*key, "PATH" | "LD_LIBRARY_PATH") {
            truncate_to_chars(value, ENV_VALUE_MAX_CHARS)
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

/// Truncate `value` to at most `max_chars` characters, appending a marker when
/// truncated. Slices on char boundaries (matching Python's `value[:n]`); a byte
/// slice would panic when the cut lands inside a multibyte character.
fn truncate_to_chars(value: String, max_chars: usize) -> String {
    if value.chars().count() > max_chars {
        let truncated: String = value.chars().take(max_chars).collect();
        format!("{truncated}...[truncated]")
    } else {
        value
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
    // `$HIP_PATH` still wins: it names the SDK specifically, where the resolver
    // answers the broader "which ROCm installs are on this machine".
    let mut root = std::env::var("HIP_PATH").unwrap_or_default();
    if root.is_empty() || !Path::new(&root).is_dir() {
        // Was a hand-rolled scan of one directory with `versions.sort()`, so
        // `6.2` outranked `6.10` and any directory whose name looked plausible
        // counted as an install. The shared resolver orders by numeric
        // component, searches the same roots as everything else, honours
        // `$ROCM_PATH`, and requires an install marker before believing a
        // directory.
        root = crate::discover_rocm_installs()
            .into_iter()
            .next()
            .map(|install| install.path.to_string_lossy().into_owned())
            .unwrap_or_default();
    }
    if root.is_empty() || !Path::new(&root).is_dir() {
        return;
    }
    e.hip_sdk_path = root.clone();
    // Prefer what the install says about itself; the directory name is only a
    // fallback for a `$HIP_PATH` pointing somewhere unversioned.
    e.hip_sdk_version = crate::rocm_install_version(Path::new(&root)).unwrap_or_else(|| {
        Path::new(&root)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    });

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

    /// Ported from the Python preflight this replaced, whose `wsl.exe` parser
    /// was covered by a self-test that CI ran on both lanes. That coverage has to
    /// land here, or deleting the script quietly drops it.
    #[test]
    fn the_distro_list_survives_the_markers_wsl_puts_around_it() {
        // `-l -q` prints one bare name per line, which is what the probe asks
        // for.
        assert_eq!(
            parse_wsl_distro_list("Ubuntu\nDebian\n"),
            vec!["Ubuntu", "Debian"]
        );
        assert_eq!(
            parse_wsl_distro_list("Ubuntu-24.04\n"),
            vec!["Ubuntu-24.04"]
        );

        // A name may contain spaces: `wsl --import "My Distro"` is legal.
        // Splitting on whitespace truncated it to "My", which then neither
        // matched what the user asked for nor named a real distribution when
        // handed back to `wsl.exe -d`.
        assert_eq!(
            parse_wsl_distro_list("My Distro\nUbuntu\n"),
            vec!["My Distro", "Ubuntu"]
        );

        // The NUL padding of the raw UTF-16 output must not become part of a
        // name, and blank lines are not distributions.
        assert_eq!(
            parse_wsl_distro_list("\0U\0b\0u\0n\0t\0u\0\n\0"),
            vec!["Ubuntu"]
        );
        assert!(parse_wsl_distro_list("").is_empty());
        assert!(parse_wsl_distro_list("\n\n  \n").is_empty());

        // Tolerance only: `-q` emits no header, but a `-l -v` header must never
        // come back as a distribution named "NAME".
        assert!(parse_wsl_distro_list("  NAME   STATE   VERSION\n").is_empty());
    }

    #[test]
    fn a_host_that_could_not_be_queried_is_unknown_not_driverless() {
        use crate::WslHostDriverProbe;

        // The distinction the catalog acts on. An earlier version flattened this
        // to `Option<String>` and defaulted the `None`, so "could not ask the
        // host" arrived as `Some("")` -- which reads as "the host has no AMD
        // adapter" and reported a missing driver on a machine never looked at.
        assert_eq!(
            host_driver_fields(WslHostDriverProbe::Unreachable),
            (false, None),
            "unreachable must stay unknown"
        );
        assert_eq!(
            host_driver_fields(WslHostDriverProbe::NoAmdDisplay),
            (true, Some(String::new())),
            "answered with no adapter is evidence, and is not the same thing"
        );
        assert_eq!(
            host_driver_fields(WslHostDriverProbe::Version("32.0.1".to_owned())),
            (true, Some("32.0.1".to_owned()))
        );
    }

    #[test]
    fn the_distro_list_decodes_as_utf16_whatever_script_it_is_in() {
        fn utf16le(text: &str, bom: bool) -> Vec<u8> {
            let mut bytes = if bom { vec![0xFF, 0xFE] } else { Vec::new() };
            for unit in text.encode_utf16() {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            bytes
        }

        assert_eq!(decode_utf16le(&utf16le("Ubuntu\n", true)), "Ubuntu\n");
        assert_eq!(decode_utf16le(&utf16le("Ubuntu\n", false)), "Ubuntu\n");
        assert_eq!(decode_utf16le(b""), "");

        // The reason this is decoded by declaration rather than sniffed: in a
        // Latin or Cyrillic script every UTF-16LE byte is below 0x80, so the
        // bytes are *valid UTF-8* and decode without error into mojibake. Read
        // as UTF-8 these names come back as "#\u{4}1\u{4}..." rather than
        // failing, so no validity or NUL-density test could catch them.
        for name in [
            "Ubuntu-24.04\nÉtat\n",
            "Ubuntu\nУбунту\n",
            "Ubuntu\n日本語\n",
        ] {
            assert_eq!(decode_utf16le(&utf16le(name, true)), name);
            assert_eq!(decode_utf16le(&utf16le(name, false)), name);
        }

        // A non-ASCII name must survive all the way through the parser.
        assert_eq!(
            parse_wsl_distro_list(&decode_utf16le(&utf16le("Ubuntu\nУбунту\n", true))),
            vec!["Ubuntu", "Убунту"]
        );

        // An odd trailing byte is dropped rather than panicking.
        let mut truncated = utf16le("Ubuntu", false);
        truncated.push(0x00);
        assert_eq!(decode_utf16le(&truncated), "Ubuntu");
    }

    #[test]
    fn command_output_that_is_not_utf8_is_kept_rather_than_dropped() {
        // `read_to_string` fails on invalid UTF-8, and the error was discarded --
        // so one stray byte emptied a whole capture, which every caller then read
        // as "the command printed nothing".
        let (rc, out, _) = run("printf", &["ok\\xffdone"], SHORT);
        if rc == 127 {
            return; // no `printf` binary on this host
        }
        assert!(
            out.starts_with("ok") && out.ends_with("done"),
            "the undecodable byte must not take the rest of the output with it: {out:?}"
        );
    }

    /// Ported from the same Python preflight, which owned this rule until the
    /// catalog took it over. Its version was well covered and its tests went with
    /// it, so the coverage has to live here or the floor becomes an untested
    /// constant.
    #[test]
    fn the_distro_floor_fails_closed_on_anything_it_cannot_read() {
        for supported in ["24.04", "24.10", "25.04", "26.04", "28.04"] {
            assert_eq!(
                distro_clears_wsl_floor("ubuntu", supported),
                Some(true),
                "ubuntu {supported} clears the floor"
            );
        }
        // 22.04 ships glibc 2.35, below the 2.38 / GLIBCXX_3.4.32 floor the
        // engines are linked against, so it cannot run them at all.
        assert_eq!(distro_clears_wsl_floor("ubuntu", "22.04"), Some(false));
        assert_eq!(distro_clears_wsl_floor("ubuntu", "20.04"), Some(false));

        // Unreadable is `None`, never `Some(true)`. Reporting a release nobody
        // could parse as supported is how a user ends up chasing a GPU fault
        // that is really a glibc floor -- but claiming it is too old would send
        // them to reinstall a perfectly good distro, so neither answer is safe.
        for unreadable in ["24.04.1", "unknown", "", "24", "24.x", "..", "24.04.1.2"] {
            assert_eq!(
                distro_clears_wsl_floor("ubuntu", unreadable),
                None,
                "{unreadable:?} cannot be read as a release"
            );
        }

        // Only Ubuntu carries a documented floor. Anything else abstains rather
        // than applying Ubuntu's numbering to a distro that does not share it --
        // Debian 12 is not "below 24.04".
        for other in ["debian", "fedora", "arch", ""] {
            assert_eq!(distro_clears_wsl_floor(other, "12.0"), None, "{other}");
        }
        assert_eq!(distro_clears_wsl_floor("UBUNTU", "24.04"), Some(true));
    }

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
        // The field set examine.py emits, plus the CLI-only `status` and `wsl`
        // additions. diagnose.py reads against these names, so this is the frozen
        // wire contract — adding/removing/renaming a top-level field is a
        // contract change and must be intentional.
        //
        // `wsl` is one of the intentional ones: WSL2 has no examine.py analogue,
        // and nesting its facts under a single key keeps the rest of the contract
        // byte-identical instead of scattering ten flat `wsl_*` fields through it.
        let expected: std::collections::BTreeSet<&str> = [
            "os_family",
            "os_version",
            "distro_id",
            "distro_version",
            "kernel_release",
            "kernel_cmdline",
            "is_wsl",
            "wsl",
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
            "status",
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
    fn asking_to_skip_the_frameworks_is_recorded_as_skipped() {
        // Host-independent on purpose: every other variant depends on what is
        // installed, so this is the one the CLI flag can be pinned against
        // anywhere. "skipped" is a distinct answer from "unknown" -- the latter
        // means the probe ran and found nothing.
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::Skip);
        assert_eq!(e.framework, "skipped");
    }

    #[test]
    fn a_skipped_framework_probe_runs_no_interpreter() {
        // The reason to offer `skip` at all: the other variants start Python to
        // read the framework's ROCm build. Nothing else on the Examination may
        // move, or "skip" would be quietly doing work.
        let mut e = Examination::default();
        probe_framework(&mut e, FrameworkProbe::Skip);
        assert!(e.framework_version.is_empty());
        assert!(e.framework_rocm_version.is_empty());
        assert!(e.framework_arch_list.is_empty());
        assert!(e.framework_notes.is_empty());
    }

    #[test]
    fn the_sysfs_fallback_leaves_a_real_pci_enumeration_alone() {
        // lspci carries the PCI id, the marketing name and the APU/discrete
        // distinction; the topology read carries none of those. So the fallback
        // must only fill a gap, never overwrite a richer answer.
        let mut e = Examination {
            os_family: "linux".to_owned(),
            gpus: vec![Gpu {
                name: "Instinct MI300X".to_owned(),
                gfx_target: "gfx942".to_owned(),
                pci_id: "1002:74a1".to_owned(),
                is_amd: true,
                is_apu: Some(false),
            }],
            ..Examination::default()
        };
        probe_gpus_sysfs_fallback(&mut e);
        assert_eq!(e.gpus.len(), 1, "the fallback must not add a second entry");
        assert_eq!(e.gpus[0].pci_id, "1002:74a1");
        assert!(
            e.notes.is_empty(),
            "a no-op fallback should not annotate the report"
        );
    }

    #[test]
    fn a_topology_sourced_gpu_counts_as_present_without_claiming_a_package() {
        // What the fallback produces: enough to stop reporting "no AMD GPU",
        // and no more. `is_apu: None` is deliberate -- the topology says which
        // target, not whether it is integrated -- so the APU and discrete
        // tallies must both stay false rather than guess.
        let mut e = Examination {
            os_family: "linux".to_owned(),
            gpus: vec![Gpu {
                name: "AMD GPU (from kernel topology)".to_owned(),
                gfx_target: "gfx942".to_owned(),
                is_amd: true,
                is_apu: None,
                ..Gpu::default()
            }],
            ..Examination::default()
        };
        summarise_gpu_categories(&mut e);
        assert!(e.has_amd_gpu, "the machine has a GPU and must say so");
        assert!(!e.has_apu);
        assert!(!e.has_discrete_amd);
        assert_eq!(e.compute_status(), "ok");
    }

    #[test]
    fn status_reflects_scope_precedence() {
        let mut e = Examination {
            os_family: "linux".to_owned(),
            has_amd_gpu: true,
            ..Examination::default()
        };
        assert_eq!(e.compute_status(), "ok");
        e.probe_failures.push("lspci missing".to_owned());
        assert_eq!(e.compute_status(), "degraded");
        e.probe_failures.clear();
        e.has_amd_gpu = false;
        assert_eq!(e.compute_status(), "no-amd-gpu");
        e.os_family = "other".to_owned();
        assert_eq!(e.compute_status(), "unsupported-os");
        e.is_wsl = true;
        assert_eq!(e.compute_status(), "wsl");
    }

    #[test]
    fn env_truncation_is_char_safe_across_multibyte_boundary() {
        // 5000 'é' chars = 10000 bytes; byte 4000 lands inside a char, which a
        // byte slice would panic on. Must truncate cleanly to 4000 chars.
        let value = "é".repeat(5000);
        let out = truncate_to_chars(value, 4000);
        assert!(out.ends_with("...[truncated]"));
        assert_eq!(out.chars().filter(|&c| c == 'é').count(), 4000);
    }

    #[test]
    fn env_truncation_leaves_short_values_untouched() {
        assert_eq!(truncate_to_chars("short".to_owned(), 4000), "short");
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
    fn gfx_apu_family_neighboring_contract() {
        // gfx110x straddles two silicon families: gfx1100/gfx1101/gfx1102 are
        // discrete RDNA3 parts (Navi 31/32/33 -> Radeon RX 7900/7800/7600),
        // while gfx1103 (Phoenix / Hawk Point) is an APU. gfx115x (Strix Point
        // / Strix Halo) are APUs. Target->product mapping per LLVM AMDGPUUsage.
        //
        // The discrete neighbors must NOT be classified as an APU family: they
        // drive `has_discrete_amd`, which gates the iGPU+dGPU collision fix.
        assert!(!gfx_is_apu_family("gfx1100"));
        assert!(!gfx_is_apu_family("gfx1101"));
        assert!(!gfx_is_apu_family("gfx1102"));
        // Integrated APUs (every gfx115x part plus gfx1103+).
        assert!(gfx_is_apu_family("gfx1103"));
        assert!(gfx_is_apu_family("gfx1150"));
        assert!(gfx_is_apu_family("gfx1151"));
        assert!(gfx_is_apu_family("gfx1152"));
        assert!(gfx_is_apu_family("gfx1153"));
        // Unrelated families are never APUs.
        assert!(!gfx_is_apu_family("gfx1200"));
        assert!(!gfx_is_apu_family("gfx942"));
        // A trailing gcnArchName feature suffix must not change the verdict.
        assert!(gfx_is_apu_family("gfx1103:sramecc+:xnack-"));
        assert!(!gfx_is_apu_family("gfx1100:xnack-"));
        // Degenerate inputs never match.
        assert!(!gfx_is_apu_family("gfx110"));
        assert!(!gfx_is_apu_family(""));
    }

    #[test]
    fn apu_plus_discrete_neighbor_sets_both_category_flags() {
        // A machine with a real APU (gfx1103 Phoenix) and its discrete RDNA3
        // neighbor (gfx1100 Navi 31) must report BOTH an APU and a discrete
        // AMD GPU. Before the gfx110x carve-out, gfx1100 was tagged an APU
        // too, so `has_discrete_amd` stayed false and the iGPU+dGPU collision
        // fix was silently suppressed for exactly this pairing.
        let mut e = Examination {
            gpus: vec![
                Gpu {
                    gfx_target: "gfx1103".to_owned(),
                    is_amd: true,
                    is_apu: Some(gfx_is_apu_family("gfx1103")),
                    ..Gpu::default()
                },
                Gpu {
                    gfx_target: "gfx1100".to_owned(),
                    is_amd: true,
                    is_apu: Some(gfx_is_apu_family("gfx1100")),
                    ..Gpu::default()
                },
            ],
            ..Examination::default()
        };
        summarise_gpu_categories(&mut e);
        assert!(e.has_apu, "gfx1103 APU should set has_apu");
        assert!(
            e.has_discrete_amd,
            "gfx1100 dGPU should set has_discrete_amd"
        );
    }

    #[test]
    fn lone_discrete_rdna3_is_not_reported_as_apu() {
        // A box with only a gfx1100 discrete card must not claim to have an APU.
        let mut e = Examination {
            gpus: vec![Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                is_apu: Some(gfx_is_apu_family("gfx1100")),
                ..Gpu::default()
            }],
            ..Examination::default()
        };
        summarise_gpu_categories(&mut e);
        assert!(!e.has_apu, "a lone gfx1100 dGPU must not set has_apu");
        assert!(e.has_discrete_amd);
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

    #[test]
    fn os_version_carries_a_version_not_the_os_name() {
        // This field used to hold `std::env::consts::OS`, so it repeated
        // `os_family` and told a reader of the report nothing.
        let mut e = Examination::default();
        probe_os(&mut e);

        // Holds on every platform: a real version on the supported ones, and
        // empty on the rest -- neither of which is the OS name.
        assert_ne!(
            e.os_version,
            std::env::consts::OS,
            "os_version must not merely repeat the OS name"
        );

        if runtime_is_linux() || runtime_is_windows() {
            assert!(
                !e.os_version.is_empty(),
                "a supported host must report an OS version"
            );
            assert_ne!(e.os_version, e.os_family);
        } else {
            // Blank by choice on a host the CLI does not support, rather than
            // a guess the diagnosis catalog would then reason over.
            assert!(
                e.os_version.is_empty(),
                "unsupported hosts report no version, got {:?}",
                e.os_version
            );
        }
    }
}
