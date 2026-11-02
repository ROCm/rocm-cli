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
//!
//! This file holds the document types and `Examination::probe`, which runs the
//! probes in order. The probes live in child modules: `linux_gpus` (Linux GPU
//! enumeration), `linux_driver` (kernel modules, device nodes, Secure Boot),
//! `rocm_install_probe` (how ROCm was installed, and the repository files),
//! `library_copies` (the loader library-copy search), `framework` (PyTorch and
//! llama.cpp) and `host_env` (OS, CPU, environment, shared memory, container
//! and kernel log).

use crate::FrameworkInterpreter;
use crate::probe_exec::{MEDIUM, run, run_utf16le, which};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

mod framework;
mod host_env;
mod library_copies;
mod linux_driver;
mod linux_gpus;
mod rocm_install_probe;

use framework::probe_framework;
use host_env::{
    probe_container, probe_cpu_linux, probe_cpu_windows, probe_dmesg_amdgpu, probe_env, probe_os,
    probe_shared_memory,
};
pub(crate) use library_copies::comgr_matches_runtime;
use library_copies::probe_comgr;
use linux_driver::{probe_devices, probe_modules, probe_secure_boot, probe_user};
pub use linux_gpus::gfx_is_apu_family;
use linux_gpus::{
    GpuProbeSources, classify_amd_marketing_name, probe_gpus_after_lspci, probe_gpus_lspci,
    summarise_gpu_categories,
};
use rocm_install_probe::probe_rocm_install;

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
///
/// `path` is always the logical host path (`/dev/kfd`), as `rocm examine
/// --json` reports it. When a test build re-roots hardware probes (see
/// [`crate::host_path`]), the node is read under that root, but the root never
/// appears here.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Device {
    /// The logical host path of the node, never the re-rooted one.
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

/// One copy of a ROCm library found on the machine.
///
/// Used for both the code object manager and the HIP runtime, because the
/// question asked of them is the same one: of the copies on this machine, which
/// would load, and which installation does it belong to.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LibraryCopy {
    /// The path as found, before symlinks are resolved -- this is the name the
    /// loader would use, and the one a user will recognise.
    pub path: String,
    /// The file the path resolves to. Two entries naming one file through a
    /// symlink are collapsed on this, so a versioned library and its unversioned
    /// alias are not reported as two copies.
    pub real_path: String,
    /// Version read from the resolved file name, empty when it carries none.
    ///
    /// Empty is an honest answer. The version is read from the name rather than
    /// by loading the library, so a renamed file yields nothing -- which is
    /// better than a confident wrong answer.
    pub version: String,
    /// Where the search found it: `active-runtime`, `ld-library-path`,
    /// `loader-cache`, `rocm-install`, or `managed-runtime`.
    pub source: String,
    /// The installation this copy belongs to, empty when no known installation
    /// claims it.
    ///
    /// Matched against the known installation roots rather than derived from the
    /// path, and this is the detail the whole diagnosis turns on. A managed
    /// runtime spreads its libraries across several `_rocm_sdk_*` directories;
    /// deriving a root by walking up from `.../_rocm_sdk_core/lib` would make
    /// each of them look like a separate installation, and the entry would then
    /// report a conflict on every healthy managed install -- the exact false
    /// report this design exists to avoid.
    pub install_root: String,
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
    /// `/etc/os-release` `ID_LIKE`, for `openmpi::resolve_package_manager`
    /// fallback matching. Not part of the examine.py wire contract, but
    /// included in JSON anyway (like the `comgr_*`/`hip_*` fields below):
    /// `rocm remote doctor` diagnoses a remote host by deserializing its
    /// `examine --json` output, and `repo_pkg_family`'s fallback needs this
    /// field populated on that deserialized value, not just on a freshly
    /// probed local `Examination`.
    ///
    /// `#[serde(default)]`: a remote on an older CLI version won't have
    /// printed this field at all, and `parse_examination` must keep accepting
    /// that document rather than failing outright over one missing field it
    /// can safely fall back to "" (unknown) for.
    #[serde(default)]
    pub distro_id_like: String,
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

    // Code object manager (Linux). HIP compiles device code at run time through
    // `libamd_comgr`, and a machine can hold more than one copy -- a system ROCm
    // install and a ROCm Python wheel each ship one. Holding two is not a fault;
    // every managed environment this CLI creates holds one, and there the wheel
    // copy is the right copy to load. What matters is which copy the loader
    // picks, and that was invisible: nothing looked past the first match.
    // `serde(default)` on all three, because this structure is read back from
    // *another machine*: `rocm remote doctor` deserializes an examination the
    // remote's own CLI produced, and that CLI may predate these fields. Without
    // a default, adding a field here refuses every remote running an older
    // build -- reported to the user as "the remote CLI is probably a different
    // version", which is true and useless, since the older CLI is the one that
    // cannot be changed.
    /// Every copy found, in an estimated loader search order. Not every tier is
    /// one the loader actually consults -- see [`Self::comgr_selected`].
    #[serde(default)]
    pub comgr_paths: Vec<LibraryCopy>,
    /// The copy that would load: the first entry in `comgr_paths` found on the
    /// `active-runtime`, `ld-library-path` or `loader-cache` tier. `None` when
    /// no copy was found on one of those tiers, even when `comgr_paths` is not
    /// empty -- a `rocm-install` or `managed-runtime` hit is evidence a copy
    /// exists, not evidence the loader would pick it.
    #[serde(default)]
    pub comgr_selected: Option<LibraryCopy>,
    /// Version of the selected copy; empty when it cannot be read from the name.
    #[serde(default)]
    pub comgr_version: String,
    /// Whether the selected code object manager copy belongs to the same
    /// installation as the selected HIP runtime.
    ///
    /// Computed by [`comgr_matches_runtime`], the same function
    /// `check_18_comgr_conflict` in `diagnose.rs` uses to decide whether to
    /// report a conflict -- so this field and that finding cannot disagree
    /// about one machine.
    ///
    /// `None` when there is not enough evidence to call it either way: either
    /// library missing, an unattributed copy on either side, or a runtime whose
    /// own installation ships no code object manager at all to compare against.
    #[serde(default)]
    pub comgr_matches_runtime: Option<bool>,

    /// Every copy of the HIP runtime found, in an estimated loader search
    /// order. Not every tier is one the loader actually consults -- see
    /// [`Self::hip_selected`].
    ///
    /// Searched the same way, and for the same reason: deciding whether the code
    /// object manager belongs to the active runtime means knowing which runtime
    /// is active, and that is the same question about a different file.
    #[serde(default)]
    pub hip_paths: Vec<LibraryCopy>,
    /// The HIP runtime copy that would load: the first entry in `hip_paths`
    /// found on the `active-runtime`, `ld-library-path` or `loader-cache`
    /// tier. `None` when no copy was found on one of those tiers, even when
    /// `hip_paths` is not empty, for the same reason `comgr_selected` can be
    /// `None` while `comgr_paths` is not.
    #[serde(default)]
    pub hip_selected: Option<LibraryCopy>,

    // HIP SDK install (Windows)
    pub hip_sdk_path: String,
    pub hip_sdk_version: String,
    pub hipinfo_present: bool,
    pub hipinfo_status: String,
    pub adrenalin_version: String,
    /// The display driver's date as `yyyy-MM-dd`. Empty when it was not read
    /// or did not parse as a date; an empty date is never treated as old.
    #[serde(default)]
    pub adrenalin_date: String,
    pub msvc_redist_present: Option<bool>,
    /// File version of the Visual C++ runtime in `System32`, for example
    /// `14.29.30133.0`. When `msvcp140.dll` and `vcruntime140.dll` differ, the
    /// older of the two. Empty when neither could be read, and always empty off
    /// Windows. Presence alone does not say the runtime is new enough: binaries
    /// built with a recent toolset crash on a runtime older than 14.40.
    #[serde(default)]
    pub msvc_redist_version: String,

    // framework
    pub framework: String,
    pub framework_version: String,
    pub framework_rocm_version: String,
    pub framework_arch_list: Vec<String>,
    pub framework_notes: Vec<String>,
    /// Which interpreter answered: `"managed-runtime"`, `"path"`, or `""` when
    /// no framework was probed.
    ///
    /// The versions themselves carry no provenance — `hip=7.2.53211` reads the
    /// same whether it came from a managed runtime or an ambient pip wheel — and
    /// the distinction decides whether comparing it against the *system* ROCm
    /// means anything. A managed runtime ships its own ROCm and never loads the
    /// system's, so for it the comparison is meaningless. See
    /// `check_8_wheel_rocm_mismatch` in `diagnose.rs`.
    pub framework_source: String,

    // environment
    pub env: BTreeMap<String, String>,

    // container
    pub in_container: bool,
    pub container_kind: String,

    // Shared memory (Linux). A serving workload needs gigabytes of `/dev/shm`;
    // a container gives it 64 MB by default. When it runs out the workload
    // crashes without the message ever naming shared memory, so the user has no
    // route from the error to the cause.
    /// Size of the `/dev/shm` filesystem in bytes.
    ///
    /// The total, not the free space, is what decides: a 64 MB allowance cannot
    /// hold an 8 GB workload even when completely empty, so judging on free
    /// space would miss the case on an idle machine entirely.
    ///
    /// `None` when the path does not exist or cannot be queried, which is a
    /// different answer from zero -- a machine we could not measure is not a
    /// machine with a shortage.
    pub shm_total_bytes: Option<u64>,
    /// Free space on `/dev/shm` in bytes. Reported because "64 MB total" and
    /// "64 MB total, 2 MB free" are different conversations.
    pub shm_available_bytes: Option<u64>,

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
            distro_id_like: String::new(),
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
            comgr_paths: Vec::new(),
            comgr_selected: None,
            comgr_version: String::new(),
            comgr_matches_runtime: None,
            hip_paths: Vec::new(),
            hip_selected: None,
            hip_sdk_path: String::new(),
            hip_sdk_version: String::new(),
            hipinfo_present: false,
            hipinfo_status: String::new(),
            adrenalin_version: String::new(),
            adrenalin_date: String::new(),
            msvc_redist_present: None,
            msvc_redist_version: String::new(),
            framework: "unknown".to_owned(),
            framework_version: String::new(),
            framework_rocm_version: String::new(),
            framework_arch_list: Vec::new(),
            framework_notes: Vec::new(),
            framework_source: String::new(),
            env: BTreeMap::new(),
            in_container: false,
            container_kind: String::new(),
            shm_total_bytes: None,
            shm_available_bytes: None,
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
    /// Record the AMD display driver from a read of the Windows device
    /// inventory, which chooses the adapter the same way `rocm examine` and
    /// `rocm install driver` do. Each command makes its own read. `None` leaves
    /// the fields empty, which no check treats as an old driver.
    pub fn set_windows_driver(&mut self, info: Option<&crate::windows_driver::WindowsDriverInfo>) {
        let Some(info) = info else {
            return;
        };
        self.adrenalin_version.clone_from(&info.version);
        self.adrenalin_date = info.date.map(|d| d.to_string()).unwrap_or_default();
    }

    /// Probe the host and return the examination. Never fails; probe errors are
    /// recorded in `probe_failures`/`notes` and the relevant fields are left at
    /// their defaults (matching `examine.py`'s degrade-gracefully behavior).
    #[must_use]
    pub fn probe(framework: FrameworkProbe) -> Self {
        Self::probe_with_interpreter(framework, None)
    }

    /// As [`Self::probe`], but running the framework probe under `interpreter`
    /// when one is given.
    ///
    /// The interpreter is injected rather than resolved here on purpose: which
    /// runtime is active is registry and config policy, and threading it in
    /// keeps that out of the host prober — and keeps the probe testable without
    /// standing up a fake `$HOME` *and* a real torch.
    #[must_use]
    pub fn probe_with_interpreter(
        framework: FrameworkProbe,
        interpreter: Option<&FrameworkInterpreter>,
    ) -> Self {
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
            probe_framework(&mut e, framework, interpreter);
            // WSL2 ships the same 64 MB default a container does, so this is one
            // of the platforms where the shortage is most likely to be real.
            probe_shared_memory(&mut e);
            // A wheel copy and a system copy collide on WSL2 exactly as they do
            // on bare metal, so skipping the search here would leave a WSL user
            // unable to see a conflict that is really there.
            probe_comgr(&mut e, interpreter);
            e.status = e.compute_status();
            return e;
        }
        if e.os_family == "linux" {
            probe_cpu_linux(&mut e);
            probe_gpus_lspci(&mut e);
            let kfd_nodes = crate::host_path("/sys/class/kfd/kfd/topology/nodes");
            probe_gpus_after_lspci(&mut e, GpuProbeSources::host(&kfd_nodes));
            summarise_gpu_categories(&mut e);
            probe_modules(&mut e);
            probe_user(&mut e);
            probe_devices(&mut e);
            probe_secure_boot(&mut e);
            probe_rocm_install(&mut e);
            probe_env(&mut e);
            // After both: the search consults `LD_LIBRARY_PATH` (read by
            // `probe_env`) and the resolved ROCm install (`probe_rocm_install`).
            probe_comgr(&mut e, interpreter);
            probe_container(&mut e);
            probe_shared_memory(&mut e);
            probe_dmesg_amdgpu(&mut e);
            probe_framework(&mut e, framework, interpreter);
        } else if e.os_family == "windows" {
            probe_cpu_windows(&mut e);
            probe_gpus_windows(&mut e);
            probe_hip_sdk_windows(&mut e);
            probe_adrenalin_windows(&mut e);
            probe_msvc_redist_windows(&mut e);
            summarise_gpu_categories(&mut e);
            probe_env(&mut e);
            probe_framework(&mut e, framework, interpreter);
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
// Platform probes
// ---------------------------------------------------------------------------

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
        wsl_lib_dir: crate::host_path("/usr/lib/wsl/lib").is_dir(),
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
///
/// The distro's `id`/`version` fields fall back from `/etc/os-release` to
/// `/usr/lib/os-release` per `os-release(5)`, matching every other reader in
/// this crate — this is the one that runs inside the distro over `wsl.exe`
/// rather than through [`crate::host_path`], so it needs its own copy of the
/// fallback rather than sharing `hardware_root::read_os_release`.
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
try_os_release() {
  if [ -e "$1" ]; then . "$1" 2>/dev/null
  elif [ -e "$2" ]; then . "$2" 2>/dev/null
  fi
}
try_os_release /etc/os-release /usr/lib/os-release
echo "id=${ID}"
echo "version=${VERSION_ID}"
"#;

/// The `try_os_release` shell function embedded in [`WSL_REMOTE_PROBE`]: sources
/// `$1`, falling back to `$2` per `os-release(5)` when `$1` does not exist — an
/// existence check (`[ -e ... ]`), not a readability one, so a permission-denied
/// primary is not silently replaced by a possibly stale fallback.
///
/// Duplicated as its own constant, with the paths left as parameters, so the
/// test below can run this exact function body against temp-directory paths
/// instead of the real `/etc`; [`try_os_release_fn_is_embedded_in_the_wsl_remote_probe`]
/// guards against the two drifting apart.
#[cfg(test)]
const TRY_OS_RELEASE_FN: &str = "try_os_release() {\n  if [ -e \"$1\" ]; then . \"$1\" 2>/dev/null\n  elif [ -e \"$2\" ]; then . \"$2\" 2>/dev/null\n  fi\n}";

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
    let selected = select_wsl_distro(distro, &distros)?;

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

/// Choose which WSL distribution to inspect, given the requested name (if
/// any) and the list of installed ones.
///
/// Pulled out of [`probe_wsl_distro_from_host`] so this refusal logic -- one
/// of the few genuinely host-independent, blocking checks in the WSL host
/// path -- can be exercised without `wsl.exe`, which the rest of that
/// function requires and which this crate's e2e coverage otherwise never
/// touches on a non-Windows test runner.
fn select_wsl_distro(distro: Option<&str>, distros: &[String]) -> Result<String, String> {
    match distro {
        Some(name) => {
            if !distros.iter().any(|d| d.eq_ignore_ascii_case(name)) {
                return Err(format!(
                    "no WSL distribution named '{name}'; found: {}",
                    distros.join(", ")
                ));
            }
            Ok(name.to_owned())
        }
        None => match distros {
            [] => Err("no WSL distributions were found".to_owned()),
            [only] => Ok(only.clone()),
            many => Err(format!(
                "several WSL distributions are installed; name one with --distro: {}",
                many.join(", ")
            )),
        },
    }
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
    parse_gpus_windows_output(&out, e);
}

/// Split out from [`probe_gpus_windows`] so the parsing can be driven with
/// synthetic `Win32_VideoController` output -- there is no Windows host to run
/// the real probe against in this test suite.
fn parse_gpus_windows_output(out: &str, e: &mut Examination) {
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
        // The PNP id first, the marketing name only as a fallback -- tried in
        // that order, not computed eagerly and discarded, since the name match
        // is wasted work whenever the id resolves.
        //
        // `Win32_VideoController` hands us `PCI\VEN_1002&DEV_1636&...` in the
        // same row as the name, and `host_gpu`'s PCI-id table answers for parts
        // the name table has no arm for -- every Vega iGPU among them. This
        // probe used to drop the id and ask the name alone, so a Renoir /
        // Lucienne / Cezanne / Barcelo laptop reported an empty `gfx_target`
        // unless a HIP SDK was installed for `hipInfo` to fill it in: exactly
        // the host that has no HIP SDK to ask. A device id is also the stronger
        // answer where both have one, being the ASIC rather than the brand.
        let (gfx_target, is_apu) = match crate::host_gpu::gfx_target_from_pnp_device_id(&pnp) {
            // `gfx_is_apu_family` knows the answer whenever the id resolved a
            // target; the name heuristic's `is_apu` guess is only consulted
            // when nothing identified the part.
            Some(target) => (target.to_owned(), gfx_is_apu_family(target)),
            None => classify_amd_marketing_name(&name),
        };
        e.gpus.push(Gpu {
            name,
            gfx_target,
            pci_id: pnp,
            is_apu: Some(is_apu),
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
            fill_unidentified_gpus_from_hipinfo(&mut e.gpus, &out);
        } else {
            e.hipinfo_status = format!("error rc={rc}");
        }
    } else {
        e.hipinfo_present = false;
        e.hipinfo_status = "missing".to_owned();
    }
}

/// Fill in the `gfx_target` of AMD GPUs the Windows probes could not identify,
/// from `hipInfo`'s `gcnArchName:` lines.
///
/// `hipInfo` prints one block per *visible* device and gives no stable handle
/// tying a block to a `Win32_VideoController` row, so the pairing is positional:
/// the Nth unidentified AMD GPU gets the Nth architecture line. That is the only
/// correspondence available, and it is a fallback — the PCI device id and the
/// marketing name both answer first, and better.
///
/// Positional pairing has to account for the GPUs that were already answered
/// for, which is the subtlety this function exists to hold. `hipInfo` still
/// prints a block for a card the PCI-id table resolved, but that card is no
/// longer an empty slot, so a naive "first empty slot" walk hands *its*
/// architecture to the next card along. The concrete case: a Radeon 680M
/// (resolved to `gfx1035` from its device id) beside an RX 6800M (in neither
/// table). `hipInfo` prints gfx1035 then gfx1031; the first line finds the 6800M
/// as the only empty slot and makes it a `gfx1035` APU. That inverts both
/// `has_apu` and `has_discrete_amd`, which gate `fix-9-igpu-dgpu`, and the
/// second line is then dropped — the discrete card ends up described as the
/// integrated one.
///
/// So an architecture already held by an identified GPU is consumed rather than
/// assigned: one line per holder, by count, not by presence. Counting matters —
/// two identical cards print the same name twice, and dropping every repeat
/// would leave the second one unidentified forever.
fn fill_unidentified_gpus_from_hipinfo(gpus: &mut [Gpu], hipinfo_stdout: &str) {
    // What the earlier, better probes already answered. Each of these has a
    // `gcnArchName` block of its own coming, which belongs to it.
    let mut already_identified: Vec<String> = gpus
        .iter()
        .filter(|g| g.is_amd && !g.gfx_target.is_empty())
        .map(|g| g.gfx_target.to_ascii_lowercase())
        .collect();

    for line in hipinfo_stdout.lines() {
        let Some(rest) = line.trim().strip_prefix("gcnArchName:") else {
            continue;
        };
        let Some(gfx) = crate::extract_first_gfx_token(rest) else {
            continue;
        };
        if let Some(position) = already_identified.iter().position(|held| *held == gfx) {
            already_identified.swap_remove(position);
            continue;
        }
        let Some(gpu) = gpus
            .iter_mut()
            .find(|g| g.is_amd && g.gfx_target.is_empty())
        else {
            continue;
        };
        gpu.is_apu = Some(gfx_is_apu_family(&gfx));
        gpu.gfx_target = gfx;
    }
}

/// Fill the driver fields from the same inventory and adapter choice that
/// `rocm diagnose` and `rocm install driver` use, so the three commands read one
/// fact.
fn probe_adrenalin_windows(e: &mut Examination) {
    e.set_windows_driver(crate::host_gpu::detect_windows_amd_driver_info().as_ref());
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

    // Presence says nothing about age. Read the version of the copies in
    // System32 only: a copy bundled with another app (Python, for example) can
    // be found on PATH and says nothing about the one Windows loads for
    // programs that do not ship their own.
    let script = "foreach ($n in 'msvcp140.dll','vcruntime140.dll') { $f = Join-Path $env:SystemRoot ('System32\\' + $n); if (Test-Path -LiteralPath $f) { (Get-Item -LiteralPath $f).VersionInfo.FileVersion } }";
    let (rc, out, _) = run("powershell", &["-NoProfile", "-Command", script], MEDIUM);
    if rc == 0 {
        e.msvc_redist_version = oldest_runtime_version(&out);
    }
}

/// Parse `14.29.30133.0` into `[14, 29, 30133, 0]`.
///
/// `None` unless every dot-separated part is a plain number, and unless there
/// are at least two parts. A caller must treat `None` as "unknown", never as
/// "old".
pub(crate) fn parse_dotted_version(text: &str) -> Option<Vec<u32>> {
    let parts: Option<Vec<u32>> = text
        .trim()
        .split('.')
        .map(|part| {
            // Digits only: `parse` alone would also accept a leading `+`.
            if part.bytes().all(|b| b.is_ascii_digit()) {
                part.parse::<u32>().ok()
            } else {
                None
            }
        })
        .collect();
    parts.filter(|p| p.len() >= 2)
}

/// The oldest parseable version among the lines of `output`, as written, or an
/// empty string when no line parses.
///
/// `FileVersion` is normally `14.29.30133.0`, but some builds append text after
/// it, so only the first word of a line is read.
fn oldest_runtime_version(output: &str) -> String {
    output
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter_map(|word| parse_dotted_version(word).map(|v| (v, word)))
        .min_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, word)| word.to_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe_exec::decode_utf16le;

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
    fn a_named_distro_that_does_not_exist_is_refused() {
        // This is the blocking refusal `probe_wsl_distro_from_host` returns
        // to the caller of `rocm diagnose --distro <name>` -- a plain error,
        // not a `continue-on-error` note the e2e suite can shrug off. It sat
        // behind `wsl.exe` and was untested outside a real Windows+WSL host,
        // which is not a lane this crate's unit tests run on.
        let distros = vec!["Ubuntu".to_owned(), "Debian".to_owned()];
        let err = select_wsl_distro(Some("Fedora"), &distros).expect_err("must be refused");
        assert!(err.contains("Fedora"), "names the distro asked for: {err}");
        assert!(err.contains("Ubuntu"), "lists what does exist: {err}");
        assert!(err.contains("Debian"), "lists what does exist: {err}");
    }

    #[test]
    fn a_named_distro_that_exists_is_selected_case_insensitively() {
        // Matched case-insensitively against the installed list, but the name
        // handed to `wsl.exe -d` afterwards is what the caller typed, not the
        // list's original casing -- pre-existing behavior, preserved as-is by
        // this extraction.
        let distros = vec!["Ubuntu-24.04".to_owned()];
        assert_eq!(
            select_wsl_distro(Some("ubuntu-24.04"), &distros).expect("must be selected"),
            "ubuntu-24.04"
        );
    }

    #[test]
    fn no_distro_named_falls_back_to_the_lone_one_or_refuses_ambiguity() {
        assert_eq!(
            select_wsl_distro(None, &["Ubuntu".to_owned()]).expect("the only one"),
            "Ubuntu"
        );
        assert!(select_wsl_distro(None, &[]).is_err(), "nothing installed");
        let many = vec!["Ubuntu".to_owned(), "Debian".to_owned()];
        let err = select_wsl_distro(None, &many).expect_err("ambiguous without --distro");
        assert!(
            err.contains("--distro"),
            "tells the user how to resolve it: {err}"
        );
    }

    #[test]
    fn sync_shared_fields_from_wsl_copies_the_actual_rocminfo_values() {
        // The fields the cross-platform PATH and wheel/ROCm checks read.
        // Asserting only that they are *set* would still pass if the sync
        // copied the wrong value or a hardcoded default -- assert the actual
        // values reached from each distinct `WslFacts` fixture instead.
        let mut missing = Examination {
            wsl: Some(WslFacts {
                rocminfo: false,
                rocm_sees_gpu: None,
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut missing);
        assert!(!missing.rocminfo_present);
        assert_eq!(missing.rocminfo_status, "missing");

        let mut healthy = Examination {
            wsl: Some(WslFacts {
                rocminfo: true,
                rocm_sees_gpu: Some(true),
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut healthy);
        assert!(healthy.rocminfo_present);
        assert_eq!(healthy.rocminfo_status, "ok");

        let mut no_agents = Examination {
            wsl: Some(WslFacts {
                rocminfo: true,
                rocm_sees_gpu: Some(false),
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut no_agents);
        assert!(no_agents.rocminfo_present);
        assert_eq!(no_agents.rocminfo_status, "no-agents");

        let mut unasked = Examination {
            wsl: Some(WslFacts {
                rocminfo: true,
                rocm_sees_gpu: None,
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut unasked);
        assert!(unasked.rocminfo_present);
        assert_eq!(unasked.rocminfo_status, "unknown");
    }

    #[test]
    fn probe_wsl_leaves_rocminfo_status_synced_not_at_its_uninitialized_default() {
        // `probe_wsl` never sets `rocminfo_status` itself -- only
        // `sync_shared_fields_from_wsl`, called at its very end, does. If that
        // call were ever removed, this field would stay at
        // `Examination::default()`'s empty string on every host, WSL or not,
        // regardless of whether rocminfo happens to be installed here -- so
        // this holds without depending on real host state, unlike a test that
        // compared against the actual probed rocminfo presence.
        let mut e = Examination::default();
        probe_wsl(&mut e);
        assert_ne!(
            e.rocminfo_status, "",
            "sync_shared_fields_from_wsl must have run and set a real status"
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

    pub(super) fn plant(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, b"").expect("plant library");
        path
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
        //
        // `framework_source` is another. examine.py only ever probes the ambient
        // interpreter, so it has no need to say which one answered; the CLI
        // prefers the active managed runtime's, and the version strings alone
        // cannot distinguish the two. Without it, `check_8_wheel_rocm_mismatch`
        // compares a managed runtime's HIP against the *system* ROCm — versions
        // that are free to differ on a perfectly healthy host — and tells the
        // user to reinstall torch.
        let expected: std::collections::BTreeSet<&str> = [
            "os_family",
            "os_version",
            "distro_id",
            "distro_version",
            // CLI addition beyond examine.py, added deliberately: `ID_LIKE`,
            // so `rocm remote doctor` can resolve a remote host's package
            // manager family from its deserialized `examine --json` output,
            // not only from a freshly probed local `Examination`.
            "distro_id_like",
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
            // CLI additions beyond examine.py, added deliberately: which copies
            // of the code object manager library the machine holds, and which
            // one would load. examine.py never looked, which is why a second
            // copy shadowing the first was invisible.
            "comgr_paths",
            "comgr_selected",
            "comgr_version",
            "comgr_matches_runtime",
            // The HIP runtime is searched the same way and for the same reason:
            // deciding whether the code object manager belongs to the active
            // runtime means knowing which runtime is active.
            "hip_paths",
            "hip_selected",
            "hip_sdk_path",
            "hip_sdk_version",
            "hipinfo_present",
            "hipinfo_status",
            "adrenalin_version",
            // CLI addition beyond examine.py, added deliberately: the driver
            // date, so an old display driver can be told from a current one.
            "adrenalin_date",
            "msvc_redist_present",
            "msvc_redist_version",
            "framework",
            "framework_version",
            "framework_rocm_version",
            "framework_arch_list",
            "framework_notes",
            "framework_source",
            "env",
            "in_container",
            "container_kind",
            // CLI additions beyond examine.py, added deliberately: the shared
            // memory allowance, which a serving workload exhausts without the
            // crash ever naming it.
            "shm_total_bytes",
            "shm_available_bytes",
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
    fn dotted_versions_parse_only_when_every_part_is_a_number() {
        assert_eq!(
            parse_dotted_version("14.29.30133.0"),
            Some(vec![14, 29, 30133, 0])
        );
        assert_eq!(parse_dotted_version(" 14.40 "), Some(vec![14, 40]));
        for bad in [
            "",
            "14",
            "x",
            "14.x",
            "14..1",
            "14.+1",
            "-1.2",
            "14.29 beta",
            "14.99999999999",
        ] {
            assert_eq!(parse_dotted_version(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_windows_driver_fills_version_and_date_and_none_changes_nothing() {
        let mut e = Examination::default();
        e.set_windows_driver(None);
        assert_eq!(
            (e.adrenalin_version.as_str(), e.adrenalin_date.as_str()),
            ("", "")
        );
        let info =
            crate::windows_driver::WindowsDriverInfo::from_probe("30.0.13014.8", "2021-08-04");
        e.set_windows_driver(info.as_ref());
        assert_eq!(e.adrenalin_version, "30.0.13014.8");
        assert_eq!(e.adrenalin_date, "2021-08-04");
        // An unreadable date stays empty, never reaching the check as text.
        let undated = crate::windows_driver::WindowsDriverInfo::from_probe("30.0.13014.8", "soon");
        let mut e = Examination::default();
        e.set_windows_driver(undated.as_ref());
        assert_eq!(e.adrenalin_version, "30.0.13014.8");
        assert_eq!(e.adrenalin_date, "");
    }

    #[test]
    fn the_oldest_runtime_version_is_recorded() {
        // Two files, the older one wins.
        assert_eq!(
            oldest_runtime_version("14.51.36231.0\r\n14.29.30133.0\r\n"),
            "14.29.30133.0"
        );
        assert_eq!(
            oldest_runtime_version("14.29.30133.0\n14.51.36231.0\n"),
            "14.29.30133.0"
        );
        // Compared as numbers, not text.
        assert_eq!(oldest_runtime_version("14.100.1.0\n14.9.1.0\n"), "14.9.1.0");
        // Text after the version is ignored; a line that does not parse is skipped.
        assert_eq!(
            oldest_runtime_version("14.44.35211.0 built by: vcwrkspc\nnot a version\n"),
            "14.44.35211.0"
        );
        // Nothing readable is "unknown", never a guess.
        assert_eq!(oldest_runtime_version(""), "");
        assert_eq!(oldest_runtime_version("error\n\n"), "");
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

    /// A wheel-format managed runtime, laid out the way the CLI installs one.
    ///
    /// `root` is one of the runtime's own `_rocm_sdk_*` package directories --
    /// `_rocm_sdk_devel`, matching what the probe script's
    /// `_devel.get_devel_root()` records as `root_path` when the `devel` extra
    /// is installed -- never a venv root above `site_packages`. The ROCm
    /// libraries do not sit under it either: they sit in the sibling
    /// `_rocm_sdk_core` package, beside `root`, both directly under
    /// `site_packages`. A fixture built as `root/lib/<python>/site-packages`
    /// (a venv layout the installer never produces) would pass against a shape
    /// production cannot create -- this one matches `apps/rocm/src/therock.rs`'s
    /// `ROCM_SDK_PROBE_SCRIPT` instead.
    #[cfg(target_os = "linux")]
    fn wheel_runtime_on_disk(tag: &str) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
        use std::fs;
        let base = std::env::temp_dir().join(format!(
            "rocm-comgr-wheel-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&base);
        let site_packages = base.join("site-packages");
        let root = site_packages.join("_rocm_sdk_devel");
        let sdk_lib = site_packages.join("_rocm_sdk_core").join("lib");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&sdk_lib).unwrap();
        fs::write(sdk_lib.join("libamd_comgr.so.3"), b"fake").unwrap();
        (root, Some(site_packages))
    }

    /// The managed copy this CLI installs itself is reachable by the search.
    ///
    /// That copy is the whole reason this entry exists: the CLI puts ROCm
    /// wheels into a managed environment, so a user who follows that path on a
    /// host already carrying a system install ends up holding both, having done
    /// nothing unusual. A search that cannot see the copy we put there reports
    /// a conflict-free machine no matter what else is true of it.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_managed_copy_this_cli_installs_is_reachable() {
        let (root, site_packages) = wheel_runtime_on_disk("found");
        let mut dirs = Vec::new();
        crate::managed_runtime::collect_managed_runtime_library_paths(
            &root,
            site_packages.as_deref(),
            &mut dirs,
        );

        let expected = site_packages
            .expect("fixture always records site_packages")
            .join("_rocm_sdk_core")
            .join("lib");
        assert!(
            dirs.contains(&expected),
            "the search missed the copy the CLI installs, which is the case this entry exists \
             for. Looked in: {dirs:?}"
        );
        let _ = std::fs::remove_dir_all(root.parent().and_then(|p| p.parent()).unwrap());
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
    fn windows_gpu_probe_classifies_a_vega_igpu_through_the_pci_table_not_the_name() {
        // ROCMAI-546: the real host this ticket was filed against. Windows
        // reports Renoir/Cezanne's Vega iGPU under the generic marketing name
        // "AMD Radeon(TM) Graphics", which `classify_amd_marketing_name` does
        // not recognize (it only maps Phoenix/Strix-and-newer names) -- so
        // without the PCI-id lookup this host's `gfx_target` stays empty and
        // `info-20-gpu-no-rocm-support` can never fire on it.
        let out = "AMD Radeon(TM) Graphics\t31.0.21004.7\tPCI\\VEN_1002&DEV_1636&SUBSYS_12341002&REV_C8\\4&1a2b3c4d&0&0008";
        let mut e = Examination::default();
        parse_gpus_windows_output(out, &mut e);
        assert_eq!(e.gpus.len(), 1, "{:?}", e.gpus);
        let gpu = &e.gpus[0];
        assert_eq!(gpu.gfx_target, "gfx90c", "{gpu:?}");
        assert!(gpu.is_amd);
        assert_eq!(gpu.is_apu, Some(true), "gfx90c is an APU family: {gpu:?}");
    }

    #[test]
    fn windows_gpu_probe_does_not_misclassify_a_rembrandt_apu_as_discrete() {
        // Regression: the PNP-id path resolves gfx1035 for a Rembrandt/
        // Barcelo-R laptop (PNP id 164d), but `gfx_is_apu_family` used to have
        // no gfx103x arm at all, so a real iGPU reported `is_apu: Some(false)`
        // -- the opposite of what the marketing-name path ("AMD Radeon 680M")
        // would have said via `APU_KEYWORDS`. That silently suppressed
        // `fix-9-igpu-dgpu` detection on exactly the iGPU+dGPU host shape this
        // PR is otherwise improving detection for.
        let out = "AMD Radeon 680M\t31.0.21012.9\tPCI\\VEN_1002&DEV_164D&SUBSYS_14F11002&REV_C8\\3&11583659&0&08";
        let mut e = Examination::default();
        parse_gpus_windows_output(out, &mut e);
        assert_eq!(e.gpus.len(), 1, "{:?}", e.gpus);
        let gpu = &e.gpus[0];
        assert_eq!(gpu.gfx_target, "gfx1035", "{gpu:?}");
        assert_eq!(
            gpu.is_apu,
            Some(true),
            "a Rembrandt/Barcelo-R iGPU must not report as discrete: {gpu:?}"
        );
    }

    #[test]
    fn windows_gpu_probe_prefers_the_pci_id_match_when_one_exists() {
        // A name `classify_amd_marketing_name` already recognizes must not be
        // asked at all when the PNP id already resolved a target -- the name
        // match is wasted work in that case, and if it ever disagreed with the
        // id, the id must still win (it names the ASIC, not the brand).
        //
        // The two inputs are made to DISAGREE on purpose, and that disagreement
        // is the whole point: PNP id `73a0` is Navi 21 (gfx1030, discrete),
        // while the name "AMD Radeon(TM) 8060S Graphics" is one of the handful
        // `classify_amd_marketing_name` answers for, as gfx1151 and an APU. No
        // such part ships -- which is exactly why it is the input. A pair the
        // two paths agree about passes with the PNP lookup deleted, silently
        // re-testing the marketing fallback that `marketing_name_maps_strix_halo`
        // already covers for this same string; only a pair that cannot both be
        // right pins *precedence* rather than *agreement*. Both halves of the
        // answer are asserted because the two paths disagree about `is_apu` as
        // well as about the target.
        let out = "AMD Radeon(TM) 8060S Graphics\t32.0.12033.1030\tPCI\\VEN_1002&DEV_73A0&SUBSYS_00000000&REV_C1\\3&11583659&0&08";
        let mut e = Examination::default();
        parse_gpus_windows_output(out, &mut e);
        assert_eq!(e.gpus.len(), 1, "{:?}", e.gpus);
        let gpu = &e.gpus[0];
        assert_eq!(
            gpu.gfx_target, "gfx1030",
            "the PNP id names the ASIC and has to win over the brand name: {gpu:?}"
        );
        assert_eq!(
            gpu.is_apu,
            Some(false),
            "`is_apu` has to follow the id's target too, not the name heuristic: {gpu:?}"
        );
    }

    /// The consequence the misclassification actually had, asserted where it
    /// lands rather than one step upstream of it.
    ///
    /// `fix-9-igpu-dgpu` is gated on `has_apu && has_discrete_amd`
    /// (`check_9_igpu_dgpu_collision`), and both are derived from `is_apu` by
    /// `summarise_gpu_categories`. A Rembrandt iGPU reported as discrete does
    /// not merely carry a wrong field: it turns the gate off on the one host
    /// shape the check exists for. Note the dGPU's marketing name carries no
    /// APU keyword either way, so nothing here rests on `APU_KEYWORDS`.
    #[test]
    fn a_rembrandt_laptop_with_a_discrete_card_opens_the_igpu_dgpu_gate() {
        let out = "AMD Radeon 680M\t31.0.21012.9\tPCI\\VEN_1002&DEV_164D&SUBSYS_14F11002&REV_C8\n\
                   AMD Radeon RX 6600M\t31.0.21012.9\tPCI\\VEN_1002&DEV_73E0&SUBSYS_14F11002&REV_C7";
        let mut e = Examination::default();
        parse_gpus_windows_output(out, &mut e);
        assert_eq!(e.gpus.len(), 2, "{:?}", e.gpus);
        assert_eq!(e.gpus[0].is_apu, Some(true), "{:?}", e.gpus[0]);
        assert_eq!(e.gpus[1].is_apu, Some(false), "{:?}", e.gpus[1]);
        summarise_gpu_categories(&mut e);
        assert!(
            e.has_apu && e.has_discrete_amd,
            "both gate fix-9-igpu-dgpu: has_apu={}, has_discrete_amd={}",
            e.has_apu,
            e.has_discrete_amd
        );
    }

    /// The `hipInfo` fallback must not hand an identified GPU's architecture to
    /// the card next to it.
    ///
    /// Same failure as `a_rembrandt_laptop_with_a_discrete_card_opens_the_igpu_dgpu_gate`
    /// and the same gate underneath it, reached by the other door. The pairing
    /// is positional, and the PCI-id lookup now fills some slots before
    /// `hipInfo` is asked — but `hipInfo` still prints a block for every device,
    /// including the filled ones. Walking to the "first empty slot" therefore
    /// gives the *iGPU's* architecture to the first card the tables could not
    /// identify.
    ///
    /// Concretely: a Radeon 680M resolved to gfx1035 from its device id, beside
    /// an RX 6800M that is in neither table. `hipInfo` prints gfx1035 then
    /// gfx1031. Unguarded, the 6800M becomes a gfx1035 APU, `has_discrete_amd`
    /// goes false, `has_apu` is the only flag left, and `fix-9-igpu-dgpu` is
    /// suppressed on the one host shape it exists for.
    #[test]
    fn hipinfo_does_not_give_an_identified_gpus_architecture_to_its_neighbour() {
        let mut gpus = vec![
            Gpu {
                name: "AMD Radeon(TM) Graphics".to_owned(),
                gfx_target: "gfx1035".to_owned(),
                is_amd: true,
                is_apu: Some(true),
                ..Gpu::default()
            },
            // In neither table: no device-id arm, and "RX 6800M" carries no APU
            // keyword, so it arrives here genuinely unidentified.
            Gpu {
                name: "AMD Radeon RX 6800M".to_owned(),
                is_amd: true,
                is_apu: Some(false),
                ..Gpu::default()
            },
        ];
        fill_unidentified_gpus_from_hipinfo(
            &mut gpus,
            "--------------------------------------------------------------------------------\n\
             device#                           0\n\
             Name:                             AMD Radeon(TM) Graphics\n\
             gcnArchName:                      gfx1035\n\
             device#                           1\n\
             Name:                             AMD Radeon RX 6800M\n\
             gcnArchName:                      gfx1031\n",
        );
        assert_eq!(
            gpus[0].gfx_target, "gfx1035",
            "the identified iGPU keeps what the device id said: {:?}",
            gpus[0]
        );
        assert_eq!(
            gpus[1].gfx_target, "gfx1031",
            "the discrete card must get its OWN architecture line, not the one \
             already spoken for by the iGPU: {:?}",
            gpus[1]
        );
        assert_eq!(
            gpus[1].is_apu,
            Some(false),
            "and must not be relabelled integrated, which is what turns \
             has_discrete_amd off: {:?}",
            gpus[1]
        );

        let mut e = Examination {
            gpus,
            ..Examination::default()
        };
        summarise_gpu_categories(&mut e);
        assert!(
            e.has_apu && e.has_discrete_amd,
            "both gate fix-9-igpu-dgpu: has_apu={}, has_discrete_amd={}",
            e.has_apu,
            e.has_discrete_amd
        );
    }

    /// Consuming by count and not by presence: two identical cards print the
    /// same architecture twice, and only one of them was identified.
    ///
    /// The cheap form of the guard above — "skip any line whose architecture
    /// some GPU already holds" — leaves the second card unidentified forever on
    /// this machine, which is the regression the guard must not introduce.
    #[test]
    fn hipinfo_still_identifies_a_second_card_of_the_same_architecture() {
        let mut gpus = vec![
            Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                is_apu: Some(false),
                ..Gpu::default()
            },
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
        ];
        fill_unidentified_gpus_from_hipinfo(
            &mut gpus,
            "gcnArchName:  gfx1100\ngcnArchName:  gfx1100\n",
        );
        assert_eq!(
            gpus[1].gfx_target, "gfx1100",
            "one line is consumed by the GPU that already holds that target; the \
             other still belongs to the card nothing has identified: {:?}",
            gpus[1]
        );
    }

    /// The probes themselves, not just `host_path`, answer from a simulated
    /// host root. Plants a bare-metal machine with one GPU, inside a Podman
    /// container, then runs the converted probes and asserts they describe it
    /// rather than the machine the test runs on. A probe that still names its
    /// absolute path directly reads the real host and fails here.
    ///
    /// The container is planted twice — first by Podman's marker, then by the
    /// cgroup alone with no marker — so a probe reading the real host's markers
    /// or cgroup answers differently whether the test runs bare, under Docker
    /// or under Podman. The GPU is a gfx1101. WSL is checked in both
    /// directions: first a non-WSL `/proc/version`, then a WSL kernel, then a
    /// `/dev/dxg` alone. The positive answers are the ones a bare-metal runner
    /// cannot give, so they catch an unrouted WSL read anywhere but on a real
    /// WSL host.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn converted_probes_read_the_simulated_host_root() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-host-root-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        let plant = |relative: &str, body: &str| {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().expect("planted path has a parent"))
                .expect("create planted directory");
            std::fs::write(path, body).expect("write planted file");
        };
        plant(
            "sys/class/kfd/kfd/topology/nodes/0/properties",
            "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\n",
        );
        plant(
            "sys/class/kfd/kfd/topology/nodes/1/properties",
            "cpu_cores_count 0\nsimd_count 120\ngfx_target_version 110001\n",
        );
        plant("dev/kfd", "");
        plant(
            "proc/version",
            "Linux version 6.8.0-generic (buildd@lcy02)\n",
        );
        plant("run/.containerenv", "");

        let (gpu_nodes, gfx_target, kfd, wsl, podman, cgroup_only) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);

            let gpu_nodes = crate::host_gpu::linux::linux_kfd_gpu_node_count();
            let gfx_target = crate::host_gpu::linux::detect_linux_kfd_gfx_target();
            let kfd = linux_driver::stat_device("/dev/kfd", "nobody", &[]);
            let wsl = crate::is_wsl_host();
            let mut podman = Examination::default();
            probe_container(&mut podman);

            // Without a marker file, only the cgroup can say "container".
            std::fs::remove_file(root.join("run/.containerenv")).expect("remove marker");
            plant("proc/1/cgroup", "0::/kubepods/besteffort/pod1234\n");
            let mut cgroup_only = Examination::default();
            probe_container(&mut cgroup_only);

            // Now make the same root read as WSL, one signal at a time. "Not
            // WSL" is what a bare-metal runner answers anyway, so only these
            // positive answers fail when a WSL read misses the root.
            plant(
                "proc/version",
                "Linux version 5.15.167.4-microsoft-standard-WSL2 (root@build)\n",
            );
            let wsl_by_kernel = crate::is_wsl_host();
            plant(
                "proc/version",
                "Linux version 6.8.0-generic (buildd@lcy02)\n",
            );
            plant("dev/dxg", "");
            let wsl_by_dxg = crate::is_wsl_host();

            (
                gpu_nodes,
                gfx_target,
                kfd,
                (wsl, wsl_by_kernel, wsl_by_dxg),
                podman,
                cgroup_only,
            )
        };
        std::fs::remove_dir_all(&root).ok();
        let (wsl, wsl_by_kernel, wsl_by_dxg) = wsl;

        assert_eq!(gpu_nodes, Some(1), "one GPU node in the planted topology");
        assert_eq!(gfx_target.as_deref(), Some("gfx1101"));
        assert!(kfd.exists, "the planted /dev/kfd is found");
        assert_eq!(kfd.path, "/dev/kfd", "the logical path is reported");
        assert!(!wsl, "the planted host is bare metal, not WSL");
        assert!(wsl_by_kernel, "a planted WSL kernel reads as WSL");
        assert!(wsl_by_dxg, "a planted /dev/dxg reads as WSL");
        assert!(podman.in_container);
        assert_eq!(podman.container_kind, "podman");
        assert!(cgroup_only.in_container);
        assert_eq!(cgroup_only.container_kind, "container");
    }

    /// Guards [`TRY_OS_RELEASE_FN`] against drifting from the copy embedded in
    /// [`WSL_REMOTE_PROBE`]: the test below exercises the standalone constant,
    /// so if a future edit touches one and not the other, this catches it
    /// rather than the test silently proving the wrong logic.
    #[test]
    fn try_os_release_fn_is_embedded_in_the_wsl_remote_probe() {
        assert!(
            WSL_REMOTE_PROBE.contains(TRY_OS_RELEASE_FN),
            "WSL_REMOTE_PROBE's os-release handling has drifted from the tested function"
        );
    }

    /// The WSL remote probe's `try_os_release` shell function, run for real
    /// under `/bin/sh` against temp-directory paths rather than the real
    /// `/etc`: it sources the first path that exists, falling back to the
    /// second only when the first is absent, matching `os-release(5)` and the
    /// Rust-side `hardware_root::read_os_release`.
    #[cfg(target_os = "linux")]
    #[test]
    fn try_os_release_sources_the_fallback_only_when_the_primary_is_absent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let primary = dir.path().join("primary");
        let fallback = dir.path().join("fallback");
        std::fs::write(&fallback, "ID=fallback\n").expect("write fallback");

        let script = format!("{TRY_OS_RELEASE_FN}\ntry_os_release \"$1\" \"$2\"\necho \"id=$ID\"");
        let run = || {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(&script)
                .arg("sh")
                .arg(&primary)
                .arg(&fallback)
                .output()
                .expect("run sh")
        };

        let before = run();
        assert_eq!(
            String::from_utf8_lossy(&before.stdout).trim(),
            "id=fallback",
            "stderr: {}",
            String::from_utf8_lossy(&before.stderr)
        );

        std::fs::write(&primary, "ID=primary\n").expect("write primary");
        let after = run();
        assert_eq!(
            String::from_utf8_lossy(&after.stdout).trim(),
            "id=primary",
            "stderr: {}",
            String::from_utf8_lossy(&after.stderr)
        );
    }

    /// `[ -e "$1" ]` is an existence check, not a readability one: a primary
    /// that exists but cannot be sourced (here, a directory in its place, the
    /// same portable non-readable fixture the Rust-side test above uses)
    /// must not fall through to the `elif` and source the fallback — that
    /// would silently read a file that may not describe this host, matching
    /// `hardware_root::read_os_release`'s own non-absence rule.
    ///
    /// This is deliberately a directory, not a permission-denied file: CI
    /// commonly runs as root, where a `chmod 000` file is still readable, so
    /// a mode-000 fixture would pass or fail depending on who runs it. A
    /// directory fails to source for a different, root-safe reason ("is a
    /// directory"), which is enough to prove this one claim — existence is
    /// not readability — but is **not** the same failure mode as a genuinely
    /// unreadable primary. Under `/bin/sh` (`dash` on Debian/Ubuntu),
    /// sourcing a *permission-denied* file is a fatal shell-level error that
    /// aborts the whole embedded script immediately, not just this function —
    /// a materially different outcome than "`$ID` stays unset" and one this
    /// test does not exercise. That abort-on-unreadable-primary behavior
    /// predates the `/usr/lib/os-release` fallback this module adds; fixing
    /// it is out of scope here (tracked as a possible follow-up, not a
    /// blocker for the fallback itself).
    #[cfg(target_os = "linux")]
    #[test]
    fn try_os_release_does_not_source_the_fallback_when_the_primary_is_not_sourceable() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let primary = dir.path().join("primary");
        let fallback = dir.path().join("fallback");
        std::fs::create_dir(&primary).expect("create primary as a directory");
        std::fs::write(&fallback, "ID=fallback\n").expect("write fallback");

        let script = format!("{TRY_OS_RELEASE_FN}\ntry_os_release \"$1\" \"$2\"\necho \"id=$ID\"");
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .arg("sh")
            .arg(&primary)
            .arg(&fallback)
            .output()
            .expect("run sh");

        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "id=",
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// Property tests for the driver-side text readers above; a child module so it
/// can reach them while they stay private.
#[cfg(test)]
#[path = "examine_driver_proptests.rs"]
mod driver_proptests;
