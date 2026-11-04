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
//! llama.cpp), `host_env` (OS, CPU, environment, shared memory, container and
//! kernel log), `wsl` (the WSL probe and the probe of a named distribution) and
//! `windows` (the native Windows probes).

use crate::FrameworkInterpreter;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod framework;
mod host_env;
mod library_copies;
mod linux_driver;
mod linux_gpus;
mod rocm_install_probe;
mod windows;
mod wsl;

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
    GpuProbeSources, probe_gpus_after_lspci, probe_gpus_lspci, summarise_gpu_categories,
};
use rocm_install_probe::probe_rocm_install;
pub(crate) use windows::parse_dotted_version;
use windows::{
    probe_adrenalin_windows, probe_gpus_windows, probe_hip_sdk_windows, probe_msvc_redist_windows,
};
use wsl::probe_wsl;
pub use wsl::{parse_wsl_distro_list, probe_wsl_distro_from_host};

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
    /// amdgpu is present: listed as a module, or registered as a PCI driver
    /// (`/sys/bus/pci/drivers/amdgpu`), which it is whether amdgpu is loaded
    /// as a module or built into the kernel (where it never appears in
    /// `lsmod`), or kfd is present on the host (kfd is linked into amdgpu).
    ///
    /// This is the driver being present, not a GPU being bound to it: a
    /// built-in driver registers at boot with or without a matching GPU, as a
    /// loaded module does when `modprobe`d on any machine. Whether there is an
    /// AMD GPU is `has_amd_gpu`. `diagnose` reads only `Some(false)`, as a
    /// module that failed to load (DKMS, a blacklist, Secure Boot), none of
    /// which can stop a driver that is present.
    pub amdgpu_loaded: Option<bool>,
    pub amdgpu_blacklisted_in: Vec<String>,
    /// kfd is available: as a separate `amdkfd` module on kernels before 4.20
    /// (where kfd was merged into `amdgpu.ko`), or through `/sys/class/kfd/kfd`
    /// or `/dev/kfd` on current ones. The name predates the merge.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

/// Property tests for the driver-side text readers above; a child module so it
/// can reach them while they stay private.
#[cfg(test)]
#[path = "examine_driver_proptests.rs"]
mod driver_proptests;
