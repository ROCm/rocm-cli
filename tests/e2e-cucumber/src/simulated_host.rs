// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Simulated hosts: a directory that stands in for `/` when the CLI asks what
//! hardware it is running on.
//!
//! An `e2e-test-hooks` build of `rocm` reads its hardware probes — the KFD
//! topology, `/dev/kfd`, `/dev/dxg`, `/sys/class/drm`, `/sys/module/amdgpu`,
//! `/proc/version`, `/etc/os-release` and the WSL plumbing under `/usr/lib/wsl`
//! — under `$ROCM_CLI_TEST_HOST_ROOT` instead of `/` (see
//! `rocm_core::host_path`). [`SimulatedHost::plant`] writes such a directory, so
//! a scenario can describe the machine it needs (eight Instinct GPUs, a WSL2
//! distribution with ROCm passthrough, a box with no GPU) and get the same
//! answer on a GitHub-hosted runner, a GPU box, or a developer's WSL2 laptop.
//!
//! The probes that run programs instead of reading files are covered by
//! controlling `PATH`. [`SimulatedHost::plant`] returns a `PATH` made of two
//! directories: the `fake-host-tool` binary linked under the name of every
//! tool the simulated machine has (it answers from the canned outputs this
//! module writes under [`TOOL_OUTPUT_DIR`]), and a filtered view of the real
//! `PATH` with every name in [`HOST_TOOLS`] removed. So the real machine's
//! `lspci`, `rocminfo` or Windows interop can never answer for the simulated
//! one, and a tool the simulated machine lacks is genuinely absent rather than
//! present-but-failing.
//!
//! The layout follows what the kernel really exposes — notably,
//! `gfx_target_version` is a line inside each KFD node's `properties` file, not
//! a file of its own. Reproducing a layout the kernel does not have would let
//! the product pass here and fail on hardware.

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

/// Directory under the root holding the canned tool outputs.
/// Must match `TOOL_OUTPUT_DIR` in `src/bin/fake-host-tool.rs`.
pub const TOOL_OUTPUT_DIR: &str = ".fake-tools";

/// The environment variable that points the CLI at a simulated root.
/// Must match `rocm_core::hardware_root::TEST_HOST_ROOT_ENV`.
pub const TEST_HOST_ROOT_ENV: &str = "ROCM_CLI_TEST_HOST_ROOT";

/// Every program the CLI runs to learn about the machine. None of them is
/// reachable from a simulated machine's `PATH` except through the stand-in, and
/// only when the simulated machine has it.
pub const HOST_TOOLS: &[&str] = &[
    "uname",
    "lsmod",
    "modinfo",
    "ldconfig",
    "lspci",
    "rocminfo",
    "rocm_agent_enumerator",
    "amd-smi",
    // OpenMPI's launcher: vLLM's dependency check wants it on PATH.
    "mpirun",
    // WSL interop: how a WSL2 distribution asks the Windows host about its
    // display adapter. On a real WSL2 machine these reach the real host.
    "powershell.exe",
    "powershell",
    "pwsh.exe",
    "cmd.exe",
    // Secure Boot state and the kernel log, which the diagnose catalog reads.
    "mokutil",
    "journalctl",
    "dmesg",
];

/// One AMD GPU, as the PCI bus and the kernel's KFD topology describe it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatedGpu {
    /// KFD's packed `gfx_target_version` (major·10000 + minor·100 + stepping):
    /// `90402` is gfx942.
    pub gfx_target_version: u32,
    /// PCI device id, e.g. `0x74a1` for MI300X.
    pub device_id: u16,
    /// The `lspci -nn` class, e.g. `Processing accelerators [1200]`.
    pub pci_class: &'static str,
    /// The `lspci` marketing name between the vendor and the `[vendor:device]`.
    pub pci_name: &'static str,
    /// KFD's `location_id`: the kernel's `(bus << 8) | devfn`.
    pub location_id: u32,
}

impl SimulatedGpu {
    /// An AMD Instinct MI300X on PCI bus `bus` (gfx942, a data-center part).
    #[must_use]
    pub const fn mi300x(bus: u8) -> Self {
        Self {
            gfx_target_version: 90402,
            device_id: 0x74a1,
            pci_class: "Processing accelerators [1200]",
            pci_name: "Aqua Vanjaram [Instinct MI300X]",
            location_id: (bus as u32) << 8,
        }
    }

    /// The integrated GPU of a Strix Halo APU (gfx1151).
    #[must_use]
    pub const fn strix_halo() -> Self {
        Self {
            gfx_target_version: 110_501,
            device_id: 0x1586,
            pci_class: "Display controller [0380]",
            pci_name: "Strix Halo [Radeon Graphics / Radeon 8050S / 8060S Graphics]",
            location_id: 0xc5 << 8,
        }
    }

    /// The gfx target the CLI should derive from this GPU, e.g. `gfx942`.
    #[must_use]
    pub fn gfx_target(&self) -> String {
        let v = self.gfx_target_version;
        format!("gfx{}{}{:x}", v / 10_000, (v / 100) % 100, v % 100)
    }

    /// The PCI address `lspci -D` prints for this GPU, e.g. `0000:11:00.0`.
    #[must_use]
    pub fn pci_address(&self) -> String {
        let bus = (self.location_id >> 8) & 0xff;
        let device = (self.location_id >> 3) & 0x1f;
        let function = self.location_id & 0x7;
        format!("0000:{bus:02x}:{device:02x}.{function}")
    }
}

/// How the GPUs reach the Linux distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Native Linux with the in-tree amdgpu driver: `/dev/kfd`, a KFD topology,
    /// DRM cards.
    BareMetal,
    /// A WSL2 distribution reaching the GPU through `/dev/dxg` and the Windows
    /// host driver. `rocdxg_ready` adds ROCDXG (the library and its linker-cache
    /// entry), without which ROCm cannot reach the device.
    Wsl2 { rocdxg_ready: bool },
}

/// A machine to simulate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatedHost {
    pub platform: Platform,
    pub gpus: Vec<SimulatedGpu>,
}

/// A WSL2 kernel string; the CLI recognises WSL by `microsoft-standard-WSL2`.
const WSL2_KERNEL: &str = "6.6.87.2-microsoft-standard-WSL2";
/// A stock Ubuntu kernel.
const BARE_METAL_KERNEL: &str = "6.8.0-60-generic";

impl SimulatedHost {
    /// Native Linux with `count` MI300X accelerators.
    #[must_use]
    pub fn instinct_mi300x(count: u8) -> Self {
        // The buses of a real eight-GPU MI300X host, in KFD node order.
        const BUSES: [u8; 8] = [0x11, 0x2f, 0x46, 0x5d, 0x8b, 0xaa, 0xc2, 0xda];
        assert!(
            (1..=8).contains(&count),
            "an MI300X host has one to eight GPUs, not {count}"
        );
        Self {
            platform: Platform::BareMetal,
            gpus: BUSES[..usize::from(count)]
                .iter()
                .map(|bus| SimulatedGpu::mi300x(*bus))
                .collect(),
        }
    }

    /// Native Linux with no AMD GPU at all.
    #[must_use]
    pub const fn bare_metal_without_gpu() -> Self {
        Self {
            platform: Platform::BareMetal,
            gpus: Vec::new(),
        }
    }

    /// A WSL2 distribution on a Strix Halo machine, with ROCm passthrough ready.
    #[must_use]
    pub fn wsl2_strix_halo() -> Self {
        Self {
            platform: Platform::Wsl2 { rocdxg_ready: true },
            gpus: vec![SimulatedGpu::strix_halo()],
        }
    }

    /// The gfx target of the first GPU, which is what the CLI reports as the
    /// host's target.
    #[must_use]
    pub fn gfx_target(&self) -> Option<String> {
        self.gpus.first().map(SimulatedGpu::gfx_target)
    }

    /// Write the simulated machine under `root`, and return the `PATH` its
    /// commands must run with: `tool` (the built `fake-host-tool`) linked under
    /// the name of every tool this machine has, then `real_path` with every
    /// [`HOST_TOOLS`] name filtered out (see the module docs).
    ///
    /// # Errors
    ///
    /// Any filesystem error while writing the tree.
    pub fn plant(
        &self,
        root: &Path,
        tool: &Path,
        real_path: &std::ffi::OsStr,
    ) -> io::Result<Vec<PathBuf>> {
        let wsl = matches!(self.platform, Platform::Wsl2 { .. });
        let kernel = if wsl { WSL2_KERNEL } else { BARE_METAL_KERNEL };

        write(
            root,
            "etc/os-release",
            "PRETTY_NAME=\"Ubuntu 24.04.2 LTS\"\nNAME=\"Ubuntu\"\nVERSION_ID=\"24.04\"\n\
             VERSION_CODENAME=noble\nID=ubuntu\nID_LIKE=debian\n",
        )?;
        write(
            root,
            "proc/version",
            &format!("Linux version {kernel} (buildd@simulated) (gcc 13.3.0) #1 SMP\n"),
        )?;
        write(
            root,
            "proc/cmdline",
            &format!("BOOT_IMAGE=/boot/vmlinuz-{kernel} root=/dev/sda1 ro quiet\n"),
        )?;
        write(
            root,
            "proc/cpuinfo",
            "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 9654 96-Core Processor\n",
        )?;
        write(root, "proc/meminfo", "MemTotal:       131072000 kB\n")?;

        let tools = root.join(TOOL_OUTPUT_DIR);
        std::fs::create_dir_all(&tools)?;
        std::fs::write(tools.join("uname"), "Linux\n")?;
        std::fs::write(tools.join("uname -s"), "Linux\n")?;
        std::fs::write(tools.join("uname -r"), format!("{kernel}\n"))?;
        std::fs::write(tools.join("uname -v"), "#1 SMP PREEMPT_DYNAMIC\n")?;
        std::fs::write(tools.join("uname -m"), "x86_64\n")?;
        std::fs::write(tools.join("ldconfig -p"), self.ldconfig_cache())?;
        std::fs::write(tools.join("lspci -nn -D"), self.lspci())?;
        std::fs::write(tools.join("mpirun"), "mpirun (Open MPI) 4.1.6\n")?;

        match self.platform {
            Platform::BareMetal => self.plant_bare_metal(root)?,
            Platform::Wsl2 { rocdxg_ready } => plant_wsl2(root, rocdxg_ready)?,
        }
        let stand_ins = link_stand_ins(&tools, tool)?;
        let filtered = filter_real_path(&tools.join("host-path"), real_path)?;
        Ok(vec![stand_ins, filtered])
    }

    fn plant_bare_metal(&self, root: &Path) -> io::Result<()> {
        let mut modules = String::new();
        // KFD node 0 is always the CPU, and reports a `gfx_target_version` of 0.
        write(
            root,
            "sys/class/kfd/kfd/topology/nodes/0/properties",
            "cpu_cores_count 96\nsimd_count 0\nmem_banks_count 1\ngfx_target_version 0\n\
             vendor_id 0\ndevice_id 0\nlocation_id 0\ndomain 0\ndrm_render_minor 0\n",
        )?;
        write(root, "sys/class/kfd/kfd/topology/nodes/0/gpu_id", "0\n")?;
        if !self.gpus.is_empty() {
            write(root, "dev/kfd", "")?;
            write(root, "sys/module/amdgpu/version", "6.12.12\n")?;
            modules.push_str("amdgpu 19943424 0 - Live 0x0000000000000000\n");
        }
        for (index, gpu) in self.gpus.iter().enumerate() {
            let node = index + 1;
            let render_minor = 128 + index;
            write(
                root,
                &format!("sys/class/kfd/kfd/topology/nodes/{node}/properties"),
                &format!(
                    "cpu_cores_count 0\nsimd_count 1216\nmem_banks_count 1\n\
                     gfx_target_version {}\nvendor_id 4098\ndevice_id {}\nlocation_id {}\n\
                     domain 0\ndrm_render_minor {render_minor}\n",
                    gpu.gfx_target_version, gpu.device_id, gpu.location_id
                ),
            )?;
            write(
                root,
                &format!("sys/class/kfd/kfd/topology/nodes/{node}/gpu_id"),
                &format!("{}\n", 10_000 + node),
            )?;
            let card = format!("sys/class/drm/card{index}/device");
            write(root, &format!("{card}/vendor"), "0x1002\n")?;
            write(
                root,
                &format!("{card}/device"),
                &format!("0x{:04x}\n", gpu.device_id),
            )?;
            write(root, &format!("dev/dri/card{index}"), "")?;
            write(root, &format!("dev/dri/renderD{render_minor}"), "")?;
        }
        write(root, "proc/modules", &modules)
    }

    /// `ldconfig -p`: the system libraries a prepared machine has, plus ROCDXG
    /// on a WSL2 host whose passthrough is ready.
    fn ldconfig_cache(&self) -> String {
        // Every simulated machine is a prepared one: the system libraries vLLM
        // and PyTorch's ROCm wheels need (OpenMPI, libatomic, libnuma) are
        // installed, as the CLI's own dependency checks would have left them.
        let mut text = String::from("5 libs found in cache `/etc/ld.so.cache'\n");
        for (soname, path) in [
            ("libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6"),
            ("libmpi.so.40", "/usr/lib/x86_64-linux-gnu/libmpi.so.40"),
            ("libatomic.so.1", "/lib/x86_64-linux-gnu/libatomic.so.1"),
            ("libnuma.so.1", "/lib/x86_64-linux-gnu/libnuma.so.1"),
        ] {
            let _ = writeln!(text, "\t{soname} (libc6,x86-64) => {path}");
        }
        if self.platform == (Platform::Wsl2 { rocdxg_ready: true }) {
            text.push_str("\tlibrocdxg.so (libc6,x86-64) => /opt/rocm/lib/librocdxg.so\n");
        }
        text
    }

    /// `lspci -nn -D`: every GPU the PCI bus carries, as the stand-in prints
    /// it. WSL2 has no PCI view of the GPU at all — the device is reached
    /// through `/dev/dxg`.
    #[must_use]
    pub fn lspci(&self) -> String {
        if matches!(self.platform, Platform::Wsl2 { .. }) {
            return String::new();
        }
        let mut text = String::from(
            "0000:00:00.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Device [1022:14a4]\n",
        );
        for gpu in &self.gpus {
            let _ = writeln!(
                text,
                "{} {}: Advanced Micro Devices, Inc. [AMD/ATI] {} [1002:{:04x}]",
                gpu.pci_address(),
                gpu.pci_class,
                gpu.pci_name,
                gpu.device_id
            );
        }
        text
    }
}

fn plant_wsl2(root: &Path, rocdxg_ready: bool) -> io::Result<()> {
    // What the CLI's video-controller inventory query prints on the Windows
    // host: one tab-separated `GPU` line per AMD display adapter.
    write(
        root,
        &format!("{TOOL_OUTPUT_DIR}/powershell.exe"),
        "GPU\tAMD Radeon(TM) 8060S Graphics\t32.0.21025.10016\t\
         PCI\\VEN_1002&DEV_1586&SUBSYS_1F111043&REV_C1\\4&1A2B3C4D&0&0041\n",
    )?;
    write(root, "dev/dxg", "")?;
    write(root, "usr/lib/wsl/lib/libdxcore.so", "")?;
    write(root, "proc/modules", "")?;
    if rocdxg_ready {
        write(root, "opt/rocm/lib/librocdxg.so", "")?;
        write(root, "opt/rocm/share/rocdxg/dids.conf", "")?;
    }
    Ok(())
}

/// Write `contents` to `root/relative`, creating parent directories.
fn write(root: &Path, relative: &str, contents: &str) -> io::Result<()> {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)
}

/// Link `tool` into `<tools>/bin` under the name of every [`HOST_TOOLS`] entry
/// that has canned output under `tools` — the tools this machine has.
fn link_stand_ins(tools: &Path, tool: &Path) -> io::Result<PathBuf> {
    let bin = tools.join("bin");
    std::fs::create_dir_all(&bin)?;
    let mut present: Vec<String> = std::fs::read_dir(tools)?
        .flatten()
        .filter(|entry| entry.path().is_file())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let program = name.split(' ').next()?.to_owned();
            HOST_TOOLS.contains(&program.as_str()).then_some(program)
        })
        .collect();
    present.sort();
    present.dedup();
    for name in present {
        link(tool, &bin.join(name))?;
    }
    Ok(bin)
}

/// Fill `dir` with links to every executable reachable through `real_path`
/// (the first of each name, as a `PATH` lookup would find it) except the
/// [`HOST_TOOLS`]. WSL's `/mnt/<drive>` entries are left out entirely: they lead
/// to the real Windows host, which no simulated machine can reach.
fn filter_real_path(dir: &Path, real_path: &std::ffi::OsStr) -> io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let mut seen = std::collections::HashSet::new();
    for entry_dir in std::env::split_paths(real_path) {
        if !entry_dir.is_absolute() || entry_dir.starts_with("/mnt") {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&entry_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if HOST_TOOLS.contains(&name.as_str()) || !is_executable(&entry.path()) {
                continue;
            }
            if seen.insert(name.clone()) {
                link(&entry.path(), &dir.join(&name))?;
            }
        }
    }
    Ok(dir.to_path_buf())
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn link(target: &Path, link: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    // A simulated host is a Linux host; elsewhere a copy keeps the build
    // honest without pretending the scenario can run there.
    #[cfg(not(unix))]
    {
        std::fs::copy(target, link).map(|_| ())
    }
}

/// The KFD topology nodes directory, relative to a root.
const KFD_NODES: &str = "sys/class/kfd/kfd/topology/nodes";

/// The `key value` pairs of a KFD node's `properties` file.
fn kfd_properties(node: &Path) -> Option<Vec<(String, String)>> {
    let text = std::fs::read_to_string(node.join("properties")).ok()?;
    Some(
        text.lines()
            .filter_map(|line| {
                let (key, value) = line.split_once(' ')?;
                Some((key.to_owned(), value.trim().to_owned()))
            })
            .collect(),
    )
}

fn property<'a>(properties: &'a [(String, String)], key: &str) -> Option<&'a str> {
    properties
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Whether a KFD property is a non-zero number, read the way the CLI reads it
/// (as an integer), so a `00` or padded value is not mistaken for a GPU.
fn is_nonzero(value: Option<&str>) -> bool {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .is_some_and(|v| v != 0)
}

/// Every way the bare-metal layout under `simulated` departs from the real
/// kernel's under `real` (normally `/`), as human-readable lines; empty when
/// they agree.
///
/// A simulated machine is only worth trusting while it is laid out the way the
/// kernel lays out the real one. If a kernel moves a property, a scenario on the
/// simulated machine keeps passing while the product fails on hardware — the
/// exact shape of the bug where `gfx_target_version` was read from a file no
/// kernel has. This is the check that notices: run on a real GPU host, it holds
/// every fixture assumption against the live `/dev` and `/sys`.
#[must_use]
pub fn bare_metal_layout_drift(real: &Path, simulated: &Path) -> Vec<String> {
    let mut drift = Vec::new();
    for device in ["dev/kfd", "sys/module/amdgpu"] {
        if simulated.join(device).exists() && !real.join(device).exists() {
            drift.push(format!(
                "the simulated machine has /{device}; this kernel does not"
            ));
        }
    }

    let gpu_nodes = |root: &Path| -> Vec<(PathBuf, Vec<(String, String)>)> {
        let Ok(entries) = std::fs::read_dir(root.join(KFD_NODES)) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter_map(|entry| Some((entry.path(), kfd_properties(&entry.path())?)))
            .filter(|(_, props)| is_nonzero(property(props, "gfx_target_version")))
            .collect()
    };
    let real_gpus = gpu_nodes(real);
    let simulated_gpus = gpu_nodes(simulated);
    if real_gpus.is_empty() {
        drift.push(format!(
            "no KFD GPU node on this kernel states gfx_target_version in its properties \
             file under /{KFD_NODES}, which is where the simulated machine puts it"
        ));
    }
    let Some((_, simulated_props)) = simulated_gpus.first() else {
        return drift;
    };
    for (node, real_props) in &real_gpus {
        let name = node.file_name().unwrap_or_default().to_string_lossy();
        for (key, _) in simulated_props {
            if property(real_props, key).is_none() {
                drift.push(format!(
                    "KFD node {name} has no `{key}` property, which the simulated machine states"
                ));
            }
        }
        if !node.join("gpu_id").is_file() {
            drift.push(format!("KFD node {name} has no gpu_id file"));
        }
        if let Some(minor) = property(real_props, "drm_render_minor")
            && !real.join(format!("dev/dri/renderD{minor}")).exists()
        {
            drift.push(format!(
                "KFD node {name} names render minor {minor}, but /dev/dri/renderD{minor} \
                 does not exist"
            ));
        }
    }
    // The simulated CPU node, like a real one, reports a target of 0.
    let cpu_nodes_with_target = std::fs::read_dir(real.join(KFD_NODES))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| kfd_properties(&entry.path()))
        .filter(|props| is_nonzero(property(props, "cpu_cores_count")))
        .filter(|props| is_nonzero(property(props, "gfx_target_version")))
        .count();
    if cpu_nodes_with_target > 0 {
        drift.push(format!(
            "{cpu_nodes_with_target} CPU KFD node(s) report a non-zero gfx_target_version"
        ));
    }

    // Each AMD DRM card exposes the files the simulated one does.
    let amd_cards: Vec<PathBuf> = std::fs::read_dir(real.join("sys/class/drm"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|card| {
            card.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.starts_with("card") && !name.contains('-')
            }) && std::fs::read_to_string(card.join("device/vendor"))
                .is_ok_and(|vendor| vendor.trim() == "0x1002")
        })
        .collect();
    if amd_cards.is_empty() {
        drift.push("no DRM card under /sys/class/drm reports vendor 0x1002".to_owned());
    }
    for card in &amd_cards {
        let device = std::fs::read_to_string(card.join("device/device")).unwrap_or_default();
        if !device.trim().starts_with("0x") || device.trim().len() != 6 {
            drift.push(format!(
                "{} has device id {:?}, not the 0xNNNN form the simulated machine writes",
                card.display(),
                device.trim()
            ));
        }
    }
    drift
}

/// Every way this WSL2 distribution under `real` (normally `/`) departs from
/// what the simulated WSL machine assumes, as human-readable lines.
///
/// The CLI recognises WSL2 by `/dev/dxg` and a `microsoft-standard-WSL2` kernel
/// string, and finds the host runtime through `/usr/lib/wsl/lib/libdxcore.so`;
/// those are the facts the simulated machine plants.
#[must_use]
pub fn wsl_layout_drift(real: &Path) -> Vec<String> {
    let mut drift = Vec::new();
    for path in ["dev/dxg", "usr/lib/wsl/lib/libdxcore.so"] {
        if !real.join(path).exists() {
            drift.push(format!(
                "the simulated WSL machine has /{path}; this one does not"
            ));
        }
    }
    let version = std::fs::read_to_string(real.join("proc/version")).unwrap_or_default();
    if !version.contains("microsoft-standard-WSL2") {
        drift.push(format!(
            "/proc/version does not carry the microsoft-standard-WSL2 kernel string the \
             simulated machine uses: {}",
            version.trim()
        ));
    }
    drift
}

/// Every way an `lspci -nn -D` listing departs from the line shape the
/// simulated machine's stand-in prints for an AMD GPU, as human-readable lines.
///
/// The product parses these lines for the PCI address and marketing name; a
/// stand-in that prints a shape the real tool does not would let the simulated
/// scenarios pass on a parse the product never gets to do on hardware.
#[must_use]
pub fn lspci_drift(real_listing: &str) -> Vec<String> {
    let gpu_classes = [
        "VGA compatible controller",
        "3D controller",
        "Display controller",
        "Processing accelerators",
    ];
    let amd_gpus: Vec<&str> = real_listing
        .lines()
        .filter(|line| line.contains("[1002:") && gpu_classes.iter().any(|c| line.contains(c)))
        .collect();
    if amd_gpus.is_empty() {
        return vec!["lspci -nn -D lists no AMD GPU".to_owned()];
    }
    amd_gpus
        .into_iter()
        .filter(|line| !matches_simulated_lspci_line(line))
        .map(|line| format!("lspci line not in the simulated shape: {line}"))
        .collect()
}

/// `DDDD:BB:DD.F <class> [cccc]: <vendor> <name> [1002:dddd]...` — the shape
/// [`SimulatedHost::lspci`] prints.
fn matches_simulated_lspci_line(line: &str) -> bool {
    let Some((address, rest)) = line.split_once(' ') else {
        return false;
    };
    let address_ok = address.len() == 12
        && address.char_indices().all(|(i, c)| match i {
            4 | 7 => c == ':',
            10 => c == '.',
            _ => c.is_ascii_hexdigit(),
        });
    let class_ok = rest.split_once("]: ").is_some_and(|(class, _)| {
        class
            .rsplit_once(" [")
            .is_some_and(|(_, code)| code.len() == 4)
    });
    let id_ok = rest.find("[1002:").is_some_and(|i| {
        rest[i + 6..].get(..5).is_some_and(|tail| {
            tail.ends_with(']') && tail[..4].chars().all(|c| c.is_ascii_hexdigit())
        })
    });
    address_ok && class_ok && id_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi300x_decodes_to_gfx942_at_the_lspci_address() {
        let gpu = SimulatedGpu::mi300x(0x11);
        assert_eq!(gpu.gfx_target(), "gfx942");
        // KFD location_id 4352 is the address `lspci -D` prints for the first
        // GPU of a real eight-GPU MI300X host.
        assert_eq!(gpu.location_id, 4352);
        assert_eq!(gpu.pci_address(), "0000:11:00.0");
    }

    #[test]
    fn strix_halo_decodes_to_gfx1151() {
        assert_eq!(SimulatedGpu::strix_halo().gfx_target(), "gfx1151");
    }

    fn planted(host: &SimulatedHost) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("tempdir");
        let tool = root.path().join("tool");
        std::fs::write(&tool, "").expect("tool");
        host.plant(root.path(), &tool, std::ffi::OsStr::new(""))
            .expect("plant");
        root
    }

    /// The kernel puts `gfx_target_version` inside `properties`; a standalone
    /// file is a layout no kernel has, and the product used to read only that.
    #[test]
    fn the_kfd_target_lives_inside_each_node_properties_file() {
        let root = planted(&SimulatedHost::instinct_mi300x(2));
        let nodes = root.path().join("sys/class/kfd/kfd/topology/nodes");
        for node in ["0", "1", "2"] {
            assert!(!nodes.join(node).join("gfx_target_version").exists());
        }
        let gpu = std::fs::read_to_string(nodes.join("2/properties")).expect("node 2");
        assert!(
            gpu.lines().any(|line| line == "gfx_target_version 90402"),
            "{gpu}"
        );
        let cpu = std::fs::read_to_string(nodes.join("0/properties")).expect("node 0");
        assert!(
            cpu.lines().any(|line| line == "gfx_target_version 0"),
            "{cpu}"
        );
    }

    #[test]
    fn a_bare_metal_gpu_host_has_kfd_and_no_wsl_device() {
        let root = planted(&SimulatedHost::instinct_mi300x(1));
        assert!(root.path().join("dev/kfd").exists());
        assert!(root.path().join("dev/dri/renderD128").exists());
        assert!(!root.path().join("dev/dxg").exists());
        let version = std::fs::read_to_string(root.path().join("proc/version")).expect("version");
        assert!(!version.to_lowercase().contains("microsoft"), "{version}");
    }

    #[test]
    fn a_host_without_a_gpu_has_no_kfd() {
        let root = planted(&SimulatedHost::bare_metal_without_gpu());
        assert!(!root.path().join("dev/kfd").exists());
        assert!(
            !root
                .path()
                .join("sys/class/kfd/kfd/topology/nodes/1")
                .exists()
        );
    }

    #[test]
    fn a_wsl2_host_has_dxg_and_rocdxg_in_the_linker_cache() {
        let root = planted(&SimulatedHost::wsl2_strix_halo());
        assert!(root.path().join("dev/dxg").exists());
        assert!(root.path().join("usr/lib/wsl/lib/libdxcore.so").exists());
        assert!(!root.path().join("dev/kfd").exists());
        let cache = std::fs::read_to_string(root.path().join(TOOL_OUTPUT_DIR).join("ldconfig -p"))
            .expect("ldconfig");
        assert!(cache.contains("librocdxg.so"), "{cache}");
    }

    /// Only the tools the simulated machine has are on its `PATH`; the real
    /// machine's hardware tools are filtered out, and everything else it has
    /// stays reachable.
    #[cfg(unix)]
    #[test]
    fn the_path_carries_the_machines_tools_and_hides_the_real_ones() {
        use std::os::unix::fs::PermissionsExt as _;
        let real = tempfile::tempdir().expect("real dir");
        for name in ["lspci", "rocminfo", "powershell.exe", "sh", "tar"] {
            let path = real.path().join(name);
            std::fs::write(&path, "").expect("tool");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        let root = tempfile::tempdir().expect("root");
        let tool = root.path().join("tool");
        std::fs::write(&tool, "").expect("tool");
        let path = SimulatedHost::instinct_mi300x(1)
            .plant(root.path(), &tool, real.path().as_os_str())
            .expect("plant");
        let on_path = |name: &str| path.iter().find(|dir| dir.join(name).exists());

        // Stand-ins the machine has, answering from canned output.
        for name in ["uname", "ldconfig", "lspci", "mpirun"] {
            assert_eq!(on_path(name), Some(&path[0]), "{name}");
        }
        // Tools the machine does not have are absent, not present-but-failing,
        // and the real machine's copies stay hidden.
        for name in ["rocminfo", "amd-smi", "powershell.exe", "journalctl"] {
            assert_eq!(on_path(name), None, "{name}");
        }
        // The rest of the real machine is still there.
        for name in ["sh", "tar"] {
            assert_eq!(on_path(name), Some(&path[1]), "{name}");
        }
    }

    #[test]
    fn a_wsl2_machine_answers_the_windows_host_query() {
        let root = planted(&SimulatedHost::wsl2_strix_halo());
        let bin = root.path().join(TOOL_OUTPUT_DIR).join("bin");
        assert!(bin.join("powershell.exe").symlink_metadata().is_ok());
        assert!(bin.join("lspci").symlink_metadata().is_ok());
    }
}

#[cfg(test)]
mod drift_tests {
    use super::*;

    fn planted(host: &SimulatedHost) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("tempdir");
        let tool = root.path().join("tool");
        std::fs::write(&tool, "").expect("tool");
        host.plant(root.path(), &tool, std::ffi::OsStr::new(""))
            .expect("plant");
        root
    }

    #[test]
    fn a_simulated_machine_does_not_drift_from_itself() {
        let a = planted(&SimulatedHost::instinct_mi300x(2));
        let b = planted(&SimulatedHost::instinct_mi300x(1));
        assert_eq!(
            bare_metal_layout_drift(a.path(), b.path()),
            Vec::<String>::new()
        );
    }

    /// The bug this exists for: a kernel that stated the target somewhere other
    /// than `properties` must be reported, not silently matched.
    #[test]
    fn a_kernel_without_the_target_in_properties_is_drift() {
        let real = planted(&SimulatedHost::instinct_mi300x(1));
        let node = real.path().join(KFD_NODES).join("1");
        let props = std::fs::read_to_string(node.join("properties")).expect("props");
        std::fs::write(
            node.join("properties"),
            props
                .lines()
                .filter(|line| !line.starts_with("gfx_target_version"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("rewrite");
        std::fs::write(node.join("gfx_target_version"), "90402\n").expect("standalone");
        let simulated = planted(&SimulatedHost::instinct_mi300x(1));
        let drift = bare_metal_layout_drift(real.path(), simulated.path());
        assert!(
            drift.iter().any(|line| line.contains("gfx_target_version")),
            "{drift:#?}"
        );
    }

    #[test]
    fn a_missing_property_is_named() {
        let real = planted(&SimulatedHost::instinct_mi300x(1));
        let node = real.path().join(KFD_NODES).join("1");
        let props = std::fs::read_to_string(node.join("properties")).expect("props");
        std::fs::write(node.join("properties"), props.replace("location_id", "loc")).expect("w");
        let simulated = planted(&SimulatedHost::instinct_mi300x(1));
        let drift = bare_metal_layout_drift(real.path(), simulated.path());
        assert!(
            drift.iter().any(|line| line.contains("`location_id`")),
            "{drift:#?}"
        );
    }

    #[test]
    fn the_stand_ins_lspci_shape_matches_its_own_grammar() {
        let host = SimulatedHost::instinct_mi300x(8);
        assert_eq!(lspci_drift(&host.lspci()), Vec::<String>::new());
    }

    #[test]
    fn a_real_lspci_line_in_another_shape_is_drift() {
        // `lspci` without `-D` drops the domain; the product's address match
        // would then never line up with the KFD-derived one.
        let listing = "11:00.0 Processing accelerators [1200]: Advanced Micro Devices, Inc. \
                       [AMD/ATI] Aqua Vanjaram [Instinct MI300X] [1002:74a1]";
        assert_eq!(lspci_drift(listing).len(), 1);
        // A real line with a trailing revision is still the same shape.
        let ok = "0000:c5:00.0 Display controller [0380]: Advanced Micro Devices, Inc. \
                  [AMD/ATI] Strix Halo [Radeon Graphics] [1002:1586] (rev c1)";
        assert_eq!(lspci_drift(ok), Vec::<String>::new());
    }
}
