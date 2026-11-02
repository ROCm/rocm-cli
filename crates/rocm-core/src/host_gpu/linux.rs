// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Linux sysfs, KFD and DRM probes: the KFD topology, the DRM ip-discovery gfx target,
//! the amdgpu device and driver-version reads, and the primary-GPU name.

#[cfg(any(target_os = "linux", test))]
use crate::gfx_target::{gfx_target_from_gc_version, parse_linux_kfd_gfx_target};
#[cfg(target_os = "linux")]
use crate::runtime_is_linux;
use crate::{host_path, probe_exec};
use std::fs;
use std::path::Path;

#[cfg(target_os = "linux")]
pub(super) fn detect_linux_primary_gpu_name() -> Option<String> {
    if !runtime_is_linux() {
        return None;
    }

    let entries = fs::read_dir(host_path("/sys/class/drm")).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device_dir = entry.path().join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        for file_name in ["product_name", "product", "model"] {
            let Some(value) = fs::read_to_string(device_dir.join(file_name)).ok() else {
                continue;
            };
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub(super) const fn detect_linux_primary_gpu_name() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
pub(crate) fn detect_linux_sysfs_gfx_target() -> Option<String> {
    if !runtime_is_linux() {
        return None;
    }

    detect_linux_kfd_gfx_target().or_else(detect_linux_drm_ip_discovery_gfx_target)
}

#[cfg(not(target_os = "linux"))]
pub(crate) const fn detect_linux_sysfs_gfx_target() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
pub(crate) fn detect_linux_kfd_gfx_target() -> Option<String> {
    detect_kfd_gfx_target_in(&host_path("/sys/class/kfd/kfd/topology/nodes"))
}

/// One GPU as the kernel's KFD topology describes it.
///
/// This is the *kernel's* answer to "which GPUs exist here", which is a
/// different question from the one the PCI bus answers. In a container the bus
/// still carries every card the host has, while KFD carries only the devices
/// passed through — so this is what `examine` must count, and the PCI scan is
/// only good for naming what it finds here.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KfdGpuNode {
    /// The node's PCI address as `lspci -D` spells it (`0000:11:00.0`), or
    /// empty when the node does not state a usable one.
    pub(crate) pci_id: String,
    /// The node's own gfx target (`gfx942`), or empty when unparseable.
    ///
    /// Per-node, so it can be attributed to a particular device: an APU+dGPU
    /// host reports two different targets and each belongs to exactly one card.
    pub(crate) gfx_target: String,
}

/// Every GPU the KFD topology describes, in node order, read from a
/// caller-supplied nodes directory. `None` when the topology could not be read
/// at all, which callers must treat as "cannot say" rather than as "no GPUs".
///
/// Same planted-directory seam and cfg gating as [`detect_kfd_gfx_target_in`].
/// The `/sys` path lives in the probe that calls this
/// (`examine::probe_gpus_kernel_membership`), so that probe's own wiring is
/// drivable from a test too rather than only the reconcile it hands off to.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn kfd_gpu_nodes_in(nodes_dir: &Path) -> Option<Vec<KfdGpuNode>> {
    let mut nodes: Vec<((u64, String), KfdGpuNode)> = fs::read_dir(nodes_dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let version = kfd_node_gfx_target_version(&path)?;
            if !kfd_gfx_target_version_is_gpu(version.trim()) {
                return None;
            }
            let properties = fs::read_to_string(path.join("properties")).unwrap_or_default();
            Some((
                natural_node_order(&entry.file_name().to_string_lossy()),
                KfdGpuNode {
                    pci_id: kfd_node_pci_id(&properties).unwrap_or_default(),
                    gfx_target: parse_linux_kfd_gfx_target(version.trim()).unwrap_or_default(),
                },
            ))
        })
        .collect();
    // `read_dir` order is filesystem-defined, so sort on the node number for the
    // same reason `detect_kfd_gfx_target_in` does: node 0 is HIP ordinal 0.
    nodes.sort_by(|(left, _), (right, _)| left.cmp(right));
    Some(nodes.into_iter().map(|(_, node)| node).collect())
}

/// The PCI address a KFD topology node reports, spelled as `lspci -D` spells it.
///
/// KFD states the address as two decimal properties. `domain` is the PCI domain;
/// `location_id` is the kernel's `pci_dev_id()`, i.e. `(bus << 8) | devfn`, with
/// `devfn` packing the device number in bits 3..8 and the function in bits 0..3.
/// So `location_id 4352` (`0x1100`) in domain 0 is `0000:11:00.0`.
///
/// Verified against an 8-GPU MI300X host: nodes 2..9 report `location_id` 4352,
/// 12032, 17920, 23808, 35584, 43520, 49664 and 55808, which decode to exactly
/// the eight addresses `lspci -D` lists for its accelerators (EAI-8449).
///
/// `None` when the property is absent or zero. Zero is refused rather than
/// decoded: `0000:00:00.0` is the host bridge, so emitting it would be a
/// wrong-but-plausible address that could match an unrelated PCI entry.
#[cfg(any(target_os = "linux", test))]
fn kfd_node_pci_id(properties: &str) -> Option<String> {
    let location = kfd_property_value(properties, "location_id")?
        .parse::<u32>()
        .ok()?;
    if location == 0 {
        return None;
    }
    let domain = kfd_property_value(properties, "domain")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let bus = (location >> 8) & 0xff;
    let device = (location >> 3) & 0x1f;
    let function = location & 0x7;
    Some(format!("{domain:04x}:{bus:02x}:{device:02x}.{function}"))
}

/// The KFD-topology read, against a caller-supplied nodes directory.
///
/// Split out so it can be driven against a planted directory: the hosts where
/// this matters most (an Instinct box with no `lspci`) are exactly the ones a
/// test cannot run on. Same seam as `discover_rocm_installs_in`.
///
/// Gated the same way as `parse_linux_kfd_gfx_target`, which it calls: present
/// on Linux and under `cfg(test)` everywhere, so the tests run on every platform
/// without the function existing in a Windows release build that can never use
/// it. (`target_os` alone would have left the tests Linux-only; `test` alone
/// would not compile on Windows CI, which is how this was found.)
#[cfg(any(target_os = "linux", test))]
fn detect_kfd_gfx_target_in(nodes_dir: &Path) -> Option<String> {
    let mut targets: Vec<(String, String)> = fs::read_dir(nodes_dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let value = kfd_node_gfx_target_version(&entry.path())?;
            let token = parse_linux_kfd_gfx_target(value.trim())?;
            Some((entry.file_name().to_string_lossy().into_owned(), token))
        })
        .collect();
    // `read_dir` order is filesystem-defined, so a multi-node box could report a
    // different GPU run to run. Node directories are numbered (`0`, `1`, ... on
    // a real KFD; `node0`, `node1`, ... in planted fixtures), so sorting on the
    // trailing number makes the answer stable and picks the lowest-numbered
    // node, which is the one HIP ordinal 0 refers to.
    targets.sort_by_key(|(name, _)| natural_node_order(name));
    targets.into_iter().next().map(|(_, token)| token)
}

/// Sort key for a KFD node directory name: its trailing number when it has one,
/// so `node9` precedes `node10` rather than following it lexicographically.
#[cfg(any(target_os = "linux", test))]
fn natural_node_order(name: &str) -> (u64, String) {
    let digits: String = name
        .chars()
        .skip_while(|ch| !ch.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    (
        digits.parse::<u64>().unwrap_or(u64::MAX),
        name.to_ascii_lowercase(),
    )
}

/// Count AMD (`amdgpu`) primary DRM cards under `/sys/class/drm` (`card0`,
/// `card1`, …), skipping connector sub-nodes like `card0-DP-1`. `None` when the
/// DRM class dir can't be read; `Some(0)` when it is readable with no AMD card.
#[cfg(target_os = "linux")]
pub(super) fn linux_drm_amdgpu_card_count() -> Option<usize> {
    let entries = fs::read_dir(host_path("/sys/class/drm")).ok()?;
    let count = entries
        .flatten()
        .filter(|entry| {
            let card_path = entry.path();
            let Some(name) = card_path.file_name().and_then(|value| value.to_str()) else {
                return false;
            };
            name.starts_with("card")
                && !name.contains('-')
                && is_amdgpu_device(&card_path.join("device"))
        })
        .count();
    Some(count)
}

/// Count AMD GPU nodes in the KFD topology. `Some(0)` is an authoritative "no
/// GPU" (topology readable with no GPU node, or no KFD device at all); `None`
/// means the topology could not be read even though `/dev/kfd` exists, so
/// availability is unknown and must not be treated as zero.
#[cfg(target_os = "linux")]
pub(crate) fn linux_kfd_gpu_node_count() -> Option<usize> {
    linux_kfd_gpu_node_count_in(&host_path("/sys/class/kfd/kfd/topology/nodes"))
}

/// The KFD GPU-node count, against a caller-supplied nodes directory.
///
/// Split out for the same reason as [`detect_kfd_gfx_target_in`]: the hosts
/// where this matters are the ones a test cannot run on. Only the readable
/// branch is driven by tests -- the unreadable branches key off the real
/// `/dev/kfd`, which a planted directory cannot stand in for.
#[cfg(any(target_os = "linux", test))]
fn linux_kfd_gpu_node_count_in(nodes_dir: &Path) -> Option<usize> {
    match fs::read_dir(nodes_dir) {
        Ok(entries) => Some(
            entries
                .flatten()
                .filter(|entry| kfd_node_is_gpu(&entry.path()))
                .count(),
        ),
        Err(_) if host_path("/dev/kfd").exists() => None,
        Err(_) => Some(0),
    }
}

/// A KFD topology node is a GPU (not the CPU node) when its
/// `gfx_target_version` is a nonzero value; CPU nodes report `0`.
#[cfg(any(target_os = "linux", test))]
fn kfd_node_is_gpu(node_dir: &Path) -> bool {
    kfd_node_gfx_target_version(node_dir)
        .is_some_and(|value| kfd_gfx_target_version_is_gpu(value.trim()))
}

/// A KFD topology node's `gfx_target_version`, or `None` when the node does not
/// state one.
///
/// The kernel exposes it as a **line inside the node's `properties` file**
/// (`gfx_target_version 90402`), not as a standalone file. Reading only the
/// standalone path found nothing on every real KFD host, so target detection
/// silently fell through to the DRM ip-discovery route, which decodes a GC IP
/// version and is wrong-but-plausible on the GC 9.4.x line: an MI300X reported
/// `gfx943` where KFD plainly said `90402` (gfx942). Hosts whose ip-discovery
/// route also came up empty got `<unknown>` instead.
///
/// The standalone file is still read as a fallback, so any layout that does
/// expose it keeps working.
#[cfg(any(target_os = "linux", test))]
fn kfd_node_gfx_target_version(node_dir: &Path) -> Option<String> {
    let from_properties = fs::read_to_string(node_dir.join("properties"))
        .ok()
        .and_then(|text| kfd_property_value(&text, "gfx_target_version"));
    if from_properties.is_some() {
        return from_properties;
    }
    fs::read_to_string(node_dir.join("gfx_target_version"))
        .ok()
        .map(|value| value.trim().to_owned())
}

/// The value of one `key value` line in a KFD `properties` body.
#[cfg(any(target_os = "linux", test))]
fn kfd_property_value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next()? != key {
            return None;
        }
        parts.next().map(str::to_owned)
    })
}

#[cfg(any(target_os = "linux", test))]
fn kfd_gfx_target_version_is_gpu(value: &str) -> bool {
    value
        .trim()
        .parse::<u64>()
        .is_ok_and(|version| version != 0)
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_ip_discovery_gfx_target() -> Option<String> {
    let drm_dir = host_path("/sys/class/drm");
    let entries = fs::read_dir(drm_dir).ok()?;
    for entry in entries.flatten() {
        let card_path = entry.path();
        let Some(card_name) = card_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !card_name.starts_with("card") || card_name.contains('-') {
            continue;
        }
        let device_dir = card_path.join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        let gc_root = device_dir.join("ip_discovery");
        let token = detect_ip_discovery_gc_target(&gc_root);
        if token.is_some() {
            return token;
        }
    }
    None
}

/// Whether a DRM `device` directory belongs to an AMD GPU: the PCI `vendor` id
/// (`0x1002`), or, when that is unreadable, an `amdgpu` `uevent` `DRIVER=` line.
///
/// Both signals are needed. A vendor-only test under-counts on hosts where
/// `vendor` is absent or unreadable, and the callers are counting *cards* — the
/// KFD/DRM count authority here and the `rocm` CLI's sysfs fallback probe — so
/// an under-count silently narrows the multi-card ordinal guard.
///
/// `pub` and not `#[cfg(target_os = "linux")]` precisely so there is one copy:
/// the CLI's fallback probe used to carry its own, with a doc comment asserting
/// the two were "the same two-signal test" and nothing holding them to it. It
/// reads files, so there is nothing platform-specific to gate, and gating it
/// would put it out of reach of a dependent crate's `cfg(test)` build.
#[must_use]
pub fn is_amdgpu_device(device_dir: &Path) -> bool {
    if let Ok(vendor) = fs::read_to_string(device_dir.join("vendor"))
        && vendor.trim().eq_ignore_ascii_case("0x1002")
    {
        return true;
    }
    if let Ok(uevent) = fs::read_to_string(device_dir.join("uevent")) {
        return uevent.lines().any(|line| line.trim() == "DRIVER=amdgpu");
    }
    false
}

#[cfg(any(target_os = "linux", test))]
fn detect_ip_discovery_gc_target(ip_discovery_dir: &Path) -> Option<String> {
    let die_entries = fs::read_dir(ip_discovery_dir.join("die")).ok()?;
    for die in die_entries.flatten() {
        let Some(gc_entries) = fs::read_dir(die.path().join("GC")).ok() else {
            continue;
        };
        for gc in gc_entries.flatten() {
            let block = gc.path();
            let Some(major) = fs::read_to_string(block.join("major"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(minor) = fs::read_to_string(block.join("minor"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(revision) = fs::read_to_string(block.join("revision"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            if let Some(token) = gfx_target_from_gc_version(major, minor, revision) {
                return Some(token);
            }
        }
    }
    None
}

/// The amdgpu kernel module's version on Linux, if it reports one.
///
/// Prefers the sysfs attribute (a plain file read, no subprocess) that DKMS
/// builds of amdgpu expose. Falls back to `modinfo`, which is the only source
/// for the in-tree kernel module -- it doesn't populate
/// `/sys/module/amdgpu/version` at all.
pub(super) fn detect_linux_amdgpu_driver_version() -> Option<String> {
    if let Ok(text) = fs::read_to_string(host_path("/sys/module/amdgpu/version")) {
        let version = text.trim();
        if !version.is_empty() {
            return Some(version.to_owned());
        }
    }
    let (rc, out, _) = probe_exec::run("modinfo", &["amdgpu"], probe_exec::SHORT);
    if rc != 0 {
        return None;
    }
    out.lines()
        .find_map(|line| line.strip_prefix("version:"))
        .map(str::trim)
        .filter(|version| !version.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx_target::normalize_therock_family;
    use crate::test_support::{temp_app_paths, workspace_test_artifact_dir};
    use anyhow::Result;

    #[test]
    fn linux_ip_discovery_gc_fixture_maps_to_gfx_target() -> Result<()> {
        let (root, _paths) = temp_app_paths("linux-ip-discovery");
        let gc = root
            .join("ip_discovery")
            .join("die")
            .join("0")
            .join("GC")
            .join("0");
        fs::create_dir_all(&gc)?;
        fs::write(gc.join("major"), "11")?;
        fs::write(gc.join("minor"), "0")?;
        fs::write(gc.join("revision"), "3")?;

        assert_eq!(
            detect_ip_discovery_gc_target(&root.join("ip_discovery")),
            Some("gfx1103".to_owned())
        );

        // KNOWN WRONG, and pinned so the gap stays visible: on the GC 9.4.x line
        // the GC IP version is not the LLVM target. Aldebaran (MI200/MI250)
        // reports GC 9.4.2 here but is gfx90a, so this path yields MI300's target
        // instead and resolves to the wrong TheRock family — the right wheel for
        // this host is `gfx90a`. Pre-existing: the decimal encoding produced
        // `gfx942` for 9/4/2 too, so the hex fix neither caused nor closes it.
        // Fixing it needs a GC-IP-version → target table for GC 9.4.x
        // (9.4.0/9.4.1/9.4.3 are gfx906/gfx908/gfx942), not a different encoding.
        fs::write(gc.join("major"), "9")?;
        fs::write(gc.join("minor"), "4")?;
        fs::write(gc.join("revision"), "2")?;
        let aldebaran = detect_ip_discovery_gc_target(&root.join("ip_discovery"));
        assert_eq!(aldebaran, Some("gfx942".to_owned()));
        assert_eq!(
            normalize_therock_family(&aldebaran.expect("target")),
            Some("gfx94X-dcgpu".to_owned())
        );

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn linux_amdgpu_device_fixture_accepts_vendor_or_uevent_driver() -> Result<()> {
        let (root, _paths) = temp_app_paths("linux-amdgpu-device");
        let vendor = root.join("vendor");
        fs::create_dir_all(&root)?;
        fs::write(&vendor, "0x1002\n")?;
        assert!(is_amdgpu_device(&root));
        fs::remove_file(&vendor)?;
        fs::write(root.join("uevent"), "DRIVER=amdgpu\n")?;
        assert!(is_amdgpu_device(&root));
        fs::write(root.join("uevent"), "DRIVER=i915\n")?;
        assert!(!is_amdgpu_device(&root));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn kfd_topology_names_the_gpu_when_no_tooling_is_installed() -> Result<()> {
        // Covers the standalone-file fallback; see
        // `kfd_topology_reads_the_target_from_the_properties_file` for the
        // layout a real KFD host has. `Examination` enumerates GPUs by shelling out to
        // lspci and rocminfo; where neither is reachable it reported no AMD GPU
        // on a machine that has one, while the human report -- which reads this
        // topology instead -- named the target correctly. Planted here because
        // the hosts where it matters are the ones a test cannot run on.
        let (root, _) = temp_app_paths("kfd-topology");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("node0"))?;
        fs::write(nodes.join("node0").join("gfx_target_version"), "90402\n")?;

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_topology_answer_does_not_depend_on_directory_order() -> Result<()> {
        // `read_dir` order is filesystem-defined, so a multi-node box could name
        // a different GPU run to run. node9 must not beat node10 lexically
        // either -- the lowest-numbered node is the one HIP ordinal 0 means.
        let (root, _) = temp_app_paths("kfd-topology-order");
        let nodes = root.join("nodes");
        for (node, version) in [("node10", "110100"), ("node9", "90402"), ("node0", "90400")] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(nodes.join(node).join("gfx_target_version"), version)?;
        }

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx940"));
        Ok(())
    }

    #[test]
    fn kfd_topology_skips_nodes_that_name_no_target() -> Result<()> {
        // A CPU node carries a zero version. Reporting `gfx0` off the first
        // directory encountered would be worse than reporting nothing.
        let (root, _) = temp_app_paths("kfd-topology-cpu-node");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("node0"))?;
        fs::write(nodes.join("node0").join("gfx_target_version"), "0\n")?;
        fs::create_dir_all(nodes.join("node1"))?;
        fs::write(nodes.join("node1").join("gfx_target_version"), "110000\n")?;

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx1100"));
        Ok(())
    }

    #[test]
    fn kfd_topology_reads_the_target_from_the_properties_file() -> Result<()> {
        // The layout every real KFD host actually has: the value is a line in
        // the node's `properties`, and there is no standalone file. Reading only
        // the standalone path found nothing here, so an MI300X fell through to
        // ip-discovery and was reported as `gfx943` while KFD said `90402`.
        // Node directories are bare integers on real hardware, not `nodeN`.
        let (root, _) = temp_app_paths("kfd-topology-properties");
        let nodes = root.join("nodes");
        // Trimmed from a live MI300X; ordering and neighbours are as found.
        for (node, properties) in [
            (
                "0",
                "cpu_cores_count 56\nsimd_count 0\ngfx_target_version 0\n",
            ),
            (
                "5",
                "cpu_cores_count 0\nsimd_count 1216\nmax_waves_per_simd 8\ngfx_target_version 90402\n",
            ),
        ] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(nodes.join(node).join("properties"), properties)?;
        }

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_gpu_node_count_reads_properties_and_skips_cpu_nodes() -> Result<()> {
        // The count used the same phantom standalone file, so it saw zero GPUs
        // on every real host and let the DRM card count stand in for it. Two
        // CPU nodes and two GPU nodes, shaped like a live Instinct topology.
        let (root, _) = temp_app_paths("kfd-node-count-properties");
        let nodes = root.join("nodes");
        for (node, properties) in [
            ("0", "cpu_cores_count 56\ngfx_target_version 0\n"),
            ("1", "cpu_cores_count 56\ngfx_target_version 0\n"),
            ("5", "simd_count 1216\ngfx_target_version 90402\n"),
            ("6", "simd_count 1216\ngfx_target_version 90402\n"),
        ] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(nodes.join(node).join("properties"), properties)?;
        }

        let count = linux_kfd_gpu_node_count_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(count, Some(2));
        Ok(())
    }

    #[test]
    fn kfd_nodes_carry_a_decoded_pci_address_and_their_own_target() -> Result<()> {
        // The first three GPU nodes of a real 8-GPU MI300X host, verbatim from
        // its `location_id` values, behind the two CPU nodes it also reports.
        // `lspci -D` lists those cards at 0000:11:00.0, 0000:2f:00.0 and
        // 0000:46:00.0, so the decode is checked against the machine rather
        // than against itself.
        let (root, _) = temp_app_paths("kfd-nodes-pci");
        let nodes = root.join("nodes");
        for node in ["0", "1"] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(
                nodes.join(node).join("properties"),
                "cpu_cores_count 56\nsimd_count 0\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
            )?;
        }
        for (node, location) in [("2", 4352), ("3", 12032), ("4", 17920)] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(
                nodes.join(node).join("properties"),
                format!(
                    "cpu_cores_count 0\nsimd_count 1216\ngfx_target_version 90402\n\
                     location_id {location}\ndomain 0\n"
                ),
            )?;
        }
        let read = kfd_gpu_nodes_in(&nodes);
        fs::remove_dir_all(&root).ok();

        let expected: Vec<KfdGpuNode> = ["0000:11:00.0", "0000:2f:00.0", "0000:46:00.0"]
            .into_iter()
            .map(|pci_id| KfdGpuNode {
                pci_id: pci_id.to_owned(),
                gfx_target: "gfx942".to_owned(),
            })
            .collect();
        // The CPU nodes are excluded, and node order is preserved so entry 0 is
        // the device HIP calls ordinal 0.
        assert_eq!(read, Some(expected));
        Ok(())
    }

    #[test]
    fn kfd_nodes_keep_their_targets_apart_and_refuse_an_unusable_address() -> Result<()> {
        // An APU + a discrete card. Each node states its own target, which is
        // the whole point of reading them per-node: a single host-wide answer
        // would stamp the APU's gfx1103 onto the discrete card.
        //
        // The dGPU node here also reports `location_id 0`. That decodes to
        // 0000:00:00.0 -- the host bridge -- so it must come back empty rather
        // than as an address that could match an unrelated PCI entry.
        let (root, _) = temp_app_paths("kfd-nodes-mixed");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("0"))?;
        fs::write(
            nodes.join("0").join("properties"),
            "cpu_cores_count 16\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
        )?;
        fs::create_dir_all(nodes.join("1"))?;
        fs::write(
            nodes.join("1").join("properties"),
            "simd_count 256\ngfx_target_version 110003\nlocation_id 25600\ndomain 0\n",
        )?;
        fs::create_dir_all(nodes.join("2"))?;
        fs::write(
            nodes.join("2").join("properties"),
            "simd_count 768\ngfx_target_version 110000\nlocation_id 0\ndomain 0\n",
        )?;
        let read = kfd_gpu_nodes_in(&nodes);
        // The host-wide answer is still the APU, which is why it must not be
        // attributed to the discrete card.
        let lowest = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(
            read,
            Some(vec![
                KfdGpuNode {
                    pci_id: "0000:64:00.0".to_owned(),
                    gfx_target: "gfx1103".to_owned(),
                },
                KfdGpuNode {
                    pci_id: String::new(),
                    gfx_target: "gfx1100".to_owned(),
                },
            ])
        );
        assert_eq!(lowest.as_deref(), Some("gfx1103"));
        Ok(())
    }

    #[test]
    fn a_kfd_node_decodes_a_nonzero_function_and_domain() {
        // devfn packs the device in bits 3..8 and the function in bits 0..3, and
        // the domain is a separate property -- so neither is assumed to be zero.
        // 0x8ffa = bus 0x8f, device 0x1f, function 2.
        assert_eq!(
            kfd_node_pci_id("location_id 36858\ndomain 5\n").as_deref(),
            Some("0005:8f:1f.2")
        );
        // A node that states no location at all cannot be placed on the bus.
        assert_eq!(kfd_node_pci_id("gfx_target_version 90402\n"), None);
    }

    #[test]
    fn kfd_properties_win_over_a_standalone_file() -> Result<()> {
        // Both present: `properties` is what the kernel maintains, so it decides.
        let (root, _) = temp_app_paths("kfd-topology-properties-precedence");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("0"))?;
        fs::write(
            nodes.join("0").join("properties"),
            "gfx_target_version 90402\n",
        )?;
        fs::write(nodes.join("0").join("gfx_target_version"), "110000\n")?;

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_topology_bare_integer_node_order_is_numeric_not_lexical() -> Result<()> {
        // Real KFD node directories are bare integers, and the only ordering that
        // separates numeric from lexical sorting is a multi-digit one: `10` must
        // not come before `2`. The other topology tests use single digits, which
        // sort the same either way, so this is the case that pins the behaviour.
        let (root, _) = temp_app_paths("kfd-topology-bare-integer-order");
        let nodes = root.join("nodes");
        for (node, version) in [("10", "110100\n"), ("2", "90402\n")] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(
                nodes.join(node).join("properties"),
                format!("gfx_target_version {version}"),
            )?;
        }

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        // Node 2 is the lowest-numbered GPU node, so it is the one HIP ordinal 0
        // refers to.
        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_property_value_reads_only_its_own_key() {
        let body = "simd_count 1216\ngfx_target_version 90402\nnum_gws 64\n";
        assert_eq!(
            kfd_property_value(body, "gfx_target_version").as_deref(),
            Some("90402")
        );
        assert_eq!(
            kfd_property_value(body, "simd_count").as_deref(),
            Some("1216")
        );
        // A key that only appears as a prefix of another must not match, and an
        // absent key yields nothing rather than an empty string.
        assert_eq!(kfd_property_value(body, "gfx_target"), None);
        assert_eq!(kfd_property_value(body, "vram_size"), None);
    }

    #[test]
    fn absent_kfd_topology_names_nothing_rather_than_guessing() {
        let missing = workspace_test_artifact_dir().join("rocm-core-kfd-absent-nodes");
        fs::remove_dir_all(&missing).ok();
        assert_eq!(detect_kfd_gfx_target_in(&missing), None);
    }

    #[test]
    fn kfd_gfx_target_version_distinguishes_gpu_from_cpu_nodes() {
        // CPU topology nodes report a zero gfx target version; GPU nodes report a
        // nonzero one.
        assert!(!kfd_gfx_target_version_is_gpu("0"));
        assert!(!kfd_gfx_target_version_is_gpu(""));
        assert!(!kfd_gfx_target_version_is_gpu("not-a-number"));
        assert!(kfd_gfx_target_version_is_gpu("90402"));
        assert!(kfd_gfx_target_version_is_gpu("110000"));
    }
}
