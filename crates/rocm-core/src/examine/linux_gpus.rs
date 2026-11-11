// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Linux GPU enumeration: the `lspci` scan, the `rocminfo` agents, the sysfs
//! fallback and the kernel topology that reconciles them, plus the marketing-name
//! and APU classification they share.

use super::{Examination, Gpu};
use crate::probe_exec::{MEDIUM, run, which};
use std::path::Path;
use std::time::Duration;

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

/// Best-effort `(gfx_target, is_apu)` for an AMD marketing name.
pub(super) fn classify_amd_marketing_name(name: &str) -> (String, bool) {
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
/// APUs: gfx90c (Vega iGPU in Renoir / Lucienne / Cezanne / Barcelo), the
/// RDNA2 iGPUs gfx1033 / gfx1035 / gfx1036 (Van Gogh, Rembrandt / Yellow Carp,
/// Raphael), gfx1103 (Phoenix / Hawk Point) and the gfx115x parts (Strix Point /
/// Strix Halo).
///
/// Their neighbours on both prefixes are *discrete* parts and must not match:
/// gfx1100 / gfx1101 / gfx1102 (Navi 31 / 32 / 33) on gfx110x, and gfx1030 /
/// gfx1031 / gfx1032 / gfx1034 (Navi 21 / 22 / 23 / 24) on gfx103x — which is
/// why the gfx103x arm enumerates rather than taking a threshold the way
/// gfx110x can. A discrete part matching here inflates `has_apu` and suppresses
/// `has_discrete_amd`, which gates the iGPU+dGPU collision fix. (Target ->
/// product per LLVM AMDGPUUsage; `pnp_device_id_table_agrees_about_apus` holds
/// this list against `gfx_target`'s PCI-id table, part by part.)
///
/// Two consumers rely on this, for different reasons:
///
/// - the doctor, to tell an integrated GPU from a discrete one when diagnosing
///   iGPU+dGPU device-visibility collisions;
/// - serve's VRAM reporting, because an APU has no private VRAM — see
///   `vram_capacity_is_meaningful` in the `rocm` binary.
///
/// The second consumer is on **every** platform, not just Windows where the
/// gfx103x arm was added for: a Linux gfx1035 host now skips serve's VRAM
/// warning. That is the intended reading of the change and not a side effect to
/// be narrowed away — Rembrandt has no private VRAM on Linux either, so warning
/// about its "capacity" was reporting a number that does not mean what the
/// warning says it means.
///
/// This answers a question about the *part*, not about the host: a machine can
/// pair an APU with a discrete card, so a true verdict here does not mean every
/// GPU on the host is integrated.
pub fn gfx_is_apu_family(gfx: &str) -> bool {
    let g = gfx.to_lowercase();
    // gfx90c: the Vega iGPU in Renoir / Lucienne / Cezanne / Barcelo. Compared
    // whole (up to a feature suffix) so other gfx90x data-center parts such as
    // gfx90a stay discrete.
    if g == "gfx90c" || g.starts_with("gfx90c:") {
        return true;
    }
    // gfx115x: every Strix part is an APU.
    if gfx_model_digit(&g, "gfx115").is_some() {
        return true;
    }
    // gfx110x: only gfx1103 and above are APUs; gfx1100/1101/1102 are discrete.
    if let Some(digit) = gfx_model_digit(&g, "gfx110") {
        return digit >= 3;
    }
    // gfx103x: non-contiguous, unlike gfx110x's threshold. gfx1033/1035/1036
    // are the RDNA2 APUs (Van Gogh variant aside, gfx_target's own PCI-id table
    // maps them as such); gfx1030/1031/1032/1034 are Navi 21-24 desktop/mobile
    // discrete cards (RX 6000-series, RX 6500 XT). A plain threshold would get
    // this family wrong in both directions.
    if let Some(digit) = gfx_model_digit(&g, "gfx103") {
        return matches!(digit, 3 | 5 | 6);
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

/// Whether an `lspci -nn` line describes a GPU this probe should enumerate.
///
/// Instinct parts report PCI class `1200` ("Processing accelerators"), not a
/// display class: an MI300X enumerates as `Processing accelerators [1200]` with
/// no display class anywhere on the device. Matching only the three display
/// classes therefore skipped every datacenter GPU, so on a bare-metal Instinct
/// host `lspci` contributed nothing and `has_amd_gpu` rested entirely on
/// `rocminfo` or the sysfs fallback — and came back false when neither was
/// reachable. Verified on an 8-GPU MI300X host, where all eight devices read
/// `class=0x120000` in sysfs (EAI-8449).
fn is_lspci_gpu_line(line: &str) -> bool {
    line.contains("VGA compatible controller")
        || line.contains("3D controller")
        || line.contains("Display controller")
        || line.contains("Processing accelerators")
}

pub(super) fn probe_gpus_lspci(e: &mut Examination) {
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
        if !is_lspci_gpu_line(line) {
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

/// The name a GPU carries when the kernel topology is all we have to go on.
///
/// A placeholder, not a name: it says "AMD GPU, source known, model unknown".
/// Both probes that can produce a GPU without a marketing name use it, and
/// [`gpu_name_is_unknown`] reads it back, so it has to be one string rather
/// than three copies that could drift apart. Public so the e2e scenario that
/// asserts a GPU has no model name (`examine-49`) reads this string rather
/// than a copy of it.
pub const KERNEL_TOPOLOGY_GPU_NAME: &str = "AMD GPU (from kernel topology)";

/// Whether this GPU still has no real model name.
///
/// [`KERNEL_TOPOLOGY_GPU_NAME`] has to count as unknown here. The membership
/// pass runs *before* `rocminfo` and stamps that placeholder on every kernel
/// node the PCI scan could not name; if a later probe treated it as a name
/// already present, the placeholder would outrank the marketing name `rocminfo`
/// supplies and the report would be strictly worse than before the membership
/// pass existed -- on exactly the host the pass was written for, the ordinary
/// ROCm container with `rocminfo` but no `pciutils`.
fn gpu_name_is_unknown(name: &str) -> bool {
    name.is_empty() || name == KERNEL_TOPOLOGY_GPU_NAME
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
    apply_rocminfo_gpu_agents(e, &out);
}

/// One GPU agent `rocminfo` listed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RocminfoGpuAgent {
    /// The agent's `Name:`, which for a GPU is its gfx target.
    gfx_target: String,
    /// The agent's `Marketing Name:`, empty when it printed none.
    marketing_name: String,
    /// The `bus:device.function` its `BDFID:` decodes to, `None` when the line
    /// is absent or zero. `rocminfo` prints no PCI domain.
    bus_address: Option<String>,
}

/// The GPU agents in a `rocminfo` listing, in the order it printed them.
fn parse_rocminfo_gpu_agents(out: &str) -> Vec<RocminfoGpuAgent> {
    let mut agents = Vec::new();
    let mut current = RocminfoGpuAgent::default();
    let mut is_gpu = false;
    let mut finish = |agent: RocminfoGpuAgent, is_gpu: bool| {
        if is_gpu && agent.gfx_target.starts_with("gfx") {
            agents.push(agent);
        }
    };
    for line in out.lines() {
        let s = line.trim();
        if s.starts_with("Agent ") {
            finish(std::mem::take(&mut current), std::mem::take(&mut is_gpu));
        } else if let Some(rest) = s.strip_prefix("Name:") {
            current.gfx_target = rest.trim().to_owned();
        } else if let Some(rest) = s.strip_prefix("Marketing Name:") {
            current.marketing_name = rest.trim().to_owned();
        } else if let Some(rest) = s.strip_prefix("Device Type:") {
            is_gpu = rest.contains("GPU");
        } else if let Some(rest) = s.strip_prefix("BDFID:") {
            current.bus_address = parse_rocminfo_bdfid(rest);
        }
    }
    finish(current, is_gpu);
    agents
}

/// The `bus:device.function` a `rocminfo` `BDFID:` value names.
///
/// `BDFID` is decimal and uses the kernel's `pci_dev_id()` encoding, the same
/// as the KFD topology's `location_id`, so both decode through
/// [`crate::host_gpu::pci_bus_device_function`] and cannot disagree about a
/// card. Zero, which `rocminfo` prints for a CPU agent, is no address.
fn parse_rocminfo_bdfid(value: &str) -> Option<String> {
    crate::host_gpu::pci_bus_device_function(value.trim().parse::<u32>().ok()?)
}

/// Split a listed GPU's `domain:bus:device.function` address into its domain
/// and the `bus:device.function` that `rocminfo` can be compared with. An
/// address with no domain (`lspci` without `-D`) is in domain 0. `None` for
/// an entry with no address.
fn split_pci_domain(pci_id: &str) -> Option<(&str, &str)> {
    if pci_id.is_empty() {
        return None;
    }
    match pci_id.split_once(':') {
        Some((domain, rest)) if rest.contains(':') => Some((domain, rest)),
        _ => Some(("0000", pci_id)),
    }
}

/// Why a `rocminfo` agent could not be attributed to a listed GPU.
fn unattributed_agent_label(agent: &RocminfoGpuAgent, why: &str) -> String {
    match &agent.bus_address {
        Some(address) => format!("{} at bus address {address} ({why})", agent.gfx_target),
        None => format!("{} ({why})", agent.gfx_target),
    }
}

/// Write one `rocminfo` agent's reading onto the listed GPU it describes.
fn apply_agent_to_gpu(gpu: &mut Gpu, agent: RocminfoGpuAgent) {
    gpu.is_apu = Some(gfx_is_apu_family(&agent.gfx_target));
    gpu.gfx_target = agent.gfx_target;
    if !agent.marketing_name.is_empty() && gpu_name_is_unknown(&gpu.name) {
        gpu.name = agent.marketing_name;
    }
}

/// List a `rocminfo` agent as a GPU of its own, for an agent no listed entry
/// can be.
fn gpu_from_agent(agent: RocminfoGpuAgent) -> Gpu {
    Gpu {
        name: if agent.marketing_name.is_empty() {
            "AMD GPU".to_owned()
        } else {
            agent.marketing_name
        },
        is_apu: Some(gfx_is_apu_family(&agent.gfx_target)),
        gfx_target: agent.gfx_target,
        is_amd: true,
        ..Gpu::default()
    }
}

/// Fold a `rocminfo` reading into the GPU list, against caller-supplied output.
///
/// Split from the process launch the same way [`apply_kernel_gpu_membership`] is
/// split from the `/sys` read, and for the same reason: the interesting
/// behaviour here is how an agent is matched onto an existing entry, and a test
/// that had to run the real `rocminfo` could only assert it on a host with a
/// GPU. See `a_rocminfo_marketing_name_outranks_the_kernel_topology_placeholder`.
///
/// Matched by PCI address, not by position: `rocminfo` lists the agents the
/// ROCm runtime exposes, which a visibility mask such as `ROCR_VISIBLE_DEVICES`
/// makes a subset of the cards listed here, in an order of its own. Pairing by
/// position wrote one card's model name, target and APU verdict onto another.
///
/// - An agent's `BDFID` is matched against each AMD entry's bus, device and
///   function. `rocminfo` prints no domain, so a match that two entries in
///   different domains share is refused. Several entries at one address are the
///   partitions of one card, and are paired with that address's agents in order.
/// - An agent no entry matches is listed as a GPU of its own when every AMD
///   entry has an address, since it then cannot be any of them; when some
///   entry has none, it could be that one, so nothing is written and a note
///   says so.
/// - Only when no agent states an address at all, and the agents and the AMD
///   entries are equal in number, are they paired by position: the last
///   resort, for a `rocminfo` too old to print `BDFID`. Unequal counts write
///   nothing onto any entry, and a note says so.
/// - An entry no agent matches is left as it was: its own target and no model
///   name, rather than another card's.
fn apply_rocminfo_gpu_agents(e: &mut Examination, out: &str) {
    let agents = parse_rocminfo_gpu_agents(out);
    if agents.is_empty() {
        return;
    }
    let amd_indices: Vec<usize> = e
        .gpus
        .iter()
        .enumerate()
        .filter(|(_, g)| g.is_amd)
        .map(|(idx, _)| idx)
        .collect();
    if amd_indices.is_empty() {
        // Nothing listed for an agent to be confused with.
        e.gpus.extend(agents.into_iter().map(gpu_from_agent));
        return;
    }

    let mut unattributed: Vec<String> = Vec::new();
    if agents.iter().all(|agent| agent.bus_address.is_none()) {
        if agents.len() == amd_indices.len() {
            for (agent, &gpu_idx) in agents.into_iter().zip(&amd_indices) {
                apply_agent_to_gpu(&mut e.gpus[gpu_idx], agent);
            }
        } else {
            unattributed.extend(agents.iter().map(|agent| {
                unattributed_agent_label(agent, "no bus address, and the GPU counts differ")
            }));
        }
    } else {
        let some_entry_has_no_address =
            amd_indices.iter().any(|&idx| e.gpus[idx].pci_id.is_empty());
        let mut claimed = vec![false; amd_indices.len()];
        let mut appended = Vec::new();
        for agent in agents {
            let Some(address) = agent.bus_address.clone() else {
                unattributed.push(unattributed_agent_label(&agent, "no bus address"));
                continue;
            };
            let candidates: Vec<(usize, &str)> = amd_indices
                .iter()
                .enumerate()
                .filter_map(|(slot, &idx)| {
                    let (domain, bdf) = split_pci_domain(&e.gpus[idx].pci_id)?;
                    bdf.eq_ignore_ascii_case(&address).then_some((slot, domain))
                })
                .collect();
            if candidates.is_empty() {
                if some_entry_has_no_address {
                    unattributed.push(unattributed_agent_label(
                        &agent,
                        "no listed GPU has that address, and one has no address at all",
                    ));
                } else {
                    appended.push(gpu_from_agent(agent));
                }
                continue;
            }
            let first_domain = candidates[0].1;
            if candidates
                .iter()
                .any(|(_, domain)| !domain.eq_ignore_ascii_case(first_domain))
            {
                unattributed.push(unattributed_agent_label(
                    &agent,
                    "GPUs in more than one PCI domain have that address",
                ));
                continue;
            }
            let Some(slot) = candidates
                .iter()
                .map(|(slot, _)| *slot)
                .find(|&slot| !claimed[slot])
            else {
                unattributed.push(unattributed_agent_label(
                    &agent,
                    "more agents than listed GPUs at that address",
                ));
                continue;
            };
            claimed[slot] = true;
            apply_agent_to_gpu(&mut e.gpus[amd_indices[slot]], agent);
        }
        e.gpus.extend(appended);
    }

    if !unattributed.is_empty() {
        e.notes.push(format!(
            "{} GPU reading(s) from rocminfo could not be matched to a listed GPU, so their \
             model name and gfx target are not reported for any GPU: {}.",
            unattributed.len(),
            unattributed.join("; ")
        ));
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
///
/// Reached only when the KFD topology could not be read or described no GPU,
/// since [`probe_gpus_kernel_membership_in`] would otherwise have contributed an
/// entry per node already. What is left for this to cover is the host whose
/// target comes from DRM ip-discovery instead — see
/// [`crate::detect_linux_sysfs_gfx_target`].
///
/// `gfx_target` is supplied by the caller rather than read here so the pass has
/// a seam like every other one in the sequence: the hosts it covers are ones a
/// test cannot run on, and it is the only pass whose *output* distinguishes the
/// sequence's correct order from running
/// [`note_unnamed_kernel_topology_gpus`] after it. It stays a function rather
/// than a value so the read is still skipped entirely when an AMD GPU is
/// already listed.
fn probe_gpus_sysfs_fallback(e: &mut Examination, gfx_target: fn() -> Option<String>) {
    if e.gpus.iter().any(|gpu| gpu.is_amd) {
        return;
    }
    let Some(gfx_target) = gfx_target() else {
        return;
    };
    // `is_apu` is left unset rather than guessed: sysfs gives the target, not
    // the packaging, and `summarise_gpu_categories` reads `Some(true)` /
    // `Some(false)` to populate has_apu / has_discrete_amd. Claiming either
    // would be inventing a fact, so both stay false and only has_amd_gpu moves.
    e.gpus.push(Gpu {
        name: KERNEL_TOPOLOGY_GPU_NAME.to_owned(),
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

/// Where the GPU passes that follow the PCI scan read the host from.
///
/// They run in one particular order, stated by [`probe_gpus_after_lspci`], and
/// until this existed that order was a claim only a doc comment made: the
/// sequence read `/sys` and launched `rocminfo` itself, so no test could drive
/// it. Moving [`note_unnamed_kernel_topology_gpus`] above `rocminfo`, below the
/// sysfs fallback, or deleting it outright each left the whole suite green,
/// even though the first reinstates the premature-note defect the pass was
/// split out to remove.
///
/// Naming the sources is what makes the sequence drivable against a planted
/// host, so its shape is pinned by assertion rather than by prose — see
/// `the_unknown_name_note_is_taken_between_rocminfo_and_the_sysfs_fallback`.
/// The passes themselves are the real ones; only where they read is injected.
pub(super) struct GpuProbeSources<'a> {
    /// The KFD topology nodes directory the membership pass reconciles against.
    kfd_nodes: &'a Path,
    /// A `rocminfo` reading to fold in, or `None` to run `rocminfo` here.
    ///
    /// `None` is what the host uses; `Some` exists because a test cannot make a
    /// machine with no GPU produce an agent listing, and the placeholder's fate
    /// turns entirely on whether one arrived.
    rocminfo: Option<&'a str>,
    /// How the sysfs fallback learns the gfx target when nothing else named a
    /// card. See [`probe_gpus_sysfs_fallback`] for why it is a function.
    sysfs_gfx_target: fn() -> Option<String>,
}

impl<'a> GpuProbeSources<'a> {
    /// The real host: the kernel's own KFD topology (read at `kfd_nodes`, which
    /// is [`crate::host_path`]'s answer for it), the `rocminfo` on `PATH`, and
    /// the sysfs target read.
    pub(super) fn host(kfd_nodes: &'a Path) -> Self {
        Self {
            kfd_nodes,
            rocminfo: None,
            sysfs_gfx_target: crate::detect_linux_sysfs_gfx_target,
        }
    }
}

/// Every GPU pass after the PCI scan, in the order they have to run in.
///
/// The order is the whole point of gathering them here: each pass is correct
/// only in one position, and three of the four constraints below are invisible
/// to any test that calls the passes directly in an order it chose itself.
///
/// - The membership pass decides *which* AMD GPUs exist, so it runs before
///   `rocminfo`: the PCI scan over-reports in a container, and `rocminfo` maps
///   its agents onto the surviving entries (by PCI address, and by position
///   only as a count-checked last resort; see [`apply_rocminfo_gpu_agents`]),
///   so it has to see the reconciled list rather than the whole bus. The
///   membership pass is also what gives a kernel-only entry the address that
///   match needs.
/// - [`note_unnamed_kernel_topology_gpus`] runs after `rocminfo`, which is the
///   probe most likely to name a card the PCI scan missed, and before the sysfs
///   fallback, which plants the same placeholder with a note of its own.
pub(super) fn probe_gpus_after_lspci(e: &mut Examination, sources: GpuProbeSources) {
    // The topology status is dropped: both answers call for the same action
    // here. It is returned at all so that a test can tell them apart -- see
    // [`KfdTopology`].
    probe_gpus_kernel_membership_in(e, sources.kfd_nodes);
    match sources.rocminfo {
        Some(out) => apply_rocminfo_gpu_agents(e, out),
        None => probe_gpus_rocminfo(e),
    }
    note_unnamed_kernel_topology_gpus(e);
    probe_gpus_sysfs_fallback(e, sources.sysfs_gfx_target);
}

/// What the kernel topology had to say, as opposed to what was done about it.
///
/// "There is no KFD to read" and "KFD listed no GPU" are different answers that
/// happen to call for the same action -- leave the PCI enumeration standing --
/// and that coincidence is precisely why the distinction went untested. With
/// both collapsed into a no-op, replacing the read's early return with
/// `unwrap_or_default()`, which turns "cannot read" into "read an empty
/// topology", left the entire suite green: `apply_kernel_gpu_membership`'s own
/// `nodes.is_empty()` guard absorbed the difference, so no assertion about the
/// resulting report could ever have noticed.
///
/// Naming the answer separately from the action is what makes it assertable.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KfdTopology {
    /// No KFD to read: `amdgpu` never loaded, the node directory is absent, or
    /// this is a container without it. "Cannot say", not "no GPUs".
    Unreadable,
    /// KFD answered, describing this many GPU nodes. Zero is a real answer --
    /// the driver is there and bound nothing -- and is not the same statement.
    Read(usize),
}

/// KFD is a Linux interface, so off Linux there is no topology to read.
///
/// The gating is not cosmetic: [`probe_gpus_after_lspci`] is reached through a
/// runtime `os_family == "linux"` test, not a `cfg`, so it and this call still
/// have to compile on Windows — and Linux clippy cannot see that. The twin
/// keeps the sequence itself in one place instead of a second, drifting copy.
#[cfg(not(any(target_os = "linux", test)))]
const fn probe_gpus_kernel_membership_in(_e: &mut Examination, _nodes_dir: &Path) {}

/// Make the reported AMD GPU list the set the *kernel* exposes, named from what
/// the PCI scan found, against a caller-supplied nodes directory.
///
/// The PCI bus answers a different question from the kernel. Matching the
/// processing-accelerator class is what finally makes Instinct parts visible to
/// [`probe_gpus_lspci`], but it also makes every card on the *host* bus visible
/// to a container that was passed one of them: an MI300X node reports eight
/// accelerators on the bus while KFD describes only the GPU the container may
/// use. Enumerating from PCI alone would therefore tell that user they have
/// eight GPUs, and `rocm serve` would then fail on a device `examine` had just
/// advertised — worse than the under-detection this replaced, which at least
/// failed honestly.
///
/// So count from the kernel and name from PCI. Only `is_amd` entries are
/// touched: KFD describes AMD compute devices and says nothing about an NVIDIA
/// card, which must survive untouched.
///
/// Split from the `/sys` path exactly as [`crate::kfd_gpu_nodes_in`] is, and for
/// the same reason one level further out: it leaves the *whole* probe drivable
/// from a test — the read, the "cannot say" early return, and the handoff to
/// [`apply_kernel_gpu_membership`] — rather than only the reconcile that handoff
/// calls.
///
/// The seam is not decoration. Before it, the handoff was a statement no test
/// could reach, because its only caller read the host's own `/sys`: deleting the
/// call left the entire workspace suite green, since every unit test drove the
/// inner function directly with a hand-supplied node list. See
/// `the_membership_probe_reconciles_the_topology_it_is_pointed_at`, which fails
/// if that handoff goes away.
#[cfg(any(target_os = "linux", test))]
fn probe_gpus_kernel_membership_in(e: &mut Examination, nodes_dir: &Path) -> KfdTopology {
    let Some(nodes) = crate::kfd_gpu_nodes_in(nodes_dir) else {
        return KfdTopology::Unreadable;
    };
    apply_kernel_gpu_membership(e, &nodes);
    KfdTopology::Read(nodes.len())
}

/// The reconcile itself, against a caller-supplied topology reading.
///
/// Split out from the sysfs read for the same reason as `detect_kfd_gfx_target_in`:
/// the hosts this matters on are the ones a test cannot run on, and reading the
/// real topology from a test would make the assertion depend on the machine.
/// Its caller holds no logic of its own, so pairing this with
/// [`crate::kfd_gpu_nodes_in`] over a planted directory covers the whole path
/// bar the `/sys` path constant — see
/// `the_membership_read_reconciles_a_planted_topology_end_to_end`.
///
/// An empty `nodes` is deliberately a no-op rather than "report no GPUs". A
/// readable topology with no GPU node is what a host whose `amdgpu` failed to
/// load looks like, and "your card is on the bus but the driver did not bind"
/// is the single most useful thing `examine` can say there — so the PCI list
/// stands, and the driver probes explain why nothing is usable.
#[cfg(any(target_os = "linux", test))]
fn apply_kernel_gpu_membership(e: &mut Examination, nodes: &[crate::KfdGpuNode]) {
    if nodes.is_empty() {
        return;
    }
    let (from_pci, others): (Vec<Gpu>, Vec<Gpu>) = std::mem::take(&mut e.gpus)
        .into_iter()
        .partition(|gpu| gpu.is_amd);
    let mut claimed = vec![false; from_pci.len()];

    let mut gpus: Vec<Gpu> = Vec::with_capacity(nodes.len());
    for node in nodes {
        // Matched by address, not by position: the two enumerations order
        // devices independently, and on a partitioned Instinct several KFD
        // nodes legitimately share one physical card's address.
        let matched = from_pci
            .iter()
            .position(|gpu| pci_ids_match(&gpu.pci_id, &node.pci_id));
        let Some(index) = matched else {
            // Kernel-visible but absent from the PCI scan: no `lspci` on PATH,
            // or a node whose address could not be decoded. The device is still
            // usable, so it must be listed -- just without the enriched name.
            //
            // A placeholder, and deliberately not a note: `rocminfo` has not run
            // yet and may well supply the real marketing name, so whether this
            // device is *finally* unnamed is not knowable here. That verdict is
            // left to `note_unnamed_kernel_topology_gpus`, after naming is done.
            gpus.push(Gpu {
                name: KERNEL_TOPOLOGY_GPU_NAME.to_owned(),
                gfx_target: node.gfx_target.clone(),
                pci_id: node.pci_id.clone(),
                is_amd: true,
                // Left unset rather than guessed, for the same reason
                // `probe_gpus_sysfs_fallback` leaves it unset: KFD gives the
                // target, not the packaging.
                is_apu: None,
            });
            continue;
        };
        claimed[index] = true;
        let mut gpu = from_pci[index].clone();
        // The node's own target, so an APU+dGPU host labels each card with the
        // target that belongs to it. Never an overwrite: `rocminfo` has not run
        // yet here, but a target `lspci` resolved from the marketing name is a
        // statement about this device and the node's is only a better one when
        // there is nothing to compare it against.
        if gpu.gfx_target.is_empty() {
            gpu.gfx_target = node.gfx_target.clone();
        }
        gpus.push(gpu);
    }

    let unexposed: Vec<&str> = from_pci
        .iter()
        .zip(&claimed)
        .filter(|(_, claimed)| !**claimed)
        .map(|(gpu, _)| gpu.pci_id.as_str())
        .collect();
    if !unexposed.is_empty() {
        // Not dropped silently: the addresses still belong in the report,
        // because "the bus has it and the kernel does not" is a diagnosis.
        e.notes.push(format!(
            "{} AMD PCI device(s) are on the bus but not exposed by the kernel here, so they \
             are not listed as GPUs: {}. In a container this is expected - only the \
             passed-through GPUs are usable.",
            unexposed.len(),
            unexposed.join(", ")
        ));
    }
    e.gpus = gpus;
    e.gpus.extend(others);
}

/// Report which kernel-sourced GPUs ended up with no model name, once nothing
/// left can supply one.
///
/// Separate from [`apply_kernel_gpu_membership`], which is where the count used
/// to be taken, because that pass runs before `probe_gpus_rocminfo`. Counting
/// there asserted "their marketing name is unknown" while the probe that most
/// often knows the name had not yet run -- so the note fired even on hosts whose
/// report went on to name every card.
///
/// Must run after `probe_gpus_rocminfo` and before [`probe_gpus_sysfs_fallback`]:
/// the first is what resolves the placeholder, and the second plants the same
/// placeholder again with a note of its own, which this would otherwise count a
/// second time. Both bounds are asserted rather than merely asked for — see
/// `the_unknown_name_note_is_taken_between_rocminfo_and_the_sysfs_fallback`,
/// which fails if this call moves to either side of them or goes away.
fn note_unnamed_kernel_topology_gpus(e: &mut Examination) {
    let unnamed = e
        .gpus
        .iter()
        .filter(|gpu| gpu.is_amd && gpu.name == KERNEL_TOPOLOGY_GPU_NAME)
        .count();
    if unnamed > 0 {
        e.notes.push(format!(
            "{unnamed} GPU(s) were taken from the kernel topology because the PCI enumeration \
             did not list them; their marketing name is unknown."
        ));
    }
}

/// Whether two PCI addresses name the same device. An empty address matches
/// nothing: it means "unknown", not "wildcard".
#[cfg(any(target_os = "linux", test))]
const fn pci_ids_match(left: &str, right: &str) -> bool {
    !left.is_empty() && left.eq_ignore_ascii_case(right)
}

pub(super) fn summarise_gpu_categories(e: &mut Examination) {
    e.has_amd_gpu = e.gpus.iter().any(|g| g.is_amd);
    e.has_apu = e.gpus.iter().any(|g| g.is_amd && g.is_apu == Some(true));
    e.has_discrete_amd = e.gpus.iter().any(|g| g.is_amd && g.is_apu == Some(false));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
        // A target is deliberately on offer: the point is that the fallback
        // declines it, not that there was nothing to take.
        probe_gpus_sysfs_fallback(&mut e, || Some("gfx942".to_owned()));
        assert_eq!(e.gpus.len(), 1, "the fallback must not add a second entry");
        assert_eq!(e.gpus[0].pci_id, "1002:74a1");
        assert!(
            e.notes.is_empty(),
            "a no-op fallback should not annotate the report"
        );
    }

    /// The eight MI300X accelerators an MI300X host's `lspci -nn -D` lists,
    /// verbatim down to the `Device` name its `pci.ids` gives them.
    fn mi300x_bus_gpus() -> Vec<Gpu> {
        [
            "0000:11:00.0",
            "0000:2f:00.0",
            "0000:46:00.0",
            "0000:5d:00.0",
            "0000:8b:00.0",
            "0000:aa:00.0",
            "0000:c2:00.0",
            "0000:da:00.0",
        ]
        .into_iter()
        .map(|pci_id| Gpu {
            name: "Advanced Micro Devices, Inc. [AMD/ATI] Device".to_owned(),
            gfx_target: String::new(),
            pci_id: pci_id.to_owned(),
            is_amd: true,
            is_apu: Some(false),
        })
        .collect()
    }

    #[test]
    fn the_report_lists_the_gpus_the_kernel_exposes_not_every_card_on_the_bus() {
        // The container this regressed in: `lspci` reads the *host* bus and
        // finds all eight MI300X accelerators, while KFD describes only the one
        // passed through. Enumerating from PCI told that user they had eight
        // GPUs, seven of which `rocm serve` could not open.
        let mut e = Examination {
            gpus: mi300x_bus_gpus(),
            ..Examination::default()
        };
        apply_kernel_gpu_membership(
            &mut e,
            &[crate::KfdGpuNode {
                pci_id: "0000:5d:00.0".to_owned(),
                gfx_target: "gfx942".to_owned(),
            }],
        );

        assert_eq!(e.gpus.len(), 1, "only the exposed GPU may be listed");
        // Matched by address, not by position: the exposed card is the fourth
        // on the bus, so taking the first entry would name the wrong device.
        assert_eq!(e.gpus[0].pci_id, "0000:5d:00.0");
        assert_eq!(e.gpus[0].gfx_target, "gfx942");
        assert!(
            e.gpus[0].name.contains("Advanced Micro Devices"),
            "the surviving entry keeps the name lspci gave it: {:?}",
            e.gpus[0].name
        );
        // The seven are not erased from the report, only from `gpus[]`: "the
        // bus has it and the kernel does not" is itself a diagnosis.
        let note = e.notes.join("\n");
        assert!(
            note.contains("not exposed by the kernel") && note.contains("0000:11:00.0"),
            "the unexposed devices must still be accounted for: {:?}",
            e.notes
        );
        // ROCMAI-543: `rocm examine` prints every note verbatim, and this one
        // carried an em dash before " only the passed-through GPUs". A Windows
        // console on the OEM code page renders that as mojibake, so the whole
        // note has to stay ASCII -- not only the substrings asserted above.
        assert!(
            note.is_ascii(),
            "examine notes must be ASCII-only; non-ASCII punctuation renders as mojibake on \
             a Windows console using the OEM code page (ROCMAI-543): {:?}",
            e.notes
        );
    }

    #[test]
    fn a_kernel_visible_gpu_the_pci_scan_missed_is_still_listed() {
        // The reverse case: no `lspci` on PATH, so nothing names the device --
        // but the kernel exposes it and it is perfectly usable, so dropping it
        // would under-report exactly the way this whole change is meant to fix.
        let mut e = Examination::default();
        apply_kernel_gpu_membership(
            &mut e,
            &[
                crate::KfdGpuNode {
                    pci_id: "0000:11:00.0".to_owned(),
                    gfx_target: "gfx942".to_owned(),
                },
                crate::KfdGpuNode {
                    pci_id: "0000:2f:00.0".to_owned(),
                    gfx_target: "gfx942".to_owned(),
                },
            ],
        );

        assert_eq!(e.gpus.len(), 2);
        assert!(e.gpus.iter().all(|gpu| gpu.is_amd));
        // The address still comes through, because KFD states it even when
        // lspci is unavailable to confirm it.
        assert_eq!(e.gpus[0].pci_id, "0000:11:00.0");
        assert_eq!(e.gpus[1].gfx_target, "gfx942");
        // The note is no longer the membership pass's to emit -- `rocminfo` runs
        // after it and may name these -- so the verdict comes from the pass that
        // runs once naming is final. Nothing named them here, so it must fire.
        note_unnamed_kernel_topology_gpus(&mut e);
        assert!(
            e.notes.join("\n").contains("marketing name is unknown"),
            "an unnamed entry must say so: {:?}",
            e.notes
        );
    }

    #[test]
    fn each_kernel_node_labels_its_own_card_and_leaves_other_vendors_alone() {
        // An APU + a discrete card. Reading the target per node is what makes
        // this safe: a single host-wide answer would stamp the APU's gfx1103
        // onto the dGPU, and diagnose's iGPU/dGPU check splits on exactly this
        // field to tell the user which GPU to pin.
        let mut e = Examination {
            gpus: vec![
                Gpu {
                    name: "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"
                        .to_owned(),
                    gfx_target: String::new(),
                    pci_id: "0000:03:00.0".to_owned(),
                    is_amd: true,
                    is_apu: Some(false),
                },
                Gpu {
                    name: "NVIDIA Corporation Device".to_owned(),
                    gfx_target: String::new(),
                    pci_id: "0000:46:00.0".to_owned(),
                    is_amd: false,
                    is_apu: Some(false),
                },
                Gpu {
                    name: "Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1".to_owned(),
                    gfx_target: "gfx1103".to_owned(),
                    pci_id: "0000:64:00.0".to_owned(),
                    is_amd: true,
                    is_apu: Some(true),
                },
            ],
            ..Examination::default()
        };
        apply_kernel_gpu_membership(
            &mut e,
            &[
                crate::KfdGpuNode {
                    pci_id: "0000:64:00.0".to_owned(),
                    gfx_target: "gfx1103".to_owned(),
                },
                crate::KfdGpuNode {
                    pci_id: "0000:03:00.0".to_owned(),
                    gfx_target: "gfx1100".to_owned(),
                },
            ],
        );

        let apu = &e.gpus[0];
        let discrete = &e.gpus[1];
        assert_eq!(apu.pci_id, "0000:64:00.0");
        assert_eq!(apu.gfx_target, "gfx1103");
        assert_eq!(apu.is_apu, Some(true), "lspci's packaging verdict survives");
        assert_eq!(discrete.pci_id, "0000:03:00.0");
        assert_eq!(
            discrete.gfx_target, "gfx1100",
            "the discrete card takes its own node's target, not the APU's"
        );
        // KFD says nothing about an NVIDIA card, so the entry must survive.
        let nvidia = e
            .gpus
            .iter()
            .find(|gpu| !gpu.is_amd)
            .expect("the NVIDIA entry must survive");
        assert_eq!(nvidia.pci_id, "0000:46:00.0");
        assert_eq!(nvidia.gfx_target, "", "a non-AMD entry gets no AMD target");
        assert!(
            e.notes.is_empty(),
            "a topology that accounts for every AMD card needs no note: {:?}",
            e.notes
        );
    }

    #[test]
    fn a_target_the_pci_scan_already_resolved_is_never_overwritten() {
        // lspci resolves a target from the marketing name, which is a statement
        // about that specific device. The node's answer is only a better one
        // when there is nothing to compare it against.
        let mut e = Examination {
            gpus: vec![Gpu {
                name: "Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1".to_owned(),
                gfx_target: "gfx1103".to_owned(),
                pci_id: "0000:64:00.0".to_owned(),
                is_amd: true,
                is_apu: Some(true),
            }],
            ..Examination::default()
        };
        apply_kernel_gpu_membership(
            &mut e,
            &[crate::KfdGpuNode {
                pci_id: "0000:64:00.0".to_owned(),
                gfx_target: "gfx1150".to_owned(),
            }],
        );
        assert_eq!(e.gpus[0].gfx_target, "gfx1103");
    }

    #[test]
    fn a_kernel_that_exposes_no_gpu_leaves_the_pci_enumeration_standing() {
        // A host whose `amdgpu` never bound: the topology is readable and
        // describes nothing. "Your card is on the bus but the driver did not
        // bind" is the most useful thing examine can say there, so the PCI
        // list must survive for the driver probes to explain.
        let mut e = Examination {
            gpus: mi300x_bus_gpus(),
            ..Examination::default()
        };
        apply_kernel_gpu_membership(&mut e, &[]);
        assert_eq!(
            e.gpus.len(),
            8,
            "an empty topology must not empty the report"
        );
        assert!(e.notes.is_empty());
    }

    #[test]
    fn the_membership_read_reconciles_a_planted_topology_end_to_end() {
        // Everything above hands `apply_kernel_gpu_membership` a node list
        // directly. This drives the real read as well -- sysfs bytes in, report
        // out -- so the `location_id` decode and the reconcile are covered
        // together rather than each assuming the other.
        // One GPU node, at the fourth accelerator on the bus: what the
        // container sees when it is passed 0000:5d:00.0 out of the host's eight.
        let (root, nodes) = plant_kfd_topology("membership", &[(23808, 90402)]);
        let read = crate::kfd_gpu_nodes_in(&nodes).expect("the planted topology must be readable");
        std::fs::remove_dir_all(&root).ok();

        let mut e = Examination {
            gpus: mi300x_bus_gpus(),
            ..Examination::default()
        };
        apply_kernel_gpu_membership(&mut e, &read);
        summarise_gpu_categories(&mut e);

        assert_eq!(e.gpus.len(), 1, "the report lists what the kernel exposes");
        assert_eq!(e.gpus[0].pci_id, "0000:5d:00.0");
        assert_eq!(e.gpus[0].gfx_target, "gfx942");
        assert!(e.has_amd_gpu);
        assert!(e.has_discrete_amd);
    }

    /// Plant a KFD topology: one CPU node, then one GPU node per
    /// `(location_id, gfx_target_version)` pair. Returns the `nodes` directory;
    /// the caller removes `root` once the read is done.
    fn plant_kfd_topology(tag: &str, gpu_nodes: &[(u32, u32)]) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "rocm-core-examine-kfd-{tag}-{}",
            crate::test_support::unique_suffix()
        ));
        let nodes = root.join("nodes");
        std::fs::create_dir_all(nodes.join("0")).expect("plant the CPU node");
        std::fs::write(
            nodes.join("0").join("properties"),
            "cpu_cores_count 56\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
        )
        .expect("plant the CPU node properties");
        for (index, (location_id, version)) in gpu_nodes.iter().enumerate() {
            let dir = nodes.join((index + 1).to_string());
            std::fs::create_dir_all(&dir).expect("plant the GPU node");
            std::fs::write(
                dir.join("properties"),
                format!(
                    "simd_count 1216\ngfx_target_version {version}\nlocation_id {location_id}\n\
                     domain 0\n"
                ),
            )
            .expect("plant the GPU node properties");
        }
        (root, nodes)
    }

    #[test]
    fn plant_kfd_topology_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| {
            plant_kfd_topology("same-label", &[]).0
        });
    }

    #[test]
    fn the_membership_probe_reconciles_the_topology_it_is_pointed_at() {
        // The *wiring*, not the reconcile. Every test above hands
        // `apply_kernel_gpu_membership` a node list directly, which proves that
        // function and says nothing about whether the probe ever reaches it --
        // deleting the handoff left the whole workspace suite green. So drive
        // the probe's own entry point instead, over a planted topology, and the
        // read, the handoff and the gfx-target fill are all covered at once.
        let (root, nodes) = plant_kfd_topology("membership-probe", &[(23808, 90402)]);
        let mut e = Examination {
            gpus: mi300x_bus_gpus(),
            ..Examination::default()
        };
        probe_gpus_kernel_membership_in(&mut e, &nodes);
        std::fs::remove_dir_all(&root).ok();
        summarise_gpu_categories(&mut e);

        assert_eq!(
            e.gpus.len(),
            1,
            "the probe must reduce the eight-card bus to the one node the kernel exposes: {:#?}",
            e.gpus
        );
        assert_eq!(e.gpus[0].pci_id, "0000:5d:00.0");
        // `mi300x_bus_gpus` leaves every target empty, the way lspci does when
        // `pci.ids` spells an Instinct part "Device 74a1". The node's target is
        // the only thing that can fill it, so this is exactly the statement the
        // review found untested.
        assert_eq!(
            e.gpus[0].gfx_target, "gfx942",
            "the probe must fill the target its own topology attributes to that card"
        );
        assert!(e.has_amd_gpu);
    }

    #[test]
    fn the_membership_probe_leaves_the_report_alone_when_the_topology_is_unreadable() {
        // "Unreadable" and "readable but empty" are different answers -- "cannot
        // say" versus "the driver bound nothing" -- and each is a no-op for its
        // own reason. Asserting only that the report is untouched cannot tell
        // them apart: it passed even with the early return replaced by
        // `unwrap_or_default()`, which collapses the first into the second,
        // because `apply_kernel_gpu_membership`'s own empty-guard absorbed it.
        //
        // So discriminate on something only the early return can produce: the
        // reconcile must never be *entered* at all. A planted topology with a
        // CPU node and no GPU node is the readable-but-empty case, and it is
        // given an empty PCI list so that entering the reconcile would be
        // observable -- if that path ran, it would take the `nodes.is_empty()`
        // guard. The unreadable case is given the eight-card bus, which the
        // reconcile would rewrite to nothing had it been reached with an empty
        // node list.
        let mut unreadable = Examination {
            gpus: mi300x_bus_gpus(),
            ..Examination::default()
        };
        let verdict = probe_gpus_kernel_membership_in(
            &mut unreadable,
            &std::env::temp_dir().join("rocm-cli-absent-kfd-topology-nodes"),
        );
        assert_eq!(
            unreadable.gpus,
            mi300x_bus_gpus(),
            "an unreadable topology must leave the PCI enumeration untouched"
        );
        assert!(
            unreadable.notes.is_empty(),
            "notes: {:#?}",
            unreadable.notes
        );
        // The discriminating assertion. Every assertion above is satisfied by
        // the reconcile's `nodes.is_empty()` guard just as well as by the early
        // return, which is why they could not fail when the two were collapsed.
        // This one can only hold if the read itself reported "cannot say".
        assert_eq!(
            verdict,
            KfdTopology::Unreadable,
            "a missing node directory is \"cannot say\", which is not the same answer as \
             \"KFD listed no GPU\" -- collapsing them must fail here"
        );

        // The other case, through the same entry point: a topology the kernel
        // *does* expose, listing zero GPUs. Also a no-op on the report, but for
        // its own reason -- the driver bound nothing, which is a fact, where the
        // case above is the absence of one.
        let (root, nodes) = plant_kfd_topology("membership-readable-empty", &[]);
        let mut empty = Examination {
            gpus: mi300x_bus_gpus(),
            ..Examination::default()
        };
        let verdict = probe_gpus_kernel_membership_in(&mut empty, &nodes);
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(
            verdict,
            KfdTopology::Read(0),
            "a readable topology with no GPU node must report itself read, not unreadable"
        );
        assert_eq!(
            empty.gpus,
            mi300x_bus_gpus(),
            "a readable topology with no GPU node must leave the PCI list standing: the card is \
             on the bus and the driver did not bind, which is the single most useful thing \
             `examine` can say there"
        );
        assert!(empty.notes.is_empty(), "notes: {:#?}", empty.notes);
    }

    #[test]
    fn a_rocminfo_marketing_name_outranks_the_kernel_topology_placeholder() {
        // The ordinary ROCm container: `rocminfo` present, `pciutils` absent --
        // the shape this whole change was written for. The PCI scan enumerates
        // nothing, so the membership pass contributes the kernel's node under a
        // placeholder name, and `rocminfo` runs *after* it.
        //
        // The placeholder must therefore not count as a name already present.
        // When it did, it outranked the marketing name and the user saw
        // "AMD GPU (from kernel topology)" on a host that previously reported
        // "AMD Instinct MI300X" -- a regression the change inflicted on its own
        // target scenario.
        let mut e = Examination::default();
        apply_kernel_gpu_membership(
            &mut e,
            &[crate::KfdGpuNode {
                pci_id: "0000:5d:00.0".to_owned(),
                gfx_target: "gfx942".to_owned(),
            }],
        );
        assert_eq!(
            e.gpus.len(),
            1,
            "the kernel's node must be listed even with no PCI scan to name it"
        );
        assert_eq!(e.gpus[0].name, KERNEL_TOPOLOGY_GPU_NAME);

        apply_rocminfo_gpu_agents(
            &mut e,
            "Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n\
             Agent 2\n  Name:  gfx942\n  Marketing Name:  AMD Instinct MI300X\n  \
             Device Type:  GPU\n",
        );
        note_unnamed_kernel_topology_gpus(&mut e);

        assert_eq!(
            e.gpus.len(),
            1,
            "`rocminfo` must map its agent onto the kernel's entry, not add a second: {:#?}",
            e.gpus
        );
        assert_eq!(
            e.gpus[0].name, "AMD Instinct MI300X",
            "the marketing name must survive the placeholder, not lose to it"
        );
        assert_eq!(e.gpus[0].gfx_target, "gfx942");
        assert_eq!(
            e.gpus[0].pci_id, "0000:5d:00.0",
            "the address the kernel supplied must not be lost in the naming"
        );
        assert!(
            !e.notes.join("\n").contains("marketing name is unknown"),
            "the name is known -- `rocminfo` just supplied it -- so the note must not fire: {:#?}",
            e.notes
        );
    }

    /// The notes [`note_unnamed_kernel_topology_gpus`] emits, and only those.
    ///
    /// Matched on "their marketing name is unknown" rather than the looser
    /// "marketing name is unknown", because `probe_gpus_sysfs_fallback` emits a
    /// note of its own ending "PCI id and marketing name are unknown" — and
    /// telling the two apart is exactly what catches the note being taken after
    /// the fallback instead of before it.
    fn unknown_name_notes(e: &Examination) -> Vec<&String> {
        e.notes
            .iter()
            .filter(|note| note.contains("their marketing name is unknown"))
            .collect()
    }

    #[test]
    fn the_unknown_name_note_is_taken_between_rocminfo_and_the_sysfs_fallback() {
        // The test above proves the *passes* compose when a caller runs them in
        // the right order. It cannot notice the real call site running them in
        // the wrong one -- it chooses the order itself. So drive
        // `probe_gpus_after_lspci`, which is the sequence `Examination::probe`
        // runs, against planted hosts. Each scenario below is failed by exactly
        // one way of misplacing `note_unnamed_kernel_topology_gpus`:
        //
        //   A. above `probe_gpus_rocminfo` -- the premature-note defect this
        //      pass was split out to remove: a named card gets noted as unnamed.
        //   B. deleted -- a genuinely unnamed card goes unremarked.
        //   C. below `probe_gpus_sysfs_fallback` -- the fallback's own
        //      placeholder is counted on top of the fallback's own note.

        // A. The ordinary ROCm container: KFD exposes the one passed-through
        //    MI300X, `pciutils` is absent so the PCI scan named nothing, and
        //    `rocminfo` supplies the marketing name. Naming is finished by the
        //    time the note is taken, so there is nothing to report.
        let (root, nodes) = plant_kfd_topology("sequence-named", &[(23808, 90402)]);
        let mut named = Examination::default();
        probe_gpus_after_lspci(
            &mut named,
            GpuProbeSources {
                kfd_nodes: &nodes,
                rocminfo: Some(
                    "Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n\
                     Agent 2\n  Name:  gfx942\n  Marketing Name:  AMD Instinct MI300X\n  \
                     Device Type:  GPU\n",
                ),
                sysfs_gfx_target: || Some("gfx942".to_owned()),
            },
        );
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(
            named.gpus.len(),
            1,
            "the kernel exposes one card and `rocminfo` names it: {:#?}",
            named.gpus
        );
        assert_eq!(
            named.gpus[0].name, "AMD Instinct MI300X",
            "the marketing name must reach the report"
        );
        assert_eq!(
            unknown_name_notes(&named),
            Vec::<&String>::new(),
            "the note must be taken after `rocminfo`, which named this card: {:#?}",
            named.notes
        );

        // B. The same host with a `rocminfo` that lists no GPU agent. Nothing
        //    ever named the card, so the note is the only thing that tells the
        //    user why the report says "AMD GPU (from kernel topology)".
        let (root, nodes) = plant_kfd_topology("sequence-unnamed", &[(23808, 90402)]);
        let mut unnamed = Examination::default();
        probe_gpus_after_lspci(
            &mut unnamed,
            GpuProbeSources {
                kfd_nodes: &nodes,
                rocminfo: Some("Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n"),
                sysfs_gfx_target: || Some("gfx942".to_owned()),
            },
        );
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(unnamed.gpus[0].name, KERNEL_TOPOLOGY_GPU_NAME);
        assert_eq!(
            unknown_name_notes(&unnamed).len(),
            1,
            "an entry nothing could name must say so exactly once: {:#?}",
            unnamed.notes
        );

        // C. No KFD topology to read at all, so the membership pass is a no-op
        //    and the sysfs fallback is what finds the card -- the DRM
        //    ip-discovery host it exists for. The fallback plants the same
        //    placeholder *and* explains it in a note of its own, so the count
        //    must already have been taken: reporting it again would tell the
        //    user twice, in two different wordings, about one card.
        let mut fallback = Examination::default();
        probe_gpus_after_lspci(
            &mut fallback,
            GpuProbeSources {
                kfd_nodes: &std::env::temp_dir().join("rocm-cli-absent-kfd-for-sequence"),
                rocminfo: Some("Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n"),
                sysfs_gfx_target: || Some("gfx1103".to_owned()),
            },
        );
        assert_eq!(
            fallback.gpus.len(),
            1,
            "the fallback must supply the card the topology could not: {:#?}",
            fallback.gpus
        );
        assert_eq!(fallback.gpus[0].name, KERNEL_TOPOLOGY_GPU_NAME);
        assert_eq!(
            unknown_name_notes(&fallback),
            Vec::<&String>::new(),
            "the fallback already explains its own placeholder; the note must be taken \
             before it, not after: {:#?}",
            fallback.notes
        );
        assert_eq!(
            fallback.notes.len(),
            1,
            "one card, one explanation: {:#?}",
            fallback.notes
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
    fn lspci_name_extraction() {
        let line = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]";
        assert_eq!(
            extract_lspci_name(line),
            "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"
        );
    }

    #[test]
    fn lspci_matches_instinct_processing_accelerator_class() {
        // An MI300X carries no display class at all: it enumerates under PCI
        // class 1200. Matching only the display classes skipped it, which is
        // how a bare-metal Instinct host reported has_amd_gpu: false (EAI-8449).
        let mi300x = "0000:11:00.0 Processing accelerators [1200]: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X] [1002:74a1] (rev 02)";
        assert!(is_lspci_gpu_line(mi300x));
        assert_eq!(
            extract_lspci_name(mi300x),
            "Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X]"
        );
        // Instinct is discrete, and has_discrete_amd depends on that verdict.
        assert!(
            !classify_amd_marketing_name(&extract_lspci_name(mi300x)).1,
            "an Instinct part must not be classified as an APU"
        );

        // The display classes still match.
        assert!(is_lspci_gpu_line(
            "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]"
        ));
        assert!(is_lspci_gpu_line(
            "0000:01:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2204]"
        ));
        assert!(is_lspci_gpu_line(
            "0000:00:02.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Device [1002:164e]"
        ));

        // The MI300X host also exposes an AMD-vendor PCI bridge per GPU; those
        // are not GPUs and must stay out of the enumeration.
        assert!(!is_lspci_gpu_line(
            "0000:10:00.0 PCI bridge [0604]: Advanced Micro Devices, Inc. [AMD/ATI] Device [1002:1501]"
        ));
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

        // Vega iGPU (Renoir / Lucienne / Cezanne / Barcelo) is an APU; the other
        // gfx90x parts are not.
        assert!(gfx_is_apu_family("gfx90c"));
        assert!(gfx_is_apu_family("GFX90C"));
        assert!(gfx_is_apu_family("gfx90c:xnack-"));
        assert!(!gfx_is_apu_family("gfx90a"));
        assert!(!gfx_is_apu_family("gfx908"));
        assert!(!gfx_is_apu_family("gfx906"));
        assert!(!gfx_is_apu_family("gfx90"));
        assert!(!gfx_is_apu_family(""));

        // gfx103x is non-contiguous, unlike gfx110x's threshold: gfx1033
        // (Van Gogh) / gfx1035 (Rembrandt/Barcelo-R) / gfx1036 are RDNA2
        // APUs, while gfx1030/1031/1032 (Navi 21/22/23, RX 6000 series) and
        // gfx1034 (Navi 24, RX 6500 XT) are discrete. A Rembrandt laptop
        // misclassified here would report as a discrete GPU and suppress
        // `fix-9-igpu-dgpu` detection on an iGPU+dGPU host.
        assert!(gfx_is_apu_family("gfx1033"));
        assert!(gfx_is_apu_family("gfx1035"));
        assert!(gfx_is_apu_family("gfx1036"));
        assert!(!gfx_is_apu_family("gfx1030"));
        assert!(!gfx_is_apu_family("gfx1031"));
        assert!(!gfx_is_apu_family("gfx1032"));
        assert!(!gfx_is_apu_family("gfx1034"));
    }

    /// Every target `gfx_target`'s PNP table can return has to be classified by
    /// `gfx_is_apu_family`, because the Windows GPU probe composes the two.
    ///
    /// The per-device tests above each pin one row of that table. This pins the
    /// *invariant* between the two tables, which is what actually broke: adding
    /// a PCI id without teaching `gfx_is_apu_family` about its target silently
    /// gives the part the fallthrough answer — `false`, "discrete" — with no
    /// test anywhere disagreeing. That is exactly how gfx1033 / gfx1035 /
    /// gfx1036 came to be labelled "RDNA2 APUs" by one table and discrete by
    /// the other.
    #[test]
    fn pnp_device_id_table_agrees_about_apus() {
        for (pnp, gfx, integrated) in [
            // Vega iGPUs: Renoir, Lucienne, Cezanne, Barcelo.
            ("PCI\\VEN_1002&DEV_1636&SUBSYS_08A71028", "gfx90c", true),
            ("PCI\\VEN_1002&DEV_164C&SUBSYS_380D103C", "gfx90c", true),
            ("PCI\\VEN_1002&DEV_1638&SUBSYS_0A2E1025", "gfx90c", true),
            ("PCI\\VEN_1002&DEV_15E7&SUBSYS_8A431043", "gfx90c", true),
            // RDNA2 iGPUs: Van Gogh, Rembrandt / Yellow Carp, Raphael.
            ("PCI\\VEN_1002&DEV_163F&SUBSYS_11111111", "gfx1033", true),
            ("PCI\\VEN_1002&DEV_164D&SUBSYS_22222222", "gfx1035", true),
            ("PCI\\VEN_1002&DEV_1681&SUBSYS_33333333", "gfx1035", true),
            ("PCI\\VEN_1002&DEV_164E&SUBSYS_44444444", "gfx1036", true),
            // RDNA3 / RDNA3.5 iGPUs: Phoenix and Strix.
            ("PCI\\VEN_1002&DEV_15BF&SUBSYS_55555555", "gfx1103", true),
            ("PCI\\VEN_1002&DEV_1114&SUBSYS_66666666", "gfx1152", true),
            // Discrete cards through the same door, so a blanket `true` fails
            // here too: Navi 21 / 22 / 23 / 24 and Navi 48.
            ("PCI\\VEN_1002&DEV_73A0&SUBSYS_77777777", "gfx1030", false),
            ("PCI\\VEN_1002&DEV_73C0&SUBSYS_88888888", "gfx1031", false),
            ("PCI\\VEN_1002&DEV_73E0&SUBSYS_99999999", "gfx1032", false),
            ("PCI\\VEN_1002&DEV_743F&SUBSYS_AAAAAAAA", "gfx1034", false),
            ("PCI\\VEN_1002&DEV_7550&SUBSYS_2435148C", "gfx1201", false),
        ] {
            assert_eq!(
                crate::host_gpu::gfx_target_from_pnp_device_id(pnp),
                Some(gfx),
                "{pnp}: the PCI-id table moved under this test"
            );
            assert_eq!(
                gfx_is_apu_family(gfx),
                integrated,
                "{pnp} -> {gfx}: the two tables disagree about whether this \
                 part is integrated"
            );
        }
    }

    /// A gcnArchName feature suffix or upper-casing must not change a gfx103x
    /// verdict, the same way the gfx110x and gfx90c arms already guarantee.
    #[test]
    fn gfx103x_verdict_survives_case_and_a_feature_suffix() {
        assert!(gfx_is_apu_family("GFX1035"));
        assert!(gfx_is_apu_family("gfx1035:xnack-"));
        assert!(!gfx_is_apu_family("GFX1034"));
        assert!(!gfx_is_apu_family("gfx1032:xnack-"));
        // Degenerate input never matches: no model digit, no verdict.
        assert!(!gfx_is_apu_family("gfx103"));
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

    // -- `rocminfo` agents matched onto the listed GPUs (ROCMAI-40) ---------
    //
    // One test per row of the state matrix in `apply_rocminfo_gpu_agents`'s
    // doc comment, plus the shapes that cross rows: a subset, a reordering,
    // an APU beside a dGPU, a domain clash and a partitioned card.

    /// A `rocminfo` listing in the shape the real tool prints: a CPU agent with
    /// `BDFID: 0`, then one GPU agent per `(gfx, marketing name, BDFID)`, each
    /// value space-padded to the tool's column. `None` omits the `BDFID` line,
    /// as a `rocminfo` too old to print it would.
    fn rocminfo_listing(gpus: &[(&str, &str, Option<u32>)]) -> String {
        use std::fmt::Write as _;
        let mut out = String::from(
            "*******\nAgent 1\n*******\n  Name:                    AMD Ryzen 9 7950X\n  \
             Marketing Name:          AMD Ryzen 9 7950X\n  Device Type:             CPU\n  \
             BDFID:                   0\n",
        );
        for (index, (gfx, marketing, bdfid)) in gpus.iter().enumerate() {
            write!(
                out,
                "*******\nAgent {}\n*******\n  Name:                    {gfx}\n  \
                 Marketing Name:          {marketing}\n  Device Type:             GPU\n",
                index + 2
            )
            .expect("writing to a String cannot fail");
            if let Some(bdfid) = bdfid {
                writeln!(out, "  BDFID:                   {bdfid}")
                    .expect("writing to a String cannot fail");
            }
        }
        out
    }

    /// Run the GPU passes after the PCI scan over a planted kernel topology
    /// (one GPU node per `(location_id, gfx_target_version)`) and a `rocminfo`
    /// listing, with an empty PCI scan in front: the ordinary ROCm container.
    fn examine_planted_host(tag: &str, nodes: &[(u32, u32)], rocminfo: &str) -> Examination {
        let (root, nodes) = plant_kfd_topology(tag, nodes);
        let mut e = Examination::default();
        probe_gpus_after_lspci(
            &mut e,
            GpuProbeSources {
                kfd_nodes: &nodes,
                rocminfo: Some(rocminfo),
                sysfs_gfx_target: || None,
            },
        );
        std::fs::remove_dir_all(&root).ok();
        summarise_gpu_categories(&mut e);
        e
    }

    fn gpu_at<'a>(e: &'a Examination, pci_id: &str) -> &'a Gpu {
        e.gpus
            .iter()
            .find(|gpu| gpu.pci_id == pci_id)
            .unwrap_or_else(|| panic!("no GPU listed at {pci_id}: {:#?}", e.gpus))
    }

    /// The note `apply_rocminfo_gpu_agents` takes for readings it could not
    /// attribute, and only that one.
    fn unattributed_notes(e: &Examination) -> Vec<&String> {
        e.notes
            .iter()
            .filter(|note| note.contains("from rocminfo could not be matched"))
            .collect()
    }

    // `location_id` / `BDFID` 768 is 03:00.0, 1024 is 04:00.0, 50176 is c4:00.0.
    const AT_03: u32 = 768;
    const AT_04: u32 = 1024;
    const AT_C4: u32 = 50176;

    #[test]
    fn a_rocminfo_bdfid_line_in_the_real_shape_decodes_to_its_bus_address() {
        // Real `rocminfo` pads the value to a column and prints `BDFID: 0` for
        // a CPU agent. 49_920 is the GPU at c3:00.0 on a ROCm 7.14 host.
        let agents = parse_rocminfo_gpu_agents(&rocminfo_listing(&[(
            "gfx1100",
            "AMD Radeon RX 7900 XTX",
            Some(49_920),
        )]));
        assert_eq!(
            agents,
            vec![RocminfoGpuAgent {
                gfx_target: "gfx1100".to_owned(),
                marketing_name: "AMD Radeon RX 7900 XTX".to_owned(),
                bus_address: Some("c3:00.0".to_owned()),
            }],
            "the CPU agent is not a GPU, and the GPU's BDFID is its address"
        );
        assert_eq!(parse_rocminfo_bdfid("                   0"), None);
        assert_eq!(parse_rocminfo_bdfid(" not-a-number"), None);
        // A missing line is no address, not a zero one.
        let without = parse_rocminfo_gpu_agents(&rocminfo_listing(&[("gfx1100", "X", None)]));
        assert_eq!(without[0].bus_address, None);
    }

    #[test]
    fn a_kfd_location_id_and_a_rocminfo_bdfid_name_the_same_card_alike() {
        // The producer and the consumer of the match: the kernel topology
        // gives each listed GPU its address from `location_id`, `rocminfo`
        // gives each agent one from `BDFID`, and the match is only right if
        // one number decodes to one address on both sides. Includes a
        // non-zero device and function, which a zero-only decode would miss.
        for dev_id in [AT_03, AT_04, AT_C4, 0x8ffa, 23808] {
            let (root, nodes) = plant_kfd_topology("bdf-tie", &[(dev_id, 110_000)]);
            let read = crate::kfd_gpu_nodes_in(&nodes).expect("planted topology must be readable");
            std::fs::remove_dir_all(&root).ok();
            let bdf = parse_rocminfo_bdfid(&format!("  {dev_id}"))
                .unwrap_or_else(|| panic!("BDFID {dev_id} must decode"));
            let (domain, kfd_bdf) =
                split_pci_domain(&read[0].pci_id).expect("the node must carry an address");
            assert_eq!(domain, "0000");
            assert_eq!(
                kfd_bdf, bdf,
                "location_id and BDFID {dev_id} must decode alike"
            );
        }
    }

    #[test]
    fn a_gpu_the_rocm_runtime_cannot_see_does_not_borrow_another_gpus_reading() {
        // Rows (a) and (f), and the reported defect: two GPUs in the kernel
        // topology, and a `rocminfo` that sees only the second (a visibility
        // mask). Pairing by position wrote the second card's model name,
        // target and APU verdict onto the first.
        let e = examine_planted_host(
            "rocminfo-subset",
            &[(AT_03, 110_000), (AT_04, 110_001)],
            &rocminfo_listing(&[("gfx1101", "AMD Radeon RX 7800 XT", Some(AT_04))]),
        );
        assert_eq!(e.gpus.len(), 2, "{:#?}", e.gpus);
        let seen = gpu_at(&e, "0000:04:00.0");
        assert_eq!(seen.name, "AMD Radeon RX 7800 XT");
        assert_eq!(seen.gfx_target, "gfx1101");
        assert_eq!(seen.is_apu, Some(false));
        let unseen = gpu_at(&e, "0000:03:00.0");
        assert_eq!(
            unseen.name, KERNEL_TOPOLOGY_GPU_NAME,
            "the card rocminfo did not describe keeps no model name"
        );
        assert_eq!(unseen.gfx_target, "gfx1100", "and keeps its own target");
        assert_eq!(
            unseen.is_apu, None,
            "and claims no packaging it was never given"
        );
        assert_eq!(unattributed_notes(&e), Vec::<&String>::new());
        // The unknown-name note now counts the right card: one is unnamed.
        assert_eq!(unknown_name_notes(&e).len(), 1, "{:#?}", e.notes);
    }

    #[test]
    fn rocminfo_agents_listed_in_another_order_each_reach_their_own_card() {
        // Row (a) with the two lists in opposite orders: every agent matched,
        // each by its own address.
        let e = examine_planted_host(
            "rocminfo-reordered",
            &[(AT_03, 110_000), (AT_04, 110_001)],
            &rocminfo_listing(&[
                ("gfx1101", "AMD Radeon RX 7800 XT", Some(AT_04)),
                ("gfx1100", "AMD Radeon RX 7900 XTX", Some(AT_03)),
            ]),
        );
        assert_eq!(e.gpus.len(), 2, "{:#?}", e.gpus);
        assert_eq!(gpu_at(&e, "0000:03:00.0").name, "AMD Radeon RX 7900 XTX");
        assert_eq!(gpu_at(&e, "0000:03:00.0").gfx_target, "gfx1100");
        assert_eq!(gpu_at(&e, "0000:04:00.0").name, "AMD Radeon RX 7800 XT");
        assert_eq!(gpu_at(&e, "0000:04:00.0").gfx_target, "gfx1101");
        assert!(e.notes.is_empty(), "{:#?}", e.notes);
    }

    #[test]
    fn an_apu_the_runtime_hides_is_still_reported_as_an_apu() {
        // The APU+dGPU host with the iGPU masked out of the runtime: the
        // dGPU's "discrete" verdict landed on the APU, so `has_apu` read
        // false, which `rocm diagnose`'s iGPU/dGPU check reads.
        let mut e = examine_planted_host(
            "rocminfo-apu-hidden",
            &[(AT_C4, 110_003), (AT_03, 110_000)],
            &rocminfo_listing(&[("gfx1100", "AMD Radeon RX 7900 XTX", Some(AT_03))]),
        );
        let apu = gpu_at(&e, "0000:c4:00.0");
        assert_eq!(apu.gfx_target, "gfx1103");
        assert_eq!(apu.name, KERNEL_TOPOLOGY_GPU_NAME);
        let dgpu = gpu_at(&e, "0000:03:00.0");
        assert_eq!(dgpu.gfx_target, "gfx1100");
        assert_eq!(dgpu.is_apu, Some(false));
        assert!(e.has_discrete_amd);
        // The APU's verdict is not invented either way: KFD gives the target,
        // not the packaging, so `has_apu` stays unclaimed rather than wrong.
        assert_eq!(apu.is_apu, None);
        assert!(!e.has_apu);

        // With `lspci` naming both (which carries the packaging), the hidden
        // APU keeps the verdict the scan gave it.
        e = Examination {
            gpus: vec![
                Gpu {
                    name: "AMD Phoenix".to_owned(),
                    gfx_target: "gfx1103".to_owned(),
                    pci_id: "0000:c4:00.0".to_owned(),
                    is_apu: Some(true),
                    is_amd: true,
                },
                Gpu {
                    name: "Navi 31 [Radeon RX 7900 XT/7900 XTX]".to_owned(),
                    gfx_target: String::new(),
                    pci_id: "0000:03:00.0".to_owned(),
                    is_apu: Some(false),
                    is_amd: true,
                },
            ],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_listing(&[("gfx1100", "AMD Radeon RX 7900 XTX", Some(AT_03))]),
        );
        summarise_gpu_categories(&mut e);
        assert_eq!(gpu_at(&e, "0000:c4:00.0").gfx_target, "gfx1103");
        assert_eq!(gpu_at(&e, "0000:c4:00.0").is_apu, Some(true));
        assert_eq!(gpu_at(&e, "0000:03:00.0").gfx_target, "gfx1100");
        assert!(e.has_apu, "the hidden APU must still be reported as one");
        assert!(e.has_discrete_amd);
    }

    #[test]
    fn an_agent_at_an_address_no_listed_gpu_has_is_listed_as_its_own_gpu() {
        // Row (b): every listed AMD GPU has an address and none is the
        // agent's, so the agent cannot be any of them and is added, as before.
        let e = examine_planted_host(
            "rocminfo-unmatched-addressed",
            &[(AT_03, 110_000)],
            &rocminfo_listing(&[
                ("gfx1100", "AMD Radeon RX 7900 XTX", Some(AT_03)),
                ("gfx1101", "AMD Radeon RX 7800 XT", Some(AT_04)),
            ]),
        );
        assert_eq!(e.gpus.len(), 2, "{:#?}", e.gpus);
        assert_eq!(gpu_at(&e, "0000:03:00.0").name, "AMD Radeon RX 7900 XTX");
        let added = e
            .gpus
            .iter()
            .find(|gpu| gpu.name == "AMD Radeon RX 7800 XT")
            .expect("the unmatched agent must be listed");
        assert_eq!(added.gfx_target, "gfx1101");
        assert!(
            added.pci_id.is_empty(),
            "rocminfo gives no domain to build one from"
        );
        assert!(unattributed_notes(&e).is_empty(), "{:#?}", e.notes);
    }

    #[test]
    fn an_agent_that_could_be_the_gpu_with_no_address_is_noted_not_added() {
        // Row (c): a listed GPU has no address (a KFD node with no
        // `location_id`), so an agent that matches no address could be that
        // very card. Adding it would list one card twice; writing onto it
        // would be a guess. Neither: the entry stays as it was, with a note.
        let mut e = Examination {
            gpus: vec![Gpu {
                name: KERNEL_TOPOLOGY_GPU_NAME.to_owned(),
                gfx_target: "gfx1100".to_owned(),
                pci_id: String::new(),
                is_apu: None,
                is_amd: true,
            }],
            ..Examination::default()
        };
        let before = e.gpus.clone();
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_listing(&[("gfx1101", "AMD Radeon RX 7800 XT", Some(AT_04))]),
        );
        assert_eq!(e.gpus, before, "nothing added and nothing written");
        let notes = unattributed_notes(&e);
        assert_eq!(notes.len(), 1, "{:#?}", e.notes);
        assert!(
            notes[0].starts_with("1 GPU reading(s)")
                && notes[0].contains("gfx1101 at bus address 04:00.0"),
            "the note must name the reading it set aside: {notes:?}"
        );
        assert!(
            notes[0].is_ascii(),
            "examine notes must be ASCII-only: {notes:?}"
        );
    }

    #[test]
    fn agents_without_an_address_pair_by_position_only_when_the_counts_agree() {
        // Row (d): a `rocminfo` that prints no `BDFID` and as many GPU agents
        // as there are listed AMD GPUs -- the documented last resort.
        let e = examine_planted_host(
            "rocminfo-no-bdfid-equal",
            &[(AT_03, 110_000), (AT_04, 110_001)],
            &rocminfo_listing(&[
                ("gfx1100", "AMD Radeon RX 7900 XTX", None),
                ("gfx1101", "AMD Radeon RX 7800 XT", None),
            ]),
        );
        assert_eq!(gpu_at(&e, "0000:03:00.0").name, "AMD Radeon RX 7900 XTX");
        assert_eq!(gpu_at(&e, "0000:04:00.0").name, "AMD Radeon RX 7800 XT");
        assert!(e.notes.is_empty(), "{:#?}", e.notes);
    }

    #[test]
    fn agents_without_an_address_write_nothing_when_the_counts_differ() {
        // Row (e): no address and one agent for two cards. Which card it
        // describes is unknowable, so neither gets its reading; both keep
        // their own target and no model name, and a note says why.
        let e = examine_planted_host(
            "rocminfo-no-bdfid-unequal",
            &[(AT_03, 110_000), (AT_04, 110_001)],
            &rocminfo_listing(&[("gfx1101", "AMD Radeon RX 7800 XT", None)]),
        );
        assert_eq!(e.gpus.len(), 2, "{:#?}", e.gpus);
        for (pci_id, own_target) in [("0000:03:00.0", "gfx1100"), ("0000:04:00.0", "gfx1101")] {
            let gpu = gpu_at(&e, pci_id);
            assert_eq!(gpu.name, KERNEL_TOPOLOGY_GPU_NAME, "{pci_id}");
            assert_eq!(gpu.gfx_target, own_target, "{pci_id}");
            assert_eq!(gpu.is_apu, None, "{pci_id}");
        }
        let notes = unattributed_notes(&e);
        assert_eq!(notes.len(), 1, "{:#?}", e.notes);
        assert!(
            notes[0].contains("gfx1101 (no bus address, and the GPU counts differ)"),
            "{notes:?}"
        );

        // Row (e)'s exception: with no AMD GPU listed at all there is nothing
        // to confuse the agent with, so it is listed, as before.
        let mut empty = Examination::default();
        apply_rocminfo_gpu_agents(
            &mut empty,
            &rocminfo_listing(&[("gfx1101", "AMD Radeon RX 7800 XT", None)]),
        );
        assert_eq!(empty.gpus.len(), 1, "{:#?}", empty.gpus);
        assert_eq!(empty.gpus[0].name, "AMD Radeon RX 7800 XT");
        assert_eq!(empty.gpus[0].gfx_target, "gfx1101");
        assert!(empty.notes.is_empty());
    }

    #[test]
    fn an_address_shared_across_pci_domains_is_refused() {
        // `rocminfo` prints no domain, so 03:00.0 in domain 0 and in domain 1
        // are indistinguishable to it. Picking either would be a guess.
        let mut e = Examination {
            gpus: ["0000:03:00.0", "0001:03:00.0"]
                .into_iter()
                .map(|pci_id| Gpu {
                    name: "Advanced Micro Devices, Inc. [AMD/ATI] Device".to_owned(),
                    gfx_target: String::new(),
                    pci_id: pci_id.to_owned(),
                    is_apu: Some(false),
                    is_amd: true,
                })
                .collect(),
            ..Examination::default()
        };
        let before = e.gpus.clone();
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_listing(&[("gfx942", "AMD Instinct MI300X", Some(AT_03))]),
        );
        assert_eq!(e.gpus, before, "neither card may take the reading");
        let notes = unattributed_notes(&e);
        assert_eq!(notes.len(), 1, "{:#?}", e.notes);
        assert!(notes[0].contains("more than one PCI domain"), "{notes:?}");
    }

    #[test]
    fn an_agent_with_no_address_beside_agents_with_one_is_set_aside() {
        // Once any agent states an address, matching is by address, and an
        // agent that states none cannot be placed: it is not paired by
        // position with whatever entry is left over.
        let e = examine_planted_host(
            "rocminfo-mixed-bdfid",
            &[(AT_03, 110_000), (AT_04, 110_001)],
            &rocminfo_listing(&[
                ("gfx1101", "AMD Radeon RX 7800 XT", Some(AT_04)),
                ("gfx1100", "AMD Radeon RX 7900 XTX", None),
            ]),
        );
        assert_eq!(e.gpus.len(), 2, "{:#?}", e.gpus);
        assert_eq!(gpu_at(&e, "0000:04:00.0").name, "AMD Radeon RX 7800 XT");
        assert_eq!(gpu_at(&e, "0000:03:00.0").name, KERNEL_TOPOLOGY_GPU_NAME);
        let notes = unattributed_notes(&e);
        assert_eq!(notes.len(), 1, "{:#?}", e.notes);
        assert!(notes[0].contains("gfx1100 (no bus address)"), "{notes:?}");
    }

    #[test]
    fn the_partitions_of_one_card_pair_within_that_card() {
        // A partitioned Instinct: KFD lists one node per partition, all at the
        // card's address, and `rocminfo` one agent per partition. Each pairs
        // within its card, in order; a second card's agent goes to it alone.
        let e = examine_planted_host(
            "rocminfo-partitions",
            &[(AT_03, 90402), (AT_03, 90402), (AT_04, 90402)],
            &rocminfo_listing(&[
                ("gfx942", "AMD Instinct MI300X", Some(AT_04)),
                ("gfx942", "AMD Instinct MI300X", Some(AT_03)),
                ("gfx942", "AMD Instinct MI300X", Some(AT_03)),
            ]),
        );
        assert_eq!(e.gpus.len(), 3, "{:#?}", e.gpus);
        assert!(
            e.gpus.iter().all(|gpu| gpu.name == "AMD Instinct MI300X"),
            "every partition is named: {:#?}",
            e.gpus
        );
        assert!(e.notes.is_empty(), "{:#?}", e.notes);

        // One agent too many at an address: the card is already listed once
        // per partition the kernel exposes, so the extra is noted, not added.
        let extra = examine_planted_host(
            "rocminfo-partitions-extra",
            &[(AT_03, 90402)],
            &rocminfo_listing(&[
                ("gfx942", "AMD Instinct MI300X", Some(AT_03)),
                ("gfx942", "AMD Instinct MI300X", Some(AT_03)),
            ]),
        );
        assert_eq!(extra.gpus.len(), 1, "{:#?}", extra.gpus);
        assert_eq!(extra.gpus[0].name, "AMD Instinct MI300X");
        let notes = unattributed_notes(&extra);
        assert_eq!(notes.len(), 1, "{:#?}", extra.notes);
        assert!(
            notes[0].contains("more agents than listed GPUs at that address"),
            "{notes:?}"
        );
    }
}
