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

/// Best-effort `(gfx_target, is_apu)` for an AMD marketing name or `lspci`
/// codename, or `("", false)` when the name resolves no part.
///
/// The target comes from the crate's one name table,
/// [`crate::gfx_target::gfx_target_from_amd_marketing_name`], and the packaging
/// from classifying that target. `examine` used to keep a second, smaller copy
/// of that table here, which disagreed with it about Krackan Point (gfx1150
/// here, gfx1152 there and in the device-id table), plus a list of codename
/// fragments that answered "APU?" without ever naming a target, so a Renoir
/// iGPU came back with an empty `gfx_target`.
///
/// A name that resolves nothing answers `false`, "not an APU", and that is a
/// deliberate default for a PCI row rather than a fact. The only consumer that
/// acts on the verdict is `check_9_igpu_dgpu_collision`, which fires on
/// `has_apu && has_discrete_amd`:
///
/// - reporting an unidentified *discrete* card as "cannot say" would clear
///   `has_discrete_amd` beside every card the tables do not name -- every RX
///   5000, Vega 56/64 and Radeon VII -- on exactly the hybrid host that check
///   exists for;
/// - reporting an unidentified *integrated* GPU as discrete sets
///   `has_discrete_amd` on an iGPU-only host, where the check still cannot fire
///   because `has_apu` is false. The cost is one wrong field in `rocm examine
///   --json`, and the integrated set is small and named in the table, so an
///   unresolved row is far more often a discrete card.
///
/// `rocminfo` and `hipInfo` revise this verdict when they report a target whose
/// packaging is known ([`gfx_apu_verdict`]); where they do not, it stands.
pub(super) fn classify_amd_marketing_name(name: &str) -> (String, bool) {
    let gfx = crate::gfx_target::gfx_target_from_amd_marketing_name(name).unwrap_or_default();
    (gfx.to_owned(), gfx_is_apu_family(gfx))
}

/// The packaging an AMD marketing name states, or `None` when the name resolves
/// no part.
///
/// Unlike [`classify_amd_marketing_name`], this does not default an
/// unresolved name to "not an APU": it is for a GPU that has no PCI row, where
/// that default would be read as "a discrete GPU" with nothing behind it.
fn marketing_name_apu_verdict(name: &str) -> Option<bool> {
    let (target, is_apu) = classify_amd_marketing_name(name);
    (!target.is_empty()).then_some(is_apu)
}

/// Whether a gfx target belongs to an AMD APU family.
///
/// APUs: the Vega iGPUs gfx902 / gfx909 (Raven Ridge, Picasso / Raven2) and
/// gfx90c (Renoir / Lucienne / Cezanne / Barcelo), the RDNA2 iGPUs gfx1033 /
/// gfx1035 / gfx1036 / gfx1037 (Van Gogh, Rembrandt / Yellow Carp, Raphael,
/// Mendocino), gfx1103 (Phoenix / Hawk Point) and the gfx115x parts (Strix Point
/// / Strix Halo / Krackan Point).
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
    // The Vega iGPUs: gfx90c in Renoir / Lucienne / Cezanne / Barcelo, and
    // gfx902 / gfx909 in Raven Ridge and Picasso / Raven2. Compared whole (up to
    // a feature or generic-target suffix, both of which follow the base
    // target) so the gfx90x data-center parts beside them -- gfx900,
    // gfx906, gfx908, gfx90a -- stay discrete. No lookup table produces gfx902
    // or gfx909: they reach the CLI from KFD's packed `gfx_target_version`
    // (90002, 90009), through `gfx_target_from_gc_version`.
    if matches!(
        g.split([':', '-']).next(),
        Some("gfx90c" | "gfx902" | "gfx909")
    ) {
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
    // maps them as such), and gfx1037 is Mendocino (Ryzen 7020), which KFD and
    // DRM ip-discovery report as GC 10.3.7; gfx1030/1031/1032/1034 are Navi
    // 21-24 desktop/mobile discrete cards (RX 6000-series, RX 6500 XT). A plain
    // threshold would get this family wrong in both directions.
    if let Some(digit) = gfx_model_digit(&g, "gfx103") {
        return matches!(digit, 3 | 5 | 6 | 7);
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

/// Discrete gfx targets: every one a card or accelerator with memory of its own.
///
/// This is the "no" half of [`gfx_apu_verdict`], which needs one because
/// [`gfx_is_apu_family`] answers `false` both for a discrete part and for a
/// target nothing here has heard of. It lists the discrete targets the lookup
/// tables can produce plus the ones the CLI installs for;
/// `every_target_a_lookup_table_produces_has_an_apu_verdict` fails when a table
/// gains a target that neither this list nor [`gfx_is_apu_family`] classifies.
///
/// gfx942 covers both the MI300X accelerator and the MI300A, which is an APU
/// with unified HBM. It is listed discrete because MI300X is what ROCm hosts
/// run and what the CLI has always answered; telling the two apart needs a
/// signal the target does not carry. gfx940 and gfx941, the earlier MI300
/// targets that `normalize_therock_family` folds into the same `gfx94X-dcgpu`
/// family, are listed for the same reason.
const DISCRETE_GFX_TARGETS: &[&str] = &[
    // GCN / CDNA.
    "gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950",
    // RDNA1: Navi 10 / 12 / 14.
    "gfx1010", "gfx1011", "gfx1012", // RDNA2: Navi 21 / 22 / 23 / 24.
    "gfx1030", "gfx1031", "gfx1032", "gfx1034", // RDNA3: Navi 31 / 32 / 33.
    "gfx1100", "gfx1101", "gfx1102", // RDNA4: Navi 44 / 48.
    "gfx1200", "gfx1201", // The gfx125X-dcgpu family.
    "gfx1250",
];

/// Whether a gfx target is an APU, or `None` when this crate does not know.
///
/// The three-way answer is the point. A probe that folds a target into a
/// verdict another probe already reached needs to tell "this is a discrete
/// card" from "never heard of this target": the second is an absence of
/// evidence, and must leave the earlier verdict standing rather than replace
/// it with [`gfx_is_apu_family`]'s default `false`. Reading `rocminfo` or
/// `hipInfo` must not make the report worse than not reading it.
///
/// A feature suffix (`gfx1036:xnack-`), a generic-target suffix
/// (`gfx1036-generic`) and case do not change the answer.
///
/// Both halves read the base target, so the suffixes are stripped once, here,
/// rather than by each half in its own way.
pub(super) fn gfx_apu_verdict(gfx: &str) -> Option<bool> {
    let base: String = gfx
        .trim()
        .to_ascii_lowercase()
        .chars()
        .take_while(char::is_ascii_alphanumeric)
        .collect();
    if gfx_is_apu_family(&base) {
        return Some(true);
    }
    DISCRETE_GFX_TARGETS
        .contains(&base.as_str())
        .then_some(false)
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
pub(super) fn is_lspci_gpu_line(line: &str) -> bool {
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
    apply_lspci_gpus(e, &out);
}

/// Fold an `lspci -nn -D` listing into the GPU list, against caller-supplied
/// output.
///
/// Split from the process launch for the same reason
/// [`apply_rocminfo_gpu_agents`] is: what matters is how a row is classified,
/// and a test that had to run the real `lspci` could only assert it on the
/// hardware it happened to run on.
pub(super) fn apply_lspci_gpus(e: &mut Examination, out: &str) {
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
        // `-nn` prints the PCI device id on the line, so ask it first, for the
        // reason the Windows probe asks the PNP id first: the id names the
        // ASIC, the marketing string only the brand. It matters most where the
        // name says nothing at all -- for an id `pci.ids` does not know, `lspci`
        // prints the bare word "Device".
        let (gfx_guess, is_apu_guess) = match extract_lspci_device_id(line)
            .and_then(|device_id| crate::gfx_target::gfx_target_from_amd_pci_device_id(&device_id))
        {
            Some(target) => (target.to_owned(), gfx_is_apu_family(target)),
            None => classify_amd_marketing_name(&name),
        };
        e.gpus.push(Gpu {
            name,
            gfx_target: gfx_guess,
            pci_id,
            is_apu: Some(is_apu_guess),
            is_amd: true,
        });
    }
}

/// The AMD PCI device id an `lspci -nn` line carries: the four hex digits after
/// `1002:` in its `[vendor:device]` tag.
///
/// Only AMD's vendor id is accepted. The tag is what tells a device id apart
/// from the other bracketed numbers on the line -- the PCI class (`[0300]`) and
/// any subsystem text `pci.ids` supplies -- and a number lifted from one of
/// those would decode to an unrelated part.
pub(super) fn extract_lspci_device_id(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    let start = lower.rfind("[1002:")? + "[1002:".len();
    let device_id: String = lower[start..]
        .chars()
        .take_while(char::is_ascii_hexdigit)
        .take(5)
        .collect();
    (device_id.len() == 4 && lower[start + 4..].starts_with(']')).then_some(device_id)
}

/// Pull the marketing name out of an `lspci -nn` line: the text between the
/// controller-kind `]:` and the trailing `[vendor:device]`.
pub(super) fn extract_lspci_name(line: &str) -> String {
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
/// than three copies that could drift apart.
const KERNEL_TOPOLOGY_GPU_NAME: &str = "AMD GPU (from kernel topology)";

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

/// Fold a `rocminfo` reading into the GPU list, against caller-supplied output.
///
/// Split from the process launch the same way [`apply_kernel_gpu_membership`] is
/// split from the `/sys` read, and for the same reason: the interesting
/// behaviour here is how an agent is matched onto an existing entry, and a test
/// that had to run the real `rocminfo` could only assert it on a host with a
/// GPU. See `a_rocminfo_marketing_name_outranks_the_kernel_topology_placeholder`.
pub(super) fn apply_rocminfo_gpu_agents(e: &mut Examination, out: &str) {
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
    let pairing_is_unambiguous = amd_indices.len() == 1 && gfx_targets.len() == 1;
    for (idx, (gfx, marketing)) in gfx_targets.into_iter().enumerate() {
        if let Some(&gpu_idx) = amd_indices.get(idx) {
            let gpu = &mut e.gpus[gpu_idx];
            gpu.gfx_target = gfx.clone();
            let named_is_apu = marketing_name_apu_verdict(&marketing);
            if !marketing.is_empty() && gpu_name_is_unknown(&gpu.name) {
                gpu.name = marketing;
            }
            // The target names the silicon, so where its packaging is known it
            // outranks the PCI scan's guess from a marketing string. A target
            // nothing here classifies is not evidence of a discrete GPU, so it
            // leaves the PCI scan's verdict standing: before this, an agent
            // reporting an unlisted target turned an APU the PCI scan had
            // identified into a discrete card, clearing `has_apu` and with it
            // the iGPU+dGPU collision diagnosis. (Latent while #393 makes the
            // parser drop every agent on a real host; live the day it is fixed.)
            //
            // A row that still has no verdict at all -- a kernel-topology
            // placeholder whose target nothing classifies, in a container
            // with no `pciutils` -- takes the agent's marketing name when that
            // resolves a part, exactly as an unpaired agent does below, but
            // only when the pairing cannot be wrong: one AMD row and one
            // agent. The agent carries no identifier this parser reads, so on
            // a multi-GPU host the pairing is by position (#508), and a
            // position is not evidence about which card a name belongs to; the
            // verdict stays unknown there rather than becoming a guess.
            if let Some(is_apu) = gfx_apu_verdict(&gfx) {
                gpu.is_apu = Some(is_apu);
            } else if gpu.is_apu.is_none() && pairing_is_unambiguous {
                gpu.is_apu = named_is_apu;
            }
        } else {
            // An agent with no PCI row and no kernel-topology node to pair
            // with: KFD unreadable, and no `pciutils`. Same precedence, over the
            // evidence there is: the target when its packaging is known, else
            // the agent's own marketing name when that resolves a part. When
            // neither answers, `is_apu` stays unset rather than defaulting to
            // `false`, which `summarise_gpu_categories` would read as "a
            // discrete GPU" -- the conclusion `probe_gpus_sysfs_fallback`
            // reaches for the same reason.
            let is_apu = gfx_apu_verdict(&gfx).or_else(|| marketing_name_apu_verdict(&marketing));
            e.gpus.push(Gpu {
                name: if marketing.is_empty() {
                    "AMD GPU".to_owned()
                } else {
                    marketing
                },
                gfx_target: gfx,
                is_amd: true,
                is_apu,
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
    // `summarise_gpu_categories` reads `Some(true)` / `Some(false)` to populate
    // has_apu / has_discrete_amd, so `is_apu` is set only where the target's
    // packaging is known ([`gfx_apu_verdict`]). For a target nothing classifies
    // it stays unset: claiming either would be inventing a fact, so both flags
    // stay false and only has_amd_gpu moves.
    let is_apu = gfx_apu_verdict(&gfx_target);
    e.gpus.push(Gpu {
        name: KERNEL_TOPOLOGY_GPU_NAME.to_owned(),
        gfx_target,
        is_amd: true,
        is_apu,
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
    pub(super) kfd_nodes: &'a Path,
    /// A `rocminfo` reading to fold in, or `None` to run `rocminfo` here.
    ///
    /// `None` is what the host uses; `Some` exists because a test cannot make a
    /// machine with no GPU produce an agent listing, and the placeholder's fate
    /// turns entirely on whether one arrived.
    pub(super) rocminfo: Option<&'a str>,
    /// How the sysfs fallback learns the gfx target when nothing else named a
    /// card. See [`probe_gpus_sysfs_fallback`] for why it is a function.
    pub(super) sysfs_gfx_target: fn() -> Option<String>,
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
///   its agents onto the surviving entries by position, so it has to see the
///   reconciled list rather than the whole bus.
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
                // The target's own verdict where its packaging is known, and
                // unset rather than guessed where it is not -- the same reading
                // `probe_gpus_sysfs_fallback` takes of the same KFD target.
                is_apu: gfx_apu_verdict(&node.gfx_target),
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
            // The verdict follows the target it now carries. A PCI row that
            // resolved no target holds the scan's "not an APU" default (see
            // `classify_amd_marketing_name`), which is a guess; a target whose
            // packaging is known replaces it, the same precedence the
            // `rocminfo` fold applies. This is the only route by which a Raven
            // Ridge or Picasso iGPU is classified on a real host: no lookup
            // table names them, and KFD's packed `gfx_target_version` is what
            // yields gfx902 / gfx909. An unknown target leaves the guess alone.
            if let Some(is_apu) = gfx_apu_verdict(&gpu.gfx_target) {
                gpu.is_apu = Some(is_apu);
            }
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
        // What the fallback produces for a target nothing classifies: enough
        // to stop reporting "no AMD GPU", and no more. `is_apu: None` is
        // deliberate -- the topology names a target whose packaging this crate
        // does not know -- so the APU and discrete tallies must both stay false
        // rather than guess.
        let mut e = Examination {
            os_family: "linux".to_owned(),
            gpus: vec![Gpu {
                name: "AMD GPU (from kernel topology)".to_owned(),
                gfx_target: "gfx1170".to_owned(),
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

    /// The Vega and RDNA2 APUs that reach the CLI only as a target synthesised
    /// from a GC version -- KFD's packed `gfx_target_version` 90002, 90009 and
    /// 100307 -- and that a prefix rule never reached: Raven Ridge, Picasso /
    /// Raven2 and Mendocino. Their gfx90x and gfx103x neighbours stay discrete.
    #[test]
    fn raven_picasso_and_mendocino_are_apus() {
        for apu in [
            "gfx902",
            "gfx909",
            "gfx1037",
            "GFX902",
            "gfx909:xnack-",
            "gfx1037:xnack-",
        ] {
            assert!(gfx_is_apu_family(apu), "{apu}");
        }
        for discrete in ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx1034", "gfx1030"] {
            assert!(!gfx_is_apu_family(discrete), "{discrete}");
        }
    }

    /// "Not an APU" and "never heard of it" are different answers, and the
    /// folds that revise a verdict need to tell them apart.
    #[test]
    fn an_unclassified_target_has_no_apu_verdict() {
        assert_eq!(gfx_apu_verdict("gfx1036"), Some(true));
        assert_eq!(gfx_apu_verdict("gfx1036-generic"), Some(true));
        assert_eq!(gfx_apu_verdict("gfx1100"), Some(false));
        assert_eq!(gfx_apu_verdict("GFX942:sramecc+:xnack-"), Some(false));
        // The earlier MI300 targets, in the same TheRock family as gfx942.
        assert_eq!(gfx_apu_verdict("gfx940"), Some(false));
        assert_eq!(gfx_apu_verdict("gfx941"), Some(false));
        assert_eq!(gfx_apu_verdict("gfx1100-generic"), Some(false));
        // The Vega iGPUs are matched whole rather than by a prefix digit, so
        // the suffix has to come off before that match, not only before the
        // discrete list.
        assert_eq!(gfx_apu_verdict("gfx90c-generic"), Some(true));
        assert_eq!(gfx_apu_verdict("gfx902-generic"), Some(true));
        assert_eq!(gfx_apu_verdict("GFX909-generic:xnack-"), Some(true));
        assert!(gfx_is_apu_family("gfx90c-generic"));
        assert_eq!(gfx_apu_verdict("gfx1170"), None);
        assert_eq!(gfx_apu_verdict(""), None);
    }

    /// The `lspci -nn` device id comes from AMD's vendor tag and nowhere else.
    #[test]
    fn the_lspci_device_id_is_read_from_the_amd_vendor_tag() {
        assert_eq!(
            extract_lspci_device_id(
                "0000:c5:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
                 [AMD/ATI] Krackan Point [Radeon 860M] [1002:1114] (rev c1)"
            )
            .as_deref(),
            Some("1114")
        );
        // The class `[0300]` is a four-digit number in brackets too, and an
        // NVIDIA line carries no AMD tag at all.
        assert_eq!(
            extract_lspci_device_id(
                "0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation \
                 GA102 [GeForce RTX 3090] [10de:2204] (rev a1)"
            ),
            None
        );
        // A tag that is not exactly four hex digits is not a device id.
        assert_eq!(extract_lspci_device_id("Device [1002:11141]"), None);
        assert_eq!(extract_lspci_device_id("Device [1002:111]"), None);
        assert_eq!(
            extract_lspci_device_id("no bracketed vendor tag here"),
            None
        );
        // Four hex digits followed by anything but the closing bracket is not
        // a tag either.
        assert_eq!(extract_lspci_device_id("Device [1002:1114g]"), None);
        // The tag is the last `[1002:` on the line: a name that carries an
        // earlier one does not decide the device.
        assert_eq!(
            extract_lspci_device_id(
                "0000:c5:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
                 [AMD/ATI] Device [1002:73a0] rebadged [1002:1114] (rev c1)"
            )
            .as_deref(),
            Some("1114")
        );
    }

    /// `examine` names a GPU from the crate's one name table, codenames
    /// included, so `lspci`'s codename for an iGPU yields its target rather
    /// than an "APU, target unknown" guess.
    ///
    /// Krackan Point is the case that used to be wrong rather than missing:
    /// `examine`'s own table called it gfx1150, while the device-id table and
    /// the 860M/840M it sells as say gfx1152, a different TheRock family.
    #[test]
    fn examine_names_a_gpu_by_its_codename_from_the_shared_table() {
        for (name, target) in [
            (
                "Advanced Micro Devices, Inc. [AMD/ATI] Krackan Point",
                "gfx1152",
            ),
            ("Advanced Micro Devices, Inc. [AMD/ATI] Renoir", "gfx90c"),
            (
                "Advanced Micro Devices, Inc. [AMD/ATI] Cezanne [Radeon Vega Series]",
                "gfx90c",
            ),
            ("Advanced Micro Devices, Inc. [AMD/ATI] Raphael", "gfx1036"),
            (
                "Advanced Micro Devices, Inc. [AMD/ATI] Mendocino",
                "gfx1037",
            ),
            ("Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1", "gfx1103"),
            (
                "Advanced Micro Devices, Inc. [AMD/ATI] VanGogh [AMD Custom GPU 0405]",
                "gfx1033",
            ),
            ("AMD Radeon(TM) 8040S Graphics", "gfx1151"),
        ] {
            assert_eq!(
                classify_amd_marketing_name(name),
                (target.to_owned(), true),
                "{name}"
            );
        }
        // Discrete names come from the same table, so they now carry a target.
        assert_eq!(
            classify_amd_marketing_name("AMD Radeon RX 7900 XTX"),
            ("gfx1100".to_owned(), false)
        );
    }

    /// Every codename and product-line row this change added to the shared
    /// name table, each through a name that carries that one pattern and no
    /// other, so deleting or retargeting any single row fails here.
    ///
    /// The `lspci` corpora cannot do this: their rows carry device ids the
    /// id table answers first, so a codename row behind them is never asked.
    /// The Phoenix spellings are `pci.ids`' names for device ids 15bf, 15c8,
    /// 1900 and 1901.
    #[test]
    fn each_codename_row_names_its_own_target() {
        for (name, target) in [
            ("[AMD/ATI] Strix Halo", "gfx1151"),
            ("AMD Ryzen AI Max", "gfx1151"),
            ("[AMD/ATI] Strix Point", "gfx1150"),
            ("[AMD/ATI] Krackan", "gfx1152"),
            ("[AMD/ATI] Phoenix", "gfx1103"),
            ("[AMD/ATI] Phoenix1", "gfx1103"),
            ("[AMD/ATI] Phoenix2", "gfx1103"),
            ("[AMD/ATI] Phoenix3", "gfx1103"),
            ("[AMD/ATI] Phoenix4", "gfx1103"),
            ("[AMD/ATI] Hawk Point", "gfx1103"),
            ("[AMD/ATI] Rembrandt", "gfx1035"),
            ("[AMD/ATI] Mendocino", "gfx1037"),
            ("[AMD/ATI] Raphael", "gfx1036"),
            ("[AMD/ATI] VanGogh", "gfx1033"),
            ("AMD Custom GPU 0405", "gfx1033"),
            ("[AMD/ATI] Renoir", "gfx90c"),
            ("[AMD/ATI] Cezanne", "gfx90c"),
            ("[AMD/ATI] Lucienne", "gfx90c"),
            ("[AMD/ATI] Barcelo", "gfx90c"),
        ] {
            assert_eq!(
                classify_amd_marketing_name(name),
                (target.to_owned(), true),
                "{name}"
            );
        }
    }

    /// The Linux PCI scan prefers the device id to the name, the way the
    /// Windows probe prefers the PNP id.
    ///
    /// The two inputs disagree on purpose: device id `73a0` is Navi 21
    /// (gfx1030, discrete) while the name is a Strix Halo SKU. No such part
    /// ships, which is why it is the input -- only a pair that cannot both be
    /// right pins precedence rather than agreement.
    #[test]
    fn lspci_scan_prefers_the_device_id_to_the_name() {
        let mut e = Examination::default();
        apply_lspci_gpus(
            &mut e,
            "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
             [AMD/ATI] Radeon 8060S [1002:73a0] (rev c1)",
        );
        assert_eq!(e.gpus.len(), 1, "{:?}", e.gpus);
        assert_eq!(e.gpus[0].gfx_target, "gfx1030", "{:?}", e.gpus);
        assert_eq!(e.gpus[0].is_apu, Some(false), "{:?}", e.gpus);
    }

    /// A Raven Ridge laptop with a discrete card, as a real host reads it.
    ///
    /// No lookup table names Raven: its PCI row resolves no target and so
    /// carries the scan's "not an APU" default. The gfx902 target arrives only
    /// from KFD's topology, so the kernel-membership pass is where the verdict
    /// must follow it -- `rocminfo`, the other route, drops every agent on a
    /// real host today (#393). Without that, the host reports no APU and the
    /// `fix-9-igpu-dgpu` gate stays shut on the shape it exists for.
    #[test]
    fn a_kernel_topology_target_decides_the_verdict_of_a_row_the_pci_scan_could_not_name() {
        let mut e = Examination::default();
        apply_lspci_gpus(
            &mut e,
            "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
             [AMD/ATI] Raven Ridge [Radeon Vega Series / Radeon Vega Mobile Series] [1002:15dd] (rev c4)\n\
             0000:01:00.0 Display controller [0380]: Advanced Micro Devices, Inc. \
             [AMD/ATI] Device [1002:7480] (rev c7)",
        );
        assert_eq!(e.gpus.len(), 2, "{:?}", e.gpus);
        assert_eq!(
            (e.gpus[0].gfx_target.as_str(), e.gpus[0].is_apu),
            ("", Some(false)),
            "the premise: nothing names Raven from its PCI row: {:?}",
            e.gpus[0]
        );

        apply_kernel_gpu_membership(
            &mut e,
            &[
                crate::KfdGpuNode {
                    pci_id: "0000:01:00.0".to_owned(),
                    gfx_target: "gfx1102".to_owned(),
                },
                crate::KfdGpuNode {
                    pci_id: "0000:03:00.0".to_owned(),
                    gfx_target: "gfx902".to_owned(),
                },
            ],
        );
        let raven = e.gpus.iter().find(|g| g.pci_id == "0000:03:00.0").unwrap();
        assert_eq!(raven.gfx_target, "gfx902", "{raven:?}");
        assert_eq!(raven.is_apu, Some(true), "{raven:?}");
        let dgpu = e.gpus.iter().find(|g| g.pci_id == "0000:01:00.0").unwrap();
        assert_eq!(dgpu.is_apu, Some(false), "{dgpu:?}");

        summarise_gpu_categories(&mut e);
        assert!(
            e.has_apu && e.has_discrete_amd,
            "both gate fix-9-igpu-dgpu: {:?}",
            e.gpus
        );
    }

    /// The kernel-membership pass revises only a verdict that came with no
    /// target. Where the PCI scan resolved a target, that target and its
    /// verdict stand; and a KFD target nothing classifies leaves the scan's
    /// verdict alone.
    #[test]
    fn a_kernel_topology_target_never_revises_a_row_that_already_had_one() {
        let row = |gfx_target: &str, is_apu| Gpu {
            name: "Advanced Micro Devices, Inc. [AMD/ATI] Device".to_owned(),
            gfx_target: gfx_target.to_owned(),
            pci_id: "0000:03:00.0".to_owned(),
            is_amd: true,
            is_apu,
        };
        for (before, node_target, expected) in [
            // Resolved by the scan: the node's target does not replace it.
            (
                row("gfx1035", Some(true)),
                "gfx1100",
                ("gfx1035", Some(true)),
            ),
            // Unresolved, and the node's target is one nothing classifies.
            (row("", Some(false)), "gfx1170", ("gfx1170", Some(false))),
        ] {
            let mut e = Examination {
                gpus: vec![before],
                ..Examination::default()
            };
            apply_kernel_gpu_membership(
                &mut e,
                &[crate::KfdGpuNode {
                    pci_id: "0000:03:00.0".to_owned(),
                    gfx_target: node_target.to_owned(),
                }],
            );
            assert_eq!(
                (e.gpus[0].gfx_target.as_str(), e.gpus[0].is_apu),
                expected,
                "node target {node_target}: {:?}",
                e.gpus[0]
            );
        }
    }

    /// A GPU known only from the kernel topology -- no `lspci`, as in an
    /// ordinary ROCm container -- is classified from its target where the
    /// target's packaging is known. Where it is not, the `rocminfo` agent's
    /// marketing name decides only when the agent cannot belong to another
    /// card; on a multi-GPU host the agents pair by position, which is no
    /// evidence, so the verdict stays unknown.
    #[test]
    fn a_kernel_topology_gpu_is_classified_from_the_evidence_there_is() {
        let node = |pci_id: &str, gfx_target: &str| crate::KfdGpuNode {
            pci_id: pci_id.to_owned(),
            gfx_target: gfx_target.to_owned(),
        };
        let agent = |gfx: &str, marketing: &str| {
            format!(
                "*******\nAgent 1\n*******\n  Name:                    {gfx}\n  \
                 Marketing Name:          {marketing}\n  \
                 Device Type:             GPU\n"
            )
        };
        let mut e = Examination::default();
        apply_kernel_gpu_membership(
            &mut e,
            &[
                node("0000:03:00.0", "gfx1151"),
                node("0000:04:00.0", "gfx942"),
                node("0000:05:00.0", "gfx1170"),
                node("0000:06:00.0", "gfx1170"),
            ],
        );
        let verdicts: Vec<_> = e.gpus.iter().map(|g| g.is_apu).collect();
        assert_eq!(
            verdicts,
            [Some(true), Some(false), None, None],
            "{:?}",
            e.gpus
        );

        // `rocminfo` then pairs one agent per row, by position. The third
        // agent's marketing name resolves an APU, but nothing ties that agent
        // to the third card rather than the fourth, so neither unknown target
        // takes a verdict from a name.
        let out = [
            agent("gfx1151", "AMD Radeon(TM) 8060S Graphics"),
            agent("gfx942", "AMD Instinct MI300X"),
            agent("gfx1170", "AMD Radeon(TM) 8060S Graphics"),
            agent("gfx1170", "AMD Radeon Graphics"),
        ]
        .concat();
        apply_rocminfo_gpu_agents(&mut e, &out);
        let verdicts: Vec<_> = e.gpus.iter().map(|g| g.is_apu).collect();
        assert_eq!(
            verdicts,
            [Some(true), Some(false), None, None],
            "{:?}",
            e.gpus
        );

        // One card and one agent: the pairing cannot be wrong, so the name
        // decides, and a name that resolves nothing still leaves it unset.
        for (marketing, expected) in [
            ("AMD Radeon(TM) 8060S Graphics", Some(true)),
            ("AMD Radeon RX 7900 XTX", Some(false)),
            ("AMD Radeon Graphics", None),
        ] {
            let mut e = Examination::default();
            apply_kernel_gpu_membership(&mut e, &[node("0000:05:00.0", "gfx1170")]);
            apply_rocminfo_gpu_agents(&mut e, &agent("gfx1170", marketing));
            assert_eq!(e.gpus.len(), 1, "{:?}", e.gpus);
            assert_eq!(e.gpus[0].is_apu, expected, "{marketing}: {:?}", e.gpus[0]);
        }
    }

    #[test]
    fn a_sysfs_fallback_gpu_carries_its_targets_verdict_where_one_is_known() {
        for (target, expected) in [
            ("gfx1151", Some(true)),
            ("gfx942", Some(false)),
            ("gfx1170", None),
        ] {
            let mut e = Examination::default();
            let probe: fn() -> Option<String> = match target {
                "gfx1151" => || Some("gfx1151".to_owned()),
                "gfx942" => || Some("gfx942".to_owned()),
                _ => || Some("gfx1170".to_owned()),
            };
            probe_gpus_sysfs_fallback(&mut e, probe);
            assert_eq!(e.gpus.len(), 1, "{:?}", e.gpus);
            assert_eq!(e.gpus[0].is_apu, expected, "{target}: {:?}", e.gpus[0]);
        }
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
}
