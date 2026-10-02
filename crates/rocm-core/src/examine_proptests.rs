// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property-based tests for AMD GPU detection and classification.
//!
//! The generators here are built from *real* fixture shapes — actual `lspci
//! -nn -D` lines, `rocminfo` agent blocks, `Win32_VideoController` rows and AMD
//! marketing names — and then mutated (case, whitespace, decorations, dropped
//! fields, truncation, duplication). A uniform-random string generator never
//! reaches the deep parser branches, so every strategy below starts from a
//! corpus and perturbs it.
//!
//! Each property states an invariant the *report* must hold, not an
//! implementation detail, so a failure names a user-visible defect.

use super::{
    Examination, GFX_TARGET_PACKAGING, Gpu, apply_hipinfo_gcn_arch_names, apply_lspci_gpus,
    apply_rocminfo_gpu_agents, apply_windows_display_rows, classify_amd_marketing_name,
    extract_lspci_name, gfx_is_apu_family, is_lspci_gpu_line, summarise_gpu_categories,
};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, TestRunner};

// ---------------------------------------------------------------------------
// Corpora: real strings, taken from pci.ids, from this repo's own fixtures and
// from the hardware named in ROCm/rocm-cli#448 / #449.
// ---------------------------------------------------------------------------

/// `(device_name_as_lspci_prints_it, pci_device_id, true_gfx_target,
/// is_really_an_apu)`.
///
/// The names are the `pci.ids` device strings an `lspci -nn` line carries after
/// the `[AMD/ATI]` vendor tag. The last two fields are ground truth about the
/// silicon, not about what the code says, so a pairing drawn from one row is
/// always a *coherent* host: the gfx target really is the one that device
/// reports.
const AMD_LSPCI_DEVICES: &[(&str, &str, &str, bool)] = &[
    // Discrete RDNA2 / RDNA3 / RDNA4.
    (
        "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]",
        "744c",
        "gfx1100",
        false,
    ),
    (
        "Navi 32 [Radeon RX 7700 XT / 7800 XT]",
        "747e",
        "gfx1101",
        false,
    ),
    (
        "Navi 33 [Radeon RX 7600/7600 XT/7600M XT/7600S/7700S / PRO W7600]",
        "7480",
        "gfx1102",
        false,
    ),
    (
        "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]",
        "73bf",
        "gfx1030",
        false,
    ),
    (
        "Navi 22 [Radeon RX 6700 XT / 6800M / 6950 XT]",
        "73df",
        "gfx1031",
        false,
    ),
    (
        "Navi 23 [Radeon RX 6600/6600 XT/6600M]",
        "73ff",
        "gfx1032",
        false,
    ),
    (
        "Navi 24 [Radeon RX 6400/6500 XT/6500M]",
        "743f",
        "gfx1034",
        false,
    ),
    ("Navi 48 [Radeon RX 9070/9070 XT]", "7550", "gfx1201", false),
    // Datacenter.
    ("Aqua Vanjaram [Instinct MI300X]", "74a1", "gfx942", false),
    // APUs, oldest first. Every one of these is an integrated GPU sharing
    // system memory; `has_apu` must be true on a host that has one.
    ("Renoir", "1636", "gfx90c", true),
    ("Cezanne", "1638", "gfx90c", true),
    ("Lucienne", "164c", "gfx90c", true),
    ("Barcelo", "15e7", "gfx90c", true),
    ("VanGogh [AMD Custom GPU 0405]", "163f", "gfx1033", true),
    ("Rembrandt [Radeon 680M]", "1681", "gfx1035", true),
    ("Raphael", "164e", "gfx1036", true),
    ("Phoenix1", "15bf", "gfx1103", true),
    ("Phoenix3", "1900", "gfx1103", true),
    ("Strix [Radeon 880M / 890M]", "150e", "gfx1150", true),
    ("Krackan Point [Radeon 860M]", "1114", "gfx1152", true),
    // Strix Halo, as reported in ROCm/rocm-cli#449: pci.ids has no entry, so
    // `lspci` prints the bare word "Device".
    ("Device", "1586", "gfx1151", true),
];

/// PCI classes an `lspci -nn` line can carry, with the ones this probe must
/// enumerate first and the bridge (which it must not) last.
const LSPCI_CLASSES: &[(&str, bool)] = &[
    ("VGA compatible controller [0300]", true),
    ("Display controller [0380]", true),
    ("3D controller [0302]", true),
    ("Processing accelerators [1200]", true),
    ("PCI bridge [0604]", false),
];

/// AMD marketing names as `rocminfo`'s `Marketing Name:` and Windows'
/// `Win32_VideoController.Name` report them, with ground truth about the part.
///
/// `(name, true_gfx_target, is_really_an_apu)`.
const AMD_MARKETING_NAMES: &[(&str, &str, bool)] = &[
    ("AMD Radeon RX 7900 XTX", "gfx1100", false),
    ("AMD Radeon RX 7800 XT", "gfx1101", false),
    ("AMD Radeon RX 7600", "gfx1102", false),
    ("AMD Radeon RX 9070 XT", "gfx1201", false),
    ("AMD Radeon PRO W7900", "gfx1100", false),
    ("AMD Instinct MI300X", "gfx942", false),
    ("AMD Radeon RX 6900 XT", "gfx1030", false),
    // APUs.
    ("AMD Radeon(TM) 610M Graphics", "gfx1036", true),
    ("AMD Radeon(TM) 660M Graphics", "gfx1035", true),
    ("AMD Radeon(TM) 680M Graphics", "gfx1035", true),
    ("AMD Radeon(TM) 740M Graphics", "gfx1103", true),
    ("AMD Radeon(TM) 760M Graphics", "gfx1103", true),
    ("AMD Radeon(TM) 780M Graphics", "gfx1103", true),
    ("AMD Radeon(TM) 820M Graphics", "gfx1153", true),
    ("AMD Radeon(TM) 840M Graphics", "gfx1152", true),
    ("AMD Radeon(TM) 860M Graphics", "gfx1152", true),
    ("AMD Radeon(TM) 880M Graphics", "gfx1150", true),
    ("AMD Radeon(TM) 890M Graphics", "gfx1150", true),
    ("AMD Radeon 8040S Graphics", "gfx1151", true),
    ("AMD Radeon 8050S Graphics", "gfx1151", true),
    ("AMD Radeon 8060S Graphics", "gfx1151", true),
    ("AMD Custom GPU 0405", "gfx1033", true),
];

/// gfx targets a real host reports, with ground truth about packaging.
///
/// Not every one of these is produced by a lookup table — gfx902, gfx909 and
/// gfx1037 reach the CLI only through
/// [`crate::gfx_target_from_gc_version`], which synthesises a target from a GC
/// version (KFD's packed `gfx_target_version`, or DRM ip-discovery). That is
/// exactly why they are here: the cross-table drift guard cannot see them, so
/// ground truth has to.
const GFX_TARGETS: &[(&str, bool)] = &[
    ("gfx900", false),
    ("gfx906", false),
    ("gfx908", false),
    ("gfx90a", false),
    ("gfx942", false),
    ("gfx950", false),
    ("gfx1010", false),
    ("gfx1030", false),
    ("gfx1031", false),
    ("gfx1032", false),
    ("gfx1034", false),
    ("gfx1100", false),
    ("gfx1101", false),
    ("gfx1102", false),
    ("gfx1200", false),
    ("gfx1201", false),
    ("gfx1250", false),
    // APU targets.
    ("gfx902", true),
    ("gfx909", true),
    ("gfx90c", true),
    ("gfx1033", true),
    ("gfx1035", true),
    ("gfx1036", true),
    ("gfx1037", true),
    ("gfx1103", true),
    ("gfx1150", true),
    ("gfx1151", true),
    ("gfx1152", true),
    ("gfx1153", true),
];

// ---------------------------------------------------------------------------
// Mutators: the perturbations real-world messiness applies to these strings.
// ---------------------------------------------------------------------------

/// Textual perturbations applied to a fixture-derived string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mutation {
    None,
    Upper,
    Lower,
    PadInnerWhitespace,
    PadOuterWhitespace,
    TrademarkDecoration,
    RegisteredDecoration,
    TruncateTail,
    TruncateHead,
    DropBrackets,
    TabsForSpaces,
}

fn mutation() -> impl Strategy<Value = Mutation> {
    prop_oneof![
        6 => Just(Mutation::None),
        2 => Just(Mutation::Upper),
        2 => Just(Mutation::Lower),
        2 => Just(Mutation::PadInnerWhitespace),
        2 => Just(Mutation::PadOuterWhitespace),
        2 => Just(Mutation::TrademarkDecoration),
        1 => Just(Mutation::RegisteredDecoration),
        1 => Just(Mutation::TruncateTail),
        1 => Just(Mutation::TruncateHead),
        1 => Just(Mutation::DropBrackets),
        1 => Just(Mutation::TabsForSpaces),
    ]
}

/// The mutations that preserve the *identity* of the device being named.
///
/// Case, whitespace and vendor decorations do; truncation and bracket removal
/// can destroy the token a lookup keys on, so properties that assert a
/// device-specific verdict draw from this narrower set rather than filtering
/// the wider one — a filter spends the generator's reject budget on draws it
/// was always going to discard, and proptest aborts the test when that budget
/// runs out.
fn identity_preserving_mutation() -> impl Strategy<Value = Mutation> {
    prop_oneof![
        6 => Just(Mutation::None),
        2 => Just(Mutation::Upper),
        2 => Just(Mutation::Lower),
        2 => Just(Mutation::PadInnerWhitespace),
        2 => Just(Mutation::PadOuterWhitespace),
        2 => Just(Mutation::TrademarkDecoration),
        1 => Just(Mutation::RegisteredDecoration),
        1 => Just(Mutation::TabsForSpaces),
    ]
}

fn apply_mutation(text: &str, m: Mutation) -> String {
    match m {
        Mutation::None => text.to_owned(),
        Mutation::Upper => text.to_uppercase(),
        Mutation::Lower => text.to_lowercase(),
        Mutation::PadInnerWhitespace => text.replace(' ', "   "),
        Mutation::PadOuterWhitespace => format!("  {text}\t "),
        Mutation::TrademarkDecoration => text.replacen("Radeon", "Radeon(TM)", 1),
        Mutation::RegisteredDecoration => text.replacen("AMD", "AMD(R)", 1),
        Mutation::TruncateTail => {
            let keep = text.len().saturating_mul(2) / 3;
            text.chars().take(keep).collect()
        }
        Mutation::TruncateHead => {
            let drop = text.len() / 4;
            text.chars().skip(drop).collect()
        }
        Mutation::DropBrackets => text.replace(['[', ']'], ""),
        Mutation::TabsForSpaces => text.replace(' ', "\t"),
    }
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// A plausible PCI address, including the host-bridge-adjacent high bus numbers
/// a Strix Halo iGPU enumerates on (`0000:66:00.0` in #449).
fn pci_address() -> impl Strategy<Value = String> {
    (0u32..2, 0u32..0x100, 0u32..0x20, 0u32..8)
        .prop_map(|(dom, bus, dev, func)| format!("{dom:04x}:{bus:02x}:{dev:02x}.{func}"))
}

/// A full `lspci -nn -D` line drawn from the real corpus, with the real vendor
/// tag, an optional revision suffix, and a mutation applied.
fn lspci_line() -> impl Strategy<Value = (String, (&'static str, &'static str, &'static str, bool))>
{
    (
        pci_address(),
        proptest::sample::select(LSPCI_CLASSES),
        proptest::sample::select(AMD_LSPCI_DEVICES),
        proptest::option::of(0u32..0x100),
        mutation(),
    )
        .prop_map(|(addr, class, device, rev, m)| {
            let rev = rev.map_or_else(String::new, |r| format!(" (rev {r:02x})"));
            let line = format!(
                "{addr} {}: Advanced Micro Devices, Inc. [AMD/ATI] {} [1002:{}]{rev}",
                class.0, device.0, device.1
            );
            (apply_mutation(&line, m), device)
        })
}

/// A marketing name drawn from the real corpus with a mutation applied, paired
/// with the ground truth for the part it names.
fn marketing_name() -> impl Strategy<Value = (String, (&'static str, &'static str, bool), Mutation)>
{
    (proptest::sample::select(AMD_MARKETING_NAMES), mutation())
        .prop_map(|(entry, m)| (apply_mutation(entry.0, m), entry, m))
}

/// The same, restricted to spellings that still name the same device.
fn marketing_name_spelled_differently()
-> impl Strategy<Value = (String, (&'static str, &'static str, bool))> {
    (
        proptest::sample::select(AMD_MARKETING_NAMES),
        identity_preserving_mutation(),
    )
        .prop_map(|(entry, m)| (apply_mutation(entry.0, m), entry))
}

/// Marketing names of parts that really are APUs, differently spelled.
fn apu_marketing_name() -> impl Strategy<Value = (String, (&'static str, &'static str, bool))> {
    let apus: Vec<_> = AMD_MARKETING_NAMES
        .iter()
        .copied()
        .filter(|entry| entry.2)
        .collect();
    assert!(!apus.is_empty(), "the marketing corpus has no APU");
    (
        proptest::sample::select(apus),
        identity_preserving_mutation(),
    )
        .prop_map(|(entry, m)| (apply_mutation(entry.0, m), entry))
}

/// `lspci` entries for parts that really are APUs *and* carry a model name.
///
/// See [`LSPCI_DEVICES_WITHOUT_A_PCI_IDS_NAME`] for why the nameless row is not
/// a candidate for any name-keyed property.
fn apu_lspci_devices() -> Vec<(&'static str, &'static str, &'static str, bool)> {
    let apus: Vec<_> = AMD_LSPCI_DEVICES
        .iter()
        .copied()
        .filter(|device| device.3 && !LSPCI_DEVICES_WITHOUT_A_PCI_IDS_NAME.contains(&device.0))
        .collect();
    assert!(!apus.is_empty(), "the lspci corpus has no named APU");
    apus
}

/// What the real PCI scan makes of one corpus device.
///
/// Every test that needs a PCI-sourced GPU goes through this rather than
/// assembling one from the classifiers, because a test that re-implements the
/// probe's logic pins the re-implementation: when the probe stops reading the
/// device id, a hand-built row keeps passing.
fn pci_scanned(device: &str, device_id: &str, addr: &str) -> Gpu {
    let line = format!(
        "{addr} VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
         [AMD/ATI] {device} [1002:{device_id}] (rev c1)"
    );
    let mut e = Examination::default();
    apply_lspci_gpus(&mut e, &line);
    assert_eq!(e.gpus.len(), 1, "{line:?} must enumerate as one GPU");
    e.gpus.remove(0)
}

/// A `Win32_VideoController` row as `WIN_GPU_SCRIPT` prints it: name, driver
/// version, PNP device id.
fn windows_row(name: &str, device_id: &str) -> String {
    format!("{name}\t32.0.1\tPCI\\VEN_1002&DEV_{device_id}&SUBSYS_00001002&REV_C1")
}

/// What the real Windows display probe makes of one row.
fn windows_scanned(name: &str, device_id: &str) -> Gpu {
    let row = windows_row(name, device_id);
    let mut e = Examination::default();
    apply_windows_display_rows(&mut e, &row);
    assert_eq!(e.gpus.len(), 1, "{row:?} must enumerate as one GPU");
    e.gpus.remove(0)
}

/// Every PCI device id the crate's device-id table maps, with its target.
///
/// The table is a `match`, not an iterable, so its whole input domain is swept.
fn every_mapped_pci_device_id() -> Vec<(String, &'static str)> {
    let ids: Vec<_> = (0..=0xffff_u32)
        .map(|id| format!("{id:04x}"))
        .filter_map(|id| crate::gfx_target_from_amd_pci_device_id(&id).map(|t| (id, t)))
        .collect();
    assert!(!ids.is_empty(), "the device-id table maps no id at all");
    ids
}

/// A `rocminfo` agent listing for the given targets, in the agent block shape
/// the current parser can digest (no ISA sub-entries).
fn rocminfo_output(agents: &[(&str, &str)]) -> String {
    rocminfo_output_shaped(agents, false)
}

/// `with_isa_entries` reproduces what every `rocminfo` since ROCm 5 actually
/// prints: each GPU agent is followed by indented ISA sub-entries that *also*
/// begin with `Name:`. The parser keeps a running `cur_name` and lets those
/// lines overwrite the agent's gfx name, so the agent is dropped and the whole
/// fold becomes a no-op — ROCm/rocm-cli#393.
///
/// Both shapes are generated on purpose. The clean one is the only shape that
/// currently reaches the `is_apu` overwrite, so it is where the APU downgrade
/// shows; the real one shows that the overwrite is unreachable on a real host
/// today, and stops being unreachable the moment #393 is fixed.
fn rocminfo_output_shaped(agents: &[(&str, &str)], with_isa_entries: bool) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "=====================\nHSA System Attributes\n=====================\n\
         Runtime Version:         1.1\n\n",
    );
    out.push_str(
        "==========\nHSA Agents\n==========\n*******\nAgent 1\n*******\n  \
         Name:                    AMD Ryzen 9 7950X\n  \
         Marketing Name:          AMD Ryzen 9 7950X\n  \
         Device Type:             CPU\n",
    );
    for (index, (gfx, marketing)) in agents.iter().enumerate() {
        let _ = write!(
            out,
            "*******\nAgent {}\n*******\n  Name:                    {gfx}\n  \
             Marketing Name:          {marketing}\n  Device Type:             GPU\n",
            index + 2
        );
        if with_isa_entries {
            let family = gfx.get(..5).unwrap_or(gfx);
            let _ = write!(
                out,
                "  ISA Info:\n    ISA 1\n      Name:                    \
                 amdgcn-amd-amdhsa--{gfx}\n    ISA 2\n      Name:                    \
                 amdgcn-amd-amdhsa--{family}-generic\n"
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Totality and determinism: no fixture-derived or mutated line may panic
    /// any of the pure classifiers, and repeated calls must agree.
    #[test]
    fn classification_is_total_and_deterministic((line, _) in lspci_line()) {
        prop_assert_eq!(is_lspci_gpu_line(&line), is_lspci_gpu_line(&line));
        let name = extract_lspci_name(&line);
        prop_assert_eq!(&name, &extract_lspci_name(&line));
        let verdict = classify_amd_marketing_name(&name);
        prop_assert_eq!(&verdict, &classify_amd_marketing_name(&name));
        prop_assert_eq!(gfx_is_apu_family(&verdict.0), gfx_is_apu_family(&verdict.0));
    }

    /// `extract_lspci_name` must recover exactly the vendor-plus-device text of
    /// an unmutated `lspci -nn` line: everything between the class `]:` and the
    /// trailing `[vendor:device]`.
    #[test]
    fn lspci_name_extraction_round_trips(
        addr in pci_address(),
        class in proptest::sample::select(LSPCI_CLASSES),
        device in proptest::sample::select(AMD_LSPCI_DEVICES),
        rev in proptest::option::of(0u32..0x100),
    ) {
        let rev = rev.map_or_else(String::new, |r| format!(" (rev {r:02x})"));
        let line = format!(
            "{addr} {}: Advanced Micro Devices, Inc. [AMD/ATI] {} [1002:{}]{rev}",
            class.0, device.0, device.1
        );
        let expected = format!("Advanced Micro Devices, Inc. [AMD/ATI] {}", device.0);
        prop_assert_eq!(extract_lspci_name(&line), expected);
    }

    /// A name the install-family detector resolves to an APU target must not be
    /// reported by `examine` as "not an APU". A lookup miss is not evidence of
    /// a discrete GPU.
    #[test]
    fn examine_never_calls_a_known_apu_discrete((name, truth) in apu_marketing_name()) {
        let (_target, is_apu) = classify_amd_marketing_name(&name);
        prop_assert!(
            is_apu,
            "{name} is an APU ({}) but classify_amd_marketing_name says is_apu=false",
            truth.1,
        );
    }

    /// Folding a `rocminfo` reading into the report must never *downgrade* an
    /// APU verdict the PCI scan already reached. More evidence must not produce
    /// a worse answer.
    #[test]
    fn rocminfo_never_downgrades_an_apu_verdict(
        device in proptest::sample::select(apu_lspci_devices()),
        addr in pci_address(),
        marketing in proptest::sample::select(&["", "AMD Radeon Graphics"][..]),
    ) {
        // A coherent host: the gfx target is the one this very device reports.
        let gpu = pci_scanned(device.0, device.1, &addr);
        prop_assert!(
            gpu.is_apu == Some(true),
            "the PCI scan must already know {} is an APU before this property \
             can say anything about preserving that verdict: {:?}",
            device.0,
            gpu,
        );

        let mut e = Examination {
            gpus: vec![gpu],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(&mut e, &rocminfo_output(&[(device.2, marketing)]));
        summarise_gpu_categories(&mut e);
        prop_assert!(
            e.has_apu,
            "lspci classified {} as an APU, then rocminfo reporting {} flipped has_apu to false",
            device.0,
            device.2,
        );
        prop_assert!(
            !e.has_discrete_amd,
            "an APU-only host must not report has_discrete_amd (gfx {})",
            device.2,
        );
    }

    /// Classification must not depend on spelling: case, inner/outer
    /// whitespace and vendor decorations name the same device.
    #[test]
    fn classification_is_spelling_invariant((name, truth) in marketing_name_spelled_differently()) {
        let canonical = classify_amd_marketing_name(truth.0);
        let mutated = classify_amd_marketing_name(&name);
        prop_assert_eq!(
            &canonical,
            &mutated,
            "{:?} and {:?} are the same device but classify differently",
            truth.0,
            name,
        );
    }

    /// `gfx_is_apu_family` must agree with ground truth for every target a real
    /// host reports, and must be suffix-insensitive.
    #[test]
    fn gfx_apu_family_matches_ground_truth(
        gfx in proptest::sample::select(GFX_TARGETS),
        suffix in proptest::sample::select(&["", ":xnack-", ":sramecc+:xnack-", ":sramecc-"][..]),
        upper in any::<bool>(),
    ) {
        let spelled = if upper {
            format!("{}{suffix}", gfx.0.to_uppercase())
        } else {
            format!("{}{suffix}", gfx.0)
        };
        prop_assert_eq!(
            gfx_is_apu_family(&spelled),
            gfx.1,
            "gfx_is_apu_family({}) disagrees with ground truth",
            spelled,
        );
    }

    /// `examine` and the install-side display probe must read the same
    /// adapter the same way.
    ///
    /// They share a decoder, so this is not two tables cross-checking each
    /// other: it guards the *wiring*. The install side hands the decoder
    /// `name<TAB>pnp`, `examine` hands it `name<TAB>driver<TAB>pnp`, and a
    /// decoder that read the PNP id by column position rather than by content
    /// would quietly resolve one and not the other.
    #[test]
    fn examine_and_the_install_probe_read_a_windows_row_alike(
        entry in proptest::sample::select(AMD_MARKETING_NAMES),
    ) {
        let Some(device_id) = entry_device_id(entry.0) else {
            return Ok(());
        };
        let install_text = format!(
            "{}\tPCI\\VEN_1002&DEV_{device_id}&SUBSYS_00001002&REV_C1",
            entry.0
        );
        let install_target = crate::parse_windows_display_gfx_target(&install_text);
        let examine_target = windows_scanned(entry.0, device_id).gfx_target;
        prop_assert_eq!(
            install_target.unwrap_or_default(),
            examine_target,
            "the install probe and examine disagree about {}",
            entry.0,
        );
    }
}

/// Corpus entries whose `lspci` text names no model at all.
///
/// `lspci` prints the `pci.ids` device string, and when that database has no
/// entry for an id it prints the bare word `Device`. The PCI scan then has only
/// the `[1002:xxxx]` id to go on, and these are ids the crate's device-id table
/// does not map yet (Strix Halo's, here), so nothing can classify them. They
/// are excluded from the APU sweeps below rather than dropped from the corpus,
/// because the parsing and totality properties still have to survive them.
const LSPCI_DEVICES_WITHOUT_A_PCI_IDS_NAME: &[&str] = &["Device"];

/// The PCI device id for a marketing name, when the corpus pins one.
fn entry_device_id(name: &str) -> Option<&'static str> {
    match name {
        "AMD Radeon(TM) 610M Graphics" => Some("164e"),
        "AMD Radeon(TM) 660M Graphics" | "AMD Radeon(TM) 680M Graphics" => Some("1681"),
        "AMD Radeon(TM) 740M Graphics"
        | "AMD Radeon(TM) 760M Graphics"
        | "AMD Radeon(TM) 780M Graphics" => Some("15bf"),
        "AMD Radeon(TM) 860M Graphics" | "AMD Radeon(TM) 840M Graphics" => Some("1114"),
        "AMD Custom GPU 0405" => Some("163f"),
        "AMD Radeon RX 9070 XT" => Some("7550"),
        _ => None,
    }
}

/// Enumerate every corpus entry the shrinker would otherwise collapse to one
/// minimal case, so the full extent of a divergence is visible in the failure
/// rather than just its first example.
#[test]
fn every_known_apu_is_classified_as_an_apu() {
    let mut misses: Vec<String> = Vec::new();
    for (name, target, is_apu) in AMD_MARKETING_NAMES {
        if !is_apu {
            continue;
        }
        let (examine_target, examine_is_apu) = classify_amd_marketing_name(name);
        if !examine_is_apu {
            misses.push(format!(
                "  marketing name {name:?} ({target}): examine says is_apu=false, \
                 gfx_target={examine_target:?}"
            ));
        }
    }
    for (gfx, is_apu) in GFX_TARGETS {
        if *is_apu && !gfx_is_apu_family(gfx) {
            misses.push(format!("  gfx target {gfx}: gfx_is_apu_family says false"));
        }
    }
    for (device, id, _gfx, is_apu) in AMD_LSPCI_DEVICES {
        if !is_apu || LSPCI_DEVICES_WITHOUT_A_PCI_IDS_NAME.contains(device) {
            continue;
        }
        let gpu = pci_scanned(device, id, "0000:04:00.0");
        if gpu.is_apu != Some(true) {
            misses.push(format!(
                "  lspci device {device:?} [1002:{id}]: the PCI scan says {gpu:?}"
            ));
        }
    }
    assert!(
        misses.is_empty(),
        "parts that are APUs but are not classified as APUs:\n{}",
        misses.join("\n")
    );
}

/// A reported `gfx_target` must never *contradict* the part: an unresolved
/// lookup is recoverable, a wrong answer is not.
#[test]
fn no_part_is_labelled_with_the_wrong_gfx_target() {
    let mut wrong: Vec<String> = Vec::new();
    for (device, id, gfx, _is_apu) in AMD_LSPCI_DEVICES {
        let guess = pci_scanned(device, id, "0000:04:00.0").gfx_target;
        if !guess.is_empty() && guess != *gfx {
            wrong.push(format!(
                "  lspci {device:?} [1002:{id}] is {gfx}, but examine reports {guess}"
            ));
        }
    }
    for (name, gfx, _is_apu) in AMD_MARKETING_NAMES {
        let guess = classify_amd_marketing_name(name).0;
        if !guess.is_empty() && guess != *gfx {
            wrong.push(format!(
                "  marketing name {name:?} is {gfx}, but examine reports {guess}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "parts labelled with a gfx target that is not theirs:\n{}",
        wrong.join("\n")
    );
}

/// The APU verdict must survive a `rocminfo` fold whether or not
/// ROCm/rocm-cli#393 has been fixed.
///
/// #393 is that `rocminfo` prints indented ISA `Name:` sub-entries after each
/// agent, the parser's running `cur_name` ends up holding
/// `amdgcn-amd-amdhsa--gfx11-generic`, the agent is dropped and the whole fold
/// returns early. That made the `is_apu` overwrite unreachable on a real host,
/// which is the only reason this misclassification was latent rather than
/// live — and fixing #393 is exactly what would have made it live.
///
/// Both shapes are driven here so the verdict cannot depend on which side of
/// #393 the parser is on. The fix for #393 belongs to #393; this only has to
/// hold under either parser.
#[test]
fn the_apu_verdict_holds_on_both_sides_of_the_rocminfo_isa_name_defect() {
    let build = |with_isa: bool| {
        let gpu = pci_scanned("Raphael", "164e", "0000:14:00.0");
        assert_eq!(
            gpu.gfx_target, "gfx1036",
            "the PCI scan alone names the part"
        );
        assert_eq!(
            gpu.is_apu,
            Some(true),
            "the PCI scan alone knows Raphael is an APU"
        );
        let mut e = Examination {
            gpus: vec![gpu],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_output_shaped(&[("gfx1036", "AMD Radeon Graphics")], with_isa),
        );
        summarise_gpu_categories(&mut e);
        e
    };

    for (with_isa, parser) in [
        (true, "with the agent-dropping parser of #393"),
        (false, "with a parser that reads the agent"),
    ] {
        let e = build(with_isa);
        assert_eq!(e.gpus[0].gfx_target, "gfx1036", "{parser}: {:?}", e.gpus);
        assert!(
            e.has_apu,
            "{parser}: the Raphael iGPU must still be an APU: {:?}",
            e.gpus
        );
        assert!(
            !e.has_discrete_amd,
            "{parser}: an iGPU-only host must not report a discrete AMD GPU: {:?}",
            e.gpus
        );
    }
}

/// Reading `rocminfo` may only ever improve the report, never worsen it.
///
/// The two parser shapes of
/// [`the_apu_verdict_holds_on_both_sides_of_the_rocminfo_isa_name_defect`] agree
/// about Raphael because the PCI name already resolves it, so that test alone
/// cannot show the fold doing anything. These two hosts separate the cases:
///
/// - a Strix Halo whose `lspci` entry has no `pci.ids` name, so the PCI scan
///   contributes nothing and only the agent can supply the target: the fold
///   must *add* the APU verdict once the agent is readable;
/// - an iGPU whose agent reports a target this crate has never heard of: the
///   fold knows the target but not its packaging, which is not evidence of a
///   discrete GPU, so the PCI scan's verdict has to survive untouched.
#[test]
fn folding_in_rocminfo_only_ever_adds_knowledge() {
    let host = |lspci_name: &str, device_id: &str, agent_target: &str, with_isa: bool| {
        let mut e = Examination {
            gpus: vec![pci_scanned(lspci_name, device_id, "0000:66:00.0")],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_output_shaped(&[(agent_target, "AMD Radeon Graphics")], with_isa),
        );
        summarise_gpu_categories(&mut e);
        e
    };

    // A nameless Strix Halo: nothing is known before the fold.
    let before = host("Device", "1586", "gfx1151", true);
    assert_eq!(before.gpus[0].gfx_target, "", "{:?}", before.gpus);
    assert!(!before.has_apu, "{:?}", before.gpus);
    // Once the agent is readable the fold supplies both the target and the
    // packaging that follows from it.
    let after = host("Device", "1586", "gfx1151", false);
    assert_eq!(after.gpus[0].gfx_target, "gfx1151", "{:?}", after.gpus);
    assert!(
        after.has_apu,
        "a readable gfx1151 agent must establish the APU verdict the PCI scan \
         could not reach: {:?}",
        after.gpus
    );

    // A Rembrandt laptop whose agent names a target this crate has no packaging
    // entry for. The target is still worth recording; the verdict is not the
    // fold's to revise.
    let unknown = host("Rembrandt [Radeon 680M]", "1681", "gfx1154", false);
    assert_eq!(unknown.gpus[0].gfx_target, "gfx1154", "{:?}", unknown.gpus);
    assert!(
        unknown.has_apu,
        "an unrecognised target is not evidence of a discrete GPU, so the PCI \
         scan's APU verdict must survive: {:?}",
        unknown.gpus
    );
    assert!(
        !unknown.has_discrete_amd,
        "and it must certainly not be promoted to a discrete GPU: {:?}",
        unknown.gpus
    );
}

/// The same rule where there is no PCI row at all: an ordinary ROCm container,
/// which ships `rocminfo` but not `pciutils`.
///
/// Every agent then lands in the branch that *creates* a GPU entry, so there is
/// no earlier verdict to preserve — but there is still evidence, and still a
/// difference between "not an APU" and "cannot say". Defaulting the latter to
/// `Some(false)` reports a discrete AMD GPU on a laptop that has none, which is
/// what gates the iGPU+dGPU diagnosis on, and it throws away the agent's own
/// marketing name while doing it. [`super::probe_gpus_sysfs_fallback`] reaches
/// the same conclusion for the same reason and leaves `is_apu` unset.
#[test]
fn an_agent_with_no_pci_row_is_classified_from_what_evidence_there_is() {
    let container = |agent_target: &str, marketing: &str| {
        let mut e = Examination::default();
        apply_rocminfo_gpu_agents(&mut e, &rocminfo_output(&[(agent_target, marketing)]));
        summarise_gpu_categories(&mut e);
        e
    };

    // The target is enough on its own.
    let known = container("gfx1036", "AMD Radeon Graphics");
    assert_eq!(known.gpus[0].is_apu, Some(true), "{:?}", known.gpus);
    assert!(!known.has_discrete_amd, "{:?}", known.gpus);

    // An unlisted target, but the agent names the part: the name answers.
    let named = container("gfx1154", "AMD Radeon(TM) 780M Graphics");
    assert_eq!(named.gpus[0].is_apu, Some(true), "{:?}", named.gpus);
    assert!(
        !named.has_discrete_amd,
        "an iGPU-only laptop must not report a discrete AMD GPU just because \
         its target is new: {:?}",
        named.gpus
    );

    // Neither answers. Saying "discrete" here would be inventing a fact.
    let silent = container("gfx1154", "");
    assert_eq!(silent.gpus[0].is_apu, None, "{:?}", silent.gpus);
    assert!(
        !silent.has_apu && !silent.has_discrete_amd,
        "{:?}",
        silent.gpus
    );
    assert!(
        silent.has_amd_gpu,
        "the GPU is still present and still AMD: {:?}",
        silent.gpus
    );

    // A discrete card still reports as one.
    let discrete = container("gfx1100", "AMD Radeon RX 7900 XTX");
    assert_eq!(discrete.gpus[0].is_apu, Some(false), "{:?}", discrete.gpus);
    assert!(discrete.has_discrete_amd, "{:?}", discrete.gpus);
}

/// Every gfx target the crate's two lookup tables can hand back must have a
/// packaging entry.
///
/// `GFX_TARGET_PACKAGING` is keyed by target while the marketing-name and PCI
/// device-id tables are keyed by name and id, so nothing in the type system
/// ties them together. An entry added to either of those for a part this one
/// has never heard of would answer `is_apu = false` — "discrete" — for an APU,
/// which is the original defect in a new place.
#[test]
fn every_target_the_lookup_tables_produce_has_a_packaging() {
    let classified = |target: &str| {
        GFX_TARGET_PACKAGING
            .iter()
            .any(|(known, _)| *known == target)
    };
    let mut unclassified: Vec<String> = Vec::new();
    for entry in crate::AMD_MARKETING_GFX_TARGETS {
        if !classified(entry.gfx_target) {
            unclassified.push(format!(
                "  marketing pattern {:?} -> {} has no packaging entry",
                entry.pattern, entry.gfx_target
            ));
        }
    }
    // The device-id lookup is a `match`, not an iterable table, so sweep its
    // whole input domain: every 4-hex-digit PCI device id.
    for id in 0..=0xffff_u32 {
        let id = format!("{id:04x}");
        if let Some(target) = crate::gfx_target_from_amd_pci_device_id(&id)
            && !classified(target)
        {
            unclassified.push(format!(
                "  PCI device id {id} -> {target} has no packaging entry"
            ));
        }
    }
    unclassified.sort();
    unclassified.dedup();
    assert!(
        unclassified.is_empty(),
        "targets a lookup table produces but GFX_TARGET_PACKAGING does not \
         classify:\n{}",
        unclassified.join("\n")
    );
}

/// What the downgrade costs a user, run through the real diagnosis catalog.
///
/// A Ryzen 7000 desktop pairing the Raphael iGPU with an RX 7900 XTX is the
/// textbook iGPU+dGPU collision host: `check_9_igpu_dgpu_collision` exists for
/// it and fires only when `has_apu && has_discrete_amd`. Once `rocminfo` marks
/// the iGPU not-an-APU, `has_apu` is false and the check scores zero, so a user
/// whose workload segfaults on the wrong device is told nothing.
#[test]
fn the_igpu_dgpu_collision_check_still_fires_on_a_raphael_plus_rx7900_host() {
    let igpu = "0000:14:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
                [AMD/ATI] Raphael [1002:164e] (rev c1)";
    let dgpu = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
                [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c]";
    let mut e = Examination {
        os_family: "linux".to_owned(),
        ..Examination::default()
    };
    apply_lspci_gpus(&mut e, &format!("{dgpu}\n{igpu}\n"));
    assert_eq!(e.gpus.len(), 2, "{:?}", e.gpus);
    // rocminfo in KFD node order, matching the PCI order the scan produced.
    apply_rocminfo_gpu_agents(
        &mut e,
        &rocminfo_output(&[
            ("gfx1100", "AMD Radeon RX 7900 XTX"),
            ("gfx1036", "AMD Radeon Graphics"),
        ]),
    );
    summarise_gpu_categories(&mut e);

    let report = crate::diagnose::diagnose(&e, "my training run segfaults");
    let collision = report.matched.iter().find(|d| d.id == "fix-9-igpu-dgpu");
    let Some(collision) = collision.filter(|d| d.score > 0) else {
        panic!(
            "an iGPU+dGPU host must raise fix-9-igpu-dgpu; has_apu={}, \
             has_discrete_amd={}, gpus={:?}",
            e.has_apu, e.has_discrete_amd, e.gpus,
        );
    };
    // And it must name the right card as the one to pin. The whole reason this
    // check exists is that the user cannot tell which ordinal is the dGPU, so a
    // diagnosis that fires but confuses the two is no better than silence.
    let notes = collision
        .fix
        .as_ref()
        .map(|fix| fix.notes.join(" "))
        .unwrap_or_default();
    assert!(
        notes.contains("[\"gfx1100\"]") && notes.contains("[\"gfx1036\"]"),
        "the collision note must name gfx1100 as the discrete GPU and gfx1036 \
         as the APU, got: {notes}"
    );
}

/// The complete list of parts whose APU verdict a `rocminfo` reading downgrades.
#[test]
fn no_known_apu_has_its_verdict_downgraded_by_rocminfo() {
    let mut downgraded: Vec<String> = Vec::new();
    for (device, id, gfx, is_apu) in AMD_LSPCI_DEVICES {
        if !is_apu {
            continue;
        }
        let gpu = pci_scanned(device, id, "0000:04:00.0");
        if gpu.is_apu != Some(true) {
            continue;
        }
        let mut e = Examination {
            gpus: vec![gpu],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(&mut e, &rocminfo_output(&[(gfx, "AMD Radeon Graphics")]));
        summarise_gpu_categories(&mut e);
        if !e.has_apu {
            downgraded.push(format!(
                "  lspci {device:?} [1002:{id}] + rocminfo {gfx} -> has_apu=false, \
                 has_discrete_amd={}",
                e.has_discrete_amd
            ));
        }
    }
    assert!(
        downgraded.is_empty(),
        "rocminfo downgraded an APU verdict the PCI scan already reached:\n{}",
        downgraded.join("\n")
    );
}

/// The downgrade driven through the *whole* post-PCI sequence, against a
/// planted KFD topology, rather than through `apply_rocminfo_gpu_agents` alone.
///
/// A Ryzen 6800H laptop: `lspci` names the iGPU "Rembrandt [Radeon 680M]", the
/// kernel exposes one KFD node for it at `0000:04:00.0` reporting
/// `gfx_target_version 100305` (gfx1035), and `rocminfo` agrees. Every source
/// describes an APU; the report must not call it a discrete GPU.
#[test]
fn a_rembrandt_laptop_is_not_reported_as_a_discrete_gpu() {
    let root = std::env::temp_dir().join(format!(
        "rocm-core-proptest-kfd-{}-{}",
        std::process::id(),
        crate::unix_time_millis()
    ));
    let nodes = root.join("nodes");
    std::fs::create_dir_all(nodes.join("0")).expect("plant the CPU node");
    std::fs::write(
        nodes.join("0").join("properties"),
        "cpu_cores_count 16\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
    )
    .expect("plant the CPU node properties");
    std::fs::create_dir_all(nodes.join("1")).expect("plant the GPU node");
    std::fs::write(
        nodes.join("1").join("properties"),
        "simd_count 12\ngfx_target_version 100305\nlocation_id 1024\ndomain 0\n",
    )
    .expect("plant the GPU node properties");

    // The whole Linux sequence from the PCI scan on: `apply_lspci_gpus` is the
    // scan itself, minus only the process launch.
    let mut e = Examination::default();
    apply_lspci_gpus(
        &mut e,
        "0000:04:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
         [AMD/ATI] Rembrandt [Radeon 680M] [1002:1681] (rev c8)",
    );
    assert_eq!(
        e.gpus.first().and_then(|gpu| gpu.is_apu),
        Some(true),
        "the PCI scan alone already knows Rembrandt is an APU: {:?}",
        e.gpus
    );
    super::probe_gpus_after_lspci(
        &mut e,
        super::GpuProbeSources {
            kfd_nodes: &nodes,
            rocminfo: Some(&rocminfo_output(&[("gfx1035", "AMD Radeon Graphics")])),
            sysfs_gfx_target: || None,
        },
    );
    summarise_gpu_categories(&mut e);
    std::fs::remove_dir_all(&root).ok();

    assert_eq!(e.gpus.len(), 1, "one GPU: {:?}", e.gpus);
    assert_eq!(e.gpus[0].gfx_target, "gfx1035");
    assert!(
        e.has_apu,
        "a Radeon 680M iGPU must report has_apu, got {:?}",
        e.gpus
    );
    assert!(
        !e.has_discrete_amd,
        "an iGPU-only laptop must not report has_discrete_amd, got {:?}",
        e.gpus
    );
}

/// The Linux PCI scan classifies a row by its device id when the name carries
/// nothing.
///
/// `lspci` prints the bare word `Device` for an id `pci.ids` does not know, so
/// on these rows only the `[1002:xxxx]` tag can identify the part. Every id the
/// device-id table maps is swept, so a scan that falls back to reading the name
/// alone fails here for all of them at once.
#[test]
fn an_lspci_row_is_classified_by_its_device_id_when_its_name_says_nothing() {
    let mut wrong: Vec<String> = Vec::new();
    for (id, target) in every_mapped_pci_device_id() {
        let gpu = pci_scanned("Device", &id, "0000:66:00.0");
        if gpu.gfx_target != target || gpu.is_apu != Some(gfx_is_apu_family(target)) {
            wrong.push(format!("  [1002:{id}] is {target}, scanned as {gpu:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "lspci rows the device id identifies but the scan did not:\n{}",
        wrong.join("\n")
    );
}

/// The Windows display probe classifies a row by its PNP id when the name
/// carries nothing.
///
/// "AMD Radeon(TM) Graphics" is not a placeholder: it is what Renoir, Rembrandt
/// and Raphael iGPUs actually report as their adapter name on Windows, so for
/// the commonest APUs the PNP id is the only thing on the row that names the
/// part.
#[test]
fn a_windows_row_is_classified_by_its_pnp_id_when_its_name_says_nothing() {
    assert_eq!(
        classify_amd_marketing_name("AMD Radeon(TM) Graphics").0,
        "",
        "the premise: the generic name alone must resolve nothing"
    );
    let mut wrong: Vec<String> = Vec::new();
    for (id, target) in every_mapped_pci_device_id() {
        let gpu = windows_scanned("AMD Radeon(TM) Graphics", &id);
        if gpu.gfx_target != target || gpu.is_apu != Some(gfx_is_apu_family(target)) {
            wrong.push(format!("  DEV_{id} is {target}, scanned as {gpu:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "Windows rows the PNP id identifies but the probe did not:\n{}",
        wrong.join("\n")
    );
}

/// `hipInfo` revises the display probe's verdict when it reports a target whose
/// packaging is known, and only then.
///
/// The first half is a host that exists: a Renoir laptop's adapter is named
/// "AMD Radeon(TM) Graphics" and its device id is one the table does not map,
/// so the display probe gives it no target and the PCI-row default, and only
/// `hipInfo`'s `gcnArchName` can say what it is.
///
/// The second half pins the fold's contract on a state the Windows display
/// probe cannot currently produce — a GPU already called an APU, with no
/// target. The display probe only reaches `Some(true)` by resolving a target,
/// so through the real pipeline the "leave an unknown target's verdict alone"
/// rule changes no output today. It is asserted anyway because it is the rule
/// every other fold in this module follows, and the first new source of a
/// target-less verdict would otherwise inherit the downgrade silently.
#[test]
fn a_hipinfo_target_revises_a_verdict_only_when_its_packaging_is_known() {
    let mut e = Examination::default();
    apply_windows_display_rows(&mut e, &windows_row("AMD Radeon(TM) Graphics", "1636"));
    assert_eq!(e.gpus[0].gfx_target, "", "{:?}", e.gpus);
    apply_hipinfo_gcn_arch_names(&mut e, "device# 0\n  gcnArchName: gfx90c:xnack-\n");
    assert_eq!(e.gpus[0].gfx_target, "gfx90c", "{:?}", e.gpus);
    assert_eq!(
        e.gpus[0].is_apu,
        Some(true),
        "a known integrated target must revise the display probe's default: {:?}",
        e.gpus
    );

    let mut e = Examination {
        gpus: vec![Gpu {
            name: "AMD Radeon(TM) Graphics".to_owned(),
            is_amd: true,
            is_apu: Some(true),
            ..Gpu::default()
        }],
        ..Examination::default()
    };
    apply_hipinfo_gcn_arch_names(&mut e, "  gcnArchName: gfx1154\n");
    assert_eq!(e.gpus[0].gfx_target, "gfx1154", "{:?}", e.gpus);
    assert_eq!(
        e.gpus[0].is_apu,
        Some(true),
        "an unrecognised target is not evidence of a discrete GPU: {:?}",
        e.gpus
    );
}

/// An AMD PCI row nothing can identify is reported as discrete, deliberately,
/// and both sides of that trade are pinned here — see `pci_row_is_apu`.
///
/// What the default buys: a Ryzen 7000 desktop with an RX 5700 XT. Navi 10 is
/// in neither SKU table, so its row resolves no target; reported as "cannot
/// say", it would clear `has_discrete_amd` and the iGPU+dGPU collision check
/// would never fire on the host it is written for.
///
/// What it costs: a Mendocino laptop, whose iGPU likewise resolves nothing, is
/// reported with `has_discrete_amd` set. The collision check still needs
/// `has_apu`, which that host does not have, so no diagnosis fires — the cost
/// is a wrong field in `rocm examine --json`, which is asserted too rather
/// than left implicit.
#[test]
fn an_unidentifiable_pci_row_is_reported_as_discrete_and_this_is_what_that_costs() {
    let raphael = "0000:14:00.0 VGA compatible controller [0300]: Advanced Micro Devices, \
                   Inc. [AMD/ATI] Raphael [1002:164e] (rev c1)";
    let navi10 = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, \
                  Inc. [AMD/ATI] Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT] \
                  [1002:731f] (rev c1)";
    let mendocino = "0000:04:00.0 VGA compatible controller [0300]: Advanced Micro Devices, \
                     Inc. [AMD/ATI] Mendocino [1002:1506] (rev c1)";
    let host = |lines: &[&str]| {
        let mut e = Examination {
            os_family: "linux".to_owned(),
            ..Examination::default()
        };
        apply_lspci_gpus(&mut e, &lines.join("\n"));
        summarise_gpu_categories(&mut e);
        let fires = crate::diagnose::diagnose(&e, "my training run segfaults")
            .matched
            .iter()
            .any(|d| d.id == "fix-9-igpu-dgpu" && d.score > 0);
        (e, fires)
    };

    let (hybrid, fires) = host(&[navi10, raphael]);
    assert_eq!(
        hybrid.gpus[0].gfx_target, "",
        "the premise: Navi 10 is unresolved"
    );
    assert!(
        hybrid.has_apu && hybrid.has_discrete_amd && fires,
        "an unresolved discrete card beside a known iGPU must still raise the \
         collision diagnosis: {:?}",
        hybrid.gpus
    );

    let (laptop, fires) = host(&[mendocino]);
    assert_eq!(
        laptop.gpus[0].gfx_target, "",
        "the premise: Mendocino is unresolved"
    );
    assert!(
        laptop.has_discrete_amd && !laptop.has_apu,
        "the known cost: an unresolved iGPU reads as discrete: {:?}",
        laptop.gpus
    );
    assert!(
        !fires,
        "and the cost stops there — no iGPU+dGPU diagnosis on a one-GPU laptop"
    );
}

/// No discrete part in the corpus is ever classified as an APU.
///
/// The APU sweeps above only look for misses; this is the other direction,
/// and it is what keeps a codename pattern in the marketing table from being
/// broader than the part it names.
#[test]
fn no_discrete_part_is_classified_as_an_apu() {
    let mut wrong: Vec<String> = Vec::new();
    for (name, gfx, is_apu) in AMD_MARKETING_NAMES {
        if !is_apu && classify_amd_marketing_name(name).1 {
            wrong.push(format!("  marketing name {name:?} ({gfx})"));
        }
    }
    for (device, id, gfx, is_apu) in AMD_LSPCI_DEVICES {
        if !is_apu && pci_scanned(device, id, "0000:03:00.0").is_apu == Some(true) {
            wrong.push(format!("  lspci {device:?} [1002:{id}] ({gfx})"));
        }
    }
    assert!(
        wrong.is_empty(),
        "discrete parts classified as APUs:\n{}",
        wrong.join("\n")
    );
}

/// Measure how far the generators actually reach, by drawing from them
/// directly and tallying which branches each draw lands in.
///
/// A generator that never samples the interesting region passes every property
/// vacuously, so the reach is asserted, not assumed: a regression that stops
/// the corpus from parsing (say, a changed `lspci` line shape) must fail here
/// rather than silently turn the whole file green.
#[test]
fn generator_reach_is_measured_and_sufficient() {
    const DRAWS: u32 = 4096;
    let mut runner = TestRunner::new(Config::with_cases(DRAWS));

    let mut lspci_parsed = 0u64;
    let mut lspci_gpu_class = 0u64;
    let mut lspci_apu_device = 0u64;
    let mut lspci_classified_apu = 0u64;
    let mut lspci_gfx_resolved = 0u64;
    for _ in 0..DRAWS {
        let (line, device) = lspci_line()
            .new_tree(&mut runner)
            .expect("lspci strategy")
            .current();
        if is_lspci_gpu_line(&line) {
            lspci_gpu_class += 1;
        }
        let name = extract_lspci_name(&line);
        if !name.is_empty() {
            lspci_parsed += 1;
        }
        if device.3 {
            lspci_apu_device += 1;
        }
        let (gfx, is_apu) = classify_amd_marketing_name(&name);
        if is_apu {
            lspci_classified_apu += 1;
        }
        if !gfx.is_empty() {
            lspci_gfx_resolved += 1;
        }
    }

    let mut marketing_examine = 0u64;
    let mut marketing_install = 0u64;
    let mut marketing_both = 0u64;
    let mut marketing_apu_truth = 0u64;
    for _ in 0..DRAWS {
        let (name, truth, _m) = marketing_name()
            .new_tree(&mut runner)
            .expect("marketing strategy")
            .current();
        let examine = classify_amd_marketing_name(&name).0;
        let install = crate::gfx_target_from_amd_marketing_name(&name);
        if !examine.is_empty() {
            marketing_examine += 1;
        }
        if install.is_some() {
            marketing_install += 1;
        }
        if !examine.is_empty() && install.is_some() {
            marketing_both += 1;
        }
        if truth.2 {
            marketing_apu_truth += 1;
        }
    }

    let pct = |n: u64| (n as f64) * 100.0 / f64::from(DRAWS);
    println!(
        "generator reach over {DRAWS} draws each:\n\
         lspci lines     : name parsed {:.1}%, GPU-class {:.1}%, APU silicon {:.1}%, \
         classified APU {:.1}%, gfx resolved {:.1}%\n\
         marketing names : examine resolved {:.1}%, install resolved {:.1}%, both {:.1}%, \
         APU silicon {:.1}%",
        pct(lspci_parsed),
        pct(lspci_gpu_class),
        pct(lspci_apu_device),
        pct(lspci_classified_apu),
        pct(lspci_gfx_resolved),
        pct(marketing_examine),
        pct(marketing_install),
        pct(marketing_both),
        pct(marketing_apu_truth),
    );
    // Floors, not exact counts: the draws are random, but a generator that
    // reaches none of these is worthless.
    assert!(
        lspci_parsed > u64::from(DRAWS) / 2,
        "fewer than half the lspci draws produced a parseable device name"
    );
    // A third, not a half: one class in five is the PCI bridge the probe must
    // *not* enumerate, and the case/tab mutations deliberately break the
    // case-sensitive class match so the negative branch is sampled too.
    assert!(
        lspci_gpu_class > u64::from(DRAWS) / 3,
        "fewer than a third of the lspci draws landed on an enumerated GPU class"
    );
    assert!(
        lspci_apu_device > u64::from(DRAWS) / 4,
        "the lspci corpus barely samples APU silicon"
    );
    assert!(
        lspci_gfx_resolved > 0,
        "no lspci draw ever resolved a gfx target"
    );
    assert!(
        marketing_both > u64::from(DRAWS) / 8,
        "the two marketing tables almost never both answer, so the agreement \
         property is near-vacuous"
    );
}
