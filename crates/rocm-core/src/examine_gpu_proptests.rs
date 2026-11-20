// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property-based tests for how `examine` enumerates and classifies AMD GPUs.
//!
//! The generators are built from *real* fixture shapes -- `lspci -nn -D` lines,
//! `rocminfo` agent blocks, `Win32_VideoController` rows and AMD marketing
//! names -- and then mutated (case, whitespace, decorations, dropped brackets,
//! truncation). A uniform-random string generator never reaches the branches
//! that matter, so every strategy starts from a corpus and perturbs it.
//!
//! Each property states an invariant the *report* must hold, so a failure names
//! a defect a user would see. `generator_reach_is_measured_and_sufficient`
//! asserts that the generators actually reach the regions the properties are
//! about, so the harness cannot quietly go vacuous.

use super::linux_gpus::{
    GpuProbeSources, apply_lspci_gpus, apply_rocminfo_gpu_agents, apu_verdict_from_target,
    classify_amd_marketing_name, extract_lspci_device_id, extract_lspci_name, gfx_is_apu_family,
    is_lspci_gpu_line, probe_gpus_after_lspci, summarise_gpu_categories,
};
use super::windows::{fill_unidentified_gpus_from_hipinfo, parse_gpus_windows_output};
use super::{Examination, Gpu};
use crate::test_support::proptest_cases;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, TestRunner};

// ---------------------------------------------------------------------------
// Corpora: real strings, from pci.ids, this repo's own fixtures and the
// hardware named in ROCm/rocm-cli#448 / #449.
// ---------------------------------------------------------------------------

/// `(device_name_as_lspci_prints_it, pci_device_id, true_gfx_target,
/// is_really_an_apu)`.
///
/// The names are the `pci.ids` device strings an `lspci -nn` line carries after
/// the `[AMD/ATI]` vendor tag. The last two fields are ground truth about the
/// silicon, not about what the code says, so one row is always a *coherent*
/// host: the gfx target really is the one that device reports.
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
    // APUs, oldest first. Every one is an integrated GPU sharing system
    // memory; `has_apu` must be true on a host that has one.
    ("Renoir", "1636", "gfx90c", true),
    ("Cezanne", "1638", "gfx90c", true),
    ("Lucienne", "164c", "gfx90c", true),
    ("Barcelo", "15e7", "gfx90c", true),
    ("VanGogh [AMD Custom GPU 0405]", "163f", "gfx1033", true),
    ("Rembrandt [Radeon 680M]", "1681", "gfx1035", true),
    ("Raphael", "164e", "gfx1036", true),
    // No device-id row: only the codename identifies it.
    ("Mendocino", "1506", "gfx1037", true),
    ("Phoenix1", "15bf", "gfx1103", true),
    ("Phoenix3", "1900", "gfx1103", true),
    ("Strix [Radeon 880M / 890M]", "150e", "gfx1150", true),
    ("Krackan Point [Radeon 860M]", "1114", "gfx1152", true),
    // Strix Halo, as reported in ROCm/rocm-cli#449: pci.ids has no entry, so
    // `lspci` prints the bare word "Device", and the device-id table has no
    // row for 1586 either (#507).
    ("Device", "1586", "gfx1151", true),
];

/// Corpus rows that nothing can identify: no `pci.ids` name, and an id the
/// device-id table does not map. They stay in the corpus for the parsing and
/// totality properties, and are excluded from the classification sweeps.
const UNIDENTIFIABLE_LSPCI_DEVICES: &[&str] = &["Device"];

/// PCI classes an `lspci -nn` line can carry: the ones the probe enumerates,
/// and last the bridge it must not.
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
/// `(name, true_gfx_target, is_really_an_apu, pci_device_id)`. The device id is
/// the one the part's Windows PNP id carries, where the device-id table maps
/// it; [`windows_corpus`] draws only those rows.
const AMD_MARKETING_NAMES: &[(&str, &str, bool, Option<&str>)] = &[
    ("AMD Radeon RX 7900 XTX", "gfx1100", false, None),
    ("AMD Radeon RX 7800 XT", "gfx1101", false, None),
    ("AMD Radeon RX 7600", "gfx1102", false, None),
    ("AMD Radeon RX 9070 XT", "gfx1201", false, Some("7550")),
    ("AMD Radeon PRO W7900", "gfx1100", false, None),
    ("AMD Instinct MI300X", "gfx942", false, None),
    ("AMD Radeon RX 6900 XT", "gfx1030", false, Some("73a0")),
    ("AMD Radeon RX 6600M", "gfx1032", false, Some("73e0")),
    ("AMD Radeon RX 6500M", "gfx1034", false, Some("743f")),
    // APUs.
    (
        "AMD Radeon(TM) 610M Graphics",
        "gfx1036",
        true,
        Some("164e"),
    ),
    (
        "AMD Radeon(TM) 660M Graphics",
        "gfx1035",
        true,
        Some("1681"),
    ),
    (
        "AMD Radeon(TM) 680M Graphics",
        "gfx1035",
        true,
        Some("1681"),
    ),
    (
        "AMD Radeon(TM) 740M Graphics",
        "gfx1103",
        true,
        Some("15bf"),
    ),
    (
        "AMD Radeon(TM) 760M Graphics",
        "gfx1103",
        true,
        Some("15bf"),
    ),
    (
        "AMD Radeon(TM) 780M Graphics",
        "gfx1103",
        true,
        Some("15bf"),
    ),
    ("AMD Radeon(TM) 820M Graphics", "gfx1153", true, None),
    (
        "AMD Radeon(TM) 840M Graphics",
        "gfx1152",
        true,
        Some("1114"),
    ),
    (
        "AMD Radeon(TM) 860M Graphics",
        "gfx1152",
        true,
        Some("1114"),
    ),
    ("AMD Radeon(TM) 880M Graphics", "gfx1150", true, None),
    ("AMD Radeon(TM) 890M Graphics", "gfx1150", true, None),
    ("AMD Radeon 8040S Graphics", "gfx1151", true, None),
    ("AMD Radeon 8050S Graphics", "gfx1151", true, None),
    ("AMD Radeon 8060S Graphics", "gfx1151", true, None),
    (
        "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        "gfx1151",
        true,
        None,
    ),
    ("AMD Custom GPU 0405", "gfx1033", true, Some("163f")),
];

/// gfx targets a real host reports, with ground truth about packaging.
///
/// gfx902, gfx909 and gfx1037 are here because no lookup table produces them:
/// they reach the CLI only through `gfx_target_from_gc_version`, from KFD's
/// packed `gfx_target_version` or DRM ip-discovery, so the table drift guard
/// cannot see them and ground truth has to.
const GFX_TARGETS: &[(&str, bool)] = &[
    ("gfx900", false),
    ("gfx906", false),
    ("gfx908", false),
    ("gfx90a", false),
    ("gfx942", false),
    ("gfx950", false),
    ("gfx1010", false),
    ("gfx1011", false),
    ("gfx1012", false),
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

/// Well-formed gfx targets that no APU family list here names: silicon newer
/// than the tables. A GPU reporting one of these is evidence of a GPU, not of
/// its packaging, so the target rule (`apu_verdict_from_target`) leaves the
/// verdict as it was.
const UNCLASSIFIED_GFX_TARGETS: &[&str] = &["gfx1170", "gfx1180", "gfx1260", "gfx1300"];

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
/// can destroy the token a lookup keys on. Properties that assert a
/// device-specific verdict draw from this narrower set rather than filtering
/// the wider one, which would spend proptest's reject budget on draws it was
/// always going to discard.
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
// Strategies and fixtures
// ---------------------------------------------------------------------------

/// A plausible PCI address, including the high bus numbers a Strix Halo iGPU
/// enumerates on (`0000:66:00.0` in #449).
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

/// A marketing name drawn from the real corpus, respelled in a way that still
/// names the same device, with the ground truth for that device.
fn marketing_name_spelled_differently() -> impl Strategy<
    Value = (
        String,
        (&'static str, &'static str, bool, Option<&'static str>),
    ),
> {
    (
        proptest::sample::select(AMD_MARKETING_NAMES),
        identity_preserving_mutation(),
    )
        .prop_map(|(entry, m)| (apply_mutation(entry.0, m), entry))
}

/// Marketing names of parts that really are APUs, differently spelled.
fn apu_marketing_name() -> impl Strategy<
    Value = (
        String,
        (&'static str, &'static str, bool, Option<&'static str>),
    ),
> {
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

/// `lspci` entries for parts that really are APUs and that the scan can
/// identify, by name or by device id.
fn apu_lspci_devices() -> Vec<(&'static str, &'static str, &'static str, bool)> {
    let apus: Vec<_> = AMD_LSPCI_DEVICES
        .iter()
        .copied()
        .filter(|device| device.3 && !UNIDENTIFIABLE_LSPCI_DEVICES.contains(&device.0))
        .collect();
    assert!(!apus.is_empty(), "the lspci corpus has no identifiable APU");
    apus
}

/// What a `rocminfo` agent at an APU's PCI address reports, as `(target,
/// Memory Properties value)`: the device's own target or another one --
/// unclassified or discrete -- and a `Memory Properties:` value that says
/// `APU`, says nothing (`Some("")`, the "discrete" reading) or is not printed.
///
/// The other targets and the empty value are the draws that can tell the rule
/// from a regression. A held APU target is never replaced by the agent's, and
/// an APU-family target outranks the empty value's "discrete" reading; an
/// agent target that replaced the held one, or a flag that outranked the
/// family list, would turn these draws discrete. On the device's own target
/// with an `APU` flag every variant computes the same answer.
///
/// Shared by [`rocminfo_never_downgrades_an_apu_verdict`] and
/// `generator_reach_is_measured_and_sufficient`, so the reach measured is the
/// reach the property draws.
fn rocminfo_agent_at_an_apu(device_target: &'static str) -> impl Strategy<Value = AgentReading> {
    let discrete: Vec<&'static str> = GFX_TARGETS
        .iter()
        .filter(|(_, apu)| !apu)
        .map(|(gfx, _)| *gfx)
        .collect();
    (
        prop_oneof![
            2 => Just(device_target),
            1 => proptest::sample::select(UNCLASSIFIED_GFX_TARGETS),
            1 => proptest::sample::select(discrete),
        ],
        proptest::sample::select(&[None, Some(""), Some("APU")][..]),
    )
}

/// An APU the PCI scan identifies, and what `rocminfo` reports at its address.
/// One [`AMD_LSPCI_DEVICES`] row.
type LspciDevice = (&'static str, &'static str, &'static str, bool);

/// What a `rocminfo` agent reports: `(target, Memory Properties value)`.
type AgentReading = (&'static str, Option<&'static str>);

fn apu_host_with_a_rocminfo_agent() -> impl Strategy<Value = (LspciDevice, AgentReading)> {
    proptest::sample::select(apu_lspci_devices())
        .prop_flat_map(|device| (Just(device), rocminfo_agent_at_an_apu(device.2)))
}

/// Windows adapters that print only the generic `AMD Radeon(TM) Graphics`, as
/// `(name, true_gfx_target, is_really_an_apu, pci_device_id)`.
///
/// The name resolves no part, so the device id is the only evidence a probe
/// has. These are the rows that tell a probe asking the id from one that asks
/// only the name: on every [`AMD_MARKETING_NAMES`] row the two agree, so a
/// Windows probe that stopped reading the PNP id would still pass on them.
/// Renoir, Cezanne and Lucienne sell their Vega iGPU under this name, and so
/// does the Raphael desktop iGPU.
const GENERIC_NAME_WINDOWS_ROWS: &[(&str, &str, bool, &str)] = &[
    ("AMD Radeon(TM) Graphics", "gfx90c", true, "1636"),
    ("AMD Radeon(TM) Graphics", "gfx90c", true, "1638"),
    ("AMD Radeon(TM) Graphics", "gfx90c", true, "164c"),
    ("AMD Radeon(TM) Graphics", "gfx1036", true, "164e"),
];

/// The Windows rows the display-probe property draws from: every
/// marketing-name row whose PNP device id the device-id table maps, plus
/// [`GENERIC_NAME_WINDOWS_ROWS`].
///
/// Asserted to stay large enough to matter, and to keep rows whose name
/// resolves nothing, so the property cannot shrink to rows on which reading
/// the name alone gives the right answer without a test failing.
fn windows_corpus() -> Vec<(&'static str, &'static str, bool, &'static str)> {
    let rows: Vec<_> = AMD_MARKETING_NAMES
        .iter()
        .filter_map(|(name, gfx, apu, id)| id.map(|id| (*name, *gfx, *apu, id)))
        .chain(GENERIC_NAME_WINDOWS_ROWS.iter().copied())
        .collect();
    assert!(
        rows.len() >= 14,
        "only {} Windows rows carry a device id; the Windows property \
         would sample a handful of parts",
        rows.len()
    );
    let name_says_nothing = rows
        .iter()
        .filter(|(name, ..)| crate::gfx_target::gfx_target_from_amd_marketing_name(name).is_none())
        .count();
    assert!(
        name_says_nothing >= 2,
        "only {name_says_nothing} Windows rows have a name that resolves no part; \
         without them the property cannot tell the device id from the name"
    );
    rows
}

/// What the real PCI scan makes of one corpus device.
///
/// Every test that needs a PCI-sourced GPU goes through this rather than
/// assembling one from the classifiers: a test that re-implements the probe's
/// logic pins the re-implementation, and keeps passing when the probe stops
/// reading the device id.
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

/// Every PCI device id the device-id table maps, with its target.
///
/// The table is a `match`, not an iterable, so its whole input domain is swept.
fn every_mapped_pci_device_id() -> Vec<(String, &'static str)> {
    let ids: Vec<_> = (0..=0xffff_u32)
        .map(|id| format!("{id:04x}"))
        .filter_map(|id| crate::gfx_target::gfx_target_from_amd_pci_device_id(&id).map(|t| (id, t)))
        .collect();
    assert!(!ids.is_empty(), "the device-id table maps no id at all");
    ids
}

/// The `BDFID` `rocminfo` prints for a `domain:bus:device.function` address:
/// the kernel's `pci_dev_id()` encoding, `(bus << 8) | (device << 3) |
/// function`, in decimal. `rocminfo` prints no domain.
fn bdfid(address: &str) -> u32 {
    let mut parts = address.split([':', '.']).skip(1);
    let mut field = |radix| {
        u32::from_str_radix(parts.next().expect("an address field"), radix).expect("a number")
    };
    let (bus, device, function) = (field(16), field(16), field(10));
    (bus << 8) | (device << 3) | function
}

/// A `rocminfo` listing: a CPU agent, then one GPU agent per
/// `(target, marketing name, PCI address, Memory Properties value)`, each with
/// the `BDFID` for its address and, where the value is `Some`, a `Memory
/// Properties:` line carrying it -- the shape ROCm 7 prints.
fn rocminfo_output(agents: &[(&str, &str, &str, Option<&str>)]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "==========\nHSA Agents\n==========\n*******\nAgent 1\n*******\n  \
         Name:                    AMD Ryzen 9 7950X\n  \
         Marketing Name:          AMD Ryzen 9 7950X\n  \
         Device Type:             CPU\n  \
         BDFID:                   0\n",
    );
    for (index, (gfx, marketing, address, memory)) in agents.iter().enumerate() {
        let _ = write!(
            out,
            "*******\nAgent {}\n*******\n  Name:                    {gfx}\n  \
             Marketing Name:          {marketing}\n  Device Type:             GPU\n  \
             BDFID:                   {}\n",
            index + 2,
            bdfid(address)
        );
        if let Some(memory) = memory {
            let _ = writeln!(out, "  Memory Properties:       {memory}");
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(Config {
        cases: proptest_cases(2048),
        failure_persistence: None,
        ..Config::default()
    })]

    /// Totality: no fixture-derived or mutated `lspci` line may panic any of
    /// the classifiers, or the PCI scan that composes them, and a single line
    /// never enumerates as more than one GPU.
    #[test]
    fn classifiers_never_panic_on_a_mutated_lspci_line((line, _) in lspci_line()) {
        let _ = is_lspci_gpu_line(&line);
        let _ = extract_lspci_device_id(&line);
        let (gfx, _) = classify_amd_marketing_name(&extract_lspci_name(&line));
        let _ = apu_verdict_from_target(None, &gfx);
        let mut e = Examination::default();
        apply_lspci_gpus(&mut e, &line);
        prop_assert!(e.gpus.len() <= 1, "{line:?} enumerated as {:?}", e.gpus);
    }

    /// `extract_lspci_name` recovers exactly the vendor-plus-device text of an
    /// unmutated `lspci -nn` line: everything between the class `]:` and the
    /// trailing `[vendor:device]`. And the scan enumerates the line exactly
    /// when its class is one the probe lists: a PCI bridge carrying an AMD
    /// GPU's name is not a GPU.
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
        prop_assert_eq!(extract_lspci_device_id(&line), Some(device.1.to_owned()));
        let mut e = Examination::default();
        apply_lspci_gpus(&mut e, &line);
        prop_assert_eq!(e.gpus.len(), usize::from(class.1), "{:?}: {:?}", line, e.gpus);
    }

    /// A name that names an APU must not be reported as "not an APU".
    #[test]
    fn examine_never_calls_a_known_apu_discrete((name, truth) in apu_marketing_name()) {
        let (target, is_apu) = classify_amd_marketing_name(&name);
        prop_assert!(
            is_apu,
            "{name} is an APU ({}) but examine says is_apu=false, gfx_target={target:?}",
            truth.1,
        );
    }

    /// Folding a `rocminfo` reading into the report never *downgrades* an APU
    /// the PCI scan identified, whatever the agent at that address reports.
    ///
    /// The scan gave the row an APU target, so by the rule table at
    /// `apu_verdict_from_target` the agent's own target does not replace it
    /// (only an empty one is filled), and an APU-family target outranks a
    /// `Memory Properties:` line without `APU`. [`rocminfo_agent_at_an_apu`]
    /// draws the agent targets and flags that would turn the row discrete if
    /// either rule were lost.
    #[test]
    fn rocminfo_never_downgrades_an_apu_verdict(
        (device, (agent_target, memory)) in apu_host_with_a_rocminfo_agent(),
        addr in pci_address(),
    ) {
        let gpu = pci_scanned(device.0, device.1, &addr);
        prop_assert!(
            gpu.is_apu == Some(true) && gfx_is_apu_family(&gpu.gfx_target),
            "the PCI scan must already know {} is an APU, by an APU target, before \
             this property can say anything about preserving that verdict: {:?}",
            device.0,
            gpu,
        );
        let held = gpu.gfx_target.clone();

        let mut e = Examination {
            gpus: vec![gpu],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_output(&[(agent_target, "AMD Radeon Graphics", &addr, memory)]),
        );
        summarise_gpu_categories(&mut e);
        prop_assert_eq!(&e.gpus[0].gfx_target, &held, "a held target is kept");
        prop_assert!(
            e.has_apu && !e.has_discrete_amd,
            "lspci classified {} as an APU, then a rocminfo agent reporting {} with \
             Memory Properties {:?} made it discrete: {:?}",
            device.0,
            agent_target,
            memory,
            e.gpus,
        );
    }

    /// Classification does not depend on spelling: case, inner/outer
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

    /// The APU family list and the target rule agree with ground truth for
    /// every target a real host reports, whatever its case or its feature or
    /// generic-target suffix: `gfx_is_apu_family` says yes or no, and
    /// `apu_verdict_from_target` makes an APU integrated and leaves any other
    /// verdict as it was.
    #[test]
    fn apu_answers_match_ground_truth(
        gfx in proptest::sample::select(GFX_TARGETS),
        suffix in proptest::sample::select(
            &["", ":xnack-", ":sramecc+:xnack-", ":sramecc-", "-generic", "-generic:xnack-"][..]
        ),
        upper in any::<bool>(),
        current in proptest::sample::select(&[None, Some(false), Some(true)][..]),
    ) {
        let spelled = if upper {
            format!("{}{suffix}", gfx.0.to_uppercase())
        } else {
            format!("{}{suffix}", gfx.0)
        };
        prop_assert_eq!(gfx_is_apu_family(&spelled), gfx.1, "gfx_is_apu_family({})", spelled);
        let expected = if gfx.1 { Some(true) } else { current };
        prop_assert_eq!(
            apu_verdict_from_target(current, &spelled),
            expected,
            "apu_verdict_from_target({:?}, {})",
            current,
            spelled,
        );
    }

    /// `examine`'s Windows display probe and the install-side display decoder
    /// read the same adapter to the same target.
    ///
    /// They share the device-id and name tables but not the wiring: `examine`
    /// splits `name<TAB>driver<TAB>pnp` and asks the PNP id first, the install
    /// side reads `name<TAB>pnp` rows and ranks a gfx90c iGPU below every other
    /// adapter. Each side is also held to ground truth, so a change to either
    /// wiring that stops reading the PNP id -- reads it from the wrong column,
    /// or resolves from the name alone -- fails on the generic-name rows
    /// ([`GENERIC_NAME_WINDOWS_ROWS`]), whose name resolves nothing.
    ///
    /// What this cannot see is the order of the two lookups where both answer:
    /// on every real row the id and the name agree, so "id first" and "name
    /// first" give the same target. That order is pinned by the hand-built
    /// disagreeing pair in `windows_gpu_probe_prefers_the_pci_id_match_when_one_exists`.
    #[test]
    fn examine_and_the_install_probe_read_a_windows_row_alike(
        (name, gfx, apu, device_id) in proptest::sample::select(windows_corpus()),
    ) {
        let mut e = Examination::default();
        parse_gpus_windows_output(&windows_row(name, device_id), &mut e);
        prop_assert_eq!(e.gpus.len(), 1);
        let install = crate::gfx_target::parse_windows_display_gfx_target(&format!(
            "{name}\tPCI\\VEN_1002&DEV_{device_id}&SUBSYS_00001002&REV_C1"
        ));
        prop_assert_eq!(install.as_deref(), Some(gfx), "install side, {}", name);
        prop_assert_eq!(&e.gpus[0].gfx_target, gfx, "examine, {}", name);
        prop_assert_eq!(e.gpus[0].is_apu, Some(apu), "examine, {}", name);
    }
}

// ---------------------------------------------------------------------------
// Exhaustive sweeps: the whole corpus, so a failure lists every case rather
// than the one minimal case a shrinker settles on.
// ---------------------------------------------------------------------------

/// Every APU in the corpora is classified as one, by every door.
#[test]
fn every_known_apu_is_classified_as_an_apu() {
    let mut misses: Vec<String> = Vec::new();
    for (name, target, is_apu, _) in AMD_MARKETING_NAMES {
        if !is_apu {
            continue;
        }
        let (examine_target, examine_is_apu) = classify_amd_marketing_name(name);
        if !examine_is_apu {
            misses.push(format!(
                "  marketing name {name:?} ({target}): is_apu=false, gfx_target={examine_target:?}"
            ));
        }
    }
    for (gfx, is_apu) in GFX_TARGETS {
        if *is_apu && !gfx_is_apu_family(gfx) {
            misses.push(format!("  gfx target {gfx}: gfx_is_apu_family says false"));
        }
    }
    for (device, id, _gfx, is_apu) in AMD_LSPCI_DEVICES {
        if !is_apu || UNIDENTIFIABLE_LSPCI_DEVICES.contains(device) {
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

/// No discrete part in the corpora is classified as an APU -- the other
/// direction, which keeps a codename pattern from reaching past its part.
#[test]
fn no_discrete_part_is_classified_as_an_apu() {
    let mut wrong: Vec<String> = Vec::new();
    for (name, gfx, is_apu, _) in AMD_MARKETING_NAMES {
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

/// A reported `gfx_target` never *contradicts* the part: an unresolved lookup
/// is recoverable, a wrong answer selects the wrong runtime build.
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
    for (name, gfx, _is_apu, _) in AMD_MARKETING_NAMES {
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

/// Every gfx target a lookup table can hand back is in the ground-truth
/// corpus, and the APU family list agrees with it there.
///
/// The name and device-id tables are keyed by name and id, and the packaging
/// knowledge by target, so nothing ties them together. A table entry for an
/// APU target the family list lacks would be reported with the scan's "not an
/// APU" default, and no later probe could correct it from that target; this
/// fails instead, and names the entry.
#[test]
fn every_target_a_lookup_table_produces_has_its_packaging_known() {
    let truth = |target: &str| {
        GFX_TARGETS
            .iter()
            .find(|(gfx, _)| *gfx == target)
            .map(|t| t.1)
    };
    let mut wrong: Vec<String> = Vec::new();
    let mut check = |source: String, target: &str| match truth(target) {
        None => wrong.push(format!("  {source} -> {target}: not in GFX_TARGETS")),
        Some(apu) if gfx_is_apu_family(target) != apu => wrong.push(format!(
            "  {source} -> {target}: gfx_is_apu_family says {}, ground truth {apu}",
            !apu
        )),
        Some(_) => {}
    };
    for entry in crate::gfx_target::AMD_MARKETING_GFX_TARGETS {
        check(
            format!("marketing pattern {:?}", entry.pattern),
            entry.gfx_target,
        );
    }
    for (id, target) in every_mapped_pci_device_id() {
        check(format!("PCI device id {id}"), target);
    }
    wrong.sort();
    wrong.dedup();
    assert!(
        wrong.is_empty(),
        "lookup-table targets whose packaging is not known:\n{}",
        wrong.join("\n")
    );
    for target in UNCLASSIFIED_GFX_TARGETS {
        assert!(
            !gfx_is_apu_family(target),
            "{target} is meant to be a target no APU family names"
        );
    }
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

// ---------------------------------------------------------------------------
// The folds that follow the PCI scan
// ---------------------------------------------------------------------------

/// Reading `rocminfo` only ever improves the report.
///
/// Each agent is at its row's PCI address and prints no `Memory Properties:`
/// line, so the target rule decides:
///
/// - A Strix Halo whose `lspci` row identifies nothing: the agent fills the
///   empty target, and its APU family makes the row integrated.
/// - A Rembrandt laptop whose agent names a target nothing here classifies: the
///   scan's gfx1035 is kept (an agent only fills an empty target), and so is
///   its APU verdict.
/// - A discrete card the PCI scan could not identify: the agent fills the
///   target, and a target outside the APU families leaves the scan's
///   "not an APU" verdict as it was.
#[test]
fn folding_in_rocminfo_only_ever_adds_knowledge() {
    let host = |lspci_name: &str, device_id: &str, agent_target: &str| {
        let mut e = Examination {
            gpus: vec![pci_scanned(lspci_name, device_id, "0000:66:00.0")],
            ..Examination::default()
        };
        apply_rocminfo_gpu_agents(
            &mut e,
            &rocminfo_output(&[(agent_target, "AMD Radeon Graphics", "0000:66:00.0", None)]),
        );
        summarise_gpu_categories(&mut e);
        e
    };

    let halo = host("Device", "1586", "gfx1151");
    assert_eq!(halo.gpus[0].gfx_target, "gfx1151", "{:?}", halo.gpus);
    assert!(
        halo.has_apu && !halo.has_discrete_amd,
        "a gfx1151 agent establishes the APU verdict the PCI scan could not: {:?}",
        halo.gpus
    );

    let unknown = host("Rembrandt [Radeon 680M]", "1681", "gfx1170");
    assert_eq!(unknown.gpus[0].gfx_target, "gfx1035", "{:?}", unknown.gpus);
    assert!(
        unknown.has_apu && !unknown.has_discrete_amd,
        "an unclassified agent target must leave the PCI scan's APU verdict standing: {:?}",
        unknown.gpus
    );

    // Navi 10 is in neither table, so the scan reports it with no target and
    // the default `Some(false)`; a gfx1010 agent fills the target.
    let navi10 = host(
        "Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT]",
        "731f",
        "gfx1010",
    );
    assert_eq!(navi10.gpus[0].gfx_target, "gfx1010", "{:?}", navi10.gpus);
    assert!(
        navi10.has_discrete_amd && !navi10.has_apu,
        "{:?}",
        navi10.gpus
    );
}

/// `hipInfo` revises the display probe's verdict when it fills a target in an
/// APU family, and only then (the target rule, `apu_verdict_from_target`).
///
/// The first half is a host that exists: a Windows Mendocino laptop reports
/// "AMD Radeon(TM) Graphics" with a device id the table does not map, so the
/// display probe gives it no target and the default `Some(false)`, and only
/// `hipInfo`'s `gcnArchName` can say what it is.
///
/// The second half pins the fold's contract on a state the display probe
/// cannot produce today -- a GPU already called an APU, with no target -- since
/// the probe only reaches `Some(true)` by resolving one. Through the real
/// pipeline the "leave an unknown target's verdict alone" rule therefore
/// changes no output yet; it is asserted so the first new source of a
/// target-less verdict does not inherit the downgrade silently.
#[test]
fn a_hipinfo_target_revises_a_verdict_only_when_its_packaging_is_known() {
    let mut e = Examination::default();
    parse_gpus_windows_output(&windows_row("AMD Radeon(TM) Graphics", "1506"), &mut e);
    assert_eq!(e.gpus[0].gfx_target, "", "the premise: {:?}", e.gpus);
    assert_eq!(e.gpus[0].is_apu, Some(false), "the premise: {:?}", e.gpus);
    fill_unidentified_gpus_from_hipinfo(&mut e.gpus, "  gcnArchName: gfx1037\n");
    assert_eq!(e.gpus[0].gfx_target, "gfx1037", "{:?}", e.gpus);
    assert_eq!(
        e.gpus[0].is_apu,
        Some(true),
        "a known integrated target must revise the default: {:?}",
        e.gpus
    );

    let mut gpus = vec![Gpu {
        name: "AMD Radeon(TM) Graphics".to_owned(),
        is_amd: true,
        is_apu: Some(true),
        ..Gpu::default()
    }];
    fill_unidentified_gpus_from_hipinfo(&mut gpus, "  gcnArchName: gfx1170\n");
    assert_eq!(gpus[0].gfx_target, "gfx1170", "{gpus:?}");
    assert_eq!(
        gpus[0].is_apu,
        Some(true),
        "an unclassified target is not evidence of a discrete GPU: {gpus:?}"
    );
}

/// The never-downgrade rule run through the whole post-PCI sequence against a
/// planted KFD topology, rather than through `apply_rocminfo_gpu_agents` alone.
///
/// A Ryzen 6800H laptop: `lspci` names the iGPU "Rembrandt [Radeon 680M]", the
/// kernel exposes one KFD node for it at `0000:04:00.0`, and `rocminfo` at
/// that address reports a target nothing here classifies (as a newer ROCm
/// would for a newer part) with a `Memory Properties:` line without `APU`.
/// The report must not call the laptop's only GPU discrete.
#[test]
fn a_rembrandt_laptop_keeps_its_apu_verdict_through_every_probe() {
    let root = tempfile::tempdir().expect("scratch dir");
    let nodes = root.path().join("nodes");
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

    let mut e = Examination::default();
    apply_lspci_gpus(
        &mut e,
        "0000:04:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
         [AMD/ATI] Rembrandt [Radeon 680M] [1002:1681] (rev c8)",
    );
    probe_gpus_after_lspci(
        &mut e,
        GpuProbeSources {
            kfd_nodes: &nodes,
            rocminfo: Some(&rocminfo_output(&[(
                "gfx1170",
                "AMD Radeon Graphics",
                "0000:04:00.0",
                Some(""),
            )])),
            sysfs_gfx_target: || None,
        },
    );
    summarise_gpu_categories(&mut e);

    assert_eq!(e.gpus.len(), 1, "one GPU: {:?}", e.gpus);
    assert!(
        e.has_apu && !e.has_discrete_amd,
        "a Radeon 680M iGPU must stay an APU: {:?}",
        e.gpus
    );
}

/// What the iGPU+dGPU collision check sees, run through the real diagnosis
/// catalog.
///
/// A Ryzen 7000 desktop pairing the Raphael iGPU with an RX 7900 XTX is the
/// host `check_9_igpu_dgpu_collision` exists for, and it fires only when
/// `has_apu && has_discrete_amd`. The note must also name each card correctly,
/// since a diagnosis that confuses the two is no better than silence.
#[test]
fn the_igpu_dgpu_collision_check_fires_on_a_raphael_plus_rx7900_host() {
    let igpu = "0000:14:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
                [AMD/ATI] Raphael [1002:164e] (rev c1)";
    let dgpu = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. \
                [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c]";
    let mut e = Examination {
        os_family: "linux".to_owned(),
        ..Examination::default()
    };
    apply_lspci_gpus(&mut e, &format!("{dgpu}\n{igpu}\n"));
    apply_rocminfo_gpu_agents(
        &mut e,
        &rocminfo_output(&[
            (
                "gfx1100",
                "AMD Radeon RX 7900 XTX",
                "0000:03:00.0",
                Some(""),
            ),
            (
                "gfx1036",
                "AMD Radeon Graphics",
                "0000:14:00.0",
                Some("APU"),
            ),
        ]),
    );
    summarise_gpu_categories(&mut e);

    let report = crate::diagnose::diagnose(&e, "my training run segfaults");
    let Some(collision) = report
        .matched
        .iter()
        .find(|d| d.id == "fix-9-igpu-dgpu" && d.score > 0)
    else {
        panic!(
            "an iGPU+dGPU host must raise fix-9-igpu-dgpu; has_apu={}, \
             has_discrete_amd={}, gpus={:?}",
            e.has_apu, e.has_discrete_amd, e.gpus,
        );
    };
    let notes = collision
        .fix
        .as_ref()
        .map(|fix| fix.notes.join(" "))
        .unwrap_or_default();
    assert!(
        notes.contains("[\"gfx1100\"]") && notes.contains("[\"gfx1036\"]"),
        "the note must name gfx1100 as the discrete GPU and gfx1036 as the APU: {notes}"
    );
}

/// An AMD PCI row nothing can identify is reported as discrete, on purpose,
/// and both sides of that trade are pinned here -- see
/// `classify_amd_marketing_name`.
///
/// What it buys: a Ryzen 7000 desktop with an RX 5700 XT. Navi 10 is in
/// neither table; reported as "cannot say", it would clear `has_discrete_amd`
/// and the collision check would never fire on the host it is written for.
///
/// What it costs: a Strix Halo whose `lspci` row is the nameless `Device` of
/// #449 is reported with `has_discrete_amd` set. The collision check still
/// needs `has_apu`, which that host lacks, so no diagnosis fires; the cost is a
/// wrong field in `rocm examine --json` until `rocminfo` names the target.
#[test]
fn an_unidentifiable_pci_row_is_reported_as_discrete_and_this_is_what_that_costs() {
    let raphael = "0000:14:00.0 VGA compatible controller [0300]: Advanced Micro Devices, \
                   Inc. [AMD/ATI] Raphael [1002:164e] (rev c1)";
    let navi10 = "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, \
                  Inc. [AMD/ATI] Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT] \
                  [1002:731f] (rev c1)";
    let halo = "0000:66:00.0 Display controller [0380]: Advanced Micro Devices, \
                Inc. [AMD/ATI] Device [1002:1586] (rev c1)";
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

    let (laptop, fires) = host(&[halo]);
    assert_eq!(
        laptop.gpus[0].gfx_target, "",
        "the premise: the row is unresolved"
    );
    assert!(
        laptop.has_discrete_amd && !laptop.has_apu,
        "the known cost: an unresolved iGPU reads as discrete: {:?}",
        laptop.gpus
    );
    assert!(
        !fires,
        "and it stops there: no collision diagnosis on one GPU"
    );
}

// ---------------------------------------------------------------------------
// Generator reach
// ---------------------------------------------------------------------------

/// Measure how far the generators reach, by drawing from them directly and
/// tallying which regions each draw lands in.
///
/// A generator that never samples the interesting region passes every property
/// vacuously, so every counter printed here has a floor. The floors sit well
/// under the measured rates so a reseed does
/// not trip them, and well over zero so a generator that stopped reaching a
/// region does.
#[test]
fn generator_reach_is_measured_and_sufficient() {
    const DRAWS: u32 = 2048;
    let draws = u64::from(DRAWS);
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
        if !extract_lspci_name(&line).is_empty() {
            lspci_parsed += 1;
        }
        if device.3 {
            lspci_apu_device += 1;
        }
        let mut e = Examination::default();
        apply_lspci_gpus(&mut e, &line);
        if let Some(gpu) = e.gpus.first() {
            if gpu.is_apu == Some(true) {
                lspci_classified_apu += 1;
            }
            if !gpu.gfx_target.is_empty() {
                lspci_gfx_resolved += 1;
            }
        }
    }

    // The marketing-name strategies the properties draw from, measured as
    // drawn: `apu_marketing_name` for `examine_never_calls_a_known_apu_discrete`
    // and `marketing_name_spelled_differently` for the spelling property.
    let mut apu_name_resolved = 0u64;
    let mut apu_name_spelled = 0u64;
    for _ in 0..DRAWS {
        let (name, truth) = apu_marketing_name()
            .new_tree(&mut runner)
            .expect("APU marketing strategy")
            .current();
        if !classify_amd_marketing_name(&name).0.is_empty() {
            apu_name_resolved += 1;
        }
        if name != truth.0 {
            apu_name_spelled += 1;
        }
    }
    let mut spelled_resolved = 0u64;
    let mut spelled_apu_truth = 0u64;
    for _ in 0..DRAWS {
        let (name, truth) = marketing_name_spelled_differently()
            .new_tree(&mut runner)
            .expect("spelling strategy")
            .current();
        if !classify_amd_marketing_name(&name).0.is_empty() {
            spelled_resolved += 1;
        }
        if truth.2 {
            spelled_apu_truth += 1;
        }
    }

    // The never-downgrade property's own strategy: the draws that tell the
    // rule from a regression are an agent target other than the device's, and
    // a `Memory Properties:` line without `APU`.
    let mut agent_other_target = 0u64;
    let mut agent_discrete_reading = 0u64;
    for _ in 0..DRAWS {
        let (device, (target, memory)) = apu_host_with_a_rocminfo_agent()
            .new_tree(&mut runner)
            .expect("agent strategy")
            .current();
        if target != device.2 {
            agent_other_target += 1;
        }
        if memory == Some("") {
            agent_discrete_reading += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let pct = |n: u64| (n as f64) * 100.0 / f64::from(DRAWS);
    println!(
        "generator reach over {DRAWS} draws each:\n\
         lspci lines     : name parsed {:.1}%, GPU-class {:.1}%, APU silicon {:.1}%, \
         scanned as APU {:.1}%, gfx resolved {:.1}%\n\
         APU names       : resolved {:.1}%, respelled {:.1}%\n\
         respelled names : resolved {:.1}%, APU silicon {:.1}%\n\
         rocminfo agents : another target {:.1}%, Memory Properties without APU {:.1}%",
        pct(lspci_parsed),
        pct(lspci_gpu_class),
        pct(lspci_apu_device),
        pct(lspci_classified_apu),
        pct(lspci_gfx_resolved),
        pct(apu_name_resolved),
        pct(apu_name_spelled),
        pct(spelled_resolved),
        pct(spelled_apu_truth),
        pct(agent_other_target),
        pct(agent_discrete_reading),
    );
    let floors = [
        ("lspci: name parsed", lspci_parsed, draws / 2),
        // A third, not a half: one class in five is the PCI bridge the probe
        // must not enumerate (`lspci_name_extraction_round_trips` asserts that
        // it is not), and the case and tab mutations break the case-sensitive
        // class match.
        ("lspci: enumerated GPU class", lspci_gpu_class, draws / 3),
        ("lspci: APU silicon", lspci_apu_device, draws / 4),
        ("lspci: scanned as an APU", lspci_classified_apu, draws / 6),
        ("lspci: gfx target resolved", lspci_gfx_resolved, draws / 4),
        ("APU names: resolved", apu_name_resolved, draws / 2),
        ("APU names: respelled", apu_name_spelled, draws / 4),
        ("respelled names: resolved", spelled_resolved, draws / 2),
        ("respelled names: APU silicon", spelled_apu_truth, draws / 4),
        (
            "rocminfo: another agent target",
            agent_other_target,
            draws / 4,
        ),
        (
            "rocminfo: Memory Properties without APU",
            agent_discrete_reading,
            draws / 6,
        ),
    ];
    for (region, reached, floor) in floors {
        assert!(
            reached > floor,
            "{region}: {reached} of {DRAWS} draws, under the floor of {floor}"
        );
    }
}
