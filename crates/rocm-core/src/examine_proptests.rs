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
//!
//! What is here is the harness and the invariants the classifiers already
//! satisfy: totality, determinism, round-tripping an `lspci` line, and
//! independence from how a name is spelled. The properties that assert
//! *correct classification* arrive with the fixes that make them pass, so
//! that no commit in this history is red.

use super::{
    classify_amd_marketing_name, extract_lspci_name, gfx_is_apu_family, is_lspci_gpu_line,
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

    /// The Windows display probe and `examine`'s own Windows GPU enumeration
    /// read the same row and must not disagree about the gfx target.
    #[test]
    fn windows_probes_agree_on_gfx_target(
        entry in proptest::sample::select(AMD_MARKETING_NAMES),
        subsys in 0u32..0x1_0000,
    ) {
        // Only coherent rows: the PNP id must name the same part the marketing
        // name does, or the two probes are being asked about different GPUs.
        let Some(device_id) = entry_device_id(entry.0) else {
            return Ok(());
        };
        // A Win32_VideoController row as `probe_gpus_windows` reads it: name,
        // driver version, PNP device id.
        let pnp = format!("PCI\\VEN_1002&DEV_{device_id}&SUBSYS_{subsys:04x}1002&REV_C1");
        let row = format!("{}\t32.0.1\t{pnp}", entry.0);
        let install_target = crate::parse_windows_display_gfx_target(&row);
        let examine_target = classify_amd_marketing_name(entry.0).0;
        if let Some(install_target) = install_target
            && !examine_target.is_empty()
        {
            prop_assert_eq!(
                &examine_target,
                &install_target,
                "the Windows display probe and examine disagree about {}",
                entry.0,
            );
        }
    }

}

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
