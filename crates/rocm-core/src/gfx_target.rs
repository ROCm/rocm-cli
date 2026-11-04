// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! gfx-target identification and TheRock package-family tables.
//!
//! Pure lookups with no probing of their own: pulling a gfx token out of text,
//! mapping a gfx target to the TheRock family that ships it (and naming the
//! target that has none), the AMD PCI-device-id and marketing-name tables, and
//! decoding the kernel's gfx-target version. [`crate::host_gpu`] feeds these
//! the strings its probes read, and [`crate::managed_runtime`] reads the family
//! mapping from here.

pub fn extract_first_gfx_token(text: &str) -> Option<String> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .find_map(|token| {
            let normalized = token.to_ascii_lowercase();
            if normalized.starts_with("gfx") {
                Some(normalized)
            } else {
                None
            }
        })
}

/// Whether `gfx` is a GPU target that is known to have no ROCm build.
///
/// This is the case when the CLI recognizes the target but no TheRock package
/// family covers it, so `normalize_therock_family` returns `None` for it on
/// purpose rather than because the target is unknown.
///
/// Today that is gfx90c, the Vega iGPU in Renoir / Lucienne / Cezanne / Barcelo
/// APUs. Callers use it to say "detected, but not supported by ROCm" instead of
/// treating the GPU as undetected or suggesting a package family for it.
///
/// The `normalize_therock_family` half of that sentence is a real conjunct, not
/// a restatement: if the gfx90c arm there were ever removed and the target fell
/// back into the loose `gfx90` catch-all, this would answer `false` and every
/// caller would go back to the old data-center behaviour together. Naming the
/// target on its own would instead leave the two halves free to disagree — the
/// install path would keep refusing gfx90c through this predicate while
/// `normalize_therock_family` quietly resolved `gfx90X-dcgpu` for it, and the
/// refusal would be certifying a mapping that no longer existed. Tests that
/// drive the refusal are only evidence about the mapping because of this.
pub fn gfx_target_has_no_rocm_build(gfx: &str) -> bool {
    extract_first_gfx_token(gfx).is_some_and(|target| target.starts_with("gfx90c"))
        && normalize_therock_family(gfx).is_none()
}

pub fn normalize_therock_family(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let target = extract_first_gfx_token(&normalized).unwrap_or(normalized);
    match target.as_str() {
        "gfx101x-dgpu" => Some("gfx101X-dgpu".to_owned()),
        "gfx103x-dgpu" => Some("gfx103X-dgpu".to_owned()),
        "gfx110x-all" => Some("gfx110X-all".to_owned()),
        "gfx120x-all" => Some("gfx120X-all".to_owned()),
        "gfx90x-dgpu" => Some("gfx90X-dgpu".to_owned()),
        "gfx94x-dcgpu" => Some("gfx94X-dcgpu".to_owned()),
        "gfx950-dcgpu" => Some("gfx950-dcgpu".to_owned()),
        value if value.starts_with("gfx101") => Some("gfx101X-dgpu".to_owned()),
        value if value.starts_with("gfx103") => Some("gfx103X-dgpu".to_owned()),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" => Some("gfx110X-all".to_owned()),
        value if value.starts_with("gfx1150") => Some("gfx1150".to_owned()),
        value if value.starts_with("gfx1151") => Some("gfx1151".to_owned()),
        value if value.starts_with("gfx1152") => Some("gfx1152".to_owned()),
        value if value.starts_with("gfx1153") => Some("gfx1153".to_owned()),
        "gfx1200" | "gfx1201" => Some("gfx120X-all".to_owned()),
        value if value.starts_with("gfx125") => Some("gfx125X-dcgpu".to_owned()),
        value if value.starts_with("gfx900") => Some("gfx900".to_owned()),
        value if value.starts_with("gfx906") => Some("gfx906".to_owned()),
        value if value.starts_with("gfx908") => Some("gfx908".to_owned()),
        value if value.starts_with("gfx90a") => Some("gfx90a".to_owned()),
        value if value.starts_with("gfx950") => Some("gfx950-dcgpu".to_owned()),
        value
            if value.starts_with("gfx942")
                || value.starts_with("gfx94")
                || value.starts_with("gfx9-4") =>
        {
            Some("gfx94X-dcgpu".to_owned())
        }
        // gfx90c is the Vega iGPU in Renoir / Lucienne / Cezanne / Barcelo
        // APUs. TheRock publishes no build for it, so it must not fall into the
        // data-center `gfx90X-dcgpu` catch-all below.
        value if value.starts_with("gfx90c") => None,
        value if value.starts_with("gfx90") => Some("gfx90X-dcgpu".to_owned()),
        _ => None,
    }
}

/// The TheRock package families the CLI recognizes.
///
/// This is the full set of values [`normalize_therock_family`] can produce. Used
/// to tell the user which `--family` values are valid when GPU auto-detection
/// cannot resolve an installable runtime. Whether a given family currently has
/// published wheels depends on the channel and release; recognition here does
/// not guarantee availability.
///
/// Kept in sync with [`normalize_therock_family`] by
/// `known_therock_families_all_round_trip` — every entry must normalize back to
/// itself.
pub const fn known_therock_families() -> &'static [&'static str] {
    &[
        "gfx90X-dgpu",
        "gfx90X-dcgpu",
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx94X-dcgpu",
        "gfx950-dcgpu",
        "gfx101X-dgpu",
        "gfx103X-dgpu",
        "gfx110X-all",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx120X-all",
        "gfx125X-dcgpu",
    ]
}

pub(crate) fn parse_windows_display_gfx_target(text: &str) -> Option<String> {
    let mut name_fallback = None;
    let mut unsupported_match = None;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let (name, pnp_id) = line.split_once('\t').unwrap_or((line, ""));
        if let Some(device_id) = amd_pci_device_id_from_pnp_id(pnp_id)
            && let Some(target) = gfx_target_from_amd_pci_device_id(&device_id)
        {
            if gfx_target_has_no_rocm_build(target) {
                // A Vega iGPU listed ahead of a discrete card must not hide it:
                // keep looking, and fall back to this target only if nothing
                // else on the host is recognized.
                unsupported_match = unsupported_match.or(Some(target));
                continue;
            }
            return Some(target.to_owned());
        }
        if name_fallback.is_none() {
            name_fallback = gfx_target_from_amd_marketing_name(name).map(str::to_owned);
        }
    }
    name_fallback.or_else(|| unsupported_match.map(str::to_owned))
}

pub(crate) fn parse_windows_display_name(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(|line| {
            let (name, _) = line.split_once('\t').unwrap_or((line, ""));
            let name = name.trim();
            (!name.is_empty()).then(|| name.to_owned())
        })
}

/// The gfx target for a Windows PNP device id such as
/// `PCI\VEN_1002&DEV_1636&SUBSYS_...`, or `None` when the id is not an AMD
/// device or its PCI device id is not in the table.
///
/// The pair of private helpers this wraps is what `host_gpu`'s own Windows
/// display probe already uses. It is public so `examine`'s Windows GPU probe
/// can answer from the same table rather than from marketing-name matching
/// alone: `Win32_VideoController` hands that probe the PNP id in the same
/// query that gives it the name, and without this it dropped the id on the
/// floor and left `gfx_target` empty for every part the name table does not
/// cover -- which is every Vega iGPU, so a Renoir/Cezanne laptop reported no
/// architecture at all unless a HIP SDK was installed to ask `hipInfo`.
///
/// Returns the first match only. Which GPU of several a caller is asking about
/// is the caller's question; this answers for one device id.
#[must_use]
pub fn gfx_target_from_pnp_device_id(pnp_id: &str) -> Option<&'static str> {
    amd_pci_device_id_from_pnp_id(pnp_id)
        .as_deref()
        .and_then(gfx_target_from_amd_pci_device_id)
}

pub(crate) fn amd_pci_device_id_from_pnp_id(pnp_id: &str) -> Option<String> {
    let upper = pnp_id.to_ascii_uppercase();
    if !upper.contains("VEN_1002") {
        return None;
    }
    let start = upper.find("DEV_")? + "DEV_".len();
    let device_id = upper[start..]
        .chars()
        .take_while(char::is_ascii_hexdigit)
        .take(4)
        .collect::<String>();
    if device_id.len() == 4 {
        Some(device_id.to_ascii_lowercase())
    } else {
        None
    }
}

pub(crate) fn gfx_target_from_amd_pci_device_id(device_id: &str) -> Option<&'static str> {
    match device_id.to_ascii_lowercase().as_str() {
        // Navi 21 / 22 / 23 / 24: Radeon RX 6000 desktop and mobile ASICs.
        "73a0" | "73a1" | "73a2" | "73a3" | "73a5" | "73a8" | "73a9" | "73ab" | "73ac" | "73ad"
        | "73ae" | "73af" => Some("gfx1030"),
        "73c0" | "73c1" | "73c3" => Some("gfx1031"),
        "73e0" | "73e1" | "73e2" | "73e3" | "73e8" | "73e9" | "73ea" | "73eb" | "73ec" | "73ed"
        | "73ef" => Some("gfx1032"),
        "7420" | "7421" | "7422" | "7423" | "7424" | "743f" => Some("gfx1034"),
        // Vega-based APUs (Renoir / Lucienne / Cezanne / Barcelo): all gfx90c.
        // Radeon Vega 6/7/8 iGPUs in Ryzen 4000/5000/7020 mobile and desktop parts.
        //
        // All four are the `/* Renoir */` block of the upstream kernel's
        // `drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c`, where they share
        // `CHIP_RENOIR | AMD_IS_APU`. coreboot's `pci_ids.h` and OpenBSD's
        // `amdgpu_devlist.h` carry the identical four. Per ID:
        //   1636  Renoir    (CPU family 17h model 60h)
        //   164c  Lucienne  (CPU family 17h model 68h)
        //   1638  Cezanne   (CPU family 19h model 51h)
        //   15e7  Barcelo   (CPU family 19h model 51h)
        // Note 164c, not 164d: 164d is Yellow Carp / Rembrandt, a different
        // ASIC, and is mapped to gfx1035 below.
        "1636" | "1638" | "164c" | "15e7" => Some("gfx90c"),
        // RDNA2 APUs.
        "163f" => Some("gfx1033"),
        "164d" | "1681" => Some("gfx1035"),
        "164e" => Some("gfx1036"),
        // RDNA3 APUs.
        "15bf" | "164f" | "1900" | "1901" => Some("gfx1103"),
        // RDNA3.5 APUs with public PCI IDs that map cleanly to one gfx target.
        "1114" => Some("gfx1152"),
        // Navi 48: Radeon RX 9070 / 9070 XT / 9070 GRE.
        "7550" => Some("gfx1201"),
        _ => None,
    }
}

pub(crate) fn gfx_target_from_amd_marketing_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let normalized = normalize_marketing_name_for_match(&lower);
    for entry in AMD_MARKETING_GFX_TARGETS {
        if marketing_name_contains(&normalized, entry.pattern) {
            return Some(entry.gfx_target);
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct AmdMarketingGfxTarget {
    pattern: &'static str,
    gfx_target: &'static str,
}

const AMD_MARKETING_GFX_TARGETS: &[AmdMarketingGfxTarget] = &[
    // RDNA4 discrete.
    AmdMarketingGfxTarget {
        pattern: "ai pro r9700",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "ai pro r9600",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 9070",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 9060",
        gfx_target: "gfx1200",
    },
    // RDNA3 discrete.
    AmdMarketingGfxTarget {
        pattern: "pro w7900",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w7800",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w7700",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7900",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7800",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7700",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7600",
        gfx_target: "gfx1102",
    },
    // RDNA2 discrete. Mobile names that share number prefixes are listed before desktop.
    AmdMarketingGfxTarget {
        pattern: "pro w6800",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w6600",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "pro v620",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6850m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6800m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700s",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6650m",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600m",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600s",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6500m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6400m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6300m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6950",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6900",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6800",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6750",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6650",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6500",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6400",
        gfx_target: "gfx1034",
    },
    // RDNA3.5 / Strix Halo APUs.
    AmdMarketingGfxTarget {
        pattern: "8060s",
        gfx_target: "gfx1151",
    },
    AmdMarketingGfxTarget {
        pattern: "8050s",
        gfx_target: "gfx1151",
    },
    AmdMarketingGfxTarget {
        pattern: "8040s",
        gfx_target: "gfx1151",
    },
    // RDNA3.5 APUs.
    AmdMarketingGfxTarget {
        pattern: "890m",
        gfx_target: "gfx1150",
    },
    AmdMarketingGfxTarget {
        pattern: "880m",
        gfx_target: "gfx1150",
    },
    AmdMarketingGfxTarget {
        pattern: "860m",
        gfx_target: "gfx1152",
    },
    AmdMarketingGfxTarget {
        pattern: "840m",
        gfx_target: "gfx1152",
    },
    AmdMarketingGfxTarget {
        pattern: "820m",
        gfx_target: "gfx1153",
    },
    // RDNA3 APUs.
    AmdMarketingGfxTarget {
        pattern: "780m",
        gfx_target: "gfx1103",
    },
    AmdMarketingGfxTarget {
        pattern: "760m",
        gfx_target: "gfx1103",
    },
    AmdMarketingGfxTarget {
        pattern: "740m",
        gfx_target: "gfx1103",
    },
    // RDNA2 APUs.
    AmdMarketingGfxTarget {
        pattern: "680m",
        gfx_target: "gfx1035",
    },
    AmdMarketingGfxTarget {
        pattern: "660m",
        gfx_target: "gfx1035",
    },
    AmdMarketingGfxTarget {
        pattern: "610m",
        gfx_target: "gfx1036",
    },
    AmdMarketingGfxTarget {
        pattern: "steam deck",
        gfx_target: "gfx1033",
    },
    AmdMarketingGfxTarget {
        pattern: "van gogh",
        gfx_target: "gfx1033",
    },
];

fn normalize_marketing_name_for_match(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | '0'..='9' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn marketing_name_contains(normalized_name: &str, pattern: &str) -> bool {
    normalized_name
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(pattern.split_whitespace().count())
        .any(|window| window.join(" ") == pattern)
}

#[cfg(any(target_os = "linux", test))]
pub(crate) fn parse_linux_kfd_gfx_target(value: &str) -> Option<String> {
    if let Some(token) = extract_first_gfx_token(value) {
        return Some(token);
    }
    let digits = value.trim();
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    match digits.len() {
        3 | 4 => Some(format!("gfx{digits}")),
        // A 5/6-digit value is KFD's *packed* version (major·10000 + minor·100 +
        // step), not a target name, so there is no `gfx{digits}` fallback here.
        // Fabricating one from a version that failed to decode fed
        // `gfx90010`-shaped tokens into `normalize_therock_family`, where the
        // loose `gfx90` arm mapped them to a plausible-looking but wrong family.
        // Yielding `None` instead lets detection try the next KFD node and then
        // `ip_discovery`, and otherwise report the target as unknown — which is
        // recoverable with `--family`, where a wrong family silently installs
        // the wrong runtime wheel.
        5 | 6 => {
            let raw: u32 = digits.parse().ok()?;
            let major = raw / 10_000;
            let minor = (raw / 100) % 100;
            let revision = raw % 100;
            gfx_target_from_gc_version(major, minor, revision)
        }
        _ => None,
    }
}

#[cfg(any(target_os = "linux", test))]
/// Render gfx target *components* as an LLVM target token.
///
/// The token is `gfx` + major in decimal + minor and revision as **single hex
/// digits** — the `a` in `gfx90a` is revision `10`. Concatenating the components
/// as decimal agrees with hex only while every component is below 10, so it
/// silently produced `gfx9010` for gfx90a hardware (MI210/MI250), which then
/// normalized to the wrong TheRock family.
///
/// **The caller owns whether its numbers are target components at all.** KFD's
/// `gfx_target_version` packs exactly this triple, so decoding it and calling
/// here is sound. A GC (Graphics Core) IP version is a *different* quantity that
/// merely coincides with the target on many parts: it does not on the GC 9.4.x
/// line, where GC 9.4.0/9.4.1/9.4.2/9.4.3 are gfx906/gfx908/gfx90a/gfx942. So
/// [`detect_ip_discovery_gc_target`] can still yield a wrong-but-plausible token
/// for those parts — pre-existing, unchanged by the hex encoding, and not
/// something this function can detect, since the components it receives are
/// well-formed either way.
///
/// A minor or revision that cannot be a single hex digit does not describe any
/// gfx target, so it yields `None` rather than a fabricated token: the caller
/// tries the next detection source and otherwise reports the target as unknown,
/// which is recoverable with `--family`, where a fabricated one silently
/// installs the wrong runtime wheel. `major` is only checked for zero — it is
/// printed in decimal and has no single-digit bound (`gfx1030`, `gfx1250`), so
/// an implausible major still concatenates into a token.
pub(crate) fn gfx_target_from_gc_version(major: u32, minor: u32, revision: u32) -> Option<String> {
    if major == 0 || minor > 0xf || revision > 0xf {
        return None;
    }
    Some(format!("gfx{major}{minor:x}{revision:x}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1201_to_gfx120x_all() {
        assert_eq!(
            normalize_therock_family("gfx1201"),
            Some("gfx120X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_accepts_canonical_family_labels() {
        assert_eq!(
            normalize_therock_family("gfx120X-all"),
            Some("gfx120X-all".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx110X-all"),
            Some("gfx110X-all".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx94X-dcgpu"),
            Some("gfx94X-dcgpu".to_owned())
        );
    }

    /// ROCm 10's next layout publishes a `gfx125X-dcgpu` family; the raw
    /// `gfx1250` arch a host reports has to land on it.
    #[test]
    fn normalize_therock_family_maps_gfx1250_to_gfx125x_dcgpu() {
        assert_eq!(
            normalize_therock_family("gfx1250"),
            Some("gfx125X-dcgpu".to_owned())
        );
    }

    /// gfx90c (Renoir / Lucienne / Cezanne / Barcelo Vega iGPUs) has no TheRock
    /// build. It used to fall into the loose `gfx90` arm and come back as the
    /// data-center family `gfx90X-dcgpu`, which `rocm install sdk` then offered.
    #[test]
    fn normalize_therock_family_has_no_family_for_gfx90c() {
        assert_eq!(normalize_therock_family("gfx90c"), None);
        assert_eq!(normalize_therock_family("GFX90C"), None);
        // A feature-suffixed spelling, as KFD / rocminfo can report it.
        assert_eq!(normalize_therock_family("gfx90c:xnack-"), None);
        // The neighbouring Vega parts keep their families.
        assert_eq!(
            normalize_therock_family("gfx900"),
            Some("gfx900".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx906"),
            Some("gfx906".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx90a"),
            Some("gfx90a".to_owned())
        );
    }

    #[test]
    fn gfx_target_has_no_rocm_build_names_only_gfx90c() {
        assert!(gfx_target_has_no_rocm_build("gfx90c"));
        assert!(gfx_target_has_no_rocm_build("GFX90C"));
        assert!(gfx_target_has_no_rocm_build("gfx90c:xnack-"));
        assert!(!gfx_target_has_no_rocm_build("gfx90a"));
        assert!(!gfx_target_has_no_rocm_build("gfx906"));
        assert!(!gfx_target_has_no_rocm_build("gfx1103"));
        assert!(!gfx_target_has_no_rocm_build(""));
    }

    /// The predicate must agree with `normalize_therock_family`, not merely
    /// happen to.
    ///
    /// "Has no ROCm build" and "normalizes to no family" are two statements
    /// about the same fact, written in two places. Today they cannot drift,
    /// because the predicate is conjoined with the mapping rather than carrying
    /// its own copy of the target name. Were that conjunct dropped, they could:
    /// the install path would refuse a target the resolver happily maps, and —
    /// worse for the test suite — every test that drives the refusal through
    /// `--family gfx90c` would keep passing after the gfx90c arm in
    /// `normalize_therock_family` was deleted, because the refusal would
    /// short-circuit before the mapping was ever consulted. That is the
    /// counterfactual, not a description of the current code; drop both and
    /// `runtime-19` goes green through the regression it exists to catch, which
    /// is the one mutant this test fires against. Measured — see the mutant
    /// list under "GPU targets with no ROCm build" in `docs/testing.md`.
    ///
    /// Note this is an implication, so it holds vacuously when the predicate
    /// answers `false`: it pins the two together but is not on its own the
    /// evidence that gfx90c has no family. `normalize_therock_family_has_no_family_for_gfx90c`
    /// is.
    #[test]
    fn no_rocm_build_implies_no_therock_family() {
        for target in [
            "gfx90c",
            "gfx90c:xnack-",
            "gfx900",
            "gfx906",
            "gfx908",
            "gfx90a",
            "gfx1030",
            "gfx1103",
            "gfx1201",
        ] {
            if gfx_target_has_no_rocm_build(target) {
                assert_eq!(
                    normalize_therock_family(target),
                    None,
                    "`{target}` is reported as having no ROCm build, so it must \
                     not normalize to a package family"
                );
            }
        }
    }

    #[test]
    fn windows_display_parser_prefers_a_supported_gpu_over_a_vega_igpu() {
        // Ryzen 4000/5000 laptop with a discrete card: the iGPU is listed first
        // and has a mapped PCI ID, but it must not hide the discrete GPU.
        let text = "AMD Radeon(TM) Graphics\tPCI\\VEN_1002&DEV_1636\n\
                    AMD Radeon RX 6600\tPCI\\VEN_1002&DEV_73E0";
        assert_eq!(
            parse_windows_display_gfx_target(text),
            Some("gfx1032".to_owned())
        );
        // The name fallback also outranks the unsupported iGPU.
        let text = "AMD Radeon(TM) Graphics\tPCI\\VEN_1002&DEV_1636\n\
                    AMD Radeon RX 6800M\tPCI\\VEN_1002&DEV_9999";
        assert_eq!(
            parse_windows_display_gfx_target(text),
            Some("gfx1031".to_owned())
        );
        // Alone, the iGPU is still reported as gfx90c.
        let text = "AMD Radeon(TM) Graphics\tPCI\\VEN_1002&DEV_1636";
        assert_eq!(
            parse_windows_display_gfx_target(text),
            Some("gfx90c".to_owned())
        );
    }

    #[test]
    fn known_therock_families_all_round_trip() {
        for family in known_therock_families() {
            assert_eq!(
                normalize_therock_family(family).as_deref(),
                Some(*family),
                "known family `{family}` must normalize back to itself"
            );
        }
    }

    #[test]
    fn known_therock_families_is_not_empty() {
        assert!(!known_therock_families().is_empty());
    }

    #[test]
    fn windows_display_parser_maps_rx_9070_xt_device_id_to_gfx1201() {
        let text = "ASPEED Graphics Family(WDDM)\tPCI\\VEN_1A03&DEV_2000\nAMD Radeon RX 9070 XT\tPCI\\VEN_1002&DEV_7550&SUBSYS_2435148C&REV_C0";
        assert_eq!(
            parse_windows_display_gfx_target(text),
            Some("gfx1201".to_owned())
        );
    }

    #[test]
    fn windows_display_parser_maps_known_amd_pci_ids() {
        for (device_id, expected) in [
            ("73A0", "gfx1030"),
            ("73C0", "gfx1031"),
            ("73E0", "gfx1032"),
            ("163F", "gfx1033"),
            ("743F", "gfx1034"),
            ("1681", "gfx1035"),
            ("164E", "gfx1036"),
            ("15BF", "gfx1103"),
            ("164F", "gfx1103"),
            ("1900", "gfx1103"),
            ("1114", "gfx1152"),
            // Vega APUs: Renoir, Cezanne, Lucienne, Barcelo.
            ("1636", "gfx90c"),
            ("1638", "gfx90c"),
            ("164C", "gfx90c"),
            ("15E7", "gfx90c"),
        ] {
            assert_eq!(
                parse_windows_display_gfx_target(&format!(
                    "AMD Display Adapter\tPCI\\VEN_1002&DEV_{device_id}"
                )),
                Some(expected.to_owned()),
                "{device_id}"
            );
        }
    }

    #[test]
    fn windows_display_parser_falls_back_to_name_when_pci_id_is_uncertain() {
        assert_eq!(
            parse_windows_display_gfx_target("AMD Radeon 820M\tPCI\\VEN_1002&DEV_1902"),
            Some("gfx1153".to_owned())
        );
    }

    #[test]
    fn windows_display_name_parser_uses_first_nonempty_adapter_name() {
        assert_eq!(
            parse_windows_display_name("\nAMD Radeon RX 9070 XT\tPCI\\VEN_1002&DEV_7550\n"),
            Some("AMD Radeon RX 9070 XT".to_owned())
        );
    }

    #[test]
    fn windows_display_parser_maps_known_marketing_names() {
        for (name, expected) in [
            ("AMD Radeon RX 9070 XT\t", "gfx1201"),
            ("AMD Radeon RX 9060 XT\t", "gfx1200"),
            ("AMD Radeon RX 7900 XTX\t", "gfx1100"),
            ("AMD Radeon RX 7800 XT\t", "gfx1101"),
            ("AMD Radeon RX 7600\t", "gfx1102"),
            ("AMD Radeon RX 6800 XT\t", "gfx1030"),
            ("AMD Radeon RX 6800M\t", "gfx1031"),
            ("AMD Radeon RX 6700 XT\t", "gfx1031"),
            ("AMD Radeon RX 6600\t", "gfx1032"),
            ("AMD Radeon RX 6500 XT\t", "gfx1034"),
            ("AMD Radeon 680M\t", "gfx1035"),
            ("AMD Radeon 660M\t", "gfx1035"),
            ("AMD Radeon 610M\t", "gfx1036"),
            ("AMD Radeon 780M\t", "gfx1103"),
            ("AMD Radeon 760M\t", "gfx1103"),
            ("AMD Radeon 740M\t", "gfx1103"),
            ("AMD Radeon 8060S\t", "gfx1151"),
            ("AMD Radeon 890M\t", "gfx1150"),
            ("AMD Radeon 860M\t", "gfx1152"),
            ("AMD Radeon 820M\t", "gfx1153"),
            ("Steam Deck\t", "gfx1033"),
        ] {
            assert_eq!(
                parse_windows_display_gfx_target(name),
                Some(expected.to_owned()),
                "{name}"
            );
        }
    }

    /// The public wrapper `examine`'s Windows GPU probe reads, over the four
    /// Vega iGPU device ids that matter to `info-20-gpu-no-rocm-support`.
    ///
    /// This is the only source of a gfx target for those parts on a Windows
    /// host with no HIP SDK: `classify_amd_marketing_name` has no Vega arm, and
    /// `hipInfo` is in the SDK that is not installed. Without it the ticket's
    /// 4700U laptop reported an empty target, and the check that exists for it
    /// could not fire on the machine it was written for.
    #[test]
    fn the_pnp_wrapper_answers_for_the_vega_igpus_with_no_rocm_build() {
        for (pnp, expected) in [
            ("PCI\\VEN_1002&DEV_1636&SUBSYS_08A71028&REV_C1", "gfx90c"),
            ("PCI\\VEN_1002&DEV_164C&SUBSYS_380D103C&REV_C1", "gfx90c"),
            ("PCI\\VEN_1002&DEV_1638&SUBSYS_0A2E1025&REV_C1", "gfx90c"),
            ("PCI\\VEN_1002&DEV_15E7&SUBSYS_8A431043&REV_C1", "gfx90c"),
            // A supported part through the same door, so this is not read as
            // "the wrapper returns gfx90c".
            ("PCI\\VEN_1002&DEV_7550&SUBSYS_2435148C", "gfx1201"),
        ] {
            assert_eq!(gfx_target_from_pnp_device_id(pnp), Some(expected), "{pnp}");
            assert_eq!(
                gfx_target_has_no_rocm_build(expected),
                expected == "gfx90c",
                "{pnp}: and the unsupported predicate has to agree about it"
            );
        }
        assert_eq!(
            gfx_target_from_pnp_device_id("PCI\\VEN_10DE&DEV_2504"),
            None,
            "an NVIDIA device id is not an AMD gfx target"
        );
        assert_eq!(
            gfx_target_from_pnp_device_id("PCI\\VEN_1002&DEV_FFFF"),
            None,
            "an AMD device id the table does not carry is unknown, not guessed"
        );
    }

    #[test]
    fn amd_pci_device_id_parser_requires_amd_vendor() {
        assert_eq!(
            amd_pci_device_id_from_pnp_id("PCI\\VEN_1002&DEV_7550&SUBSYS_2435148C"),
            Some("7550".to_owned())
        );
        assert_eq!(
            amd_pci_device_id_from_pnp_id("PCI\\VEN_1A03&DEV_2000"),
            None
        );
    }

    #[test]
    fn gc_version_converts_to_gfx_target() {
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 1),
            Some("gfx1101".to_owned())
        );
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 3),
            Some("gfx1103".to_owned())
        );
        // Two digits of major stay decimal.
        assert_eq!(
            gfx_target_from_gc_version(12, 5, 0),
            Some("gfx1250".to_owned())
        );
    }

    #[test]
    fn gc_version_encodes_components_above_nine_as_hex_digits() {
        // GC 9.0.10 is gfx90a (MI210/MI250), not "gfx9010": minor and revision
        // are single hex digits. Decimal concatenation agreed with hex only
        // while every component stayed below 10.
        assert_eq!(
            gfx_target_from_gc_version(9, 0, 10),
            Some("gfx90a".to_owned())
        );
        assert_eq!(
            gfx_target_from_gc_version(9, 4, 12),
            Some("gfx94c".to_owned())
        );

        // A component that is not a single hex digit describes no gfx target,
        // so detection falls through rather than acting on a fabricated one.
        assert_eq!(gfx_target_from_gc_version(12, 16, 0), None);
        assert_eq!(gfx_target_from_gc_version(12, 0, 16), None);
        assert_eq!(gfx_target_from_gc_version(0, 0, 1), None);
    }

    #[test]
    fn gfx90a_gc_version_normalizes_to_its_own_therock_family() {
        // The point of the fix, end to end: "gfx9010" missed every specific arm
        // of `normalize_therock_family` and fell through to the loose `gfx90`
        // one, yielding "gfx90X-dcgpu" — the wrong runtime wheel, and outside
        // `VLLM_PREFERRED_THEROCK_FAMILIES`, so engine selection changed too.
        let target = gfx_target_from_gc_version(9, 0, 10).expect("gfx90a target");
        assert_eq!(normalize_therock_family(&target), Some("gfx90a".to_owned()));
    }

    #[test]
    fn linux_kfd_gfx_target_parser_accepts_numeric_and_direct_tokens() {
        assert_eq!(
            parse_linux_kfd_gfx_target("110003"),
            Some("gfx1103".to_owned())
        );
        assert_eq!(
            parse_linux_kfd_gfx_target("120001"),
            Some("gfx1201".to_owned())
        );
        assert_eq!(
            parse_linux_kfd_gfx_target("gfx1103"),
            Some("gfx1103".to_owned())
        );
        // KFD packs gfx90a as 9·10000 + 0·100 + 10.
        assert_eq!(
            parse_linux_kfd_gfx_target("90010"),
            Some("gfx90a".to_owned())
        );
        // A packed version whose components are not single hex digits is not a
        // target; it must not be passed through as `gfx{digits}`, which used to
        // normalize to a plausible-looking but wrong family.
        assert_eq!(parse_linux_kfd_gfx_target("121600"), None);
        assert_eq!(parse_linux_kfd_gfx_target("not-a-target"), None);
    }
}
