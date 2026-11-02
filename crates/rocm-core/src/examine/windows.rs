// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The native Windows probes: the GPU query, the HIP SDK, the Adrenalin driver and
//! the MSVC redistributable.

use super::linux_gpus::{classify_amd_marketing_name, gfx_is_apu_family};
use super::{Examination, Gpu};
use crate::probe_exec::{MEDIUM, run};
use std::path::Path;
use std::time::Duration;

const WIN_GPU_SCRIPT: &str = "Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -match 'VEN_1002' -or $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' -or $_.PNPDeviceID -match 'VEN_10DE' -or $_.Name -match 'NVIDIA' } | ForEach-Object { \"$($_.Name)`t$($_.DriverVersion)`t$($_.PNPDeviceID)\" }";

pub(super) fn probe_gpus_windows(e: &mut Examination) {
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
        // same row as the name, and `gfx_target`'s PCI-id table answers for parts
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

pub(super) fn probe_hip_sdk_windows(e: &mut Examination) {
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
pub(super) fn probe_adrenalin_windows(e: &mut Examination) {
    e.set_windows_driver(crate::host_gpu::detect_windows_amd_driver_info().as_ref());
}

pub(super) fn probe_msvc_redist_windows(e: &mut Examination) {
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
    use crate::examine::linux_gpus::summarise_gpu_categories;

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
}
