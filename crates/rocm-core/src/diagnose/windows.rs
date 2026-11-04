// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The checks that apply on Windows only: the HIP SDK, the Adrenalin driver, the
//! Visual C++ runtime and the display driver date.

use super::scoring::{
    KEYWORDS_HIP_SDK_MISSING, KEYWORDS_MSVC_REDIST, KEYWORDS_MSVC_REDIST_OUTDATED, finalize,
    keyword_score, keyword_score_collapsing_overlaps, symptom_matches, zero,
};
use super::{Diagnosis, Fix};
use crate::examine::Examination;

/// The oldest Visual C++ runtime that programs built with a current toolset can
/// run on, as `major.minor`.
///
/// Evidence: Visual Studio 2022 17.10 made the standard library's `mutex`
/// constructor `constexpr`. A program built that way and run on an older
/// `msvcp140.dll` reads a null pointer in the mutex code and dies with an access
/// violation (0xC0000005). Microsoft's STL issue tracker has the report: 14.32
/// crashed and 14.40.33810 worked, and the maintainers closed it as by design,
/// because a redistributable must be at least as new as the toolset that built
/// the program. 17.10 ships runtime 14.40. On the machine this check was written
/// from, 14.29.30133 crashed the llama.cpp server that Lemonade installs and
/// 14.51 ran it.
///
/// Confidence: high that anything below 14.40 breaks; not confirmed that 14.40
/// itself is enough for every build. A program built with a newer toolset can
/// ask for a newer runtime, so 14.40 is a floor and the fix tells the user to
/// install the latest one. It is a floor on purpose: raising it would flag
/// runtimes that work.
const MSVC_REDIST_MIN_VERSION: [u32; 2] = [14, 40];

/// Score for a runtime older than [`MSVC_REDIST_MIN_VERSION`] with no symptom.
/// Equal to [`HIGH_CONFIDENCE`](super::HIGH_CONFIDENCE): the version is read from the machine and the
/// threshold is a known break, so no symptom is needed to act on it.
const MSVC_REDIST_OUTDATED_SCORE: i32 = 75;

/// Whether `version` (for example `14.29.30133.0`) is older than `minimum`.
///
/// `false` for an empty or unparsable version: unknown is not old, and a false
/// match would send a user to reinstall a runtime that is fine. Parts missing on
/// either side count as zero, so `14.40` equals `14.40.0.0`.
fn msvc_redist_older_than(version: &str, minimum: &[u32]) -> bool {
    let Some(found) = crate::examine::parse_dotted_version(version) else {
        return false;
    };
    for i in 0..found.len().max(minimum.len()) {
        let have = found.get(i).copied().unwrap_or(0);
        let want = minimum.get(i).copied().unwrap_or(0);
        if have != want {
            return have < want;
        }
    }
    false
}

pub(super) fn check_13_hip_sdk_missing(e: &Examination, symptom: &str) -> Diagnosis {
    if e.os_family != "windows" {
        return zero("fix-13-hip-sdk-missing", "HIP SDK not installed");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    let sdk_path = &e.hip_sdk_path;

    // A missing or broken HIP SDK is only evidence of a fault when something on
    // this host actually calls into HIP: an ambient PyTorch HIP build, a
    // llama.cpp binary built with GGML_HIP=ON, or -- scored first below -- the
    // user's own pasted error. The CLI's own default (Lemonade's bundled
    // llama.cpp on Vulkan) never touches the system HIP SDK, and a
    // managed-runtime PyTorch wheel brings its own runtime too. Scoring the
    // SDK's absence unconditionally used to rank this fix first on exactly
    // those hosts: the ones least likely to need it.
    //
    // The pasted error is the strongest of the three signals, not the weakest.
    // `amdhip64.dll was not found` or `'hipInfo' is not recognized` is a
    // *running* process failing to load HIP, which is a fact about this host
    // that no probe can contradict -- the framework probe answering "nothing
    // detected" means it did not find a framework it knows, not that nothing
    // here calls into HIP. Gating the SDK evidence on the detected framework
    // alone dropped the +35 from the textbook case this entry exists for: a
    // Windows host with no SDK, pasting precisely that error.
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_HIP_SDK_MISSING);
    let pytorch_is_ambient_hip_build = e.has_amd_gpu
        && e.framework == "pytorch"
        && e.framework_rocm_version.starts_with("hip=")
        && e.framework_source != "managed-runtime";
    let llama_cpp_is_hip_build =
        e.has_amd_gpu && e.framework == "llama-cpp" && e.framework_rocm_version == "GGML_HIP=ON";
    let symptom_names_hip_loading = kw_score > 0;
    let needs_hip =
        pytorch_is_ambient_hip_build || llama_cpp_is_hip_build || symptom_names_hip_loading;

    if needs_hip {
        if sdk_path.is_empty() {
            score += 35;
            evidence.push("No HIP SDK install found under C:\\Program Files\\AMD\\ROCm".to_owned());
        } else if !e.hipinfo_present {
            score += 30;
            evidence.push(format!(
                "HIP SDK at {sdk_path} but hipInfo.exe is missing from its bin directory"
            ));
        }
        if pytorch_is_ambient_hip_build {
            score += 25;
            // Two wordings, because this clause is about PyTorch needing a
            // system HIP SDK and the arm above already said what is wrong with
            // the one here. Saying "the HIP SDK is not present" beside "HIP SDK
            // at C:\... but hipInfo.exe is missing" put two contradictory lines
            // in one finding, and the user had no way to tell which was true.
            evidence.push(if sdk_path.is_empty() {
                "PyTorch is a HIP build but the HIP SDK is not present on this host".to_owned()
            } else {
                "PyTorch is a HIP build, so it needs a working system HIP SDK".to_owned()
            });
        }
    }

    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-13-hip-sdk-missing", "HIP SDK not installed");
    }
    let fix = Fix {
        summary: "Install the AMD HIP SDK for Windows; the HIP runtime DLLs and hipInfo.exe come from there.".to_owned(),
        commands: vec![
            "# Download and install the HIP SDK (matched to your framework's HIP major):".to_owned(),
            "#   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html".to_owned(),
            "# After install, reopen the shell so HIP_PATH and PATH pick up the new install.".to_owned(),
        ],
        fix_id: "fix-13-hip-sdk-missing".to_owned(),
        verify: "powershell -NoProfile -Command \"& (Join-Path ([Environment]::GetEnvironmentVariable('HIP_PATH')) 'bin\\hipInfo.exe') | Select-Object -First 5\"".to_owned(),
        notes: vec!["If you only need PyTorch on Windows AMD and don't need the C/C++ HIP toolchain, the TheRock wheels bundle their own HIP runtime and may not require a system HIP SDK install.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-13-hip-sdk-missing",
        "HIP SDK not installed (Windows)",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_14_adrenalin_too_old(e: &Examination, symptom: &str) -> Diagnosis {
    if e.os_family != "windows" {
        return zero("fix-14-adrenalin-too-old", "Adrenalin driver too old");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    let sdk_path = &e.hip_sdk_path;
    if !sdk_path.is_empty() && e.hipinfo_present && !matches!(e.hipinfo_status.as_str(), "ok" | "")
    {
        score += 35;
        evidence.push(format!(
            "HIP SDK at {sdk_path} is installed but hipInfo.exe reports {:?}; this typically means the kernel-mode driver doesn't match the SDK.",
            e.hipinfo_status
        ));
    }
    if !e.adrenalin_version.is_empty() {
        evidence.push(format!(
            "Adrenalin / kernel-mode driver version: {}",
            e.adrenalin_version
        ));
    }
    if symptom_matches(symptom, r"driver.*(too old|out of date|unsupported)") {
        score += 35;
        evidence.push("error mentions 'driver too old / out of date / unsupported'".to_owned());
    }
    if symptom_matches(symptom, r"hsa.*invalid agent|no agents (were )?found") {
        score += 25;
        evidence.push("HSA error suggests driver/runtime can't enumerate the GPU".to_owned());
    }

    if score <= 0 {
        return zero("fix-14-adrenalin-too-old", "Adrenalin driver too old");
    }
    let fix = Fix {
        summary: "Update the AMD Adrenalin (or PRO) graphics driver to the version the HIP SDK release notes call out as the supported pairing.".to_owned(),
        commands: vec![
            "# Cross-check the HIP SDK release notes for the exact driver pairing:".to_owned(),
            "#   https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/install.html".to_owned(),
            "# Then download the matching driver from:".to_owned(),
            "#   https://www.amd.com/en/support".to_owned(),
            "# Reboot after the install for the kernel-mode driver to take effect.".to_owned(),
        ],
        needs_reboot: true,
        fix_id: "fix-14-adrenalin-too-old".to_owned(),
        verify: "powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | Where-Object Name -match 'AMD|Radeon' | Select-Object -First 1).DriverVersion\"".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-14-adrenalin-too-old",
        "Adrenalin / kernel-mode driver too old for the installed HIP SDK",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_15_msvc_redist(e: &Examination, symptom: &str) -> Diagnosis {
    if e.os_family != "windows" {
        return zero("fix-15-msvc-redist", "MSVC runtime missing");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    if e.msvc_redist_present == Some(false) {
        score += 45;
        evidence.push("vcruntime140.dll / vcruntime140_1.dll not resolvable on PATH".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_MSVC_REDIST);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-15-msvc-redist", "MSVC runtime missing");
    }
    let fix = Fix {
        summary: "Install the Microsoft Visual C++ 2015-2022 redistributable so the programs that link it, such as the HIP SDK's amdhip64_*.dll and the llama.cpp server that Lemonade installs, can load.".to_owned(),
        commands: vec![
            "# Download & install (x64):".to_owned(),
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe".to_owned(),
            "# After the install, reopen the shell and re-run your import / hipInfo check.".to_owned(),
        ],
        fix_id: "fix-15-msvc-redist".to_owned(),
        verify: "powershell -NoProfile -Command \"(Get-Item ([Environment]::GetFolderPath('System') + '\\vcruntime140.dll'),([Environment]::GetFolderPath('System') + '\\vcruntime140_1.dll') -ErrorAction Stop).VersionInfo | Select-Object FileName,FileVersion\"".to_owned(),
        notes: vec!["If installing the redistributable still leaves a missing-DLL error, the failing DLL is probably amdhip64_X.dll itself; that points at fix-13-hip-sdk-missing rather than this fix. If the runtime is present but the program crashes with an access violation, see fix-21-msvc-redist-outdated.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-15-msvc-redist",
        "MSVC runtime missing (programs that link it cannot load)",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_21_msvc_redist_outdated(e: &Examination, symptom: &str) -> Diagnosis {
    const ID: &str = "fix-21-msvc-redist-outdated";
    const TITLE: &str = "MSVC runtime too old (programs built with a recent toolset crash)";

    // Belt and braces with the `windows` tag in `CHECKERS`: a non-Windows host
    // has no such runtime, so it never matches, whatever it was handed.
    if e.os_family != "windows" {
        return zero(ID, TITLE);
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    if msvc_redist_older_than(&e.msvc_redist_version, &MSVC_REDIST_MIN_VERSION) {
        score += MSVC_REDIST_OUTDATED_SCORE;
        evidence.push(format!(
            "Visual C++ runtime {} in System32 is older than {}.{}",
            e.msvc_redist_version, MSVC_REDIST_MIN_VERSION[0], MSVC_REDIST_MIN_VERSION[1]
        ));
    }
    let (kw_score, kw_ev) =
        keyword_score_collapsing_overlaps(symptom, KEYWORDS_MSVC_REDIST_OUTDATED);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero(ID, TITLE);
    }
    let fix = Fix {
        summary: "Update the Microsoft Visual C++ 2015-2022 redistributable. Programs built with a recent toolset, such as the llama.cpp server that Lemonade installs, crash on a runtime older than 14.40.".to_owned(),
        commands: vec![
            "# Update the runtime (x64). This needs the Windows administrator prompt, so".to_owned(),
            "# the command is shown here and not run for you:".to_owned(),
            "winget install --id Microsoft.VCRedist.2015+.x64 --exact".to_owned(),
            "# Or download and run the installer:".to_owned(),
            "#   https://aka.ms/vs/17/release/vc_redist.x64.exe".to_owned(),
            "# Then close and restart the program (or Lemonade) so it loads the new runtime.".to_owned(),
        ],
        fix_id: ID.to_owned(),
        verify: "powershell -NoProfile -Command \"(Get-Item ([Environment]::GetFolderPath('System') + '\\msvcp140.dll') -ErrorAction Stop).VersionInfo.FileVersion\"".to_owned(),
        notes: vec![
            "The verify command prints the runtime version. It should read 14.40 or higher.".to_owned(),
            "`rocm fix` only prints this plan. It does not run the install: the install needs the Windows administrator prompt and your consent.".to_owned(),
            "14.40 is the first runtime that matches the toolset where the standard library changed. A newer program can still ask for a newer runtime, so install the latest one.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(ID, TITLE, score, evidence, fix)
}

/// The score for an old display driver.
///
/// Below [`MIN_SCORE_FOR_MATCH`](super::MIN_SCORE_FOR_MATCH) on purpose. The entry is advice to look at the
/// driver, not a finding that anything is broken: AMD publishes no minimum
/// driver version for the HIP SDK or for the Lemonade Vulkan backend. A WEAK
/// row is shown by `rocm diagnose` and does not set `has_match`.
pub(super) const DISPLAY_DRIVER_OLD_SCORE: i32 = 25;

/// An AMD display driver dated before the first HIP SDK for Windows.
///
/// Separate from [`check_14_adrenalin_too_old`] on purpose. That entry needs a
/// HIP SDK and a failing `hipInfo` or a symptom. This one needs neither, so it
/// also covers a machine that only runs the Vulkan backend. Only evidence is
/// read: the driver date from the examination, never the symptom text.
pub(super) fn check_22_display_driver_old(e: &Examination, _symptom: &str) -> Diagnosis {
    const TITLE: &str = "AMD display driver is older than the first HIP SDK for Windows";
    let id = crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID;
    if e.os_family != "windows" {
        return zero(id, TITLE);
    }
    let Some(info) = crate::windows_driver::WindowsDriverInfo::from_probe(
        &e.adrenalin_version,
        &e.adrenalin_date,
    ) else {
        return zero(id, TITLE);
    };
    let Some(evidence) = crate::windows_driver::old_driver_evidence(&info) else {
        return zero(id, TITLE);
    };
    let fix = Fix {
        summary: "Look at AMD's support page for a newer display driver for this GPU. This is advice only: no minimum driver version is established, and rocm-cli does not install drivers on Windows.".to_owned(),
        commands: vec![
            "# Find the current display driver for this GPU on AMD's support page:".to_owned(),
            format!("#   {}", crate::windows_driver::AMD_SUPPORT_URL),
            "# Restart the machine after the driver install.".to_owned(),
        ],
        needs_reboot: true,
        fix_id: id.to_owned(),
        verify: "powershell -NoProfile -Command \"Get-CimInstance Win32_VideoController | Where-Object Name -match 'AMD|Radeon' | Select-Object Name,DriverVersion,DriverDate\"".to_owned(),
        notes: vec![
            crate::windows_driver::NO_MINIMUM_EXPLANATION.to_owned(),
            crate::windows_driver::FLOOR_NOTE.to_owned(),
        ],
        ..Fix::default()
    };
    finalize(id, TITLE, DISPLAY_DRIVER_OLD_SCORE, vec![evidence], fix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::test_support::{linux_base, windows_driver_exam, wsl_base};
    use crate::diagnose::{
        DiagnoseReport, HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH, diagnose, render_report_text,
    };
    use crate::examine::Gpu;

    #[test]
    fn a_managed_runtimes_hip_build_is_not_evidence_of_a_missing_hip_sdk() {
        // The Windows-shaped sibling of the check_8 gate: TheRock wheels bring
        // their own HIP runtime, so a HIP-build torch on a host with no system
        // HIP SDK says nothing when that torch came from a managed runtime --
        // not even the base "no system HIP SDK" evidence, since nothing on this
        // host actually needs one.
        let windows = |source: &str| Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            framework: "pytorch".to_owned(),
            framework_rocm_version: "hip=7.14.60850".to_owned(),
            framework_source: source.to_owned(),
            ..Examination::default()
        };

        let managed = check_13_hip_sdk_missing(&windows("managed-runtime"), "");
        assert_eq!(
            managed.score, 0,
            "a managed runtime's own HIP build is not evidence a system HIP SDK \
             is missing: {managed:?}"
        );

        let ambient = check_13_hip_sdk_missing(&windows("path"), "");
        assert_eq!(
            ambient.score, 60,
            "the ambient build needs both the base SDK-missing evidence (35) and \
             the HIP-build clause (25): {ambient:?}"
        );
    }

    #[test]
    fn fix_13_scores_an_sdk_whose_hipinfo_is_missing_only_when_something_needs_hip() {
        // The `else if !hipinfo_present` arm: an SDK directory is present but
        // its bin/hipInfo.exe is gone -- a half-installed or hand-pruned SDK.
        // This PR moved that arm inside the `needs_hip` gate, so it now has to
        // be read as two cases, not one: the same broken SDK is evidence on a
        // host that calls into HIP, and silence on one that does not.
        let host = |framework: &str, rocm_version: &str| Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            hip_sdk_path: "C:\\Program Files\\AMD\\ROCm\\6.4".to_owned(),
            hipinfo_present: false,
            framework: framework.to_owned(),
            framework_rocm_version: rocm_version.to_owned(),
            framework_source: "path".to_owned(),
            ..Examination::default()
        };

        let needs_hip = check_13_hip_sdk_missing(&host("pytorch", "hip=7.14.60850"), "");
        assert_eq!(
            needs_hip.score, 55,
            "a broken SDK under an ambient HIP torch scores the missing-hipInfo \
             evidence (30) plus the HIP-build clause (25), and NOT the \
             SDK-absent 35 -- the SDK directory is present: {needs_hip:?}"
        );
        assert!(
            needs_hip
                .evidence
                .iter()
                .any(|ev| ev.contains("hipInfo.exe is missing")),
            "the missing-hipInfo evidence must name what is broken: {needs_hip:?}"
        );

        // Same half-installed SDK, nothing on the host built against HIP: the
        // Lemonade/Vulkan shape of ROCMAI-546. A broken SDK nobody calls into
        // is not a fault worth ranking.
        let no_hip = check_13_hip_sdk_missing(&host("llama-cpp", ""), "");
        assert_eq!(
            no_hip.score, 0,
            "with no HIP-based setup the SDK's state is not evidence of \
             anything: {no_hip:?}"
        );
    }

    #[test]
    fn fix_13_does_not_contradict_itself_about_whether_the_hip_sdk_is_there() {
        // The PyTorch clause has two wordings, chosen on `sdk_path.is_empty()`,
        // and the choice had nothing standing on it: forcing that condition to
        // `true` left every fix-13 test green while putting "the HIP SDK is not
        // present on this host" in the same finding as "HIP SDK at C:\... but
        // hipInfo.exe is missing". A user reading two opposite lines under one
        // cause has no way to tell which is true, which is the whole reason the
        // branch exists.
        //
        // Both arms are asserted together because the failure is a pair: the
        // wording that is right for one host is wrong for the other, so a test
        // holding only one arm is satisfied by a constant.
        let host = |sdk_path: &str| Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            hip_sdk_path: sdk_path.to_owned(),
            hipinfo_present: false,
            framework: "pytorch".to_owned(),
            framework_rocm_version: "hip=7.14.60850".to_owned(),
            framework_source: "path".to_owned(),
            ..Examination::default()
        };

        let absent = check_13_hip_sdk_missing(&host(""), "");
        let absent_text = absent.evidence.join("\n");
        assert!(
            absent_text.contains("the HIP SDK is not present on this host"),
            "with no SDK anywhere, the PyTorch clause has to say the SDK is \
             absent: {absent:?}"
        );
        assert!(
            !absent_text.contains("needs a working system HIP SDK"),
            "and must not instead imply one is there but not working: {absent:?}"
        );

        let broken = check_13_hip_sdk_missing(&host("C:\\Program Files\\AMD\\ROCm\\6.4"), "");
        let broken_text = broken.evidence.join("\n");
        assert!(
            broken_text.contains("needs a working system HIP SDK"),
            "with an SDK directory present, the clause is about it not working, \
             not about it being absent: {broken:?}"
        );
        assert!(
            !broken_text.contains("the HIP SDK is not present on this host"),
            "saying the SDK is absent beside `HIP SDK at C:\\... but hipInfo.exe \
             is missing` puts two contradictory lines in one finding: {broken:?}"
        );
        // And the line it would contradict is in that same finding, so the
        // assertion above is about a real pairing and not a hypothetical one.
        assert!(
            broken_text.contains("hipInfo.exe is missing"),
            "the arm this wording exists to agree with must be present: {broken:?}"
        );
    }

    #[test]
    fn fix_13_does_not_fire_on_a_lemonade_vulkan_host_with_no_hip_evidence() {
        // ROCMAI-546: the laptop that opened this ticket. A GPU with no ROCm/HIP
        // support (gfx90c) served over Lemonade's bundled, Vulkan-built
        // llama.cpp -- nothing on the host ever calls into HIP, so the absence
        // of a system HIP SDK must not itself read as a fault.
        let e = Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            gpus: vec![Gpu {
                name: "AMD Radeon Vega 7".to_owned(),
                gfx_target: "gfx90c".to_owned(),
                is_amd: true,
                ..Gpu::default()
            }],
            framework: "llama-cpp".to_owned(),
            framework_source: "path".to_owned(),
            // Lemonade's llama.cpp build advertises Vulkan, not HIP/ROCm.
            framework_rocm_version: String::new(),
            ..Examination::default()
        };
        let report = diagnose(&e, "");
        assert!(
            !report
                .matched
                .iter()
                .any(|d| d.id == "fix-13-hip-sdk-missing"),
            "fix-13 must not fire with no HIP-based setup configured: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
        let info = report
            .matched
            .iter()
            .find(|d| d.id == "info-20-gpu-no-rocm-support")
            .expect("the no-ROCm-support diagnosis must appear with the detected gfx target");
        assert!(
            info.evidence.iter().any(|ev| ev.contains("gfx90c")),
            "the diagnosis must name the detected GPU architecture: {info:?}"
        );
        let fix = info
            .fix
            .as_ref()
            .expect("every catalog entry carries a Fix, including the fixless one");
        assert_eq!(
            fix.class,
            crate::fix::FixClass::DiagnoseOnly,
            "the class is where \"detected, explained, no remedy\" is recorded, and it is \
             what every consumer reads to tell this entry from a repair: {info:?}"
        );
        assert!(
            !fix.auto_applicable && fix.commands.is_empty() && fix.verify.is_empty(),
            "nothing to run, nothing to copy, nothing to verify: {info:?}"
        );
        assert_eq!(
            report.matched.first().map(|d| d.id.as_str()),
            Some("info-20-gpu-no-rocm-support"),
            "the no-ROCm-support diagnosis, not fix-13, must rank first: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    /// ROCMAI-830: the other host ROCMAI-546's gate has to hold on. A native
    /// Windows Strix Halo (gfx1151) whose default engine is Lemonade, with no
    /// system HIP SDK, diagnosed twice: before any runtime was installed, and
    /// after `rocm install sdk` put a managed TheRock wheel runtime in place.
    ///
    /// Unlike the gfx90c laptop above, gfx1151 is a supported ROCm target, so
    /// `info-20` does not fire and nothing outranks fix-13 here: on v0.2.1,
    /// which was cut before the gate landed, fix-13 was the *only* entry and
    /// ranked #1 at 35 on a machine serving models fine. A managed runtime is
    /// not a HIP-based setup that needs the system SDK either -- its wheels
    /// bundle their own HIP runtime -- whether the torch probe through it
    /// answered (a `hip=` build) or timed out (`unknown`, as on the reporting
    /// machine under load).
    ///
    /// The fields mirror that machine's `rocm examine --json`. The final
    /// positive-control arm keeps the gate honest: the same host with an
    /// ambient HIP-built torch on PATH does need the SDK, so fix-13 still
    /// ranks first there.
    #[test]
    fn fix_13_does_not_fire_on_a_lemonade_strix_halo_host_with_a_managed_runtime() {
        let strix_halo = |framework: &str, rocm_version: &str, source: &str| Examination {
            os_family: "windows".to_owned(),
            cpu_vendor: "amd".to_owned(),
            gpus: vec![Gpu {
                name: "AMD Radeon(TM) 8060S Graphics".to_owned(),
                gfx_target: "gfx1151".to_owned(),
                is_apu: Some(true),
                is_amd: true,
                ..Gpu::default()
            }],
            has_amd_gpu: true,
            has_apu: true,
            // No system HIP SDK: the arm that used to score 35 on its own.
            hip_sdk_path: String::new(),
            hipinfo_present: false,
            adrenalin_version: "32.0.32015.7029".to_owned(),
            msvc_redist_present: Some(true),
            framework: framework.to_owned(),
            framework_rocm_version: rocm_version.to_owned(),
            framework_source: source.to_owned(),
            ..Examination::default()
        };
        let summary = |report: &DiagnoseReport| {
            report
                .matched
                .iter()
                .map(|d| (d.id.clone(), d.score))
                .collect::<Vec<_>>()
        };

        for (label, host) in [
            (
                "before any runtime was installed",
                strix_halo("unknown", "", ""),
            ),
            // Framework "unknown" means nothing HIP-based is detected, so
            // `needs_hip` is false whatever `framework_source` says: this arm
            // documents the reporting machine's shape but does not exercise
            // the managed-runtime exemption. The next arm does.
            (
                "managed runtime whose torch probe timed out (no HIP evidence)",
                strix_halo("unknown", "", "managed-runtime"),
            ),
            (
                "managed runtime whose torch is a HIP build",
                strix_halo("pytorch", "hip=7.14.60850", "managed-runtime"),
            ),
        ] {
            assert_eq!(
                check_13_hip_sdk_missing(&host, "").score,
                0,
                "{label}: nothing on this host calls into the system HIP SDK"
            );
            let report = diagnose(&host, "");
            assert!(
                report.matched.is_empty() && !report.has_match,
                "{label}: a healthy Lemonade host must report no match, not a WEAK \
                 fix-13 at #1: {:?}",
                summary(&report)
            );
        }

        let ambient = diagnose(&strix_halo("pytorch", "hip=7.14.60850", "path"), "");
        assert_eq!(
            ambient.matched.first().map(|d| d.id.as_str()),
            Some("fix-13-hip-sdk-missing"),
            "the same host with a HIP-built torch on PATH really needs the SDK: {:?}",
            summary(&ambient)
        );
    }

    /// Pins what production does today for an ambient HIP-built PyTorch beside
    /// a healthy system SDK: the +25 "PyTorch needs a working SDK" bonus is not
    /// gated on the SDK being unusable, so fix-13 still lists (score 25, below
    /// `MIN_SCORE_FOR_MATCH`, so advice only). The diagnose-48 oracle mirrors
    /// this. If the bonus is later gated on `sdk_unusable`, change both.
    #[test]
    fn fix_13_lists_weakly_for_ambient_hip_torch_beside_a_healthy_system_sdk() {
        let host = |hip_sdk_path: &str, hipinfo_present: bool, source: &str| Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            hip_sdk_path: hip_sdk_path.to_owned(),
            hipinfo_present,
            framework: "pytorch".to_owned(),
            framework_rocm_version: "hip=7.14.60850".to_owned(),
            framework_source: source.to_owned(),
            ..Examination::default()
        };
        let sdk = r"C:\Program Files\AMD\ROCm\6.4";

        let healthy = host(sdk, true, "path");
        assert_eq!(check_13_hip_sdk_missing(&healthy, "").score, 25);
        let report = diagnose(&healthy, "");
        let row = report
            .matched
            .iter()
            .find(|d| d.id == "fix-13-hip-sdk-missing")
            .expect("ambient HIP torch lists fix-13 even with a healthy SDK");
        assert!(
            row.score < MIN_SCORE_FOR_MATCH,
            "advice only: {}",
            row.score
        );

        // Control: the same healthy SDK under a managed runtime is silent.
        assert_eq!(
            check_13_hip_sdk_missing(&host(sdk, true, "managed-runtime"), "").score,
            0
        );
    }

    /// The textbook fix-13 host: Windows, an AMD GPU, no HIP SDK, no framework
    /// the probe recognises — and a pasted error that is HIP failing to load.
    ///
    /// The gate that stops fix-13 top-ranking a Vulkan host must not also stop
    /// it on this one. The error text *is* the HIP-use evidence: something on
    /// the machine tried to load HIP and could not, which no framework probe
    /// can contradict — "no framework detected" means none that `examine`
    /// knows, not that nothing here calls into HIP. Gating on the detected
    /// framework alone dropped the +35 from exactly the case the entry exists
    /// for, taking `'hipInfo' is not recognized` from 80 to 45 and
    /// `could not find HIP` from 75 to 40 — both below `MIN_SCORE_FOR_MATCH`,
    /// so `has_match` went false and the user was routed upstream for a
    /// problem the catalog has a fix for.
    ///
    /// Measured, not reasoned: these three numbers were read off a run on the
    /// base branch before the gate existed.
    #[test]
    fn fix_13_scores_a_pasted_hip_load_failure_with_no_framework_detected() {
        let windows_no_sdk = Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            gpus: vec![Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                ..Gpu::default()
            }],
            // No framework, and no HIP SDK: `hip_sdk_path` empty is the +35 arm.
            ..Examination::default()
        };

        for (symptom, expected) in [
            (
                "'hipInfo' is not recognized as an internal or external command",
                80,
            ),
            ("could not find HIP", 75),
            ("amdhip64.dll was not found", 85),
        ] {
            let finding = check_13_hip_sdk_missing(&windows_no_sdk, symptom);
            assert_eq!(
                finding.score, expected,
                "the pasted error is what establishes that this host loads HIP, so the \
                 SDK-absent 35 belongs on top of the keyword score for {symptom:?}: {finding:?}"
            );
            assert!(
                finding
                    .evidence
                    .iter()
                    .any(|ev| ev.contains("No HIP SDK install found")),
                "and the finding has to say the SDK is missing, which is the thing to \
                 fix, not only quote the error back: {finding:?}"
            );
            let report = diagnose(&windows_no_sdk, symptom);
            assert!(
                report.has_match,
                "{symptom:?} must establish a cause rather than routing the user \
                 upstream: {:?}",
                report
                    .matched
                    .iter()
                    .map(|d| (&d.id, d.score))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// The gate still holds on the host it was built for: a pasted error that
    /// says nothing about HIP opens nothing.
    ///
    /// Without this, "let a keyword count as HIP use" could be read as "let any
    /// symptom count", which would put the entry back where it started.
    #[test]
    fn fix_13_stays_silent_for_a_symptom_that_does_not_name_hip() {
        let finding = check_13_hip_sdk_missing(
            &Examination {
                os_family: "windows".to_owned(),
                has_amd_gpu: true,
                ..Examination::default()
            },
            "the server exits with code 3221225781 on startup",
        );
        assert_eq!(
            finding.score, 0,
            "an unrelated paste on a Vulkan host is not evidence the HIP SDK is \
             missing: {finding:?}"
        );
    }

    #[test]
    fn fix_13_keeps_its_weak_signal_for_a_native_rocm_llama_cpp_build() {
        // Regression guard for the other half of the acceptance criteria: a host
        // that genuinely runs a HIP-built llama.cpp (e.g. a manual/vLLM build,
        // as opposed to Lemonade's Vulkan one) with no system HIP SDK keeps the
        // base SDK-missing evidence rather than being gated to silence.
        //
        // "Keeps", not "still fires": 35 is below MIN_SCORE_FOR_MATCH, so this
        // is a WEAK entry that does not set `has_match`. The name used to say
        // "still fires", which claimed more than the assertion pins -- the score
        // is the same 35 the pre-gate code produced, so the number alone cannot
        // tell the two versions apart. What this test actually guards is the
        // `llama_cpp_is_hip_build` arm of `needs_hip`: drop that arm and the
        // score falls to 0.
        let e = Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: true,
            framework: "llama-cpp".to_owned(),
            framework_source: "path".to_owned(),
            framework_rocm_version: "GGML_HIP=ON".to_owned(),
            ..Examination::default()
        };
        let finding = check_13_hip_sdk_missing(&e, "");
        assert_eq!(
            finding.score, 35,
            "a HIP-built llama.cpp with no system HIP SDK must still score the base \
             SDK-missing evidence: {finding:?}"
        );
        assert!(
            finding.score > 0 && finding.score < MIN_SCORE_FOR_MATCH,
            "and it stays a weak signal, not a match: {finding:?}"
        );
    }

    #[test]
    fn fix_13_does_not_fire_for_a_hip_built_llama_cpp_binary_with_no_amd_gpu() {
        // A HIP/ROCm-advertising llama.cpp binary on PATH is not, by itself,
        // evidence this host has AMD hardware to run it on -- the same reasoning
        // `pytorch_is_ambient_hip_build` already applies. Without it, this host
        // would get the same spurious fix-13 the Lemonade/Vulkan laptop did.
        let e = Examination {
            os_family: "windows".to_owned(),
            has_amd_gpu: false,
            framework: "llama-cpp".to_owned(),
            framework_source: "path".to_owned(),
            framework_rocm_version: "GGML_HIP=ON".to_owned(),
            ..Examination::default()
        };
        assert_eq!(
            check_13_hip_sdk_missing(&e, "").score,
            0,
            "no AMD GPU means nothing on this host needs the HIP SDK"
        );
    }

    #[test]
    fn the_display_driver_old_plan_matches_the_catalog_copy() {
        // The `diagnose` copy of fix-22 and the `rocm fix` recipe are one plan.
        let fix =
            check_22_display_driver_old(&windows_driver_exam("30.0.13014.8", "2021-08-04"), "")
                .fix
                .expect("an old driver must carry a plan");
        let commands: Vec<&str> = fix.commands.iter().map(String::as_str).collect();
        crate::fix::assert_plan_matches_the_catalog_copy(&fix.fix_id, &commands);
    }

    #[test]
    fn msvc_redist_diagnosed_on_windows() {
        let mut e = Examination {
            os_family: "windows".to_owned(),
            ..Examination::default()
        };
        e.msvc_redist_present = Some(false);
        let report = diagnose(
            &e,
            "The program can't start because vcruntime140.dll is missing",
        );
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-15-msvc-redist");
        assert!(top.score >= MIN_SCORE_FOR_MATCH);
    }

    #[test]
    fn windows_missing_dll_pair_still_reaches_the_msvc_redist_fix() {
        // The user-visible half of the guard above: with no PATH probe result
        // (`msvc_redist_present == None`) the keyword score alone decides, so a
        // demoted table makes fix-15 vanish from the report entirely.
        let e = Examination {
            os_family: "windows".to_owned(),
            ..Examination::default()
        };
        let report = diagnose(
            &e,
            "api-ms-win-crt-runtime-l1-1-0.dll missing and msvcp140.dll not found",
        );
        let msvc = report
            .matched
            .iter()
            .find(|d| d.id == "fix-15-msvc-redist")
            .expect("the MSVC redistributable fix must still be reported");
        assert!(
            msvc.score >= MIN_SCORE_FOR_MATCH,
            "score was {}",
            msvc.score
        );
        assert!(report.has_match());
    }

    /// What the Windows event log and Lemonade print when the llama.cpp server
    /// dies on a Visual C++ runtime that is too old.
    const MSVC_OUTDATED_CRASH_TEXT: &str = "llama-server.exe exited with code 3221225477. \
        Faulting application name: llama-server.exe, version: 0.0.0.0, \
        Faulting module name: MSVCP140.dll, version: 14.29.30133.0, \
        Exception code: 0xc0000005 (access violation)";

    /// The same crash as the server's own log shows it, with no event-log line.
    const MSVC_OUTDATED_CRASH_TEXT_SHORT: &str =
        "llama-server failed: process exited with 0xC0000005 in MSVCP140.dll";

    fn windows_with_runtime(version: &str) -> Examination {
        Examination {
            os_family: "windows".to_owned(),
            msvc_redist_version: version.to_owned(),
            ..Examination::default()
        }
    }

    fn outdated_runtime_entry(report: &DiagnoseReport) -> Option<&Diagnosis> {
        report
            .matched
            .iter()
            .find(|d| d.id == "fix-21-msvc-redist-outdated")
    }

    #[test]
    fn an_outdated_runtime_is_a_high_confidence_match_with_no_symptom() {
        let report = diagnose(&windows_with_runtime("14.29.30133.0"), "");
        let hit = outdated_runtime_entry(&report).expect("14.29 is below the minimum");
        assert!(
            hit.score >= HIGH_CONFIDENCE,
            "no symptom needed: score {}",
            hit.score
        );
        assert!(report.has_match);
        assert!(hit.evidence.iter().any(|ev| ev.contains("14.29.30133.0")));
    }

    #[test]
    fn the_minimum_runtime_version_is_14_40() {
        // Pins the chosen floor. The evidence for it is on the constant.
        assert_eq!(MSVC_REDIST_MIN_VERSION, [14, 40]);
    }

    #[test]
    fn runtime_version_comparison_edge_cases() {
        let min = &MSVC_REDIST_MIN_VERSION;
        // Older.
        assert!(msvc_redist_older_than("14.29.30133.0", min));
        assert!(msvc_redist_older_than("14.39.99999.0", min));
        assert!(msvc_redist_older_than("13.99", min));
        assert!(msvc_redist_older_than("14.32.31326.0", min));
        // Equal to the minimum is not older, in any spelling.
        assert!(!msvc_redist_older_than("14.40", min));
        assert!(!msvc_redist_older_than("14.40.0.0", min));
        assert!(!msvc_redist_older_than("14.40.33810.0", min));
        // Newer, including a major version above 14 and a four-part version.
        assert!(!msvc_redist_older_than("14.51.36231.0", min));
        assert!(!msvc_redist_older_than("15.0", min));
        // Compared as numbers, not as text: 14.9 is older than 14.40.
        assert!(msvc_redist_older_than("14.9.1.0", min));
        assert!(!msvc_redist_older_than("14.100.0.0", min));
        // Unknown is never old.
        for unknown in [
            "",
            "   ",
            "14",
            "unknown",
            "14.x.1",
            "v14.29",
            "14.29.beta",
            "-1.2",
        ] {
            assert!(
                !msvc_redist_older_than(unknown, min),
                "{unknown:?} must not count as old"
            );
        }
    }

    #[test]
    fn a_missing_or_unparsable_runtime_version_alone_does_not_match() {
        for version in ["", "garbage", "14"] {
            let report = diagnose(&windows_with_runtime(version), "");
            assert!(
                outdated_runtime_entry(&report).is_none(),
                "{version:?} produced a match"
            );
        }
    }

    #[test]
    fn a_current_runtime_does_not_match_without_a_symptom() {
        for version in ["14.40.33810.0", "14.51.36231.0"] {
            let report = diagnose(&windows_with_runtime(version), "");
            assert!(outdated_runtime_entry(&report).is_none(), "{version}");
        }
    }

    #[test]
    fn an_outdated_runtime_never_matches_off_windows() {
        for family in ["linux", "unknown", ""] {
            let exam = Examination {
                os_family: family.to_owned(),
                msvc_redist_version: "14.29.30133.0".to_owned(),
                ..Examination::default()
            };
            let direct = check_21_msvc_redist_outdated(&exam, MSVC_OUTDATED_CRASH_TEXT);
            assert_eq!(direct.score, 0, "{family:?}");
            let report = diagnose(&exam, MSVC_OUTDATED_CRASH_TEXT);
            assert!(outdated_runtime_entry(&report).is_none(), "{family:?}");
        }
        let mut wsl = linux_base();
        wsl.is_wsl = true;
        wsl.msvc_redist_version = "14.29.30133.0".to_owned();
        let report = diagnose(&wsl, MSVC_OUTDATED_CRASH_TEXT);
        assert!(outdated_runtime_entry(&report).is_none());
    }

    #[test]
    fn the_crash_text_from_an_outdated_runtime_ranks_first() {
        // With the runtime version known, and with nothing known about it.
        for exam in [
            windows_with_runtime("14.29.30133.0"),
            windows_with_runtime(""),
        ] {
            for text in [MSVC_OUTDATED_CRASH_TEXT, MSVC_OUTDATED_CRASH_TEXT_SHORT] {
                let report = diagnose(&exam, text);
                let top = &report.matched[0];
                assert_eq!(
                    top.id,
                    "fix-21-msvc-redist-outdated",
                    "runtime {:?}, text {text:?}: {:?}",
                    exam.msvc_redist_version,
                    report
                        .matched
                        .iter()
                        .map(|d| (&d.id, d.score))
                        .collect::<Vec<_>>()
                );
                assert!(top.score >= MIN_SCORE_FOR_MATCH);
            }
        }
        // The full event-log text reaches high confidence from the symptom alone.
        let report = diagnose(&windows_with_runtime(""), MSVC_OUTDATED_CRASH_TEXT);
        assert!(report.matched[0].score >= HIGH_CONFIDENCE);
    }

    #[test]
    fn an_access_violation_without_the_runtime_is_not_blamed_on_it() {
        // A bare crash code on a machine with a current runtime stays unmatched.
        let report = diagnose(
            &windows_with_runtime("14.51.36231.0"),
            "exit code 0xC0000005",
        );
        assert!(outdated_runtime_entry(&report).is_none_or(|d| d.score < MIN_SCORE_FOR_MATCH));
    }

    #[test]
    fn the_outdated_runtime_remediation_covers_the_llama_server_and_stays_print_only() {
        let report = diagnose(&windows_with_runtime("14.29.30133.0"), "");
        let fix = outdated_runtime_entry(&report)
            .and_then(|d| d.fix.as_ref())
            .expect("a matched diagnosis carries a fix");
        assert!(fix.summary.contains("llama.cpp"));
        assert!(fix.summary.contains("Lemonade"));
        assert!(
            fix.commands
                .iter()
                .any(|c| c == "winget install --id Microsoft.VCRedist.2015+.x64 --exact")
        );
        assert!(
            fix.commands
                .iter()
                .any(|c| c.contains("aka.ms/vs/17/release/vc_redist.x64.exe"))
        );
        assert!(fix.verify.contains("msvcp140.dll") && fix.verify.contains("FileVersion"));
        // The plan promises it only prints. The class is checked against the
        // catalog by `the_outdated_runtime_recipe_is_print_only_on_windows` in
        // fix.rs, which is what makes this sentence true.
        assert!(
            fix.notes
                .iter()
                .any(|n| n.contains("only prints this plan"))
        );
    }

    #[test]
    fn the_missing_runtime_fix_no_longer_says_only_the_hip_sdk_needs_it() {
        let report = diagnose(
            &Examination {
                os_family: "windows".to_owned(),
                msvc_redist_present: Some(false),
                ..Examination::default()
            },
            "",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-15-msvc-redist")
            .expect("a missing runtime still matches");
        let fix = hit.fix.as_ref().expect("fix");
        assert!(fix.summary.contains("llama.cpp"), "{}", fix.summary);
        assert!(!hit.title.contains("HIP DLLs cannot load"), "{}", hit.title);
    }

    fn old_driver_entry(e: &Examination) -> Option<Diagnosis> {
        diagnose(e, "")
            .matched
            .into_iter()
            .find(|d| d.id == crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID)
    }

    #[test]
    fn a_2021_display_driver_is_flagged_as_advice_only() {
        let hit = old_driver_entry(&windows_driver_exam("30.0.13014.8", "2021-08-04"))
            .expect("a 2021 driver must be flagged");
        assert_eq!(hit.score, DISPLAY_DRIVER_OLD_SCORE);
        assert!(
            hit.score < MIN_SCORE_FOR_MATCH,
            "advice must not read as an established cause"
        );
        assert!(hit.evidence.iter().any(|ev| ev.contains("30.0.13014.8")));
        assert!(hit.evidence.iter().any(|ev| ev.contains("2021-08-04")));
        assert!(hit.evidence.iter().any(|ev| ev.contains("not established")));
        let fix = hit.fix.expect("fix");
        assert!(!fix.commands.is_empty());
        assert!(
            fix.commands.iter().all(|c| c.starts_with('#')),
            "the recipe must print no command that installs anything: {:?}",
            fix.commands
        );
        assert!(fix.commands.iter().any(|c| c.contains("Restart")));
        assert!(
            fix.commands
                .iter()
                .any(|c| c.contains("amd.com/en/support"))
        );
    }

    fn old_driver_only_exam() -> Examination {
        Examination {
            hip_sdk_path: "C:\\Program Files\\AMD\\ROCm\\6.2".to_owned(),
            hipinfo_present: true,
            ..windows_driver_exam("30.0.13014.8", "2021-08-04")
        }
    }

    #[test]
    fn an_old_driver_alone_keeps_the_upstream_escalation_in_the_text_report() {
        // State: the old-driver row is present, nothing else is, and no
        // established cause exists.
        let report = diagnose(&old_driver_only_exam(), "");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, [crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID]);
        assert!(!report.has_match);

        // Text: the symptom was not recognised, so the report says where to
        // file it, and it does not present the advice as the best match.
        let text = render_report_text(&report, 5);
        assert!(text.contains("no known misconfiguration matched"), "{text}");
        assert!(text.contains(&report.route_when_no_match.url), "{text}");
        assert!(text.contains("rocm examine --json"), "{text}");
        assert!(text.contains("fix-22-display-driver-old"), "{text}");
        assert!(!text.contains("highest-scoring match"), "{text}");
        assert!(!text.contains("HIGH_CONFIDENCE"), "{text}");

        // The line that introduces the advice makes a promise: nothing in it is
        // applied automatically. The promise and the behaviour are asserted in
        // the same test: every command the entry carries is a printed comment
        // (`a_2021_display_driver_is_flagged_as_advice_only` pins the same for
        // the recipe), so nothing in the advice can run anything.
        assert!(
            text.contains(
                "Advice about this host follows. It does not explain the symptom, and nothing in it is applied automatically."
            ),
            "{text}"
        );
        let fix = report.matched[0]
            .fix
            .as_ref()
            .expect("the advice has a fix");
        assert!(
            fix.commands.iter().all(|c| c.starts_with('#')),
            "the advice must apply nothing: {:?}",
            fix.commands
        );
    }

    #[test]
    fn a_real_cause_beside_an_old_driver_is_still_the_highest_scoring_match() {
        // The advice row must not be named as the best match, and a real
        // finding must keep that role.
        let mut e = old_driver_only_exam();
        e.msvc_redist_version = "14.29.30133.0".to_owned();
        let report = diagnose(&e, "");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert!(ids.contains(&crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));
        assert!(ids.contains(&"fix-21-msvc-redist-outdated"));
        let text = render_report_text(&report, 5);
        assert!(
            !text.contains("no known misconfiguration matched"),
            "{text}"
        );
        assert!(
            text.contains("run `rocm fix fix-21-msvc-redist-outdated`"),
            "{text}"
        );
        assert!(
            !text.contains("run `rocm fix fix-22-display-driver-old` (the highest"),
            "{text}"
        );
    }

    #[test]
    fn a_below_high_real_cause_beside_an_old_driver_is_the_one_named_as_next_step() {
        for (score, tier) in [(60, "LIKELY"), (30, "WEAK")] {
            let mut report = diagnose(&old_driver_only_exam(), "");
            assert_eq!(report.matched.len(), 1, "fixture: only the advice row");
            report.matched.insert(
                0,
                Diagnosis {
                    id: "fix-99-synthetic-real-cause".to_owned(),
                    title: "A real cause below HIGH_CONFIDENCE".to_owned(),
                    score,
                    evidence: vec!["synthetic".to_owned()],
                    fix: None,
                },
            );
            assert!(score < HIGH_CONFIDENCE);
            let text = render_report_text(&report, 5);
            assert!(
                text.contains(&format!("[{tier} score={score}/100]")),
                "{text}"
            );
            assert!(
                text.contains("below the HIGH_CONFIDENCE threshold"),
                "{text}"
            );
            assert!(
                text.contains(
                    "run `rocm fix fix-99-synthetic-real-cause` (the highest-scoring match)"
                ),
                "{text}"
            );
            assert!(
                !text.contains("run `rocm fix fix-22-display-driver-old` (the highest"),
                "{text}"
            );
            // The advice row is still listed, after the real cause.
            assert!(text.contains("#2 (fix-22-display-driver-old)"), "{text}");
        }
    }

    #[test]
    fn an_old_driver_leaves_the_json_report_as_it_was() {
        let report = diagnose(&old_driver_only_exam(), "");
        let json = serde_json::to_value(&report).expect("serialises");
        assert_eq!(json["has_match"], false);
        assert_eq!(json["matched"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["matched"][0]["id"], "fix-22-display-driver-old");
        assert!(json["route_when_no_match"]["url"].is_string());
    }

    #[test]
    fn a_current_display_driver_is_not_flagged() {
        assert!(old_driver_entry(&windows_driver_exam("32.0.23033.1002", "2026-05-02")).is_none());
        // Exactly the cutoff day is not older than it.
        assert!(old_driver_entry(&windows_driver_exam("32.0.1.1", "2023-07-27")).is_none());
    }

    #[test]
    fn a_missing_or_unreadable_display_driver_date_never_flags() {
        for (version, date) in [
            ("30.0.13014.8", ""),
            ("", "2021-08-04"),
            ("", ""),
            ("30.0.13014.8", "garbage"),
        ] {
            assert!(
                old_driver_entry(&windows_driver_exam(version, date)).is_none(),
                "{version:?} {date:?}"
            );
        }
    }

    #[test]
    fn the_old_display_driver_entry_never_applies_off_windows() {
        for family in ["linux", "other"] {
            let mut e = windows_driver_exam("30.0.13014.8", "2021-08-04");
            e.os_family = family.to_owned();
            assert!(old_driver_entry(&e).is_none(), "{family}");
        }
        let mut wsl = wsl_base();
        wsl.adrenalin_version = "30.0.13014.8".to_owned();
        wsl.adrenalin_date = "2021-08-04".to_owned();
        assert!(old_driver_entry(&wsl).is_none());
    }

    #[test]
    fn the_old_display_driver_entry_does_not_change_fix_14() {
        // fix-14 keeps its meaning: an old date alone must not raise it.
        let report = diagnose(&windows_driver_exam("30.0.13014.8", "2021-08-04"), "");
        assert!(
            report
                .matched
                .iter()
                .all(|d| d.id != "fix-14-adrenalin-too-old")
        );
        assert!(!report.has_match());
    }
}
