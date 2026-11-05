// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The checks that apply on WSL2 only.

use super::scoring::{KEYWORDS_WSL_LOADER, KEYWORDS_WSL_NO_DEVICE, finalize, keyword_score, zero};
use super::{Diagnosis, Fix};
use crate::examine::{Examination, WslFacts};

// ---------------------------------------------------------------------------
// WSL2 catalog
//
// A parallel catalog, not a port of the bare-metal one. WSL2 reaches the GPU
// through /dev/dxg and the Windows host driver (dxgkrnl), so the questions worth
// asking are about the DXCore handoff, the ROCDXG userspace, and the host — none
// of which the bare-metal checks know anything about.
// ---------------------------------------------------------------------------

/// The WSL facts, or a default set when the probe did not populate them.
///
/// A WSL host whose `wsl` section is missing is a probe failure, not a healthy
/// machine, so every field reads false and the checks fire on the missing
/// plumbing rather than silently passing.
fn wsl_facts(e: &Examination) -> WslFacts {
    e.wsl.clone().unwrap_or_default()
}

pub(super) fn check_wsl_1_gpu_not_exposed(e: &Examination, symptom: &str) -> Diagnosis {
    let w = wsl_facts(e);
    if w.dxg_device {
        return zero("fix-wsl-1-gpu-not-exposed", "GPU not exposed to the distro");
    }
    // WSL 1 has no GPU path at all; fix-wsl-7 says so in terms the user can act
    // on, and two findings for one cause is noise.
    if w.version == 1 {
        return zero("fix-wsl-1-gpu-not-exposed", "GPU not exposed to the distro");
    }
    let mut score = 55;
    let mut evidence = vec!["/dev/dxg is missing, so the distro has no GPU path".to_owned()];
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_WSL_NO_DEVICE);
    score += kw_score;
    evidence.extend(kw_ev);

    // Three different causes produce the same missing device, and they need
    // different actions. Naming which one this is spares the user from updating a
    // Windows driver that was never the problem.
    let (cause, commands, notes) = if e.in_container {
        (
            "this is a container, and containers only see /dev/dxg when it is passed in",
            vec![
                "# Re-run the container with the WSL GPU device and libraries:".to_owned(),
                "#   --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl".to_owned(),
                "# and add /usr/lib/wsl/lib to the loader path inside it.".to_owned(),
            ],
            vec![
                "The Windows host driver is probably fine here: the device is missing because this container was not given it, not because the host lacks GPU support.".to_owned(),
            ],
        )
    } else if !w.wsl_lib_dir {
        (
            "/usr/lib/wsl is absent too, so this distro never had WSL GPU support wired in",
            vec![
                "# Update WSL itself, then restart the distro from Windows:".to_owned(),
                "#   wsl --update".to_owned(),
                "#   wsl --shutdown".to_owned(),
            ],
            Vec::new(),
        )
    } else {
        (
            "/usr/lib/wsl is present but the device is not, which points at the Windows host driver or the WSL kernel",
            vec![
                "# On the Windows host: install a WSL-capable AMD Adrenalin driver,".to_owned(),
                "# then update the WSL kernel and restart the distro:".to_owned(),
                "#   wsl --update".to_owned(),
                "#   wsl --shutdown".to_owned(),
            ],
            vec![format!("Driver and WSL setup steps: {WSL_DOCS_URL}")],
        )
    };
    evidence.push(cause.to_owned());

    let fix = Fix {
        summary: "Expose the GPU to the distro: /dev/dxg is how WSL reaches it, and nothing works until it is there.".to_owned(),
        commands,
        fix_id: "fix-wsl-1-gpu-not-exposed".to_owned(),
        verify: "ls -l /dev/dxg".to_owned(),
        notes,
        ..Fix::default()
    };
    finalize(
        "fix-wsl-1-gpu-not-exposed",
        "GPU not exposed to the distro (/dev/dxg missing)",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_wsl_2_dxcore_missing(e: &Examination, symptom: &str) -> Diagnosis {
    let w = wsl_facts(e);
    // Without the device there is nothing for DXCore to talk to; fix-wsl-1 is the
    // cause and this would only add a second finding for it.
    if w.dxcore || !w.dxg_device {
        return zero("fix-wsl-2-dxcore-missing", "WSL DXCore libraries missing");
    }
    let mut score = 50;
    let mut evidence =
        vec!["/usr/lib/wsl/lib/libdxcore.so is missing, so the ROCm runtime cannot reach the host driver".to_owned()];
    if !w.wsl_lib_dir {
        score += 15;
        evidence.push("/usr/lib/wsl/lib does not exist at all".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_WSL_LOADER);
    score += kw_score;
    evidence.extend(kw_ev);

    let fix = Fix {
        summary: "Restore the WSL DXCore libraries, then make sure they are on the loader path."
            .to_owned(),
        commands: vec![
            "# From Windows, refresh the WSL runtime that ships these libraries:".to_owned(),
            "#   wsl --update".to_owned(),
            "#   wsl --shutdown".to_owned(),
            "# Inside the distro, confirm the loader can see them:".to_owned(),
            "echo /usr/lib/wsl/lib | sudo tee /etc/ld.so.conf.d/wsl.conf".to_owned(),
            "sudo ldconfig".to_owned(),
        ],
        needs_sudo: true,
        fix_id: "fix-wsl-2-dxcore-missing".to_owned(),
        verify: "ls -l /usr/lib/wsl/lib/libdxcore.so && ldconfig -p | grep libdxcore".to_owned(),
        notes: vec![
            "/usr/lib/wsl is mounted by WSL itself, not installed by the distro's package manager, so apt cannot repair it -- the fix is on the Windows side.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-wsl-2-dxcore-missing",
        "WSL DXCore libraries missing or not on the loader path",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_wsl_3_rocdxg_missing(e: &Examination, symptom: &str) -> Diagnosis {
    let w = wsl_facts(e);
    if w.librocdxg || !w.dxg_device {
        return zero("fix-wsl-3-rocdxg-missing", "ROCDXG not installed");
    }
    let mut score = 50;
    let mut evidence = vec!["librocdxg.so was not found under any ROCm install".to_owned()];
    if w.dxcore {
        // The host side is ready and only the distro-side package is absent, which
        // is both the most common case and the one the user can fix alone.
        score += 15;
        evidence.push(
            "the WSL DXCore handoff is present, so only the distro-side ROCDXG is missing"
                .to_owned(),
        );
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_WSL_LOADER);
    score += kw_score;
    evidence.extend(kw_ev);

    let fix = Fix {
        summary: "Install ROCDXG inside the distro: it is the ROCm-to-DXCore shim the WSL path runs on.".to_owned(),
        commands: vec![
            "rocm install driver".to_owned(),
            "# Then, once the plan looks right:".to_owned(),
            "#   rocm install driver --yes".to_owned(),
        ],
        needs_sudo: true,
        fix_id: "fix-wsl-3-rocdxg-missing".to_owned(),
        verify: "ldconfig -p | grep librocdxg".to_owned(),
        notes: vec![
            "This downloads and installs a .deb with sudo, so `rocm fix` prints it rather than running it. `rocm install driver` shows the full plan, and checks the download against a digest pinned for that ROCDXG release.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-wsl-3-rocdxg-missing",
        "ROCDXG not installed in the distro",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_wsl_4_rocdxg_not_linked(e: &Examination, symptom: &str) -> Diagnosis {
    let w = wsl_facts(e);
    // Only meaningful once the library is actually on disk: when it is not,
    // fix-wsl-3 is the finding and the missing linker entry is a consequence.
    //
    // The device guard matches its siblings. librocdxg can be installed while
    // /dev/dxg is absent, and running `ldconfig` fixes nothing then -- offering
    // it beside the real cause just leaves the user to guess which to act on.
    // `Some(false)` only: `None` means ldconfig could not be run, and an
    // unreadable linker cache is not an unregistered library.
    if !w.librocdxg || w.ldconfig_librocdxg != Some(false) || !w.dxg_device {
        return zero(
            "fix-wsl-4-rocdxg-not-linked",
            "ROCDXG installed but not on the loader path",
        );
    }
    let mut score = 55;
    let mut evidence = vec![
        "librocdxg.so is installed but does not appear in `ldconfig -p`, so the runtime will not load it".to_owned(),
    ];
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_WSL_LOADER);
    score += kw_score;
    evidence.extend(kw_ev);

    let fix = Fix {
        summary: "Refresh the linker cache so the installed ROCDXG becomes loadable.".to_owned(),
        commands: vec!["sudo ldconfig".to_owned()],
        needs_sudo: true,
        fix_id: "fix-wsl-4-rocdxg-not-linked".to_owned(),
        verify: "ldconfig -p | grep librocdxg".to_owned(),
        notes: vec![
            "If `ldconfig` alone does not fix it, the install went somewhere outside the linker's search path: add that directory under /etc/ld.so.conf.d/ and re-run.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-wsl-4-rocdxg-not-linked",
        "ROCDXG installed but invisible to the dynamic linker",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_wsl_5_distro_too_old(e: &Examination, _symptom: &str) -> Diagnosis {
    let w = wsl_facts(e);
    // `None` means the release could not be read. That is not evidence of an old
    // distro, so this stays silent rather than guessing.
    if w.distro_supported != Some(false) {
        return zero("fix-wsl-5-distro-too-old", "Distro release below the floor");
    }
    let (major, minor) = crate::examine::WSL_MIN_UBUNTU;
    let evidence = vec![format!(
        "distro is {} {}, below the {major}.{minor:02} floor the WSL path requires",
        e.distro_id, e.distro_version
    )];
    let fix = Fix {
        summary: format!(
            "Move to Ubuntu {major}.{minor:02} or newer: older releases ship a glibc the engines cannot run against."
        ),
        commands: vec![
            "# From Windows, install a supported distro alongside the current one:".to_owned(),
            "#   wsl --install -d Ubuntu-24.04".to_owned(),
        ],
        fix_id: "fix-wsl-5-distro-too-old".to_owned(),
        verify: "f=/etc/os-release; [ -e \"$f\" ] || f=/usr/lib/os-release; grep VERSION_ID \"$f\" 2>/dev/null | head -1"
            .to_owned(),
        notes: vec![
            "This is a hard floor, not a recommendation: Ubuntu 22.04 ships glibc 2.35, below the glibc 2.38 / GLIBCXX_3.4.32 that every published Lemonade embeddable is linked against, so the engine cannot start there at all.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-wsl-5-distro-too-old",
        "Distro release is below the supported floor for WSL",
        70,
        evidence,
        fix,
    )
}

pub(super) fn check_wsl_6_host_driver_too_old(e: &Examination, symptom: &str) -> Diagnosis {
    let w = wsl_facts(e);
    // Abstain unless the host was actually reached. Treating "could not ask" as
    // "driver is old" would fire on every host with interop switched off and on
    // every container, where the question is unanswerable rather than answered.
    //
    // WSL 1 abstains too: it has no GPU path for any driver to serve, so the
    // host's driver version cannot be the reason anything failed. Without this
    // the symptom keywords alone were enough to raise it alongside fix-wsl-7,
    // pointing the user at a Windows driver that could never have helped.
    if !w.host_reachable || w.version == 1 {
        return zero(
            "fix-wsl-6-host-driver-too-old",
            "Windows host GPU driver too old",
        );
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    match w.host_driver_version.as_deref() {
        // Guarded on the device like its siblings: without /dev/dxg, fix-wsl-1
        // is the finding, and adding a second one for the same root cause just
        // leaves the user choosing between them.
        Some("") if w.dxg_device => {
            score += 45;
            evidence.push(
                "the Windows host reports no AMD display adapter, so no WSL-capable AMD driver is installed there"
                    .to_owned(),
            );
        }
        // The plumbing is present but the runtime still cannot see a GPU, which
        // on a machine with a working DXCore handoff points at the host driver.
        //
        // `rocm_sees_gpu`, not `has_amd_gpu`: the probes that populate the latter
        // are skipped on WSL, so it reads false on every host here, healthy or
        // not, and this check fired on a complete working stack. `Some(false)`
        // specifically -- `None` means rocminfo was absent so the question went
        // unasked, which is not evidence of anything.
        //
        // Guarded on the linker cache like fix-wsl-4: when `ldconfig` positively
        // shows librocdxg is not registered, that is the root cause and this
        // check must not also fire for the same symptom -- the user would be
        // left choosing between "run ldconfig" and "update the host driver" for
        // a fault that only the former explains.
        Some(version)
            if w.dxg_device
                && w.dxcore
                && w.librocdxg
                && w.ldconfig_librocdxg != Some(false)
                && w.rocm_sees_gpu == Some(false) =>
        {
            score += 40;
            evidence.push(format!(
                "host AMD display driver {version} is installed and the distro plumbing is complete, but rocminfo enumerates no GPU"
            ));
        }
        Some(_) | None => {}
    }

    // Symptom keywords only ever CORROBORATE machine evidence here; they cannot
    // stand alone. Every other WSL entry either carries a base score from a fact
    // or returns before scoring keywords, so this was the one place where the
    // words a user typed could clear the threshold by themselves -- and the
    // table it shares with fix-wsl-1 is generic enough ("/dev/dxg", "no
    // HIP-capable device") that an ordinary description of any GPU failure hit
    // 85 on a completely healthy machine, outranking the check that had found
    // the real fault. The WSL-1 guard above was one instance of this; the floor
    // is the general fix.
    if score <= 0 {
        return zero(
            "fix-wsl-6-host-driver-too-old",
            "Windows host GPU driver too old",
        );
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_WSL_NO_DEVICE);
    score += kw_score;
    evidence.extend(kw_ev);
    let fix = Fix {
        summary: "Update the AMD driver on the Windows host: the GPU driver WSL uses lives there, not in the distro.".to_owned(),
        commands: vec![
            "# On the Windows host, not in this distro:".to_owned(),
            "#   install a WSL-capable AMD Adrenalin driver, then `wsl --shutdown`.".to_owned(),
        ],
        fix_id: "fix-wsl-6-host-driver-too-old".to_owned(),
        verify: "rocminfo | head -n 20".to_owned(),
        notes: vec![
            format!("Driver and ROCm version pairing: {WSL_DOCS_URL}"),
            "Nothing inside the distro can carry this out -- `rocm fix` prints the steps because the change belongs to the Windows host.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-wsl-6-host-driver-too-old",
        "Windows host AMD driver missing or too old for ROCm on WSL",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_wsl_7_wsl1(e: &Examination, _symptom: &str) -> Diagnosis {
    if wsl_facts(e).version != 1 {
        return zero("fix-wsl-7-wsl1", "Distro is running under WSL 1");
    }
    let evidence = vec![format!(
        "kernel '{}' is a WSL 1 kernel; WSL 1 translates syscalls and exposes no GPU device at all",
        e.kernel_release
    )];
    let fix = Fix {
        summary: "Convert the distro to WSL 2: WSL 1 has no GPU path, so no amount of driver or package work will help.".to_owned(),
        commands: vec![
            "# From Windows PowerShell:".to_owned(),
            "#   wsl --set-version <distro> 2".to_owned(),
            "#   wsl --set-default-version 2".to_owned(),
        ],
        fix_id: "fix-wsl-7-wsl1".to_owned(),
        verify: "uname -r".to_owned(),
        notes: vec![
            "Converting rewrites the distro's filesystem and can take a while on a large install; back up anything you cannot lose first.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-wsl-7-wsl1",
        "Distro is running under WSL 1, which has no GPU support",
        80,
        evidence,
        fix,
    )
}

/// ROCm-on-WSL2 setup guidance (distinct from the bare-metal catalog).
const WSL_DOCS_URL: &str = "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html";

#[cfg(test)]
mod tests {
    use crate::diagnose::test_support::wsl_base;
    use crate::diagnose::{HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH, diagnose};

    #[test]
    fn a_healthy_wsl_host_gets_no_diagnosis() {
        // Scenario 2: the platform is covered, so `out_of_scope` stays clear, and
        // a symptom the catalog does not recognise must not manufacture a cause.
        let report = diagnose(&wsl_base(), "the model output looks wrong");
        assert!(
            report.out_of_scope.is_none(),
            "WSL2 is a covered platform now"
        );
        // Emptiness, not just `!has_match()`. A sub-threshold finding is still
        // printed and still tells the user something is wrong with their machine,
        // so a healthy host must produce NO entries at all. Asserting only on the
        // verdict let a permanent score-40 false positive through: the host-driver
        // check read a field the WSL probe never populates.
        assert!(
            report.matched.is_empty(),
            "a healthy WSL host must produce no findings at all, got: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
        assert!(!report.has_match());
        assert!(!report.route_when_no_match.url.is_empty());
    }

    #[test]
    fn an_unlinked_rocdxg_is_not_reported_when_the_device_is_missing() {
        // librocdxg can be installed while /dev/dxg is absent. Running `ldconfig`
        // then fixes nothing, and offering it alongside the real cause leaves the
        // user to guess which to act on -- the same reason the DXCore and ROCDXG
        // checks already stand down without the device.
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.dxg_device = false;
        w.ldconfig_librocdxg = Some(false);
        let report = diagnose(&e, "");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["fix-wsl-1-gpu-not-exposed"], "ids: {ids:?}");
    }

    #[test]
    fn a_missing_dxg_device_is_the_top_finding() {
        let mut e = wsl_base();
        e.wsl.as_mut().expect("wsl facts").dxg_device = false;
        let report = diagnose(&e, "no ROCm-capable device is detected");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-1-gpu-not-exposed");
        assert!(top.score >= HIGH_CONFIDENCE, "score {}", top.score);
    }

    #[test]
    fn a_container_without_the_device_is_not_blamed_on_the_windows_driver() {
        // A container on WSL2 reports itself as WSL but only sees /dev/dxg when
        // it was started with it. Telling that user to update a Windows driver
        // sends them to fix a machine that was never broken.
        let mut e = wsl_base();
        e.wsl.as_mut().expect("wsl facts").dxg_device = false;
        e.in_container = true;
        e.container_kind = "docker".to_owned();
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-1-gpu-not-exposed");
        assert!(
            top.evidence.iter().any(|line| line.contains("container")),
            "evidence must name the container as the reason: {:?}",
            top.evidence
        );
        let fix = top.fix.as_ref().expect("carries a fix");
        assert!(
            fix.notes
                .iter()
                .any(|n| n.contains("host driver is probably fine")),
            "must say the Windows driver is not the suspect: {:?}",
            fix.notes
        );
    }

    #[test]
    fn only_the_root_cause_of_a_broken_stack_is_reported() {
        // With no device, the missing DXCore shim and the missing ROCDXG package
        // are consequences, not causes. Reporting all three would leave the user
        // to guess which one to act on.
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.dxg_device = false;
        w.dxcore = false;
        w.librocdxg = false;
        w.ldconfig_librocdxg = Some(false);
        let report = diagnose(&e, "");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["fix-wsl-1-gpu-not-exposed"], "ids: {ids:?}");
    }

    #[test]
    fn a_missing_rocdxg_package_is_reported_when_the_host_side_is_ready() {
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.librocdxg = false;
        w.ldconfig_librocdxg = Some(false);
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-3-rocdxg-missing");
        let fix = top.fix.as_ref().expect("carries a fix");
        assert!(
            !fix.auto_applicable,
            "installing a .deb with sudo must stay print-only"
        );
        // Verification is no longer something the user has to remember to turn
        // on: `rocm install driver` pins a digest per release. The note has to
        // say so, because a print-only recipe is all the user sees here.
        assert!(
            fix.notes.iter().any(|n| n.contains("digest")),
            "must state that the download is verified: {:?}",
            fix.notes
        );
        assert!(
            fix.commands
                .iter()
                .any(|c| c.contains("rocm install driver")),
            "must route to the command that carries the plan: {:?}",
            fix.commands
        );
    }

    #[test]
    fn an_installed_but_unlinked_rocdxg_is_a_distinct_finding() {
        let mut e = wsl_base();
        e.wsl.as_mut().expect("wsl facts").ldconfig_librocdxg = Some(false);
        let report = diagnose(&e, "librocdxg.so: cannot open shared object file");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-4-rocdxg-not-linked");
    }

    #[test]
    fn an_unlinked_rocdxg_does_not_also_raise_the_host_driver_finding() {
        // Same fault as above, but with `rocm_sees_gpu = Some(false)` added so
        // this machine also satisfies every other condition
        // check_wsl_6_host_driver_too_old's second arm checks (dxg_device,
        // dxcore, librocdxg, rocminfo seeing no GPU). Without the ldconfig
        // guard on that arm, it fires alongside fix-wsl-4 for the same
        // underlying fault, leaving the user to guess between "run ldconfig"
        // and "update the Windows host driver" when only the former is true.
        //
        // The symptom text also carries a generic "cannot open shared object
        // file" keyword that fix-8-wheel-rocm scores on regardless of WSL
        // state -- that overlap is real and expected, so this only asserts on
        // the two WSL findings that fix #2's guard actually governs.
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.ldconfig_librocdxg = Some(false);
        w.rocm_sees_gpu = Some(false);
        let report = diagnose(&e, "librocdxg.so: cannot open shared object file");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert!(
            ids.contains(&"fix-wsl-4-rocdxg-not-linked"),
            "the unlinked-library finding must still fire: ids: {ids:?}"
        );
        assert!(
            !ids.contains(&"fix-wsl-6-host-driver-too-old"),
            "the host-driver-too-old finding must not overlap with the unlinked-library \
             finding on the same fault: ids: {ids:?}"
        );
    }

    #[test]
    fn a_distro_below_the_floor_is_reported() {
        let mut e = wsl_base();
        e.distro_version = "22.04".to_owned();
        e.wsl.as_mut().expect("wsl facts").distro_supported = Some(false);
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-5-distro-too-old");
        assert!(
            top.evidence[0].contains("22.04"),
            "evidence must name the release found: {:?}",
            top.evidence
        );
    }

    #[test]
    fn an_unreadable_distro_release_is_not_reported_as_too_old() {
        // `None` means the release could not be parsed. That is not evidence of
        // an old distro, and a finding here would send the user to reinstall a
        // perfectly supported one.
        let mut e = wsl_base();
        e.wsl.as_mut().expect("wsl facts").distro_supported = None;
        let report = diagnose(&e, "");
        assert!(
            !report
                .matched
                .iter()
                .any(|d| d.id == "fix-wsl-5-distro-too-old"),
            "must not guess: {:?}",
            report.matched
        );
    }

    #[test]
    fn a_typed_symptom_alone_never_blames_the_windows_host_driver() {
        // The host-driver check shares a generic keyword table with fix-wsl-1
        // ("/dev/dxg", "no HIP-capable device"), and unlike every other WSL entry
        // it has no base score from a fact. Scoring keywords before checking that
        // anything was actually measured let an ordinary description of a GPU
        // problem reach 85 -- HIGH confidence -- on a fully healthy machine,
        // with both evidence lines being the user's own words.
        let symptom = "no hip-capable device found, see /dev/dxg";
        let report = diagnose(&wsl_base(), symptom);
        assert!(
            report.matched.is_empty(),
            "a healthy host must stay silent whatever the user typed: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );

        // And the real cause must outrank a keyword-only guess. With ROCDXG
        // missing, the driver check previously scored 85 against fix-wsl-3's 65
        // and sent the user to reinstall a current driver.
        let mut broken = wsl_base();
        let w = broken.wsl.as_mut().expect("wsl facts");
        w.librocdxg = false;
        w.ldconfig_librocdxg = Some(false);
        let report = diagnose(&broken, symptom);
        assert_eq!(
            report.matched[0].id,
            "fix-wsl-3-rocdxg-missing",
            "the measured fault must rank first: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_unreadable_linker_cache_is_not_an_unregistered_library() {
        // `ldconfig` lives in /sbin, off a non-root user's PATH on Debian. When
        // it cannot be run the cache is unknown, not empty -- reading it as empty
        // told users with a correctly installed ROCDXG to re-run `ldconfig`.
        let mut e = wsl_base();
        e.wsl.as_mut().expect("wsl facts").ldconfig_librocdxg = None;
        let report = diagnose(&e, "");
        assert!(
            !report
                .matched
                .iter()
                .any(|d| d.id == "fix-wsl-4-rocdxg-not-linked"),
            "unknown must not be reported as not-linked: {:?}",
            report.matched
        );
    }

    #[test]
    fn a_host_with_no_amd_adapter_is_not_reported_twice_with_the_device_missing() {
        // Without /dev/dxg, fix-wsl-1 is the cause. The driver check's
        // no-adapter arm lacked the device guard its siblings carry, so both
        // fired for one root cause.
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.dxg_device = false;
        w.host_driver_version = Some(String::new());
        let report = diagnose(&e, "");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["fix-wsl-1-gpu-not-exposed"], "ids: {ids:?}");
    }

    #[test]
    fn a_missing_dxcore_shim_is_reported_when_the_device_is_present() {
        // The only WSL entry with no fire-case test: both existing tests that
        // clear `dxcore` also clear `dxg_device`, so they exercised the abstain
        // path and an inverted condition here would have passed CI.
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.dxcore = false;
        w.wsl_lib_dir = false;
        let report = diagnose(&e, "libdxcore.so: cannot open shared object file");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-2-dxcore-missing");
        assert!(top.score >= MIN_SCORE_FOR_MATCH, "score {}", top.score);
    }

    #[test]
    fn an_unreachable_windows_host_never_blames_the_host_driver() {
        // Scenario 6. Interop is off or this is a container, so the host driver
        // is unknown -- and unknown must not read as "too old".
        let mut e = wsl_base();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.host_reachable = false;
        w.host_driver_version = None;
        let report = diagnose(&e, "no ROCm-capable device is detected");
        assert!(
            !report
                .matched
                .iter()
                .any(|d| d.id == "fix-wsl-6-host-driver-too-old"),
            "the check must abstain when the host was never asked: {:?}",
            report.matched
        );
    }

    #[test]
    fn a_host_with_no_amd_adapter_is_reported() {
        let mut e = wsl_base();
        e.wsl.as_mut().expect("wsl facts").host_driver_version = Some(String::new());
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-wsl-6-host-driver-too-old");
        let fix = top.fix.as_ref().expect("carries a fix");
        assert!(
            fix.notes
                .iter()
                .any(|n| n.contains("Nothing inside the distro")),
            "must say the remedy belongs to the Windows host: {:?}",
            fix.notes
        );
    }

    #[test]
    fn wsl1_is_reported_instead_of_a_missing_device() {
        // WSL 1 has no GPU path at all, so "install a driver" is the wrong advice
        // and fix-wsl-1 must stand down in favour of the conversion.
        let mut e = wsl_base();
        e.kernel_release = "4.4.0-19041-Microsoft".to_owned();
        let w = e.wsl.as_mut().expect("wsl facts");
        w.version = 1;
        w.dxg_device = false;
        w.dxcore = false;
        w.wsl_lib_dir = false;
        w.librocdxg = false;
        w.ldconfig_librocdxg = Some(false);
        let report = diagnose(&e, "no ROCm-capable device is detected");
        let ids: Vec<&str> = report.matched.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["fix-wsl-7-wsl1"], "ids: {ids:?}");
    }
}
