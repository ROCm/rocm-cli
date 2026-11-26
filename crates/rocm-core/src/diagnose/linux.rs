// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The checks that apply on Linux only: the kernel and the amdgpu module, device
//! node access, the package repositories and the repo-native install, and the
//! helpers that pick which package manager's commands to print.

use super::scoring::{
    KEYWORDS_APU_UNIFIED_MEMORY, KEYWORDS_CONTAINER, KEYWORDS_DPKG_BROKEN, KEYWORDS_IOMMU_HANG,
    KEYWORDS_KFD_PERMISSION, KEYWORDS_MODULE_NOT_LOADED, KEYWORDS_REPO_BROKEN, amd_gpu_count,
    finalize, keyword_score, zero,
};
use super::{Diagnosis, Fix};
use crate::examine::Examination;

/// The phrase `check_3_rocm_kernel_unsupported` puts in its evidence when
/// `amdgpu_loaded` is `Some(false)` and ROCm is installed.
const AMDGPU_KERNEL_MODULE_NOT_LOADED: &str = "the amdgpu kernel module is not loaded";
/// The evidence line `check_5_amdgpu_blacklisted` adds when `amdgpu_loaded` is
/// `Some(false)`.
const AMDGPU_MODULE_NOT_LOADED: &str = "amdgpu module is not loaded";
/// The start of the evidence line `check_5_amdgpu_blacklisted` adds when Secure
/// Boot is enabled and `amdgpu_loaded` is `Some(false)`.
const SECURE_BOOT_BLOCKED_AMDGPU: &str = "Secure Boot is enabled and amdgpu didn't load";

/// One phrase per evidence line that only `amdgpu_loaded == Some(false)` produces.
///
/// The checks build their lines from these same constants, so a test that asserts
/// a diagnosis cites none of them (the e2e scenario for a built-in `amdgpu`)
/// cannot drift from the wording and pass vacuously.
pub const AMDGPU_MISSING_EVIDENCE: [&str; 3] = [
    AMDGPU_KERNEL_MODULE_NOT_LOADED,
    AMDGPU_MODULE_NOT_LOADED,
    SECURE_BOOT_BLOCKED_AMDGPU,
];

pub(super) fn check_3_rocm_kernel_unsupported(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let kernel = &e.kernel_release;
    let distro = &e.distro_id;
    let distro_v = &e.distro_version;
    let rocm_version = &e.rocm_version;

    if !rocm_version.is_empty() && e.amdgpu_loaded == Some(false) {
        score += 30;
        evidence.push(format!(
            "ROCm {rocm_version} is installed but {AMDGPU_KERNEL_MODULE_NOT_LOADED}; this is typical when DKMS failed against an unsupported kernel."
        ));
    }

    let (_kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_DPKG_BROKEN);
    if kw_ev.iter().any(|l| l.to_lowercase().contains("dkms")) {
        score += 30;
        evidence.extend(kw_ev);
    }

    if score <= 0 {
        return zero("fix-3-rocm-kernel", "ROCm/distro/kernel triple unsupported");
    }
    let mut commands = vec![
        format!("# Current: kernel={kernel} distro={distro} {distro_v} rocm={rocm_version}"),
        "# Compare to the live AMD matrix:".to_owned(),
        "#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html".to_owned(),
    ];
    if dkms_driver_plan_is_supported(e) {
        commands.push("# If your kernel is above the supported range, install the HWE".to_owned());
        commands.push(
            "# kernel that matches ROCm, then rerun `rocm install driver --dkms --yes`.".to_owned(),
        );
    } else {
        commands.push(format!(
            "# `rocm install driver --dkms` plans nothing on {distro} {distro_v}: it isn't one of the AMD-documented distros/versions above, so rerunning it won't fix this."
        ));
    }
    let fix = Fix {
        summary: "Cross-check your kernel/distro against the live AMD compatibility matrix before reinstalling.".to_owned(),
        commands,
        fix_id: "fix-3-rocm-kernel".to_owned(),
        needs_reboot: true,
        // Check rocminfo's own exit code: piping through `head` would report
        // the pipeline's last stage and mask a missing/failing `rocminfo`.
        verify: "lsmod | grep amdgpu && rocminfo >/dev/null 2>&1".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-3-rocm-kernel",
        "ROCm version + distro/kernel form an unsupported triple",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_4_render_group(e: &Examination, symptom: &str) -> Diagnosis {
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_KFD_PERMISSION);
    // The user can already open the device, so group membership is not what
    // stands between them and the GPU. Only a pasted error that names a
    // permission failure on the device keeps the entry: then the examination
    // and the error disagree, and the error is the stronger evidence.
    if kw_score == 0
        && e.kfd
            .as_ref()
            .is_some_and(|kfd| kfd.exists && kfd.user_can_write == Some(true))
    {
        return zero("fix-4-render-group", "User missing render/video group");
    }
    let mut score = 0;
    let mut evidence = Vec::new();
    if e.in_render_group == Some(false) {
        score += 35;
        evidence.push("user is NOT in the 'render' group".to_owned());
    }
    if e.in_video_group == Some(false) {
        score += 10;
        evidence.push("user is NOT in the 'video' group".to_owned());
    }
    if let Some(kfd) = &e.kfd
        && kfd.exists
        && kfd.user_can_write == Some(false)
    {
        score += 25;
        evidence.push(format!(
            "/dev/kfd exists (mode {}, group {}) but the current user can't write to it",
            kfd.mode, kfd.owner_group
        ));
    }
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-4-render-group", "User missing render/video group");
    }
    let owner_group = e.kfd.as_ref().map_or("", |k| k.owner_group.as_str());
    let mut fix = render_group_plan(owner_group, e.groups_on_host.as_deref());
    fix.fix_id = "fix-4-render-group".to_owned();
    // With no group to name there is nothing to run, no re-login to wait for
    // and nothing to verify: the flags line then says `rocm fix` changes
    // nothing (it exits 3 with the reason), not that it can run the command.
    if !fix.unavailable_here {
        fix.needs_sudo = true;
        fix.needs_relogin = true;
        fix.verify = crate::fix::RENDER_GROUP_VERIFY.to_owned();
        fix.notes = vec![
            "Group membership only takes effect after a full re-login (or reboot). `newgrp <group>` gives the current shell access but not other terminals or services.".to_owned(),
        ];
    }
    finalize(
        "fix-4-render-group",
        "User not in render/video group (or /dev/kfd owned by the other group)",
        score,
        evidence,
        fix,
    )
}

/// The summary and the command of fix-4's plan, from the group that owns
/// `/dev/kfd` and the groups the host's database names (`None` when it was
/// not read).
///
/// The groups come from `crate::fix::render_group_targets` and the arguments
/// from `crate::fix::usermod_args`, which `rocm fix` builds its own `usermod`
/// from, so the line printed here is the one the fix runs (held by
/// `the_render_group_plan_diagnose_prints_is_the_one_fix_runs`). When no
/// group can be named the plan carries no command, only the reason: a
/// `usermod` naming a group the host does not have fails, and `stat` prints
/// `UNKNOWN` for a group it cannot name.
fn render_group_plan(owner_group: &str, groups_on_host: Option<&[String]>) -> Fix {
    let targets = crate::fix::render_group_targets(owner_group, |group| {
        groups_on_host.is_some_and(|known| known.iter().any(|g| g == group))
    });
    if targets.is_empty() {
        let reason = crate::fix::no_target_reason(
            owner_group,
            crate::fix::PlanReader::Examination {
                database_read: groups_on_host.is_some(),
            },
        );
        return Fix {
            summary: format!(
                "No group to add the current user to: {reason}. Give the user access to /dev/kfd another way, or create the group, then re-run `rocm diagnose`."
            ),
            commands: vec![
                format!("# {reason}."),
                "# Check with: ls -l /dev/kfd; getent group render video".to_owned(),
            ],
            // Only when the host's groups were read: with `None` the diagnosis
            // cannot tell, and `rocm fix` reads them live and may well run.
            unavailable_here: groups_on_host.is_some(),
            ..Fix::default()
        };
    }
    let groups = targets.join(",");
    let named = targets
        .iter()
        .map(|g| format!("'{g}'"))
        .collect::<Vec<_>>()
        .join(" and ");
    Fix {
        summary: format!("Add the current user to {named} and log out/in."),
        commands: vec![format!(
            "sudo usermod {}",
            crate::fix::usermod_args(&groups, "\"$USER\"").join(" ")
        )],
        ..Fix::default()
    }
}

pub(super) fn check_5_amdgpu_blacklisted(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let blacklisted = &e.amdgpu_blacklisted_in;
    if !blacklisted.is_empty() {
        score += 55;
        evidence.push(format!("amdgpu is blacklisted in: {blacklisted:?}"));
    }
    if e.amdgpu_loaded == Some(false) {
        score += 35;
        evidence.push(AMDGPU_MODULE_NOT_LOADED.to_owned());
    }
    if e.rocminfo_status == "not-loaded" {
        score += 25;
        evidence.push("rocminfo says 'ROCk module is NOT loaded'".to_owned());
    }
    if e.secure_boot == "enabled" && e.amdgpu_loaded == Some(false) {
        score += 10;
        evidence.push(format!("{SECURE_BOOT_BLOCKED_AMDGPU} -- DKMS modules are often blocked until you sign them or disable Secure Boot."));
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_MODULE_NOT_LOADED);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-5-amdgpu-load", "amdgpu not loaded");
    }
    let mut commands = Vec::new();
    if !blacklisted.is_empty() {
        for f in blacklisted {
            commands.push(format!(
                "# Inspect & remove the blacklist line: sudo $EDITOR {f}"
            ));
        }
        commands.push("sudo update-initramfs -u   # Debian/Ubuntu".to_owned());
        commands.push("sudo dracut -f             # Fedora/RHEL".to_owned());
    }
    commands.push("sudo modprobe amdgpu".to_owned());
    if e.secure_boot == "enabled" {
        commands.push("# Secure Boot is on; if amdgpu still won't load, the DKMS module isn't signed. Sign it (mokutil) or disable Secure Boot.".to_owned());
    }
    let fix = Fix {
        summary: "Remove amdgpu from any modprobe blacklist and load it.".to_owned(),
        commands,
        needs_sudo: true,
        // Catalog-aligned, not state-derived: fix/catalog.rs's FixRecipe for this fix-id sets
        // needs_reboot unconditionally. On this no-blacklist path the plan is just
        // `modprobe amdgpu` (or that plus a Secure Boot signing note, whose own
        // remedy can require a reboot) -- diagnose conforms to the catalog rather
        // than the reverse (see fix/catalog.rs's assert_needs_reboot_matches_the_catalog) so
        // `rocm diagnose` and `rocm fix` never disagree; over-warning is limited to
        // the plain no-Secure-Boot sub-case, judged cheaper than the drift it replaces.
        needs_reboot: true,
        fix_id: "fix-5-amdgpu-load".to_owned(),
        verify: "lsmod | grep amdgpu && rocminfo | head -n 5".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-5-amdgpu-load",
        "amdgpu kernel module not loaded (or blacklisted)",
        score,
        evidence,
        fix,
    )
}

/// The package manager's own "re-resolve the repo metadata" command for
/// `family`, without a `sudo` prefix. Shared by `check_7_stale_repos` and
/// `check_12_repo_native_broken` so a future per-family tweak (e.g. adding
/// `--non-interactive` to one of these) only has one copy to update instead
/// of drifting across the two checks.
fn refresh_command_for_family(family: &str) -> Option<&'static str> {
    match family {
        "apt" => Some("apt update"),
        // `dnf clean all` only deletes the local cache and always exits 0; it
        // can't fail even when a repo file is malformed or unreachable, which
        // is exactly what `verify` needs to catch. `dnf makecache` re-fetches
        // and parses the metadata, so it fails loudly in that case, matching
        // `apt update` and `zypper refresh` below.
        "dnf" => Some("dnf makecache"),
        "zypper" => Some("zypper refresh"),
        _ => None,
    }
}

pub(super) fn check_7_stale_repos(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let repos = &e.rocm_repos_seen;
    if repos.len() >= 2 {
        score += 40;
        evidence.push(format!(
            "{} ROCm/AMDGPU repo files present: {repos:?}",
            repos.len()
        ));
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_REPO_BROKEN);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-7-stale-repos", "Stale ROCm repos");
    }
    let family = repo_pkg_family(e);
    let mut commands = Vec::new();
    match family {
        "apt" => commands
            .push("ls /etc/apt/sources.list.d/ | grep -iE 'rocm|amdgpu|radeon' || true".to_owned()),
        "dnf" => {
            commands
                .push("ls /etc/yum.repos.d/ | grep -iE 'rocm|amdgpu|radeon' || true".to_owned());
        }
        "zypper" => commands
            .push("ls /etc/zypp/repos.d/ | grep -iE 'rocm|amdgpu|radeon' || true".to_owned()),
        _ => {}
    }
    for r in repos {
        commands.push(format!(
            "# sudo mv {r} {r}.bak     # quarantine, do not delete yet"
        ));
    }
    if let Some(cmd) = refresh_command_for_family(family) {
        commands.push(format!("sudo {cmd}"));
    }
    commands.push(
        "# If the package manager now resolves, reinstall the driver only (review the plan, then approve):"
            .to_owned(),
    );
    if dkms_driver_plan_is_supported(e) {
        commands.push("#   rocm install driver --dkms   # review the plan".to_owned());
        commands
            .push("#   rocm install driver --dkms --yes   # reinstalls the driver only".to_owned());
    } else {
        commands.push(format!(
            "#   `rocm install driver --dkms` plans nothing on {} {}: it isn't one of the AMD-documented distros/versions; see https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
            e.distro_id, e.distro_version
        ));
    }
    let verify = match refresh_command_for_family(family) {
        // `set -o pipefail` so the exit status reflects `{cmd}` itself, not
        // `tail` (which exits 0 regardless) -- otherwise a copy-pasted
        // failure of this command silently reads as success.
        Some(cmd) => format!("bash -c 'set -o pipefail; sudo {cmd} 2>&1 | tail -n 20'"),
        // No package manager to refresh on this host, so the only real
        // post-condition the suggested (commented-out) `mv ... .bak`
        // commands above leave behind is that the flagged repo files are
        // gone from their original paths -- unlike
        // `rocm examine --json | grep -q '"rocm_install_method"'`, which is
        // present in every `examine --json` output regardless of host state
        // and so can never fail. When no repo files were actually flagged
        // (the symptom text alone triggered this fix), there is nothing to
        // assert about a `mv` command that never ran, so fall back to a
        // real, still-falsifiable check: no ROCm/AMDGPU repo file exists
        // under any of the three families' repo directories.
        None if repos.is_empty() => {
            "! ls /etc/apt/sources.list.d/ /etc/yum.repos.d/ /etc/zypp/repos.d/ 2>/dev/null \
             | grep -iE 'rocm|amdgpu|radeon'"
                .to_owned()
        }
        None => repos
            .iter()
            .map(|r| format!("! test -e '{r}'"))
            .collect::<Vec<_>>()
            .join(" && "),
    };
    let fix = Fix {
        summary: "Quarantine duplicate ROCm/AMDGPU repo files and resolve the package manager before re-running any installer.".to_owned(),
        commands,
        needs_sudo: true,
        fix_id: "fix-7-stale-repos".to_owned(),
        verify,
        ..Fix::default()
    };
    finalize(
        "fix-7-stale-repos",
        "Stale or conflicting APT/DNF/zypper repos from prior installer runs",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_10_container_devices(e: &Examination, symptom: &str) -> Diagnosis {
    if !e.in_container {
        return zero("fix-10-container", "Container missing devices");
    }
    let kind = if e.container_kind.is_empty() {
        "container".to_owned()
    } else {
        e.container_kind.clone()
    };
    let mut score = 25;
    let mut evidence = vec![format!("running inside a {kind}")];
    // Mirror diagnose.py: a null kfd contributes 0 (the script reads
    // `kfd.get("exists") is False`, which is not True for a missing key). The
    // probe always populates kfd, so this only matters for hand-built exams.
    if let Some(kfd) = &e.kfd {
        if !kfd.exists {
            score += 40;
            evidence.push("/dev/kfd is not present in the container".to_owned());
        } else if kfd.user_can_write == Some(false) {
            score += 30;
            evidence.push("/dev/kfd is present but not writable by the container user".to_owned());
        }
    }
    if e.render_devices.is_empty() {
        score += 20;
        evidence.push("no /dev/dri/renderD* visible in the container".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_CONTAINER);
    score += kw_score;
    evidence.extend(kw_ev);

    let fix = Fix {
        summary: "Re-launch the container with the AMD devices and the render group passed through.".to_owned(),
        commands: vec![
            "# Docker / Podman flags AMD-recommends:".to_owned(),
            "docker run --rm -it \\".to_owned(),
            "  --device=/dev/kfd \\".to_owned(),
            "  --device=/dev/dri \\".to_owned(),
            "  --group-add render \\".to_owned(),
            "  --security-opt seccomp=unconfined \\".to_owned(),
            "  --shm-size=8g \\".to_owned(),
            "  rocm/pytorch:latest".to_owned(),
            "# Rootless podman: also pass `--userns=keep-id` and ensure the".to_owned(),
            "# host user is in the render group; podman maps it through.".to_owned(),
        ],
        fix_id: "fix-10-container".to_owned(),
        verify: "rocminfo | head -n 5".to_owned(),
        notes: vec!["Use rocm/pytorch or rocm/dev-ubuntu-22.04 as a known-good image. Mixing host ROCm + container ROCm versions is a separate footgun.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-10-container",
        "Container can't see /dev/kfd or /dev/dri/renderD*",
        score,
        evidence,
        fix,
    )
}

pub(super) fn check_11_iommu_hang(e: &Examination, symptom: &str) -> Diagnosis {
    if amd_gpu_count(e) < 2 {
        return zero("fix-11-iommu", "Multi-GPU IOMMU hang");
    }
    let mut score = 0;
    let mut evidence = vec![format!("{} AMD GPUs detected", amd_gpu_count(e))];
    let iommu = &e.iommu_kernel_param;
    if !iommu.is_empty() && iommu != "pt" {
        score += 25;
        evidence.push(format!("kernel cmdline has iommu={iommu} (not 'pt')"));
    }
    if iommu.is_empty() {
        score += 10;
        evidence.push("no iommu= flag on kernel cmdline (default may be 'on')".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_IOMMU_HANG);
    score += kw_score;
    evidence.extend(kw_ev);

    if score < 25 {
        return zero("fix-11-iommu", "Multi-GPU IOMMU hang");
    }
    let fix = Fix {
        summary: "Add `iommu=pt` to the kernel command line so DMA goes through pass-through mode. This requires editing GRUB and rebooting.".to_owned(),
        commands: vec![
            "# Inspect the current cmdline:".to_owned(),
            "cat /proc/cmdline".to_owned(),
            "# Edit /etc/default/grub and add iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT:".to_owned(),
            "sudo $EDITOR /etc/default/grub".to_owned(),
            "sudo update-grub                # Debian/Ubuntu".to_owned(),
            "sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora/RHEL".to_owned(),
            "# Reboot for the change to take effect, then retry the multi-GPU job.".to_owned(),
        ],
        needs_sudo: true,
        needs_reboot: true,
        fix_id: "fix-11-iommu".to_owned(),
        verify: "cat /proc/cmdline | grep -o 'iommu=\\w*'".to_owned(),
        ..Fix::default()
    };
    finalize(
        "fix-11-iommu",
        "Multi-GPU hang on systems with IOMMU enabled",
        score,
        evidence,
        fix,
    )
}

/// Best-effort package-manager family for this host: prefer the family
/// implied by the repo dir the marker files live under, since that's the
/// config these commands act on, and fall back to the same `ID_LIKE`-aware
/// resolver `openmpi` uses only when no repo file was found -- restricted to
/// apt/dnf/zypper, the only families this check has remediation for: a
/// pacman host (which `openmpi`'s resolver can return, for its own,
/// unrelated library installers) falls back to `"unknown"`, rather than the
/// remediation commands mismatching a real "pacman" answer.
fn repo_pkg_family(e: &Examination) -> &'static str {
    if e.rocm_repos_seen.iter().any(|r| r.contains("/etc/apt/")) {
        return "apt";
    }
    if e.rocm_repos_seen
        .iter()
        .any(|r| r.contains("/etc/yum.repos.d/"))
    {
        return "dnf";
    }
    if e.rocm_repos_seen.iter().any(|r| r.contains("/etc/zypp/")) {
        return "zypper";
    }
    // The two `match family` sites below only implement apt/dnf/zypper
    // remediation (this check has no pacman-based repo-native ROCm install to
    // recover, unlike `openmpi`'s library installers); their `_` arms already
    // treat any unhandled string, including a literal `"pacman"`, as "no
    // package-manager-specific command" -- so collapsing pacman to
    // `"unknown"` here doesn't change that match's behavior. It's kept so the
    // family string this function returns always matches what its callers
    // actually handle, rather than leaking a fourth value they merely happen
    // to no-op on today.
    crate::openmpi::resolve_package_manager(
        &e.distro_id.to_ascii_lowercase(),
        &e.distro_id_like.to_ascii_lowercase(),
    )
    .filter(|pm| {
        matches!(
            pm,
            crate::openmpi::PackageManager::Apt
                | crate::openmpi::PackageManager::Dnf
                | crate::openmpi::PackageManager::Zypper
        )
    })
    .map_or("unknown", crate::openmpi::PackageManager::as_str)
}

/// Whether `rocm_install_method` indicates a package-manager-driven install,
/// covering both the marker-file-based `"repo-native"` value and the
/// dpkg/rpm-query fallback's `"apt"`/`"dnf"` (see `probe_rocm_install` in
/// `examine/rocm_install_probe.rs`): a host without one of `REPO_NATIVE_INSTALL_MARKERS` present
/// but whose package manager confirms `rocm-hip-runtime` is installed is
/// still repo-native-managed, just detected a different way.
///
/// `"amdgpu-install"` is kept as a legacy alias of `"repo-native"`: an older
/// `rocm remote doctor` client can still send the retired value for a while
/// after this rename, and dropping it here would silently lose fix-12
/// evidence for that machine rather than just serving a stale label.
fn is_repo_native_installed(method: &str) -> bool {
    matches!(method, "repo-native" | "amdgpu-install" | "apt" | "dnf")
}

/// Whether `rocm install driver --dkms` plans any commands for this host.
///
/// Mirrors the full `(os_id, version_id)` / `ID_LIKE`-derivative
/// matrix in `apps/rocm/src/driver_plan.rs::build_driver_install_plan_at`
/// (`rocm-core` can't call that plan builder directly: `apps/rocm` depends
/// on `rocm-core`, not the reverse).
pub fn dkms_driver_plan_is_supported(e: &Examination) -> bool {
    let os_id = e.distro_id.to_ascii_lowercase();
    let version_id = e.distro_version.as_str();
    match (os_id.as_str(), version_id) {
        ("ubuntu", "22.04" | "24.04" | "26.04") => true,
        ("debian", "12" | "13") => true,
        ("rhel", "10.1" | "10.0" | "9.7" | "9.6" | "9.4" | "8.10") => true,
        ("ol", "10.1" | "9.7" | "8.10") => true,
        ("rocky", "9.4" | "9.6" | "9.7") => true,
        ("sles" | "sle", "15.7") => true,
        _ => {
            let likes: Vec<String> = e
                .distro_id_like
                .split_whitespace()
                .map(str::to_ascii_lowercase)
                .collect();
            let mentions = |family: &str| likes.iter().any(|like| like == family);
            (mentions("ubuntu") && matches!(version_id, "22.04" | "24.04" | "26.04"))
                || (mentions("debian") && matches!(version_id, "12" | "13"))
                || (mentions("rhel")
                    && !matches!(os_id.as_str(), "ol" | "oracle")
                    && matches!(version_id, "10.1" | "10.0" | "9.7" | "9.6" | "9.4" | "8.10"))
        }
    }
}

pub(super) fn check_12_repo_native_broken(e: &Examination, symptom: &str) -> Diagnosis {
    let mut score = 0;
    let mut evidence = Vec::new();
    let method = &e.rocm_install_method;
    if is_repo_native_installed(method) {
        evidence.push("ROCm was installed via the repo-native package-manager flow".to_owned());
    }
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_DPKG_BROKEN);
    score += kw_score;
    evidence.extend(kw_ev);
    if is_repo_native_installed(method) && kw_score > 0 {
        score += 20;
    }

    if score <= 0 {
        return zero("fix-12-installer", "repo-native install broken state");
    }
    let family = repo_pkg_family(e);
    // Purging `amdgpu-dkms` is only the first half of purge-then-reinstall:
    // where rocm-cli plans no reinstall (undocumented distro) it would strip
    // the driver with nothing to put back, so that step is withheld there.
    let reinstall_planned = dkms_driver_plan_is_supported(e);
    let mut commands = vec!["# Clear the half-configured package/DKMS state first.".to_owned()];
    match family {
        "apt" => {
            commands.push("sudo dpkg --configure -a".to_owned());
            commands.push("sudo apt-get install -f".to_owned());
            if reinstall_planned {
                commands.push("sudo apt-get purge -y amdgpu-dkms".to_owned());
            }
        }
        "dnf" if reinstall_planned => commands.push("sudo dnf remove -y amdgpu-dkms".to_owned()),
        "zypper" if reinstall_planned => {
            commands.push("sudo zypper remove -y amdgpu-dkms".to_owned());
        }
        _ => {}
    }
    commands.push(
        "# Quarantine (do not delete) the repo files, then let the package manager".to_owned(),
    );
    commands.push("# forget the broken state before reinstalling.".to_owned());
    if e.rocm_repos_seen.is_empty() {
        // No repo file was actually seen on this host, so show an example
        // matching the detected package family rather than always the apt
        // path -- a dnf/zypper host has no `/etc/apt/` at all.
        let example = match family {
            "dnf" => "/etc/yum.repos.d/amdgpu.repo",
            "zypper" => "/etc/zypp/repos.d/amdgpu.repo",
            _ => "/etc/apt/sources.list.d/amdgpu.list",
        };
        commands.push(format!(
            "# sudo mv {example} {example}.bak     # or the equivalent for your package manager"
        ));
    } else {
        for r in &e.rocm_repos_seen {
            commands.push(format!(
                "# sudo mv {r} {r}.bak     # quarantine, do not delete yet"
            ));
        }
    }
    if let Some(cmd) = refresh_command_for_family(family) {
        commands.push(format!("sudo {cmd}"));
    }
    let summary = if dkms_driver_plan_is_supported(e) {
        commands.push("# Reinstall via rocm-cli's repo-native flow:".to_owned());
        commands.push("rocm install driver --dkms --yes".to_owned());
        "Clear the half-configured package/DKMS state, quarantine the repo-native install's repo files, THEN reinstall via `rocm install driver --dkms --yes`.".to_owned()
    } else {
        commands.push(
            "# rocm-cli has no automated DKMS driver plan for this distro; reinstall the"
                .to_owned(),
        );
        commands.push("# driver manually per AMD's install docs:".to_owned());
        commands.push("#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html".to_owned());
        format!(
            "Clear the half-configured package/DKMS state and quarantine the repo-native install's repo files; rocm-cli has no automated DKMS driver plan for {} {}, so reinstall the driver manually per AMD's docs.",
            e.distro_id, e.distro_version
        )
    };

    // Query the package manager's own records, not `rocm examine`'s
    // self-reported `rocm_install_method` marker: that marker is set from
    // repo-marker-file presence before dpkg/rpm is consulted (see
    // `probe_rocm_install`), so a repo file left behind by a failed
    // `install --dkms` keeps the marker true even though the package
    // never actually installed.
    //
    // Must check the *targeted* package's installed status, not just that
    // some `rocm`/`amdgpu` name appears anywhere: a bare `grep -qE
    // 'rocm|amdgpu'` also matches unrelated stock packages (e.g.
    // `libdrm-amdgpu1`), and matches a healthy `rocm-*` package sitting
    // next to the broken `amdgpu-dkms` this fix targets. So query
    // `amdgpu-dkms` specifically, via dpkg's full `Status` field
    // (`install ok installed`, not a `^ii` substring) or `rpm -q`. Also
    // confirm DKMS itself reports the module built, and check
    // `rocminfo`'s own exit code directly -- piping through `head` would
    // report the pipeline's last stage, masking a missing/failing
    // `rocminfo`.
    //
    let verify = if dkms_driver_plan_is_supported(e) {
        "(dpkg-query -W -f='${Status}' amdgpu-dkms 2>/dev/null | grep -q 'install ok installed' \
         || rpm -q amdgpu-dkms >/dev/null 2>&1) \
         && dkms status amdgpu 2>/dev/null | grep -q installed \
         && rocminfo >/dev/null 2>&1"
            .to_owned()
    } else {
        // No AMD-documented package/DKMS name to check on an undocumented
        // distro -- the manual reinstall above is on the user, not rocm-cli,
        // so the only thing this fix itself can still assert is that the
        // GPU stack actually works afterward.
        "rocminfo >/dev/null 2>&1".to_owned()
    };
    let fix = Fix {
        summary,
        commands,
        needs_sudo: true,
        needs_reboot: true,
        fix_id: "fix-12-installer".to_owned(),
        verify,
        notes: vec!["If the purge/remove step warns it will take unrelated packages with it, stop and resolve those by hand before continuing.".to_owned()],
        ..Fix::default()
    };
    finalize(
        "fix-12-installer",
        "Repo-native install left a broken state (repo regression / partial DKMS)",
        score,
        evidence,
        fix,
    )
}

/// The stable handle for the unified-memory entry, shared with its recipe in
/// [`crate::fix`].
pub const APU_UNIFIED_MEMORY_FIX_ID: &str = "fix-24-apu-unified-memory";

/// 4 KiB pages per MiB.
///
/// The two kernel parameters this check reads are in different units:
/// `amdgpu.gttsize` counts MiB and `ttm.pages_limit` counts 4 KiB pages. A
/// consistent pair therefore satisfies
/// `pages_limit >= gttsize_mib * TTM_PAGES_PER_MIB`; a smaller `pages_limit`
/// caps the aperture at `pages_limit / TTM_PAGES_PER_MIB` MiB however large
/// `gttsize` is, which is the misconfiguration this entry exists to name.
const TTM_PAGES_PER_MIB: u64 = 256;

/// The share of installed RAM the recommended `amdgpu.gttsize` takes.
///
/// Three quarters, and both halves of the reason have to hold.
///
/// It has to stay under the kernel's own ceiling on host memory the KFD will
/// hand out. `amdgpu_amdkfd_gpuvm_init_mem_limits` in
/// `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c` sets `max_system_mem_limit`
/// to `RAM - RAM/64 - 1.5 GiB`, and a reservation refused against that limit is
/// what prints the `SVM mapping failed` line this check reads. Three quarters is
/// below `RAM - RAM/64 - 1.5 GiB` whenever RAM is at least 6.4 GiB, so on any
/// machine with 8 GiB or more the recommendation cannot itself be what trips the
/// limit. `the_recommended_pair_stays_under_the_kernels_own_memory_limit` holds
/// that arithmetic.
///
/// The quarter it leaves is the operating system's: the page cache the weights
/// are read through, the host-side copy the loader makes, and everything else
/// running. A larger share is a defensible choice on a box that does nothing
/// else, which is why this entry prints the figure instead of applying it.
const RECOMMENDED_GTT_SHARE_OF_RAM: f64 = 0.75;

/// The pair to recommend on a host with `ram_gib` of installed memory, as
/// `(gttsize_mib, pages_limit)`.
///
/// Consistent by construction: the page limit is derived from the MiB figure
/// through [`TTM_PAGES_PER_MIB`], so the advice can never reproduce the
/// inconsistency it is given to fix. `None` when the figure is not a usable
/// measurement or rounds down to nothing.
fn recommended_limits(ram_gib: f64) -> Option<(u64, u64)> {
    if !ram_gib.is_finite() || ram_gib <= 0.0 {
        return None;
    }
    // Whole GiB, rounded down, so the printed command line is one a user can
    // read back against `free -g`.
    let gib = (ram_gib * RECOMMENDED_GTT_SHARE_OF_RAM).floor() as u64;
    let mib = gib.checked_mul(1024)?;
    let pages = mib.checked_mul(TTM_PAGES_PER_MIB)?;
    (mib > 0).then_some((mib, pages))
}

/// The recent kernel log's verdict on SVM mapping, as an evidence line.
///
/// Both answers are stated, and neither is "this machine has never failed".
/// The probe reads the last [`DMESG_LINES_SCANNED`](crate::examine::DMESG_LINES_SCANNED)
/// lines of the kernel log and keeps at most
/// [`DMESG_HITS_KEPT`](crate::examine::DMESG_HITS_KEPT) amdgpu matches out of
/// them, so a failure from an earlier run is simply not in the evidence this
/// check can see. Saying so is the difference between "it did not happen" and
/// "it is not in the window that was read".
fn svm_evidence(e: &Examination) -> (bool, String) {
    let hit = e.dmesg_amdgpu_tail.iter().find(|line| {
        let lower = line.to_lowercase();
        lower.contains(crate::examine::SVM_MAPPING_FAILED)
            || lower.contains(crate::examine::SVM_ALREADY_ALLOCATED)
    });
    match hit {
        Some(line) => (
            true,
            format!("the kernel log reports an SVM mapping failure: {line}"),
        ),
        None => (
            false,
            format!(
                "no SVM mapping failure in the recent kernel log (the last {} lines, of which at \
                 most {} amdgpu matches are kept), so an older one would not be seen here",
                crate::examine::DMESG_LINES_SCANNED,
                crate::examine::DMESG_HITS_KEPT,
            ),
        ),
    }
}

pub(super) fn check_24_apu_unified_memory(e: &Examination, symptom: &str) -> Diagnosis {
    const TITLE: &str = "Unified memory GTT/TTM limits on an integrated GPU";

    let gttsize = e.amdgpu_gttsize_mib;
    let pages_limit = e.ttm_pages_limit;
    // Nothing to say about a machine with no integrated GPU that has set
    // neither parameter and whose recent kernel log is clean: every piece of
    // evidence this entry reasons over is absent.
    if !e.has_apu && gttsize.is_none() && pages_limit.is_none() && e.dmesg_amdgpu_tail.is_empty() {
        return zero(APU_UNIFIED_MEMORY_FIX_ID, TITLE);
    }

    let mut score = 0;
    let mut evidence = Vec::new();

    let (svm_failed, svm_line) = svm_evidence(e);
    if svm_failed {
        score += 50;
    }
    evidence.push(svm_line);

    match (gttsize, pages_limit) {
        (Some(mib), Some(pages)) => {
            let needed = mib.saturating_mul(TTM_PAGES_PER_MIB);
            if pages < needed {
                score += 50;
                evidence.push(format!(
                    "amdgpu.gttsize={mib} (MiB) asks for {needed} pages of 4 KiB, but \
                     ttm.pages_limit={pages} allows only {} MiB: the page limit is the one that \
                     decides, so the aperture is capped at {} MiB whatever gttsize says",
                    pages / TTM_PAGES_PER_MIB,
                    pages / TTM_PAGES_PER_MIB,
                ));
            } else {
                evidence.push(format!(
                    "amdgpu.gttsize={mib} (MiB) and ttm.pages_limit={pages} agree: the page limit \
                     covers the aperture"
                ));
            }
        }
        (Some(mib), None) => {
            score += 25;
            evidence.push(format!(
                "amdgpu.gttsize={mib} (MiB) is set but ttm.pages_limit is not, so TTM's default \
                 of about half of installed RAM decides instead"
            ));
        }
        (None, Some(pages)) => {
            score += 25;
            evidence.push(format!(
                "ttm.pages_limit={pages} is set but amdgpu.gttsize is not, so the aperture is \
                 whatever amdgpu defaults to and the page limit allows at most {} MiB",
                pages / TTM_PAGES_PER_MIB,
            ));
        }
        (None, None) if e.has_apu => {
            score += 15;
            evidence.push(
                "this machine has an integrated GPU and sets neither amdgpu.gttsize nor \
                 ttm.pages_limit, so the memory it may be given is whatever the driver defaults \
                 and the BIOS carve-out leave it"
                    .to_owned(),
            );
        }
        (None, None) => {}
    }

    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_APU_UNIFIED_MEMORY);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero(APU_UNIFIED_MEMORY_FIX_ID, TITLE);
    }

    let recommended = e.system_ram_gib.and_then(recommended_limits);
    let mut commands = vec![
        "# Inspect the current cmdline:".to_owned(),
        "cat /proc/cmdline".to_owned(),
        "# Edit /etc/default/grub and put BOTH parameters on".to_owned(),
        "# GRUB_CMDLINE_LINUX_DEFAULT; ttm.pages_limit counts 4 KiB pages, so it".to_owned(),
        format!("# must be at least {TTM_PAGES_PER_MIB} x the MiB figure in amdgpu.gttsize:"),
    ];
    let summary = if let Some((mib, pages)) = recommended {
        commands.push(format!(
            "#   amdgpu.gttsize={mib} ttm.pages_limit={pages}   # {} GiB",
            mib / 1024
        ));
        format!(
            "Give the integrated GPU a {} GiB aperture by setting amdgpu.gttsize={mib} together \
             with ttm.pages_limit={pages}, and raise the BIOS carve-out as the notes say. This \
             means editing GRUB and rebooting.",
            mib / 1024
        )
    } else {
        commands.push("#   amdgpu.gttsize=<MiB> ttm.pages_limit=<MiB x 256>".to_owned());
        commands
            .push("# Installed RAM could not be read here, so size the pair yourself:".to_owned());
        commands.push("free -g".to_owned());
        "Set amdgpu.gttsize and ttm.pages_limit as a consistent pair sized to installed RAM, and \
         raise the BIOS carve-out as the notes say. This means editing GRUB and rebooting."
            .to_owned()
    };
    commands.extend([
        "sudo $EDITOR /etc/default/grub".to_owned(),
        "sudo update-grub                # Debian/Ubuntu".to_owned(),
        "sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora/RHEL".to_owned(),
        "# Reboot for the change to take effect, then retry the model load.".to_owned(),
    ]);

    let fix = Fix {
        summary,
        commands,
        needs_sudo: true,
        needs_reboot: true,
        fix_id: APU_UNIFIED_MEMORY_FIX_ID.to_owned(),
        // Reads the parameters back off the live command line rather than
        // reporting that the editor exited: the only thing that proves the
        // change took is the kernel having been booted with it.
        verify:
            "grep -o 'amdgpu\\.gttsize=[0-9]*\\|\\(amd\\)\\?ttm\\.pages_limit=[0-9]*' /proc/cmdline"
                .to_owned(),
        notes: vec![
            "The BIOS carve-out is a firmware setting, not a command: in the UEFI setup look for \
             UMA Frame Buffer Size, Dedicated Graphics Memory or GART size, and give it the \
             largest value the board offers that still leaves the operating system enough RAM. \
             The carve-out is what the GPU reports as its own VRAM; the GTT aperture above is \
             memory on top of it."
                .to_owned(),
            "ttm.pages_limit is called amdttm.pages_limit when the out-of-tree amdgpu-dkms \
             packages are installed, because they ship their own copy of TTM. Set the name your \
             own driver uses; setting the other one has no effect."
                .to_owned(),
        ],
        ..Fix::default()
    };
    finalize(APU_UNIFIED_MEMORY_FIX_ID, TITLE, score, evidence, fix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::test_support::linux_base;
    #[cfg(unix)]
    use crate::diagnose::test_support::{run_in_sh, stub_bin_dir};
    use crate::diagnose::{HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH, diagnose, render_report_text};
    use crate::examine::{Device, Gpu};

    #[test]
    fn every_amdgpu_missing_phrase_is_in_the_evidence_the_checks_produce() {
        let mut e = linux_base();
        e.rocm_version = "7.1.0".to_owned();
        e.amdgpu_loaded = Some(false);
        e.secure_boot = "enabled".to_owned();
        let lines: Vec<String> = [
            check_3_rocm_kernel_unsupported(&e, ""),
            check_5_amdgpu_blacklisted(&e, ""),
        ]
        .into_iter()
        .flat_map(|d| d.evidence)
        .collect();
        for phrase in AMDGPU_MISSING_EVIDENCE {
            assert!(
                lines.iter().any(|l| l.contains(phrase)),
                "no evidence line contains {phrase:?}: {lines:#?}"
            );
        }
    }

    #[test]
    fn built_in_amdgpu_produces_none_of_the_amdgpu_missing_phrases() {
        let mut e = linux_base();
        e.rocm_version = "7.1.0".to_owned();
        e.amdgpu_loaded = Some(true);
        e.secure_boot = "enabled".to_owned();
        for d in [
            check_3_rocm_kernel_unsupported(&e, ""),
            check_5_amdgpu_blacklisted(&e, ""),
        ] {
            for line in &d.evidence {
                assert!(
                    !AMDGPU_MISSING_EVIDENCE.iter().any(|p| line.contains(p)),
                    "{line}"
                );
            }
        }
    }

    #[test]
    fn render_group_missing_is_diagnosed() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-4-render-group");
        assert_eq!(top.score, 45); // 35 render + 10 video
        assert!(top.fix.as_ref().unwrap().auto_applicable);
    }

    #[test]
    fn render_group_with_symptom_is_high_confidence() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "RuntimeError: unable to open /dev/kfd");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-4-render-group");
        assert!(top.score >= HIGH_CONFIDENCE, "score was {}", top.score);
        assert!(report.has_match());
    }

    #[test]
    fn kfd_not_writable_adds_score() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.kfd = Some(Device {
            path: "/dev/kfd".to_owned(),
            exists: true,
            mode: "crw-rw----".to_owned(),
            owner_group: "render".to_owned(),
            user_can_write: Some(false),
            ..Device::default()
        });
        let report = diagnose(&e, "");
        let top = &report.matched[0];
        assert_eq!(top.id, "fix-4-render-group");
        assert_eq!(top.score, 60); // 35 render + 25 kfd
    }

    /// fix-4's verify line is printed by `rocm diagnose` (from this check's
    /// `Fix`) and by `rocm fix fix-4-render-group` (from the recipe). The two
    /// copies drifted once, with only this one listing `/dev/kfd`. Both now read
    /// `crate::fix::RENDER_GROUP_VERIFY`, and this keeps either side from going
    /// back to a literal of its own. The check builds its verify line without
    /// looking at the host, so one host that fires it is enough.
    #[test]
    fn the_render_group_fix_verify_matches_the_catalog_copy() {
        let (catalog_verify, _, _) =
            crate::fix::recipe_verify_notes_and_run_commands("fix-4-render-group")
                .expect("fix-4-render-group is in the recipe catalog");
        let not_in_render = Examination {
            in_render_group: Some(false),
            ..linux_base()
        };
        let fix = check_4_render_group(&not_in_render, "")
            .fix
            .expect("check 4 builds a Fix when the user is not in render");
        assert_eq!(
            fix.verify, catalog_verify,
            "the verify line for fix-4-render-group diverged between `diagnose` and `rocm fix`"
        );
        // The owning group is in the line, as it is in the command the check
        // tells the user to run; `ls -l` prints it for the user to read.
        // `the_render_group_verify_accepts_the_group_that_owns_kfd` runs it.
        assert!(
            catalog_verify.contains("stat -c %G /dev/kfd")
                && catalog_verify.contains("ls -l /dev/kfd"),
            "{catalog_verify}"
        );
    }

    /// fix-4's verify line, run in `/bin/sh` with stand-ins for `groups`,
    /// `stat`, `ls` and `rocminfo`. The check tells the user to join the group
    /// that owns `/dev/kfd`, which need not be `render`, so a member of that
    /// group alone must pass, and a user in none of the groups must fail,
    /// including when `stat` fails or names no group (`UNKNOWN`).
    #[cfg(unix)]
    #[test]
    fn the_render_group_verify_accepts_the_group_that_owns_kfd() {
        fn real(tool: &str) -> String {
            ["/usr/bin", "/bin"]
                .iter()
                .map(|dir| format!("{dir}/{tool}"))
                .find(|path| std::path::Path::new(path).exists())
                .unwrap_or_else(|| panic!("{tool} is installed"))
        }
        let passes = |groups: &str, owner: Option<&str>| {
            let groups = format!("echo '{groups}'");
            let stat = owner.map_or_else(
                || "echo 'stat: cannot stat' >&2; exit 1".to_owned(),
                |owner| format!("echo '{owner}'"),
            );
            let tr = format!("exec {} \"$@\"", real("tr"));
            let grep = format!("exec {} \"$@\"", real("grep"));
            let head = format!("exec {} \"$@\"", real("head"));
            let bin = stub_bin_dir(&[
                ("groups", groups.as_str()),
                ("stat", stat.as_str()),
                ("ls", "echo 'crw-rw---- 1 root kfd 235, 0 /dev/kfd'"),
                ("rocminfo", "echo agent"),
                ("tr", tr.as_str()),
                ("grep", grep.as_str()),
                ("head", head.as_str()),
            ]);
            let (ok, _, _) = run_in_sh(crate::fix::RENDER_GROUP_VERIFY, bin.path(), &[]);
            ok
        };
        assert!(
            passes("user kfd", Some("kfd")),
            "a member of the group that owns /dev/kfd must pass"
        );
        assert!(passes("user render", Some("kfd")), "render still counts");
        assert!(
            !passes("user", Some("kfd")),
            "a user in none of them must fail"
        );
        assert!(
            !passes("user", Some("UNKNOWN")),
            "an owner with no name must not let a user in none of them pass"
        );
        assert!(
            !passes("user", None),
            "a failing stat leaves an empty alternative that must match nothing"
        );
    }

    #[test]
    fn unknown_kfd_group_falls_back_to_render() {
        // `stat -c %G /dev/kfd` prints "UNKNOWN" when the GID has no group
        // name. That value must not leak into the remediation command, which
        // would otherwise suggest `usermod -a -G UNKNOWN,video` -- a group that
        // does not exist. Fall back to the render group instead, and only when
        // the host has one: with no `render` either, there is no group to name
        // and the plan carries no command at all. Exercise both casings so the
        // case-insensitive guard cannot silently regress to an exact-match
        // check while this test stays green.
        let host_has = |groups: &[&str]| Some(groups.iter().map(|g| (*g).to_owned()).collect());
        for sentinel in ["UNKNOWN", "unknown"] {
            let mut e = linux_base();
            e.in_render_group = Some(false);
            e.kfd = Some(Device {
                path: "/dev/kfd".to_owned(),
                exists: true,
                mode: "crw-rw----".to_owned(),
                owner_group: sentinel.to_owned(),
                user_can_write: Some(false),
                ..Device::default()
            });
            e.groups_on_host = host_has(&["render", "video"]);
            let report = diagnose(&e, "");
            let top = &report.matched[0];
            assert_eq!(top.id, "fix-4-render-group");
            let fix = top.fix.as_ref().expect("fix-4 carries a fix");
            assert_eq!(
                fix.commands,
                vec!["sudo usermod -a -G render,video \"$USER\"".to_owned()],
                "{sentinel} group must fall back to render, not leak into the command"
            );
            assert!(
                !fix.summary.to_uppercase().contains("UNKNOWN"),
                "summary must not name the bogus {sentinel} group:\n{}",
                fix.summary
            );

            // The ROCMAI-296 open point: a container whose group database has
            // no `render` group.
            e.groups_on_host = host_has(&["video"]);
            let report = diagnose(&e, "");
            let fix = report.matched[0].fix.as_ref().expect("fix-4 carries a fix");
            assert!(
                fix.commands.iter().all(|c| c.starts_with('#')),
                "{sentinel} with no render group: nothing may be offered to run: {:?}",
                fix.commands
            );
            assert!(
                fix.summary
                    .starts_with("No group to add the current user to:"),
                "{}",
                fix.summary
            );
            // The flags line must not promise a run: `rocm fix fix-4-render-group`
            // exits 3 with the reason here and changes nothing (held by
            // `render_group_with_no_group_to_name_changes_nothing_and_says_why` in fix/runners.rs).
            assert_eq!(fix.class, crate::fix::FixClass::DiagnoseOnly, "{fix:?}");
            assert!(!fix.auto_applicable, "{fix:?}");
            assert!(
                !fix.needs_sudo && !fix.needs_relogin && fix.verify.is_empty(),
                "{fix:?}"
            );
            let text = crate::diagnose::render_report_text(&report, 5);
            assert!(
                !text.contains("rocm fix can run it") && text.contains("will NOT change anything"),
                "{text}"
            );
        }
    }

    /// The examination did not read the host's groups (an older remote CLI):
    /// the diagnosis cannot say no group exists, so it prints the reason and
    /// no command but does not claim `rocm fix` changes nothing -- `rocm fix`
    /// reads the groups live and may run.
    #[test]
    fn unread_host_groups_print_no_command_but_leave_the_fix_runnable() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.groups_on_host = None;
        let report = diagnose(&e, "");
        let fix = report.matched[0].fix.as_ref().expect("fix-4 carries a fix");
        assert_eq!(report.matched[0].id, "fix-4-render-group");
        assert!(fix.commands.iter().all(|c| c.starts_with('#')), "{fix:?}");
        assert!(!fix.unavailable_here && fix.auto_applicable, "{fix:?}");
    }

    /// A `/dev/kfd` node the user can write to, owned by a group the user is
    /// not in.
    fn writable_kfd_host() -> Examination {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        e.kfd = Some(Device {
            path: "/dev/kfd".to_owned(),
            exists: true,
            mode: "crw-rw-rw-".to_owned(),
            owner_user: "root".to_owned(),
            owner_group: "render".to_owned(),
            user_can_read: Some(true),
            user_can_write: Some(true),
        });
        e.groups_on_host = Some(vec!["render".to_owned(), "video".to_owned()]);
        e
    }

    /// ROCMAI-400: the user can already write `/dev/kfd`, so joining a group
    /// is not what stands between them and the GPU, whatever groups they are
    /// in. The entry is not offered at all, nor any `usermod`.
    #[test]
    fn a_writable_kfd_does_not_raise_the_render_group_fix() {
        let e = writable_kfd_host();
        let d = check_4_render_group(&e, "");
        assert_eq!(d.score, 0, "{:?}", d.evidence);
        let report = diagnose(&e, "");
        assert!(
            !report.matched.iter().any(|d| d.id == "fix-4-render-group"),
            "{:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
        // The device has to be the reason: the same host with a device the
        // user cannot write still raises it.
        let mut locked = writable_kfd_host();
        if let Some(kfd) = locked.kfd.as_mut() {
            kfd.mode = "crw-rw----".to_owned();
            kfd.user_can_write = Some(false);
        }
        assert_eq!(check_4_render_group(&locked, "").score, 70);
    }

    /// The exception to the rule above: an error the user pasted that names a
    /// permission failure on the device. The examination and the error then
    /// disagree, and the entry stays, scored as before.
    #[test]
    fn a_kfd_permission_symptom_still_raises_it_on_a_writable_device() {
        let e = writable_kfd_host();
        let symptom = "RuntimeError: unable to open /dev/kfd";
        let d = check_4_render_group(&e, symptom);
        let (kw_score, _) = keyword_score(symptom, KEYWORDS_KFD_PERMISSION);
        assert!(
            kw_score > 0,
            "the symptom has to name a kfd permission failure"
        );
        assert_eq!(d.score, (35 + 10 + kw_score).min(100));
        let fix = d.fix.expect("fix-4 carries a fix");
        assert_eq!(fix.commands, ["sudo usermod -a -G render,video \"$USER\""]);
    }

    #[test]
    fn fix_11_iommu_rendered_flags_line_is_exact() {
        // Companion to
        // `fix_9_igpu_dgpu_waits_for_an_argument_rather_than_claiming_it_will_act`:
        // that test only pins a fix-id with a single class flag and no
        // sudo/reboot/relogin flags set, which an exact-match assertion can't
        // distinguish from a render path that drops the comma-joining or
        // misorders multiple flags. fix-11-iommu carries sudo+reboot+manual, so
        // this pins the full comma-joined, reworded `flags:` line through the
        // real render call site.
        let mut e = linux_base();
        e.iommu_kernel_param = "on".to_owned();
        e.gpus = vec![
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
        ];
        let report = diagnose(&e, "");
        let text = render_report_text(&report, report.matched.len());
        let lines: Vec<&str> = text.lines().collect();
        let id_line = lines
            .iter()
            .position(|l| l.trim_start() == "id: fix-11-iommu")
            .expect("fix-11-iommu should appear in the rendered report");
        let flags_line = lines[id_line..]
            .iter()
            .find(|l| l.trim_start().starts_with("flags:"))
            .expect("fix-11-iommu should have a flags: line");
        assert_eq!(
            flags_line.trim_start(),
            "flags: requires sudo, requires reboot, manual only (`rocm fix` will NOT run it automatically)",
            "rendered flags: line for fix-11-iommu: {flags_line}"
        );
    }

    fn assert_fix_5_needs_reboot(e: &Examination) {
        let report = diagnose(e, "");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-5-amdgpu-load")
            .expect("amdgpu-not-loaded should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        crate::fix::assert_needs_reboot_matches_the_catalog("fix-5-amdgpu-load", fix.needs_reboot);

        let text = render_report_text(&report, report.matched.len());
        let lines: Vec<&str> = text.lines().collect();
        let id_line = lines
            .iter()
            .position(|l| l.trim_start() == "id: fix-5-amdgpu-load")
            .expect("fix-5-amdgpu-load should appear in the rendered report");
        let flags_line = lines[id_line..]
            .iter()
            .find(|l| l.trim_start().starts_with("flags:"))
            .expect("fix-5-amdgpu-load should have a flags: line");
        assert_eq!(
            flags_line.trim_start(),
            "flags: requires sudo, requires reboot, manual only (`rocm fix` will NOT run it automatically)",
            "rendered flags: line for fix-5-amdgpu-load: {flags_line}"
        );
    }

    #[test]
    fn fix_5_amdgpu_load_needs_reboot_matches_catalog_when_blacklisted() {
        let mut e = linux_base();
        e.amdgpu_loaded = Some(false);
        e.amdgpu_blacklisted_in = vec!["/etc/modprobe.d/blacklist.conf".to_owned()];
        assert_fix_5_needs_reboot(&e);
    }

    #[test]
    fn fix_5_amdgpu_load_needs_reboot_matches_catalog_when_not_blacklisted() {
        // `check_5_amdgpu_blacklisted` used to compute `needs_reboot` as
        // `!blacklisted.is_empty()`, so this case (module simply not loaded,
        // no blacklist entry involved) used to report `false` here while the
        // `fix-5-amdgpu-load` FixRecipe catalog said `true` unconditionally --
        // the drift issue #418 fixed. Pin the now-unconditional `true` so a
        // regression back to the conditional fails here rather than only
        // being visible by eyeballing `rocm diagnose` vs `rocm fix` output.
        let mut e = linux_base();
        e.amdgpu_loaded = Some(false);
        assert!(e.amdgpu_blacklisted_in.is_empty());
        assert_fix_5_needs_reboot(&e);
    }

    #[test]
    fn a_healthy_container_reports_no_match_despite_a_nonempty_list() {
        // The case `has_match` exists for. `check_10_container_devices` opens at
        // 25 for merely being in a container, before it has looked at anything,
        // and this fixture adds the 20 for no visible render device (a probe
        // that found nothing, not a device that is missing) -- 45, short of the
        // threshold and still in `matched`. A caller reading "is `matched`
        // empty?" as "did anything match?" would propose re-launching a
        // container over what is only a thin probe, and would never route the
        // user upstream.
        let mut e = linux_base();
        e.in_container = true;
        e.container_kind = "docker".to_owned();
        let report = diagnose(&e, "");

        assert!(
            !report.matched.is_empty(),
            "fixture must produce an entry for this test to mean anything"
        );
        assert!(
            report.matched.iter().all(|d| d.score < MIN_SCORE_FOR_MATCH),
            "fixture must stay below the threshold; got {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
        assert!(
            !report.has_match,
            "a list of sub-threshold entries is not a match"
        );
    }

    #[test]
    fn iommu_requires_two_gpus_and_min_score() {
        // Single GPU: never fires.
        let mut e = linux_base();
        e.gpus = vec![Gpu {
            is_amd: true,
            ..Gpu::default()
        }];
        e.iommu_kernel_param = "on".to_owned();
        assert!(
            diagnose(&e, "hang")
                .matched
                .iter()
                .all(|d| d.id != "fix-11-iommu")
        );
        // Two GPUs + iommu=on (25) clears the per-rule >=25 gate.
        e.gpus = vec![
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
            Gpu {
                is_amd: true,
                ..Gpu::default()
            },
        ];
        let report = diagnose(&e, "");
        assert!(report.matched.iter().any(|d| d.id == "fix-11-iommu"));
    }

    #[test]
    fn repo_pkg_family_prefers_seen_repos_over_distro_id() {
        let mut e = linux_base();
        e.distro_id = "sles".to_owned();
        e.rocm_repos_seen = vec!["/etc/apt/sources.list.d/amdgpu.list".to_owned()];
        assert_eq!(repo_pkg_family(&e), "apt");
    }

    #[test]
    fn repo_pkg_family_detects_dnf_and_zypper_from_seen_repos() {
        let mut e = linux_base();
        e.rocm_repos_seen = vec!["/etc/yum.repos.d/amdgpu.repo".to_owned()];
        assert_eq!(repo_pkg_family(&e), "dnf");

        e.rocm_repos_seen = vec!["/etc/zypp/repos.d/amdgpu.repo".to_owned()];
        assert_eq!(repo_pkg_family(&e), "zypper");
    }

    #[test]
    fn repo_pkg_family_falls_back_to_distro_id_when_no_repo_seen() {
        let mut e = linux_base();
        e.distro_id = "sles".to_owned();
        assert_eq!(repo_pkg_family(&e), "zypper");

        e.distro_id = "rocky".to_owned();
        assert_eq!(repo_pkg_family(&e), "dnf");

        e.distro_id = "ubuntu".to_owned();
        assert_eq!(repo_pkg_family(&e), "apt");
    }

    #[test]
    fn repo_pkg_family_fallback_is_id_like_aware() {
        // Amazon Linux has no ROCm repo file in this scenario and isn't in
        // the old hardcoded list; it must not default to "apt".
        let mut e = linux_base();
        e.distro_id = "amzn".to_owned();
        assert_eq!(repo_pkg_family(&e), "dnf");

        // A RHEL rebuild with an unrecognized ID must still resolve via
        // ID_LIKE, same as openmpi's resolver.
        let mut e = linux_base();
        e.distro_id = "some-rhel-rebuild".to_owned();
        e.distro_id_like = "rhel fedora".to_owned();
        assert_eq!(repo_pkg_family(&e), "dnf");
    }

    #[test]
    fn repo_pkg_family_degrades_to_unknown_on_pacman_hosts() {
        // openmpi's resolver can answer "pacman" (it has its own, unrelated
        // library installers for Arch et al.), but this check has no pacman
        // remediation -- it must not surface "pacman" verbatim, nor coerce to
        // "apt" (which would make check_12 print apt/dpkg commands on an Arch
        // host); both `match family` arms in check_12 treat "unknown" as "no
        // package-manager-specific command".
        let mut e = linux_base();
        e.distro_id = "arch".to_owned();
        assert_eq!(repo_pkg_family(&e), "unknown");
    }

    #[test]
    fn kernel_unsupported_on_an_undocumented_distro_does_not_suggest_a_dkms_rerun() {
        let mut e = linux_base();
        e.distro_id = "fedora".to_owned();
        e.distro_version = "40".to_owned();
        e.rocm_version = "7.0.0".to_owned();
        e.amdgpu_loaded = Some(false);
        let report = diagnose(&e, "kernel/rocm mismatch");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-3-rocm-kernel")
            .expect("unsupported kernel triple should be diagnosed");
        let commands = hit.fix.as_ref().unwrap().commands.join("\n");
        assert!(
            commands.contains("plans nothing on fedora 40"),
            "an undocumented distro must be told the dkms command plans nothing, not rerun it: {commands}"
        );
        assert!(
            !commands.contains("rerun `rocm install driver --dkms --yes`"),
            "an undocumented distro must not be told to rerun a command that plans nothing: {commands}"
        );
    }

    #[test]
    fn stale_repos_on_an_undocumented_dnf_distro_does_not_suggest_a_dkms_reinstall() {
        let mut e = linux_base();
        e.distro_id = "fedora".to_owned();
        e.distro_version = "40".to_owned();
        e.rocm_repos_seen = vec![
            "/etc/yum.repos.d/amdgpu.repo".to_owned(),
            "/etc/yum.repos.d/rocm.repo".to_owned(),
        ];
        let report = diagnose(&e, "stale repo conflict");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-7-stale-repos")
            .expect("duplicate repo files should be diagnosed");
        let commands = hit.fix.as_ref().unwrap().commands.join("\n");
        assert!(
            commands.contains("plans nothing on fedora 40"),
            "an undocumented dnf distro must be told the dkms command plans nothing: {commands}"
        );
        assert!(
            !commands.contains("rocm install driver --dkms --yes")
                && !commands.contains("rocm install driver --dkms   # review the plan"),
            "an undocumented distro must not be offered a DKMS reinstall that plans nothing: {commands}"
        );
    }

    #[test]
    fn repo_native_broken_on_an_undocumented_distro_tells_the_user_to_reinstall_manually() {
        let mut e = linux_base();
        e.distro_id = "fedora".to_owned();
        e.distro_version = "40".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        e.rocm_repos_seen = vec!["/etc/yum.repos.d/amdgpu.repo".to_owned()];
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        let commands = fix.commands.join("\n");
        assert!(
            commands.contains("no automated DKMS driver plan for this distro")
                && commands.contains("system-requirements.html"),
            "an undocumented distro must be sent to AMD's manual install docs: {commands}"
        );
        assert!(
            !commands.contains("rocm install driver --dkms --yes"),
            "an undocumented distro must not be offered a DKMS reinstall that plans nothing: {commands}"
        );
        assert_eq!(
            fix.verify, "rocminfo >/dev/null 2>&1",
            "with no AMD-documented package/DKMS name to check, verify must fall back to \
             confirming the GPU stack works, not assert an unknowable package state: {}",
            fix.verify
        );
    }

    #[test]
    fn stale_repos_on_a_sles_host_gets_zypper_commands_not_apt() {
        // A SLES host whose repo files now live under /etc/zypp/repos.d/ (the
        // new zypp scan this diff adds) must not be told to run apt/dpkg
        // commands: `check_7` must branch on `repo_pkg_family` the same way
        // `check_12` already does.
        let mut e = linux_base();
        e.distro_id = "sles".to_owned();
        e.rocm_repos_seen = vec![
            "/etc/zypp/repos.d/amdgpu.repo".to_owned(),
            "/etc/zypp/repos.d/rocm.repo".to_owned(),
        ];
        let report = diagnose(&e, "stale repo conflict");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-7-stale-repos")
            .expect("duplicate repo files should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        let commands = fix.commands.join("\n");
        assert!(
            commands.contains("/etc/zypp/repos.d/") && commands.contains("sudo zypper refresh"),
            "a SLES host must get zypper remediation: {commands}"
        );
        assert!(
            !commands.contains("apt") && !commands.contains("dpkg"),
            "a SLES host must not be told to run apt/dpkg commands: {commands}"
        );
        assert!(
            fix.verify.contains("sudo zypper refresh"),
            "a SLES host's verify step must also use zypper, not apt: {}",
            fix.verify
        );
    }

    #[test]
    fn stale_repos_on_a_dnf_host_gets_dnf_commands_not_apt() {
        // Same drift guard as the SLES/zypper test above, for the dnf family.
        let mut e = linux_base();
        e.distro_id = "rocky".to_owned();
        e.rocm_repos_seen = vec![
            "/etc/yum.repos.d/amdgpu.repo".to_owned(),
            "/etc/yum.repos.d/rocm.repo".to_owned(),
        ];
        let report = diagnose(&e, "stale repo conflict");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-7-stale-repos")
            .expect("duplicate repo files should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        let commands = fix.commands.join("\n");
        assert!(
            commands.contains("/etc/yum.repos.d/") && commands.contains("sudo dnf makecache"),
            "a dnf host must get dnf remediation: {commands}"
        );
        assert!(
            !commands.contains("apt") && !commands.contains("dpkg") && !commands.contains("zypper"),
            "a dnf host must not be told to run apt/dpkg/zypper commands: {commands}"
        );
        assert!(
            fix.verify.contains("sudo dnf makecache"),
            "a dnf host's verify step must also use dnf, not apt: {}",
            fix.verify
        );
    }

    #[test]
    fn stale_repos_emits_no_package_manager_commands_on_pacman_hosts() {
        let mut e = linux_base();
        e.distro_id = "arch".to_owned();
        e.rocm_repos_seen = vec!["/some/rocm.repo".to_owned(), "/some/amdgpu.repo".to_owned()];
        let report = diagnose(&e, "stale repo conflict");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-7-stale-repos")
            .expect("duplicate repo files should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        let commands: Vec<&str> = fix
            .commands
            .iter()
            .map(String::as_str)
            .filter(|c| !c.trim_start().starts_with('#'))
            .collect();
        assert!(
            commands.iter().all(|c| !c.contains("apt")
                && !c.contains("dpkg")
                && !c.contains("dnf")
                && !c.contains("zypper")),
            "must not print a foreign package manager's commands on an unresolved/pacman host: {commands:?}"
        );
        assert!(
            !fix.verify.contains("apt")
                && !fix.verify.contains("dnf")
                && !fix.verify.contains("zypper"),
            "verify step must not assume a foreign package manager on a pacman host: {}",
            fix.verify
        );
        // Not just "no foreign package manager" -- the verify must be a real,
        // falsifiable post-condition. `rocm examine --json | grep -q
        // '"rocm_install_method"'` is present in every `examine --json` output
        // regardless of host state, so it would pass this assertion too while
        // never actually being able to fail. Pin it to the one thing the
        // suggested (commented) `mv {r} {r}.bak` commands actually change: the
        // flagged repo files no longer existing at their original paths.
        for r in &e.rocm_repos_seen {
            assert!(
                fix.verify.contains(r),
                "verify must check each flagged repo file is gone, not a tautology: {}",
                fix.verify
            );
        }
    }

    #[test]
    fn stale_repos_verify_is_not_empty_when_symptom_alone_triggers_on_an_unresolved_host() {
        // Same unresolved/pacman family as the test above, but this time no
        // repo files were ever recorded -- only the symptom text scores the
        // fix. `repos.iter().join(" && ")` over an empty `repos` produces an
        // empty string, which would make `verify` a blank, always-passing
        // check instead of the real repo-marker check below.
        let mut e = linux_base();
        e.distro_id = "arch".to_owned();
        let report = diagnose(&e, "404 against repo.radeon.com");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-7-stale-repos")
            .expect("symptom-only match should still diagnose fix-7");
        let fix = hit.fix.as_ref().unwrap();
        assert!(
            !fix.verify.trim().is_empty(),
            "verify must not be blank when no repo files were flagged"
        );
        assert!(
            fix.verify.contains("rocm|amdgpu|radeon"),
            "verify must fall back to a real, falsifiable repo-marker check: {}",
            fix.verify
        );
    }

    #[test]
    fn repo_native_broken_emits_no_package_manager_commands_on_pacman_hosts() {
        let mut e = linux_base();
        e.distro_id = "arch".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        // Only real (non-comment) commands are package-manager-specific here;
        // the commented-out quarantine example always names the apt path as a
        // generic illustration ("or the yum/zypp equivalent"), unrelated to
        // `family` and out of scope for this check.
        let commands: Vec<&str> = hit
            .fix
            .as_ref()
            .unwrap()
            .commands
            .iter()
            .map(String::as_str)
            .filter(|c| !c.trim_start().starts_with('#'))
            .collect();
        assert!(
            commands.iter().all(|c| !c.contains("apt")
                && !c.contains("dpkg")
                && !c.contains("dnf")
                && !c.contains("zypper")),
            "must not print a foreign package manager's commands on an unresolved/pacman host: {commands:?}"
        );
    }

    #[test]
    fn is_repo_native_installed_still_accepts_the_retired_amdgpu_install_value() {
        // An older `rocm remote doctor` client can report the pre-rename
        // value for a while after this PR; it must still count as
        // repo-native-managed, not silently fall through to "not a
        // package-manager install" and lose fix-12 evidence.
        assert!(is_repo_native_installed("amdgpu-install"));
        assert!(is_repo_native_installed("repo-native"));
        assert!(!is_repo_native_installed("runfile-or-tarball"));
        assert!(!is_repo_native_installed(""));
    }

    #[test]
    fn repo_native_broken_verify_requires_dpkg_status_not_just_name_presence() {
        // `method == "apt"` here is the exact pre-fix state `is_repo_native_installed`
        // treats as already repo-native-installed (see its doc comment): the dpkg/rpm
        // query fallback found an amdgpu/rocm package by name, with no distinction
        // between a healthy install and a half-configured one. If `verify` only
        // re-ran that same bare name grep, it would already be true here, before any
        // of this fix's remediation commands run.
        let mut e = linux_base();
        e.distro_id = "ubuntu".to_owned();
        e.distro_version = "22.04".to_owned();
        e.rocm_install_method = "apt".to_owned();
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let verify = &hit.fix.as_ref().unwrap().verify;
        assert!(
            verify.contains("grep -q 'install ok installed'"),
            "verify must check dpkg's full installed-status field, not just that the \
             package name appears anywhere -- a bare name grep also matches unrelated \
             stock packages (e.g. libdrm-amdgpu1) and a healthy rocm-* package sitting \
             next to the broken amdgpu-dkms this fix targets: {verify}"
        );
        assert!(
            verify.contains("dpkg-query -W -f='${Status}' amdgpu-dkms"),
            "verify must query dpkg's status for amdgpu-dkms by name, not merely contain \
             the right-shaped status-field query against some other package -- pinning \
             the package name closes the gap where the targeted name could be swapped \
             for an unrelated one and this test would still pass: {verify}"
        );
        assert!(
            verify.contains("dkms status amdgpu"),
            "verify must also confirm dkms reports the module actually built: {verify}"
        );
        assert!(
            !verify.contains("| head"),
            "verify must not pipe rocminfo through head, which would report head's exit \
             code instead of rocminfo's: {verify}"
        );
    }

    #[test]
    fn repo_native_broken_fires_and_clears_state_before_reinstall() {
        let mut e = linux_base();
        e.distro_id = "ubuntu".to_owned();
        e.distro_version = "22.04".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        e.rocm_repos_seen = vec!["/etc/apt/sources.list.d/amdgpu.list".to_owned()];
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let fix = hit.fix.as_ref().unwrap();
        let commands = fix.commands.join("\n");
        assert!(
            commands.contains("dpkg --configure -a"),
            "must clear the half-configured dpkg state before reinstalling: {commands}"
        );
        assert!(
            commands.contains("purge -y amdgpu-dkms"),
            "must purge the broken driver package: {commands}"
        );
        assert!(
            commands.contains("# sudo mv /etc/apt/sources.list.d/amdgpu.list"),
            "repo quarantine command must be commented out, not ready-to-run: {commands}"
        );
        assert!(
            commands.contains("rocm install driver --dkms --yes"),
            "reinstall command must actually install (bare 'rocm install driver' is a non-mutating preflight no-op): {commands}"
        );
        assert!(
            fix.verify.contains("dpkg") && fix.verify.contains("rpm"),
            "verify must query real dpkg/rpm package state, not the self-reported \
             rocm_install_method marker (set from repo-file presence before the \
             package actually installs): {}",
            fix.verify
        );
        assert!(
            hit.evidence
                .iter()
                .any(|l| l.contains("installed via the repo-native package-manager flow")),
            "positive control: the repo-native evidence line must appear when the method really is repo-native: {:?}",
            hit.evidence
        );
    }

    #[test]
    fn repo_native_broken_on_a_dnf_host_gets_dnf_commands_not_apt() {
        let mut e = linux_base();
        e.distro_id = "rocky".to_owned();
        e.distro_version = "9.6".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        e.rocm_repos_seen = vec!["/etc/yum.repos.d/amdgpu.repo".to_owned()];
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let commands = hit.fix.as_ref().unwrap().commands.join("\n");
        assert!(
            commands.contains("sudo dnf remove -y amdgpu-dkms")
                && commands.contains("sudo dnf makecache"),
            "a dnf host must get dnf remediation: {commands}"
        );
        assert!(
            !commands.contains("apt") && !commands.contains("dpkg") && !commands.contains("zypper"),
            "a dnf host must not be told to run apt/dpkg/zypper commands: {commands}"
        );
    }

    #[test]
    fn repo_native_broken_on_a_zypper_host_gets_zypper_commands_not_apt() {
        let mut e = linux_base();
        e.distro_id = "sles".to_owned();
        e.distro_version = "15.7".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        e.rocm_repos_seen = vec!["/etc/zypp/repos.d/amdgpu.repo".to_owned()];
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let commands = hit.fix.as_ref().unwrap().commands.join("\n");
        assert!(
            commands.contains("sudo zypper remove -y amdgpu-dkms")
                && commands.contains("sudo zypper refresh"),
            "a zypper host must get zypper remediation: {commands}"
        );
        assert!(
            !commands.contains("apt") && !commands.contains("dpkg") && !commands.contains("dnf"),
            "a zypper host must not be told to run apt/dpkg/dnf commands: {commands}"
        );
    }

    #[test]
    fn repo_native_broken_quarantine_example_matches_the_family_when_no_repo_was_seen() {
        // With no repo file recorded (`rocm_repos_seen` empty), the quarantine
        // line falls back to an example path -- it must match the host's own
        // package family, not always show the apt path on a dnf/zypper host
        // that has no `/etc/apt/` at all.
        let mut e = linux_base();
        e.distro_id = "sles".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        e.rocm_repos_seen = Vec::new();
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let commands = hit.fix.as_ref().unwrap().commands.join("\n");
        assert!(
            commands.contains("/etc/zypp/repos.d/amdgpu.repo"),
            "a zypper host's fallback quarantine example must show a zypp path: {commands}"
        );
        assert!(
            !commands.contains("/etc/apt/"),
            "a zypper host must not show an apt path as its quarantine example: {commands}"
        );
    }

    #[test]
    fn repo_native_broken_does_not_claim_repo_native_evidence_for_other_install_methods() {
        // Regression test: an empty symptom would short-circuit keyword_score to
        // 0 before the install-method gate is ever reached, making this pass
        // vacuously regardless of whether the gate works. Pass a real dpkg
        // symptom instead, matching the positive-control test above, so fix-12
        // still fires (kw_score alone clears MIN_SCORE_FOR_MATCH) but must not
        // claim repo-native evidence for a non-repo-native install.
        let mut e = linux_base();
        e.rocm_install_method = "runfile-or-tarball".to_owned();
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("dpkg keyword evidence alone should still surface fix-12");
        assert!(
            !hit.evidence
                .iter()
                .any(|l| l.contains("installed via the repo-native package-manager flow")),
            "must not claim repo-native evidence for a non-repo-native install: {:?}",
            hit.evidence
        );
    }

    #[test]
    fn repo_native_broken_claims_repo_native_evidence_for_dpkg_rpm_fallback_methods() {
        // probe_rocm_install (examine/rocm_install_probe.rs) falls back to the literal "apt"/"dnf"
        // when dpkg/rpm confirms the package is installed but no marker file
        // was found. Those hosts are still repo-native-managed, so the evidence
        // line must fire for them too, not just the marker-file-detected
        // literal "repo-native" case covered above. The +20 score bonus that
        // fires alongside it is asserted separately, in
        // `repo_native_broken_score_bonus_actually_raises_confidence`.
        for method in ["apt", "dnf"] {
            let mut e = linux_base();
            e.rocm_install_method = method.to_owned();
            let report = diagnose(
                &e,
                "dpkg: error processing package amdgpu-dkms (half-configured)",
            );
            let hit = report
                .matched
                .iter()
                .find(|d| d.id == "fix-12-installer")
                .expect("dpkg keyword evidence alone should still surface fix-12");
            assert!(
                hit.evidence
                    .iter()
                    .any(|l| l.contains("installed via the repo-native package-manager flow")),
                "method {method:?} is a repo-native fallback and must claim repo-native evidence: {:?}",
                hit.evidence
            );
        }
    }

    #[test]
    fn repo_native_broken_score_bonus_actually_raises_confidence() {
        // The two tests above only assert the evidence *line*, which fires
        // from the same `is_repo_native_installed(method)` check that gates
        // the `+20` bonus a few lines later -- so they'd stay green even if
        // the bonus itself were deleted. Diagnose the same dpkg symptom on a
        // repo-native host and a runfile-or-tarball host and assert the
        // former's score is exactly 20 higher.
        let symptom = "dpkg: error processing package amdgpu-dkms (half-configured)";
        let score_for = |method: &str| {
            let mut e = linux_base();
            e.rocm_install_method = method.to_owned();
            diagnose(&e, symptom)
                .matched
                .iter()
                .find(|d| d.id == "fix-12-installer")
                .expect("dpkg keyword evidence alone should still surface fix-12")
                .score
        };
        assert_eq!(
            score_for("apt"),
            score_for("runfile-or-tarball") + 20,
            "the +20 repo-native score bonus must actually raise the computed score, \
             not just the evidence text"
        );
    }

    #[test]
    fn repo_native_broken_fires_on_dpkg_symptom_regardless_of_install_method() {
        // The keyword score alone (half-configured + generic dpkg error = 75)
        // clears MIN_SCORE_FOR_MATCH without the +20 repo-native bonus, so
        // fix-12 still fires even when rocm_install_method isn't
        // "repo-native" -- the method only adds confidence, it doesn't gate
        // whether the check fires at all. See the sibling test above for the
        // repo-native-evidence-line assertion this used to also make.
        let mut e = linux_base();
        e.rocm_install_method = "runfile-or-tarball".to_owned();
        let report = diagnose(
            &e,
            "dpkg: error processing package amdgpu-dkms (half-configured)",
        );
        assert!(
            report.matched.iter().any(|d| d.id == "fix-12-installer"),
            "dpkg keyword evidence alone should still surface fix-12"
        );
    }

    #[test]
    fn dkms_driver_plan_support_matrix_covers_every_arm() {
        // (distro_id, distro_version, distro_id_like, supported). Kept in step
        // with `build_driver_install_plan_at` by
        // `driver_plan_support_matches_rocm_core_diagnose` in
        // `apps/rocm/src/driver_plan.rs`.
        let table = [
            ("ubuntu", "22.04", "debian", true),
            ("ubuntu", "24.04", "debian", true),
            ("ubuntu", "26.04", "debian", true),
            ("debian", "12", "", true),
            ("debian", "13", "", true),
            ("rhel", "9.7", "fedora", true),
            ("rhel", "10.0", "fedora", true),
            ("ol", "9.7", "fedora", true),
            ("rocky", "9.6", "rhel centos fedora", true),
            ("sles", "15.7", "suse", true),
            ("sle", "15.7", "suse", true),
            ("almalinux", "9.6", "rhel centos fedora", true),
            ("pop", "24.04", "ubuntu debian", true),
            ("ol", "9.7", "rhel fedora", true),
            ("ol", "9.5", "rhel fedora", false),
            // OL's own ID_LIKE row must not inherit the `rhel` family's
            // versions: OL 9.6 is not in the documented OL list.
            ("ol", "9.6", "rhel fedora", false),
            ("fedora", "41", "", false),
            ("opensuse-leap", "15.7", "suse opensuse", false),
            ("linuxmint", "22", "ubuntu debian", false),
        ];
        for (id, version, like, expected) in table {
            let mut e = linux_base();
            e.distro_id = id.to_owned();
            e.distro_version = version.to_owned();
            e.distro_id_like = like.to_owned();
            assert_eq!(
                dkms_driver_plan_is_supported(&e),
                expected,
                "{id} {version} (ID_LIKE={like})"
            );
        }
    }

    #[test]
    fn repo_native_broken_on_an_undocumented_distro_does_not_purge_the_driver() {
        let mut e = linux_base();
        e.distro_id = "fedora".to_owned();
        e.distro_version = "41".to_owned();
        e.rocm_install_method = "repo-native".to_owned();
        let report = diagnose(&e, "dpkg: error processing package amdgpu-dkms");
        let hit = report
            .matched
            .iter()
            .find(|d| d.id == "fix-12-installer")
            .expect("repo-native broken state should be diagnosed");
        let commands = hit.fix.as_ref().unwrap().commands.join("\n");
        assert!(
            !commands.contains("amdgpu-dkms") && !commands.contains("rocm install driver"),
            "no purge without a planned reinstall: {commands}"
        );
    }

    fn fix_commands(e: &Examination, symptom: &str, id: &str) -> String {
        diagnose(e, symptom)
            .matched
            .iter()
            .find(|d| d.id == id)
            .unwrap_or_else(|| panic!("{id} should be diagnosed"))
            .fix
            .as_ref()
            .unwrap()
            .commands
            .join("\n")
    }

    #[test]
    fn fix_12_purges_the_driver_only_where_a_reinstall_is_planned() {
        // (id, version, id_like, repo seen, purge command, expect purge)
        let cases = [
            (
                "ubuntu",
                "24.04",
                "debian",
                "/etc/apt/sources.list.d/amdgpu.list",
                "sudo apt-get purge -y amdgpu-dkms",
                true,
            ),
            (
                "linuxmint",
                "22",
                "ubuntu debian",
                "/etc/apt/sources.list.d/amdgpu.list",
                "sudo apt-get purge -y amdgpu-dkms",
                false,
            ),
            (
                "rhel",
                "9.7",
                "fedora",
                "/etc/yum.repos.d/amdgpu.repo",
                "sudo dnf remove -y amdgpu-dkms",
                true,
            ),
            (
                "fedora",
                "41",
                "",
                "/etc/yum.repos.d/amdgpu.repo",
                "sudo dnf remove -y amdgpu-dkms",
                false,
            ),
            (
                "sles",
                "15.7",
                "suse",
                "/etc/zypp/repos.d/amdgpu.repo",
                "sudo zypper remove -y amdgpu-dkms",
                true,
            ),
            (
                "opensuse-leap",
                "15.7",
                "suse opensuse",
                "/etc/zypp/repos.d/amdgpu.repo",
                "sudo zypper remove -y amdgpu-dkms",
                false,
            ),
        ];
        for (id, version, like, repo, purge, expect) in cases {
            let mut e = linux_base();
            e.distro_id = id.to_owned();
            e.distro_version = version.to_owned();
            e.distro_id_like = like.to_owned();
            e.rocm_install_method = "repo-native".to_owned();
            e.rocm_repos_seen = vec![repo.to_owned()];
            let commands = fix_commands(
                &e,
                "dpkg: error processing package amdgpu-dkms",
                "fix-12-installer",
            );
            assert_eq!(
                commands.contains(purge),
                expect,
                "{id} {version}: {commands}"
            );
        }
    }

    #[test]
    fn supported_distro_fix_3_and_fix_7_name_the_dkms_reinstall() {
        let mut e = linux_base();
        e.distro_id = "ubuntu".to_owned();
        e.distro_version = "24.04".to_owned();
        e.rocm_version = "7.0.0".to_owned();
        e.amdgpu_loaded = Some(false);
        e.rocm_repos_seen = vec![
            "/etc/apt/sources.list.d/amdgpu.list".to_owned(),
            "/etc/apt/sources.list.d/rocm.list".to_owned(),
        ];
        let fix3 = fix_commands(&e, "kernel/rocm mismatch", "fix-3-rocm-kernel");
        assert!(fix3.contains("rocm install driver --dkms --yes"), "{fix3}");
        let fix7 = fix_commands(&e, "stale repo conflict", "fix-7-stale-repos");
        assert!(fix7.contains("rocm install driver --dkms --yes"), "{fix7}");
    }

    #[test]
    fn kernel_unsupported_does_not_offer_the_sdk_install_or_a_maskable_verify() {
        let mut e = linux_base();
        e.distro_id = "ubuntu".to_owned();
        e.distro_version = "24.04".to_owned();
        e.rocm_version = "7.0.0".to_owned();
        e.amdgpu_loaded = Some(false);
        let report = diagnose(&e, "kernel/rocm mismatch");
        let fix = report
            .matched
            .iter()
            .find(|d| d.id == "fix-3-rocm-kernel")
            .and_then(|d| d.fix.as_ref())
            .expect("unsupported kernel triple should be diagnosed");
        assert!(!fix.commands.join("\n").contains("rocm install sdk"));
        assert!(!fix.verify.contains("| head"), "{}", fix.verify);
    }

    // ── fix-24: unified memory GTT/TTM limits ──────────────────────

    /// A Strix Halo-class APU with 128 GiB of RAM and nothing set.
    fn apu_host() -> Examination {
        let mut e = linux_base();
        e.has_apu = true;
        e.system_ram_gib = Some(128.0);
        e.gpus = vec![Gpu {
            gfx_target: "gfx1151".to_owned(),
            is_amd: true,
            is_apu: Some(true),
            ..Gpu::default()
        }];
        e
    }

    /// The headline finding: the pair is set, and the page limit is the one
    /// that decides.
    ///
    /// Asserted against the arithmetic rather than against a repeated literal:
    /// 98304 MiB needs 25165824 pages, and 8388608 pages is only 32768 MiB.
    /// The evidence has to name that capped figure, because "the parameters
    /// are set" is exactly what the user already believes.
    #[test]
    fn an_inconsistent_gttsize_and_pages_limit_pair_is_named_as_the_cap() {
        let mut e = apu_host();
        e.amdgpu_gttsize_mib = Some(98_304);
        e.ttm_pages_limit = Some(8_388_608);
        let finding = check_24_apu_unified_memory(&e, "");
        assert!(
            finding.score >= MIN_SCORE_FOR_MATCH,
            "an inconsistent pair must match on host state alone: {finding:?}"
        );
        let evidence = finding.evidence.join("\n");
        assert!(
            evidence.contains("capped at 32768 MiB"),
            "the evidence must name the ceiling the page limit really imposes: {evidence}"
        );
        assert!(
            evidence.contains("ttm.pages_limit=8388608")
                && evidence.contains("amdgpu.gttsize=98304"),
            "both parameters must be quoted as set: {evidence}"
        );

        // The same host with a pair that agrees is not told anything is wrong.
        let mut consistent = apu_host();
        consistent.amdgpu_gttsize_mib = Some(98_304);
        consistent.ttm_pages_limit = Some(98_304 * TTM_PAGES_PER_MIB);
        let ok = check_24_apu_unified_memory(&consistent, "");
        assert_eq!(
            ok.score, 0,
            "a consistent pair with a clean log is not a finding: {ok:?}"
        );
    }

    /// The recommendation is sized from installed RAM, and the two numbers it
    /// prints are consistent with each other by construction.
    #[test]
    fn the_recommended_pair_is_sized_to_installed_ram_and_is_self_consistent() {
        let mut e = apu_host();
        e.amdgpu_gttsize_mib = Some(98_304);
        e.ttm_pages_limit = Some(8_388_608);
        let fix = check_24_apu_unified_memory(&e, "")
            .fix
            .expect("a matched finding carries a fix");
        // Three quarters of 128 GiB is 96 GiB = 98304 MiB = 25165824 pages.
        assert!(
            fix.commands
                .iter()
                .any(|c| c.contains("amdgpu.gttsize=98304 ttm.pages_limit=25165824")),
            "the plan must print the sized pair: {:?}",
            fix.commands
        );
        assert!(fix.needs_sudo && fix.needs_reboot, "{fix:?}");
        // The carve-out is firmware, so it is advice and not a command line.
        assert!(
            fix.notes
                .iter()
                .any(|n| n.contains("UMA Frame Buffer Size")),
            "the BIOS carve-out belongs in the notes: {:?}",
            fix.notes
        );
        assert!(
            !fix.commands
                .iter()
                .any(|c| c.to_lowercase().contains("bios")),
            "nothing about the carve-out may look like a command to run: {:?}",
            fix.commands
        );
        // The verify step has to read the parameter back off the booted
        // kernel, not report that an editor exited.
        assert!(
            fix.verify.contains("/proc/cmdline") && fix.verify.contains("gttsize"),
            "verify must re-read the parameter: {}",
            fix.verify
        );

        // A host whose RAM could not be read is told to size the pair itself
        // rather than given a figure derived from nothing.
        let mut unknown_ram = e;
        unknown_ram.system_ram_gib = None;
        let fix = check_24_apu_unified_memory(&unknown_ram, "")
            .fix
            .expect("a matched finding carries a fix");
        assert!(
            fix.commands.iter().any(|c| c.trim() == "free -g"),
            "with no RAM figure the plan must show how to read it: {:?}",
            fix.commands
        );
        assert!(
            !fix.commands.iter().any(|c| c.contains("gttsize=98304")),
            "no figure may be invented when RAM is unknown: {:?}",
            fix.commands
        );
    }

    /// The recommendation must never be the thing that trips the kernel limit
    /// whose refusal prints `SVM mapping failed`.
    ///
    /// `amdgpu_amdkfd_gpuvm_init_mem_limits` caps host memory the KFD hands out
    /// at `RAM - RAM/64 - 1.5 GiB`; this is the arithmetic that
    /// [`RECOMMENDED_GTT_SHARE_OF_RAM`]'s doc comment claims, run over the RAM
    /// sizes an APU actually ships with.
    #[test]
    fn the_recommended_pair_stays_under_the_kernels_own_memory_limit() {
        for ram_gib in [8.0, 16.0, 32.0, 64.0, 96.0, 128.0, 256.0] {
            let (mib, pages) = recommended_limits(ram_gib).expect("a usable RAM figure");
            let kernel_limit_gib = ram_gib - ram_gib / 64.0 - 1.5;
            assert!(
                (mib / 1024) as f64 <= kernel_limit_gib,
                "{ram_gib} GiB: recommending {} GiB exceeds the kernel's own \
                 {kernel_limit_gib} GiB limit",
                mib / 1024
            );
            assert_eq!(
                pages,
                mib * TTM_PAGES_PER_MIB,
                "{ram_gib} GiB: the recommended pair must be consistent with itself"
            );
        }
        // Figures that are not a measurement yield no recommendation at all.
        for bad in [0.0, -1.0, f64::NAN, 0.5] {
            assert_eq!(recommended_limits(bad), None, "{bad}");
        }
    }

    /// The SVM failure reaches the report, and its absence is reported as
    /// "not in the window that was read" rather than as "it never happened".
    #[test]
    fn an_svm_mapping_failure_is_reported_and_its_absence_is_qualified() {
        let mut e = apu_host();
        e.dmesg_amdgpu_tail = vec![
            "[ 1234.567890] amdgpu: SVM mapping failed, exceeds resident system memory limit"
                .to_owned(),
        ];
        let finding = check_24_apu_unified_memory(&e, "");
        assert!(
            finding.score >= MIN_SCORE_FOR_MATCH,
            "the kernel's own SVM failure is enough on its own: {finding:?}"
        );
        assert!(
            finding
                .evidence
                .iter()
                .any(|line| line.contains("SVM mapping failed, exceeds resident system memory")),
            "the matched kernel line must be quoted: {:?}",
            finding.evidence
        );

        // Without it, the finding says where it looked and does not claim the
        // machine never failed.
        let mut clean = apu_host();
        clean.amdgpu_gttsize_mib = Some(98_304);
        clean.ttm_pages_limit = Some(8_388_608);
        let qualified = check_24_apu_unified_memory(&clean, "");
        let window = qualified
            .evidence
            .iter()
            .find(|line| line.contains("SVM mapping failure"))
            .expect("the log verdict is always stated");
        assert!(
            window.contains("recent kernel log")
                && window.contains(&crate::examine::DMESG_LINES_SCANNED.to_string())
                && window.contains("would not be seen here"),
            "the absence has to be bounded by the window that was read: {window}"
        );
    }

    /// A host with no integrated GPU, neither parameter set and a clean log is
    /// not given unified-memory advice at all, and this entry never fires off
    /// bare-metal Linux.
    #[test]
    fn an_ordinary_discrete_linux_host_is_told_nothing_about_unified_memory() {
        let mut e = linux_base();
        e.has_discrete_amd = true;
        e.system_ram_gib = Some(64.0);
        assert_eq!(check_24_apu_unified_memory(&e, "").score, 0);

        let report = diagnose(&e, "");
        assert!(
            report
                .matched
                .iter()
                .all(|d| d.id != crate::diagnose::APU_UNIFIED_MEMORY_FIX_ID),
            "{:?}",
            report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
        );

        // The same facts on WSL2 reach no unified-memory entry either: the
        // distro's own kernel command line does not govern a /dev/dxg GPU.
        let mut wsl = crate::diagnose::test_support::wsl_base();
        wsl.has_apu = true;
        wsl.amdgpu_gttsize_mib = Some(98_304);
        wsl.ttm_pages_limit = Some(8_388_608);
        let report = diagnose(&wsl, "");
        assert!(
            report
                .matched
                .iter()
                .all(|d| d.id != crate::diagnose::APU_UNIFIED_MEMORY_FIX_ID),
            "{:?}",
            report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
        );
    }

    /// An APU that has set neither parameter is advised, but only advised: the
    /// entry stays below the match threshold on that alone, because a default
    /// aperture is a tuning opportunity, not a fault.
    #[test]
    fn an_untuned_apu_is_advised_without_being_told_it_is_broken() {
        let finding = check_24_apu_unified_memory(&apu_host(), "");
        assert!(
            finding.score > 0 && finding.score < MIN_SCORE_FOR_MATCH,
            "an untuned APU is advice, not a diagnosis: {finding:?}"
        );
        // The user's own paste of the kernel line is what turns it into one.
        let matched = check_24_apu_unified_memory(
            &apu_host(),
            "amdgpu: SVM mapping failed, exceeds resident system memory limit",
        );
        assert!(
            matched.score >= MIN_SCORE_FOR_MATCH,
            "the pasted kernel line must carry it over the threshold: {matched:?}"
        );
    }
}
