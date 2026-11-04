// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The checks that apply on Linux only: the kernel and the amdgpu module, device
//! node access, the package repositories and the repo-native install, and the
//! helpers that pick which package manager's commands to print.

use super::scoring::{
    KEYWORDS_CONTAINER, KEYWORDS_DPKG_BROKEN, KEYWORDS_IOMMU_HANG, KEYWORDS_KFD_PERMISSION,
    KEYWORDS_MODULE_NOT_LOADED, KEYWORDS_REPO_BROKEN, amd_gpu_count, finalize, keyword_score, zero,
};
use super::{Diagnosis, Fix};
use crate::examine::Examination;

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
            "ROCm {rocm_version} is installed but the amdgpu kernel module is not loaded; this is typical when DKMS failed against an unsupported kernel."
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
    let (kw_score, kw_ev) = keyword_score(symptom, KEYWORDS_KFD_PERMISSION);
    score += kw_score;
    evidence.extend(kw_ev);

    if score <= 0 {
        return zero("fix-4-render-group", "User missing render/video group");
    }
    // `stat -c %G` prints the literal "UNKNOWN" when the device GID has no
    // matching group name (common in containers). Treat that the same as an
    // empty value and fall back to the render group, so the suggested command
    // never names a group that does not exist.
    let kfd_group = e
        .kfd
        .as_ref()
        .map(|k| k.owner_group.clone())
        .filter(|g| !g.is_empty() && !g.eq_ignore_ascii_case("UNKNOWN"))
        .unwrap_or_else(|| "render".to_owned());
    let fix = Fix {
        summary: format!("Add the current user to '{kfd_group}' (and 'video' for safety) and log out/in."),
        commands: vec![format!("sudo usermod -a -G {kfd_group},video \"$USER\"")],
        needs_sudo: true,
        needs_relogin: true,
        fix_id: "fix-4-render-group".to_owned(),
        verify: "groups | tr ' ' '\\n' | grep -E '^(render|video)$' && ls -l /dev/kfd && rocminfo | head -n 5".to_owned(),
        notes: vec![
            "Group membership only takes effect after a full re-login (or reboot). `newgrp render` will give the current shell access but not other terminals or services.".to_owned(),
        ],
        ..Fix::default()
    };
    finalize(
        "fix-4-render-group",
        "User not in render/video group (or /dev/kfd owned by the other group)",
        score,
        evidence,
        fix,
    )
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
        evidence.push("amdgpu module is not loaded".to_owned());
    }
    if e.rocminfo_status == "not-loaded" {
        score += 25;
        evidence.push("rocminfo says 'ROCk module is NOT loaded'".to_owned());
    }
    if e.secure_boot == "enabled" && e.amdgpu_loaded == Some(false) {
        score += 10;
        evidence.push("Secure Boot is enabled and amdgpu didn't load -- DKMS modules are often blocked until you sign them or disable Secure Boot.".to_owned());
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
        // Catalog-aligned, not state-derived: fix.rs's FixRecipe for this fix-id sets
        // needs_reboot unconditionally. On this no-blacklist path the plan is just
        // `modprobe amdgpu` (or that plus a Secure Boot signing note, whose own
        // remedy can require a reboot) -- diagnose conforms to the catalog rather
        // than the reverse (see fix.rs's assert_needs_reboot_matches_the_catalog) so
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::test_support::linux_base;
    use crate::diagnose::{HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH, diagnose, render_report_text};
    use crate::examine::{Device, Gpu};

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

    #[test]
    fn unknown_kfd_group_falls_back_to_render() {
        // `stat -c %G /dev/kfd` prints "UNKNOWN" when the GID has no group
        // name. That value must not leak into the remediation command, which
        // would otherwise suggest `usermod -a -G UNKNOWN,video` -- a group that
        // does not exist. Fall back to the render group instead. Exercise both
        // casings so the case-insensitive guard cannot silently regress to an
        // exact-match check while this test stays green.
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
        }
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
}
