// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm examine` text rendering of an [`ExamineSummary`].

use super::ExamineSummary;
use super::os_facts::format_gib_value;
use crate::gfx_target::{GpuSupportLevel, gpu_support_level};

impl ExamineSummary {
    /// The `rocm_support` line of the examine text.
    ///
    /// ROCMAI-550 asks `rocm examine` to "report the support level with one
    /// line of what the user can do", so every level gets a line — including
    /// the ones where the answer is "this build cannot tell". The earlier draft
    /// printed only the `VulkanOnly` line, which made this report unable to
    /// answer the very question the other surfaces send people here to ask:
    /// `lemonade_engine_note` in `apps/rocm/src/main.rs` tells an `Unknown`
    /// host to run `rocm examine`, and silence there is not an answer.
    ///
    /// Without the `VulkanOnly` line specifically,
    /// `compatible_therock_family: <unknown>` reads like a probe failure when
    /// the GPU was in fact recognized and simply has no ROCm build (the gfx90c
    /// Vega iGPU in Renoir / Cezanne APUs). The `Unknown` line exists for the
    /// mirror-image misreading: the same `<unknown>` value, where the CLI
    /// genuinely does not know, must not be read as either claim.
    ///
    /// Classified by [`gpu_support_level`] rather than re-deriving the same
    /// `(detected, covering family)` test here: this line and that function are
    /// two renderings of one judgement, and the whole point of this change is
    /// that support claims come from a single place. Every variant is spelled
    /// out so that adding one is a compile error here rather than a silently
    /// blank line.
    fn rocm_support_line(&self) -> String {
        let detected = self.detected_gfx_target.as_deref();
        match gpu_support_level(detected, self.compatible_therock_family.as_deref()) {
            // `VulkanOnly` is only ever returned for a detected target, so the
            // fallback below is unreachable; it keeps the arm total without a
            // panic path.
            GpuSupportLevel::VulkanOnly => format!(
                "  rocm_support: GPU detected ({}) but not supported by ROCm; \
                 the Vulkan backend can still be used - serve on it with \
                 `rocm serve <model> --engine lemonade`\n",
                detected.unwrap_or_default()
            ),
            // The covering family is the fact; it is already printed on its own
            // line above, so this one says what it means rather than repeating
            // it.
            //
            // The advice is conditional because AGENTS.md wants a named command
            // to be able to clear the condition it is printed for. On a host
            // that already has a managed SDK, `rocm install sdk` clears
            // nothing — it is the install that is already done — so the line
            // says so and points at what is left to do instead.
            // `detected_therock_family` is the installed managed runtime (see
            // `detect_managed_therock_family`), as opposed to
            // `compatible_therock_family`, which is what this GPU *could* run.
            GpuSupportLevel::RocmHip => match self.detected_therock_family.as_deref() {
                Some(installed) => format!(
                    "  rocm_support: GPU detected ({}) with full ROCm/HIP support, and the \
                     managed ROCm SDK for {installed} is already installed - serve on it with \
                     `rocm serve <model>`\n",
                    detected.unwrap_or("<unknown>")
                ),
                None => format!(
                    "  rocm_support: GPU detected ({}) with full ROCm/HIP support; \
                     install the SDK for it with `rocm install sdk`\n",
                    detected.unwrap_or("<unknown>")
                ),
            },
            // Not folded in with `VulkanOnly`: a detection gap is not a
            // confirmed absence of ROCm support, and saying "Vulkan" here would
            // be the same false claim in the other direction. What the user can
            // do is let an engine decide and then read back what it chose -
            // which is the `device:` row this same ticket added, and the only
            // thing on this host that can actually settle the question.
            GpuSupportLevel::Unknown => format!(
                "  rocm_support: unknown - {}, so this build can confirm neither \
                 ROCm support nor its absence; `rocm serve <model> --engine \
                 lemonade` reports the backend it picked on its `device:` row\n",
                match detected {
                    Some(gfx) => format!("GPU target {gfx} is not in this build's catalog"),
                    None => "no AMD GPU was detected".to_owned(),
                }
            ),
        }
    }

    pub fn render_text(&self) -> String {
        let legacy_paths = if self.legacy_rocm.paths.is_empty() {
            "<none>".to_owned()
        } else {
            self.legacy_rocm
                .paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let wsl = self.wsl.as_ref();
        // Every other field here describes the MACHINE; this one describes the
        // invocation, which is what made it ambiguous -- `true` from a terminal
        // and `false` under the dashboard both look like claims about the host.
        // Say which it is on the line itself, so pasted output explains itself.
        let interactive_terminal = if self.interactive_terminal {
            "true (this run has a terminal; the CLI may prompt)"
        } else {
            "false (this run's output is captured, so the CLI will not prompt)"
        };
        format!(
            "rocm examine\n  os: {}\n  arch: {}\n  kernel: {}\n  distro: {}\n  cpu: {}\n  system_ram: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  compatible_therock_family: {}\n{}  detected_therock_family: {}\n  driver_policy: {}\n  driver_status: {}\n  driver_detail: {}\n{}  legacy_rocm_status: {}\n  legacy_rocm_paths: {}\n  legacy_rocm_version: {}\n  legacy_rocm_detail: {}\n  legacy_rocm_guidance: {}\n  wsl: {}\n  wsl_dxg_device: {}\n  wsl_dxcore: {}\n  wsl_librocdxg: {}\n  wsl_rocdxg_dids: {}\n  wsl_ldconfig_librocdxg: {}\n  wsl_global_rocminfo: {}\n  wsl_cargo: {}\n  wsl_detail: {}\n  managed_runtimes: {}\n  managed_services: {}\n  model_cache_entries: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
            self.os,
            self.arch,
            self.kernel.as_deref().unwrap_or("<unknown>"),
            self.distro.as_deref().unwrap_or("<unknown>"),
            self.cpu.as_deref().unwrap_or("<unknown>"),
            self.system_ram_gib
                .map_or_else(|| "<unknown>".to_owned(), format_gib_value),
            interactive_terminal,
            self.default_engine,
            self.detected_gfx_target.as_deref().unwrap_or("<unknown>"),
            self.compatible_therock_family
                .as_deref()
                .unwrap_or("<unknown>"),
            self.rocm_support_line(),
            self.detected_therock_family
                .as_deref()
                .unwrap_or("<not detected>"),
            self.driver.policy,
            self.driver.status,
            self.driver.detail.as_deref().unwrap_or("<unknown>"),
            self.windows_driver_lines(),
            self.legacy_rocm.status,
            legacy_paths,
            self.legacy_rocm.version.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm.detail.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm_guidance(),
            wsl.is_some_and(|summary| summary.is_wsl),
            wsl.is_some_and(|summary| summary.dxg_device),
            wsl.is_some_and(|summary| summary.dxcore),
            wsl.is_some_and(|summary| summary.librocdxg),
            wsl.is_some_and(|summary| summary.rocdxg_dids),
            wsl.is_some_and(|summary| summary.ldconfig_librocdxg),
            wsl.is_some_and(|summary| summary.rocminfo),
            wsl.is_some_and(|summary| summary.cargo),
            wsl.and_then(|summary| summary.detail.as_deref())
                .unwrap_or("<not WSL>"),
            self.managed_runtime_count,
            self.managed_service_count,
            self.model_cache_entries,
            self.config_dir.display(),
            self.data_dir.display(),
            self.cache_dir.display(),
        )
    }

    /// The `driver_version`, `driver_date` and `driver_age` lines. Empty unless
    /// this is a Windows host where a driver version was read, so other
    /// platforms print exactly what they printed before.
    fn windows_driver_lines(&self) -> String {
        let Some(info) = self.driver.windows.as_ref() else {
            return String::new();
        };
        format!(
            "  driver_version: {}\n  driver_date: {}\n  driver_age: {}\n",
            info.version,
            info.date
                .map_or_else(|| "<unknown>".to_owned(), |d| d.to_string()),
            crate::windows_driver::age_summary(info),
        )
    }

    const fn legacy_rocm_guidance(&self) -> &'static str {
        if self.legacy_rocm.paths.is_empty() {
            return "none";
        }
        if self.managed_runtime_count == 0 {
            return "legacy ROCm detected; install a managed TheRock runtime with `rocm install sdk --channel release --format wheel` and keep legacy ROCm unmanaged";
        }
        "legacy ROCm detected; keep it side-by-side and use rocm-cli managed TheRock runtimes for local engines"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LegacyRocmSummary;
    use crate::gfx_target::normalize_therock_family;
    use crate::host_gpu::DriverSummary;
    use crate::host_gpu::tests::windows_summary_with_driver;
    use std::path::PathBuf;

    #[test]
    fn examine_render_includes_driver_and_state_counts() {
        let summary = ExamineSummary {
            os: "windows".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: Some("10.0.26100".to_owned()),
            distro: Some("Windows".to_owned()),
            cpu: Some("AMD Ryzen".to_owned()),
            system_ram_gib: Some(64.0),
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: Some("gfx120X-all".to_owned()),
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "windows_validate_only".to_owned(),
                status: "amd_display_driver_detected".to_owned(),
                detail: Some("AMD Radeon driver 1.2.3".to_owned()),
                windows: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "detected_unmanaged".to_owned(),
                paths: vec![PathBuf::from("C:\\Program Files\\AMD\\ROCm")],
                detail: Some("legacy install".to_owned()),
                version: Some("6.4.1".to_owned()),
            },
            wsl: None,
            managed_runtime_count: 2,
            managed_service_count: 1,
            model_cache_entries: 3,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let rendered = summary.render_text();
        assert!(rendered.contains("distro: Windows"));
        assert!(rendered.contains("cpu: AMD Ryzen"));
        assert!(rendered.contains("system_ram: 64 GiB"));
        assert!(rendered.contains("compatible_therock_family: gfx120X-all"));
        assert!(rendered.contains("detected_therock_family: <not detected>"));
        assert!(rendered.contains("driver_policy: windows_validate_only"));
        assert!(rendered.contains("driver_status: amd_display_driver_detected"));
        assert!(rendered.contains("legacy_rocm_status: detected_unmanaged"));
        assert!(rendered.contains("legacy_rocm_paths: C:\\Program Files\\AMD\\ROCm"));
        assert!(
            rendered.contains("legacy_rocm_guidance: legacy ROCm detected; keep it side-by-side")
        );
        assert!(rendered.contains("wsl: false"));
        assert!(rendered.contains("managed_runtimes: 2"));
        assert!(rendered.contains("managed_services: 1"));
        assert!(rendered.contains("model_cache_entries: 3"));
    }

    fn examine_summary_with_gpu(
        detected_gfx_target: Option<&str>,
        compatible_therock_family: Option<&str>,
    ) -> ExamineSummary {
        ExamineSummary {
            os: "windows".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "lemonade".to_owned(),
            detected_gfx_target: detected_gfx_target.map(str::to_owned),
            compatible_therock_family: compatible_therock_family.map(str::to_owned),
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "windows_validate_only".to_owned(),
                status: "amd_display_driver_detected".to_owned(),
                detail: None,
                windows: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "not_detected".to_owned(),
                paths: Vec::new(),
                detail: None,
                version: None,
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        }
    }

    #[test]
    fn examine_render_says_a_gpu_without_a_rocm_family_is_unsupported() {
        // gfx90c is detected from its PCI ID but has no ROCm package family.
        let family = normalize_therock_family("gfx90c");
        let summary = examine_summary_with_gpu(Some("gfx90c"), family.as_deref());
        let rendered = summary.render_text();
        assert!(
            rendered.contains("detected_gfx_target: gfx90c"),
            "{rendered}"
        );
        assert!(
            rendered.contains("rocm_support: GPU detected (gfx90c) but not supported by ROCm"),
            "{rendered}"
        );
        assert!(
            rendered.contains("the Vulkan backend can still be used"),
            "{rendered}"
        );
    }

    /// An unmapped target must not be *claimed* unsupported, even though every
    /// host now gets a `rocm_support` line.
    ///
    /// This replaces `examine_render_omits_the_support_note_for_supported_or_missing_gpus`,
    /// which asserted the line was absent for supported, missing and unmapped
    /// GPUs. ROCMAI-550 deliberately reverses the absence: silence could not
    /// answer the question `lemonade_engine_note` sends users here to ask. The
    /// half of that test worth keeping is the discrimination it protected —
    /// "no ROCm build" is a fact about gfx90c, and a gap in this catalog is not
    /// the same claim about gfx803.
    #[test]
    fn examine_render_never_claims_an_unmapped_target_is_unsupported() {
        for (gfx, family) in [
            ("gfx1103", normalize_therock_family("gfx1103")),
            ("gfx803", None),
        ] {
            let rendered = examine_summary_with_gpu(Some(gfx), family.as_deref()).render_text();
            assert!(
                rendered.contains("rocm_support"),
                "every host gets a support line now: {rendered}"
            );
        }

        // Covered by a family: the line must not say it has no ROCm build.
        let family = normalize_therock_family("gfx1103");
        let supported = examine_summary_with_gpu(Some("gfx1103"), family.as_deref()).render_text();
        assert!(
            !supported.contains("not supported by ROCm"),
            "a covered GPU must not be reported as unsupported: {supported}"
        );

        // Unmapped: a gap in the catalog, NOT a confirmed absence of support.
        let unknown = examine_summary_with_gpu(Some("gfx803"), None).render_text();
        assert!(
            !unknown.contains("not supported by ROCm"),
            "an unmapped target must not be claimed unsupported: {unknown}"
        );

        // Nothing detected: likewise a gap, not a verdict about hardware.
        let missing = examine_summary_with_gpu(None, None).render_text();
        assert!(
            missing.contains("rocm_support"),
            "an undetected GPU still gets a line: {missing}"
        );
        assert!(
            !missing.contains("not supported by ROCm"),
            "an undetected GPU must not be claimed unsupported: {missing}"
        );
    }

    #[test]
    fn examine_shows_the_windows_driver_version_date_and_age() {
        let old = windows_summary_with_driver(
            crate::windows_driver::WindowsDriverInfo::from_probe("30.0.13014.8", "2021-08-04"),
        )
        .render_text();
        assert!(old.contains("  driver_version: 30.0.13014.8\n"));
        assert!(old.contains("  driver_date: 2021-08-04\n"));
        assert!(old.contains("  driver_age: older than the first HIP SDK for Windows"));
        assert!(old.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));
        assert!(old.contains("not established"));

        let current = windows_summary_with_driver(
            crate::windows_driver::WindowsDriverInfo::from_probe("32.0.23033.1002", "2026-05-02"),
        )
        .render_text();
        assert!(current.contains("  driver_version: 32.0.23033.1002\n"));
        assert!(current.contains("  driver_age: not older than"));
        assert!(!current.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));

        let undated = windows_summary_with_driver(
            crate::windows_driver::WindowsDriverInfo::from_probe("30.0.13014.8", ""),
        )
        .render_text();
        assert!(undated.contains("  driver_date: <unknown>\n"));
        assert!(undated.contains("  driver_age: unknown"));
        assert!(!undated.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));
    }

    #[test]
    fn examine_prints_no_driver_age_lines_when_no_version_was_read() {
        let rendered = windows_summary_with_driver(None).render_text();
        assert!(!rendered.contains("driver_version:"));
        assert!(!rendered.contains("driver_age:"));
        assert!(rendered.contains("driver_detail: AMD Radeon(TM) Graphics driver"));
    }

    #[test]
    fn examine_render_explains_what_interactive_terminal_means() {
        // The same machine reports `true` from a shell and `false` under the
        // dashboard, because the field describes the invocation rather than the
        // host. Both are correct, and the line has to say so on its own — a
        // pasted report is usually all a reader gets.
        let mut summary = ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: true,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
                windows: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "not_detected".to_owned(),
                paths: Vec::new(),
                detail: None,
                version: None,
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let interactive = summary.render_text();
        assert!(
            interactive.contains("interactive_terminal: true (this run has a terminal"),
            "the true case must say it is about this run:\n{interactive}"
        );

        summary.interactive_terminal = false;
        let captured = summary.render_text();
        assert!(
            captured.contains("interactive_terminal: false (this run's output is captured"),
            "the false case must explain why, not just report it:\n{captured}"
        );
        // The reason it matters to the reader: it is why they saw no prompt.
        assert!(
            captured.contains("will not prompt"),
            "the false case must connect to the visible consequence:\n{captured}"
        );
    }

    #[test]
    fn examine_render_guides_managed_runtime_install_when_only_legacy_rocm_exists() {
        let summary = ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
                windows: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "detected_unmanaged".to_owned(),
                paths: vec![PathBuf::from("/opt/rocm")],
                detail: Some("legacy install".to_owned()),
                version: Some("7.14.0".to_owned()),
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let rendered = summary.render_text();

        assert!(rendered.contains(
            "legacy_rocm_guidance: legacy ROCm detected; install a managed TheRock runtime"
        ));
        assert!(rendered.contains("rocm install sdk --channel release --format wheel"));
    }

    #[test]
    fn examine_answers_the_support_question_the_unknown_engine_note_sends_users_here_for() {
        for detected in [None, Some("gfx803")] {
            let rendered = examine_summary_with_gpu(detected, None).render_text();
            let line = rendered
                .lines()
                .find(|line| line.trim_start().starts_with("rocm_support:"))
                .unwrap_or_else(|| {
                    panic!(
                        "`rocm examine` is where the Unknown-tier engine note sends \
                         the user; it must print a support level:\n{rendered}"
                    )
                });
            // Says which level it is, and leaves a command that can settle it.
            assert!(line.contains("unknown"), "{line}");
            assert!(
                line.contains("rocm serve <model> --engine lemonade"),
                "an `unknown` answer still has to leave somewhere to go: {line}"
            );
            assert!(
                line.contains("device:"),
                "and name where the real backend shows up: {line}"
            );
        }
    }

    #[test]
    fn examine_render_reports_a_support_level_for_every_host() {
        let family = normalize_therock_family("gfx1103");
        let supported = examine_summary_with_gpu(Some("gfx1103"), family.as_deref());
        let rendered = supported.render_text();
        assert!(
            rendered.contains("rocm_support: GPU detected (gfx1103) with full ROCm/HIP support"),
            "{rendered}"
        );
        assert!(rendered.contains("rocm install sdk"), "{rendered}");

        // Same GPU, but the SDK it was being told to install is already there:
        // the advice has to be something that is still left to do (AGENTS.md
        // wants a named command to be able to clear the condition).
        let mut installed = supported;
        installed.detected_therock_family = Some("gfx110X-dgpu".to_owned());
        let rendered = installed.render_text();
        assert!(
            rendered.contains("rocm_support: GPU detected (gfx1103) with full ROCm/HIP support"),
            "{rendered}"
        );
        assert!(
            rendered.contains("managed ROCm SDK for gfx110X-dgpu is already installed"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("rocm install sdk"),
            "an install that is already done must not be the advice: {rendered}"
        );

        // No GPU found: the level is reported as unknown, and nothing about the
        // hardware is asserted in either direction.
        let missing = examine_summary_with_gpu(None, None);
        let rendered = missing.render_text();
        assert!(
            rendered.contains("rocm_support: unknown - no AMD GPU was detected"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("Vulkan backend can still be used"),
            "nothing was detected, so no backend may be promised: {rendered}"
        );

        // An unmapped target the CLI knows nothing about is not claimed to be
        // unsupported; only targets known to have no ROCm build are.
        let unknown = examine_summary_with_gpu(Some("gfx803"), None);
        let rendered = unknown.render_text();
        assert!(
            rendered.contains(
                "rocm_support: unknown - GPU target gfx803 is not in this build's catalog"
            ),
            "{rendered}"
        );
        assert!(
            !rendered.contains("not supported by ROCm"),
            "an unrecognized target must not be reported as confirmed \
             ROCm-less: {rendered}"
        );
        assert!(
            !rendered.contains("full ROCm/HIP support"),
            "nor as supported: {rendered}"
        );
    }
}
