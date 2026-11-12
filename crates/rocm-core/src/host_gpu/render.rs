// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm examine` text rendering of an [`ExamineSummary`].

use super::ExamineSummary;
use super::os_facts::format_gib_value;
use super::vulkan::VulkanStatus;
use crate::gfx_target::{GpuSupportLevel, gpu_support_level};

impl ExamineSummary {
    /// What the Vulkan probe found, or [`VulkanStatus::Unknown`] when this
    /// summary carries no probe result — an older host's JSON read back by
    /// `rocm remote doctor`, or a caller that built the struct by hand.
    ///
    /// Public because the `rocm` binary classifies its own support level from a
    /// summary it already gathered, and must read the probe result the same way
    /// this module does rather than reaching into the field and inventing its
    /// own `None` handling.
    #[must_use]
    pub fn vulkan_status(&self) -> VulkanStatus {
        self.vulkan
            .as_ref()
            .map_or(VulkanStatus::Unknown, |vulkan| vulkan.status)
    }

    /// The `vulkan` line: the status, and the evidence behind it.
    fn vulkan_line(&self) -> String {
        let Some(vulkan) = self.vulkan.as_ref() else {
            return "<not probed>".to_owned();
        };
        let status = match vulkan.status {
            VulkanStatus::Available => "available",
            VulkanStatus::DriverPresentDevicesUnverified => "driver_present_devices_unverified",
            VulkanStatus::Absent => "absent",
            VulkanStatus::Inconclusive => "inconclusive",
            VulkanStatus::Unknown => "unknown",
        };
        vulkan.detail.as_deref().map_or_else(
            || status.to_owned(),
            |detail| format!("{status} ({detail})"),
        )
    }

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
        let vulkan = self.vulkan_status();
        match gpu_support_level(detected, self.compatible_therock_family.as_deref(), vulkan) {
            // `VulkanOnly` is only ever returned for a detected target, so the
            // fallback below is unreachable; it keeps the arm total without a
            // panic path.
            //
            // The claim is graded by what the probe actually established. An
            // enumerated device is named, because naming it is what lets the
            // reader check the claim rather than take it; an installed-but-
            // unenumerated driver says so instead of borrowing the stronger
            // sentence; and a platform that could not be probed admits it. All
            // three still point at the same command, because all three are
            // hosts worth trying.
            GpuSupportLevel::VulkanOnly => format!(
                "  rocm_support: GPU detected ({}) but not supported by ROCm; {} - serve on it \
                 with `rocm serve <model> --engine lemonade`\n",
                detected.unwrap_or_default(),
                match vulkan {
                    VulkanStatus::Available => self.vulkan.as_ref().map_or_else(
                        || "the Vulkan backend can be used".to_owned(),
                        |support| support.driver_label().map_or_else(
                            || "the Vulkan backend can be used".to_owned(),
                            |driver| format!("the Vulkan backend can be used ({driver})"),
                        ),
                    ),
                    VulkanStatus::DriverPresentDevicesUnverified =>
                        "a Vulkan driver is installed, so the Vulkan backend is expected to work"
                            .to_owned(),
                    VulkanStatus::Unknown =>
                        "Vulkan was not probed on this platform, so the Vulkan backend is \
                         unconfirmed"
                            .to_owned(),
                    // Unreachable: `gpu_support_level` routes `Inconclusive` to
                    // `VulkanUnconfirmed` and `Absent` to `NoGpuBackend`, so
                    // neither can arrive under `VulkanOnly`. Kept spelled out
                    // rather than folded into a `_` so that adding a
                    // `VulkanStatus` stays a compile error here, and worded
                    // hedged rather than promising so that a future routing
                    // change cannot silently resurrect the claim this level
                    // exists to stop making.
                    VulkanStatus::Inconclusive | VulkanStatus::Absent =>
                        "the Vulkan backend is unconfirmed".to_owned(),
                }
            ),
            // Looked at, nothing found, and nothing that could have found it.
            // Says what was searched and how to settle it, rather than either
            // promising a backend or declaring one absent — neither of which
            // the evidence supports. See `GpuSupportLevel::VulkanUnconfirmed`.
            GpuSupportLevel::VulkanUnconfirmed => format!(
                // Deliberately does NOT say "install `vulkaninfo`". Measured on
                // a real driverless Linux host: `vulkaninfo --summary` is
                // installed, prints "Found no drivers!" and exits 1 — which
                // lands here. Telling that user to install the tool they are
                // already running is advice that can never clear the state.
                // Install a *driver* is the step that can.
                "  rocm_support: GPU detected ({}) but not supported by ROCm; no Vulkan driver \
                 was found, and `vulkaninfo` could not confirm that either (not installed, or \
                 it failed to run) - install a Vulkan driver (Mesa RADV on Linux, the AMD \
                 display driver on Windows) to run models on this GPU, and run `vulkaninfo` \
                 yourself to see what it says\n",
                detected.unwrap_or_default()
            ),
            // Both halves are established facts: the catalog confirmed there is
            // no ROCm build, and the probe looked for a Vulkan driver and found
            // none. Unlike `VulkanUnconfirmed` above, there is nothing left to
            // go and find out on this host, so "install a driver" is the whole
            // story. Only `Absent` reaches this level — `gpu_support_level`
            // routes the probed-but-inconclusive host to `VulkanUnconfirmed`
            // precisely so that this sentence is never said to one.
            GpuSupportLevel::NoGpuBackend => format!(
                "  rocm_support: GPU detected ({}) but not supported by ROCm, and no Vulkan GPU \
                 driver was found - install one (Mesa RADV on Linux, the AMD display driver on \
                 Windows) to run models on this GPU\n",
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
            // The command names `rocm services list` rather than the serve
            // summary on purpose. The summary reads the device once, right
            // after `ready`, and lemonade answers `unknown` until its own
            // inference probe returns — so the row it prints is legitimately
            // absent for a short window. Pointing at the surface that shows it
            // *once known* keeps the advice able to clear the condition it is
            // printed for, which naming the summary would not.
            //
            // Not folded in with `VulkanOnly`: a detection gap is not a
            // confirmed absence of ROCm support, and saying "Vulkan" here would
            // be the same false claim in the other direction. What the user can
            // do is let an engine decide and then read back what it chose -
            // which is the `device:` row this same ticket added, and the only
            // thing on this host that can actually settle the question.
            // Two different hosts reach `Unknown`, and they need different
            // advice because only one of them can serve anything.
            //
            // A detected-but-unrecognised target means a GPU *is* present, so
            // `rocm serve` runs and the engine's own `device:` row settles what
            // this catalog could not. With nothing detected, that same command
            // is refused outright by the no-usable-GPU pre-flight under the
            // default `gpu_required` policy — naming it would be advice that
            // cannot clear the condition it is printed for (AGENTS.md section 3).
            // `rocm diagnose` is the command that can: its `fix-3-rocm-kernel`,
            // `fix-4-render-group` and `fix-5-amdgpu-load` entries are exactly
            // the reasons a present GPU goes undetected.
            //
            // Kept to one sentence each: the level word already says the CLI
            // cannot answer, so spelling out "can confirm neither ROCm support
            // nor its absence" in both arms was length without information.
            // The distinction it was carrying lives in this comment and in
            // `GpuSupportLevel`'s own docs, which is where it is load-bearing.
            // The command is deliberately NOT `--engine lemonade`. An
            // unrecognised target is a gap in this build's catalog, and a newer
            // ROCm-capable GPU falls into it just as readily as a Vulkan-only
            // one — naming the Vulkan backend would make exactly the kind of
            // unestablished claim this level exists to avoid, only in the other
            // direction. Plain `rocm serve` lets the engine choose, and the
            // `device:` row then reports what it actually used, which is the
            // answer this host needs and nothing here can predict.
            GpuSupportLevel::Unknown => match detected {
                Some(gfx) => format!(
                    "  rocm_support: unknown - {gfx} is not in this build's \
                     catalog; serve with `rocm serve <model>`, then read the \
                     `device:` row of `rocm services list` for the backend it \
                     used\n"
                ),
                None => "  rocm_support: unknown - no AMD GPU was detected; run \
                         `rocm diagnose` to find out why none is visible\n"
                    .to_owned(),
            },
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
            "rocm examine\n  os: {}\n  arch: {}\n  kernel: {}\n  distro: {}\n  cpu: {}\n  system_ram: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  compatible_therock_family: {}\n{}  detected_therock_family: {}\n  vulkan: {}\n  driver_policy: {}\n  driver_status: {}\n  driver_detail: {}\n{}  legacy_rocm_status: {}\n  legacy_rocm_paths: {}\n  legacy_rocm_version: {}\n  legacy_rocm_detail: {}\n  legacy_rocm_guidance: {}\n  wsl: {}\n  wsl_dxg_device: {}\n  wsl_dxcore: {}\n  wsl_librocdxg: {}\n  wsl_rocdxg_dids: {}\n  wsl_ldconfig_librocdxg: {}\n  wsl_librocdxg_on_ld_library_path: {}\n  wsl_librocdxg_beside_runtime: {}\n  wsl_global_rocminfo: {}\n  wsl_cargo: {}\n  wsl_detail: {}\n  managed_runtimes: {}\n  managed_services: {}\n  model_cache_entries: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
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
            self.vulkan_line(),
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
            wsl.is_some_and(|summary| summary.librocdxg_on_ld_library_path),
            wsl.is_some_and(|summary| summary.librocdxg_beside_runtime),
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
    use crate::host_gpu::tests::windows_summary_with_driver;
    use crate::host_gpu::{DriverSummary, VulkanDevice, VulkanSupport};
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
            vulkan: None,
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

    /// The exact placeholders the report prints for the three GPU fields when
    /// it has no value, pinned because they are read back as data.
    ///
    /// examine-49 decides which support tier a host is in by filtering these
    /// out of `detected_gfx_target`, `compatible_therock_family` and
    /// `detected_therock_family`. Reword one and that scenario stops seeing an
    /// absent value as absent: a lane would be classified into the wrong tier
    /// and its tier-specific command assertion would silently stop describing
    /// the host it ran on. The two strings differ on purpose — "not detected"
    /// is about a managed install, "unknown" about a probe — so neither may
    /// stand in for the other.
    #[test]
    fn examine_render_spells_its_absent_value_placeholders() {
        let rendered = examine_summary_with_gpu(None, None).render_text();
        for field in [
            "detected_gfx_target: <unknown>",
            "compatible_therock_family: <unknown>",
            "detected_therock_family: <not detected>",
        ] {
            assert!(rendered.contains(field), "missing `{field}`:\n{rendered}");
        }
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
            vulkan: None,
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

    /// A gfx90c summary carrying the given Vulkan evidence.
    ///
    /// gfx90c is detected from its PCI ID but has no ROCm package family, so it
    /// is the target that reaches the Vulkan-graded arms at all.
    fn gfx90c_summary_with_vulkan(vulkan: Option<VulkanSupport>) -> ExamineSummary {
        let family = normalize_therock_family("gfx90c");
        let mut summary = examine_summary_with_gpu(Some("gfx90c"), family.as_deref());
        summary.vulkan = vulkan;
        summary
    }

    fn vulkan_support(status: VulkanStatus, detail: &str) -> Option<VulkanSupport> {
        Some(VulkanSupport {
            status,
            detail: Some(detail.to_owned()),
            devices: vec![VulkanDevice {
                name: "AMD Radeon Graphics (RADV RENOIR)".to_owned(),
                driver_name: Some("radv".to_owned()),
                vendor_id: Some("0x1002".to_owned()),
                device_type: Some("PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU".to_owned()),
                ..VulkanDevice::default()
            }],
            ..VulkanSupport::default()
        })
    }

    #[test]
    fn examine_render_says_a_gpu_without_a_rocm_family_is_unsupported() {
        let summary = gfx90c_summary_with_vulkan(vulkan_support(
            VulkanStatus::Available,
            "AMD Radeon Graphics (RADV RENOIR) through radv",
        ));

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
            rendered.contains("the Vulkan backend can be used (radv)"),
            "a confirmed device must be named, so the claim can be checked: {rendered}"
        );
        assert!(
            rendered.contains("vulkan: available"),
            "the evidence line must carry the verdict too: {rendered}"
        );
    }

    /// The defect this guards against: the `VulkanOnly` line used to promise
    /// "the Vulkan backend can still be used" on every gfx90c host, derived from
    /// the gfx table alone. A Renoir laptop with no Vulkan driver installed was
    /// told to go and serve a model on a backend that cannot start — and under
    /// the GPU-required policy there is nothing for it to fall back to.
    #[test]
    fn examine_render_does_not_promise_vulkan_on_a_host_that_has_none() {
        let summary = gfx90c_summary_with_vulkan(Some(VulkanSupport {
            status: VulkanStatus::Absent,
            detail: Some("no Vulkan driver found".to_owned()),
            ..VulkanSupport::default()
        }));

        let rendered = summary.render_text();

        assert!(
            !rendered.contains("the Vulkan backend can be used"),
            "a host with no Vulkan driver must not be promised one: {rendered}"
        );
        assert!(
            rendered.contains("no Vulkan GPU driver was found"),
            "{rendered}"
        );
        assert!(
            rendered.contains("install one"),
            "the advice must be able to clear the condition it is printed for: {rendered}"
        );
        assert!(rendered.contains("vulkan: absent"), "{rendered}");
    }

    /// An installed driver that enumerated no device is a weaker claim than a
    /// device that answered, and says so rather than borrowing the stronger
    /// sentence.
    #[test]
    fn examine_render_separates_an_installed_driver_from_a_confirmed_device() {
        let summary = gfx90c_summary_with_vulkan(Some(VulkanSupport {
            status: VulkanStatus::DriverPresentDevicesUnverified,
            detail: Some("Mesa RADV is installed, but no device was enumerated".to_owned()),
            ..VulkanSupport::default()
        }));

        let rendered = summary.render_text();

        assert!(
            rendered.contains("a Vulkan driver is installed, so the Vulkan backend is expected"),
            "{rendered}"
        );
        assert!(
            rendered.contains("vulkan: driver_present_devices_unverified"),
            "{rendered}"
        );
    }

    /// A summary with no probe result at all — an older host's JSON read back
    /// through `rocm remote doctor` — must not read as "no Vulkan". It keeps
    /// the Vulkan advice, because withholding the only path such a host has on
    /// the strength of a probe that never ran would be worse than the
    /// assumption this change replaces.
    #[test]
    fn examine_render_treats_an_unprobed_summary_as_unconfirmed_not_absent() {
        let summary = gfx90c_summary_with_vulkan(None);

        let rendered = summary.render_text();

        assert!(rendered.contains("vulkan: <not probed>"), "{rendered}");
        assert!(
            rendered.contains("the Vulkan backend is unconfirmed"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("no Vulkan GPU driver was found"),
            "an unprobed host is not a host without Vulkan: {rendered}"
        );
        assert!(
            rendered.contains("rocm serve <model> --engine lemonade"),
            "it is still worth trying, so the command stays: {rendered}"
        );
    }

    /// The motivating case, and the one nothing rendered: a host that **was**
    /// probed and showed nothing either way — no ICD, no enumerated device, and
    /// no `vulkaninfo` to ask. On Windows the Khronos `Drivers` key is legacy
    /// and `vulkaninfo` is rarely on PATH, so a working Adrenalin laptop (the
    /// machine ROCMAI-550 was reported from) lands here.
    ///
    /// `gfx_target.rs` pins that the classifier turns this into
    /// `VulkanUnconfirmed`; this pins what the user actually reads, which is the
    /// half that can go wrong on its own. Both directions are asserted: it must
    /// not promise a backend, and it must not declare one absent either.
    #[test]
    fn examine_render_hedges_a_host_that_was_probed_and_showed_nothing() {
        // Built directly rather than through `vulkan_support`, which plants an
        // AMD device: an enumerated device is the one thing this state cannot
        // have, and a fixture that contradicts its own status proves nothing.
        let summary = gfx90c_summary_with_vulkan(Some(VulkanSupport {
            status: VulkanStatus::Inconclusive,
            detail: Some(
                "no Vulkan driver was found, and `vulkaninfo` could not confirm that either (not \
                 installed, or it failed to run), so nothing here confirms one is absent"
                    .to_owned(),
            ),
            ..VulkanSupport::default()
        }));

        let rendered = summary.render_text();

        assert!(
            rendered.contains("`vulkaninfo` could not confirm"),
            "it must name the gap that stopped it confirming: {rendered}"
        );
        assert!(
            !rendered.contains("the Vulkan backend can be used"),
            "nothing here established a Vulkan path: {rendered}"
        );
        assert!(
            !rendered.contains("and no Vulkan GPU driver was found - install one"),
            "and nothing established its absence either, so the confirmed-absence \
             sentence must not be used: {rendered}"
        );
        assert!(rendered.contains("vulkan: inconclusive"), "{rendered}");
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
            vulkan: None,
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
            vulkan: None,
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
        // Both `Unknown` sub-cases must answer, and both must leave somewhere
        // to go — but not the same somewhere. A detected-but-unrecognised
        // target has a GPU, so `rocm serve` runs and its `device:` row settles
        // the backend. With nothing detected that command is refused by the
        // no-usable-GPU pre-flight, so the only honest next step is finding out
        // why the GPU is invisible. See
        // `no_gpu_advice_names_a_command_that_host_can_run`, which pins that
        // distinction against the behaviour rather than the wording.
        for (detected, expected_command) in [
            (None, "rocm diagnose"),
            // Plain `rocm serve`, not `--engine lemonade`: see the arm's own
            // comment. The reasoning this test already gave — "`rocm serve`
            // runs and its `device:` row settles the backend" — is only true of
            // the unpinned command.
            (Some("gfx803"), "rocm serve <model>"),
        ] {
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
                line.contains(expected_command),
                "an `unknown` answer still has to leave somewhere to go, and it \
                 has to be somewhere this host can actually go: {line}"
            );
        }

        // The detected sub-case also names where the backend shows up; the
        // undetected one has no server to read it from.
        let unrecognised = examine_summary_with_gpu(Some("gfx803"), None).render_text();
        assert!(
            unrecognised.contains("device:"),
            "a host that can serve must be told where the real backend shows up: \
             {unrecognised}"
        );
    }

    /// The advice a host is given must name a command that host can actually
    /// run — asserted against the serve pre-flight's own predicate, not against
    /// the wording.
    ///
    /// `rocm serve` refuses on a host with no usable AMD GPU under the default
    /// `gpu_required` policy, so telling an undetected host to serve a model is
    /// advice that cannot clear the condition it is printed for. Both `Unknown`
    /// sub-cases are driven here because they differ precisely on this: a
    /// detected-but-unrecognised target has a GPU and may serve, nothing
    /// detected may not.
    ///
    /// Written against `gpu_support_level` + the rendered line rather than a
    /// string match on the sentence: a reworded message that still sent a
    /// GPU-less host to `rocm serve` would pass a wording assertion and fail
    /// this one.
    #[test]
    fn no_gpu_advice_names_a_command_that_host_can_run() {
        let undetected = examine_summary_with_gpu(None, None).render_text();
        let line = undetected
            .lines()
            .find(|l| l.trim_start().starts_with("rocm_support:"))
            .expect("every host gets a support line");
        assert_eq!(
            gpu_support_level(None, None, VulkanStatus::Unknown),
            GpuSupportLevel::Unknown,
            "precondition: nothing detected is the Unknown tier"
        );
        assert!(
            !line.contains("rocm serve"),
            "`rocm serve` is refused on a host with no usable AMD GPU, so it \
             cannot be the next step offered to one: {line}"
        );
        assert!(
            line.contains("rocm diagnose"),
            "dropping the unusable command must still leave a usable one: {line}"
        );

        // The sibling sub-case keeps the serve advice, because that host has a
        // GPU and the engine's own device row is what settles the question.
        let unrecognised = examine_summary_with_gpu(Some("gfx-not-in-catalog"), None).render_text();
        let line = unrecognised
            .lines()
            .find(|l| l.trim_start().starts_with("rocm_support:"))
            .expect("every host gets a support line");
        assert!(
            line.contains("rocm serve"),
            "a detected GPU can serve, so the advice that settles its backend \
             must not be withheld: {line}"
        );
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
            rendered.contains("rocm_support: unknown - gfx803 is not in this build's catalog"),
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
