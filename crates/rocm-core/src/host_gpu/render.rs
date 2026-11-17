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
        // `VulkanStatus::as_str`, not a second table here: this word and the
        // one `--json` prints are the same status, and a local mapping lets the
        // two renderings of one report drift apart silently.
        let status = vulkan.status.as_str();
        vulkan.detail.as_deref().map_or_else(
            || status.to_owned(),
            |detail| format!("{status} ({detail})"),
        )
    }

    /// Whether this summary's own evidence already contradicts "install a
    /// Vulkan driver".
    ///
    /// `DriverSummary::status` only ever reads `AMD_DISPLAY_DRIVER_DETECTED`
    /// on Windows — `windows_driver_summary` is the sole producer of that word
    /// — so matching on it *is* the Windows cross-check, with no second read
    /// of `os` to keep in step with it. On that platform the display driver
    /// carries the Vulkan ICD, so its presence contradicts that advice: the
    /// report would print `driver_status: amd_display_driver_detected` a few
    /// lines above a sentence telling the reader to go and install what it
    /// names.
    ///
    /// The cross-check is Windows-only for that reason, and not by accident:
    /// Linux's `driver_status` reports the `amdgpu` kernel module, which says
    /// nothing about whether a userspace ICD such as Mesa RADV is installed,
    /// so suppressing the install advice there would withhold the one step
    /// that can help.
    ///
    /// One predicate rather than a test spelled out in each arm that gives
    /// install advice. Two arms now need it — `VulkanUnconfirmed` and
    /// `NoGpuBackend` — and the second one shipped without it precisely
    /// because the first one's check lived inside that first one's `match`
    /// arm. Each arm's own test pins its half:
    /// `examine_render_does_not_tell_a_windows_host_to_install_the_driver_it_has`
    /// and
    /// `examine_render_does_not_tell_a_no_backend_windows_host_to_install_the_driver_it_has`.
    fn install_advice_is_contradicted_by_the_driver(&self) -> bool {
        self.driver.status == super::AMD_DISPLAY_DRIVER_DETECTED
    }

    /// What a [`GpuSupportLevel::VulkanUnconfirmed`] host is told to do next.
    fn vulkan_unconfirmed_next_step(&self) -> &'static str {
        if self.install_advice_is_contradicted_by_the_driver() {
            crate::gfx_target::VULKAN_UNCONFIRMED_NEXT_STEP_WITH_DISPLAY_DRIVER
        } else {
            crate::gfx_target::VULKAN_UNCONFIRMED_NEXT_STEP
        }
    }

    /// What a [`GpuSupportLevel::NoGpuBackend`] host is told to do next.
    ///
    /// The unconditional branch is the whole story on a host with nothing
    /// installed: both halves of the level are established facts, there is
    /// nothing left to go and find out, and installing a driver is the one
    /// thing that can clear it.
    ///
    /// The other branch is a Windows host that already has the driver that
    /// sentence names. Its probe answered — `VulkanStatus::Absent` is never
    /// reached without `vulkaninfo`'s answer behind it — so "run `vulkaninfo`"
    /// is not left either, the way it is under `VulkanUnconfirmed`. What is
    /// left is the display driver itself: it is installed and is exposing no
    /// Vulkan GPU device here, which a repair or update of that driver can
    /// change and a fresh install cannot.
    fn no_gpu_backend_next_step(&self) -> &'static str {
        if self.install_advice_is_contradicted_by_the_driver() {
            "the AMD display driver is installed (see `driver_status`) but is exposing no Vulkan \
             GPU device here, so repairing or updating it is the step that could open one"
        } else {
            "install one (Mesa RADV on Linux, the AMD display driver on Windows) to run models on \
             this GPU"
        }
    }

    /// The `rocm_support` line of the examine text.
    ///
    /// ROCMAI-550 asks `rocm examine` to "report the support level with one
    /// line of what the user can do", so every level gets a line — including
    /// the ones where the answer is "this build cannot tell". Printing only the
    /// `VulkanOnly` line leaves this report unable to answer the very question
    /// the other surfaces send people here to ask: `lemonade_engine_note` in
    /// `apps/rocm/src/main.rs` tells an `Unknown` host to run `rocm examine`,
    /// and silence there is not an answer.
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
            // Looked at and nothing settled either way. Says the probe could
            // not determine it and how to settle it, rather than promising a
            // backend or declaring one absent — neither of which the evidence
            // supports. See `GpuSupportLevel::VulkanUnconfirmed`.
            //
            // The claim comes from `VULKAN_UNCONFIRMED_CLAIM` because
            // `lemonade_engine_note` renders the same level a few lines further
            // down the same report; one constant is what keeps the two from
            // contradicting each other.
            //
            // The next step is cross-checked against `driver.status`, the one
            // other fact on this summary that bears on it. On Windows the AMD
            // display driver *is* the Vulkan driver, so a host that already has
            // it must not be told to install one — the exact misdirection this
            // level exists to avoid, reached through its own advice. The
            // cross-check is Windows-only: Linux's `driver_status` reports the
            // `amdgpu` kernel module, which says nothing about whether a
            // userspace ICD is installed.
            GpuSupportLevel::VulkanUnconfirmed => format!(
                "  rocm_support: GPU detected ({}) but not supported by ROCm; {} - {}\n",
                detected.unwrap_or_default(),
                crate::gfx_target::VULKAN_UNCONFIRMED_CLAIM,
                self.vulkan_unconfirmed_next_step(),
            ),
            // Both halves are established facts: the catalog confirmed there is
            // no ROCm build, and the probe looked for a Vulkan driver and found
            // none. Unlike `VulkanUnconfirmed` above, there is nothing left to
            // go and find out on this host. Only `Absent` reaches this level —
            // `gpu_support_level` routes the probed-but-inconclusive host to
            // `VulkanUnconfirmed` precisely so that this sentence is never said
            // to one.
            //
            // The next step is cross-checked against `driver.status` for the
            // same reason `VulkanUnconfirmed`'s is, and through the same
            // predicate: a Windows host whose report already says
            // `driver_status: amd_display_driver_detected` must not be told, a
            // few lines below, to install the driver it is showing. The
            // confirmed absence itself is unaffected — the probe did look and
            // did find no Vulkan GPU device — only what the host can do about
            // it differs.
            GpuSupportLevel::NoGpuBackend => format!(
                "  rocm_support: GPU detected ({}) but not supported by ROCm, and no Vulkan GPU \
                 driver was found - {}\n",
                detected.unwrap_or_default(),
                self.no_gpu_backend_next_step(),
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
            //
            // The two are computed from unrelated facts and really can differ.
            // `compatible_therock_family` is `normalize_therock_family` applied
            // to the gfx target detected right now; `detect_managed_therock_family`
            // is the family of the most recently installed managed runtime on
            // disk and never looks at the GPU at all. Swap the card, plug in an
            // eGPU, or keep two cards and boot the other one, and the install
            // on disk is for a family this GPU was never built for. Saying
            // "already installed - serve on it" there would promise a serve
            // this host cannot do, which is exactly the kind of unestablished
            // claim ROCMAI-550 exists to remove, so the match is a condition of
            // that sentence rather than an assumption behind it.
            GpuSupportLevel::RocmHip => {
                let compatible = self.compatible_therock_family.as_deref();
                match self.detected_therock_family.as_deref() {
                    Some(installed) if Some(installed) == compatible => format!(
                        "  rocm_support: GPU detected ({}) with full ROCm/HIP support, and the \
                         managed ROCm SDK for {installed} is already installed - serve on it with \
                         `rocm serve <model>`\n",
                        detected.unwrap_or("<unknown>")
                    ),
                    // An install is present but for another family, so
                    // `rocm install sdk` still has something to do and is still
                    // advice that can clear the condition it is printed for.
                    Some(installed) => format!(
                        "  rocm_support: GPU detected ({}) with full ROCm/HIP support, but the \
                         managed ROCm SDK installed here is for {installed}, not {} - install the \
                         one this GPU needs with `rocm install sdk`\n",
                        detected.unwrap_or("<unknown>"),
                        compatible.unwrap_or("<unknown>")
                    ),
                    None => format!(
                        "  rocm_support: GPU detected ({}) with full ROCm/HIP support; \
                         install the SDK for it with `rocm install sdk`\n",
                        detected.unwrap_or("<unknown>")
                    ),
                }
            }
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

    /// The `wsl_runtime_librocdxg` line of the examine text, or an empty string.
    ///
    /// Present only when a managed runtime ships its own `librocdxg.so.1` that its
    /// HSA runtime does not load. `driver_status` describes the host's plumbing
    /// and stays as it is; this line is about the runtime. Each runtime's fix is
    /// the `ln -sfn` command [`crate::runtime_rocdxg::link_command`] builds (`-fn`
    /// so it also clears a dangling `librocdxg.so`). It makes the same relative
    /// link `rocm install sdk` creates for a runtime it installs (through
    /// `link_runtime_copy`, not through the command).
    fn runtime_librocdxg_line(&self) -> String {
        if self.unlinked_runtime_librocdxg.is_empty() {
            return String::new();
        }
        let commands = self
            .unlinked_runtime_librocdxg
            .iter()
            .map(|copy| crate::runtime_rocdxg::link_command(copy))
            .collect::<Vec<_>>()
            .join("; ");
        let count = self.unlinked_runtime_librocdxg.len();
        let noun = if count == 1 { "file sits" } else { "files sit" };
        format!(
            "  wsl_runtime_librocdxg: {count} managed-runtime librocdxg.so.1 {noun} beside libhsa-runtime64 with \
             no librocdxg.so (or a dangling one), and the runtime opens librocdxg.so, so it loads the \
             host's librocdxg if there is one; a host copy older than the runtime expects fails at first GPU \
             use (for example `undefined symbol: hsaKmtSetPersistingCacheSize`). This lists what the files \
             show, not whether this machine is affected. To make the runtime use its own copy (this adds a \
             link inside the runtime's directory, including for a runtime the CLI did not install) run: \
             {commands}\n"
        )
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
            "rocm examine\n  os: {}\n  arch: {}\n  kernel: {}\n  distro: {}\n  cpu: {}\n  system_ram: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  compatible_therock_family: {}\n{}  detected_therock_family: {}\n  vulkan: {}\n  driver_policy: {}\n  driver_status: {}\n  driver_detail: {}\n{}  legacy_rocm_status: {}\n  legacy_rocm_paths: {}\n  legacy_rocm_version: {}\n  legacy_rocm_detail: {}\n  legacy_rocm_guidance: {}\n  wsl: {}\n  wsl_dxg_device: {}\n  wsl_dxcore: {}\n  wsl_librocdxg: {}\n  wsl_rocdxg_dids: {}\n  wsl_ldconfig_librocdxg: {}\n  wsl_librocdxg_on_ld_library_path: {}\n  wsl_librocdxg_beside_runtime: {}\n  wsl_global_rocminfo: {}\n  wsl_cargo: {}\n  wsl_detail: {}\n{}  managed_runtimes: {}\n  managed_services: {}\n  model_cache_entries: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
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
            self.runtime_librocdxg_line(),
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
            unlinked_runtime_librocdxg: Vec::new(),
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
    /// examine-55 decides which support tier a host is in by filtering these
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
            unlinked_runtime_librocdxg: Vec::new(),
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
        // AGENTS.md §3: advice naming a command is asserted together with the
        // behaviour. Saying the Vulkan backend "can be used" without naming
        // how to use it leaves the user where they started, so the command is
        // part of the claim, not decoration — without this assertion the
        // recommendation could be deleted or garbled and this test would still
        // pass.
        //
        // That the named command *parses* (the subcommand exists and accepts
        // `--engine lemonade`) is proved mechanically rather than here:
        // `advised_commands::every_advised_command_parses` in `apps/rocm`
        // scans production string literals under `crates/`, including this
        // one, and routes each through the real clap tree. rocm-core cannot
        // see the CLI's argument parser, so this half pins that the sentence
        // is printed at all for a Vulkan-only host.
        assert!(
            rendered.contains("serve on it with `rocm serve <model> --engine lemonade`"),
            "a Vulkan-only host must be told the command that serves on it:\n{rendered}"
        );
    }

    /// The defect this guards against: promising "the Vulkan backend can still
    /// be used" on every gfx90c host, from the gfx table alone. That sends a
    /// Renoir laptop with no Vulkan driver installed to serve a model on a
    /// backend that cannot start — and under the GPU-required policy there is
    /// nothing for it to fall back to.
    #[test]
    fn examine_render_does_not_promise_vulkan_on_a_host_that_has_none() {
        let mut summary = gfx90c_summary_with_vulkan(Some(VulkanSupport {
            status: VulkanStatus::Absent,
            detail: Some("no Vulkan driver found".to_owned()),
            ..VulkanSupport::default()
        }));
        // The shared fixture is a Windows host carrying
        // `AMD_DISPLAY_DRIVER_DETECTED`, which is the one state where "install
        // one" is the wrong advice (see
        // `examine_render_does_not_tell_a_no_backend_windows_host_to_install_the_driver_it_has`).
        // This test is about the host that genuinely has nothing, so it plants
        // that host rather than asserting the install sentence on a summary
        // whose own `driver_status` line contradicts it.
        summary.os = "linux".to_owned();
        summary.driver = DriverSummary {
            policy: "amdgpu_dkms".to_owned(),
            status: "amdgpu_module_loaded".to_owned(),
            detail: None,
            windows: None,
        };

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
                "no installed Vulkan driver was listed, and `vulkaninfo` could not confirm \
                 anything either (not installed, or it failed to run), so nothing here shows \
                 whether a driver is present"
                    .to_owned(),
            ),
            ..VulkanSupport::default()
        }));

        let rendered = summary.render_text();

        assert!(
            rendered.contains(crate::gfx_target::VULKAN_UNCONFIRMED_CLAIM),
            "it must say the probe could not settle it: {rendered}"
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

    /// The Windows machine this level exists for: the Khronos `Drivers` key it
    /// reads is legacy and `vulkaninfo` is rarely on PATH, so an **Adrenalin
    /// laptop with a working Vulkan driver** lands in `VulkanUnconfirmed`.
    /// Telling it to install the AMD display driver is advice it has already
    /// taken, and nothing it does can clear the state.
    ///
    /// `driver_status` is the summary's own evidence against that advice, and
    /// on Windows the display driver carries the Vulkan ICD, so its presence is
    /// decisive. The line must say so instead, and still name the one step that
    /// can settle the question.
    #[test]
    fn examine_render_does_not_tell_a_windows_host_to_install_the_driver_it_has() {
        let mut summary = gfx90c_summary_with_vulkan(Some(VulkanSupport {
            status: VulkanStatus::Inconclusive,
            detail: Some("the legacy Khronos key listed nothing".to_owned()),
            ..VulkanSupport::default()
        }));
        summary.os = "windows".to_owned();
        summary.driver = DriverSummary {
            policy: "windows_validate_only".to_owned(),
            status: super::super::AMD_DISPLAY_DRIVER_DETECTED.to_owned(),
            detail: Some("AMD display driver 31.0.21925.1001".to_owned()),
            windows: None,
        };

        let rendered = summary.render_text();
        let support = rendered
            .lines()
            .find(|line| line.contains("rocm_support:"))
            .expect("every host gets a support line");

        assert!(
            !support.contains("install a Vulkan driver"),
            "this host already has the driver that would be installed: {support}"
        );
        assert!(
            support.contains("the AMD display driver is installed"),
            "and it is owed the reason that advice does not apply: {support}"
        );
        assert!(
            support.contains("run `vulkaninfo`"),
            "the one step that can still settle it must stay: {support}"
        );
        assert!(
            support.contains(crate::gfx_target::VULKAN_UNCONFIRMED_CLAIM),
            "the claim is the same on both platforms; only the next step differs: {support}"
        );

        // The Linux shape of the same status, which is what makes the two halves
        // a discrimination rather than one assertion twice. `driver` has to be
        // overwritten explicitly: the shared fixture is a Windows host and
        // already carries `AMD_DISPLAY_DRIVER_DETECTED`, so reusing it unchanged
        // would re-enter the branch above and assert nothing about Linux.
        //
        // There `driver_status` reports the `amdgpu` kernel module, which says
        // nothing about whether a userspace ICD such as Mesa RADV is installed,
        // so the install advice is the true one and must survive.
        let mut linux = gfx90c_summary_with_vulkan(Some(VulkanSupport {
            status: VulkanStatus::Inconclusive,
            detail: Some("no ICD manifest was found".to_owned()),
            ..VulkanSupport::default()
        }));
        linux.os = "linux".to_owned();
        linux.driver = DriverSummary {
            policy: "amdgpu_dkms".to_owned(),
            status: "amdgpu_module_loaded".to_owned(),
            detail: None,
            windows: None,
        };
        let linux = linux.render_text();
        assert!(
            linux.contains(crate::gfx_target::VULKAN_UNCONFIRMED_NEXT_STEP),
            "a host with no display-driver evidence still gets the install step: {linux}"
        );
        assert!(
            !linux.contains("the AMD display driver is installed"),
            "and must not be told about a Windows driver it cannot have: {linux}"
        );
    }

    /// The same misdirection one level down, and the one the
    /// `VulkanUnconfirmed` cross-check above did not stop: a Windows host whose
    /// report prints `driver_status: amd_display_driver_detected` must not be
    /// told, a few lines higher in the same output, to install the AMD display
    /// driver.
    ///
    /// `NoGpuBackend` keeps the confirmed absence — the probe did look and did
    /// find no Vulkan GPU device, and that is not in question here — but the
    /// *advice* has to be one the host can act on, and "install the driver you
    /// are already showing" is not. `vulkaninfo` is not offered as the next
    /// step either, unlike under `VulkanUnconfirmed`: reaching `Absent` takes
    /// `vulkaninfo`'s own answer, so that question is already settled.
    ///
    /// Both directions are asserted, against the two hosts that differ only in
    /// `driver_status`, so an arm that dropped the cross-check again fails here
    /// rather than passing on the Linux half alone.
    #[test]
    fn examine_render_does_not_tell_a_no_backend_windows_host_to_install_the_driver_it_has() {
        let absent = || {
            Some(VulkanSupport {
                status: VulkanStatus::Absent,
                detail: Some("no Vulkan driver found".to_owned()),
                ..VulkanSupport::default()
            })
        };
        // The shared fixture is already a Windows host carrying
        // `AMD_DISPLAY_DRIVER_DETECTED` — the exact summary this arm shipped
        // being rendered against, and the reason the omission was invisible.
        let windows = gfx90c_summary_with_vulkan(absent());
        assert_eq!(
            windows.driver.status,
            super::super::AMD_DISPLAY_DRIVER_DETECTED,
            "this test is only a discrimination while the fixture it reuses is the \
             display-driver host"
        );
        let windows = windows.render_text();
        let support = windows
            .lines()
            .find(|line| line.contains("rocm_support:"))
            .expect("every host gets a support line");

        assert!(
            windows.contains("driver_status: amd_display_driver_detected"),
            "the contradiction this guards against is between two lines of one report: {windows}"
        );
        assert!(
            !support.contains("install one"),
            "this host already has the driver that sentence names: {support}"
        );
        assert!(
            support.contains("repairing or updating it"),
            "and it is owed the step that can actually open a GPU path: {support}"
        );
        assert!(
            support.contains("no Vulkan GPU driver was found"),
            "the confirmed absence itself is unchanged; only the advice is: {support}"
        );
        assert!(
            !support.contains("run `vulkaninfo`"),
            "`Absent` already has `vulkaninfo`'s answer behind it, so there is nothing \
             left to ask it: {support}"
        );

        // The Linux shape of the same status: `driver_status` there reports the
        // `amdgpu` kernel module, which says nothing about a userspace ICD, so
        // the install advice is true and must survive.
        let mut linux = gfx90c_summary_with_vulkan(absent());
        linux.os = "linux".to_owned();
        linux.driver = DriverSummary {
            policy: "amdgpu_dkms".to_owned(),
            status: "amdgpu_module_loaded".to_owned(),
            detail: None,
            windows: None,
        };
        let linux = linux.render_text();
        assert!(
            linux.contains("install one (Mesa RADV on Linux, the AMD display driver on Windows)"),
            "a host with no display-driver evidence still gets the install step: {linux}"
        );
        assert!(
            !linux.contains("repairing or updating it"),
            "and must not be told about a Windows driver it cannot have: {linux}"
        );
    }

    /// An unmapped target must not be *claimed* unsupported, even though every
    /// host now gets a `rocm_support` line.
    ///
    /// Every host gets the line, so what this protects is the discrimination
    /// rather than the silence: "no ROCm build" is a fact about gfx90c, and a
    /// gap in this catalog is not the same claim about gfx803.
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

    // Pins POSIX path strings; the finding is WSL-only.
    #[cfg(unix)]
    #[test]
    fn examine_names_a_runtime_librocdxg_its_runtime_does_not_load() {
        let mut summary = examine_summary_with_gpu(None, None);
        summary.unlinked_runtime_librocdxg = vec![PathBuf::from("/rt/lib/librocdxg.so.1")];
        let rendered = summary.render_text();
        // The wording of the line is pinned here; that the command it names clears
        // the finding is proved by scenario `examine-51`, which runs it.
        assert!(
            rendered.contains("wsl_runtime_librocdxg: 1 managed-runtime librocdxg.so.1 file sits beside libhsa-runtime64 with no librocdxg.so (or a dangling one)"),
            "{rendered}"
        );
        assert!(
            rendered.contains("run: ln -sfn librocdxg.so.1 '/rt/lib/librocdxg.so'"),
            "{rendered}"
        );
        // The host's own plumbing verdict is a separate line and is not rewritten.
        assert!(rendered.contains("driver_status: amd_display_driver_detected"));
    }

    #[test]
    fn examine_prints_no_runtime_librocdxg_line_when_every_runtime_loads_its_own() {
        let clean = examine_summary_with_gpu(None, None).render_text();
        assert!(!clean.contains("wsl_runtime_librocdxg"), "{clean}");
        // The same summary with one copy listed does print it, so the absence
        // above is the absence of the finding and not of the line.
        let mut listed = examine_summary_with_gpu(None, None);
        listed.unlinked_runtime_librocdxg = vec![PathBuf::from("/rt/lib/librocdxg.so.1")];
        assert!(listed.render_text().contains("wsl_runtime_librocdxg"));
    }

    // Pins POSIX path strings; the finding is WSL-only.
    #[cfg(unix)]
    #[test]
    fn examine_joins_the_commands_for_several_copies() {
        let mut summary = examine_summary_with_gpu(None, None);
        summary.unlinked_runtime_librocdxg = vec![
            PathBuf::from("/a/lib/librocdxg.so.1"),
            PathBuf::from("/b/lib/librocdxg.so.1"),
        ];
        let rendered = summary.render_text();
        assert!(rendered.contains("wsl_runtime_librocdxg: 2 managed-runtime librocdxg.so.1 files sit beside libhsa-runtime64 with no librocdxg.so (or a dangling one)"));
        assert!(rendered.contains(
            "run: ln -sfn librocdxg.so.1 '/a/lib/librocdxg.so'; ln -sfn librocdxg.so.1 '/b/lib/librocdxg.so'"
        ));
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
            unlinked_runtime_librocdxg: Vec::new(),
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
            unlinked_runtime_librocdxg: Vec::new(),
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
        // wants a named command to be able to clear the condition). The
        // installed family must be the GPU's own - see
        // `an_sdk_for_another_family_is_not_reported_as_ready_for_this_gpu`.
        let mut installed = supported;
        installed.detected_therock_family = family;
        let rendered = installed.render_text();
        assert!(
            rendered.contains("rocm_support: GPU detected (gfx1103) with full ROCm/HIP support"),
            "{rendered}"
        );
        assert!(
            rendered.contains("managed ROCm SDK for gfx110X-all is already installed"),
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

    /// A managed SDK on disk is not, by itself, an SDK for *this* GPU.
    ///
    /// `detected_therock_family` is the most recently installed managed runtime
    /// (`detect_managed_therock_family` reads the registry and never looks at
    /// the GPU); `compatible_therock_family` is normalised from the gfx target
    /// detected right now. A GPU swap, an eGPU or a second card leaves them
    /// pointing at different families, and the `already installed - serve on
    /// it` sentence then promises a serve this host cannot do.
    ///
    /// The matching half is asserted by
    /// `examine_render_reports_a_support_level_for_every_host`; only this one
    /// fails if the two families stop being compared, because the wording of
    /// the matching case is unchanged.
    #[test]
    fn an_sdk_for_another_family_is_not_reported_as_ready_for_this_gpu() {
        let compatible = normalize_therock_family("gfx1103").expect("gfx1103 has a family");
        let mut host = examine_summary_with_gpu(Some("gfx1103"), Some(&compatible));
        // A real family, and not this GPU's: the gfx110X split between the
        // discrete and the all-targets build is exactly the pair a user is
        // likeliest to have both of.
        host.detected_therock_family = Some("gfx110X-dgpu".to_owned());
        assert_ne!(
            host.detected_therock_family.as_deref(),
            Some(compatible.as_str()),
            "precondition: the installed family is not the compatible one"
        );

        let rendered = host.render_text();
        let line = rendered
            .lines()
            .find(|line| line.trim_start().starts_with("rocm_support:"))
            .expect("every host gets a support line")
            .to_owned();

        assert!(
            !line.contains("already installed"),
            "an SDK built for another family must not be reported as this GPU's: {line}"
        );
        assert!(
            line.contains("gfx110X-dgpu") && line.contains(compatible.as_str()),
            "the line has to name both families, or the user cannot see which \
             one is missing: {line}"
        );
        assert!(
            line.contains("rocm install sdk"),
            "there is still an install left to do, so the advice must name the \
             command that does it: {line}"
        );
    }
}
