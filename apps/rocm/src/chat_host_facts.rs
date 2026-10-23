// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host facts injected into the chat assistant's system prompt.
//!
//! The assistant used to answer platform questions purely from pretraining —
//! "ROCm is not compatible with Windows, use CUDA" was the reported result. It
//! never learned which machine it was running on, even though `rocm-core`
//! detects all of it. This module turns that detection into a short, plain-text
//! block the bin appends to the assistant prompt.
//!
//! Two constraints shape it:
//!
//! - **Cheap.** It is built in [`crate::dash::resolved_args`], on the
//!   synchronous pre-runtime path, before the dashboard draws. So it uses the
//!   fast public getters (`runtime_os_name`, `is_wsl_host`,
//!   `detect_host_gpu_summary`) and never `ExamineSummary::gather()`, whose
//!   Windows path chains several process-spawning inventory queries.
//! - **Plain data.** The dash crates carry no `rocm-core` dependency; the bin
//!   detects and renders, and the rendered `String` rides down through
//!   `ResolvedArgs`.
//!
//! Rendering is separated from detection ([`HostFacts::render`] is pure) so the
//! wording is unit-testable without a machine.

use rocm_core::{AppPaths, HostGpuSummary, detect_host_gpu_summary, is_wsl_host, runtime_os_name};

/// What the assistant is told about the machine it is answering for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostFacts {
    /// `windows` / `linux` (or whatever `std::env::consts::OS` reports).
    os_name: &'static str,
    /// Linux running under WSL — a Windows box where the Linux engines apply.
    wsl: bool,
    /// Marketing name of the primary AMD adapter, when one was detected.
    gpu_name: Option<String>,
    /// LLVM target of that adapter (`gfx1151`, …), when it was detected.
    gfx_target: Option<String>,
    /// TheRock family the target maps to, when it differs from the target.
    therock_family: Option<String>,
}

impl HostFacts {
    /// Detect this machine's facts. Fast: one const OS lookup, up to three
    /// filesystem/env reads for the WSL signal, and the fast GPU summary.
    pub(crate) fn detect(paths: Option<&AppPaths>) -> Self {
        Self::from_gpu_summary(
            runtime_os_name(),
            is_wsl_host(),
            detect_host_gpu_summary(paths),
        )
    }

    /// The pure half of [`Self::detect`], so the rendering tests can pin every
    /// host shape (Windows, WSL, bare Linux, no GPU) from one machine.
    fn from_gpu_summary(os_name: &'static str, wsl: bool, gpu: HostGpuSummary) -> Self {
        let HostGpuSummary {
            name,
            gfx_target,
            therock_family,
        } = gpu;
        // The family is only worth a line when it says something the target
        // does not (`gfx110X-all` for `gfx1100`, but not `gfx1151` twice).
        let therock_family = therock_family.filter(|f| Some(f) != gfx_target.as_ref());
        Self {
            os_name,
            wsl,
            gpu_name: name,
            gfx_target,
            therock_family,
        }
    }

    /// Whether vLLM can serve here. Mirrors
    /// [`rocm_core::preferred_serve_engine_for_host_gpu_summary`]: the vLLM
    /// adapter bails out on native Windows, and WSL builds as Linux.
    fn vllm_available(&self) -> bool {
        self.os_name != "windows"
    }

    /// How to describe the operating system to a non-technical user. WSL is
    /// called out because it is the case the old prompt got wrong: it asserted
    /// "on native Windows, vLLM is skipped" at a user who was on WSL, where
    /// vLLM is the supported path.
    fn operating_system(&self) -> &'static str {
        match (self.os_name, self.wsl) {
            ("windows", _) => "Windows",
            ("linux", true) => "Linux running under WSL2 on a Windows host",
            ("linux", false) => "Linux",
            (other, _) => other,
        }
    }

    /// The GPU line's value. `None` renders as an explicit "not detected" so
    /// the assistant knows the check ran and came back empty, rather than
    /// filling the silence from pretraining.
    fn gpu(&self) -> String {
        match (self.gpu_name.as_deref(), self.gfx_target.as_deref()) {
            (Some(name), Some(target)) => format!("{name} ({target})"),
            (Some(name), None) => name.to_owned(),
            (None, Some(target)) => target.to_owned(),
            (None, None) => "no AMD GPU detected on this machine".to_owned(),
        }
    }

    /// Render the block appended to the assistant's system prompt.
    ///
    /// Deliberately short: for the built-in `local` provider the whole system
    /// prompt is folded into the user turn, and the built-in assistant is a 4B
    /// model — a long facts block would crowd out the question.
    pub(crate) fn render(&self) -> String {
        let family = self
            .therock_family
            .as_deref()
            .map_or_else(String::new, |f| {
                format!("- ROCm/TheRock family for this GPU: {f}\n")
            });
        let engines = if self.vllm_available() {
            "- Serving engines that run here: Lemonade and vLLM.\n"
        } else {
            "- Serving engines that run here: Lemonade. vLLM is skipped on native Windows; \
             tell the user to use WSL2 or Linux for that ROCm GPU engine.\n"
        };
        format!(
            "Facts about the machine you are running on, detected by ROCm CLI. Trust these \
             over anything you remember about ROCm:\n\
             - Operating system: {os}\n\
             - AMD GPU: {gpu}\n\
             {family}{engines}\
             ROCm and ROCm CLI support this operating system. Never tell the user ROCm is \
             unavailable on their platform, and never suggest CUDA, DirectML, or CPU fallback.",
            os = self.operating_system(),
            gpu = self.gpu(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: Option<&str>, target: Option<&str>, family: Option<&str>) -> HostGpuSummary {
        HostGpuSummary {
            name: name.map(str::to_owned),
            gfx_target: target.map(str::to_owned),
            therock_family: family.map(str::to_owned),
        }
    }

    #[test]
    fn wsl_is_described_as_linux_on_a_windows_host_with_vllm_available() {
        // The reporter's shape: a Windows box, but the CLI runs as Linux, so
        // vLLM IS the supported path. The old prompt asserted the opposite.
        let facts = HostFacts::from_gpu_summary(
            "linux",
            true,
            gpu(
                Some("AMD Radeon Graphics"),
                Some("gfx1151"),
                Some("gfx1151"),
            ),
        );
        let block = facts.render();
        assert!(
            block.contains("- Operating system: Linux running under WSL2 on a Windows host"),
            "WSL must be named, not flattened to bare Linux:\n{block}"
        );
        assert!(
            block.contains("- AMD GPU: AMD Radeon Graphics (gfx1151)"),
            "the detected adapter and target must both appear:\n{block}"
        );
        assert!(
            block.contains("Lemonade and vLLM"),
            "vLLM is available under WSL:\n{block}"
        );
        assert!(
            !block.contains("ROCm/TheRock family"),
            "a family identical to the target adds nothing:\n{block}"
        );
    }

    #[test]
    fn native_windows_keeps_the_vllm_caveat_that_left_the_static_prompt() {
        let facts = HostFacts::from_gpu_summary(
            "windows",
            false,
            gpu(
                Some("AMD Radeon RX 9070 XT"),
                Some("gfx1201"),
                Some("gfx120X-all"),
            ),
        );
        let block = facts.render();
        assert!(block.contains("- Operating system: Windows"), "{block}");
        assert!(
            block.contains("vLLM is skipped on native Windows"),
            "the caveat must survive, now stated only where it is true:\n{block}"
        );
        assert!(
            block.contains("- ROCm/TheRock family for this GPU: gfx120X-all"),
            "a family that differs from the target is worth a line:\n{block}"
        );
    }

    #[test]
    fn bare_linux_is_not_reported_as_wsl() {
        let facts =
            HostFacts::from_gpu_summary("linux", false, gpu(None, Some("gfx942"), Some("gfx942")));
        let block = facts.render();
        assert!(block.contains("- Operating system: Linux\n"), "{block}");
        assert!(!block.contains("WSL"), "{block}");
        assert!(block.contains("- AMD GPU: gfx942"), "{block}");
    }

    #[test]
    fn an_undetected_gpu_is_stated_rather_than_omitted() {
        // A missing line would leave the model free to invent one; an explicit
        // negative keeps the block's shape stable for every host.
        let facts = HostFacts::from_gpu_summary("linux", false, gpu(None, None, None));
        let block = facts.render();
        assert!(
            block.contains("- AMD GPU: no AMD GPU detected on this machine"),
            "{block}"
        );
    }

    #[test]
    fn every_host_is_told_rocm_supports_its_platform() {
        // The reported defect: the assistant answered "ROCm is not compatible
        // with Windows; use CUDA/DirectX". Both halves are refused here.
        for (os, wsl) in [("windows", false), ("linux", true), ("linux", false)] {
            let block = HostFacts::from_gpu_summary(os, wsl, gpu(None, None, None)).render();
            assert!(
                block.contains("Never tell the user ROCm is unavailable on their platform"),
                "{os} (wsl={wsl}):\n{block}"
            );
            assert!(block.contains("never suggest CUDA"), "{os}:\n{block}");
        }
    }

    #[test]
    fn detection_describes_this_machine() {
        // Not a fixture: this pins that `detect` reads the real host and that
        // the block it produces is well-formed wherever the suite runs.
        let facts = HostFacts::detect(None);
        let block = facts.render();
        assert!(block.contains("- Operating system: "), "{block}");
        assert!(block.contains("- AMD GPU: "), "{block}");
        assert!(
            block.contains(if cfg!(windows) {
                "Operating system: Windows"
            } else if cfg!(target_os = "linux") {
                "Operating system: Linux"
            } else {
                "Operating system: "
            }),
            "the OS line must match the machine the test runs on:\n{block}"
        );
    }
}
