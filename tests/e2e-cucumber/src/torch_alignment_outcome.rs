// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which `torch_alignment:` outcomes the runtime scenarios accept as settled.

/// Verdicts that mean the runtime was settled and its torch left in a state the
/// CLI produces on purpose.
///
/// Four, not one. Whether a reinstall rewrites torch or finds it already correct
/// depends on what the shared pre-warm tree held when the scenario started, and a
/// torch that has already executed a GPU kernel with this SDK is kept exactly as
/// it is and reported as `retained_*`: settled without anything being installed.
pub const SETTLED_VERDICTS: [&str; 4] = [
    "realigned",
    "already_aligned",
    "retained_sdk_build",
    "retained_engine_build",
];

/// The reason the CLI prints when the installed engine declares no torch
/// requirement at all.
const NO_PIN_REASON: &str = "(vllm does not pin torch)";

/// Whether the installed engine's own package metadata requires torch.
///
/// Read from the runtime's interpreter by the scenario, never from the CLI's
/// output: the output is the thing being judged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineTorchRequirement {
    Declared,
    Absent,
}

/// Whether the text after `torch_alignment: ` describes a settled runtime.
///
/// `not_applicable (vllm does not pin torch)` counts only when the engine really
/// declares no torch requirement. That is the ROCm 10.x discover install: the
/// CLI pins torch itself and installs the engine without dependencies, so there
/// is no engine pin for the alignment to settle against. The same text over an
/// engine that does declare one would be the skip this check exists to catch, as
/// would any other `not_applicable` reason, such as a manifest that identifies no
/// SDK torch build.
#[must_use]
pub fn is_settled(verdict_line: &str, engine: EngineTorchRequirement) -> bool {
    let verdict_line = verdict_line.trim();
    let verdict = verdict_line.split_whitespace().next().unwrap_or_default();
    if SETTLED_VERDICTS.contains(&verdict) {
        return true;
    }
    verdict == "not_applicable"
        && engine == EngineTorchRequirement::Absent
        && verdict_line.strip_prefix(verdict).map(str::trim) == Some(NO_PIN_REASON)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_PIN: &str = "not_applicable (vllm does not pin torch)";

    #[test]
    fn every_settled_verdict_is_accepted_whatever_the_engine_declares() {
        for verdict in SETTLED_VERDICTS {
            for engine in [
                EngineTorchRequirement::Declared,
                EngineTorchRequirement::Absent,
            ] {
                assert!(is_settled(verdict, engine), "{verdict} with {engine:?}");
            }
        }
    }

    #[test]
    fn no_pin_is_settled_only_when_the_engine_declares_no_torch() {
        assert!(is_settled(NO_PIN, EngineTorchRequirement::Absent));
        assert!(!is_settled(NO_PIN, EngineTorchRequirement::Declared));
    }

    #[test]
    fn any_other_not_applicable_reason_stays_unsettled() {
        for line in [
            "not_applicable (the runtime manifest does not identify the SDK's torch build)",
            "not_applicable (vllm does not pin torch to an exact version (torch>=2.0))",
            "not_applicable",
        ] {
            assert!(!is_settled(line, EngineTorchRequirement::Absent), "{line}");
        }
    }

    #[test]
    fn the_other_unsettled_verdicts_stay_unsettled() {
        for line in ["unavailable", "install_failed", "disabled", ""] {
            assert!(!is_settled(line, EngineTorchRequirement::Absent), "{line}");
        }
    }
}
