// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The tensorizer release rocm-cli installs over vLLM's own exact pin, and the
//! judgement that a `uv pip check` finding about it is deliberate.
//!
//! vLLM's wheel declares `tensorizer==2.10.1`, and that release does not import
//! under Python 3.14 (its libsodium-constants metaclass reads class annotations
//! the way Python 3.13 and earlier stored them). ROCm 10.1.0's vLLM is published
//! for Python 3.14 only, so the engine install upgrades tensorizer to
//! [`TENSORIZER_PYTHON314_FIX`] when the pinned release cannot be imported. The
//! exact pin then reads as unmet in `uv pip check`, which both the vLLM engine
//! (to decide whether to reinstall) and the CLI (to report `dependency_check`)
//! must tell apart from a defect; they share the rule here.

use crate::ViolationSubject;

/// The first tensorizer release that imports under Python 3.14 (coreweave/tensorizer#204).
pub const TENSORIZER_PYTHON314_FIX: &str = "2.12.1";

/// The package whose pin rocm-cli deliberately overrides.
pub const TENSORIZER_PACKAGE: &str = "tensorizer";

/// `uv pip install` requirement for the override.
pub fn tensorizer_override_requirement() -> String {
    format!("{TENSORIZER_PACKAGE}=={TENSORIZER_PYTHON314_FIX}")
}

/// Whether `version` is at least [`TENSORIZER_PYTHON314_FIX`].
///
/// Compares the dotted numeric release. A version that does not start with digits, or that is
/// a pre-release (`2.12.1rc1`), is not treated as fixed.
pub fn tensorizer_version_is_fixed(version: &str) -> bool {
    fn release(version: &str) -> Option<Vec<u64>> {
        // `uv` prints PEP 440 versions: a local segment follows `+`, and no `-` or
        // space appears in a normalised release.
        let public = version.split('+').next()?;
        let mut parts = Vec::new();
        for part in public.split('.') {
            let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
            // A post-release of a release is that release or later.
            if digits.is_empty() && part.starts_with("post") {
                break;
            }
            // `dev0` and the like: not a plain release.
            if digits.is_empty() {
                return None;
            }
            // `1rc1`, `1a2`, `1b3`: a pre-release is not the release it precedes.
            if digits.len() != part.len() {
                return None;
            }
            parts.push(digits.parse().ok()?);
        }
        (!parts.is_empty()).then_some(parts)
    }
    match (release(version), release(TENSORIZER_PYTHON314_FIX)) {
        (Some(have), Some(need)) => have >= need,
        _ => false,
    }
}

/// Whether this unmet requirement is the tensorizer override.
///
/// The requirement must be an exact pin on tensorizer below the fix (the pin that
/// cannot be imported) and the installed release at least the fix. A tensorizer
/// that is missing or older is a real defect, and so is a pin on a release that
/// is itself fixed: that one is unmet for some other reason.
pub fn is_tensorizer_override(subject: &ViolationSubject) -> bool {
    subject.package.eq_ignore_ascii_case(TENSORIZER_PACKAGE)
        && subject
            .required
            .as_deref()
            .is_some_and(|required| !tensorizer_version_is_fixed(required))
        && subject
            .installed
            .as_deref()
            .is_some_and(tensorizer_version_is_fixed)
}

/// The tensorizer release installed when `detail` is vLLM's own unmet tensorizer
/// pin that [`is_tensorizer_override`] accepts.
///
/// `detail` is one `uv pip check` line. Only vLLM's requirement qualifies: another
/// package's unmet tensorizer pin is not something the vLLM install put there.
/// Returning the installed release lets a caller print the version the runtime
/// holds instead of the constant it was upgraded to.
pub fn vllm_tensorizer_override_installed(detail: &str) -> Option<String> {
    let (requirer, _) = detail.split_once(" requires `")?;
    if !requirer.contains("`vllm`") {
        return None;
    }
    let subject = crate::violation_subject(detail)?;
    if !is_tensorizer_override(&subject) {
        return None;
    }
    subject.installed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(package: &str, installed: Option<&str>) -> ViolationSubject {
        ViolationSubject {
            package: package.to_owned(),
            required: Some("2.10.1".to_owned()),
            installed: installed.map(str::to_owned),
        }
    }

    #[test]
    fn versions_at_or_above_the_fix_count_as_fixed() {
        assert!(tensorizer_version_is_fixed("2.12.1"));
        assert!(tensorizer_version_is_fixed("2.12.2"));
        assert!(tensorizer_version_is_fixed("2.13.0"));
        assert!(tensorizer_version_is_fixed("3.0"));
        assert!(!tensorizer_version_is_fixed("2.12.0"));
        assert!(!tensorizer_version_is_fixed("2.10.1"));
        assert!(!tensorizer_version_is_fixed("2.9.9"));
        assert!(!tensorizer_version_is_fixed("garbage"));
    }

    #[test]
    fn only_a_fixed_tensorizer_is_the_override() {
        assert!(is_tensorizer_override(&subject(
            "tensorizer",
            Some("2.12.1")
        )));
        assert!(is_tensorizer_override(&subject(
            "Tensorizer",
            Some("2.12.1")
        )));
        assert!(!is_tensorizer_override(&subject(
            "tensorizer",
            Some("2.9.0")
        )));
        assert!(!is_tensorizer_override(&subject("tensorizer", None)));
        assert!(!is_tensorizer_override(&subject("torch", Some("2.12.1"))));
        // A pin on a release that is not the broken one is unmet for another reason.
        let mut newer_pin = subject("tensorizer", Some("2.12.1"));
        newer_pin.required = Some("2.13.0".to_owned());
        assert!(!is_tensorizer_override(&newer_pin));
        let mut loose_pin = subject("tensorizer", Some("2.12.1"));
        loose_pin.required = None;
        assert!(!is_tensorizer_override(&loose_pin));
    }

    #[test]
    fn a_prerelease_of_the_fix_is_not_fixed() {
        assert!(!tensorizer_version_is_fixed("2.12.1rc1"));
        assert!(!tensorizer_version_is_fixed("2.12.1.dev0"));
        assert!(tensorizer_version_is_fixed("2.12.1+local"));
        // Only a `+` local segment is dropped; anything else after the release is
        // not a plain release.
        assert!(!tensorizer_version_is_fixed("2.12.1-1"));
        assert!(!tensorizer_version_is_fixed("2.12.1 (x)"));
        // A post-release of the fixed release is at least that release.
        assert!(tensorizer_version_is_fixed("2.12.1.post1"));
        assert!(!tensorizer_version_is_fixed("2.12.0.post1"));
    }

    #[test]
    fn only_vllms_own_unmet_pin_is_reported_with_the_installed_release() {
        let pin = |requirer: &str, installed: &str| {
            format!(
                "The package `{requirer}` requires `tensorizer==2.10.1`, but `{installed}` is installed"
            )
        };
        assert_eq!(
            vllm_tensorizer_override_installed(&pin("vllm", "2.12.1")).as_deref(),
            Some("2.12.1")
        );
        assert_eq!(
            vllm_tensorizer_override_installed(&pin("vllm", "2.13.0")).as_deref(),
            Some("2.13.0")
        );
        assert_eq!(
            vllm_tensorizer_override_installed(&pin("lemonade", "2.12.1")),
            None
        );
        assert_eq!(
            vllm_tensorizer_override_installed(&pin("vllm", "2.9.0")),
            None
        );
    }

    #[test]
    fn the_override_requirement_names_the_fix() {
        assert_eq!(tensorizer_override_requirement(), "tensorizer==2.12.1");
    }
}
