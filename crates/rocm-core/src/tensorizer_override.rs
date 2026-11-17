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

/// The numeric release of a plain version, or `None` for anything else.
///
/// Accepted: dotted components that are runs of ASCII digits, then optionally one
/// last `post<digits>` component (a post-release counts as the release it follows),
/// and then optionally one `+local` segment made only of ASCII letters, digits and
/// dots with no empty component (`+rocm10.1.0.g98dff2a81.d20261002`, `+x`). The
/// local segment is checked but not read.
/// Everything else has no release and fails closed: a pre-release (`2.12.1rc1`,
/// `dev0`), a bare `post`, text after the `post<digits>` component, a marker or
/// other trailing text, `-`, spaces, non-ASCII digits, a component too large for
/// `u64`, an empty or repeated `+` segment, and the empty string.
///
/// Leading zeros read as the number (`2.10.01` is 2.10.1). Trailing zero components
/// are kept: a longer release compares greater than its prefix, which already gives
/// the right answer for the one release this is compared with (`2.12.1.0` is not
/// below 2.12.1, `2.10.1.0` is, and `2.12` is) without normalising them away.
fn release(version: &str) -> Option<Vec<u64>> {
    let (public, local) = match version.split_once('+') {
        Some((public, local)) => (public, Some(local)),
        None => (version, None),
    };
    if let Some(local) = local
        && !local
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_alphanumeric()))
    {
        // Empty, a second `+`, or anything but letters, digits and dots.
        return None;
    }
    let mut parts = Vec::new();
    let mut components = public.split('.').peekable();
    while let Some(part) = components.next() {
        if let Some(rest) = part.strip_prefix("post")
            && !parts.is_empty()
            && components.peek().is_none()
            && !rest.is_empty()
            && rest.bytes().all(|byte| byte.is_ascii_digit())
        {
            // `post<digits>` as the last component: the release it follows.
            break;
        }
        // `u64::parse` is the digit guard: it rejects an empty component, a sign
        // (`-1`; a `+` cannot reach here, the string was split on it), non-ASCII
        // digits, anything non-numeric and an overflow.
        parts.push(part.parse::<u64>().ok()?);
    }
    (!parts.is_empty()).then_some(parts)
}

/// Whether `version` is at least [`TENSORIZER_PYTHON314_FIX`].
///
/// Compares the dotted numeric release. A version that has no plain release is
/// not treated as fixed.
pub fn tensorizer_version_is_fixed(version: &str) -> bool {
    match (release(version), release(TENSORIZER_PYTHON314_FIX)) {
        (Some(have), Some(need)) => have >= need,
        _ => false,
    }
}

/// Whether `version` is a plain release strictly below the fix. Unlike
/// `!tensorizer_version_is_fixed`, a version that cannot be read is not below it.
fn tensorizer_version_is_below_fix(version: &str) -> bool {
    match (release(version), release(TENSORIZER_PYTHON314_FIX)) {
        (Some(have), Some(need)) => have < need,
        _ => false,
    }
}

/// Whether this unmet requirement is the tensorizer override.
///
/// `violation_subject` fills `required` only for a lone `==` pin, as the text after
/// the `==`. That text must read as a plain release strictly below the fix (the
/// pin that cannot be imported); a pin that does not parse (garbage, empty, or
/// qualified by an environment marker, which leaves trailing text) stays a
/// violation, as does a pin on a release at or above the fix. The installed
/// release must be at least the fix. A tensorizer that is missing or older is a
/// real defect.
pub fn is_tensorizer_override(subject: &ViolationSubject) -> bool {
    subject.package.eq_ignore_ascii_case(TENSORIZER_PACKAGE)
        && subject
            .required
            .as_deref()
            .is_some_and(tensorizer_version_is_below_fix)
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
    let (requirer, _) = detail.split_once(crate::uv::UV_CHECK_REQUIREMENT_INFIX)?;
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
        // A pin that cannot be read must not pass as the deliberate override.
        for required in [
            "garbage",
            "",
            "2.10.1; python_version >= \"3.14\"",
            "2.10.1rc1",
            "2.12.1",
            "2.12.1+local",
        ] {
            let mut pin = subject("tensorizer", Some("2.12.1"));
            pin.required = Some(required.to_owned());
            assert!(
                !is_tensorizer_override(&pin),
                "{required:?} must stay a violation"
            );
        }
        // The plain pin still counts, with a local segment too.
        for required in ["2.10.1", "2.11.0", "2.10.1+local"] {
            let mut pin = subject("tensorizer", Some("2.12.1"));
            pin.required = Some(required.to_owned());
            assert!(is_tensorizer_override(&pin), "{required:?}");
        }
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
    fn a_post_release_is_read_only_in_its_plain_shape() {
        // Accepted: post<digits> as the last component, with or without +local.
        assert!(tensorizer_version_is_fixed("2.12.1.post1"));
        assert!(tensorizer_version_is_fixed("2.12.1.post0+local"));
        assert!(!tensorizer_version_is_fixed("2.12.0.post1"));
        let pin = |required: &str| {
            let mut pin = subject("tensorizer", Some("2.12.1"));
            pin.required = Some(required.to_owned());
            pin
        };
        assert!(is_tensorizer_override(&pin("2.10.1.post1")));
        assert!(is_tensorizer_override(&pin("2.10.1.post1+local")));
        // Anything else around `post` fails closed: the pin stays a violation and
        // an installed version of that shape is not fixed.
        for text in [
            "2.10.1.postgarbage",
            "2.10.1.post1.foo",
            "2.10.1.post",
            "2.10.1.post1x",
            "2.10.1.post1.post2",
            "2.10.1.post-1",
            "post1",
        ] {
            assert!(!is_tensorizer_override(&pin(text)), "{text:?} pin");
            let installed = text.replace("2.10.1", "2.12.1");
            assert!(
                !tensorizer_version_is_fixed(&installed),
                "{installed:?} installed"
            );
        }
    }

    #[test]
    fn spellings_of_one_release_compare_equal_and_unreadable_ones_fail_closed() {
        // Trailing zero components and leading zeros do not change the verdict.
        assert!(tensorizer_version_is_fixed("2.12.1.0"));
        assert!(tensorizer_version_is_fixed("2.12.01"));
        assert!(!tensorizer_version_is_fixed("2.12.0.0"));
        assert!(!tensorizer_version_is_fixed("2.12"));
        assert!(tensorizer_version_is_below_fix("2.10.1.0"));
        assert!(tensorizer_version_is_below_fix("2.10.01"));
        assert!(tensorizer_version_is_below_fix("2.12"));
        assert!(!tensorizer_version_is_below_fix("2.12.1.0"));
        assert!(!tensorizer_version_is_below_fix("2.12.01"));
        // Unreadable: neither fixed nor below the fix.
        for text in [
            "",
            "2..1",
            ".2.1",
            "2.10.1-1",
            "2.10.1 ",
            "2.１0.1",
            "2.10.99999999999999999999999",
        ] {
            assert!(!tensorizer_version_is_fixed(text), "{text:?} fixed");
            assert!(!tensorizer_version_is_below_fix(text), "{text:?} below");
        }
    }

    #[test]
    fn the_requirement_frame_is_the_one_uv_check_parsing_uses() {
        // Built from the shared constant, not a literal, so a change to the frame
        // `violation_subject` splits on is a change here too.
        let detail = format!(
            "The package `vllm`{}tensorizer==2.10.1`, but `2.12.1` is installed",
            crate::uv::UV_CHECK_REQUIREMENT_INFIX
        );
        assert_eq!(
            vllm_tensorizer_override_installed(&detail).as_deref(),
            Some("2.12.1")
        );
    }

    #[test]
    fn a_local_segment_is_read_only_in_its_plain_shape() {
        let pin = |required: &str| {
            let mut pin = subject("tensorizer", Some("2.12.1"));
            pin.required = Some(required.to_owned());
            pin
        };
        // Real local segments, and the short ones, still read.
        for local in ["+rocm10.1.0.g98dff2a81.d20261002", "+local", "+x", "+1"] {
            assert!(
                is_tensorizer_override(&pin(&format!("2.10.1{local}"))),
                "pin {local}"
            );
            assert!(
                tensorizer_version_is_fixed(&format!("2.12.1{local}")),
                "installed {local}"
            );
        }
        // Malformed ones do not: the pin stays a violation, the installed version
        // is not fixed.
        for local in [
            "+",
            "+a+b",
            "+a..b",
            "+.a",
            "+a.",
            "+a-b",
            "+a b",
            "+\u{ff11}",
        ] {
            assert!(
                !is_tensorizer_override(&pin(&format!("2.10.1{local}"))),
                "pin {local:?}"
            );
            assert!(
                !tensorizer_version_is_fixed(&format!("2.12.1{local}")),
                "installed {local:?}"
            );
        }
    }

    #[test]
    fn the_override_requirement_names_the_fix() {
        assert_eq!(tensorizer_override_requirement(), "tensorizer==2.12.1");
    }
}
