// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The case-count rule for property tests that set their own count. This file
//! is its one implementation: `rocm`'s and `rocm-core`'s crate-internal
//! `test_support`, the `rocm-core` and `rocm-engine-protocol` integration
//! tests, and the Lemonade engine's unit tests all include it by `#[path]`.
//! `archive_props::run_property` applies the same rule inline, since that file
//! is itself included by `#[path]` into three crates.

/// Case count for a property test: `PROPTEST_CASES` (proptest's own knob,
/// which an explicit count would otherwise override) when it is set and above
/// `default`, otherwise `default`.
///
/// Never below `default`: each count is chosen so its property reaches the
/// cases it targets, and some of those are hit by a small share of cases, so a
/// lower count would pass with the defect present. `PROPTEST_CASES=0` would
/// otherwise pass every property without running one.
pub fn proptest_cases(default: u32) -> u32 {
    case_count_from_env("PROPTEST_CASES", default)
}

/// Case count read from `var`, raising `default` but never lowering it, for
/// the same reason as [`proptest_cases`]. `rocm`'s uninstall properties use it
/// with their own knob, `ROCM_DELETION_PROP_CASES`. A value that is ignored,
/// because it is lower or does not parse, is reported on stderr.
pub fn case_count_from_env(var: &str, default: u32) -> u32 {
    let requested = std::env::var(var).ok();
    let cases = raised_case_count(requested.as_deref(), default);
    if let Some(requested) = requested
        && requested.parse::<u32>().ok() != Some(cases)
    {
        eprintln!("{var}={requested} ignored: this property runs at least {default} cases");
    }
    cases
}

/// The pure half of [`case_count_from_env`]: `requested` raises `default` and
/// never lowers it.
pub fn raised_case_count(requested: Option<&str>, default: u32) -> u32 {
    requested
        .and_then(|value| value.parse().ok())
        .map_or(default, |requested: u32| requested.max(default))
}

#[test]
fn proptest_cases_can_raise_the_default_but_never_lower_it() {
    assert_eq!(raised_case_count(None, 24), 24);
    assert_eq!(raised_case_count(Some("1000"), 24), 1000);
    assert_eq!(raised_case_count(Some("0"), 24), 24);
    assert_eq!(raised_case_count(Some("8"), 24), 24);
    assert_eq!(raised_case_count(Some("lots"), 24), 24);
}
