// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The case-count rule for property tests that set their own count, shared by
//! the `rocm-core` and `rocm-engine-protocol` integration tests and the
//! Lemonade engine's unit tests, which include this file by `#[path]`.
//!
//! It is the same rule as `rocm-core`'s crate-internal
//! `test_support::proptest_cases`, which integration tests and other crates
//! cannot reach.

/// Case count for a property test: `PROPTEST_CASES` (proptest's own knob,
/// which an explicit count would otherwise override) when it is set and above
/// `default`, otherwise `default`.
///
/// Never below `default`: each count is chosen so its property reaches the
/// cases it targets, and `PROPTEST_CASES=0` would otherwise pass every
/// property without running one. A value that is ignored, because it is lower
/// or does not parse, is reported on stderr.
pub fn proptest_cases(default: u32) -> u32 {
    let requested = std::env::var("PROPTEST_CASES").ok();
    let cases = raised_case_count(requested.as_deref(), default);
    if let Some(requested) = requested
        && requested.parse::<u32>().ok() != Some(cases)
    {
        eprintln!(
            "PROPTEST_CASES={requested} ignored: this property runs at least {default} cases"
        );
    }
    cases
}

/// The pure half of [`proptest_cases`]: `requested` raises `default` and never
/// lowers it.
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
