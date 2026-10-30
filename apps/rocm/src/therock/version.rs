// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Version grammar for TheRock runtimes: the strict gate an install is routed
//! on ([`parse_version`]), the PEP 440 ordering used to pick the newest build
//! ([`compare_version_strings`], [`version_relation`]), the host-version
//! reader, and the build-date helpers behind `--build-date` and the version
//! display.
//!
//! A leaf of the `therock` module: nothing here depends on the rest of it.

use anyhow::{Context, Result, bail};
use std::cmp::Ordering;

/// What [`parse_version`]'s strict grammar gate yields: only the parts a
/// caller reads. The patch and stage number are still parsed, so a malformed
/// one still fails the gate, but nothing orders by them here — ordering goes
/// through [`VersionOrderKey`].
#[derive(Debug, Clone)]
pub(super) struct ParsedVersion {
    pub(super) major: u32,
    pub(super) minor: u32,
    pub(super) stage: VersionStage,
}

/// A version's position in the PEP 440 order, as produced by
/// [`parse_version_for_ordering`].
///
/// Separate from [`ParsedVersion`] because the two answer different questions.
/// `ParsedVersion` is what [`parse_version`]'s strict grammar gate yields, and
/// is never compared: the PyTorch stack matchers read its `major`, `minor` and
/// `stage`, and the layout selection its `major`. This one exists only to be
/// compared, so its release segment is a vector. Callers do not build or edit
/// one; only `release` and `stage` are `pub(super)`, because the property
/// tests read them.
///
/// Field order is the comparison order, and it is PEP 440's: release segment
/// first, then stage, then the stage's number.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct VersionOrderKey {
    /// Release components, trailing zeros removed so that `7.9`, `7.9.0` and
    /// `7.9.0.0` share one key and plain `Vec` ordering matches PEP 440's
    /// zero-padded comparison.
    pub(super) release: Vec<u64>,
    pub(super) stage: VersionStage,
    stage_number: u64,
}

/// Where a version sits within its release.
///
/// Declaration order is the ordering, and it is PEP 440's: a dev release
/// precedes its alpha, which precedes beta, release candidate, the final
/// release, and finally any post-release of that final release.
///
/// [`parse_version`] only ever yields `Alpha`, `Rc` or `Stable` — it is the
/// strict grammar gate for "a version shape this tool will route an install
/// on", not an ordering parser. The remaining variants come from
/// [`parse_version_for_ordering`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(super) enum VersionStage {
    Dev,
    Alpha,
    Beta,
    Rc,
    Stable,
    Post,
}

/// Total order over published version strings — the total order used for
/// sorting candidates in the `therock` module's "which build does the user get?"
/// decisions. It is not always the first key: [`super::select_tarball_candidate`]
/// orders a nightly catalog by upload time first and uses this only to break
/// ties. The release channel orders by this comparison first.
///
/// Ordering is PEP 440's, because these strings come from a Python simple
/// index: within one release `7.0.0.dev1` < `7.0.0a1` < `7.0.0b1` <
/// `7.0.0rc1` < `7.0.0` < `7.0.0.post1`. The release segment is compared
/// component-wise with the shorter side zero-padded, so `7.9` is `7.9.0` and
/// precedes `7.10.0`, and `7.2.4.70204` — the shape ROCm's own packages are
/// named with — is above `7.2.4` and below `7.2.5`.
///
/// Where this is not PEP 440, it declines to answer rather than guessing: an
/// epoch (`1!2.0`), a combined stage suffix (`7.0.0rc1.dev1`) and anything else
/// unreadable all return `None` from [`parse_version_for_ordering`]. Local
/// version metadata is dropped rather than ordered, so `7.9.0+local` ties with
/// `7.9.0` and is separated by the string tiebreak below.
///
/// A string that cannot be read keeps a *defined* position instead of switching
/// the comparator to a different relation: it sorts below everything readable,
/// and unreadable strings are ordered against each other by the string itself.
/// Sorting below is the safe end for the "newest candidate" pickers, which take
/// the maximum: an unreadable string can only win when every candidate is
/// unreadable, rather than beating a real release by being lexicographically
/// longer. It would cut the *other* way for [`super::runtime_freshness`], where the
/// unreadable string is the installed version and "lowest" would read as "older
/// than the index" — an offer that could be a downgrade.
///
/// Versions that are equal but textually different (`7.9` and `7.9.0`,
/// `7.0.0-rc1` and `7.0.0rc1`, `7.9.0` and `7.9.0+local`) are separated by the
/// raw string, so no two distinct strings compare `Equal` and `sort_by` is
/// deterministic. That tiebreak is right for sorting and wrong for deciding
/// whether a build is new: a caller asking "is there a newer runtime?" must use
/// [`version_relation`], which compares the parsed keys, so that a spelling
/// difference is not mistaken for an update.
///
/// This replaces an implementation that compared numerically only when *both*
/// sides matched [`parse_version`]'s grammar and fell back to a byte-wise
/// string compare otherwise. Mixing two relations in one comparator is not
/// transitive — `7.0.0 < 7.0.0b1`, `7.0.0b1 < 7.0.0rc1`, yet `7.0.0 >
/// 7.0.0rc1` — so `sort_by` had no defined result, "newest" depended on the
/// order the index listed its candidates in, and `runtime_freshness` could
/// offer a downgrade as an update.
pub(super) fn compare_version_strings(left: &str, right: &str) -> Ordering {
    match (
        parse_version_for_ordering(left),
        parse_version_for_ordering(right),
    ) {
        (Some(left_parsed), Some(right_parsed)) => {
            left_parsed.cmp(&right_parsed).then_with(|| left.cmp(right))
        }
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => left.cmp(right),
    }
}

/// How two versions relate *as versions*, for the callers asking "is this a
/// different build?" rather than "where does this sort?".
///
/// Those are not the same question, and [`compare_version_strings`] only
/// answers the second. It is a total order over *strings*, which is what
/// sorting needs, and it gets there two ways this caller must not inherit:
///
/// * Its last tiebreak is the raw string, so two spellings of one version sort
///   apart. `7.0.0-rc1` and `7.0.0rc1` are the same version — that is why the
///   parser accepts both — but they are different strings. Treating that as a
///   difference offers an update that re-downloads the runtime already
///   installed. PyPI normalises versions, so an index can serve one spelling
///   while an older manifest recorded the other.
/// * A string it cannot read sorts below everything readable. That is the safe
///   end for the "newest candidate" pickers, which take a maximum, and the
///   wrong end here, where the unreadable string is the *installed* version and
///   "lowest" reads as "older than the index" — an offer that could be a
///   downgrade. A runtime adopted from an existing environment carries whatever
///   its probe reported, so this is reachable without any index being odd.
///
/// So: compare the parsed keys, which are equal exactly when the two strings
/// denote one version, and return `None` when either side cannot be identified
/// at all. Byte-identical strings short-circuit, which keeps the
/// repair/up-to-date path working for a runtime whose version this tool cannot
/// parse but which plainly matches what the index offers.
pub(super) fn version_relation(left: &str, right: &str) -> Option<Ordering> {
    if left == right {
        return Some(Ordering::Equal);
    }
    let left_key = parse_version_for_ordering(left)?;
    let right_key = parse_version_for_ordering(right)?;
    Some(left_key.cmp(&right_key))
}

/// Lenient parse of a published version string, for ordering only.
///
/// Reads a PEP 440 public version: a release segment of any length, followed by
/// at most one `dev`/`a`/`b`/`rc`/`post` stage. Because the module declares the
/// order is PEP 440's, the spellings PEP 440 treats as equal are accepted as
/// equal — case (`7.0.0RC1`), any of `.`, `-`, `_` or nothing as the separator
/// before a stage label (`7.0.0-rc1`, `7.0.0_rc1`, `7.0.0rc1`), the alias
/// labels (`alpha`, `beta`, `c`, `pre`, `preview`, `rev`, `r`), an omitted
/// stage numeral (`7.0.0rc` is `rc0`), and a leading `v`.
///
/// The release segment is a *vector*, not three fixed components, because a
/// four-component version is a shape this project really meets: ROCm's own
/// packages are named `7.2.4.70204`. Folding that onto `7.2.4`, or rejecting
/// it, both put it in the wrong place relative to a plain `6.4.3`.
///
/// Returns `None` rather than approximating where this stops being PEP 440,
/// because a shape read *approximately* is a silently wrong order while `None`
/// lands in the position [`compare_version_strings`] documents:
/// * An epoch (`1!2.0`), which changes what the release numbers mean.
/// * A combined stage suffix (`7.0.0rc1.dev1`).
/// * PEP 440's implicit post-release (`7.2.4-98` meaning `7.2.4.post98`).
///   That spelling collides with the packaging build number a system ROCm
///   reports, which [`parse_host_version`] must read as plain `7.2.4`.
///   Implementing the PEP 440 reading here would put the two parsers back into
///   disagreement about one string, which is the defect this shared parser
///   exists to remove; leaving it unreadable keeps them consistent and costs
///   only a defined position.
///
///   What makes the two sides agree is that [`parse_host_version`] strips a
///   `-` suffix only when it is *numeric*. A host `7.2.4-98` is reduced to
///   `7.2.4` before it reaches here, so this function never sees that shape
///   from the host; and `7.0.0-rc1` is left intact, so both sides read it as a
///   release candidate. Widening either side of that rule re-opens the
///   disagreement.
///
/// Local version metadata is dropped rather than ordered, so `7.9.0+local` and
/// `7.9.0` produce the same key and [`compare_version_strings`] separates them
/// by the string. PEP 440 would sort the local version above the plain one;
/// nothing here depends on that, and the string tiebreak happens to agree.
pub(super) fn parse_version_for_ordering(value: &str) -> Option<VersionOrderKey> {
    // Drop local version metadata: `7.9.0+local` -> `7.9.0`.
    let value = value.split('+').next().unwrap_or(value).trim();
    // PEP 440 is case-insensitive (`7.0.0RC1` is `7.0.0rc1`).
    let value = value.to_ascii_lowercase();
    let value = value.strip_prefix('v').unwrap_or(&value);
    if value.is_empty() {
        return None;
    }

    // An epoch (`1!2.0`) changes what the release numbers mean, so a version
    // carrying one must not be ordered as though it did not. This check is
    // redundant today and kept as documentation: the `!` is neither a release
    // digit nor a stage label, so `parse_version_stage` below rejects every
    // epoch anyway. That rejection is incidental, though; this is the line
    // that states the rule, and it keeps holding if the stage parser ever
    // accepts more.
    if value.contains('!') {
        return None;
    }

    let mut release = Vec::new();
    let mut rest = value;
    loop {
        let (number, tail) = split_leading_number(rest)?;
        release.push(number);
        match tail.strip_prefix('.') {
            // Another release component only if a digit follows the dot;
            // `.post1` and `.dev1` are stages, not components.
            Some(next) if next.starts_with(|ch: char| ch.is_ascii_digit()) => rest = next,
            _ => {
                rest = tail;
                break;
            }
        }
    }
    // `7.9`, `7.9.0` and `7.9.0.0` are one release. Stripping trailing zeros
    // makes the plain `Vec` comparison below agree with PEP 440's rule of
    // zero-padding the shorter release to the longer one.
    while release.last() == Some(&0) {
        release.pop();
    }

    let (stage, stage_number) = parse_version_stage(rest)?;
    Some(VersionOrderKey {
        release,
        stage,
        stage_number,
    })
}

/// Split a leading run of ASCII digits off `value`, returning it and the rest.
/// `None` when `value` does not start with a digit, or the digits overflow.
fn split_leading_number(value: &str) -> Option<(u64, &str)> {
    let digits = value.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    Some((value[..digits].parse().ok()?, &value[digits..]))
}

/// Map a PEP 440 stage suffix onto its [`VersionStage`] and number.
///
/// Accepts the spellings PEP 440 normalises away: any of `.`, `-`, `_` or no
/// separator before the label, the alias labels (`alpha`, `beta`, `c`, `pre`,
/// `preview`, `rev`, `r`), and an omitted numeral, which PEP 440 defines as
/// zero. `value` is already lowercased by the caller.
///
/// A combined suffix (`7.0.0rc1.dev1`) is rejected rather than folded onto the
/// first stage it happens to match: an approximate position is a silently wrong
/// order, while `None` lands in the place [`compare_version_strings`]
/// documents.
fn parse_version_stage(suffix: &str) -> Option<(VersionStage, u64)> {
    if suffix.is_empty() {
        return Some((VersionStage::Stable, 0));
    }
    let label = suffix.strip_prefix(['.', '-', '_']).unwrap_or(suffix);
    // Longest spelling first: `a` would otherwise swallow `alpha`, and `r`
    // would swallow `rev`, matching and then failing on the leftover letters.
    for (prefix, stage) in [
        ("preview", VersionStage::Rc),
        ("alpha", VersionStage::Alpha),
        ("beta", VersionStage::Beta),
        ("post", VersionStage::Post),
        ("pre", VersionStage::Rc),
        ("dev", VersionStage::Dev),
        ("rev", VersionStage::Post),
        ("rc", VersionStage::Rc),
        ("a", VersionStage::Alpha),
        ("b", VersionStage::Beta),
        ("c", VersionStage::Rc),
        ("r", VersionStage::Post),
    ] {
        let Some(number) = label.strip_prefix(prefix) else {
            continue;
        };
        // PEP 440: an omitted numeral is zero, so `7.0.0rc` is `7.0.0rc0`.
        let number = number.trim_start_matches(['.', '-', '_']);
        return Some((
            stage,
            if number.is_empty() {
                0
            } else {
                number.parse().ok()?
            },
        ));
    }
    None
}

pub(super) fn parse_version(value: &str) -> Option<ParsedVersion> {
    let value = value.split('+').next().unwrap_or(value);
    let mut parts = value.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_and_rest = parts.next()?;

    let patch_len = patch_and_rest
        .chars()
        .take_while(char::is_ascii_digit)
        .count();
    if patch_len == 0 {
        return None;
    }
    // Parsed only to be validated: nothing reads the patch or stage number.
    patch_and_rest[..patch_len].parse::<u32>().ok()?;
    let suffix = &patch_and_rest[patch_len..];

    let stage = if suffix.is_empty() {
        VersionStage::Stable
    } else if let Some(rest) = suffix.strip_prefix("rc") {
        rest.parse::<u64>().ok()?;
        VersionStage::Rc
    } else if let Some(rest) = suffix.strip_prefix('a') {
        rest.parse::<u64>().ok()?;
        VersionStage::Alpha
    } else {
        return None;
    };

    Some(ParsedVersion {
        major,
        minor,
        stage,
    })
}

/// Lenient parse of a host-reported ROCm version for the "is the host newer?"
/// decision.
///
/// [`parse_version_for_ordering`] already handles every shape this needs except
/// one: a legacy/system ROCm reports a packaging build number as `7.2.4-98`,
/// which is not a PEP 440 spelling of anything.
///
/// Only a *numeric* `-` suffix is dropped. `-` is also a PEP 440 pre-release
/// separator, so stripping it unconditionally would read `7.0.0-rc1` as the
/// final `7.0.0` — and then this function and the comparator would disagree
/// about that string, which is exactly the self-contradiction the shared parser
/// exists to remove.
///
/// Sharing the parser is the point: a host version and an index version are
/// compared against each other, and the two must not disagree about which is
/// newer. Returns `None` when the string still cannot be read, so an
/// unparseable host string is "can't tell" rather than a lexicographic guess.
pub(super) fn parse_host_version(value: &str) -> Option<VersionOrderKey> {
    // `7.2.4-98` -> `7.2.4`, but `7.0.0-rc1` is left for the PEP 440 parser.
    let trimmed = value.trim();
    let candidate = match trimmed.split_once('-') {
        Some((head, tail)) if tail.starts_with(|ch: char| ch.is_ascii_digit()) => head,
        _ => trimmed,
    };
    parse_version_for_ordering(candidate)
}

pub(super) fn is_stable_runtime_version(version: &str) -> bool {
    parse_version(version).is_some_and(|parsed| parsed.stage == VersionStage::Stable)
}

pub(crate) fn runtime_version_display(version: &str) -> String {
    if let Some(date) = runtime_version_build_date(version) {
        format!("{version} (build {date})")
    } else {
        version.to_owned()
    }
}

pub(super) fn runtime_version_build_date(version: &str) -> Option<String> {
    let bytes = version.as_bytes();
    if bytes.len() < 8 {
        return None;
    }
    for window in bytes.windows(8) {
        if !window.iter().all(u8::is_ascii_digit) {
            continue;
        }
        let digits = std::str::from_utf8(window).ok()?;
        let year = digits[0..4].parse::<u32>().ok()?;
        let month = digits[4..6].parse::<u32>().ok()?;
        let day = digits[6..8].parse::<u32>().ok()?;
        if !(2000..=2099).contains(&year) || month == 0 || month > 12 {
            continue;
        }
        let max_day = days_in_month(year, month);
        if day == 0 || day > max_day {
            continue;
        }
        return Some(format!("{year:04}-{month:02}-{day:02}"));
    }
    None
}

pub(super) fn normalize_requested_build_date(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("TheRock build date cannot be empty");
    }
    let digits = trimmed
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>();
    if digits.len() != 8 {
        bail!("TheRock build date `{trimmed}` must use YYYY-MM-DD, YYYYMMDD, or MMDDYYYY");
    }

    let parsed = if digits.starts_with("20") {
        parse_yyyy_mm_dd(&digits)
    } else if digits[4..].starts_with("20") {
        parse_mm_dd_yyyy(&digits)
    } else {
        None
    };
    let Some((year, month, day)) = parsed else {
        bail!("TheRock build date `{trimmed}` must use YYYY-MM-DD, YYYYMMDD, or MMDDYYYY");
    };
    validate_date_components(year, month, day)
        .with_context(|| format!("invalid TheRock build date `{trimmed}`"))?;
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

fn parse_yyyy_mm_dd(digits: &str) -> Option<(u32, u32, u32)> {
    Some((
        digits[0..4].parse().ok()?,
        digits[4..6].parse().ok()?,
        digits[6..8].parse().ok()?,
    ))
}

fn parse_mm_dd_yyyy(digits: &str) -> Option<(u32, u32, u32)> {
    Some((
        digits[4..8].parse().ok()?,
        digits[0..2].parse().ok()?,
        digits[2..4].parse().ok()?,
    ))
}

fn validate_date_components(year: u32, month: u32, day: u32) -> Result<()> {
    if !(2000..=2099).contains(&year) {
        bail!("year must be between 2000 and 2099");
    }
    if month == 0 || month > 12 {
        bail!("month must be between 1 and 12");
    }
    let max_day = days_in_month(year, month);
    if day == 0 || day > max_day {
        bail!("day must be between 1 and {max_day}");
    }
    Ok(())
}

const fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

const fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_version_display_mentions_embedded_build_date() {
        assert_eq!(
            runtime_version_display("7.14.0a20260601"),
            "7.14.0a20260601 (build 2026-06-01)"
        );
        assert_eq!(
            runtime_version_display("2.11.0+rocm7.13.0a20260416"),
            "2.11.0+rocm7.13.0a20260416 (build 2026-04-16)"
        );
        assert_eq!(runtime_version_display("7.14.0"), "7.14.0");
        assert_eq!(
            runtime_version_build_date("7.14.0a20260230"),
            None,
            "invalid calendar dates should not be displayed"
        );
    }
}
