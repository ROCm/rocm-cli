// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The AMD display driver on Windows: what version and date it has, and
//! whether it is old.
//!
//! Three commands report this driver: `rocm examine` prints it,
//! `rocm install driver` describes it, and `rocm diagnose` raises
//! [`DISPLAY_DRIVER_OLD_FIX_ID`] from it. What is guaranteed:
//!
//! - The comparison is one function here, so the same version and date always
//!   give the same verdict.
//! - The facts come from one place. All three commands take the version and
//!   date from the same Windows device inventory and the same adapter choice
//!   (`preferred_amd_display` in `host_gpu.rs`), which ranks a ROCm-supported
//!   discrete GPU above an integrated one. Given the same inventory, all three
//!   commands select the same adapter.
//!
//! What is not guaranteed: the inventory can fail to read the driver (a query
//! that times out, or an adapter without a version). The commands run their
//! queries separately, so one run can succeed where another fails. A machine
//! that runs the Microsoft Basic Display Adapter has no AMD driver loaded and is
//! reported as having no driver version, not as an old AMD driver. That last
//! case is argued from how Windows names the fallback driver, and has not been
//! checked on a Windows machine.
//!
//! This module decides nothing about compatibility. AMD publishes no minimum
//! display driver version for the HIP SDK for Windows. Its documentation says
//! only that the installer bundles a driver. AMD also publishes no minimum for
//! the Vulkan backend that Lemonade uses. Because no minimum is established,
//! the only comparison made here is against a fact that is: the first HIP SDK
//! for Windows shipped in July 2023, so a driver dated earlier cannot have been
//! released alongside any HIP SDK. That is advice to look at the driver, never
//! a claim that anything is broken. The date is only a floor: it cannot judge
//! GPUs that launched later, such as gfx120X, gfx1150 and gfx1151, which may
//! need newer drivers than a driver dated after the cutoff.
//!
//! Pure functions only. Nothing here runs a process or reads the machine.

use serde::{Deserialize, Serialize};

/// The catalog entry `rocm diagnose` raises for an old driver.
///
/// One constant, because `rocm install driver` names it on Windows and the
/// catalog registers it. A text that names an id the catalog does not hold
/// would send the user to a command that finds nothing.
pub const DISPLAY_DRIVER_OLD_FIX_ID: &str = "fix-22-display-driver-old";

/// Where AMD publishes display drivers for each GPU.
pub const AMD_SUPPORT_URL: &str = "https://www.amd.com/en/support";

/// A calendar day, ordered by year, then month, then day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CalendarDate {
    pub year: u16,
    pub month: u8,
    pub day: u8,
}

impl CalendarDate {
    /// Read `yyyy-MM-dd`, optionally followed by a time after `T` or a space.
    ///
    /// Anything else is `None`. A year before 2000 is also `None`: Windows
    /// reports an unset driver date as a placeholder far in the past, and
    /// reading that as "very old" would flag a driver for a missing value.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let (date, rest) = match text.split_once(['T', ' ']) {
            Some((date, rest)) => (date, Some(rest)),
            None => (text, None),
        };
        // A trailing part must exist for a reason: an empty one means the text
        // was `2021-08-04T`, which is not a date.
        if rest.is_some_and(|rest| rest.trim().is_empty()) {
            return None;
        }
        let mut parts = date.split('-');
        let year = parts.next()?;
        let month = parts.next()?;
        let day = parts.next()?;
        if parts.next().is_some() || year.len() != 4 || month.len() != 2 || day.len() != 2 {
            return None;
        }
        // Digits only: `str::parse` accepts a leading `+`, so `2021-+1-+4`
        // would otherwise read as a date.
        if ![year, month, day]
            .iter()
            .all(|part| part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return None;
        }
        let year: u16 = year.parse().ok()?;
        let month: u8 = month.parse().ok()?;
        let day: u8 = day.parse().ok()?;
        if year < 2000 || !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month)
        {
            return None;
        }
        Some(Self { year, month, day })
    }
}

/// Days in `month` of `year` (a Gregorian calendar month, 1 to 12).
const fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) => {
            29
        }
        2 => 28,
        _ => 31,
    }
}

impl Serialize for CalendarDate {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl std::fmt::Display for CalendarDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// The day the first HIP SDK for Windows was released (ROCm 5.5 based).
///
/// A driver dated before this day predates every HIP SDK for Windows. It is not
/// a minimum driver version. AMD does not state one. It is only a floor and
/// cannot judge GPUs that launched later (see [`FLOOR_NOTE`]).
pub const FIRST_HIP_SDK_FOR_WINDOWS: CalendarDate = CalendarDate {
    year: 2023,
    month: 7,
    day: 27,
};

/// What one driver record says, as read from the machine.
///
/// The date is parsed once, when the record is built, so a value that is not a
/// date cannot be stored. It is written to JSON as the `yyyy-MM-dd` text it
/// came from, and `age` is added to the output so a consumer does not have to
/// repeat the cutoff.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WindowsDriverInfo {
    /// The version string Windows reports, such as `30.0.13014.8`.
    pub version: String,
    /// The driver date, only when it parsed as a date.
    pub date: Option<CalendarDate>,
}

impl Serialize for WindowsDriverInfo {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let len = if self.date.is_some() { 3 } else { 2 };
        let mut out = serializer.serialize_struct("WindowsDriverInfo", len)?;
        out.serialize_field("version", &self.version)?;
        if let Some(date) = &self.date {
            out.serialize_field("date", date)?;
        }
        // Derived, never read back: `deserialize` ignores it, so it cannot
        // disagree with the date.
        out.serialize_field("age", self.age().verdict())?;
        out.end()
    }
}

impl<'de> Deserialize<'de> for WindowsDriverInfo {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            version: String,
            // A date that does not parse is dropped, not an error: the record
            // stays usable and reads as an unknown age.
            #[serde(default)]
            date: Option<String>,
        }
        let wire = Wire::deserialize(deserializer)?;
        Ok(Self {
            version: wire.version,
            date: wire.date.as_deref().and_then(CalendarDate::parse),
        })
    }
}

/// How the driver date compares with [`FIRST_HIP_SDK_FOR_WINDOWS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverAge {
    /// Dated before the first HIP SDK for Windows.
    Predates(CalendarDate),
    /// Dated on or after that day.
    NotOlder(CalendarDate),
    /// No date, or one that did not parse. Never treated as old.
    Unknown,
}

impl DriverAge {
    /// The verdict as a stable lowercase word, for machine-readable output.
    #[must_use]
    pub const fn verdict(self) -> &'static str {
        match self {
            Self::Predates(_) => "predates_first_hip_sdk",
            Self::NotOlder(_) => "not_older",
            Self::Unknown => "unknown",
        }
    }
}

/// The version prefix of a driver that ships with Windows itself.
const OS_DRIVER_VERSION_PREFIX: &str = "10.0.";

/// Whether a display adapter record is the fallback driver Windows loads when
/// no AMD driver is installed, not an AMD driver.
///
/// That adapter keeps the AMD PCI vendor id, so the inventory still lists it.
/// It is the "Microsoft Basic Display Adapter", and Windows gives its driver
/// the operating system's own `10.0.<build>` version and a placeholder date.
/// Reading that date as an AMD driver date would tell the user to update a
/// driver that is not installed.
///
/// Not verified on a Windows machine: the name and the `10.0.` version are
/// recalled from how Windows reports this driver, not observed.
#[must_use]
pub fn is_microsoft_fallback_driver(name: &str, version: &str) -> bool {
    name.to_ascii_lowercase()
        .contains("microsoft basic display")
        || version.trim().starts_with(OS_DRIVER_VERSION_PREFIX)
}

impl WindowsDriverInfo {
    /// Build a record from the two probe fields, keeping only a date that parses.
    ///
    /// `None` when there is no version, because a record without one carries
    /// nothing worth showing, and when the version is the operating system's
    /// own `10.0.` fallback driver (see [`is_microsoft_fallback_driver`]).
    #[must_use]
    pub fn from_probe(version: &str, date: &str) -> Option<Self> {
        let version = version.trim();
        if version.is_empty() || is_microsoft_fallback_driver("", version) {
            return None;
        }
        Some(Self {
            version: version.to_owned(),
            date: CalendarDate::parse(date),
        })
    }

    /// Compare the driver date with the first HIP SDK for Windows.
    #[must_use]
    pub fn age(&self) -> DriverAge {
        self.date.map_or(DriverAge::Unknown, assess_date)
    }
}

/// Compare one date with [`FIRST_HIP_SDK_FOR_WINDOWS`].
///
/// A driver dated exactly on that day is not older than it.
#[must_use]
pub fn assess_date(date: CalendarDate) -> DriverAge {
    if date < FIRST_HIP_SDK_FOR_WINDOWS {
        DriverAge::Predates(date)
    } else {
        DriverAge::NotOlder(date)
    }
}

/// Why the cutoff is a date and not a documented minimum. Shared by the `rocm fix`
/// recipe and the `rocm diagnose` copy so they cannot drift.
pub const NO_MINIMUM_EXPLANATION: &str = "The comparison is with the date of the first HIP SDK for Windows, not with a documented minimum. AMD publishes no minimum driver version for the HIP SDK or the Lemonade Vulkan backend.";

/// States that the date cutoff is a floor and not a compatibility check. Shared
/// by the `rocm fix` recipe and the `rocm diagnose` copy so they cannot drift.
pub const FLOOR_NOTE: &str = "The date is only a floor: it flags drivers older than the first HIP SDK for Windows and cannot judge newer GPUs such as gfx120X, gfx1150 and gfx1151, which may need newer drivers.";

/// The caveat that must travel with every statement about driver age.
const NO_MINIMUM_NOTE: &str = "AMD publishes no minimum driver version for the HIP SDK or for the Lemonade Vulkan backend, so the minimum is not established and this is advice only";

/// The sentence `rocm diagnose` uses as evidence for an old driver.
///
/// `None` unless the driver predates the first HIP SDK for Windows.
#[must_use]
pub fn old_driver_evidence(info: &WindowsDriverInfo) -> Option<String> {
    let DriverAge::Predates(date) = info.age() else {
        return None;
    };
    Some(format!(
        "AMD display driver {} is dated {date}, before the first HIP SDK for Windows ({FIRST_HIP_SDK_FOR_WINDOWS}); {NO_MINIMUM_NOTE}",
        info.version
    ))
}

/// The `driver_age` line of `rocm examine`.
#[must_use]
pub fn age_summary(info: &WindowsDriverInfo) -> String {
    match info.age() {
        DriverAge::Predates(_) => format!(
            "older than the first HIP SDK for Windows ({FIRST_HIP_SDK_FOR_WINDOWS}); {NO_MINIMUM_NOTE}; see {AMD_SUPPORT_URL}, restart after installing, or run `rocm diagnose` ({DISPLAY_DRIVER_OLD_FIX_ID})"
        ),
        DriverAge::NotOlder(_) => format!(
            "not older than the first HIP SDK for Windows ({FIRST_HIP_SDK_FOR_WINDOWS}); this is not a compatibility check, because the minimum driver version is not established, and the date cannot judge newer GPUs such as gfx120X, gfx1150 and gfx1151, which may need newer drivers"
        ),
        DriverAge::Unknown => "unknown (the driver date could not be read)".to_owned(),
    }
}

/// The paragraph `rocm install driver` prints on Windows.
///
/// Two truths travel together here: what the driver is, and that this command
/// installs nothing. `info` is `None` when the probe found no AMD driver.
#[must_use]
pub fn install_driver_summary(info: Option<&WindowsDriverInfo>) -> String {
    let Some(info) = info else {
        return "Windows driver install is validate-only in rocm-cli; no driver is installed or changed. The AMD display driver version could not be read; run `rocm examine` to inspect it.".to_owned();
    };
    let date = info
        .date
        .map_or_else(|| "unknown".to_owned(), |date| date.to_string());
    let verdict = match info.age() {
        DriverAge::Predates(_) => format!(
            "outdated: yes, dated before the first HIP SDK for Windows ({FIRST_HIP_SDK_FOR_WINDOWS}); {NO_MINIMUM_NOTE}; fix id {DISPLAY_DRIVER_OLD_FIX_ID}, run `rocm diagnose` for the advice, then update from {AMD_SUPPORT_URL} and restart"
        ),
        DriverAge::NotOlder(_) => "outdated: no, the date is not older than the first HIP SDK for Windows; the minimum driver version is not established".to_owned(),
        DriverAge::Unknown => "outdated: unknown, the driver date could not be read".to_owned(),
    };
    format!(
        "Windows driver install is validate-only in rocm-cli; no driver is installed or changed. AMD display driver version {}, dated {date}, {verdict}.",
        info.version
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(version: &str, date: &str) -> WindowsDriverInfo {
        WindowsDriverInfo::from_probe(version, date).expect("a version was given")
    }

    #[test]
    fn dates_parse_in_the_one_shape_the_probe_writes() {
        let date = CalendarDate::parse("2021-08-04").expect("parses");
        assert_eq!(date.to_string(), "2021-08-04");
        assert_eq!(CalendarDate::parse(" 2021-08-04T10:11:12 "), Some(date));
        assert_eq!(CalendarDate::parse("2021-08-04 10:11:12"), Some(date));
    }

    #[test]
    fn unparsable_dates_are_none() {
        for bad in [
            "",
            "   ",
            "8/4/2021",
            "2021-8-4",
            "2021-13-01",
            "2021-00-10",
            "2021-01-32",
            "2021-08",
            "2021-08-04-01",
            "2021-08-04T",
            "abcd-ef-gh",
            "0001-01-01",
            "1601-01-01",
            // A sign is not a digit, and the day has to exist in that month.
            "2021-+1-+4",
            "+021-08-04",
            "2023-02-31",
            "2023-02-29",
            "2021-04-31",
            "2021-06-00",
        ] {
            assert_eq!(CalendarDate::parse(bad), None, "{bad:?}");
        }
        // A leap day exists in a leap year, including the century rule.
        assert!(CalendarDate::parse("2024-02-29").is_some());
        assert!(CalendarDate::parse("2100-02-29").is_none());
        assert!(CalendarDate::parse("2400-02-29").is_some());
    }

    #[test]
    fn a_2021_driver_predates_the_first_hip_sdk() {
        let i = info("30.0.13014.8", "2021-08-04");
        assert_eq!(
            i.age(),
            DriverAge::Predates(CalendarDate {
                year: 2021,
                month: 8,
                day: 4
            })
        );
        assert!(old_driver_evidence(&i).is_some());
    }

    #[test]
    fn a_driver_on_the_cutoff_day_is_not_older() {
        let on = info("32.0.1.1", "2023-07-27");
        assert!(matches!(on.age(), DriverAge::NotOlder(_)));
        assert_eq!(old_driver_evidence(&on), None);
        let day_before = info("32.0.1.1", "2023-07-26");
        assert!(matches!(day_before.age(), DriverAge::Predates(_)));
    }

    #[test]
    fn a_current_driver_is_never_flagged() {
        let i = info("32.0.23033.1002", "2026-05-02");
        assert!(matches!(i.age(), DriverAge::NotOlder(_)));
        assert_eq!(old_driver_evidence(&i), None);
        assert!(age_summary(&i).starts_with("not older"));
    }

    #[test]
    fn a_missing_or_unreadable_date_is_unknown_and_never_flagged() {
        for date in ["", "garbage", "0001-01-01"] {
            let i = info("30.0.13014.8", date);
            assert_eq!(i.date, None, "{date:?}");
            assert_eq!(i.age(), DriverAge::Unknown);
            assert_eq!(old_driver_evidence(&i), None);
            assert!(age_summary(&i).starts_with("unknown"));
        }
    }

    #[test]
    fn a_record_without_a_version_does_not_exist() {
        assert_eq!(WindowsDriverInfo::from_probe("", "2021-08-04"), None);
        assert_eq!(WindowsDriverInfo::from_probe("  ", "2021-08-04"), None);
    }

    #[test]
    fn every_statement_about_age_says_the_minimum_is_not_established() {
        let old = info("30.0.13014.8", "2021-08-04");
        let current = info("32.0.23033.1002", "2026-05-02");
        for text in [
            old_driver_evidence(&old).expect("old"),
            age_summary(&old),
            install_driver_summary(Some(&old)),
            age_summary(&current),
            install_driver_summary(Some(&current)),
        ] {
            assert!(
                text.contains("not established"),
                "a statement about driver age must not read as a compatibility verdict: {text}"
            );
        }
    }

    #[test]
    fn the_old_driver_text_never_claims_the_driver_is_broken() {
        let text = install_driver_summary(Some(&info("30.0.13014.8", "2021-08-04")));
        for word in ["broken", "incompatible", "unsupported", "will fail"] {
            assert!(!text.contains(word), "{word}: {text}");
        }
    }

    #[test]
    fn the_install_summary_reports_version_date_verdict_and_fix_id() {
        let old = install_driver_summary(Some(&info("30.0.13014.8", "2021-08-04")));
        assert!(old.contains("version 30.0.13014.8"));
        assert!(old.contains("dated 2021-08-04"));
        assert!(old.contains("outdated: yes"));
        assert!(old.contains(DISPLAY_DRIVER_OLD_FIX_ID));
        assert!(old.contains("no driver is installed or changed"));

        let current = install_driver_summary(Some(&info("32.0.23033.1002", "2026-05-02")));
        assert!(current.contains("outdated: no"));
        assert!(
            !current.contains(DISPLAY_DRIVER_OLD_FIX_ID),
            "a current driver must not point at the old-driver advice: {current}"
        );

        let unknown = install_driver_summary(Some(&info("30.0.13014.8", "")));
        assert!(unknown.contains("outdated: unknown"));
        assert!(!unknown.contains(DISPLAY_DRIVER_OLD_FIX_ID));

        let none = install_driver_summary(None);
        assert!(none.contains("could not be read"));
        assert!(none.contains("no driver is installed or changed"));
    }

    /// The Microsoft Basic Display Adapter keeps the AMD vendor id, reports an
    /// operating system `10.0.<build>` version and a 2006 placeholder date. That
    /// date is after the year-2000 guard, so only the version or name keeps it
    /// from reading as an old AMD driver. Inputs are recalled, not observed on
    /// Windows.
    #[test]
    fn the_microsoft_fallback_driver_is_not_an_amd_driver() {
        assert_eq!(
            WindowsDriverInfo::from_probe("10.0.26100.1", "2006-06-21"),
            None
        );
        assert_eq!(
            WindowsDriverInfo::from_probe("10.0.19041.3636", "2006-06-21"),
            None
        );
        assert!(is_microsoft_fallback_driver(
            "Microsoft Basic Display Adapter",
            "30.0.13014.8"
        ));
        assert!(is_microsoft_fallback_driver("AMD Radeon", "10.0.22621.1"));
        assert!(!is_microsoft_fallback_driver(
            "AMD Radeon(TM) Graphics",
            "30.0.13014.8"
        ));
        // A real AMD version that merely starts with the digits 10 is kept.
        assert!(WindowsDriverInfo::from_probe("10.1.2.3", "2021-08-04").is_some());
        assert!(WindowsDriverInfo::from_probe("100.0.1", "2021-08-04").is_some());
    }

    #[test]
    fn json_keeps_the_date_text_and_adds_the_verdict() {
        let old = info("30.0.13014.8", "2021-08-04");
        assert_eq!(
            serde_json::to_value(&old).unwrap(),
            serde_json::json!({
                "version": "30.0.13014.8",
                "date": "2021-08-04",
                "age": "predates_first_hip_sdk",
            })
        );
        let current = info("32.0.23033.1002", "2026-05-02");
        assert_eq!(serde_json::to_value(&current).unwrap()["age"], "not_older");
        let undated = serde_json::to_value(info("30.0.13014.8", "")).unwrap();
        assert_eq!(
            undated,
            serde_json::json!({"version": "30.0.13014.8", "age": "unknown"})
        );
    }

    #[test]
    fn json_without_the_verdict_still_reads_and_a_bad_date_reads_as_unknown() {
        // The shape written before `age` existed.
        let old: WindowsDriverInfo =
            serde_json::from_str(r#"{"version":"30.0.13014.8","date":"2021-08-04"}"#).unwrap();
        assert_eq!(old, info("30.0.13014.8", "2021-08-04"));
        // A written record reads back to itself; the stored `age` is ignored.
        let again: WindowsDriverInfo =
            serde_json::from_value(serde_json::to_value(&old).unwrap()).unwrap();
        assert_eq!(again, old);
        let lying: WindowsDriverInfo = serde_json::from_str(
            r#"{"version":"30.0.13014.8","date":"2021-08-04","age":"not_older"}"#,
        )
        .unwrap();
        assert!(matches!(lying.age(), DriverAge::Predates(_)));
        let bad: WindowsDriverInfo =
            serde_json::from_str(r#"{"version":"30.0.13014.8","date":"soon"}"#).unwrap();
        assert_eq!(bad.date, None);
        assert_eq!(bad.age(), DriverAge::Unknown);
    }

    #[test]
    fn the_fix_id_is_a_catalog_entry() {
        assert!(crate::fix::is_catalog_id(DISPLAY_DRIVER_OLD_FIX_ID));
    }
}
