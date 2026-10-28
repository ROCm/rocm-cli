// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The weekly trend of Doctor reports, and the weeks in which one problem rose
//! sharply.
//!
//! A count says how many reports describe a problem. It does not say whether
//! that number is steady or new, and the proposal this follows treats the two
//! differently: "treat the trend as the signal and the absolute count as soft".
//! So reports are also counted per week, and a week whose count for a known
//! problem is several times its recent average is named. A rise is not a claim
//! about anybody's machine; it says that something changed, and in which week.
//!
//! Nothing here reads a date. The caller hands over the week a report belongs
//! to, already reduced to a [`Week`], so the time of day, the time zone and even
//! the day of the week were dropped where the date was read. What reaches this
//! module cannot be finer than a week.
//!
//! Like [`crate::report_rollup`], this knows nothing about where a report came
//! from, and it sends and files nothing.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use serde_json::{Value, json};

use crate::report::{Report, UNRECOGNISED};

/// A calendar week, numbered from the week that began on Monday 1970-01-05.
///
/// A whole number, not a year and a week of that year, so the week before and
/// the week after are plain arithmetic. The ISO week of the year restarts at
/// New Year, and a baseline that has to reach back across it should not need to
/// know how many weeks the previous year had.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Week(i64);

impl Week {
    /// The week holding `day`, counted in days since 1970-01-01 in UTC.
    ///
    /// 1970-01-01 was a Thursday, so the first Monday is day 4.
    #[must_use]
    pub const fn from_unix_day(day: i64) -> Self {
        Self((day - 4).div_euclid(7))
    }

    /// The week `weeks` after this one, or before it when negative.
    #[must_use]
    pub const fn offset(self, weeks: i64) -> Self {
        Self(self.0.saturating_add(weeks))
    }

    /// The Monday that starts this week, in days since 1970-01-01.
    const fn monday(self) -> i64 {
        self.0.saturating_mul(7).saturating_add(4)
    }

    /// The Monday as `MM-DD`, for a column heading. The table's header states
    /// the first and last week in full, so the year of a column is never in doubt.
    #[must_use]
    pub fn short(self) -> String {
        let (_, month, day) = civil_from_days(self.monday());
        format!("{month:02}-{day:02}")
    }
}

/// The Monday that starts the week, as `YYYY-MM-DD`.
///
/// Named by its Monday rather than by an ISO week number: the date reads the
/// same in every year, and it does not change meaning at New Year.
impl fmt::Display for Week {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (year, month, day) = civil_from_days(self.monday());
        write!(f, "{year:04}-{month:02}-{day:02}")
    }
}

/// The calendar date `days` after 1970-01-01, as year, month, day.
///
/// Howard Hinnant's `civil_from_days`, which needs no date library. It is
/// exact for every day count whose shift by 719,468 does not overflow an `i64`,
/// far beyond any real date. `saturating_add` only keeps the arithmetic from
/// panicking past that, where the result is not a date. A test checks it against
/// `chrono` for every day from 2000 to 2100.
const fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days.saturating_add(719_468);
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// What one problem had in one week.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cell {
    pub count: usize,
    /// The ROCm release each of those reports came from, by the report's own
    /// `rocm` field. A rise is tied to a release through this, so no table of
    /// release dates is needed.
    pub rocm: BTreeMap<String, usize>,
}

/// Reports per problem per week.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trend {
    /// By catalog entry, then by week. A week with no reports is absent, and
    /// reads as zero.
    pub weeks: BTreeMap<String, BTreeMap<Week, Cell>>,
    /// Understood reports that could not be placed in a week. They are in the
    /// rollup's counts and in no week, and never count as a quiet one.
    pub undated: usize,
}

/// What counts as a sharp rise, and how much of the trend to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrendView {
    /// How many of the newest weeks the table shows.
    pub weeks: usize,
    pub rise: RiseRule,
}

impl Default for TrendView {
    fn default() -> Self {
        Self {
            weeks: 8,
            rise: RiseRule::default(),
        }
    }
}

/// When a week's count for a known problem is a rise.
///
/// A week rises when its count is at least `factor` times the average of the
/// `baseline_weeks` weeks before it, and is at least `min_reports`. The defaults
/// are the proposal's own example ("triple in the week after a ROCm release"),
/// with a floor so that two reports after one is not called a tripling. They are
/// flags rather than constants because there is no data yet to set them from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiseRule {
    pub factor: usize,
    pub baseline_weeks: usize,
    pub min_reports: usize,
}

impl Default for RiseRule {
    fn default() -> Self {
        Self {
            factor: 3,
            baseline_weeks: 4,
            min_reports: 5,
        }
    }
}

impl RiseRule {
    /// Whether the rule can mean a rise at all. A factor under 2 calls a week at
    /// the same rate as before a rise, and a baseline of no weeks has nothing to
    /// compare with. Neither is ever evaluated, and the output says so.
    #[must_use]
    pub const fn is_meaningful(&self) -> bool {
        self.factor >= 2 && self.baseline_weeks >= 1
    }

    /// How many weeks an export has to cover before any week can be judged: the
    /// baseline, after the export's first week, then the week itself.
    #[must_use]
    pub const fn weeks_needed(&self) -> usize {
        self.baseline_weeks.saturating_add(2)
    }
}

/// One week in which one problem rose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rise {
    pub entry: String,
    pub week: Week,
    pub count: usize,
    /// Reports in the weeks before it, and how many weeks that was.
    pub baseline_reports: usize,
    pub baseline_weeks: usize,
    /// The releases the week's reports came from, most reports first.
    pub rocm: Vec<(String, usize)>,
}

impl Trend {
    /// Place one understood report in a week, or count it as undated.
    pub(crate) fn add(&mut self, week: Option<Week>, report: &Report) {
        let Some(week) = week else {
            self.undated += 1;
            return;
        };
        let cell = self
            .weeks
            .entry(report.entry.clone())
            .or_default()
            .entry(week)
            .or_default();
        cell.count += 1;
        *cell.rocm.entry(report.rocm.clone()).or_default() += 1;
    }

    /// The first and last week holding any report at all.
    fn span(&self) -> Option<(Week, Week)> {
        let all = || self.weeks.values().flat_map(BTreeMap::keys).copied();
        Some((all().min()?, all().max()?))
    }

    /// How many weeks the reports cover, first to last, quiet ones included.
    #[must_use]
    pub fn span_weeks(&self) -> usize {
        self.span().map_or(0, |(first, last)| {
            usize::try_from(last.0 - first.0 + 1).unwrap_or(usize::MAX)
        })
    }

    /// The weeks in which a known problem rose, oldest first.
    ///
    /// Four rules keep this from naming a rise that is not one:
    /// - Only a matched catalog entry is considered. `unrecognised` mixes every
    ///   cause the catalog does not know, so a rise in it says nothing about one
    ///   problem. It is in the table, and never flagged.
    /// - A problem with no reports in the baseline is new, not risen. A new
    ///   group and a steady one need different responses.
    /// - The whole baseline has to lie inside the weeks the export covers.
    ///   Otherwise the first weeks of any export would read as rises against
    ///   weeks before collection began.
    /// - The export's first week is never part of a baseline. An export that
    ///   starts mid-week holds only the end of that week, which understates a
    ///   baseline and makes the next week look like a rise it is not.
    ///
    /// A rule that cannot mean a rise (see [`RiseRule::is_meaningful`]) finds
    /// none.
    #[must_use]
    pub fn rises(&self, rule: &RiseRule) -> Vec<Rise> {
        let Some((first, _)) = self.span() else {
            return Vec::new();
        };
        if !rule.is_meaningful() {
            return Vec::new();
        }
        let back = i64::try_from(rule.baseline_weeks).unwrap_or(i64::MAX);
        let mut found = Vec::new();
        for (entry, weeks) in &self.weeks {
            if entry == UNRECOGNISED {
                continue;
            }
            for (&week, cell) in weeks {
                if week.offset(-back) <= first || cell.count < rule.min_reports {
                    continue;
                }
                let baseline: usize = (1..=back)
                    .map(|ago| {
                        weeks
                            .get(&week.offset(-ago))
                            .map_or(0, |earlier| earlier.count)
                    })
                    .sum();
                // count / (baseline / weeks) >= factor, without a division.
                let rose = baseline > 0
                    && cell.count.saturating_mul(rule.baseline_weeks)
                        >= rule.factor.saturating_mul(baseline);
                if rose {
                    let mut rocm: Vec<(String, usize)> = cell
                        .rocm
                        .iter()
                        .map(|(release, count)| (release.clone(), *count))
                        .collect();
                    // Most reports first, ties by name. The tie-break is stated rather
                    // than left to the order `cell.rocm` iterates in: it is a
                    // `BTreeMap` and the sort is stable, so equal counts would come out
                    // by name without it. That makes it unobservable to a test, so
                    // `releases_with_equal_counts_are_listed_by_name` pins the result
                    // and not the clause.
                    rocm.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    found.push(Rise {
                        entry: entry.clone(),
                        week,
                        count: cell.count,
                        baseline_reports: baseline,
                        baseline_weeks: rule.baseline_weeks,
                        rocm,
                    });
                }
            }
        }
        found.sort_by(|a, b| a.week.cmp(&b.week).then_with(|| a.entry.cmp(&b.entry)));
        found
    }

    /// The newest `shown` weeks that end at the last week with a report.
    fn columns(&self, shown: usize) -> Vec<Week> {
        let Some((first, last)) = self.span() else {
            return Vec::new();
        };
        let available = usize::try_from(last.0 - first.0 + 1).unwrap_or(usize::MAX);
        let shown = shown.clamp(1, available);
        let back = i64::try_from(shown - 1).unwrap_or(i64::MAX);
        (0..shown)
            .map(|n| last.offset(i64::try_from(n).unwrap_or(0) - back))
            .collect()
    }

    /// Problems by their reports in all weeks, most first, ties by name.
    ///
    /// The tie-break is stated rather than left to the order the entries are
    /// stored in. `weeks` is a `BTreeMap` and the sort is stable, so equal totals
    /// would come out in name order without it. Removing it changes nothing a
    /// test can see, which is why none pins it separately: the order a person
    /// reads is pinned by `the_table_lists_the_busiest_problem_first_and_ties_by_name`.
    fn ranked(&self) -> Vec<(&String, &BTreeMap<Week, Cell>)> {
        let total = |weeks: &BTreeMap<Week, Cell>| weeks.values().map(|c| c.count).sum::<usize>();
        let mut ranked: Vec<_> = self.weeks.iter().collect();
        ranked.sort_by(|a, b| total(b.1).cmp(&total(a.1)).then_with(|| a.0.cmp(b.0)));
        ranked
    }

    /// The trend as a person reads it. Empty when there is nothing to say.
    #[must_use]
    pub fn render_text(&self, view: &TrendView) -> String {
        let mut out = String::new();
        let columns = self.columns(view.weeks);
        if columns.is_empty() {
            if self.undated > 0 {
                let _ = writeln!(
                    out,
                    "\nWeekly trend: none of the {} reports carried a usable date, so none is \
                     placed in a week.",
                    self.undated
                );
            }
            return out;
        }

        let _ = writeln!(
            out,
            "\nWeekly trend: reports per week, for the weeks starting {} to {} (columns are those \
             Mondays, MM-DD), newest last. The first and the latest week of the export may be \
             partial.",
            columns[0],
            columns[columns.len() - 1]
        );
        let mut header = format!("{:<44}", "entry");
        for week in &columns {
            let _ = write!(header, "{:>6}", week.short());
        }
        let _ = writeln!(out, "{header}");
        for (entry, weeks) in self.ranked() {
            // A problem with nothing in the weeks shown has no row. It is still
            // counted in the tree, and a rise is judged over every week.
            if !columns.iter().any(|week| weeks.contains_key(week)) {
                continue;
            }
            let mut row = format!("{entry:<44}");
            for week in &columns {
                match weeks.get(week).map_or(0, |cell| cell.count) {
                    0 => row.push_str("     ."),
                    count => {
                        let _ = write!(row, "{count:>6}");
                    }
                }
            }
            let _ = writeln!(out, "{row}");
        }
        if self.undated > 0 {
            let _ = writeln!(
                out,
                "Undated: {} reports carried no usable date and are in no week.",
                self.undated
            );
        }

        let rule = &view.rise;
        let _ = writeln!(
            out,
            "\nRises: a week's count for a known problem at least {}x the average of the {} weeks \
             before it, with at least {} reports.",
            rule.factor, rule.baseline_weeks, rule.min_reports
        );
        let rises = self.rises(rule);
        if rises.is_empty() {
            if !rule.is_meaningful() {
                out.push_str(
                    "  not evaluated: a rise needs a factor of at least 2 and a baseline of at \
                     least 1 week\n",
                );
            } else if self.span_weeks() < rule.weeks_needed() {
                let _ = writeln!(
                    out,
                    "  not evaluated: the reports cover {} weeks, and a rise needs {}: a baseline \
                     of {} weeks after the export's first week, then the week itself",
                    self.span_weeks(),
                    rule.weeks_needed(),
                    rule.baseline_weeks
                );
            } else {
                out.push_str("  none found\n");
            }
        }
        for rise in &rises {
            // The average to the nearest tenth, halves up, in integers.
            // `rises` returns nothing for a baseline of no weeks, so the divisor
            // is never zero.
            let tenths =
                (rise.baseline_reports * 10 + rise.baseline_weeks / 2) / rise.baseline_weeks;
            let releases: Vec<String> = rise
                .rocm
                .iter()
                .map(|(release, count)| format!("ROCm {release} ({count})"))
                .collect();
            let _ = writeln!(
                out,
                "  {}: week of {}, {} reports against {}.{} a week; {}",
                rise.entry,
                rise.week,
                rise.count,
                tenths / 10,
                tenths % 10,
                releases.join(", ")
            );
        }
        out
    }

    /// The trend as a script reads it.
    #[must_use]
    pub fn to_json(&self, view: &TrendView) -> Value {
        let columns = self.columns(view.weeks);
        let entries: Vec<Value> = self
            .ranked()
            .into_iter()
            .filter(|(_, weeks)| columns.iter().any(|week| weeks.contains_key(week)))
            .map(|(entry, weeks)| {
                json!({
                    "entry": entry,
                    "counts": columns
                        .iter()
                        .map(|week| weeks.get(week).map_or(0, |cell| cell.count))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let rises: Vec<Value> = self
            .rises(&view.rise)
            .into_iter()
            .map(|rise| {
                json!({
                    "entry": rise.entry,
                    "week": rise.week.to_string(),
                    "count": rise.count,
                    "baseline_reports": rise.baseline_reports,
                    "baseline_weeks": rise.baseline_weeks,
                    "rocm": rise
                        .rocm
                        .iter()
                        .map(|(release, count)| json!({"value": release, "count": count}))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({
            "weeks": columns.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "entries": entries,
            "undated": self.undated,
            "span_weeks": self.span_weeks(),
            "rises_evaluated": view.rise.is_meaningful()
                && self.span_weeks() >= view.rise.weeks_needed(),
            "rule": {
                "factor": view.rise.factor,
                "baseline_weeks": view.rise.baseline_weeks,
                "min_reports": view.rise.min_reports,
            },
            "rises": rises,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::REPORT_SCHEMA_VERSION;

    /// A report for `entry` on ROCm `rocm`. Every other field is the same, so a
    /// test varies only what it is about.
    fn report(entry: &str, rocm: &str) -> Report {
        Report {
            schema: REPORT_SCHEMA_VERSION,
            architecture: "gfx942".to_owned(),
            architecture_matrix: "matrix".to_owned(),
            entry: entry.to_owned(),
            os_family: "linux".to_owned(),
            os_major: "22".to_owned(),
            distro: "ubuntu".to_owned(),
            rocm: rocm.to_owned(),
            engine: "pytorch".to_owned(),
            engine_version: "2.5".to_owned(),
            cli_version: "0.1.0".to_owned(),
            fix_offered: false,
        }
    }

    /// Week number `n`, so a test reads in weeks and never in days: week 0 is
    /// the week that starts on Monday 1970-01-05, and each number after is the
    /// next week.
    const fn week(n: i64) -> Week {
        Week(n)
    }

    /// `n` reports of `entry` in `week`.
    fn put(trend: &mut Trend, entry: &str, week: Week, n: usize, rocm: &str) {
        for _ in 0..n {
            trend.add(Some(week), &report(entry, rocm));
        }
    }

    /// Reports placed in a week, of any entry.
    fn dated(trend: &Trend) -> usize {
        trend
            .weeks
            .values()
            .flat_map(BTreeMap::values)
            .map(|cell| cell.count)
            .sum()
    }

    /// The export's first week, which is never part of a baseline, so a fixture
    /// that wants week 10 onward judged starts one week before it.
    fn partial_first_week(trend: &mut Trend) {
        put(trend, "fix-1-arch", week(9), 1, "7.1");
    }

    #[test]
    fn a_day_falls_in_the_week_that_starts_on_the_monday_before_it() {
        // (day, the Monday that starts its week)
        for (day, monday) in [
            (0, "1969-12-29"),      // Thursday 1970-01-01
            (3, "1969-12-29"),      // Sunday: the last day of that week
            (4, "1970-01-05"),      // Monday: the first day of the next
            (20_752, "2026-10-26"), // Monday 2026-10-26
            (20_755, "2026-10-26"), // Thursday 2026-10-29
            (20_751, "2026-10-19"), // Sunday 2026-10-25
        ] {
            assert_eq!(Week::from_unix_day(day).to_string(), monday, "day {day}");
        }
    }

    /// Weeks are a plain count, so a baseline that reaches back over New Year, or
    /// over a leap day, needs no calendar knowledge.
    #[test]
    fn adjacent_weeks_differ_by_one_across_new_year_and_a_leap_day() {
        let thursday_new_year = Week::from_unix_day(20_454); // Thursday 2026-01-01
        let sunday_before = Week::from_unix_day(20_450); // Sunday 2025-12-28
        assert_eq!(thursday_new_year.to_string(), "2025-12-29");
        assert_eq!(sunday_before.to_string(), "2025-12-22");
        assert_eq!(sunday_before.offset(1), thursday_new_year);

        let leap = Week::from_unix_day(19_782); // Thursday 2024-02-29
        assert_eq!(leap.to_string(), "2024-02-26");
        assert_eq!(leap.offset(1).to_string(), "2024-03-04");
        assert_eq!(leap.offset(-1).to_string(), "2024-02-19");
    }

    #[test]
    fn a_week_names_itself_by_its_monday_in_short_form_too() {
        assert_eq!(Week::from_unix_day(20_755).short(), "10-26");
    }

    #[test]
    fn reports_are_counted_per_entry_per_week_with_their_releases() {
        let mut trend = Trend::default();
        put(&mut trend, "fix-6-path", week(10), 3, "7.1");
        put(&mut trend, "fix-6-path", week(10), 1, "7.0");
        put(&mut trend, "fix-6-path", week(11), 2, "7.1");
        trend.add(None, &report("fix-6-path", "7.1"));

        let cell = &trend.weeks["fix-6-path"][&week(10)];
        assert_eq!(cell.count, 4);
        assert_eq!(cell.rocm["7.1"], 3);
        assert_eq!(cell.rocm["7.0"], 1);
        assert_eq!(dated(&trend), 6);
        assert_eq!(trend.undated, 1, "an undated report is in no week");
    }

    /// A steady problem, then a week at three times its average. Week 9 is the
    /// export's first, partial week; weeks 10 to 13 hold 4 reports each; week 14
    /// holds `spike`.
    fn steady_then(spike: usize) -> Trend {
        let mut trend = Trend::default();
        partial_first_week(&mut trend);
        for n in 10..=13 {
            put(&mut trend, "fix-6-path", week(n), 4, "7.1");
        }
        put(&mut trend, "fix-6-path", week(14), spike, "7.2");
        trend
    }

    #[test]
    fn a_week_at_the_threshold_is_a_rise_and_one_below_it_is_not() {
        let rule = RiseRule::default(); // 3x the average of the 4 weeks before, at least 5
        // The average is 4, so 12 is exactly 3x.
        assert_eq!(steady_then(12).rises(&rule).len(), 1);
        assert!(
            steady_then(11).rises(&rule).is_empty(),
            "11 is under 3x of 4 a week"
        );
    }

    #[test]
    fn a_rise_names_the_entry_the_week_the_rate_and_the_releases() {
        let mut trend = steady_then(0);
        put(&mut trend, "fix-6-path", week(14), 9, "7.2");
        put(&mut trend, "fix-6-path", week(14), 4, "7.1");

        let rises = trend.rises(&RiseRule::default());
        assert_eq!(rises.len(), 1);
        let rise = &rises[0];
        assert_eq!(rise.entry, "fix-6-path");
        assert_eq!(rise.week, week(14));
        assert_eq!(rise.count, 13);
        assert_eq!((rise.baseline_reports, rise.baseline_weeks), (16, 4));
        assert_eq!(
            rise.rocm,
            vec![("7.2".to_owned(), 9), ("7.1".to_owned(), 4)],
            "most reports first"
        );

        let text = trend.render_text(&TrendView::default());
        assert!(text.contains("fix-6-path: week of "), "{text}");
        assert!(text.contains("13 reports against 4.0 a week"), "{text}");
        assert!(text.contains("ROCm 7.2 (9), ROCm 7.1 (4)"), "{text}");
    }

    /// An average that is not a whole number reads to the nearest tenth: 17
    /// reports over 4 weeks is 4.25 a week, which prints as 4.3 and not as 4.2.
    #[test]
    fn the_average_in_a_rise_is_rounded_to_the_nearest_tenth_not_truncated() {
        let mut trend = Trend::default();
        partial_first_week(&mut trend);
        for (n, reports) in [(10, 5), (11, 4), (12, 4), (13, 4)] {
            put(&mut trend, "fix-6-path", week(n), reports, "7.1");
        }
        put(&mut trend, "fix-6-path", week(14), 13, "7.1");

        // 13 against 4.25 a week is a rise: 13 * 4 = 52, and 3 * 17 = 51.
        assert_eq!(trend.rises(&RiseRule::default()).len(), 1);
        let text = trend.render_text(&TrendView::default());
        assert!(text.contains("13 reports against 4.3 a week"), "{text}");
    }

    #[test]
    fn each_part_of_the_rule_can_be_changed() {
        let trend = steady_then(12);
        let default = RiseRule::default;

        let rule = RiseRule {
            factor: 4,
            ..default()
        };
        assert!(trend.rises(&rule).is_empty(), "12 is not 4x of 4 a week");

        let rule = RiseRule {
            min_reports: 13,
            ..default()
        };
        assert!(trend.rises(&rule).is_empty(), "12 is under a floor of 13");

        // A count exactly at the floor is enough: the floor is "at least".
        let rule = RiseRule {
            min_reports: 12,
            ..default()
        };
        assert_eq!(trend.rises(&rule).len(), 1, "12 meets a floor of 12");

        // Two weeks of baseline: weeks 12 and 13, still 4 a week.
        let rule = RiseRule {
            baseline_weeks: 2,
            ..default()
        };
        assert_eq!(trend.rises(&rule).len(), 1);

        let rule = RiseRule {
            baseline_weeks: 0,
            ..default()
        };
        assert!(trend.rises(&rule).is_empty(), "no baseline, no rise");
    }

    #[test]
    fn a_problem_with_nothing_in_its_baseline_is_new_not_risen() {
        let mut trend = Trend::default();
        // Another problem makes the export long enough to hold a baseline.
        for n in 9..=14 {
            put(&mut trend, "fix-1-arch", week(n), 1, "7.1");
        }
        put(&mut trend, "fix-6-path", week(14), 30, "7.1");
        assert!(trend.rises(&RiseRule::default()).is_empty());
    }

    /// `unrecognised` mixes every cause the catalog does not know.
    #[test]
    fn the_unrecognised_entry_is_shown_and_never_flagged() {
        let mut trend = Trend::default();
        for n in 9..=13 {
            put(&mut trend, UNRECOGNISED, week(n), 4, "7.1");
        }
        put(&mut trend, UNRECOGNISED, week(14), 40, "7.1");

        assert!(trend.rises(&RiseRule::default()).is_empty());
        assert!(
            trend
                .render_text(&TrendView::default())
                .contains(UNRECOGNISED)
        );
    }

    /// Without this, the first weeks of any export would look like rises against
    /// weeks before anyone was collecting.
    #[test]
    fn a_baseline_that_starts_before_the_export_does_is_not_used() {
        let mut trend = Trend::default();
        put(&mut trend, "fix-6-path", week(10), 1, "7.1");
        put(&mut trend, "fix-6-path", week(11), 20, "7.1");
        // Week 11's baseline would be weeks 7 to 10, and the export begins at 10.
        assert!(trend.rises(&RiseRule::default()).is_empty());

        // Four weeks later the baseline is wholly inside the export.
        put(&mut trend, "fix-6-path", week(12), 1, "7.1");
        put(&mut trend, "fix-6-path", week(13), 1, "7.1");
        put(&mut trend, "fix-6-path", week(14), 1, "7.1");
        put(&mut trend, "fix-6-path", week(15), 60, "7.1");
        let rises = trend.rises(&RiseRule::default());
        assert_eq!(rises.len(), 1);
        assert_eq!(rises[0].week, week(15));
    }

    /// A quiet week inside the baseline is zero, not missing.
    #[test]
    fn a_week_with_no_reports_counts_as_zero_in_the_baseline() {
        let mut trend = Trend::default();
        partial_first_week(&mut trend);
        put(&mut trend, "fix-6-path", week(10), 8, "7.1");
        // Weeks 11 to 13 are silent. Baseline for week 14 is 8 over 4 weeks.
        put(&mut trend, "fix-6-path", week(14), 5, "7.1");
        // 5 reports against 2 a week is 2.5x, under 3x.
        assert!(trend.rises(&RiseRule::default()).is_empty());
        put(&mut trend, "fix-6-path", week(14), 1, "7.1");
        // 6 against 2 a week is exactly 3x.
        assert_eq!(trend.rises(&RiseRule::default()).len(), 1);
    }

    #[test]
    fn the_table_shows_the_newest_weeks_and_a_dot_for_a_quiet_one() {
        let mut trend = Trend::default();
        put(&mut trend, "fix-6-path", week(10), 4, "7.1");
        put(&mut trend, "fix-6-path", week(12), 2, "7.1");
        // Reports only from before the weeks shown: counted, and no row.
        put(&mut trend, "fix-1-arch", week(10), 3, "7.1");
        trend.add(None, &report("fix-6-path", "7.1"));

        let view = TrendView {
            weeks: 2,
            rise: RiseRule::default(),
        };
        let text = trend.render_text(&view);
        assert!(
            !text.lines().any(|l| l.starts_with("fix-1-arch")),
            "an entry with nothing in the weeks shown has no row:\n{text}"
        );
        let json = trend.to_json(&view);
        assert_eq!(json["entries"].as_array().unwrap().len(), 1, "{json}");
        let row = text
            .lines()
            .find(|l| l.starts_with("fix-6-path"))
            .expect("a row");
        assert!(
            row.ends_with("     .     2"),
            "newest two weeks, 11 quiet: {row:?}"
        );
        assert!(
            text.contains("Undated: 1 reports carried no usable date"),
            "{text}"
        );
        assert!(text.contains("may be partial"), "{text}");
        // Both ends of the table are named in full, so a table that crosses New
        // Year still says which year each column is in.
        assert!(
            text.contains("for the weeks starting 1970-03-23 to 1970-03-30"),
            "{text}"
        );
    }

    #[test]
    fn a_trend_with_no_dated_report_says_so_instead_of_staying_silent() {
        let mut trend = Trend::default();
        trend.add(None, &report("fix-6-path", "7.1"));
        trend.add(None, &report("fix-6-path", "7.1"));
        let text = trend.render_text(&TrendView::default());
        assert!(
            text.contains("none of the 2 reports carried a usable date"),
            "{text}"
        );
        assert!(
            Trend::default()
                .render_text(&TrendView::default())
                .is_empty()
        );
    }

    #[test]
    fn the_json_carries_the_weeks_the_counts_and_the_rises() {
        let mut trend = steady_then(12);
        // Two reports with no date, which no week holds.
        trend.add(None, &report("fix-6-path", "7.1"));
        trend.add(None, &report("fix-6-path", "7.1"));
        let value = trend.to_json(&TrendView::default());

        assert_eq!(value["weeks"].as_array().unwrap().len(), 6);
        assert_eq!(value["entries"][0]["entry"], "fix-6-path");
        assert_eq!(value["entries"][0]["counts"], json!([0, 4, 4, 4, 4, 12]));
        assert_eq!(value["span_weeks"], 6);
        assert_eq!(value["rises_evaluated"], true);
        assert_eq!(
            value["rule"],
            json!({"factor": 3, "baseline_weeks": 4, "min_reports": 5})
        );
        assert_eq!(value["rises"][0]["count"], 12);
        assert_eq!(
            value["rises"][0]["rocm"][0],
            json!({"value": "7.2", "count": 12})
        );
        assert_eq!(value["undated"], 2, "{value}");
    }

    /// Cross-check of the hand-written civil date against `chrono`, for every day
    /// from 2000 to 2100. `chrono` is a dev-dependency of this test only; the
    /// module itself needs no date library.
    #[test]
    fn the_civil_date_agrees_with_chrono_for_every_day_from_2000_to_2100() {
        use chrono::{Datelike, NaiveDate};
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let mut day = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
        while day.year() <= 2100 {
            let days = (day - epoch).num_days();
            let (year, month, date) = civil_from_days(days);
            assert_eq!(
                (year, month, date),
                (
                    i64::from(day.year()),
                    i64::from(day.month()),
                    i64::from(day.day())
                ),
                "{day}"
            );
            // And the week agrees with chrono's: its Monday is on or before the day,
            // fewer than seven days earlier, and is a Monday.
            let monday = epoch + chrono::Duration::days(Week::from_unix_day(days).monday());
            assert_eq!(monday.weekday(), chrono::Weekday::Mon, "{day}");
            let behind = (day - monday).num_days();
            assert!(
                (0..7).contains(&behind),
                "{day} is {behind} days after {monday}"
            );
            day = day.succ_opt().unwrap();
        }
    }

    /// An export that starts mid-week holds only the end of that week. If that
    /// week were part of a baseline it would understate it, and the next week
    /// would look like a rise it is not.
    #[test]
    fn the_exports_first_week_is_never_part_of_a_baseline() {
        // Weeks 11 to 13 hold 6 reports each, and week 14 holds 15.
        let build = |first_week_reports: usize| {
            let mut trend = Trend::default();
            put(
                &mut trend,
                "fix-6-path",
                week(10),
                first_week_reports,
                "7.1",
            );
            for n in 11..=13 {
                put(&mut trend, "fix-6-path", week(n), 6, "7.1");
            }
            put(&mut trend, "fix-6-path", week(14), 15, "7.1");
            trend
        };
        let rule = RiseRule::default();
        // The first week held only 1 report because the export began late in it.
        // Counted in the baseline it would read 15 against 19, a rise.
        assert!(build(1).rises(&rule).is_empty());
        assert!(build(6).rises(&rule).is_empty());

        // Once an earlier week exists, week 10 is an ordinary week and week 14's
        // baseline is wholly after the export's first.
        let mut trend = build(1);
        put(&mut trend, "fix-6-path", week(9), 6, "7.1");
        assert_eq!(trend.rises(&rule).len(), 1);
    }

    /// An export too short to judge any week says so. It does not print the same
    /// "none found" as a rule that was evaluated and found nothing.
    #[test]
    fn an_export_too_short_to_judge_says_so_and_does_not_say_none_found() {
        let rule = RiseRule::default();
        assert_eq!(
            rule.weeks_needed(),
            6,
            "4 baseline weeks, the first week, the week"
        );
        let covering = |weeks: i64| {
            let mut trend = Trend::default();
            for n in 0..weeks {
                put(&mut trend, "fix-6-path", week(10 + n), 4, "7.1");
            }
            trend
        };
        let view = TrendView::default();

        let five = covering(5);
        let text = five.render_text(&view);
        assert!(
            text.contains("not evaluated: the reports cover 5 weeks, and a rise needs 6"),
            "{text}"
        );
        assert!(!text.contains("none found"), "{text}");
        assert_eq!(five.to_json(&view)["rises_evaluated"], false);

        // One week more and it is evaluated, and finds nothing.
        let six = covering(6);
        let text = six.render_text(&view);
        assert!(text.contains("none found"), "{text}");
        assert!(!text.contains("not evaluated"), "{text}");
        assert_eq!(six.to_json(&view)["rises_evaluated"], true);

        // A baseline longer than the export, however long, is not evaluated.
        let absurd = TrendView {
            rise: RiseRule {
                baseline_weeks: usize::MAX,
                ..RiseRule::default()
            },
            ..TrendView::default()
        };
        assert!(six.render_text(&absurd).contains("not evaluated"));
    }

    /// A rule that could not mean a rise finds none, and says it was not
    /// evaluated. A factor of 0 with a floor of 0 would otherwise call any week
    /// with a baseline a rise.
    #[test]
    fn a_rule_that_cannot_mean_a_rise_is_not_evaluated() {
        let mut trend = steady_then(5);
        let view = |rise: RiseRule| TrendView { weeks: 8, rise };
        for rule in [
            RiseRule {
                factor: 0,
                min_reports: 0,
                ..RiseRule::default()
            },
            RiseRule {
                factor: 1,
                min_reports: 0,
                ..RiseRule::default()
            },
            RiseRule {
                baseline_weeks: 0,
                ..RiseRule::default()
            },
        ] {
            assert!(!rule.is_meaningful(), "{rule:?}");
            assert!(trend.rises(&rule).is_empty(), "{rule:?}");
            let text = trend.render_text(&view(rule.clone()));
            // For a script as well: the reports cover enough weeks, but the rule
            // cannot mean a rise, so it was not evaluated.
            assert_eq!(
                trend.to_json(&view(rule))["rises_evaluated"],
                false,
                "{text}"
            );
            assert!(
                text.contains("not evaluated: a rise needs a factor of at least 2"),
                "{text}"
            );
        }
        // The same data under a meaningful rule, to show the guard is the rule's.
        assert!(
            RiseRule {
                factor: 2,
                baseline_weeks: 1,
                ..RiseRule::default()
            }
            .is_meaningful()
        );
        put(&mut trend, "fix-6-path", week(14), 7, "7.2");
        assert_eq!(trend.rises(&RiseRule::default()).len(), 1);
    }

    /// Rises come oldest week first, then by entry, and each carries the count,
    /// the baseline and how many weeks it was over. The text and the JSON agree.
    #[test]
    fn rises_are_listed_oldest_week_first_with_their_counts_and_baselines() {
        let mut trend = Trend::default();
        partial_first_week(&mut trend);
        // Three problems at 4 a week over weeks 10 to 13.
        for entry in ["fix-1-arch", "fix-2-unset-override", "fix-6-path"] {
            for n in 10..=13 {
                put(&mut trend, entry, week(n), 4, "7.1");
            }
        }
        // Two rise in week 14, tied on the week. The third holds, then rises in 15.
        put(&mut trend, "fix-6-path", week(14), 14, "7.2");
        put(&mut trend, "fix-2-unset-override", week(14), 14, "7.1");
        put(&mut trend, "fix-1-arch", week(14), 4, "7.1");
        put(&mut trend, "fix-1-arch", week(15), 13, "7.1");

        // Three baseline weeks, so baseline_weeks (3) differs from every count.
        let rule = RiseRule {
            baseline_weeks: 3,
            ..RiseRule::default()
        };
        let rises = trend.rises(&rule);
        let order: Vec<(Week, &str)> = rises.iter().map(|r| (r.week, r.entry.as_str())).collect();
        assert_eq!(
            order,
            [
                (week(14), "fix-2-unset-override"),
                (week(14), "fix-6-path"),
                (week(15), "fix-1-arch"),
            ],
            "oldest week first, then by entry; not the order the entries are stored in"
        );

        let view = TrendView {
            weeks: 8,
            rise: rule,
        };
        let json = trend.to_json(&view);
        for (i, (week, entry, count, baseline)) in [
            (week(14), "fix-2-unset-override", 14, 12),
            (week(14), "fix-6-path", 14, 12),
            (week(15), "fix-1-arch", 13, 12),
        ]
        .into_iter()
        .enumerate()
        {
            let rise = &json["rises"][i];
            assert_eq!(rise["entry"], entry);
            assert_eq!(rise["week"], week.to_string());
            assert_eq!(rise["count"], count, "{rise}");
            assert_eq!(rise["baseline_reports"], baseline, "{rise}");
            assert_eq!(rise["baseline_weeks"], 3, "{rise}");
        }
        assert_eq!(json["rises"].as_array().unwrap().len(), 3);

        // And the text lists them in the same order.
        let text = trend.render_text(&view);
        let at = |needle: &str| {
            text.find(needle)
                .unwrap_or_else(|| panic!("{needle}\n{text}"))
        };
        assert!(
            at("fix-2-unset-override: week of") < at("fix-6-path: week of")
                && at("fix-6-path: week of") < at("fix-1-arch: week of"),
            "{text}"
        );
    }

    /// Releases with equal counts are listed by name, so the output does not
    /// depend on the order they were seen in.
    #[test]
    fn releases_with_equal_counts_are_listed_by_name() {
        let mut trend = steady_then(0);
        // Seen in the opposite order to the one they must be listed in.
        put(&mut trend, "fix-6-path", week(14), 6, "7.2");
        put(&mut trend, "fix-6-path", week(14), 6, "7.1");
        put(&mut trend, "fix-6-path", week(14), 1, "7.0");

        let rises = trend.rises(&RiseRule::default());
        assert_eq!(rises.len(), 1);
        assert_eq!(
            rises[0].rocm,
            [
                ("7.1".to_owned(), 6),
                ("7.2".to_owned(), 6),
                ("7.0".to_owned(), 1),
            ],
            "most reports first, and the tied two by name"
        );
    }

    /// The baseline's length is what the count is averaged over.
    #[test]
    fn the_baseline_length_is_what_the_average_is_taken_over() {
        let mut trend = Trend::default();
        partial_first_week(&mut trend);
        for n in 11..=13 {
            put(&mut trend, "fix-6-path", week(n), 4, "7.1");
        }
        let with = |baseline_weeks: usize| RiseRule {
            baseline_weeks,
            ..RiseRule::default()
        };
        // Three weeks at 4 is 12, so 4 a week. 11 is under 3x of that.
        put(&mut trend, "fix-6-path", week(14), 11, "7.1");
        assert!(trend.rises(&with(3)).is_empty());
        // 12 is exactly 3x.
        put(&mut trend, "fix-6-path", week(14), 1, "7.1");
        assert_eq!(trend.rises(&with(3)).len(), 1);
        // Over two weeks the baseline is 8, still 4 a week: the same verdict.
        assert_eq!(trend.rises(&with(2)).len(), 1);
        // Over one week it is 4 a week too, and 12 is exactly 3x.
        assert_eq!(trend.rises(&with(1)).len(), 1);
    }

    /// The table lists the problem with the most reports first, and ties by name.
    #[test]
    fn the_table_lists_the_busiest_problem_first_and_ties_by_name() {
        let mut trend = Trend::default();
        put(&mut trend, "fix-6-path", week(10), 10, "7.1");
        put(&mut trend, "fix-1-arch", week(10), 10, "7.1");
        put(&mut trend, "fix-2-unset-override", week(10), 30, "7.1");
        let text = trend.render_text(&TrendView::default());
        let rows: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("fix-"))
            .map(|l| l.split_whitespace().next().unwrap())
            .collect();
        assert_eq!(
            rows,
            ["fix-2-unset-override", "fix-1-arch", "fix-6-path"],
            "{text}"
        );
    }
}
