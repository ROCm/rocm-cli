// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Count Doctor reports by what they describe, at several levels at once.
//!
//! One level of grouping fails in both directions. A narrow key gives many
//! groups that never reach a useful size, and a wide one gives a single group
//! that mixes causes. So every level of [`Report::group_key`] is counted, and
//! the result reads as a tree: a person sees which field separates the reports
//! by watching where the counts split, instead of guessing the axis first.
//!
//! Nothing here knows where a report came from. The input is what
//! [`read_report`](crate::report::read_report) made of each one, so a mailbox
//! export and an issue tracker feed the same counter, and only the code that
//! fetches report bodies changes when the transport does. Nothing here sends
//! or files anything either.
//!
//! Every count is of reports, not of people. A report carries no machine
//! identifier, deliberately, so one person who hits the same failure five times
//! counts as five. The rendered rollup says so, because the number is otherwise
//! easy to read as people affected.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Value, json};

use crate::report::{GROUP_AXES, ReadOutcome, Report, UNRECOGNISED};

/// Reports sharing one value at one level, and how they split below it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Group {
    pub count: usize,
    pub children: BTreeMap<String, Self>,
}

impl Group {
    fn add(&mut self, key: &[String]) {
        self.count += 1;
        if let Some((head, rest)) = key.split_first() {
            self.children.entry(head.clone()).or_default().add(rest);
        }
    }

    /// Children largest first, ties by value, so the order is stable across
    /// runs and the biggest problem is read first.
    fn ranked(&self) -> Vec<(&String, &Self)> {
        let mut ranked: Vec<_> = self.children.iter().collect();
        ranked.sort_by(|a, b| b.1.count.cmp(&a.1.count).then_with(|| a.0.cmp(b.0)));
        ranked
    }
}

/// Every report a counter was given, sorted into what could and could not be
/// counted.
///
/// The three outcomes are kept apart because they answer different questions.
/// A refusal is the disclosure guard working, and an unread report is a reader
/// that is out of date. Folding either into the groups, or dropping it, would
/// make the counts look complete while being wrong.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rollup {
    /// Reports that were read, grouped level by level along [`GROUP_AXES`].
    pub understood: Group,
    /// Refusals by the marker that refused.
    pub refused: BTreeMap<String, usize>,
    /// Reports this reader could not read, by the schema version they
    /// declared. `0` means no schema could be found at all.
    pub unread: BTreeMap<u32, usize>,
}

impl Rollup {
    /// Count one report.
    pub fn add(&mut self, outcome: &ReadOutcome) {
        match outcome {
            ReadOutcome::Understood(report) => self.add_report(report),
            ReadOutcome::Refused { reason } => {
                *self.refused.entry(reason.clone()).or_default() += 1;
            }
            ReadOutcome::Unread { schema_seen } => {
                *self.unread.entry(*schema_seen).or_default() += 1;
            }
        }
    }

    fn add_report(&mut self, report: &Report) {
        self.understood.add(&report.group_key());
    }

    /// Count every report.
    #[must_use]
    pub fn of<'a>(outcomes: impl IntoIterator<Item = &'a ReadOutcome>) -> Self {
        let mut rollup = Self::default();
        for outcome in outcomes {
            rollup.add(outcome);
        }
        rollup
    }

    /// Every report given, whatever became of it.
    #[must_use]
    pub fn total(&self) -> usize {
        self.understood.count
            + self.refused.values().sum::<usize>()
            + self.unread.values().sum::<usize>()
    }

    /// Understood reports whose cause the catalog recognised.
    #[must_use]
    pub fn recognised(&self) -> usize {
        self.understood.count - self.unrecognised()
    }

    /// Understood reports whose cause the catalog did not recognise.
    ///
    /// Counted apart because they cannot separate beyond the platform: a report
    /// carries no error text to tell two unrecognised causes apart. A set
    /// dominated by them looks ungroupable for reasons that have nothing to do
    /// with the field set.
    #[must_use]
    pub fn unrecognised(&self) -> usize {
        self.understood
            .children
            .get(UNRECOGNISED)
            .map_or(0, |group| group.count)
    }

    /// The rollup as a person reads it.
    #[must_use]
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        let refused: usize = self.refused.values().sum();
        let unread: usize = self.unread.values().sum();
        let _ = writeln!(
            out,
            "Doctor reports: {} (understood {}, refused {refused}, unread {unread})",
            self.total(),
            self.understood.count,
        );
        let _ = writeln!(
            out,
            "Recognised cause: {} in {} entries. Unrecognised: {}.",
            self.recognised(),
            self.understood.children.len() - usize::from(self.unrecognised() > 0),
            self.unrecognised(),
        );
        out.push_str(
            "Counts are reports, not people: one person may have sent the same report more than \
             once.\n",
        );

        if self.understood.count > 0 {
            let _ = writeln!(out, "\nBy {}:", GROUP_AXES.join(" > "));
            render_level(&mut out, &self.understood, 0);
        }
        if refused > 0 {
            out.push_str("\nRefused, by the rule that refused:\n");
            for (reason, count) in &self.refused {
                let _ = writeln!(out, "{reason:<44}{count:>6}");
            }
        }
        if unread > 0 {
            out.push_str(
                "\nUnread, by declared schema (counted, never as zero; no group includes them):\n",
            );
            for (schema, count) in &self.unread {
                let label = if *schema == 0 {
                    "no readable schema".to_owned()
                } else {
                    format!("schema {schema}")
                };
                let _ = writeln!(out, "{label:<44}{count:>6}");
            }
        }
        out
    }

    /// The rollup as a script reads it. Groups are ranked as in the text.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "total": self.total(),
            "understood": self.understood.count,
            "recognised": self.recognised(),
            "unrecognised": self.unrecognised(),
            "refused": self.refused,
            "unread": self.unread.iter().map(|(schema, count)| (schema.to_string(), *count)).collect::<BTreeMap<_, _>>(),
            "axes": GROUP_AXES,
            "groups": groups_json(&self.understood),
        })
    }
}

fn render_level(out: &mut String, group: &Group, depth: usize) {
    for (value, child) in group.ranked() {
        let label = if depth == 0 {
            value.clone()
        } else {
            format!("{}+ {} {value}", "  ".repeat(depth - 1), GROUP_AXES[depth])
        };
        let _ = writeln!(out, "{label:<44}{:>6}", child.count);
        render_level(out, child, depth + 1);
    }
}

fn groups_json(group: &Group) -> Vec<Value> {
    group
        .ranked()
        .into_iter()
        .map(|(value, child)| {
            json!({
                "value": value,
                "count": child.count,
                "groups": groups_json(child),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{REPORT_SCHEMA_VERSION, read_report};

    fn report(entry: &str, arch: &str, rocm: &str, distro: (&str, &str)) -> ReadOutcome {
        ReadOutcome::Understood(Box::new(Report {
            schema: REPORT_SCHEMA_VERSION,
            architecture: arch.to_owned(),
            architecture_matrix: "matrix".to_owned(),
            entry: entry.to_owned(),
            os_family: "linux".to_owned(),
            os_major: distro.1.to_owned(),
            distro: distro.0.to_owned(),
            rocm: rocm.to_owned(),
            engine: "pytorch".to_owned(),
            engine_version: "2.5".to_owned(),
            cli_version: "0.1.0".to_owned(),
            fix_offered: false,
        }))
    }

    fn assert_levels_sum(group: &Group) {
        if !group.children.is_empty() {
            assert_eq!(
                group.children.values().map(|c| c.count).sum::<usize>(),
                group.count
            );
            group.children.values().for_each(assert_levels_sum);
        }
    }

    /// The proposal's worked example: the architecture and the ROCm release
    /// carry the signal, and the distribution does not. The tree has to show
    /// that without anyone choosing the axis first.
    #[test]
    fn every_level_is_counted_so_the_separating_field_shows() {
        let mut outcomes = Vec::new();
        outcomes.extend((0..44).map(|_| report(UNRECOGNISED, "gfx1201", "7.1", ("ubuntu", "24"))));
        outcomes.extend((0..4).map(|_| report(UNRECOGNISED, "gfx1201", "7.1", ("rhel", "9"))));
        outcomes.extend((0..4).map(|_| report(UNRECOGNISED, "gfx1201", "7.0", ("ubuntu", "24"))));
        outcomes.extend((0..6).map(|_| report(UNRECOGNISED, "gfx90a", "7.1", ("ubuntu", "24"))));
        let rollup = Rollup::of(&outcomes);

        let unrecognised = &rollup.understood.children[UNRECOGNISED];
        assert_eq!(unrecognised.count, 58);
        let gfx1201 = &unrecognised.children["gfx1201"];
        assert_eq!(gfx1201.count, 52);
        assert_eq!(gfx1201.children["7.1"].count, 48);
        assert_eq!(gfx1201.children["7.1"].children["ubuntu-24"].count, 44);
        assert_eq!(gfx1201.children["7.1"].children["rhel-9"].count, 4);
        assert_eq!(unrecognised.children["gfx90a"].count, 6);

        // Every level sums to its parent: no report is lost between levels.
        assert_levels_sum(&rollup.understood);

        let text = rollup.render_text();
        let order: Vec<usize> = ["gfx1201", "gfx90a"]
            .iter()
            .map(|a| text.find(a).unwrap())
            .collect();
        assert!(
            order[0] < order[1],
            "the largest group is read first:\n{text}"
        );
    }

    /// A report this reader cannot read is reported as unread, never counted
    /// as zero, and kept out of every group.
    #[test]
    fn an_unread_report_is_counted_and_grouped_nowhere() {
        let newer = format!(
            r#"{{"schema": {}, "entry": "fix-6-path"}}"#,
            REPORT_SCHEMA_VERSION + 1
        );
        let outcomes = [
            read_report(&newer),
            read_report("not json at all"),
            report("fix-6-path", "gfx942", "7.1", ("ubuntu", "22")),
        ];
        let rollup = Rollup::of(&outcomes);

        assert_eq!(rollup.total(), 3);
        assert_eq!(rollup.understood.count, 1);
        assert_eq!(rollup.unread[&(REPORT_SCHEMA_VERSION + 1)], 1);
        assert_eq!(rollup.unread[&0], 1);

        let text = rollup.render_text();
        assert!(text.contains("unread 2"), "{text}");
        assert!(
            text.contains(&format!("schema {}", REPORT_SCHEMA_VERSION + 1)),
            "{text}"
        );
        assert!(text.contains("no readable schema"), "{text}");
        assert_eq!(rollup.to_json()["unread"]["0"], 1);
    }

    /// A refusal is the disclosure guard working, not a reader fault, so it is
    /// counted by its rule and apart from both other outcomes.
    #[test]
    fn a_refusal_is_counted_by_its_rule_and_apart_from_unread() {
        let outcomes = [
            ReadOutcome::Refused {
                reason: "unreleased-hardware".to_owned(),
            },
            ReadOutcome::Refused {
                reason: "unreleased-hardware".to_owned(),
            },
            ReadOutcome::Refused {
                reason: "platform-not-probed".to_owned(),
            },
        ];
        let rollup = Rollup::of(&outcomes);

        assert_eq!(rollup.total(), 3);
        assert_eq!(rollup.refused["unreleased-hardware"], 2);
        assert!(rollup.unread.is_empty());
        assert_eq!(rollup.understood.count, 0);
        assert!(rollup.render_text().contains("refused 3"));
    }

    /// Reports with no recognised cause are judged apart from the rest.
    #[test]
    fn recognised_and_unrecognised_reports_are_counted_apart() {
        let outcomes = [
            report("fix-6-path", "gfx942", "7.1", ("ubuntu", "22")),
            report("fix-6-path", "gfx942", "7.1", ("ubuntu", "22")),
            report("fix-2-group", "gfx942", "7.1", ("ubuntu", "22")),
            report(UNRECOGNISED, "gfx942", "7.1", ("ubuntu", "22")),
        ];
        let rollup = Rollup::of(&outcomes);

        assert_eq!(rollup.recognised(), 3);
        assert_eq!(rollup.unrecognised(), 1);
        assert!(
            rollup
                .render_text()
                .contains("Recognised cause: 3 in 2 entries. Unrecognised: 1."),
            "{}",
            rollup.render_text()
        );
    }

    /// A count is never presented as a number of people.
    #[test]
    fn the_rollup_says_counts_are_reports_not_people() {
        let rollup = Rollup::of(&[report("fix-6-path", "gfx942", "7.1", ("ubuntu", "22"))]);
        assert!(
            rollup
                .render_text()
                .contains("Counts are reports, not people")
        );
    }

    #[test]
    fn the_json_ranks_groups_as_the_text_does() {
        let outcomes = [
            report("fix-2-group", "gfx942", "7.1", ("ubuntu", "22")),
            report("fix-6-path", "gfx942", "7.1", ("ubuntu", "22")),
            report("fix-6-path", "gfx942", "7.1", ("ubuntu", "22")),
        ];
        let value = Rollup::of(&outcomes).to_json();

        assert_eq!(value["total"], 3);
        assert_eq!(value["groups"][0]["value"], "fix-6-path");
        assert_eq!(value["groups"][0]["count"], 2);
        assert_eq!(value["groups"][0]["groups"][0]["value"], "gfx942");
        assert_eq!(value["axes"][2], "rocm");
    }
}
