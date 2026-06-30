// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pass^N / Pass@N trial-group rollups over benchmark rows.
//!
//! Pure: no rendering, no async. The rollup is cheap, so the TUI recomputes
//! it whenever the row set changes rather than relying on the upstream
//! `pass_n_of_n` / `pass_at_n` CSV columns.
//!
//! See `../wiki/concepts/benchmark-result-schema.md` and
//! `../wiki/entities/normalize-results.md`: rows are grouped by
//! `(cell, model, backend, concurrency)` and each group of N trials yields
//! two verdicts — strict (all N passed) and lenient (at least one passed).

use std::collections::BTreeMap;

use crate::bench_schema::{BenchmarkRow, PassFail};

/// Effective verdict for one row: prefer the rolled-up `pass_fail`, fall back
///
/// to the judge verdict when the rollup is `Unknown`. A row that is `Unknown`
/// under both returns `Unknown` and never counts as a pass.
#[must_use]
pub const fn row_verdict(row: &BenchmarkRow) -> PassFail {
    match row.pass_fail {
        PassFail::Unknown => row.judge_pass_fail,
        v => v,
    }
}

/// Grouping key for a trial set. Trials within a group differ only by `run`
/// (and `trial_index`); everything that defines the backend config is held
/// fixed so Pass^N / Pass@N compare like-for-like.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RollupKey {
    cell: String,
    model: Option<String>,
    engine: Option<String>,
    tp: Option<u32>,
    dtype: Option<String>,
    concurrency: Option<u32>,
}

impl RollupKey {
    fn from_row(r: &BenchmarkRow) -> Self {
        Self {
            cell: r.cell.clone(),
            model: r.model.clone(),
            engine: r.engine.clone(),
            tp: r.tp,
            dtype: r.dtype.clone(),
            concurrency: r.concurrency,
        }
    }
}

/// Running accumulator while folding rows into a group.
#[derive(Default)]
struct Acc {
    n_trials: usize,
    n_passed: usize,
    ptps_sum: f64,
    ptps_n: usize,
    gtps_sum: f64,
    gtps_n: usize,
}

/// One trial-group rollup with Pass^N / Pass@N verdicts.
#[derive(Debug, Clone, PartialEq)]
pub struct PassNRollup {
    pub cell: String,
    pub model: Option<String>,
    pub engine: Option<String>,
    pub tp: Option<u32>,
    pub dtype: Option<String>,
    pub concurrency: Option<u32>,
    /// Number of trials in the group (N).
    pub n_trials: usize,
    /// How many trials passed.
    pub n_passed: usize,
    /// Strict verdict: all N trials passed (and N > 0).
    pub pass_n_of_n: bool,
    /// Lenient verdict: at least one of N trials passed.
    pub pass_at_n: bool,
    /// Mean `prompt_tps` over trials that reported it (`None` if none did).
    pub mean_prompt_tps: Option<f64>,
    /// Mean `gen_tps` over trials that reported it (`None` if none did).
    pub mean_gen_tps: Option<f64>,
}

/// Group `rows` into trial sets and compute Pass^N / Pass@N per group.
///
/// Output is sorted deterministically by the grouping key (cell first, then
/// model where `None` sorts before any name, then engine/tp/dtype/concurrency),
/// so callers can render it directly without re-sorting. Accepts any iterator
/// of row references so callers can avoid materializing a contiguous slice.
#[must_use]
pub fn rollup_pass_n<'a, I>(rows: I) -> Vec<PassNRollup>
where
    I: IntoIterator<Item = &'a BenchmarkRow>,
{
    let mut groups: BTreeMap<RollupKey, Acc> = BTreeMap::new();

    for r in rows {
        let acc = groups.entry(RollupKey::from_row(r)).or_default();
        acc.n_trials += 1;
        if row_verdict(r) == PassFail::Pass {
            acc.n_passed += 1;
        }
        if let Some(p) = r.prompt_tps {
            acc.ptps_sum += p;
            acc.ptps_n += 1;
        }
        if let Some(g) = r.gen_tps {
            acc.gtps_sum += g;
            acc.gtps_n += 1;
        }
    }

    groups
        .into_iter()
        .map(|(key, acc)| PassNRollup {
            cell: key.cell,
            model: key.model,
            engine: key.engine,
            tp: key.tp,
            dtype: key.dtype,
            concurrency: key.concurrency,
            n_trials: acc.n_trials,
            n_passed: acc.n_passed,
            pass_n_of_n: acc.n_trials > 0 && acc.n_passed == acc.n_trials,
            pass_at_n: acc.n_passed > 0,
            mean_prompt_tps: mean(acc.ptps_sum, acc.ptps_n),
            mean_gen_tps: mean(acc.gtps_sum, acc.gtps_n),
        })
        .collect()
}

fn mean(sum: f64, n: usize) -> Option<f64> {
    if n > 0 { Some(sum / n as f64) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cell: &str, run: u32, verdict: PassFail, ptps: Option<f64>) -> BenchmarkRow {
        BenchmarkRow {
            cell: cell.to_string(),
            run,
            model: Some("m".into()),
            engine: Some("vllm".into()),
            tp: Some(4),
            dtype: Some("fp8".into()),
            concurrency: Some(64),
            pass_fail: verdict,
            prompt_tps: ptps,
            ..Default::default()
        }
    }

    #[test]
    fn all_trials_pass_is_strict_and_lenient() {
        let rows = [
            row("A", 1, PassFail::Pass, Some(100.0)),
            row("A", 2, PassFail::Pass, Some(200.0)),
        ];
        let out = rollup_pass_n(&rows);
        assert_eq!(out.len(), 1);
        let g = &out[0];
        assert_eq!(g.n_trials, 2);
        assert_eq!(g.n_passed, 2);
        assert!(g.pass_n_of_n);
        assert!(g.pass_at_n);
        assert!((g.mean_prompt_tps.unwrap() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn mixed_trials_fail_strict_but_pass_lenient() {
        let rows = [
            row("A", 1, PassFail::Fail, None),
            row("A", 2, PassFail::Pass, Some(50.0)),
        ];
        let g = &rollup_pass_n(&rows)[0];
        assert_eq!(g.n_trials, 2);
        assert_eq!(g.n_passed, 1);
        assert!(!g.pass_n_of_n, "not all trials passed");
        assert!(g.pass_at_n, "one trial passed");
    }

    #[test]
    fn all_trials_fail_is_neither() {
        let rows = [
            row("A", 1, PassFail::Fail, None),
            row("A", 2, PassFail::Fail, None),
        ];
        let g = &rollup_pass_n(&rows)[0];
        assert!(!g.pass_n_of_n);
        assert!(!g.pass_at_n);
        assert_eq!(g.n_passed, 0);
    }

    #[test]
    fn single_trial_group() {
        let g = &rollup_pass_n(&[row("A", 1, PassFail::Pass, Some(10.0))])[0];
        assert_eq!(g.n_trials, 1);
        assert!(g.pass_n_of_n);
        assert!(g.pass_at_n);
    }

    #[test]
    fn distinct_configs_do_not_merge() {
        let mut r2 = row("A", 1, PassFail::Pass, None);
        r2.tp = Some(8); // different backend config → separate group
        let rows = [row("A", 1, PassFail::Pass, None), r2];
        assert_eq!(rollup_pass_n(&rows).len(), 2);
    }

    #[test]
    fn unknown_verdict_does_not_count_as_pass() {
        let rows = [
            row("A", 1, PassFail::Unknown, None),
            row("A", 2, PassFail::Pass, None),
        ];
        let g = &rollup_pass_n(&rows)[0];
        assert_eq!(g.n_trials, 2);
        assert_eq!(g.n_passed, 1);
        assert!(!g.pass_n_of_n);
        assert!(g.pass_at_n);
    }

    #[test]
    fn row_verdict_falls_back_to_judge() {
        let mut r = row("X", 1, PassFail::Unknown, None);
        r.judge_pass_fail = PassFail::Pass;
        assert_eq!(row_verdict(&r), PassFail::Pass);
        r.pass_fail = PassFail::Fail; // explicit rollup wins over judge
        assert_eq!(row_verdict(&r), PassFail::Fail);
    }

    #[test]
    fn mean_throughput_is_none_when_no_trial_reports() {
        let g = &rollup_pass_n(&[row("A", 1, PassFail::Pass, None)])[0];
        assert!(g.mean_prompt_tps.is_none());
    }

    #[test]
    fn output_sorted_by_cell() {
        let rows = [
            row("B", 1, PassFail::Pass, None),
            row("A", 1, PassFail::Pass, None),
        ];
        let out = rollup_pass_n(&rows);
        assert_eq!(out[0].cell, "A");
        assert_eq!(out[1].cell, "B");
    }

    #[test]
    fn within_cell_none_model_sorts_before_named() {
        let mut named = row("A", 1, PassFail::Pass, None);
        named.model = Some("z-model".into());
        let mut unnamed = row("A", 1, PassFail::Pass, None);
        unnamed.model = None;
        // Pass in named-first order; output must put the None-model group first.
        let out = rollup_pass_n(&[named, unnamed]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].model, None);
        assert_eq!(out[1].model.as_deref(), Some("z-model"));
    }

    #[test]
    fn accepts_borrowed_iterator() {
        // Exercises the IntoIterator signature with a `.iter()` source.
        let v = [row("A", 1, PassFail::Pass, Some(10.0))];
        let out = rollup_pass_n(v.iter());
        assert_eq!(out.len(), 1);
        assert!(out[0].pass_n_of_n);
    }
}
