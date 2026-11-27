// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Measure how often a GPU lane is re-run for a merge-queue entry:
//! `cargo xtask gpu-rerun-rate`.
//!
//! The figure is the share of merge-queue entries for which at least one GPU
//! lane was re-run, held against [`GPU_RERUN_TARGET_PERCENT`]. It reads the
//! GitHub Actions API through `gh` and prints; it gates nothing, re-runs
//! nothing and publishes nothing.
//!
//! What the words mean here, as the workflows define them:
//!
//! - A **merge-queue entry** is one `merge_group` run of [`ENTRY_WORKFLOW`]. Every
//!   group the queue builds fires it once, and that workflow holds the required
//!   checks, so it is the one run every entry is certain to have. Entries are
//!   keyed by head commit; a pull request that re-enters the queue is a new
//!   entry, because the queue built a new group for it.
//! - A **GPU lane** is a job of [`GPU_WORKFLOW`] whose name starts with
//!   [`GPU_LANE_PREFIX`]. GitHub fires that workflow's `merge_group` trigger
//!   once per group too, so an entry normally has one such run. Nothing in
//!   the workflow enforces that, though (its `concurrency:` group only orders
//!   runs), so [`summarize`] takes every GPU run with the entry's head commit.
//!   `every_hardware_lane_is_named_with_the_gpu_lane_prefix` ties the prefix
//!   to the lanes in the workflow file.
//! - A **re-run** is a GPU-lane job that ran again on a later attempt of its
//!   run: a `run_attempt` above 1, not skipped on that attempt, and neither a
//!   `started_at` nor a `completed_at` that an earlier attempt of the same job
//!   has. The jobs endpoint's `filter=all` returns every attempt. A later
//!   attempt also lists, under a new job id, each job it did not run again,
//!   with that job's conclusion and some of its times: a lane that passed or
//!   failed keeps its original `started_at`
//!   (`a_lane_carried_over_into_a_later_attempt_is_not_a_rerun`), and a
//!   cancelled one gets a fresh `started_at` but keeps its original
//!   `completed_at` (`a_cancelled_lane_carried_over_is_not_a_rerun`; seen on
//!   run 40526200755). Matching either time marks the copy. A later-attempt
//!   lane that has not started yet has neither time and counts: it was asked
//!   to run again (`a_lane_queued_on_a_later_attempt_counts_as_a_rerun`). So a
//!   "re-run failed jobs", a "re-run all jobs" and a single-job re-run each
//!   count the lanes they actually ran again. A retry loop inside a step
//!   (the WSL lane's `apt-get update` loop, for one) leaves no trace in the API
//!   and is not counted.
//!
//! The computation, [`summarize`], is a pure function over the fetched runs and
//! jobs, so its tests need no network.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// The workflow whose `merge_group` runs are the merge-queue entries.
pub(crate) const ENTRY_WORKFLOW: &str = "ci.yml";
/// The workflow that holds the GPU lanes.
pub(crate) const GPU_WORKFLOW: &str = "e2e-selfhosted.yml";
/// The name every hardware lane in [`GPU_WORKFLOW`] starts with, and no other
/// job there does.
pub(crate) const GPU_LANE_PREFIX: &str = "E2E tests (";
/// The target: GPU-lane re-runs stay under this percentage of merge-queue
/// entries. Strictly under: an entry count exactly at it is above target.
pub(crate) const GPU_RERUN_TARGET_PERCENT: usize = 5;
/// The GitHub API returns at most this many runs for a filtered run listing,
/// whatever the pagination, so a window that reaches it was cut short.
const RUN_LISTING_CAP: usize = 1000;
/// The placeholders `gh api` replaces with the current repository's owner and
/// name.
const GH_CURRENT_REPO: &str = "{owner}/{repo}";

/// The command line of `gpu-rerun-rate`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// How many days back to count merge-queue entries, by the day each run was
    /// created (UTC), at least 1.
    #[arg(long, default_value_t = 30)]
    pub days: u32,
    /// Repository to read, as OWNER/NAME. Defaults to the one `gh` resolves
    /// from the current checkout (or `GH_REPO`).
    #[arg(long)]
    pub repo: Option<String>,
    /// Print the summary as JSON.
    #[arg(long)]
    pub json: bool,
}

/// One workflow run, with the fields this reads.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Run {
    pub id: u64,
    pub head_sha: String,
    #[serde(default)]
    pub head_branch: Option<String>,
    /// `completed` once every attempt so far has finished.
    #[serde(default)]
    pub status: Option<String>,
}

/// One attempt of one job, with the fields this reads.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Job {
    pub run_id: u64,
    pub run_attempt: u32,
    pub name: String,
    #[serde(default)]
    pub conclusion: Option<String>,
    /// When this attempt started. A copy carried over into a later attempt
    /// keeps the original time, unless the job was cancelled.
    #[serde(default)]
    pub started_at: Option<String>,
    /// When this attempt finished. A carried-over copy keeps the original
    /// time, cancelled or not.
    #[serde(default)]
    pub completed_at: Option<String>,
}

/// The first-attempt conclusion counted as a lane failing. `cancelled` is
/// left out because it cannot tell a lane apart from its surroundings: the
/// API reports a lane that hit its `timeout-minutes` as `cancelled` (job
/// 136846652960 on run 44753799819), and a group the queue dropped cancels
/// its lanes the same way. So lane timeouts are not in this count.
const FAILED_CONCLUSION: &str = "failure";

/// An entry with at least one GPU lane re-run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct RerunEntry {
    /// The queue's branch for the group (`gh-readonly-queue/main/pr-N-<sha>`).
    pub branch: Option<String>,
    pub head_sha: String,
    /// The GPU workflow's runs for this entry with a re-run lane (normally one).
    pub gpu_run_ids: Vec<u64>,
    /// Each re-run lane and the highest attempt it reached.
    pub lanes: BTreeMap<String, u32>,
}

/// What the command reports.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Summary {
    /// First day of the window (UTC, inclusive), `YYYY-MM-DD`.
    pub since: String,
    pub entries: usize,
    /// Entries for which no GPU workflow run was found; they count in
    /// `entries` and can never count as re-run.
    pub entries_without_gpu_run: usize,
    /// Entries whose GPU run had not completed when this read it: they count
    /// in `entries`, and a re-run still to come is not in the figure yet.
    pub entries_in_progress: usize,
    pub entries_with_gpu_rerun: usize,
    /// Entries in which some GPU lane concluded `failure` on its first
    /// attempt, re-run or not: what a re-run would have been for.
    pub entries_with_first_attempt_gpu_failure: usize,
    /// `entries_with_gpu_rerun` as a percentage of `entries`, rounded down to one decimal,
    /// or `None` with no entries.
    pub rerun_percent: Option<f64>,
    pub target_percent: usize,
    /// Whether the rate is under the target; `None` with no entries.
    pub under_target: Option<bool>,
    pub reruns: Vec<RerunEntry>,
}

/// Count the entries in `entry_runs` and their GPU-lane re-runs.
///
/// `gpu_runs` are the GPU workflow's `merge_group` runs and `gpu_jobs` every
/// attempt of every job in them. A GPU run whose head commit is no entry (a run
/// just inside the window whose entry is just outside it) is ignored.
pub(crate) fn summarize(
    since: &str,
    entry_runs: &[Run],
    gpu_runs: &[Run],
    gpu_jobs: &[Job],
) -> Summary {
    let mut entries: BTreeMap<&str, Option<&str>> = BTreeMap::new();
    for run in entry_runs {
        entries
            .entry(run.head_sha.as_str())
            .or_insert(run.head_branch.as_deref());
    }
    let mut gpu_runs_of: HashMap<&str, Vec<&Run>> = HashMap::new();
    for run in gpu_runs {
        gpu_runs_of
            .entry(run.head_sha.as_str())
            .or_default()
            .push(run);
    }

    let lanes: Vec<&Job> = gpu_jobs
        .iter()
        .filter(|job| job.name.starts_with(GPU_LANE_PREFIX))
        .collect();
    // The first attempt each start and each finish time appears in, per job:
    // a later attempt sharing either time with an earlier one is a
    // carried-over copy, not a run.
    let mut first_seen: HashMap<(u64, &str, char, &str), u32> = HashMap::new();
    for job in &lanes {
        for (kind, time) in stamps(job) {
            let attempt = first_seen
                .entry((job.run_id, job.name.as_str(), kind, time))
                .or_insert(job.run_attempt);
            *attempt = (*attempt).min(job.run_attempt);
        }
    }
    let mut reran: HashMap<u64, BTreeMap<String, u32>> = HashMap::new();
    let mut failed_first: BTreeSet<u64> = BTreeSet::new();
    for job in &lanes {
        let ran_again = job.run_attempt > 1
            && job.conclusion.as_deref() != Some("skipped")
            && stamps(job).all(|(kind, time)| {
                first_seen.get(&(job.run_id, job.name.as_str(), kind, time))
                    == Some(&job.run_attempt)
            });
        if ran_again {
            let attempt = reran
                .entry(job.run_id)
                .or_default()
                .entry(job.name.clone())
                .or_insert(0);
            *attempt = (*attempt).max(job.run_attempt);
        }
        if job.run_attempt == 1 && job.conclusion.as_deref() == Some(FAILED_CONCLUSION) {
            failed_first.insert(job.run_id);
        }
    }

    let mut reruns = Vec::new();
    let mut entries_without_gpu_run = 0;
    let mut entries_in_progress = 0;
    let mut entries_with_first_attempt_gpu_failure = 0;
    for (sha, branch) in &entries {
        let Some(runs) = gpu_runs_of.get(sha) else {
            entries_without_gpu_run += 1;
            continue;
        };
        if runs
            .iter()
            .any(|run| run.status.as_deref() != Some("completed"))
        {
            entries_in_progress += 1;
        }
        if runs.iter().any(|run| failed_first.contains(&run.id)) {
            entries_with_first_attempt_gpu_failure += 1;
        }
        let mut gpu_run_ids = Vec::new();
        let mut rerun_lanes = BTreeMap::new();
        for run in runs {
            if let Some(lanes) = reran.get(&run.id) {
                gpu_run_ids.push(run.id);
                for (lane, attempt) in lanes {
                    let highest = rerun_lanes.entry(lane.clone()).or_insert(0);
                    *highest = (*highest).max(*attempt);
                }
            }
        }
        if !gpu_run_ids.is_empty() {
            reruns.push(RerunEntry {
                branch: branch.map(str::to_owned),
                head_sha: (*sha).to_owned(),
                gpu_run_ids,
                lanes: rerun_lanes,
            });
        }
    }

    let total = entries.len();
    let rerun = reruns.len();
    Summary {
        since: since.to_owned(),
        entries: total,
        entries_without_gpu_run,
        entries_in_progress,
        entries_with_gpu_rerun: rerun,
        entries_with_first_attempt_gpu_failure,
        rerun_percent: (total > 0).then(|| percent_tenths(rerun, total) as f64 / 10.0),
        target_percent: GPU_RERUN_TARGET_PERCENT,
        under_target: (total > 0).then(|| rerun * 100 < GPU_RERUN_TARGET_PERCENT * total),
        reruns,
    }
}

/// A job's start and finish times, each tagged with which it is.
fn stamps(job: &Job) -> impl Iterator<Item = (char, &str)> {
    [
        job.started_at.as_deref().map(|t| ('s', t)),
        job.completed_at.as_deref().map(|t| ('c', t)),
    ]
    .into_iter()
    .flatten()
}

/// `part / whole` as a percentage in tenths, rounded down. Rounding down is
/// what keeps the printed figure on the same side of the target as the exact
/// comparison in [`summarize`]: a rate under 5% prints as at most 4.9, and a
/// rate of 5% or more as at least 5.0, where rounding to nearest would print
/// 4.96% as 5.0% beside "under the 5% target"
/// (`a_rate_just_under_the_target_prints_below_it`).
const fn percent_tenths(part: usize, whole: usize) -> usize {
    part * 1000 / whole
}

/// [`percent_tenths`] as text with one decimal.
fn percent(part: usize, whole: usize) -> String {
    let tenths = percent_tenths(part, whole);
    format!("{}.{}", tenths / 10, tenths % 10)
}

impl Summary {
    /// The text form, ending in a newline.
    pub(crate) fn render_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "GPU-lane re-runs in the merge queue since {} (UTC)",
            self.since
        );
        let _ = writeln!(out, "  merge-queue entries:        {}", self.entries);
        if self.rerun_percent.is_none() {
            let _ = writeln!(
                out,
                "No merge-queue entries in this window, so there is no rate to compare with the {}% target.",
                self.target_percent
            );
            return out;
        }
        let percent = percent(self.entries_with_gpu_rerun, self.entries);
        let _ = writeln!(
            out,
            "  entries with a GPU re-run:  {} ({percent}%)",
            self.entries_with_gpu_rerun
        );
        let _ = writeln!(
            out,
            "  entries with a GPU lane failing on its first attempt: {}",
            self.entries_with_first_attempt_gpu_failure
        );
        if self.entries_in_progress > 0 {
            let _ = writeln!(
                out,
                "  entries whose GPU run is still in progress: {} (a re-run still to come is not counted)",
                self.entries_in_progress
            );
        }
        if self.entries_without_gpu_run > 0 {
            let _ = writeln!(
                out,
                "  entries with no {GPU_WORKFLOW} run found: {}",
                self.entries_without_gpu_run
            );
        }
        let verdict = if self.under_target == Some(true) {
            "under"
        } else {
            "not under"
        };
        let _ = writeln!(
            out,
            "Re-run rate {percent}% is {verdict} the {}% target.",
            self.target_percent
        );
        for entry in &self.reruns {
            let lanes: Vec<String> = entry
                .lanes
                .iter()
                .map(|(lane, attempt)| format!("{lane} (attempt {attempt})"))
                .collect();
            let runs: Vec<String> = entry.gpu_run_ids.iter().map(u64::to_string).collect();
            let _ = writeln!(
                out,
                "  {} run {}: {}",
                entry.branch.as_deref().unwrap_or(&entry.head_sha),
                runs.join(", "),
                lanes.join(", ")
            );
        }
        out
    }
}

/// Fetch the window's runs and jobs through `gh` and print the summary.
pub fn run(args: &Args) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("the system clock reads before 1970")?;
    let since = window_start(now.as_secs(), args.days)?;
    // `gh api` fills these two placeholders from the current repository.
    let repo = args.repo.as_deref().unwrap_or(GH_CURRENT_REPO);
    let summary = fetch_and_summarize(&since, repo, gh_api_lines)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print!("{}", summary.render_text());
    }
    Ok(())
}

/// The first day (UTC, `YYYY-MM-DD`) of a window of `days` days ending at
/// `now_secs` seconds after 1970-01-01.
fn window_start(now_secs: u64, days: u32) -> Result<String> {
    if days < 1 {
        bail!("--days must be at least 1: the window needs a day to count");
    }
    let start = i64::try_from(now_secs)? - i64::from(days) * 86_400;
    Ok(chrono::DateTime::from_timestamp(start, 0)
        .context("the window start is out of range")?
        .format("%Y-%m-%d")
        .to_string())
}

/// Read the window from the API through `api`, which takes an endpoint and a
/// jq filter and returns one JSON value per line, as `gh api --paginate --jq`
/// prints them. A parameter so a test can hand it canned pages.
fn fetch_and_summarize(
    since: &str,
    repo: &str,
    api: impl Fn(&str, &str) -> Result<String> + Sync,
) -> Result<Summary> {
    let runs_of = |workflow: &str| -> Result<Vec<Run>> {
        let endpoint = format!(
            "repos/{repo}/actions/workflows/{workflow}/runs?event=merge_group&created=%3E%3D{since}&per_page=100"
        );
        let runs: Vec<Run> = parse_lines(&api(&endpoint, ".workflow_runs[]")?)
            .with_context(|| format!("reading the {workflow} runs"))?;
        if runs.len() >= RUN_LISTING_CAP {
            bail!(
                "{workflow} has {RUN_LISTING_CAP} or more merge_group runs since {since}, which is the most the API lists; use a shorter --days"
            );
        }
        Ok(runs)
    };
    let entry_runs = runs_of(ENTRY_WORKFLOW)?;
    let gpu_runs = runs_of(GPU_WORKFLOW)?;

    let jobs_of = |run: &Run| -> Result<Vec<Job>> {
        let endpoint = format!(
            "repos/{repo}/actions/runs/{}/jobs?filter=all&per_page=100",
            run.id
        );
        parse_lines(&api(&endpoint, ".jobs[]")?)
            .with_context(|| format!("reading the jobs of run {}", run.id))
    };
    // One `gh` call per run; a few at a time keeps a month's window to seconds
    // without hammering the API.
    let mut gpu_jobs = Vec::new();
    for chunk in gpu_runs.chunks(8) {
        let fetched: Vec<Result<Vec<Job>>> = std::thread::scope(|scope| {
            // Collected so every thread is spawned before the first join.
            #[expect(
                clippy::needless_collect,
                reason = "joining lazily would run the fetches one at a time"
            )]
            let handles: Vec<_> = chunk
                .iter()
                .map(|run| scope.spawn(|| jobs_of(run)))
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| bail!("a job-fetching thread panicked"))
                })
                .collect()
        });
        for jobs in fetched {
            gpu_jobs.extend(jobs?);
        }
    }
    Ok(summarize(since, &entry_runs, &gpu_runs, &gpu_jobs))
}

fn parse_lines<T: for<'de> Deserialize<'de>>(text: &str) -> Result<Vec<T>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("the API returned a value this cannot read"))
        .collect()
}

/// `gh api --paginate <endpoint> --jq <jq>`, its stdout.
fn gh_api_lines(endpoint: &str, jq: &str) -> Result<String> {
    // git exports GIT_DIR/GIT_WORK_TREE to hooks, and gh would resolve the repo
    // from those; same as `release_readiness::collect_known_issues_in`.
    let output = Command::new("gh")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .args(["api", "--paginate", endpoint, "--jq", jq])
        .output()
        .context("failed to run `gh api` (is the GitHub CLI installed and authenticated?)")?;
    if !output.status.success() {
        bail!(
            "`gh api {endpoint}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("`gh api` printed something that is not UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: u64, sha: &str) -> Run {
        Run {
            id,
            head_sha: sha.to_owned(),
            head_branch: Some(format!("gh-readonly-queue/main/pr-{id}-{sha}")),
            status: Some("completed".to_owned()),
        }
    }

    /// A job that ran on `attempt`, so it starts at a time of its own.
    fn job(run_id: u64, attempt: u32, name: &str, conclusion: &str) -> Job {
        Job {
            run_id,
            run_attempt: attempt,
            name: name.to_owned(),
            conclusion: Some(conclusion.to_owned()),
            started_at: Some(format!("2026-11-0{attempt}T10:00:00Z")),
            completed_at: Some(format!("2026-11-0{attempt}T11:00:00Z")),
        }
    }

    /// The copy of `original` GitHub lists under attempt `attempt` when that
    /// attempt did not run it again: same times, same conclusion.
    fn carried(original: &Job, attempt: u32) -> Job {
        Job {
            run_attempt: attempt,
            ..original.clone()
        }
    }

    /// The copy of a cancelled `original` under attempt `attempt`: a fresh
    /// start time, the original finish time (as on run 40526200755).
    fn carried_cancelled(original: &Job, attempt: u32) -> Job {
        Job {
            run_attempt: attempt,
            started_at: Some(format!("2026-11-0{attempt}T10:00:00Z")),
            ..original.clone()
        }
    }

    const WINDOWS: &str = "E2E tests (Strix Halo, Windows, current)";
    const WSL: &str = "E2E tests (Strix Halo, WSL2, current)";

    /// Twenty entries, each with a GPU run whose two lanes passed first time.
    fn clean_window() -> (Vec<Run>, Vec<Run>, Vec<Job>) {
        let entries: Vec<Run> = (1..=20).map(|i| run(i, &format!("sha{i}"))).collect();
        let gpu: Vec<Run> = (1..=20).map(|i| run(100 + i, &format!("sha{i}"))).collect();
        let jobs = gpu
            .iter()
            .flat_map(|r| {
                [
                    job(r.id, 1, "changes", "success"),
                    job(r.id, 1, WINDOWS, "success"),
                    job(r.id, 1, WSL, "success"),
                ]
            })
            .collect();
        (entries, gpu, jobs)
    }

    #[test]
    fn a_window_with_no_reruns_is_zero_percent_and_under_target() {
        let (entries, gpu, jobs) = clean_window();
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries, 20);
        assert_eq!(summary.entries_with_gpu_rerun, 0);
        assert_eq!(summary.rerun_percent, Some(0.0));
        assert_eq!(summary.under_target, Some(true));
        let text = summary.render_text();
        assert!(
            text.contains("Re-run rate 0.0% is under the 5% target."),
            "{text}"
        );
    }

    #[test]
    fn one_entry_rerun_counts_once_however_many_lanes_and_attempts() {
        let (entries, gpu, mut jobs) = clean_window();
        // Entry 3: Windows failed and was re-run twice; one "re-run all jobs"
        // ran WSL again with it once (a new start time).
        jobs.push(job(103, 1, WINDOWS, "failure"));
        jobs.push(job(103, 2, WINDOWS, "failure"));
        jobs.push(job(103, 3, WINDOWS, "success"));
        jobs.push(job(103, 2, WSL, "success"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 1);
        assert_eq!(summary.entries_with_first_attempt_gpu_failure, 1);
        assert_eq!(summary.rerun_percent, Some(5.0));
        assert_eq!(summary.reruns.len(), 1);
        assert_eq!(summary.reruns[0].gpu_run_ids, vec![103]);
        assert_eq!(
            summary.reruns[0].lanes,
            BTreeMap::from([(WINDOWS.to_owned(), 3), (WSL.to_owned(), 2)])
        );
    }

    #[test]
    fn a_lane_carried_over_into_a_later_attempt_is_not_a_rerun() {
        // "Re-run failed jobs" on entry 3: Windows ran again, WSL passed on
        // attempt 1 and is listed again under attempt 2 with its old start
        // time, then under attempt 3 too.
        let (entries, gpu, mut jobs) = clean_window();
        let wsl = jobs
            .iter()
            .find(|j| j.run_id == 103 && j.name == WSL)
            .unwrap()
            .clone();
        jobs.retain(|j| !(j.run_id == 103 && j.name == WINDOWS));
        jobs.push(job(103, 1, WINDOWS, "failure"));
        jobs.push(job(103, 2, WINDOWS, "success"));
        jobs.push(carried(&wsl, 2));
        jobs.push(carried(&wsl, 3));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 1);
        assert_eq!(
            summary.reruns[0].lanes,
            BTreeMap::from([(WINDOWS.to_owned(), 2)])
        );
    }

    #[test]
    fn re_running_only_the_report_job_is_no_gpu_rerun() {
        // Every lane of entry 4 is carried over into attempt 2; only the
        // consolidated report ran again.
        let (entries, gpu, mut jobs) = clean_window();
        let lanes: Vec<Job> = jobs
            .iter()
            .filter(|j| j.run_id == 104 && j.name.starts_with(GPU_LANE_PREFIX))
            .cloned()
            .collect();
        jobs.extend(lanes.iter().map(|lane| carried(lane, 2)));
        jobs.push(job(
            104,
            2,
            "E2E consolidated report (self-hosted)",
            "success",
        ));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 0);
    }

    #[test]
    fn only_a_first_attempt_failure_counts_as_failing_first() {
        let (entries, gpu, mut jobs) = clean_window();
        // Entry 1 fails only on a later attempt; entry 2 is cancelled, which
        // is also how the API reports a lane timeout; entry 6 fails.
        jobs.push(job(101, 2, WINDOWS, "failure"));
        jobs.push(job(102, 1, "E2E tests (W7900, current)", "cancelled"));
        jobs.push(job(106, 1, WSL, "failure"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_first_attempt_gpu_failure, 1);
    }

    #[test]
    fn a_cancelled_lane_carried_over_is_not_a_rerun() {
        // Entry 4: W7900 was cancelled on attempt 1; attempt 2 re-ran only
        // the report job and lists the lane again with a fresh start time.
        let (entries, gpu, mut jobs) = clean_window();
        let w7900 = job(104, 1, "E2E tests (W7900, current)", "cancelled");
        jobs.push(carried_cancelled(&w7900, 2));
        jobs.push(w7900);
        jobs.push(job(
            104,
            2,
            "E2E consolidated report (self-hosted)",
            "success",
        ));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 0);
    }

    #[test]
    fn a_lane_queued_on_a_later_attempt_counts_as_a_rerun() {
        // Entry 5's Windows lane was asked to run again and has not started:
        // no times yet.
        let (entries, gpu, mut jobs) = clean_window();
        jobs.push(Job {
            started_at: None,
            completed_at: None,
            conclusion: None,
            ..job(105, 2, WINDOWS, "")
        });
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 1);
        assert_eq!(summary.reruns[0].gpu_run_ids, vec![105]);
    }

    #[test]
    fn one_entry_s_runs_merge_by_highest_attempt_and_any_failure() {
        // Entry 9 has runs 109 and 300. Both re-ran Windows, to attempts 3
        // and 2; only run 300 failed a lane first time.
        let (entries, mut gpu, mut jobs) = clean_window();
        gpu.push(run(300, "sha9"));
        jobs.push(job(109, 3, WINDOWS, "success"));
        jobs.push(job(300, 1, WSL, "failure"));
        jobs.push(job(300, 2, WINDOWS, "success"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_first_attempt_gpu_failure, 1);
        assert_eq!(
            summary.reruns[0].lanes,
            BTreeMap::from([(WINDOWS.to_owned(), 3)])
        );
        // The same, with the runs listed the other way round.
        gpu.reverse();
        let reversed = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(reversed.reruns[0].lanes, summary.reruns[0].lanes);
    }

    #[test]
    fn a_rerun_entry_without_a_branch_is_named_by_its_commit() {
        let (mut entries, gpu, mut jobs) = clean_window();
        entries[1].head_branch = None;
        jobs.push(job(102, 2, WSL, "success"));
        let text = summarize("2026-10-28", &entries, &gpu, &jobs).render_text();
        assert!(
            text.contains(&format!("  sha2 run 102: {WSL} (attempt 2)")),
            "{text}"
        );
    }

    #[test]
    fn the_json_form_carries_every_field() {
        let (entries, gpu, mut jobs) = clean_window();
        jobs.push(job(103, 2, WSL, "success"));
        let summary = summarize("2026-10-28", &entries[..10], &gpu, &jobs);
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "since": "2026-10-28",
                "entries": 10,
                "entries_without_gpu_run": 0,
                "entries_in_progress": 0,
                "entries_with_gpu_rerun": 1,
                "entries_with_first_attempt_gpu_failure": 0,
                "rerun_percent": 10.0,
                "target_percent": 5,
                "under_target": false,
                "reruns": [{
                    "branch": "gh-readonly-queue/main/pr-3-sha3",
                    "head_sha": "sha3",
                    "gpu_run_ids": [103],
                    "lanes": { WSL: 2 }
                }]
            })
        );
    }

    #[test]
    fn two_gpu_runs_for_one_entry_are_both_read() {
        let (entries, mut gpu, mut jobs) = clean_window();
        // Entry 8 has its usual run 108 and a second run 200; each re-ran a
        // different lane, and both count, once, for the one entry.
        gpu.push(run(200, "sha8"));
        jobs.push(job(108, 2, WINDOWS, "success"));
        jobs.push(job(200, 2, WSL, "success"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 1);
        assert_eq!(summary.reruns[0].gpu_run_ids, vec![108, 200]);
        assert_eq!(summary.reruns[0].head_sha, "sha8");
        assert_eq!(
            summary.reruns[0].lanes,
            BTreeMap::from([(WINDOWS.to_owned(), 2), (WSL.to_owned(), 2)])
        );
    }

    #[test]
    fn an_entry_still_running_is_counted_and_reported_as_in_progress() {
        let (entries, mut gpu, jobs) = clean_window();
        gpu[0].status = Some("in_progress".to_owned());
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries, 20);
        assert_eq!(summary.entries_in_progress, 1);
        assert!(
            summary
                .render_text()
                .contains("entries whose GPU run is still in progress: 1"),
            "{}",
            summary.render_text()
        );
    }

    #[test]
    fn the_window_starts_days_before_now_and_needs_a_day() {
        // 2026-11-26T23:00:00Z: an hour before midnight, so a window a few
        // hours off lands on the wrong day.
        let now = 1_795_734_000;
        assert_eq!(window_start(now, 30).unwrap(), "2026-10-27");
        assert_eq!(window_start(now, 1).unwrap(), "2026-11-25");
        let err = window_start(now, 0).unwrap_err();
        assert!(
            err.to_string().contains("--days must be at least 1"),
            "{err}"
        );
    }

    /// 19 re-runs in 383 entries is 4.96%: under the target, so it must not
    /// print as 5.0% beside a verdict saying it is under 5%.
    #[test]
    fn a_rate_just_under_the_target_prints_below_it() {
        let entries: Vec<Run> = (1..=383).map(|i| run(i, &format!("sha{i}"))).collect();
        let gpu: Vec<Run> = (1..=383)
            .map(|i| run(1000 + i, &format!("sha{i}")))
            .collect();
        let jobs: Vec<Job> = (1..=19).map(|i| job(1000 + i, 2, WSL, "success")).collect();
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.under_target, Some(true));
        assert_eq!(summary.rerun_percent, Some(4.9));
        let text = summary.render_text();
        assert!(text.contains("19 (4.9%)"), "{text}");
        assert!(
            text.contains("Re-run rate 4.9% is under the 5% target."),
            "{text}"
        );
    }

    /// The API lists attempts in ascending order today; the counting must not
    /// depend on it.
    #[test]
    fn the_order_jobs_arrive_in_does_not_change_the_count() {
        let (entries, gpu, mut jobs) = clean_window();
        let wsl = jobs
            .iter()
            .find(|j| j.run_id == 103 && j.name == WSL)
            .unwrap()
            .clone();
        jobs.retain(|j| !(j.run_id == 103 && j.name == WINDOWS));
        jobs.push(job(103, 1, WINDOWS, "failure"));
        jobs.push(job(103, 2, WINDOWS, "failure"));
        jobs.push(job(103, 3, WINDOWS, "success"));
        jobs.push(carried(&wsl, 2));
        jobs.push(carried(&wsl, 3));
        let forward = summarize("2026-10-28", &entries, &gpu, &jobs);
        jobs.reverse();
        let backward = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(forward, backward);
        assert_eq!(
            backward.reruns[0].lanes,
            BTreeMap::from([(WINDOWS.to_owned(), 3)])
        );
    }

    #[test]
    fn exactly_the_target_is_not_under_it() {
        // 1 of 20 is 5.0%: the target is "under 5%", so this misses it, and
        // the printed verdict says the same thing the flag does.
        let (entries, gpu, mut jobs) = clean_window();
        jobs.push(job(107, 2, WSL, "success"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.under_target, Some(false));
        let text = summary.render_text();
        assert!(
            text.contains("Re-run rate 5.0% is not under the 5% target."),
            "{text}"
        );
        assert!(
            text.contains(&format!("run 107: {WSL} (attempt 2)")),
            "{text}"
        );
    }

    #[test]
    fn a_rerun_of_a_non_gpu_job_or_a_skipped_lane_is_not_a_gpu_rerun() {
        let (entries, gpu, mut jobs) = clean_window();
        jobs.push(job(104, 2, "changes", "success"));
        jobs.push(job(
            104,
            2,
            "E2E consolidated report (self-hosted)",
            "success",
        ));
        jobs.push(job(105, 2, "E2E tests (MI300X, current)", "skipped"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries_with_gpu_rerun, 0);
    }

    #[test]
    fn an_entry_without_a_gpu_run_counts_but_cannot_be_rerun() {
        let (entries, mut gpu, jobs) = clean_window();
        gpu.retain(|r| r.head_sha != "sha9");
        // A GPU run whose entry lies outside the window is not an entry.
        gpu.push(run(999, "sha-outside"));
        let mut jobs = jobs;
        jobs.push(job(999, 2, WINDOWS, "success"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries, 20);
        assert_eq!(summary.entries_without_gpu_run, 1);
        assert_eq!(summary.entries_with_gpu_rerun, 0);
        assert!(
            summary
                .render_text()
                .contains("entries with no e2e-selfhosted.yml run found: 1")
        );
    }

    #[test]
    fn duplicate_entry_runs_for_one_commit_are_one_entry() {
        let (mut entries, gpu, jobs) = clean_window();
        entries.push(run(50, "sha1"));
        let summary = summarize("2026-10-28", &entries, &gpu, &jobs);
        assert_eq!(summary.entries, 20);
    }

    #[test]
    fn no_entries_has_no_rate_and_says_so() {
        let summary = summarize("2026-10-28", &[], &[], &[]);
        assert_eq!(summary.rerun_percent, None);
        assert_eq!(summary.under_target, None);
        let text = summary.render_text();
        assert!(
            text.contains("No merge-queue entries in this window"),
            "{text}"
        );
        assert!(
            !text.contains("Re-run rate"),
            "no verdict without a rate: {text}"
        );
    }

    #[test]
    fn percent_rounds_down_to_one_decimal() {
        assert_eq!(percent(0, 376), "0.0");
        assert_eq!(percent(1, 3), "33.3");
        assert_eq!(percent(2, 3), "66.6");
        assert_eq!(percent(1, 400), "0.2");
        assert_eq!(percent(19, 376), "5.0");
        assert_eq!(percent(20, 20), "100.0");
    }

    #[test]
    fn fetch_reads_paginated_lines_for_both_workflows_and_every_gpu_run() {
        let calls = std::sync::Mutex::new(Vec::new());
        let api = |endpoint: &str, jq: &str| -> Result<String> {
            calls.lock().unwrap().push(endpoint.to_owned());
            Ok(match (endpoint, jq) {
                (e, ".workflow_runs[]") if e.contains("/workflows/ci.yml/") => concat!(
                    r#"{"id":1,"head_sha":"a","head_branch":"q/a","extra":true}"#,
                    "\n",
                    r#"{"id":2,"head_sha":"b","head_branch":"q/b"}"#,
                    "\n"
                )
                .to_owned(),
                (e, ".workflow_runs[]") if e.contains("/workflows/e2e-selfhosted.yml/") => concat!(
                    r#"{"id":11,"head_sha":"a"}"#,
                    "\n",
                    r#"{"id":12,"head_sha":"b"}"#,
                    "\n"
                )
                .to_owned(),
                (e, ".jobs[]") if e.contains("/runs/12/") => format!(
                    "{{\"run_id\":12,\"run_attempt\":2,\"name\":\"{WSL}\",\"conclusion\":null}}\n"
                ),
                (_, ".jobs[]") => String::new(),
                other => panic!("unexpected call {other:?}"),
            })
        };
        let summary = fetch_and_summarize("2026-10-28", "o/r", api).unwrap();
        assert_eq!(summary.entries, 2);
        assert_eq!(summary.entries_with_gpu_rerun, 1);
        assert_eq!(summary.reruns[0].branch.as_deref(), Some("q/b"));
        let calls = calls.into_inner().unwrap();
        assert!(calls.iter().any(|c| c
            == "repos/o/r/actions/workflows/ci.yml/runs?event=merge_group&created=%3E%3D2026-10-28&per_page=100"));
        assert!(
            calls
                .iter()
                .any(|c| c == "repos/o/r/actions/runs/11/jobs?filter=all&per_page=100")
        );
    }

    #[test]
    fn fetch_reads_the_jobs_of_every_gpu_run_past_the_first_batch() {
        // More GPU runs than one batch of threads; the only re-run is in the
        // last one.
        let count = 19;
        let runs = |base: u64| -> String {
            let mut page = String::new();
            for i in 1..=count {
                let _ = writeln!(
                    page,
                    "{{\"id\":{},\"head_sha\":\"s{i}\",\"status\":\"completed\"}}",
                    base + i
                );
            }
            page
        };
        let api = |endpoint: &str, jq: &str| -> Result<String> {
            Ok(match jq {
                ".workflow_runs[]" if endpoint.contains("/workflows/ci.yml/") => runs(0),
                ".workflow_runs[]" => runs(100),
                _ if endpoint.contains(&format!("/runs/{}/", 100 + count)) => format!(
                    "{{\"run_id\":{},\"run_attempt\":2,\"name\":\"{WSL}\",\"conclusion\":\"success\",\"started_at\":\"x\"}}\n",
                    100 + count
                ),
                _ => String::new(),
            })
        };
        let summary = fetch_and_summarize("2026-10-28", "o/r", api).unwrap();
        assert_eq!(summary.entries, 19);
        assert_eq!(summary.entries_with_gpu_rerun, 1);
        assert_eq!(summary.reruns[0].gpu_run_ids, vec![100 + count]);
    }

    #[test]
    fn a_window_at_the_listing_cap_is_refused_rather_than_undercounted() {
        let line = r#"{"id":1,"head_sha":"a"}"#;
        let page = format!("{line}\n").repeat(RUN_LISTING_CAP);
        // Each workflow on its own: the cap guards both listings.
        for capped in ["/workflows/ci.yml/", "/workflows/e2e-selfhosted.yml/"] {
            let err = fetch_and_summarize("2026-10-28", "o/r", |endpoint: &str, _: &str| {
                Ok(if endpoint.contains(capped) {
                    page.clone()
                } else {
                    String::new()
                })
            })
            .unwrap_err();
            assert!(
                err.to_string().contains("use a shorter --days"),
                "{capped}: {err}"
            );
        }
    }

    /// Each job of a workflow file: its id, its `name:` and whether its
    /// `runs-on:` names a self-hosted runner. Only the one-line label or list
    /// form of `runs-on:` is read; any other form (a block list, an
    /// expression) could hide a self-hosted label, so it panics instead.
    fn jobs_of_workflow(workflow: &str) -> Vec<(String, Option<String>, bool)> {
        let mut in_jobs = false;
        let mut jobs = Vec::new();
        let mut current: Option<(String, Option<String>, bool)> = None;
        for line in workflow.lines() {
            if line == "jobs:" {
                in_jobs = true;
                continue;
            }
            if !in_jobs {
                continue;
            }
            if let Some(id) = line.strip_prefix("  ").and_then(|l| l.strip_suffix(':'))
                && !id.starts_with(' ')
                && !id.starts_with('#')
            {
                jobs.extend(current.take());
                current = Some((id.to_owned(), None, false));
            } else if let Some((job_id, name, hardware)) = current.as_mut() {
                if let Some(n) = line.strip_prefix("    name: ") {
                    *name = Some(n.to_owned());
                } else if let Some(r) = line.strip_prefix("    runs-on:") {
                    let r = r.trim();
                    assert!(
                        !r.is_empty() && !r.contains("${{"),
                        "`runs-on` of job `{job_id}` is a form this test cannot classify: {line:?}"
                    );
                    *hardware = r.contains("self-hosted");
                }
            }
        }
        jobs.extend(current.take());
        jobs
    }

    #[test]
    fn the_workflow_scan_reads_labels_and_refuses_forms_it_cannot_classify() {
        let jobs = jobs_of_workflow(
            "jobs:\n  a:\n    name: E2E tests (X)\n    runs-on: [self-hosted, linux]\n  b:\n    name: other\n    runs-on: ubuntu-24.04 # comment\n",
        );
        assert_eq!(
            jobs,
            vec![
                ("a".to_owned(), Some("E2E tests (X)".to_owned()), true),
                ("b".to_owned(), Some("other".to_owned()), false),
            ]
        );
        for hidden in [
            "jobs:\n  c:\n    runs-on:\n      - self-hosted\n",
            "jobs:\n  d:\n    runs-on: ${{ matrix.runner }}\n",
        ] {
            let refused = std::panic::catch_unwind(|| jobs_of_workflow(hidden));
            assert!(refused.is_err(), "accepted {hidden:?}");
        }
    }

    /// The prefix is matched against job names in the workflow file, so tie it
    /// there: every job whose one-line `runs-on:` names a self-hosted runner
    /// (the scan refuses any other form), and the queue lane that drives the
    /// shared GPU queue, is named with it, and no other job is.
    #[test]
    fn every_hardware_lane_is_named_with_the_gpu_lane_prefix() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/e2e-selfhosted.yml"
        );
        let lanes = jobs_of_workflow(&std::fs::read_to_string(path).unwrap());
        let mut hardware_ids = Vec::new();
        for (id, name, hardware) in &lanes {
            let name = name.as_deref().unwrap_or("");
            let is_lane = *hardware || id == "e2e-gpu-queue";
            assert_eq!(
                name.starts_with(GPU_LANE_PREFIX),
                is_lane,
                "job `{id}` (name `{name}`): a hardware lane must start with {GPU_LANE_PREFIX:?} and nothing else may"
            );
            if *hardware {
                hardware_ids.push(id.as_str());
            }
        }
        // The scan found the lanes at all (a parse that sees no jobs passes
        // the loop above vacuously).
        for id in ["e2e-gpu", "e2e-gpu-strix-windows", "e2e-wsl"] {
            assert!(
                hardware_ids.contains(&id),
                "{id} not found among {hardware_ids:?}"
            );
        }
    }
}
