// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure reducer. `State::apply(StateEvent) -> Vec<SideEffect>`.
//! See `../wiki/concepts/tea-reducer-pattern.md`.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::bench_schema::BenchmarkRow;
use crate::metrics::{Instance, Snapshot};

/// Maximum sparkline history we keep in-state.
pub const SNAPSHOT_RING_CAP: usize = 300;

/// Maximum benchmark rows kept in memory (FIFO).
pub const BENCH_RING_CAP: usize = 10_000;

/// Maximum streamed output lines retained per job (FIFO ring).
pub const JOB_OUTPUT_RING_CAP: usize = 1_000;

/// Identifier for a background job. A caller-supplied, human-meaningful string
/// (e.g. `"serve-llama3"`); the reducer treats it as opaque.
pub type JobId = String;

/// Lifecycle of a background job, as the reducer sees it. A semantic enum — the
/// core never carries `tokio`/`ratatui` types;
/// the async runtime that actually spawns the process lives in `rocm-dash-tui`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// The job has been requested / is streaming output.
    Running,
    /// The process exited with this status code.
    Done { code: i32 },
    /// The job failed before/while running (spawn error, non-UTF8, etc.).
    Failed { message: String },
    /// The user cancelled the job; the runtime is tearing the process down.
    Cancelled,
}

/// Per-job model: the streamed output ring plus the shared cancel flag the
/// async runtime watches. `Arc<AtomicBool>` is `std` only (no `tokio`), so it
/// is safe at the core boundary.
#[derive(Debug)]
pub struct JobState {
    pub cmd: String,
    pub args: Vec<String>,
    pub status: JobStatus,
    /// Streamed stdout/stderr lines, bounded to [`JOB_OUTPUT_RING_CAP`].
    pub output: VecDeque<String>,
    /// Set to `true` by [`StateEvent::CancelJob`]; the runtime polls it and
    /// kills the child process. The runtime holds a clone handed to it via
    /// [`SideEffect::SpawnJob`].
    pub cancel: Arc<AtomicBool>,
}

impl JobState {
    /// `true` once the job has reached any terminal status.
    pub const fn is_terminal(&self) -> bool {
        !matches!(self.status, JobStatus::Running)
    }

    /// The most recent output line, if any. (D4 `ring.latest()` convenience.)
    pub fn latest(&self) -> Option<&str> {
        self.output.back().map(String::as_str)
    }
}

#[derive(Debug, Default)]
pub struct State {
    pub latest: Option<Snapshot>,
    pub history: VecDeque<Snapshot>,
    pub instances: HashMap<String, Instance>,
    pub bench_rows: VecDeque<BenchmarkRow>,
    pub paused: bool,
    /// Background jobs keyed by [`JobId`]. The async runtime feeds line/done/err
    /// events back through the reducer.
    pub jobs: HashMap<JobId, JobState>,
}

impl State {
    /// Read-only accessor for a job by id.
    pub fn job(&self, id: &str) -> Option<&JobState> {
        self.jobs.get(id)
    }
}

#[derive(Debug, Clone)]
pub enum StateEvent {
    Tick(Snapshot),
    InstanceUpserted(Instance),
    InstanceRemoved(String),
    BenchmarkRows(Vec<BenchmarkRow>),
    Pause,
    Resume,
    Reset,
    // --- Job bridge (Phase 3 Wave 0) ---
    /// Request a background job. The reducer registers it `Running` and emits
    /// [`SideEffect::SpawnJob`] for the async runtime to actually launch.
    StartJob {
        id: JobId,
        cmd: String,
        args: Vec<String>,
    },
    /// One streamed output line from a running job.
    JobLine {
        id: JobId,
        line: String,
    },
    /// The job's process exited with `code`.
    JobDone {
        id: JobId,
        code: i32,
    },
    /// The job failed (spawn error / I/O error). Carries a human message.
    JobErr {
        id: JobId,
        message: String,
    },
    /// User-requested cancellation. Flips the shared cancel flag and marks the
    /// job `Cancelled` immediately; the runtime tears the process down.
    CancelJob(JobId),
}

#[derive(Debug, Clone)]
pub enum SideEffect {
    Persist,
    BroadcastSnapshot,
    BroadcastInstance(String),
    BroadcastInstanceRemoved(String),
    BroadcastBenchRows(usize),
    // --- Job bridge (Phase 3 Wave 0) ---
    /// Launch a background process. The async runtime (in `rocm-dash-tui`)
    /// interprets this: spawn the child, stream lines back as
    /// [`StateEvent::JobLine`], finish with [`StateEvent::JobDone`] /
    /// [`StateEvent::JobErr`], and watch `cancel` to kill it early.
    SpawnJob {
        id: JobId,
        cmd: String,
        args: Vec<String>,
        cancel: Arc<AtomicBool>,
    },
}

impl State {
    pub fn apply(&mut self, event: StateEvent) -> Vec<SideEffect> {
        match event {
            StateEvent::Tick(snap) => {
                if self.paused {
                    return Vec::new();
                }
                self.history.push_back(snap.clone());
                while self.history.len() > SNAPSHOT_RING_CAP {
                    self.history.pop_front();
                }
                self.latest = Some(snap);
                vec![SideEffect::BroadcastSnapshot]
            }
            StateEvent::InstanceUpserted(inst) => {
                let id = inst.container_id.clone();
                self.instances.insert(id.clone(), inst);
                vec![SideEffect::BroadcastInstance(id)]
            }
            StateEvent::InstanceRemoved(id) => {
                self.instances.remove(&id);
                vec![SideEffect::BroadcastInstanceRemoved(id)]
            }
            StateEvent::BenchmarkRows(rows) => {
                let n = rows.len();
                for r in rows {
                    self.bench_rows.push_back(r);
                    while self.bench_rows.len() > BENCH_RING_CAP {
                        self.bench_rows.pop_front();
                    }
                }
                vec![SideEffect::BroadcastBenchRows(n), SideEffect::Persist]
            }
            StateEvent::Pause => {
                self.paused = true;
                Vec::new()
            }
            StateEvent::Resume => {
                self.paused = false;
                Vec::new()
            }
            StateEvent::Reset => {
                self.latest = None;
                self.history.clear();
                self.instances.clear();
                self.bench_rows.clear();
                Vec::new()
            }
            StateEvent::StartJob { id, cmd, args } => {
                // Idempotent: never double-spawn a job that is already running.
                if self.jobs.get(&id).is_some_and(|j| !j.is_terminal()) {
                    return Vec::new();
                }
                let cancel = Arc::new(AtomicBool::new(false));
                self.jobs.insert(
                    id.clone(),
                    JobState {
                        cmd: cmd.clone(),
                        args: args.clone(),
                        status: JobStatus::Running,
                        output: VecDeque::new(),
                        cancel: Arc::clone(&cancel),
                    },
                );
                vec![SideEffect::SpawnJob {
                    id,
                    cmd,
                    args,
                    cancel,
                }]
            }
            StateEvent::JobLine { id, line } => {
                if let Some(job) = self.jobs.get_mut(&id) {
                    // Drop late lines for terminal jobs (cancel/done race).
                    if !job.is_terminal() {
                        job.output.push_back(line);
                        while job.output.len() > JOB_OUTPUT_RING_CAP {
                            job.output.pop_front();
                        }
                    }
                }
                Vec::new()
            }
            StateEvent::JobDone { id, code } => {
                if let Some(job) = self.jobs.get_mut(&id)
                    && !job.is_terminal()
                {
                    job.status = JobStatus::Done { code };
                }
                Vec::new()
            }
            StateEvent::JobErr { id, message } => {
                if let Some(job) = self.jobs.get_mut(&id)
                    && !job.is_terminal()
                {
                    job.status = JobStatus::Failed { message };
                }
                Vec::new()
            }
            StateEvent::CancelJob(id) => {
                if let Some(job) = self.jobs.get_mut(&id)
                    && !job.is_terminal()
                {
                    // Signal the runtime, then reflect the cancel immediately so
                    // the UI updates without waiting for the process teardown.
                    job.cancel.store(true, Ordering::SeqCst);
                    job.status = JobStatus::Cancelled;
                }
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn snap_at(secs: i64) -> Snapshot {
        Snapshot {
            timestamp: chrono::DateTime::<Utc>::from_timestamp(secs, 0).unwrap(),
            ..Snapshot::default()
        }
    }

    #[test]
    fn tick_pushes_history_and_broadcasts() {
        let mut s = State::default();
        let fx = s.apply(StateEvent::Tick(snap_at(1)));
        assert_eq!(s.history.len(), 1);
        assert!(s.latest.is_some());
        assert!(matches!(fx.as_slice(), [SideEffect::BroadcastSnapshot]));
    }

    #[test]
    fn pause_drops_ticks() {
        let mut s = State::default();
        s.apply(StateEvent::Pause);
        let fx = s.apply(StateEvent::Tick(snap_at(1)));
        assert!(fx.is_empty());
        assert_eq!(s.history.len(), 0);
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn history_caps_at_ring_size() {
        let mut s = State::default();
        for i in 0..(SNAPSHOT_RING_CAP + 5) as i64 {
            s.apply(StateEvent::Tick(snap_at(i)));
        }
        assert_eq!(s.history.len(), SNAPSHOT_RING_CAP);
    }

    #[test]
    fn instance_lifecycle() {
        let mut s = State::default();
        let inst = Instance {
            container_id: "c1".into(),
            ..Instance::default()
        };
        s.apply(StateEvent::InstanceUpserted(inst));
        assert!(s.instances.contains_key("c1"));
        s.apply(StateEvent::InstanceRemoved("c1".into()));
        assert!(!s.instances.contains_key("c1"));
    }

    // --- Job bridge (Phase 3 Wave 0) ---

    fn start(s: &mut State, id: &str) -> Vec<SideEffect> {
        s.apply(StateEvent::StartJob {
            id: id.into(),
            cmd: "echo".into(),
            args: vec!["hi".into()],
        })
    }

    #[test]
    fn start_job_registers_running_and_emits_spawn() {
        let mut s = State::default();
        let fx = start(&mut s, "j1");
        let job = s.job("j1").expect("job registered");
        assert_eq!(job.status, JobStatus::Running);
        assert!(job.output.is_empty());
        assert!(!job.cancel.load(Ordering::SeqCst));
        match fx.as_slice() {
            [
                SideEffect::SpawnJob {
                    id,
                    cmd,
                    args,
                    cancel,
                },
            ] => {
                assert_eq!(id, "j1");
                assert_eq!(cmd, "echo");
                assert_eq!(args, &vec!["hi".to_string()]);
                // The spawned cancel handle is the same flag the job holds.
                assert!(!cancel.load(Ordering::SeqCst));
            }
            other => panic!("expected one SpawnJob, got {other:?}"),
        }
    }

    #[test]
    fn restarting_a_running_job_is_idempotent() {
        let mut s = State::default();
        start(&mut s, "j1");
        let fx = start(&mut s, "j1");
        assert!(fx.is_empty(), "running job must not respawn");
        assert_eq!(s.jobs.len(), 1);
    }

    #[test]
    fn terminal_job_can_be_restarted() {
        let mut s = State::default();
        start(&mut s, "j1");
        s.apply(StateEvent::JobDone {
            id: "j1".into(),
            code: 0,
        });
        let fx = start(&mut s, "j1");
        assert_eq!(fx.len(), 1, "a finished job may be relaunched");
        assert_eq!(s.job("j1").unwrap().status, JobStatus::Running);
    }

    #[test]
    fn job_lines_append_and_are_bounded() {
        let mut s = State::default();
        start(&mut s, "j1");
        for i in 0..(JOB_OUTPUT_RING_CAP + 25) {
            s.apply(StateEvent::JobLine {
                id: "j1".into(),
                line: format!("line {i}"),
            });
        }
        let job = s.job("j1").unwrap();
        assert_eq!(job.output.len(), JOB_OUTPUT_RING_CAP);
        // Oldest evicted; newest retained.
        assert_eq!(
            job.latest(),
            Some(format!("line {}", JOB_OUTPUT_RING_CAP + 24)).as_deref()
        );
        assert_eq!(job.output.front().map(String::as_str), Some("line 25"));
    }

    #[test]
    fn job_done_and_err_mark_terminal_once() {
        let mut s = State::default();
        start(&mut s, "ok");
        s.apply(StateEvent::JobDone {
            id: "ok".into(),
            code: 3,
        });
        assert_eq!(s.job("ok").unwrap().status, JobStatus::Done { code: 3 });
        // A later JobErr must not overwrite a terminal status.
        s.apply(StateEvent::JobErr {
            id: "ok".into(),
            message: "ignored".into(),
        });
        assert_eq!(s.job("ok").unwrap().status, JobStatus::Done { code: 3 });

        let mut s = State::default();
        start(&mut s, "bad");
        s.apply(StateEvent::JobErr {
            id: "bad".into(),
            message: "boom".into(),
        });
        assert_eq!(
            s.job("bad").unwrap().status,
            JobStatus::Failed {
                message: "boom".into()
            }
        );
    }

    #[test]
    fn cancel_sets_flag_and_status_and_drops_late_lines() {
        let mut s = State::default();
        let fx = start(&mut s, "j1");
        // Grab the runtime's cancel handle from the emitted effect.
        let SideEffect::SpawnJob { cancel, .. } = fx.into_iter().next().unwrap() else {
            unreachable!()
        };
        s.apply(StateEvent::CancelJob("j1".into()));
        assert_eq!(s.job("j1").unwrap().status, JobStatus::Cancelled);
        assert!(cancel.load(Ordering::SeqCst), "runtime flag must be set");

        // Lines arriving after cancellation are dropped.
        s.apply(StateEvent::JobLine {
            id: "j1".into(),
            line: "late".into(),
        });
        assert!(s.job("j1").unwrap().output.is_empty());
        // A racing JobDone does not resurrect a cancelled job.
        s.apply(StateEvent::JobDone {
            id: "j1".into(),
            code: 0,
        });
        assert_eq!(s.job("j1").unwrap().status, JobStatus::Cancelled);
    }

    #[test]
    fn events_for_unknown_jobs_are_ignored() {
        let mut s = State::default();
        // None of these panic or create a job.
        s.apply(StateEvent::JobLine {
            id: "ghost".into(),
            line: "x".into(),
        });
        s.apply(StateEvent::JobDone {
            id: "ghost".into(),
            code: 0,
        });
        s.apply(StateEvent::CancelJob("ghost".into()));
        assert!(s.jobs.is_empty());
    }
}
