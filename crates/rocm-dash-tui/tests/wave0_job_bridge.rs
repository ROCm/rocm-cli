// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Wave 0 exit gate.
//!
//! Proves the job-bridge spine end-to-end: a real long-running child process is
//! launched through the reducer's [`SideEffect::SpawnJob`], its output streams
//! back as [`StateEvent::JobLine`] events that the pure reducer accumulates, and
//! a [`StateEvent::CancelJob`] tears the process down via the shared
//! `Arc<AtomicBool>`. Plus `TestBackend` buffer snapshots of the two Wave-0
//! render seams (job console + approval gate).

use std::time::Duration;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use rocm_dash_core::state::{JobStatus, State, StateEvent};
use rocm_dash_tui::jobs;
use rocm_dash_tui::ui::approval::{ApprovalChoice, ApprovalRequest, draw_approval};
use rocm_dash_tui::ui::job_console::draw_job_console;
use rocm_dash_tui::ui::theme::Theme;
use tokio::sync::mpsc;

/// Drive a real long-running command through the bridge, stream a few lines,
/// then cancel it. The job must end `Cancelled` with output captured.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn long_job_streams_then_cancels() {
    let (tx, mut rx) = mpsc::unbounded_channel::<StateEvent>();
    let mut state = State::default();

    // A shell loop that prints 200 lines slowly — long enough to cancel mid-run.
    let fx = state.apply(StateEvent::StartJob {
        id: "demo".into(),
        cmd: "sh".into(),
        args: vec![
            "-c".into(),
            "i=0; while [ $i -lt 200 ]; do echo line$i; i=$((i+1)); sleep 0.05; done".into(),
        ],
    });
    // The runtime interprets SpawnJob: launches the child, streams lines back.
    jobs::run_effects(fx, &tx);

    // Collect streamed lines until we have several, then cancel.
    let mut seen = 0;
    while seen < 3 {
        let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("job produced output before timeout")
            .expect("channel open");
        if matches!(ev, StateEvent::JobLine { .. }) {
            seen += 1;
        }
        state.apply(ev);
    }
    assert!(seen >= 3, "streamed at least three lines");
    assert_eq!(state.job("demo").unwrap().status, JobStatus::Running);

    // Cancel: flips the shared flag (reducer) → runtime kills the child.
    state.apply(StateEvent::CancelJob("demo".into()));
    assert_eq!(state.job("demo").unwrap().status, JobStatus::Cancelled);

    // Drain any in-flight events; the process should stop producing shortly.
    // A racing JobDone/JobLine must not resurrect or grow the cancelled job.
    let lines_at_cancel = state.job("demo").unwrap().output.len();
    while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_millis(400), rx.recv()).await {
        state.apply(ev);
    }
    let job = state.job("demo").unwrap();
    assert_eq!(job.status, JobStatus::Cancelled, "stays cancelled");
    assert_eq!(
        job.output.len(),
        lines_at_cancel,
        "no lines accepted after cancel"
    );
    assert!(lines_at_cancel >= 3, "captured the streamed output");
}

/// A spawn of a non-existent binary surfaces a `JobErr` → `Failed` status.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_binary_fails_cleanly() {
    let (tx, mut rx) = mpsc::unbounded_channel::<StateEvent>();
    let mut state = State::default();
    let fx = state.apply(StateEvent::StartJob {
        id: "nope".into(),
        cmd: "this-binary-does-not-exist-rocmdash".into(),
        args: vec![],
    });
    jobs::run_effects(fx, &tx);

    let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("error reported before timeout")
        .expect("channel open");
    state.apply(ev);
    assert!(matches!(
        state.job("nope").unwrap().status,
        JobStatus::Failed { .. }
    ));
}

/// A short command that exits on its own reaches `Done { code: 0 }` and all of
/// its output is drained before completion is reported.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn short_job_completes_with_all_output() {
    let (tx, mut rx) = mpsc::unbounded_channel::<StateEvent>();
    let mut state = State::default();
    let fx = state.apply(StateEvent::StartJob {
        id: "echo".into(),
        cmd: "sh".into(),
        args: vec!["-c".into(), "printf 'a\\nb\\nc\\n'".into()],
    });
    jobs::run_effects(fx, &tx);

    loop {
        let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("completes before timeout")
            .expect("channel open");
        let done = matches!(ev, StateEvent::JobDone { .. });
        state.apply(ev);
        if done {
            break;
        }
    }
    let job = state.job("echo").unwrap();
    assert_eq!(job.status, JobStatus::Done { code: 0 });
    let out: Vec<&str> = job.output.iter().map(String::as_str).collect();
    assert_eq!(out, vec!["a", "b", "c"], "all lines drained before done");
}

fn render<F: FnOnce(&mut ratatui::Frame)>(cols: u16, rows: u16, draw: F) -> String {
    let backend = TestBackend::new(cols, rows);
    let mut term = Terminal::new(backend).unwrap();
    term.draw(|f| draw(f)).unwrap();
    let buf = term.backend().buffer().clone();
    buf.content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect()
}

#[test]
fn job_console_snapshot_renders_status_and_output() {
    let theme = Theme::from_name("default-dark");
    let mut state = State::default();
    state.apply(StateEvent::StartJob {
        id: "serve".into(),
        cmd: "rocm".into(),
        args: vec!["serve".into(), "llama3".into()],
    });
    for i in 0..4 {
        state.apply(StateEvent::JobLine {
            id: "serve".into(),
            line: format!("loading shard {i}"),
        });
    }
    let job = state.job("serve").unwrap();
    let out = render(140, 32, |f| draw_job_console(f, f.area(), job, 0, &theme));

    assert!(out.contains("rocm serve llama3"), "title shows the command");
    assert!(out.contains("status"), "status badge present");
    assert!(out.contains("running"), "running status label");
    assert!(out.contains("loading shard 0"), "streamed output rendered");
    assert!(out.contains("Ctrl+C cancel"), "cancel hint while running");
}

#[test]
fn approval_snapshot_renders_request_and_buttons() {
    let theme = Theme::from_name("default-dark");
    let req = ApprovalRequest::new(
        "serve llama3 (managed)",
        vec![
            "rocm serve llama3 --engine vllm --port 8000 --managed".into(),
            "Starts a managed vLLM service and registers it.".into(),
        ],
    );
    let out = render(120, 26, |f| {
        draw_approval(f, f.area(), &req, ApprovalChoice::Approve, &theme);
    });

    assert!(
        out.contains("Review: serve llama3"),
        "title with screen name"
    );
    assert!(out.contains("rocm serve llama3"), "command preview shown");
    assert!(out.contains("Approve"), "approve button");
    assert!(out.contains("Deny"), "deny button");
    assert!(out.contains("Esc/q cancel"), "cancel hint");
}
