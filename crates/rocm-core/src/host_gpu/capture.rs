// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Optional-command capture for the host probes.
//!
//! Runs a tool that may not be installed under a short timeout and keeps its
//! stdout only on success, trying each spelling of the program that
//! [`crate::probe_exec::tool_path_candidates`] gives (`PATHEXT` and the fixed
//! Windows system locations included). A missing, slow or failing tool reads as
//! `None`, never as an error.

use crate::command_capture::{CommandCapture, CommandOutcome};
use crate::probe_exec::tool_path_candidates;
use crate::process::run_bounded;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

pub(super) const OPTIONAL_COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);

pub(super) fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    capture_optional_command_with_timeout(program, args, OPTIONAL_COMMAND_TIMEOUT)
}

pub(super) fn capture_optional_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    for candidate in tool_path_candidates(program) {
        if let Some(output) =
            capture_optional_command_candidate_with_timeout(Path::new(&candidate), args, timeout)
        {
            return Some(output);
        }
    }
    None
}

/// One spelling of the program, run through [`run_bounded`], the executor every
/// host probe shares; [`optional_stdout`] says what the run is worth.
fn capture_optional_command_candidate_with_timeout(
    program: &Path,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    candidate_with_debug(program, args, timeout, &mut debug_command_capture)
}

/// [`capture_optional_command_candidate_with_timeout`] with the place its debug
/// lines go as an argument, so a test can read them. The sink is handed every
/// line; whether it is printed is its own decision.
fn candidate_with_debug(
    program: &Path,
    args: &[&str],
    timeout: Duration,
    debug: &mut impl FnMut(String),
) -> Option<String> {
    let mut command = Command::new(program);
    command.args(args);
    let capture = run_bounded(command, timeout, "process");
    if let Some(note) = incomplete_output_note(&capture) {
        debug(format!(
            "rocm debug: command capture output from {} may be incomplete: {note}",
            program.display()
        ));
    }
    match optional_stdout(capture) {
        Ok(stdout) => stdout,
        Err((stage, detail)) => {
            debug(format!(
                "rocm debug: command capture {stage} failed for {}: {detail}",
                program.display()
            ));
            None
        }
    }
}

/// Why the stdout of a run that exited 0 may be shorter than the program wrote:
/// a descendant still writing when the executor stopped waiting for it, or a
/// pipe that failed to read. [`optional_stdout`] gives the bytes that arrived
/// either way, so this is the only place the shortfall shows.
fn incomplete_output_note(capture: &CommandCapture) -> Option<&str> {
    match capture.outcome {
        CommandOutcome::Exited(0) => capture.detail.as_deref(),
        _ => None,
    }
}

/// What an optional capture makes of a finished run: the stdout of a program
/// that exited 0 and wrote valid UTF-8, and nothing for any other run.
///
/// `Ok(None)` is a successful run whose output is not UTF-8, which reads as no
/// answer and is not worth a debug line. `Err` is a run that did not succeed,
/// with the stage and wording for that line. A pipe that failed to read part way
/// does not change the answer: the bytes that arrived are used, as they were
/// before the capture moved onto the shared executor.
fn optional_stdout(capture: CommandCapture) -> Result<Option<String>, (&'static str, String)> {
    match capture.outcome {
        CommandOutcome::Exited(0) => Ok(String::from_utf8(capture.stdout_raw).ok()),
        CommandOutcome::Exited(_) | CommandOutcome::Signaled(_) => Err((
            "exit",
            capture.status.map_or_else(
                || "no status".to_owned(),
                |status| format!("status {status}"),
            ),
        )),
        CommandOutcome::TimedOut => Err(("timeout", "timed out".to_owned())),
        CommandOutcome::WaitFailed => Err(("wait", "failed to wait".to_owned())),
        CommandOutcome::NotFound | CommandOutcome::SpawnFailed(_) => Err((
            "spawn",
            capture
                .detail
                .unwrap_or_else(|| "failed to start".to_owned()),
        )),
    }
}

/// Print one debug line, when `ROCM_CLI_DEBUG_COMMAND_CAPTURE` asks for them.
fn debug_command_capture(line: String) {
    if !crate::runtime::env_flag("ROCM_CLI_DEBUG_COMMAND_CAPTURE") {
        return;
    }
    eprintln!("{line}");
}

/// The subprocess-runners pair of `docs/host-probing.md`: the three runners the
/// host probes use, which are the summary stack's
/// `capture_optional_command_with_timeout` (this file), the examination stack's
/// `run` over `run_raw` ([`crate::probe_exec`]), and `run_with_timeout`
/// ([`crate::process`]), with the first two now thin adapters over the
/// executor `run_with_timeout` shares.
///
/// Every case runs one stand-in program through all three and pins what each
/// returns. The cases that stayed as the characterisation left them are the
/// adapters keeping their old answers; the ones that moved are the differences
/// the note lists, each named for the answer it now gives. Programs are planted
/// with [`crate::fake_executable::write_fake_executable`] and started by
/// absolute path, so no `PATH` is involved. Each case that leaves a descendant
/// holding a pipe gives it a short, fixed life, so a runner that waits for it is
/// slow and cannot hang the suite.
#[cfg(all(test, unix))]
mod subprocess_runners_pair {
    use super::{candidate_with_debug, capture_optional_command_with_timeout};
    use crate::fake_executable::write_fake_executable;
    use crate::probe_exec::{run, run_utf16le};
    use crate::process::run_with_timeout;
    use std::os::unix::process::ExitStatusExt as _;
    use std::path::Path;
    use std::process::{Command, Output};
    use std::time::{Duration, Instant};

    /// How long a descendant that keeps a pipe open lives. An unbounded reader
    /// join waits this long; a bounded one does not.
    const DESCENDANT_LIFE: &str = "2";

    /// Under this a call did not wait for a [`DESCENDANT_LIFE`] descendant, and
    /// over it a call did, with the margin on both sides for a loaded host.
    const WAITED: Duration = Duration::from_millis(1_500);

    struct Fixture {
        _dir: tempfile::TempDir,
        program: String,
    }

    /// Plant `body` as a shell script and return where it is.
    fn fixture(body: &str) -> Fixture {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("fake-tool");
        write_fake_executable(&path, format!("#!/bin/sh\n{body}\n")).expect("plant the fake");
        let program = path.to_str().expect("a UTF-8 temp path").to_owned();
        Fixture { _dir: dir, program }
    }

    fn timed<T>(call: impl FnOnce() -> T) -> (T, Duration) {
        let started = Instant::now();
        let value = call();
        (value, started.elapsed())
    }

    fn bounded(program: &str, timeout: Duration) -> anyhow::Result<Output> {
        run_with_timeout(Command::new(program), timeout, "process")
    }

    const LONG: Duration = Duration::from_secs(20);

    #[test]
    fn a_program_that_succeeds_is_read_alike_by_all_three() {
        let tool = fixture("printf 'out\\n'; printf 'err\\n' >&2; exit 0");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (0, "out\n".to_owned(), "err\n".to_owned())
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            Some("out\n".to_owned()),
            "the optional capture keeps stdout and discards stderr"
        );
        let output = bounded(&tool.program, LONG).expect("a success is not an error");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"out\n");
        assert_eq!(output.stderr, b"err\n");
    }

    #[test]
    fn a_non_zero_exit_keeps_its_output_for_run_but_reads_as_nothing_to_the_optional_capture() {
        let tool = fixture("printf 'partial\\n'; printf 'bad\\n' >&2; exit 3");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (3, "partial\n".to_owned(), "bad\n".to_owned())
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            None,
            "the optional capture keeps stdout only on success"
        );
        let output = bounded(&tool.program, LONG).expect("a non-zero exit is not an error");
        assert_eq!(output.status.code(), Some(3));
        assert_eq!(output.stdout, b"partial\n");
        assert_eq!(output.stderr, b"bad\n");
    }

    #[test]
    fn a_program_that_cannot_be_started_is_127_empty_none_or_an_error() {
        let missing = "/nonexistent-dir-for-the-runner-pair/tool";

        assert_eq!(run(missing, &[], LONG), (127, String::new(), String::new()));
        assert_eq!(
            capture_optional_command_with_timeout(missing, &[], LONG),
            None
        );
        let error = bounded(missing, LONG).expect_err("a spawn failure is an error here");
        assert!(
            format!("{error:#}").contains("failed to spawn process"),
            "{error:#}"
        );
    }

    #[test]
    fn a_timeout_keeps_partial_output_for_run_and_quotes_it_for_the_bounded_executor() {
        // `exec` so killing the shell at the deadline leaves no descendant on the
        // pipe: this case is about what a timeout returns, not about the join.
        let tool = fixture("printf 'before\\n'; exec sleep 30");
        // Room for a loaded host to start the shell and run its `printf` before
        // the deadline, since the assertion is on what was printed by then.
        let timeout = Duration::from_secs(3);

        assert_eq!(
            run(&tool.program, &[], timeout),
            (124, "before\n".to_owned(), String::new())
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], timeout),
            None
        );
        let error = bounded(&tool.program, timeout).expect_err("a timeout is an error here");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("exceeded 3s timeout"), "{rendered}");
        assert!(rendered.contains("before"), "{rendered}");
    }

    /// The difference the note asks to be pinned first: the child prints `early`
    /// and exits 0, and a descendant prints `late` half a second later, past the
    /// 200 ms the executor gives the readers once the child has gone. All three
    /// now return `early` alone; `run` and the optional capture used to join
    /// their readers and return both lines.
    #[test]
    fn output_written_more_than_the_drain_grace_after_the_child_exits_is_lost_to_all_three() {
        let tool = fixture("printf 'early\\n'; (sleep 0.5; printf 'late\\n') &");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (0, "early\n".to_owned(), String::new())
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            Some("early\n".to_owned())
        );
        let output = bounded(&tool.program, LONG).expect("the child exited 0");
        assert_eq!(output.stdout, b"early\n");
    }

    /// The other side of the grace: a descendant that writes 0.1 s after the
    /// child exits is inside the 200 ms the readers are given, so every runner
    /// still returns both lines. This is what keeps a probe whose tool leaves a
    /// short-lived helper behind from losing its tail.
    #[test]
    fn output_written_within_the_drain_grace_after_the_child_exits_is_kept_by_all_three() {
        let tool = fixture("printf 'early\\n'; (sleep 0.1; printf 'late\\n') &");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (0, "early\nlate\n".to_owned(), String::new())
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            Some("early\nlate\n".to_owned())
        );
        let output = bounded(&tool.program, LONG).expect("the child exited 0");
        assert_eq!(output.stdout, b"early\nlate\n");
    }

    /// A descendant holds both pipes for [`DESCENDANT_LIFE`] seconds after the
    /// child exits 0. `run` and the optional capture used to join their readers
    /// with no limit, so each call lasted as long as the descendant; now none of
    /// the three waits for it, and each still returns what the child wrote.
    #[test]
    fn a_descendant_holding_the_pipe_no_longer_extends_run_or_the_optional_capture() {
        let tool = fixture(&format!("printf 'ready\\n'; sleep {DESCENDANT_LIFE} &"));

        let (answer, elapsed) = timed(|| run(&tool.program, &[], LONG));
        assert_eq!(answer, (0, "ready\n".to_owned(), String::new()));
        assert!(elapsed < WAITED, "run returned after {elapsed:?}");

        let (answer, elapsed) =
            timed(|| capture_optional_command_with_timeout(&tool.program, &[], LONG));
        assert_eq!(answer, Some("ready\n".to_owned()));
        assert!(
            elapsed < WAITED,
            "the optional capture returned after {elapsed:?}"
        );

        let (answer, elapsed) = timed(|| bounded(&tool.program, LONG));
        assert_eq!(answer.expect("the child exited 0").stdout, b"ready\n");
        assert!(elapsed < WAITED, "the executor returned after {elapsed:?}");
    }

    /// The same descendant behind a child that is killed at its deadline: the
    /// timeout used to bound nothing for the two older runners (issue 623), and
    /// now bounds all three. `run` keeps the partial output and its `124`.
    #[test]
    fn a_descendant_holding_the_pipe_no_longer_outlasts_the_timeout_of_run_or_the_optional_capture()
    {
        let tool = fixture(&format!(
            "printf 'start\\n'; sleep {DESCENDANT_LIFE} & wait"
        ));
        let timeout = Duration::from_millis(500);

        let (answer, elapsed) = timed(|| run(&tool.program, &[], timeout));
        // The rc and the time are the point here; what was printed before the
        // deadline is `a_timeout_keeps_partial_output_for_run_...`'s, with room
        // for a loaded host that this short deadline does not leave.
        assert_eq!(answer.0, 124);
        assert!(elapsed < WAITED, "run returned after {elapsed:?}");

        let (answer, elapsed) =
            timed(|| capture_optional_command_with_timeout(&tool.program, &[], timeout));
        assert_eq!(answer, None);
        assert!(
            elapsed < WAITED,
            "the optional capture returned after {elapsed:?}"
        );

        let (answer, elapsed) = timed(|| bounded(&tool.program, timeout));
        assert!(answer.is_err(), "a timeout is an error here");
        assert!(elapsed < WAITED, "the executor returned after {elapsed:?}");
    }

    /// What the optional capture says on its debug channel for the same two
    /// runs: a descendant that writes after the grace gives a short answer and a
    /// line saying so, and a run that finished cleanly gives no line. The answer
    /// and the line are read from one call, so the line cannot be claiming a
    /// shortfall the answer does not have.
    #[test]
    fn a_short_answer_from_the_optional_capture_comes_with_a_debug_line_and_a_whole_one_does_not() {
        let late = fixture("printf 'early\\n'; (sleep 0.5; printf 'late\\n') &");
        let mut lines = Vec::new();
        let answer = candidate_with_debug(Path::new(&late.program), &[], LONG, &mut |line| {
            lines.push(line);
        });
        assert_eq!(answer, Some("early\n".to_owned()));
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].starts_with("rocm debug: command capture output from ")
                && lines[0].contains(&late.program)
                && lines[0].contains("may be incomplete: output was still arriving"),
            "{}",
            lines[0]
        );

        let whole = fixture("printf 'early\\n'; (sleep 0.1; printf 'late\\n') &");
        let mut lines = Vec::new();
        let answer = candidate_with_debug(Path::new(&whole.program), &[], LONG, &mut |line| {
            lines.push(line);
        });
        assert_eq!(answer, Some("early\nlate\n".to_owned()));
        assert!(lines.is_empty(), "{lines:?}");
    }

    /// A run that failed is described by the failure line alone, not also as an
    /// incomplete answer: the second line would add nothing to the first.
    #[test]
    fn a_failed_run_gets_its_failure_line_and_no_incomplete_line() {
        let tool = fixture("printf 'partial\\n'; (sleep 0.5; printf 'late\\n') & exit 3");
        let mut lines = Vec::new();
        let answer = candidate_with_debug(Path::new(&tool.program), &[], LONG, &mut |line| {
            lines.push(line);
        });
        assert_eq!(answer, None);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].starts_with("rocm debug: command capture exit failed for "),
            "{}",
            lines[0]
        );
    }

    /// A descendant that holds only stderr. The optional capture used to send
    /// stderr to the null device, so such a descendant held nothing of its and
    /// the call returned at once; it now pipes stderr like the others and gives
    /// the descendant the drain grace. The answer is unchanged. The extra cost,
    /// up to the 200 ms grace, is too small for a timing assertion to tell from
    /// none, so the assertion is only that the call does not wait out the
    /// descendant's life. `run` pipes stderr as it always did, and used to wait
    /// the descendant out and return its late line.
    #[test]
    fn a_descendant_holding_only_stderr_costs_the_optional_capture_at_most_the_drain_grace() {
        let tool = fixture(&format!(
            "printf 'x\\n'; (exec >/dev/null; sleep {DESCENDANT_LIFE}; printf 'y\\n' >&2) &"
        ));

        let (answer, elapsed) =
            timed(|| capture_optional_command_with_timeout(&tool.program, &[], LONG));
        assert_eq!(answer, Some("x\n".to_owned()));
        assert!(elapsed < WAITED, "returned after {elapsed:?}");

        let (answer, elapsed) = timed(|| run(&tool.program, &[], LONG));
        assert_eq!(answer, (0, "x\n".to_owned(), String::new()));
        assert!(elapsed < WAITED, "run returned after {elapsed:?}");
    }

    #[test]
    fn output_that_is_not_utf8_is_lossy_for_run_none_for_the_optional_capture_and_raw_for_the_executor()
     {
        let tool = fixture("printf 'ok\\377done\\n'");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (0, "ok\u{FFFD}done\n".to_owned(), String::new())
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            None,
            "strict UTF-8: a stray byte reads as no answer"
        );
        let output = bounded(&tool.program, LONG).expect("the child exited 0");
        assert_eq!(output.stdout, b"ok\xffdone\n");
    }

    #[test]
    fn utf16le_output_is_decoded_by_declaration_from_the_raw_bytes() {
        // `h` `i` as UTF-16LE, preceded by the byte order mark.
        let tool = fixture("printf '\\377\\376h\\000i\\000'");

        assert_eq!(run_utf16le(&tool.program, &[], LONG), (0, "hi".to_owned()));
    }

    #[test]
    fn a_program_killed_by_a_signal_has_no_exit_code() {
        let tool = fixture("printf 'partial\\n'; kill -9 $$");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (-1, "partial\n".to_owned(), String::new()),
            "run reports -1 where the status has no code"
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            None
        );
        let output = bounded(&tool.program, LONG).expect("a signal death is not an error");
        assert_eq!(output.status.code(), None);
        assert_eq!(output.status.signal(), Some(9));
        assert_eq!(output.stdout, b"partial\n");
    }
}

/// What the optional capture makes of each way a run can end. Runs on every
/// host, over captures built by hand, so the Windows job checks the translation
/// the real-process cases above cannot drive there.
#[cfg(test)]
mod optional_stdout_tests {
    use super::{CommandCapture, CommandOutcome, incomplete_output_note, optional_stdout};

    fn capture(outcome: CommandOutcome, stdout: &[u8], detail: Option<&str>) -> CommandCapture {
        CommandCapture::from_run(
            vec!["tool".to_owned()],
            outcome,
            None,
            stdout.to_vec(),
            b"ignored".to_vec(),
            detail.map(str::to_owned),
        )
    }

    #[test]
    fn a_program_that_exited_0_gives_its_stdout_and_nothing_else() {
        assert_eq!(
            optional_stdout(capture(CommandOutcome::Exited(0), b"out\n", None)),
            Ok(Some("out\n".to_owned()))
        );
    }

    #[test]
    fn a_program_that_exited_0_with_output_that_is_not_utf8_gives_no_answer_and_no_debug_line() {
        assert_eq!(
            optional_stdout(capture(CommandOutcome::Exited(0), b"ok\xffdone", None)),
            Ok(None)
        );
    }

    #[test]
    fn a_pipe_that_failed_to_read_does_not_change_the_answer() {
        // Kept as it was: the bytes that arrived are used, and the error is not
        // an answer of its own.
        assert_eq!(
            optional_stdout(capture(
                CommandOutcome::Exited(0),
                b"partial",
                Some("failed to read the output: broken pipe")
            )),
            Ok(Some("partial".to_owned()))
        );
    }

    #[test]
    fn every_other_way_a_run_ends_is_no_answer_with_its_stage_named() {
        let cases = [
            (CommandOutcome::Exited(3), "exit"),
            (CommandOutcome::Signaled(Some(9)), "exit"),
            (CommandOutcome::TimedOut, "timeout"),
            (CommandOutcome::WaitFailed, "wait"),
            (CommandOutcome::NotFound, "spawn"),
            (
                CommandOutcome::SpawnFailed(std::io::ErrorKind::PermissionDenied),
                "spawn",
            ),
        ];
        for (outcome, stage) in cases {
            // Stdout the child wrote before ending badly is not an answer.
            let result = optional_stdout(capture(outcome, b"partial", Some("the os said no")));
            assert_eq!(
                result.map_err(|(stage, _)| stage),
                Err(stage),
                "{outcome:?}"
            );
        }
    }

    #[test]
    fn only_a_run_that_exited_0_with_something_cutting_its_output_short_is_called_incomplete() {
        let note = "output was still arriving when the drain grace ended";
        assert_eq!(
            incomplete_output_note(&capture(CommandOutcome::Exited(0), b"x", Some(note))),
            Some(note)
        );
        assert_eq!(
            incomplete_output_note(&capture(CommandOutcome::Exited(0), b"x", None)),
            None,
            "a whole capture has nothing to report"
        );
        // Every other ending already has a failure line of its own.
        for outcome in [
            CommandOutcome::Exited(3),
            CommandOutcome::Signaled(Some(9)),
            CommandOutcome::TimedOut,
            CommandOutcome::WaitFailed,
            CommandOutcome::NotFound,
        ] {
            assert_eq!(
                incomplete_output_note(&capture(outcome, b"x", Some(note))),
                None,
                "{outcome:?}"
            );
        }
    }

    #[test]
    fn a_spawn_failure_reports_the_error_the_os_gave() {
        let (stage, detail) = optional_stdout(capture(
            CommandOutcome::NotFound,
            b"",
            Some("No such file or directory (os error 2)"),
        ))
        .expect_err("a program that cannot be started is no answer");

        assert_eq!(stage, "spawn");
        assert_eq!(detail, "No such file or directory (os error 2)");
    }
}
