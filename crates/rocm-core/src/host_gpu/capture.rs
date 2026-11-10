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

use crate::probe_exec::tool_path_candidates;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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

fn capture_optional_command_candidate_with_timeout(
    program: &Path,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            debug_command_capture_failure(program, "spawn", &error.to_string());
            return None;
        }
    };
    let mut stdout_reader = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        })
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = stdout_reader
                    .take()
                    .map(|reader| reader.join().unwrap_or_default())
                    .unwrap_or_default();
                if status.success() {
                    return String::from_utf8(bytes).ok();
                }
                debug_command_capture_failure(program, "exit", &format!("status {status}"));
                return None;
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "timeout", "timed out");
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "wait", "failed to wait");
                return None;
            }
        }
    }
}

fn debug_command_capture_failure(program: &Path, stage: &str, detail: &str) {
    if !crate::runtime::env_flag("ROCM_CLI_DEBUG_COMMAND_CAPTURE") {
        return;
    }
    eprintln!(
        "rocm debug: command capture {stage} failed for {}: {detail}",
        program.display()
    );
}

/// Characterisation of the three subprocess runners the host probes use, for the
/// convergence of the subprocess-runners pair of `docs/host-probing.md`: the
/// summary stack's `capture_optional_command_with_timeout` (this file), the
/// examination stack's `run` over `run_raw` ([`crate::probe_exec`]), and
/// `run_with_timeout` ([`crate::process`]) the other two are to be thin adapters
/// over.
///
/// Every case runs one stand-in program through all three and pins what each
/// returns, so where they differ the difference is written down as an assertion
/// rather than a claim. Programs are planted with
/// [`crate::fake_executable::write_fake_executable`] and started by absolute
/// path, so no `PATH` is involved. Each case that leaves a descendant holding a
/// pipe gives it a short, fixed life, so a runner that waits for it is slow and
/// cannot hang the suite.
#[cfg(all(test, unix))]
mod subprocess_runners_pair {
    use super::capture_optional_command_with_timeout;
    use crate::fake_executable::write_fake_executable;
    use crate::probe_exec::{run, run_utf16le};
    use crate::process::run_with_timeout;
    use std::os::unix::process::ExitStatusExt as _;
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
        let timeout = Duration::from_secs(1);

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
        assert!(rendered.contains("exceeded 1s timeout"), "{rendered}");
        assert!(rendered.contains("before"), "{rendered}");
    }

    /// The difference the note asks to be pinned first: the child prints `early`
    /// and exits 0, and a descendant prints `late` half a second later, past the
    /// 200 ms the bounded executor gives the readers once the child has gone.
    #[test]
    fn output_written_after_the_child_exits_is_kept_by_run_and_the_optional_capture_only() {
        let tool = fixture("printf 'early\\n'; (sleep 0.5; printf 'late\\n') &");

        assert_eq!(
            run(&tool.program, &[], LONG),
            (0, "early\nlate\n".to_owned(), String::new()),
            "run joins its readers, so it waits for the descendant's output"
        );
        assert_eq!(
            capture_optional_command_with_timeout(&tool.program, &[], LONG),
            Some("early\nlate\n".to_owned())
        );
        let output = bounded(&tool.program, LONG).expect("the child exited 0");
        assert_eq!(
            output.stdout, b"early\n",
            "the bounded executor abandons its readers after the drain grace"
        );
    }

    /// A descendant holds both pipes for [`DESCENDANT_LIFE`] seconds after the
    /// child exits 0. The readers of `run` and the optional capture are joined
    /// with no limit, so each call lasts as long as the descendant.
    #[test]
    fn a_descendant_holding_the_pipe_extends_run_and_the_optional_capture_but_not_the_executor() {
        let tool = fixture(&format!("printf 'ready\\n'; sleep {DESCENDANT_LIFE} &"));

        let (answer, elapsed) = timed(|| run(&tool.program, &[], LONG));
        assert_eq!(answer, (0, "ready\n".to_owned(), String::new()));
        assert!(elapsed >= WAITED, "run returned after {elapsed:?}");

        let (answer, elapsed) =
            timed(|| capture_optional_command_with_timeout(&tool.program, &[], LONG));
        assert_eq!(answer, Some("ready\n".to_owned()));
        assert!(
            elapsed >= WAITED,
            "the optional capture returned after {elapsed:?}"
        );

        let (answer, elapsed) = timed(|| bounded(&tool.program, LONG));
        assert_eq!(answer.expect("the child exited 0").stdout, b"ready\n");
        assert!(elapsed < WAITED, "the executor returned after {elapsed:?}");
    }

    /// The same descendant behind a child that is killed at its deadline: the
    /// timeout bounds nothing for the two older runners, which is issue 623.
    #[test]
    fn a_descendant_holding_the_pipe_outlasts_the_timeout_of_run_and_the_optional_capture() {
        let tool = fixture(&format!(
            "printf 'start\\n'; sleep {DESCENDANT_LIFE} & wait"
        ));
        let timeout = Duration::from_millis(500);

        let (answer, elapsed) = timed(|| run(&tool.program, &[], timeout));
        assert_eq!(answer, (124, "start\n".to_owned(), String::new()));
        assert!(elapsed >= WAITED, "run returned after {elapsed:?}");

        let (answer, elapsed) =
            timed(|| capture_optional_command_with_timeout(&tool.program, &[], timeout));
        assert_eq!(answer, None);
        assert!(
            elapsed >= WAITED,
            "the optional capture returned after {elapsed:?}"
        );

        let (answer, elapsed) = timed(|| bounded(&tool.program, timeout));
        assert!(answer.is_err(), "a timeout is an error here");
        assert!(elapsed < WAITED, "the executor returned after {elapsed:?}");
    }

    /// A descendant that holds only stderr. The optional capture sends stderr to
    /// the null device, so such a descendant never held anything of its.
    #[test]
    fn a_descendant_holding_only_stderr_does_not_hold_the_optional_capture() {
        let tool = fixture(&format!(
            "printf 'x\\n'; (exec >/dev/null; sleep {DESCENDANT_LIFE}; printf 'y\\n' >&2) &"
        ));

        let (answer, elapsed) =
            timed(|| capture_optional_command_with_timeout(&tool.program, &[], LONG));
        assert_eq!(answer, Some("x\n".to_owned()));
        assert!(elapsed < WAITED, "returned after {elapsed:?}");

        // `run` pipes stderr too, so the same descendant holds it for the life
        // of the descendant.
        let (answer, elapsed) = timed(|| run(&tool.program, &[], LONG));
        assert_eq!(answer, (0, "x\n".to_owned(), "y\n".to_owned()));
        assert!(elapsed >= WAITED, "run returned after {elapsed:?}");
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
