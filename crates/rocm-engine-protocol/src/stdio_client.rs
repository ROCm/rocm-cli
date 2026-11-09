// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The client side of the engine stdio protocol: run an engine as a child
//! process, send it one [`EngineRequestEnvelope`] on stdin as a single line of
//! JSON, and read one [`EngineResponseEnvelope`] back from stdout.
//!
//! `rocm` and `rocmd` both talk to engines this way and share this one
//! implementation. They differ in *which* process they start (`rocm` runs the
//! resolved engine binary with `stdio`, or answers a built-in engine in its own
//! process without a child at all; `rocmd` runs `rocm __engine-stdio <engine>`),
//! and that stays with each caller. What is shared is everything that follows
//! the choice of program: the request framing, the environment, how stderr is
//! handled, the write guard against SIGPIPE, what a failing exit status or an
//! unusable answer is reported as, and decoding the response. The few things
//! the callers want differently are the fields of [`EngineStdioOptions`].
//!
//! The request and the child's lifetime are not bounded by a timeout: an engine
//! install legitimately runs for minutes, and no caller has ever set one.
//!
//! Nor is the wait for the pipes after the child has exited, and that is a known
//! limit rather than an oversight. The answer is read to the end of the child's
//! stdout, and the stderr collector is joined, so an engine that leaves a
//! grandchild holding either pipe open makes the request wait for that grandchild
//! to exit. Both clients this one replaces behaved the same way (`rocm` read
//! stdout to the end then joined its stderr thread; `rocmd` used
//! `wait_with_output`), so nothing new is exposed to it, but the daemon is long
//! lived and a hung request holds its caller. Bounding it exactly would mean
//! reading both pipes on their own threads, waiting for the child itself without a
//! limit and for the pipes only up to a deadline after it exits, and choosing what
//! to report when the deadline passes (an answer already complete? a partial
//! one?). That is a timeout policy for an engine contract nothing here sets, so
//! this change leaves it to be decided on its own. The one wait this client adds
//! itself, for the stderr of a child whose request could not be written, is
//! bounded by `EXITED_STDERR_GRACE`.
//!
//! A request that cannot be written is the other place the answer-wins rule does
//! not apply: if the engine answered and exited without draining a request larger
//! than a pipe buffer, the write fails, the child is killed and its stdout is
//! never read, so the error carries its stderr and not its answer. Every engine
//! here (`lemonade`, `vllm`, `aim`, and the `rocm __engine-stdio` verb that serves
//! the built-in ones) reads stdin to the end before it answers, so reaching this
//! needs an engine that breaks the protocol. Decoding such an answer would mean
//! waiting, up to a deadline, for the child to finish printing it and to exit
//! before deciding it is complete, and an engine that has stopped reading is no
//! engine to wait for, so it is left as a documented limit.
//!
//! Forwarding the child's stderr guards each write against SIGPIPE on its own
//! (see `forward_best_effort`), and the guard is held for as long as that write
//! blocks. If the sink is stalled rather than closed (a terminal that has stopped
//! reading), SIGPIPE therefore stays ignored process-wide for that time, and a
//! concurrent `print!` to a closed pipe on another thread fails with an error
//! (which `print!` turns into a panic) instead of ending the process quietly. The
//! guard is not held across the reads between lines, so this needs a stalled
//! sink, not just a quiet engine. It is accepted: a guard that could be taken
//! back mid-write does not exist, and without one a closed sink would end
//! `rocm` before the request could report why.

use std::ffi::OsStr;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ENGINE_ENVS_ROOT_ENV, with_sigpipe_ignored};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{
    ENDPOINT_API_KEY_FILE_ENV, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope,
};

/// Set to `1` on an engine child whose caller wants its progress lines on
/// stderr as they happen (an engine install).
pub const ENGINE_PROGRESS_STDERR_ENV: &str = "ROCM_ENGINE_PROGRESS_STDERR";

/// What a caller wants different about the child [`run_engine_stdio`] starts.
///
/// The default asks for nothing beyond the request itself: the child inherits
/// the caller's environment and its stderr is only collected.
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineStdioOptions<'a> {
    /// Run the child on these roots, by setting the `ROCM_CLI_*_DIR` variables
    /// ([`AppPaths::env_vars`]). `None` leaves the child to inherit whatever the
    /// caller's own environment says.
    pub app_paths: Option<&'a AppPaths>,
    /// The root engine virtualenvs go under for this request, set as
    /// [`ENGINE_ENVS_ROOT_ENV`]. Only an install into a particular root needs it.
    pub env_root: Option<&'a Path>,
    /// A key file to hand the engine through [`ENDPOINT_API_KEY_FILE_ENV`], so
    /// its HTTP probe of a protected endpoint authenticates. The caller decides
    /// whether a file holds a usable key; this only carries it.
    pub endpoint_key_file: Option<&'a Path>,
    /// Ask the engine for progress on stderr ([`ENGINE_PROGRESS_STDERR_ENV`]).
    pub request_progress: bool,
    /// Write the child's stderr to this process's stderr as it arrives, so a
    /// terminal user watches an install. Either way the whole of it is kept and
    /// shown in the error when the request fails.
    pub forward_stderr: bool,
}

/// Build the request envelope for `method` from `request`.
pub fn engine_request_envelope<T: Serialize>(
    method: EngineMethod,
    request: &T,
) -> Result<EngineRequestEnvelope> {
    Ok(EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request)
            .context("failed to encode engine request payload")?,
    })
}

/// The command [`run_engine_stdio`] runs: `program` with `args`, all three
/// standard streams piped, and the environment `options` asks for. Separate so
/// a test can read the command line and environment without starting anything.
fn engine_stdio_command<I, S>(program: &Path, args: I, options: &EngineStdioOptions<'_>) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(paths) = options.app_paths {
        for (key, value) in paths.env_vars() {
            command.env(key, value);
        }
    }
    if let Some(env_root) = options.env_root {
        command.env(ENGINE_ENVS_ROOT_ENV, env_root);
    }
    if let Some(key_file) = options.endpoint_key_file {
        command.env(ENDPOINT_API_KEY_FILE_ENV, key_file);
    }
    if options.request_progress {
        command.env(ENGINE_PROGRESS_STDERR_ENV, "1");
    }
    command
}

/// Send `envelope` to the engine child `program args…` and return what it
/// answered.
///
/// A failing exit status is an error only when the child printed no answer: an
/// engine that reports its own failure in an error envelope and then exits
/// non-zero is read as that envelope, which says more than the status does. The
/// same holds for a success envelope (`ok: true`): a child that answers
/// successfully and then exits non-zero is reported as the success it answered,
/// and the failing exit is not reported or logged anywhere, by design, because
/// the answer is what the caller asked for.
/// Every other failure names what went wrong and carries the child's stderr.
pub fn run_engine_stdio<I, S>(
    program: &Path,
    args: I,
    options: &EngineStdioOptions<'_>,
    envelope: &EngineRequestEnvelope,
) -> Result<EngineResponseEnvelope>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run_engine_stdio_forwarding(
        program,
        args,
        options,
        envelope,
        options.forward_stderr.then(io::stderr),
    )
}

/// [`run_engine_stdio`] with the sink the child's stderr is forwarded to chosen
/// by the caller (`None` forwards nothing), so a test can see what is forwarded
/// without capturing this process's own stderr. [`run_engine_stdio`] is the one
/// place the `forward_stderr` option becomes a sink.
fn run_engine_stdio_forwarding<I, S>(
    program: &Path,
    args: I,
    options: &EngineStdioOptions<'_>,
    envelope: &EngineRequestEnvelope,
    forward: Option<impl Write + Send + 'static>,
) -> Result<EngineResponseEnvelope>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut child = engine_stdio_command(program, args, options)
        .spawn()
        .with_context(|| format!("failed to spawn engine stdio process {}", program.display()))?;

    let mut stdin = child
        .stdin
        .take()
        .context("engine stdio child did not expose stdin")?;
    // Ignore SIGPIPE for the duration of the write. Under the default action, an
    // engine that exits before reading its stdin would kill this process with
    // SIGPIPE (`rocm` runs that way, and so does the daemon hosted in it);
    // ignored, the write returns EPIPE and fails as an ordinary error.
    let written = with_sigpipe_ignored(|| {
        serde_json::to_writer(&mut stdin, envelope).map_err(io::Error::from)?;
        stdin.write_all(b"\n")
    });
    if let Err(error) = written {
        return Err(abandon_unwritten_request(
            child,
            &error,
            EXITED_STDERR_GRACE,
        ));
    }
    // End of the request: the engine reads to EOF before it answers.
    drop(stdin);

    let stderr_reader = child
        .stderr
        .take()
        .map(|stderr| thread::spawn(move || collect_stderr(stderr, forward)));
    let stdout_pipe = child
        .stdout
        .take()
        .context("engine stdio child did not expose stdout")?;
    let stdout = read_response(&mut child, stdout_pipe)?;
    let status = child
        .wait()
        .context("failed waiting for engine stdio response")?;
    let stderr = stderr_reader
        .map(|reader| reader.join().unwrap_or_default())
        .unwrap_or_default();

    let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
    if !status.success() && stdout.is_empty() {
        let exited = exit_note(status);
        if stderr.is_empty() {
            bail!("{exited}");
        }
        bail!("{exited}: {stderr}");
    }
    serde_json::from_slice(&stdout).with_context(|| unparseable_answer_message(status, &stderr))
}

/// How a failing exit is worded wherever the status is reported. Built from the
/// exit code (or the signal that ended the process) rather than from the status's
/// own `Display`, which already says "exit status: N" on Unix and "exit code: N"
/// on Windows.
fn exit_note(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("engine stdio process exited with code {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        if let Some(signal) = status.signal() {
            return format!("engine stdio process was terminated by signal {signal}");
        }
    }
    "engine stdio process ended without an exit code".to_owned()
}

/// The error for an answer that is not a response envelope. A child that also
/// exited with a failing status says so, because that is often why the answer is
/// cut short; the stderr it wrote follows when there is any.
fn unparseable_answer_message(status: ExitStatus, stderr: &str) -> String {
    let mut message = "failed to parse engine response envelope".to_owned();
    if !status.success() {
        message = format!("{message} ({})", exit_note(status));
    }
    if !stderr.is_empty() {
        message = format!("{message}; stderr: {stderr}");
    }
    message
}

/// How long the stderr of a child that was killed, or had exited, is waited for
/// when its request could not be written.
const EXITED_STDERR_GRACE: Duration = Duration::from_secs(1);

/// The request could not be written to `child`'s stdin, so it will never answer:
/// do not leave it running or unreaped (a long-lived caller would collect one
/// zombie per failure), and report why it was not listening.
///
/// The child is killed (nothing to do for one that has already exited) and
/// waited for, then whatever it wrote to stderr is added to the error: an engine
/// that exits at once usually says why there, and that is the explanation the
/// caller wants. The kill does not wait on `try_wait` showing an exit first,
/// because the pipe closes while the process is still exiting, so a check made
/// right after the failed write can still see it running. The read is bounded
/// by `grace`: a grandchild that inherited the pipe keeps it open past the exit,
/// and this error path must not block on it.
fn abandon_unwritten_request(
    mut child: Child,
    write_error: &io::Error,
    grace: Duration,
) -> anyhow::Error {
    let _ = child.kill();
    let _ = child.wait();
    let stderr = child
        .stderr
        .take()
        .map(|stderr| read_available(stderr, grace))
        .unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
    let mut message = format!("failed to write engine request: {write_error}");
    if !stderr.is_empty() {
        message = format!("{message}; stderr: {stderr}");
    }
    anyhow::anyhow!(message)
}

/// Read `source` to its end, or for at most `grace` if it does not end, and
/// return what arrived. The reading thread is left to finish by itself when the
/// pipe closes.
fn read_available(mut source: impl Read + Send + 'static, grace: Duration) -> Vec<u8> {
    let (sender, receiver) = mpsc::channel::<Vec<u8>>();
    thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        loop {
            match source.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if sender.send(buffer[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    collect_until(&receiver, Instant::now() + grace)
}

/// Gather what arrives on `receiver` until it closes or `deadline` passes.
///
/// The deadline is compared with the clock before every receive, and nothing is
/// received once it has passed: `recv_timeout` with an expired timeout still
/// hands back a message that is already queued, so a backlog, or a descendant
/// that keeps writing, would otherwise keep this collecting past the bound.
fn collect_until(receiver: &mpsc::Receiver<Vec<u8>>, deadline: Instant) -> Vec<u8> {
    let mut collected = Vec::new();
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match receiver.recv_timeout(deadline - now) {
            Ok(chunk) => collected.extend(chunk),
            Err(_) => break,
        }
    }
    collected
}

/// Read the child's whole answer from `stdout`.
///
/// A failed read would leave the child running with nobody waiting on it, and
/// this client lives in the long-lived daemon, so the child is killed and
/// reaped before the error is returned, as it is when the request cannot be
/// written. The stderr collector is not joined on that path: a grandchild that
/// inherited the pipe could hold it open, and the thread ends by itself once
/// the pipe closes.
fn read_response(child: &mut Child, mut stdout: impl Read) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    if let Err(error) = stdout.read_to_end(&mut response) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(anyhow::Error::new(error).context("failed to read engine stdio response"));
    }
    Ok(response)
}

/// Read `stderr` to the end line by line, optionally writing each line to
/// `forward` (this process's stderr, in production) as it arrives, and return
/// everything read. A failed forward is ignored: a closed terminal must not
/// cost the request its answer. That holds for the signal as well as the error:
/// under SIGPIPE's default action (`rocm` runs that way) a write to a closed
/// pipe ends the whole process before `write_all` can return, so each forward
/// runs inside [`with_sigpipe_ignored`] (see [`forward_best_effort`]). A read error ends the
/// collection and is itself recorded as the last line.
fn collect_stderr(stderr: impl Read, mut forward: Option<impl Write>) -> Vec<u8> {
    let mut reader = io::BufReader::new(stderr);
    let mut collected = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => break,
            Ok(_) => {
                if let Some(sink) = forward.as_mut() {
                    forward_best_effort(sink, &line);
                }
                collected.extend_from_slice(&line);
            }
            Err(error) => {
                let message = format!("failed to read engine progress: {error}\n");
                if let Some(sink) = forward.as_mut() {
                    forward_best_effort(sink, message.as_bytes());
                }
                collected.extend_from_slice(message.as_bytes());
                break;
            }
        }
    }
    collected
}

/// Write `bytes` to the forwarding sink and drop the result: a sink that has
/// gone away must not cost the request its answer. SIGPIPE is ignored for the
/// write, so a closed pipe comes back as the error that is then discarded
/// rather than as a signal that kills the process.
fn forward_best_effort(sink: &mut impl Write, bytes: &[u8]) {
    let _ = with_sigpipe_ignored(|| sink.write_all(bytes));
}

/// Turn an engine's answer into the response type the caller asked for: the
/// engine's own error when it reported one, otherwise the decoded `data`.
pub fn decode_engine_response<R>(envelope: EngineResponseEnvelope) -> Result<R>
where
    R: DeserializeOwned,
{
    if !envelope.ok {
        let error = envelope.error.map_or_else(
            || "unknown engine error".to_owned(),
            |value| format!("{}: {}", value.code, value.message),
        );
        bail!("{error}");
    }
    let data = envelope
        .data
        .context("engine response envelope did not contain data")?;
    serde_json::from_value(data).context("failed to decode engine response payload")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineErrorDetail, StopResponse};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    /// The environment `command` would add to its child, as plain strings.
    fn added_env(command: &Command) -> BTreeMap<String, String> {
        command
            .get_envs()
            .filter_map(|(key, value)| {
                Some((
                    key.to_string_lossy().into_owned(),
                    value?.to_string_lossy().into_owned(),
                ))
            })
            .collect()
    }

    fn args_of(command: &Command) -> Vec<String> {
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn the_default_options_add_nothing_to_the_childs_environment() {
        let command = engine_stdio_command(
            Path::new("/an/engine"),
            ["stdio"],
            &EngineStdioOptions::default(),
        );
        assert_eq!(command.get_program(), "/an/engine");
        assert_eq!(args_of(&command), ["stdio"]);
        assert!(added_env(&command).is_empty(), "{:?}", added_env(&command));
    }

    #[test]
    fn each_option_sets_exactly_its_own_variable() {
        let paths = AppPaths {
            config_dir: "/c".into(),
            data_dir: "/d".into(),
            cache_dir: "/k".into(),
        };
        let command = engine_stdio_command(
            Path::new("/an/engine"),
            ["__engine-stdio", "vllm"],
            &EngineStdioOptions {
                app_paths: Some(&paths),
                env_root: Some(Path::new("/envs")),
                endpoint_key_file: Some(Path::new("/keys/svc.endpoint-key")),
                request_progress: true,
                forward_stderr: true,
            },
        );
        assert_eq!(args_of(&command), ["__engine-stdio", "vllm"]);
        assert_eq!(
            added_env(&command),
            BTreeMap::from(
                [
                    ("ROCM_CLI_CONFIG_DIR", "/c"),
                    ("ROCM_CLI_DATA_DIR", "/d"),
                    ("ROCM_CLI_CACHE_DIR", "/k"),
                    ("ROCM_CLI_ENGINE_ENVS_ROOT", "/envs"),
                    ("ROCM_SERVE_API_KEY_FILE", "/keys/svc.endpoint-key"),
                    ("ROCM_ENGINE_PROGRESS_STDERR", "1"),
                ]
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
            )
        );

        for (name, options) in [
            (
                "ROCM_CLI_ENGINE_ENVS_ROOT",
                EngineStdioOptions {
                    env_root: Some(Path::new("/envs")),
                    ..EngineStdioOptions::default()
                },
            ),
            (
                "ROCM_SERVE_API_KEY_FILE",
                EngineStdioOptions {
                    endpoint_key_file: Some(Path::new("/keys/k")),
                    ..EngineStdioOptions::default()
                },
            ),
            (
                "ROCM_ENGINE_PROGRESS_STDERR",
                EngineStdioOptions {
                    request_progress: true,
                    ..EngineStdioOptions::default()
                },
            ),
        ] {
            let command = engine_stdio_command(Path::new("/an/engine"), ["stdio"], &options);
            let env = added_env(&command);
            assert_eq!(
                env.keys().map(String::as_str).collect::<Vec<_>>(),
                [name],
                "an option must set only its own variable"
            );
        }
    }

    #[test]
    fn the_variable_names_are_the_ones_the_engines_read() {
        assert_eq!(ENGINE_ENVS_ROOT_ENV, "ROCM_CLI_ENGINE_ENVS_ROOT");
        assert_eq!(ENDPOINT_API_KEY_FILE_ENV, "ROCM_SERVE_API_KEY_FILE");
        assert_eq!(ENGINE_PROGRESS_STDERR_ENV, "ROCM_ENGINE_PROGRESS_STDERR");
    }

    #[test]
    fn the_request_envelope_is_the_method_and_the_encoded_payload() {
        let envelope =
            engine_request_envelope(EngineMethod::Stop, &json!({"service_id": "svc"})).unwrap();
        assert_eq!(envelope.method, EngineMethod::Stop);
        assert_eq!(envelope.payload, json!({"service_id": "svc"}));
        assert_eq!(
            serde_json::to_string(&envelope).unwrap(),
            r#"{"method":"stop","payload":{"service_id":"svc"}}"#
        );
    }

    #[test]
    fn decode_engine_response_reports_each_way_an_answer_can_be_unusable() {
        let failed = EngineResponseEnvelope {
            ok: false,
            data: None,
            error: Some(EngineErrorDetail {
                code: "stop_failed".to_owned(),
                message: "it broke".to_owned(),
            }),
        };
        let error = decode_engine_response::<Value>(failed).unwrap_err();
        assert_eq!(error.to_string(), "stop_failed: it broke");

        let unexplained = EngineResponseEnvelope {
            ok: false,
            data: None,
            error: None,
        };
        let error = decode_engine_response::<Value>(unexplained).unwrap_err();
        assert_eq!(error.to_string(), "unknown engine error");

        let empty = EngineResponseEnvelope {
            ok: true,
            data: None,
            error: None,
        };
        let error = decode_engine_response::<Value>(empty).unwrap_err();
        assert_eq!(
            error.to_string(),
            "engine response envelope did not contain data"
        );

        let wrong_shape = EngineResponseEnvelope::success(json!({"stopped": "yes"}));
        let error = decode_engine_response::<StopResponse>(wrong_shape).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to decode engine response payload"
        );
    }

    /// A status that is neither an exit nor a signal (a stopped process, on
    /// Unix) has neither a code nor a signal to name, and is reported as that.
    #[cfg(unix)]
    #[test]
    fn a_status_with_no_code_and_no_signal_gets_the_neutral_wording() {
        use std::os::unix::process::ExitStatusExt as _;
        // Stopped by SIGSTOP: WIFSTOPPED, which is neither WIFEXITED nor WIFSIGNALED.
        let stopped = ExitStatus::from_raw(0x137f);
        assert_eq!(stopped.code(), None);
        assert_eq!(stopped.signal(), None);
        assert_eq!(
            exit_note(stopped),
            "engine stdio process ended without an exit code"
        );
    }

    /// A source that does not answer until released, then never stops answering,
    /// and says when it is dropped.
    struct Gated {
        release: mpsc::Receiver<()>,
        dropped: mpsc::Sender<()>,
        released: bool,
    }
    impl Read for Gated {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if !self.released {
                self.release.recv().unwrap();
                self.released = true;
            }
            buffer[..4].copy_from_slice(b"late");
            Ok(4)
        }
    }
    impl Drop for Gated {
        fn drop(&mut self) {
            let _ = self.dropped.send(());
        }
    }

    /// Once the bounded read has given up, the reading thread that is still
    /// blocked is not left spinning: when its source finally answers it finds
    /// nobody listening and ends (it drops the source, which the test sees). The
    /// source never reaches its end, so a thread that kept reading after the
    /// failed send would never end.
    #[test]
    fn the_reader_left_behind_by_a_bounded_read_ends_when_nobody_is_listening() {
        let (release, released) = mpsc::channel();
        let (dropped, dropped_signal) = mpsc::channel();
        let source = Gated {
            release: released,
            dropped,
            released: false,
        };
        let collected = read_available(source, Duration::from_millis(50));
        assert!(collected.is_empty());

        release.send(()).unwrap();
        dropped_signal
            .recv_timeout(Duration::from_secs(10))
            .expect("the abandoned reader thread did not end");
    }

    /// A stderr whose reads fail.
    struct Fails;
    impl Read for Fails {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("pipe went away"))
        }
    }

    /// A sink that hands every write to a closure: one that refuses them is a
    /// closed terminal, one that records them shows what was forwarded.
    struct FnSink<F>(F);
    impl<F: FnMut(&[u8]) -> io::Result<usize>> Write for FnSink<F> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            (self.0)(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_closure_sink_writes_through_and_flushes_nothing() {
        let mut sink = FnSink(|bytes: &[u8]| Ok(bytes.len()));
        assert_eq!(sink.write(b"abc").unwrap(), 3);
        sink.flush().unwrap();
    }

    #[test]
    fn stderr_is_collected_whole_and_a_read_error_is_recorded() {
        let collected = collect_stderr(&b"first\nsecond"[..], None::<Vec<u8>>);
        assert_eq!(collected, b"first\nsecond");

        let collected = String::from_utf8(collect_stderr(Fails, None::<Vec<u8>>)).unwrap();
        assert_eq!(
            collected,
            "failed to read engine progress: pipe went away\n"
        );
    }

    /// Output already queued when the deadline has passed is not collected:
    /// `recv_timeout(Duration::ZERO)` would still return it, one message after
    /// another, for as long as the queue stays non-empty.
    #[test]
    fn collecting_stops_at_the_deadline_even_with_output_queued() {
        let (sender, receiver) = mpsc::channel::<Vec<u8>>();
        for _ in 0..100_000 {
            sender.send(vec![b'x'; 16]).unwrap();
        }
        let started = Instant::now();
        let collected = collect_until(&receiver, Instant::now());
        let after_deadline = collected.len();
        assert_eq!(after_deadline, 0, "collected bytes after the deadline");
        assert!(started.elapsed() < Duration::from_secs(1));

        // Before the deadline, queued output is collected, and a closed channel
        // ends the collection early.
        drop(sender);
        let collected = collect_until(&receiver, Instant::now() + Duration::from_secs(30));
        assert_eq!(collected.len(), 100_000 * 16);
    }

    /// A producer that keeps writing (a descendant holding the pipe) cannot hold
    /// the collection past its deadline. A secondary guard: it is timing-based
    /// and also passes with the loop that let queued output bypass the deadline,
    /// because that producer sleeps between messages. The fix is covered by
    /// `collecting_stops_at_the_deadline_even_with_output_queued` above, which is
    /// deterministic.
    #[test]
    fn collecting_returns_at_the_deadline_while_output_keeps_arriving() {
        let (sender, receiver) = mpsc::channel::<Vec<u8>>();
        let producer = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(3);
            while Instant::now() < until {
                if sender.send(vec![b'y'; 1024]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
        });
        let started = Instant::now();
        let collected = collect_until(&receiver, Instant::now() + Duration::from_millis(100));
        let elapsed = started.elapsed();
        drop(receiver);
        producer.join().unwrap();
        assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
        assert!(!collected.is_empty(), "what arrived in time is kept");
    }

    #[test]
    fn forwarded_stderr_is_written_out_and_still_collected() {
        let mut shown = Vec::new();
        let collected = collect_stderr(&b"step 1\nstep 2\n"[..], Some(&mut shown));
        assert_eq!(shown, b"step 1\nstep 2\n");
        assert_eq!(collected, b"step 1\nstep 2\n");

        // A sink that refuses writes (a closed terminal) costs nothing.
        let collected = collect_stderr(
            &b"still kept\n"[..],
            Some(FnSink(|_: &[u8]| {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            })),
        );
        assert_eq!(collected, b"still kept\n");
    }

    #[cfg(unix)]
    mod child {
        use super::*;
        use crate::fake_engine::FakeEngine;
        use std::os::unix::process::ExitStatusExt as _;

        const STOPPED: &str = r#"{"ok":true,"data":{"stopped":true}}"#;

        fn stop_request() -> EngineRequestEnvelope {
            engine_request_envelope(EngineMethod::Stop, &json!({"service_id": "svc"})).unwrap()
        }

        fn run(engine: &FakeEngine, options: &EngineStdioOptions<'_>) -> Result<Value> {
            decode_engine_response(run_engine_stdio(
                engine.path(),
                ["stdio"],
                options,
                &stop_request(),
            )?)
        }

        #[test]
        fn a_request_is_one_newline_terminated_line_on_the_childs_stdin() {
            let engine = FakeEngine::answering("client-framing", STOPPED, 0);
            let answer = run(&engine, &EngineStdioOptions::default()).unwrap();
            assert_eq!(answer, json!({"stopped": true}));
            assert_eq!(engine.argv(), ["stdio"]);
            assert_eq!(
                engine.stdin(),
                "{\"method\":\"stop\",\"payload\":{\"service_id\":\"svc\"}}\n"
            );
        }

        #[test]
        fn a_response_with_an_apostrophe_reaches_the_caller_intact() {
            // The fake writes its answer through a shell literal; an apostrophe
            // in valid JSON must not end that literal early.
            let engine = FakeEngine::answering(
                "client-apostrophe",
                r#"{"ok":false,"error":{"code":"stop_failed","message":"can't stop"}}"#,
                0,
            );
            let error = run(&engine, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(error.to_string(), "stop_failed: can't stop");
        }

        #[test]
        fn the_options_reach_the_running_child() {
            let engine = FakeEngine::answering("client-options", STOPPED, 0);
            let paths = AppPaths {
                config_dir: "/c".into(),
                data_dir: "/d".into(),
                cache_dir: "/k".into(),
            };
            run(
                &engine,
                &EngineStdioOptions {
                    app_paths: Some(&paths),
                    env_root: Some(Path::new("/envs")),
                    endpoint_key_file: Some(Path::new("/keys/k")),
                    request_progress: true,
                    forward_stderr: false,
                },
            )
            .unwrap();
            assert_eq!(engine.env("ROCM_CLI_DATA_DIR").as_deref(), Some("/d"));
            assert_eq!(
                engine.env("ROCM_CLI_ENGINE_ENVS_ROOT").as_deref(),
                Some("/envs")
            );
            assert_eq!(
                engine.env("ROCM_SERVE_API_KEY_FILE").as_deref(),
                Some("/keys/k")
            );
            assert_eq!(
                engine.env("ROCM_ENGINE_PROGRESS_STDERR").as_deref(),
                Some("1")
            );
        }

        #[test]
        fn a_nonzero_exit_with_no_answer_reports_the_status_and_stderr() {
            let engine = FakeEngine::new("client-crash", true, "echo boom >&2\nexit 3");
            let error = run(&engine, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine stdio process exited with code 3: boom"
            );

            let silent = FakeEngine::new("client-silent-crash", true, "exit 4");
            let error = run(&silent, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(error.to_string(), "engine stdio process exited with code 4");
        }

        #[test]
        fn a_nonzero_exit_that_still_answered_is_decoded_not_discarded() {
            let engine = FakeEngine::answering(
                "client-answers-then-fails",
                r#"{"ok":false,"error":{"code":"stop_failed","message":"it broke"}}"#,
                1,
            );
            let error = run(&engine, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(error.to_string(), "stop_failed: it broke");

            let engine = FakeEngine::answering("client-answers-then-exits-2", STOPPED, 2);
            assert_eq!(
                run(&engine, &EngineStdioOptions::default()).unwrap(),
                json!({"stopped": true})
            );
        }

        #[test]
        fn an_unparseable_answer_carries_the_childs_stderr() {
            let engine = FakeEngine::new(
                "client-garbage",
                true,
                "printf 'not json'\necho oops >&2\nexit 0",
            );
            let error = run(&engine, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope; stderr: oops"
            );

            let silent = FakeEngine::new("client-garbage-quiet", true, "printf 'not json'");
            let error = run(&silent, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope"
            );
        }

        /// A child ended by a signal has no exit code; it is reported by the signal,
        /// with or without a half-written answer, and never as a code of its own
        /// invention.
        #[test]
        fn a_child_ended_by_a_signal_is_reported_by_the_signal() {
            let killed = FakeEngine::new("client-killed", true, "kill -KILL $$");
            let error = run(&killed, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine stdio process was terminated by signal 9"
            );

            let cut_short = FakeEngine::new(
                "client-killed-mid-answer",
                true,
                "printf '{\"ok\"'\nkill -KILL $$",
            );
            let error = run(&cut_short, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope \
                 (engine stdio process was terminated by signal 9)"
            );
        }

        /// A child that fails and also leaves something that is not an envelope on
        /// stdout is reported with its exit status (the status says the answer was
        /// cut short, where the parse error alone would not), as a child that
        /// printed nothing is. The same unreadable answer from a child that exited
        /// cleanly carries no status, so the note is the condition's own and not
        /// added to every parse failure.
        #[test]
        fn an_unparseable_answer_from_a_failing_child_names_the_exit_status() {
            let crashed = FakeEngine::new(
                "client-garbage-crash",
                true,
                "printf 'not json'\necho oops >&2\nexit 4",
            );
            let error = run(&crashed, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope \
                 (engine stdio process exited with code 4); stderr: oops"
            );

            let silent = FakeEngine::new(
                "client-garbage-crash-quiet",
                true,
                "printf 'not json'\nexit 5",
            );
            let error = run(&silent, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope \
                 (engine stdio process exited with code 5)"
            );

            let clean = FakeEngine::new("client-garbage-clean", true, "printf 'not json'\nexit 0");
            let error = run(&clean, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope"
            );
        }

        /// Runs `work` on its own thread and fails the test if it has not finished
        /// within `limit`, so a step that should not block turns red instead of
        /// hanging the run.
        fn finishes_within<T: Send + 'static>(
            limit: Duration,
            work: impl FnOnce() -> T + Send + 'static,
        ) -> T {
            let (sender, receiver) = mpsc::channel();
            thread::spawn(move || {
                let _ = sender.send(work());
            });
            receiver
                .recv_timeout(limit)
                .unwrap_or_else(|_| panic!("did not finish within {limit:?}"))
        }

        /// A request that cannot be written because the engine is gone reports
        /// what the engine said on stderr, not only that the write failed: the
        /// engine's own words are the explanation. The request is larger than a
        /// pipe buffer, so the write reaches the closed read end.
        #[test]
        fn a_request_to_an_engine_that_exited_reports_what_it_said_on_stderr() {
            let engine = FakeEngine::new("client-no-listen", false, "echo 'no thanks' >&2\nexit 3");
            let envelope =
                engine_request_envelope(EngineMethod::Stop, &json!({"blob": "x".repeat(1 << 20)}))
                    .unwrap();
            let error = run_engine_stdio(
                engine.path(),
                ["stdio"],
                &EngineStdioOptions::default(),
                &envelope,
            )
            .expect_err("the engine never read its request");
            let message = error.to_string();
            assert!(
                message.starts_with("failed to write engine request: "),
                "{message}"
            );
            assert!(message.ends_with("; stderr: no thanks"), "{message}");
        }

        /// The child of a failed write is killed and reaped when it is still
        /// running, which a failed write does not show it to have stopped.
        /// Linux-only for the `/proc` entry, which a zombie keeps until waited on.
        #[cfg(target_os = "linux")]
        #[test]
        fn a_request_that_cannot_be_written_kills_and_reaps_a_child_that_is_still_running() {
            let child = Command::new("sleep")
                .arg("60")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let proc_entry = format!("/proc/{}", child.id());
            assert!(Path::new(&proc_entry).exists(), "the child is running");

            let error = finishes_within(Duration::from_secs(10), move || {
                abandon_unwritten_request(
                    child,
                    &io::Error::from(io::ErrorKind::BrokenPipe),
                    Duration::from_secs(1),
                )
            });

            assert!(
                error
                    .to_string()
                    .starts_with("failed to write engine request: "),
                "{error}"
            );
            assert!(
                !error.to_string().contains("stderr"),
                "a child that said nothing adds no stderr: {error}"
            );
            assert!(
                !Path::new(&proc_entry).exists(),
                "the child must be killed and reaped, not left running or as a zombie"
            );
        }

        /// A grandchild that holds the child's stderr open past its exit cannot
        /// make the error path wait for it: what had been written is reported
        /// and the rest is not waited for.
        #[test]
        fn the_stderr_of_a_failed_write_is_not_waited_for_past_the_grace() {
            let mut child = Command::new("sh")
                .args(["-c", "echo said >&2; sleep 5 & exit 3"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            // The engine has said its piece and exited before the failure is
            // handled, as it has in the case this bounds.
            child.wait().unwrap();
            let started = Instant::now();

            let error = finishes_within(Duration::from_secs(4), move || {
                abandon_unwritten_request(
                    child,
                    &io::Error::from(io::ErrorKind::BrokenPipe),
                    Duration::from_millis(300),
                )
            });

            let waited = started.elapsed();
            assert!(
                waited < Duration::from_secs(4),
                "waited for the grandchild: {waited:?}"
            );
            assert!(error.to_string().ends_with("; stderr: said"), "{error}");
        }

        /// An answer whose read fails kills the child and reaps it, so a daemon
        /// that hits this repeatedly collects neither a running engine nor a
        /// zombie per failure. A zombie keeps its `/proc` entry until it is
        /// waited on, so the entry being gone shows the reap and not only the
        /// kill (hence Linux only).
        #[cfg(target_os = "linux")]
        #[test]
        fn a_failed_read_of_the_answer_kills_and_reaps_the_child() {
            let mut child = Command::new("sleep")
                .arg("60")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            let pid = child.id();
            let proc_entry = format!("/proc/{pid}");
            assert!(Path::new(&proc_entry).exists(), "the child is running");

            let error = read_response(&mut child, Fails).unwrap_err();

            assert_eq!(error.to_string(), "failed to read engine stdio response");
            assert_eq!(
                format!("{error:#}"),
                "failed to read engine stdio response: pipe went away"
            );
            assert!(
                !Path::new(&proc_entry).exists(),
                "the child must be killed and reaped, not left running or as a zombie"
            );
            // The status was collected by the reap, so it is still reported.
            assert_eq!(
                child.try_wait().unwrap().and_then(|status| status.signal()),
                Some(libc::SIGKILL)
            );
        }

        /// The success path of the same helper hands the bytes back and leaves
        /// the child alone.
        #[test]
        fn a_successful_read_of_the_answer_leaves_the_child_to_the_caller() {
            let mut child = Command::new("sleep")
                .arg("60")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();

            let bytes = read_response(&mut child, &b"{\"ok\":true}"[..]).unwrap();

            assert_eq!(bytes, b"{\"ok\":true}");
            assert!(child.try_wait().unwrap().is_none(), "still running");
            child.kill().unwrap();
            child.wait().unwrap();
        }

        #[test]
        fn a_program_that_cannot_start_is_named_in_the_error() {
            let error = run_engine_stdio(
                Path::new("/no/such/engine"),
                ["stdio"],
                &EngineStdioOptions::default(),
                &stop_request(),
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to spawn engine stdio process /no/such/engine"
            );
        }

        /// What the running child writes to its stderr reaches the forwarding
        /// sink as it arrives. (That nothing is forwarded without the option is
        /// the `off` arm of the test below.)
        #[test]
        fn a_childs_stderr_reaches_the_forwarding_sink_when_there_is_one() {
            let progress = "downloading 1/2\ndownloading 2/2\n";
            let body = format!("printf '%s' '{progress}' >&2\nprintf '%s' '{STOPPED}'");
            let engine = FakeEngine::new("client-forward-sink", true, &body);

            let forwarded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = {
                let forwarded = std::sync::Arc::clone(&forwarded);
                FnSink(move |bytes: &[u8]| {
                    forwarded.lock().unwrap().extend_from_slice(bytes);
                    Ok(bytes.len())
                })
            };
            let answer = run_engine_stdio_forwarding(
                engine.path(),
                ["stdio"],
                &EngineStdioOptions::default(),
                &stop_request(),
                Some(sink),
            )
            .unwrap();
            assert!(answer.ok);
            assert_eq!(forwarded.lock().unwrap().as_slice(), progress.as_bytes());
        }

        /// The public entry point sends the child's stderr to THIS process's
        /// stderr when `forward_stderr` is set, and does not when it is not. That
        /// is the one place the option takes effect, and it is what shows a user
        /// an engine install's progress. The test binary re-executes itself so the
        /// parent can read the stderr the arm writes.
        #[test]
        fn forward_stderr_sends_the_childs_stderr_to_this_processs_stderr() {
            const LINE: &str = "engine progress line for the terminal";
            if let Some(arm) = std::env::var_os("ROCM_TEST_CLIENT_FORWARD_OPTION_ARM") {
                let engine = FakeEngine::new(
                    "client-forward-option",
                    true,
                    &format!("echo '{LINE}' >&2\nprintf '%s' '{STOPPED}'"),
                );
                let options = EngineStdioOptions {
                    forward_stderr: arm == "on",
                    ..EngineStdioOptions::default()
                };
                run_engine_stdio(engine.path(), ["stdio"], &options, &stop_request()).unwrap();
                return;
            }

            for (arm, expected) in [("on", true), ("off", false)] {
                let output = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "stdio_client::tests::child::forward_stderr_sends_the_childs_stderr_to_this_processs_stderr",
                        "--exact",
                        "--nocapture",
                    ])
                    .env("ROCM_TEST_CLIENT_FORWARD_OPTION_ARM", arm)
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{arm}: {:?}", output.status);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert_eq!(
                    stderr.contains(LINE),
                    expected,
                    "forward_stderr {arm}: the child's stderr on this process's stderr:\n{stderr}"
                );
            }
        }

        /// Progress forwarded to a stderr whose reader has gone away is dropped, not
        /// fatal, also where SIGPIPE is at its default action (`rocm` runs that
        /// way). The `Closed` sink above cannot show this: it returns the error
        /// itself, where the real failure is a signal that ends the process before
        /// any error is returned. So this re-executes itself in a fresh process
        /// that installs `SIG_DFL` and forwards into one end of a real pipe whose
        /// other end is closed. Two arms, one per forwarding write in
        /// `collect_stderr`: a line of the child's stderr, and the note recorded
        /// when reading that stderr fails.
        #[test]
        #[allow(unsafe_code)] // libc::signal, to install the default action in the child
        fn forwarding_stderr_to_a_closed_pipe_is_not_fatal_under_sig_dfl() {
            if let Some(arm) = std::env::var_os("ROCM_TEST_CLIENT_FORWARD_ARM") {
                // SAFETY: installing a signal disposition is always safe.
                unsafe {
                    libc::signal(libc::SIGPIPE, libc::SIG_DFL);
                }
                let (reader, writer) = io::pipe().unwrap();
                drop(reader);
                let collected = if arm == "line" {
                    collect_stderr(&b"progress\n"[..], Some(writer))
                } else {
                    collect_stderr(Fails, Some(writer))
                };
                // Surviving is the point; the text is still kept for the error.
                assert!(!collected.is_empty());
                return;
            }

            for arm in ["line", "read-error"] {
                let status = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "stdio_client::tests::child::forwarding_stderr_to_a_closed_pipe_is_not_fatal_under_sig_dfl",
                        "--exact",
                        "--nocapture",
                    ])
                    .env("ROCM_TEST_CLIENT_FORWARD_ARM", arm)
                    .status()
                    .unwrap();
                assert_eq!(
                    status.signal(),
                    None,
                    "{arm}: SIGPIPE killed the process instead of the write failing: {status:?}"
                );
                assert!(status.success(), "{arm}: {status:?}");
            }
        }

        /// Whether SIGPIPE is ignored right now, read by installing the same
        /// disposition back.
        #[allow(unsafe_code)] // libc::signal
        fn sigpipe_is_ignored() -> bool {
            // SAFETY: reading/installing a signal disposition is always safe.
            let current = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
            // SAFETY: putting back what was just read.
            unsafe {
                libc::signal(libc::SIGPIPE, current);
            }
            current == libc::SIG_IGN
        }

        /// Forwarding guards each write on its own and never holds SIGPIPE ignored
        /// across the reads between them. The reads block for as long as the engine
        /// is quiet (an install runs for minutes), and the disposition is
        /// process-wide, so a guard around the whole loop would leave every other
        /// thread's write to a closed stdout returning an error for that whole time
        /// instead of ending `rocm` quietly. Each line costs one pair of
        /// `signal` calls, which is nothing next to the write itself. A fresh
        /// process with `SIG_DFL` records the disposition seen inside every read
        /// and every write, and after the loop.
        #[test]
        #[allow(unsafe_code)] // libc::signal, to install the default action in the child
        fn forwarding_guards_each_write_and_never_the_reads_between_them() {
            use std::cell::RefCell;
            use std::rc::Rc;

            type Seen = Rc<RefCell<Vec<(&'static str, bool)>>>;

            struct Source {
                lines: Vec<&'static [u8]>,
                seen: Seen,
            }
            impl Read for Source {
                fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                    self.seen.borrow_mut().push(("read", sigpipe_is_ignored()));
                    if self.lines.is_empty() {
                        return Ok(0);
                    }
                    let line = self.lines.remove(0);
                    buffer[..line.len()].copy_from_slice(line);
                    Ok(line.len())
                }
            }

            if std::env::var_os("ROCM_TEST_CLIENT_GUARD_SCOPE_ARM").is_some() {
                // SAFETY: installing a signal disposition is always safe.
                unsafe {
                    libc::signal(libc::SIGPIPE, libc::SIG_DFL);
                }
                let seen: Seen = Rc::default();
                let source = Source {
                    lines: vec![b"one\n", b"two\n", b"three\n"],
                    seen: Rc::clone(&seen),
                };
                let sink_seen = Rc::clone(&seen);
                let sink = FnSink(move |bytes: &[u8]| {
                    sink_seen.borrow_mut().push(("write", sigpipe_is_ignored()));
                    Ok(bytes.len())
                });
                collect_stderr(source, Some(sink));
                let seen = seen.borrow();
                let writes = seen.iter().filter(|(what, _)| *what == "write").count();
                assert_eq!(writes, 3, "{seen:?}");
                for (what, ignored) in seen.iter() {
                    assert_eq!(
                        *ignored,
                        *what == "write",
                        "SIGPIPE must be ignored inside each write and only there: {seen:?}"
                    );
                }
                assert!(
                    !sigpipe_is_ignored(),
                    "the default action is back after the loop"
                );
                return;
            }

            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "stdio_client::tests::child::forwarding_guards_each_write_and_never_the_reads_between_them",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_CLIENT_GUARD_SCOPE_ARM", "1")
                .status()
                .unwrap();
            assert!(status.success(), "{status:?}");
        }

        /// A request written to an engine that has already exited comes back as an
        /// ordinary error, also where SIGPIPE is at its default action. Re-executes
        /// this test in a fresh process that installs `SIG_DFL` first, because the
        /// harness leaves SIGPIPE ignored and changing it process-wide would race
        /// sibling tests; a write that is not guarded kills that process, and the
        /// parent reports the signal.
        #[test]
        #[allow(unsafe_code)] // libc::signal, to install the default action in the child
        fn a_request_to_an_engine_that_already_exited_is_an_error_under_sig_dfl() {
            if std::env::var_os("ROCM_TEST_CLIENT_SIGPIPE_ARM").is_some() {
                // SAFETY: installing a signal disposition is always safe.
                unsafe {
                    libc::signal(libc::SIGPIPE, libc::SIG_DFL);
                }
                let engine = FakeEngine::new("client-sigpipe", false, "exit 3");
                // Larger than a pipe buffer, so the write reaches the closed read end.
                let envelope = engine_request_envelope(
                    EngineMethod::Stop,
                    &json!({"blob": "x".repeat(1 << 20)}),
                )
                .unwrap();
                let error = run_engine_stdio(
                    engine.path(),
                    ["stdio"],
                    &EngineStdioOptions::default(),
                    &envelope,
                )
                .expect_err("the engine never answered");
                assert!(
                    format!("{error:#}").contains("failed to write engine request"),
                    "{error:#}"
                );
                return;
            }

            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "stdio_client::tests::child::a_request_to_an_engine_that_already_exited_is_an_error_under_sig_dfl",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_CLIENT_SIGPIPE_ARM", "1")
                .status()
                .unwrap();
            assert_eq!(
                status.signal(),
                None,
                "SIGPIPE killed the process instead of surfacing as an error: {status:?}"
            );
            assert!(status.success(), "{status:?}");
        }
    }
}
