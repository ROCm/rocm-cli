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
//! It is also where the engine protocol version is enforced on the client side
//! (the rule is in the `protocol_version` module). [`decode_engine_response`]
//! checks an answer's `protocol_version` before anything else in it, and
//! [`run_engine_stdio`], before an `install` or a `launch`, first sends a
//! `capabilities` request through the same path and does not send the real one
//! unless that answer is in a protocol this build can use. The other methods are
//! sent once, as they are.
//!
//! The whole of an engine's stdout must be its one answer: an engine writes
//! only its JSON answer to stdout and its logs to stderr. Output that is not
//! exactly one envelope is refused, never salvaged, and the error says what was
//! wrong with it (nothing printed, other output beside an answer, JSON that is
//! not an envelope, not JSON) and quotes the start of it.
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
//!
//! These three are deliberate tradeoffs, not oversights: the wait on pipes held
//! by a grandchild is unbounded, an answer from an engine that exits without
//! draining a large request is not decoded, and SIGPIPE stays ignored
//! process-wide while a stalled stderr forward blocks. Only the first is
//! partly cheap to bound: the stderr side would be one deadline on
//! `StderrCollector::finish`, but stdout is read to its end on the calling thread,
//! so a bound there needs a second reader thread and a decision on what to report
//! when it fires, and bounding one pipe and not the other would be a half
//! measure. The stderr is read from the moment the child starts, on its own
//! thread, so an engine that fills it before draining a large request cannot
//! deadlock the write.

use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ENGINE_ENVS_ROOT_ENV, with_sigpipe_ignored};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{
    ENDPOINT_API_KEY_FILE_ENV, ENGINE_PROTOCOL_VERSION, EngineMethod, EngineRequestEnvelope,
    EngineResponseEnvelope, ProtocolMismatch, engine_protocol_compatible,
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
    /// With no `endpoint_key_file`, also CLEAR [`ENDPOINT_API_KEY_FILE_ENV`]
    /// rather than leave whatever this process inherited to pass through.
    ///
    /// For a child that must not see a credential path at all — `rocm`'s
    /// external engine plugins are third-party binaries. Off by default,
    /// because "no key file for this request" and "this child may not have one"
    /// are different statements: `rocmd` passes `None` for a loopback service
    /// that simply needs no key, and its children are its own.
    pub withhold_inherited_key_file: bool,
    /// Ask the engine for progress on stderr ([`ENGINE_PROGRESS_STDERR_ENV`]).
    pub request_progress: bool,
    /// Write the child's stderr to this process's stderr as it arrives, so a
    /// terminal user watches an install. Either way the whole of it is kept and
    /// shown in the error when the request fails.
    pub forward_stderr: bool,
    /// Say, on this process's stderr, when the child answered and then exited
    /// with a failing status. The answer is used either way; this only leaves a
    /// trace of the status, which the caller would otherwise never see. For a
    /// daemon, whose stderr is its log; a terminal user has no use for it.
    pub note_failing_exit_after_answer: bool,
}

/// Build the request envelope for `method` from `request`.
pub fn engine_request_envelope<T: Serialize>(
    method: EngineMethod,
    request: &T,
) -> Result<EngineRequestEnvelope> {
    Ok(EngineRequestEnvelope::new(
        method,
        serde_json::to_value(request).context("failed to encode engine request payload")?,
    ))
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
        paths.apply_env(&mut command);
    }
    if let Some(env_root) = options.env_root {
        command.env(ENGINE_ENVS_ROOT_ENV, env_root);
    }
    if let Some(key_file) = options.endpoint_key_file {
        command.env(ENDPOINT_API_KEY_FILE_ENV, key_file);
    } else if options.withhold_inherited_key_file {
        // Cleared, not merely left unset: a value this process inherited must
        // not reach a third-party plugin binary just because nothing
        // overwrote it. Only for callers that ask — `rocmd` passes `None` for
        // a loopback service that needs no key, and clearing there would
        // change what its own children see. See
        // `an_external_plugin_child_is_not_handed_the_endpoint_key_file`.
        command.env_remove(ENDPOINT_API_KEY_FILE_ENV);
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
///
/// For a method that changes the machine
/// ([`EngineMethod::requires_protocol_handshake`]), the engine is first asked
/// for its capabilities, as a request of its own, and `envelope` is sent only
/// when that answer is in a compatible protocol and is not an error; otherwise
/// the error says `envelope` was not sent, and why.
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
    let args: Vec<OsString> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_owned())
        .collect();
    if envelope.method.requires_protocol_handshake() {
        check_protocol_before(program, &args, options, &envelope.method)?;
    }
    exchange(program, &args, options, envelope, forward)
}

/// Before a request that changes the machine (see
/// [`EngineMethod::requires_protocol_handshake`]), ask the engine for its
/// capabilities, through the same path, and check the protocol it answers in.
/// The request itself is sent only when that answer is in a protocol this
/// build can use; otherwise the error says it was not sent, and why.
///
/// Nothing is cached: an engine can be replaced between two requests, and the
/// extra child is small next to the install or launch it guards. The
/// capabilities child's stderr is not forwarded (it is in the error when the
/// check fails), and it is not asked for progress.
fn check_protocol_before(
    program: &Path,
    args: &[OsString],
    options: &EngineStdioOptions<'_>,
    method: &EngineMethod,
) -> Result<()> {
    let refused = format!("the {} request was not sent to the engine", method.name());
    let handshake_options = EngineStdioOptions {
        request_progress: false,
        forward_stderr: false,
        note_failing_exit_after_answer: false,
        ..*options
    };
    let capabilities =
        EngineRequestEnvelope::new(EngineMethod::Capabilities, serde_json::Value::Null);
    let answer = exchange(
        program,
        args,
        &handshake_options,
        &capabilities,
        None::<io::Stderr>,
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "{refused}: its protocol could not be checked against this release's \
             (engine protocol {ENGINE_PROTOCOL_VERSION}) because asking for its capabilities \
             failed: {error:#}"
        )
    })?;
    match engine_protocol_compatible(ENGINE_PROTOCOL_VERSION, answer.protocol_version.as_deref()) {
        Ok(()) => {}
        Err(ProtocolMismatch::Incompatible { ours, theirs }) => bail!(
            "{refused}: the engine speaks engine protocol {theirs}, which this release \
             (engine protocol {ours}) cannot use"
        ),
        Err(ProtocolMismatch::Unreadable { ours, theirs }) => bail!(
            "{refused}: the engine named its protocol version {theirs:?}, which is not a \
             MAJOR.MINOR.PATCH version, so this release (engine protocol {ours}) cannot tell \
             whether it can use it"
        ),
    }
    if !answer.ok {
        bail!(
            "{refused}: its protocol could not be checked against this release's \
             (engine protocol {ENGINE_PROTOCOL_VERSION}) because it refused the capabilities \
             request: {}",
            engine_error_text(answer.error)
        );
    }
    Ok(())
}

/// One request and its answer: start the child, write `envelope`, read what it
/// printed. [`run_engine_stdio_forwarding`] is this plus the protocol check that
/// goes before a request that changes the machine.
fn exchange(
    program: &Path,
    args: &[OsString],
    options: &EngineStdioOptions<'_>,
    envelope: &EngineRequestEnvelope,
    forward: Option<impl Write + Send + 'static>,
) -> Result<EngineResponseEnvelope> {
    let mut child = engine_stdio_command(program, args, options)
        .spawn()
        .with_context(|| format!("failed to spawn engine stdio process {}", program.display()))?;

    let stdin = child.stdin.take();
    let mut stdin = require_pipe(&mut child, stdin, "stdin")?;
    // Read the child's stderr from the start, on its own thread: an engine that
    // fills the stderr pipe before it has drained a request larger than a pipe
    // buffer would otherwise block on that write while this process blocks on
    // its own, with nobody reading either side.
    let stderr_collector = child
        .stderr
        .take()
        .map(|stderr| StderrCollector::start(stderr, forward));
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
            stderr_collector,
            &error,
            EXITED_STDERR_GRACE,
        ));
    }
    // End of the request: the engine reads to EOF before it answers.
    drop(stdin);

    let stdout_pipe = child.stdout.take();
    let stdout_pipe = require_pipe(&mut child, stdout_pipe, "stdout")?;
    let stdout = match read_response(&mut child, stdout_pipe) {
        Ok(stdout) => stdout,
        Err(error) => {
            return Err(read_failure_with_stderr(
                error,
                stderr_collector,
                EXITED_STDERR_GRACE,
            ));
        }
    };
    let status = child
        .wait()
        .context("failed waiting for engine stdio response")?;
    let stderr = stderr_collector
        .map(|collector| collector.finish(None))
        .unwrap_or_default();

    let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
    if !status.success() && stdout.is_empty() {
        let exited = exit_note(status);
        if stderr.is_empty() {
            bail!("{exited}");
        }
        bail!("{exited}: {stderr}");
    }
    let answer = serde_json::from_slice::<EngineResponseEnvelope>(&stdout)
        .map_err(|_| anyhow::anyhow!(unparseable_answer_message(status, &stdout, &stderr)));
    // The note says the answer was used, so it is only made for an answer that
    // parsed; an unparseable one is reported by the error above, with the status.
    if answer.is_ok() && !status.success() && options.note_failing_exit_after_answer {
        note_failing_exit_after_answer(&mut io::stderr(), status);
    }
    answer
}

/// Say on `sink` (this process's stderr, in production) that the child answered
/// and then failed. Written like the stderr forwarding, through the SIGPIPE guard
/// and ignoring a failed write: a closed stderr must neither end a daemon hosted
/// in `rocm` (SIGPIPE at its default action) nor panic it, as `eprintln!` would.
fn note_failing_exit_after_answer(sink: &mut impl Write, status: ExitStatus) {
    let note = format!(
        "{} after printing its answer; the answer was used\n",
        exit_note(status)
    );
    forward_best_effort(sink, note.as_bytes());
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
/// cut short; then what was wrong with what it printed ([`unreadable_stdout`]);
/// the stderr it wrote follows when there is any.
fn unparseable_answer_message(status: ExitStatus, stdout: &[u8], stderr: &str) -> String {
    let mut message = "failed to parse engine response envelope".to_owned();
    if !status.success() {
        message = format!("{message} ({})", exit_note(status));
    }
    message = format!("{message}: {}", unreadable_stdout(stdout));
    if !stderr.is_empty() {
        message = format!("{message}; stderr: {stderr}");
    }
    message
}

/// What an engine's stdout must hold, said wherever it held something else.
const STDOUT_RULE: &str = "an engine writes only its JSON answer to stdout; logs go to stderr";

/// How much of an unreadable stdout an error quotes: its first non-blank line,
/// cut to this many characters.
const STDOUT_EXCERPT_CHARS: usize = 120;

/// How many line starts [`answer_among_other_output`] tries an envelope at, so
/// a large unreadable stdout costs a bounded amount of work to describe.
const ENVELOPE_SEARCH_LINES: usize = 256;

/// What is wrong with a stdout that does not parse as one response envelope,
/// for the error that reports it.
///
/// The whole of stdout must be the answer, so nothing here salvages one: an
/// answer found among other output is still refused, and only named as the
/// reason. The cases, in the order they are told apart:
/// - nothing (or only whitespace) was printed;
/// - an envelope is there, but so is other output before or after it (a log
///   line on the wrong stream);
/// - the whole of it is JSON, but not an envelope;
/// - it is not JSON.
fn unreadable_stdout(stdout: &[u8]) -> String {
    if stdout.iter().all(u8::is_ascii_whitespace) {
        return "the engine printed no answer on stdout".to_owned();
    }
    let excerpt = stdout_excerpt(stdout);
    if answer_among_other_output(stdout) {
        return format!(
            "the engine wrote other output to stdout alongside its answer, starting with \
             `{excerpt}` ({STDOUT_RULE})"
        );
    }
    match serde_json::from_slice::<serde_json::Value>(stdout) {
        Ok(_) => {
            let reason = serde_json::from_slice::<EngineResponseEnvelope>(stdout)
                .err()
                .map_or_else(String::new, |error| format!(" ({error})"));
            format!(
                "the engine's stdout is JSON but not an answer envelope{reason}, starting with \
                 `{excerpt}` ({STDOUT_RULE})"
            )
        }
        Err(error) => format!(
            "the engine's stdout is not JSON ({error}), starting with `{excerpt}` \
             ({STDOUT_RULE})"
        ),
    }
}

/// Whether a whole response envelope starts at the beginning of one of the
/// first lines of `stdout` (leading whitespace aside), with other output before
/// or after it. Engines pretty-print their answer over many lines, so only the
/// start of the envelope is looked for at a line start; its end is wherever
/// the JSON ends.
fn answer_among_other_output(stdout: &[u8]) -> bool {
    let mut line_start = 0;
    for _ in 0..ENVELOPE_SEARCH_LINES {
        let rest = &stdout[line_start..];
        let indent = rest
            .iter()
            .take_while(|byte| matches!(byte, b' ' | b'\t'))
            .count();
        if rest.get(indent) == Some(&b'{') {
            let mut values = serde_json::Deserializer::from_slice(&rest[indent..])
                .into_iter::<EngineResponseEnvelope>();
            if let Some(Ok(_)) = values.next() {
                let end = line_start + indent + values.byte_offset();
                let before = &stdout[..line_start];
                let after = &stdout[end..];
                let blank = |bytes: &[u8]| bytes.iter().all(u8::is_ascii_whitespace);
                return !blank(before) || !blank(after);
            }
        }
        match rest.iter().position(|byte| *byte == b'\n') {
            Some(newline) => line_start += newline + 1,
            None => return false,
        }
    }
    false
}

/// The first non-blank line of `stdout`, cut to [`STDOUT_EXCERPT_CHARS`]
/// characters, with control characters escaped so an engine's output cannot
/// reshape the error (or the terminal) that quotes it.
fn stdout_excerpt(stdout: &[u8]) -> String {
    let text = String::from_utf8_lossy(stdout);
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let mut excerpt = String::new();
    for (index, ch) in line.chars().enumerate() {
        if index == STDOUT_EXCERPT_CHARS {
            excerpt.push_str("...");
            break;
        }
        if ch.is_control() {
            excerpt.extend(ch.escape_default());
        } else {
            excerpt.push(ch);
        }
    }
    excerpt
}

/// `pipe` of `child`, or an error that also kills and reaps the child: a child
/// whose request cannot be sent must not be left running, as on every other
/// failure path. Unreachable while the command pipes all three streams, which is
/// why it is a function of its own, testable with a pipe that is missing.
fn require_pipe<T>(child: &mut Child, pipe: Option<T>, name: &str) -> Result<T> {
    pipe.ok_or_else(|| {
        let _ = child.kill();
        let _ = child.wait();
        anyhow::anyhow!("engine stdio child did not expose {name}")
    })
}

/// The error for a stdout that could not be read (the child is already killed
/// and reaped), with what its stderr had said, as every other failure carries
/// it: a crashing or signalled engine is where the explanation matters most.
fn read_failure_with_stderr(
    error: anyhow::Error,
    stderr: Option<StderrCollector>,
    grace: Duration,
) -> anyhow::Error {
    let stderr = stderr
        .map(|collector| collector.finish(Some(Instant::now() + grace)))
        .unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
    if stderr.is_empty() {
        return error;
    }
    anyhow::anyhow!("{error:#}; stderr: {stderr}")
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
    stderr: Option<StderrCollector>,
    write_error: &io::Error,
    grace: Duration,
) -> anyhow::Error {
    let _ = child.kill();
    let _ = child.wait();
    let deadline = Instant::now() + grace;
    let stderr = stderr
        .map(|collector| collector.finish(Some(deadline)))
        .unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
    let mut message = format!("failed to write engine request: {write_error}");
    if !stderr.is_empty() {
        message = format!("{message}; stderr: {stderr}");
    }
    anyhow::anyhow!(message)
}

/// A child's stderr being collected on a thread of its own, from the moment it
/// is started, whatever the main thread is doing meanwhile.
///
/// The thread outlives the collector when a request fails and a descendant of
/// the engine still holds the pipe: nobody will read what it collects, but it
/// must keep draining the pipe (closing it would hand that descendant a SIGPIPE
/// it did not get before). So dropping the collector marks the capture
/// abandoned, which releases what was retained and makes the thread discard
/// whatever arrives later, in chunks of a fixed size, so a long-lived daemon does
/// not grow with every failed request.
struct StderrCollector {
    shared: Arc<Mutex<Captured>>,
    /// Receives one message when the stderr has been read to its end.
    finished: mpsc::Receiver<()>,
}

/// What the collecting thread has read, and whether anyone still wants it.
#[derive(Default)]
struct Captured {
    bytes: Vec<u8>,
    /// Set when the collector is dropped. Read and written under the same lock as
    /// `bytes`, so "emptied and marked" is one step as far as the reader is
    /// concerned and the buffer cannot grow after the drop returns.
    abandoned: bool,
}

impl StderrCollector {
    /// Start reading `stderr`, forwarding what arrives to `forward` when there is
    /// one (see [`collect_stderr_into`]).
    fn start(
        stderr: impl Read + Send + 'static,
        forward: Option<impl Write + Send + 'static>,
    ) -> Self {
        let shared = Arc::new(Mutex::new(Captured::default()));
        let (done, finished) = mpsc::channel();
        let sink = Arc::clone(&shared);
        thread::spawn(move || {
            collect_stderr_into(stderr, forward, &sink);
            let _ = done.send(());
        });
        Self { shared, finished }
    }

    /// What was collected, once the stderr has ended, or what has arrived by
    /// `deadline` when there is one and the stderr has not ended by then (a
    /// grandchild holding the pipe open). With no deadline this waits for the
    /// end, which is the unbounded wait the module comment states.
    fn finish(&self, deadline: Option<Instant>) -> Vec<u8> {
        match deadline {
            None => {
                let _ = self.finished.recv();
            }
            Some(deadline) => {
                let _ = self
                    .finished
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()));
            }
        }
        std::mem::take(&mut lock(&self.shared).bytes)
    }
}

impl Drop for StderrCollector {
    fn drop(&mut self) {
        let mut captured = lock(&self.shared);
        captured.abandoned = true;
        captured.bytes = Vec::new();
    }
}

fn lock(shared: &Mutex<Captured>) -> std::sync::MutexGuard<'_, Captured> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
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
#[cfg(test)]
fn collect_stderr(stderr: impl Read, forward: Option<impl Write>) -> Vec<u8> {
    let shared = Mutex::new(Captured::default());
    collect_stderr_into(stderr, forward, &shared);
    shared
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .bytes
}

/// [`collect_stderr`], publishing into `shared` as each chunk arrives, not when a
/// line is complete, so a caller that stops waiting has everything that had been
/// written (a diagnostic with no trailing newline included) and not only the
/// complete lines. Once the capture is abandoned the thread keeps reading, so the
/// pipe stays drained, but keeps and forwards nothing.
fn collect_stderr_into(
    mut stderr: impl Read,
    mut forward: Option<impl Write>,
    shared: &Mutex<Captured>,
) {
    let mut chunk = [0_u8; 4096];
    let mut carry = Utf8Carry::default();
    loop {
        let bytes: Vec<u8> = match stderr.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => chunk[..read].to_vec(),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                let message = format!("failed to read engine progress: {error}\n").into_bytes();
                publish(shared, forward.as_mut(), &mut carry, &message);
                break;
            }
        };
        publish(shared, forward.as_mut(), &mut carry, &bytes);
    }
    // What the pipe ended in the middle of a character of is forwarded lossily.
    if let Some(sink) = forward.as_mut()
        && !lock(shared).abandoned
    {
        forward_best_effort(sink, &carry.finish());
    }
}

/// Keep `bytes` and forward the complete UTF-8 of them, unless the capture has
/// been abandoned. The kept copy is the raw bytes; only what is forwarded is
/// held back at an incomplete trailing character.
fn publish(
    shared: &Mutex<Captured>,
    forward: Option<&mut impl Write>,
    carry: &mut Utf8Carry,
    bytes: &[u8],
) {
    {
        let mut captured = lock(shared);
        if captured.abandoned {
            return;
        }
        captured.bytes.extend_from_slice(bytes);
    }
    if let Some(sink) = forward {
        let complete = carry.push(bytes);
        if !complete.is_empty() {
            forward_best_effort(sink, &complete);
        }
    }
}

/// U+FFFD in UTF-8, as bytes: a literal here would be a non-ASCII string in
/// production code, which the output contract forbids.
const REPLACEMENT_CHARACTER: &[u8] = &[0xEF, 0xBF, 0xBD];

/// Turns arbitrary chunks of a byte stream into text that is valid UTF-8 at every
/// boundary, for a sink that requires it. Rust's stdio on a Windows console
/// refuses a write that is not valid UTF-8, so a chunk that ends in the middle of
/// a multi-byte character would fail, and the forwarding discards that error,
/// losing valid output. The incomplete trailing sequence is carried to the next
/// chunk instead; bytes that can never be valid are replaced with U+FFFD, and
/// what is still carried when the stream ends is flushed the same way.
#[derive(Default)]
struct Utf8Carry {
    pending: Vec<u8>,
}

impl Utf8Carry {
    /// The part of `pending + bytes` that is complete UTF-8 (with invalid bytes
    /// replaced), keeping an incomplete trailing sequence for the next call.
    fn push(&mut self, bytes: &[u8]) -> Vec<u8> {
        self.pending.extend_from_slice(bytes);
        let mut out = Vec::with_capacity(self.pending.len());
        let mut rest = std::mem::take(&mut self.pending);
        loop {
            match std::str::from_utf8(&rest) {
                Ok(valid) => {
                    out.extend_from_slice(valid.as_bytes());
                    return out;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    out.extend_from_slice(&rest[..valid_up_to]);
                    match error.error_len() {
                        // The input ends inside a character: wait for the rest.
                        None => {
                            self.pending = rest[valid_up_to..].to_vec();
                            return out;
                        }
                        Some(len) => {
                            out.extend_from_slice(REPLACEMENT_CHARACTER);
                            rest.drain(..valid_up_to + len);
                        }
                    }
                }
            }
        }
    }

    /// What is carried at the end of the stream, replaced lossily.
    fn finish(&mut self) -> Vec<u8> {
        let pending = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&pending).into_owned().into_bytes()
    }
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
///
/// The protocol the answer was written in is checked first, before `ok`: an
/// answer in a protocol this build cannot use is refused whatever it says,
/// and so is an engine's own error in such a protocol, because its meaning
/// cannot be trusted either and the mismatch is the thing to fix.
pub fn decode_engine_response<R>(envelope: EngineResponseEnvelope) -> Result<R>
where
    R: DeserializeOwned,
{
    if let Err(mismatch) = engine_protocol_compatible(
        ENGINE_PROTOCOL_VERSION,
        envelope.protocol_version.as_deref(),
    ) {
        bail!("{}", incompatible_answer_message(&mismatch));
    }
    if !envelope.ok {
        bail!("{}", engine_error_text(envelope.error));
    }
    let data = envelope
        .data
        .context("engine response envelope did not contain data")?;
    serde_json::from_value(data)
        .map_err(|error| anyhow::anyhow!("failed to decode engine response payload: {error}"))
}

/// The error for an answer in a protocol this build cannot use.
fn incompatible_answer_message(mismatch: &ProtocolMismatch) -> String {
    match mismatch {
        ProtocolMismatch::Incompatible { ours, theirs } => format!(
            "engine answered in engine protocol {theirs}, which this release \
             (protocol {ours}) cannot use"
        ),
        ProtocolMismatch::Unreadable { ours, theirs } => format!(
            "engine answered with protocol version {theirs:?}, which is not a \
             MAJOR.MINOR.PATCH version this release (protocol {ours}) can read"
        ),
    }
}

/// An engine's own error, as `code: message`.
fn engine_error_text(error: Option<crate::EngineErrorDetail>) -> String {
    error.map_or_else(
        || "unknown engine error".to_owned(),
        |value| format!("{}: {}", value.code, value.message),
    )
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

    /// A child told it gets no endpoint key must have the variable CLEARED,
    /// not merely left unset.
    ///
    /// `rocm` passes `endpoint_key_file: None` for every external engine
    /// plugin — a third-party binary — and the path it would otherwise carry
    /// points at a mode-0600 file holding a live endpoint credential. Merely
    /// not setting it would still hand the value through whenever this process
    /// inherited one.
    ///
    /// Asserts the distinction directly: `Command` records `env_remove` as an
    /// explicit `(key, None)` entry, so a plain "is it absent" check would pass
    /// even with the clearing deleted.
    #[test]
    fn an_external_plugin_child_is_not_handed_the_endpoint_key_file() {
        let key_var = std::ffi::OsStr::new(ENDPOINT_API_KEY_FILE_ENV);
        let command = engine_stdio_command(
            Path::new("/an/engine"),
            ["stdio"],
            &EngineStdioOptions {
                app_paths: None,
                env_root: None,
                endpoint_key_file: None,
                withhold_inherited_key_file: true,
                request_progress: false,
                forward_stderr: false,
                note_failing_exit_after_answer: false,
            },
        );

        let entry = command
            .get_envs()
            .find(|(key, _)| *key == key_var)
            .map(|(_, value)| value);
        assert_eq!(
            entry,
            Some(None),
            "the key-file variable must be explicitly CLEARED when no key file \
             is passed, not merely left unset"
        );

        // The other direction, which is what keeps this scoped: a caller that
        // did not ask (`rocmd`, for a loopback service that simply needs no
        // key) must leave the child's inherited value alone.
        let kept = engine_stdio_command(
            Path::new("/an/engine"),
            ["stdio"],
            &EngineStdioOptions {
                app_paths: None,
                env_root: None,
                endpoint_key_file: None,
                withhold_inherited_key_file: false,
                request_progress: false,
                forward_stderr: false,
                note_failing_exit_after_answer: false,
            },
        );
        assert!(
            kept.get_envs().all(|(key, _)| key != key_var),
            "a caller that did not ask to withhold must not touch the variable"
        );
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
                withhold_inherited_key_file: false,
                request_progress: true,
                forward_stderr: true,
                note_failing_exit_after_answer: false,
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
                    withhold_inherited_key_file: false,
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
            r#"{"protocol_version":"0.1.0","method":"stop","payload":{"service_id":"svc"}}"#
        );
    }

    #[test]
    fn decode_engine_response_reports_each_way_an_answer_can_be_unusable() {
        let failed = EngineResponseEnvelope {
            protocol_version: None,
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
            protocol_version: None,
            ok: false,
            data: None,
            error: None,
        };
        let error = decode_engine_response::<Value>(unexplained).unwrap_err();
        assert_eq!(error.to_string(), "unknown engine error");

        let empty = EngineResponseEnvelope {
            protocol_version: None,
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
            "failed to decode engine response payload: \
             invalid type: string \"yes\", expected a boolean"
        );
    }

    /// An answer, as a client reads it, from an engine that wrote it in
    /// `version` (`None`: before the field existed).
    fn answer_in(version: Option<&str>, ok: bool) -> EngineResponseEnvelope {
        let mut answer = if ok {
            EngineResponseEnvelope::success(json!({"stopped": true, "graceful": true}))
        } else {
            EngineResponseEnvelope::failure("incompatible_protocol", "the engine's own refusal")
        };
        answer.protocol_version = version.map(str::to_owned);
        answer
    }

    /// The answer's protocol is checked before anything else in it: a success
    /// with perfectly good data is refused when it is in a protocol this
    /// release cannot use, and an engine's own error in such a protocol is
    /// reported as the mismatch, which is what has to be fixed.
    #[test]
    fn an_answer_in_an_incompatible_protocol_is_refused_whatever_it_says() {
        for ok in [true, false] {
            let error =
                decode_engine_response::<StopResponse>(answer_in(Some("9.0.0"), ok)).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine answered in engine protocol 9.0.0, which this release \
                 (protocol 0.1.0) cannot use",
                "ok: {ok}"
            );
            let error =
                decode_engine_response::<StopResponse>(answer_in(Some("0.2.0"), ok)).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine answered in engine protocol 0.2.0, which this release \
                 (protocol 0.1.0) cannot use",
                "ok: {ok}"
            );
            let error =
                decode_engine_response::<StopResponse>(answer_in(Some("banana"), ok)).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine answered with protocol version \"banana\", which is not a \
                 MAJOR.MINOR.PATCH version this release (protocol 0.1.0) can read",
                "ok: {ok}"
            );
        }
    }

    /// The same answer in this protocol, in a newer one with only additions,
    /// or from an engine older than the field, is read as answered.
    #[test]
    fn an_answer_in_a_compatible_or_unnamed_protocol_is_read() {
        for version in [Some("0.1.0"), Some("0.1.9"), None] {
            let decoded = decode_engine_response::<StopResponse>(answer_in(version, true))
                .unwrap_or_else(|error| panic!("{version:?}: {error:#}"));
            assert!(decoded.stopped, "{version:?}");
            let error =
                decode_engine_response::<StopResponse>(answer_in(version, false)).unwrap_err();
            assert_eq!(
                error.to_string(),
                "incompatible_protocol: the engine's own refusal",
                "{version:?}"
            );
        }
    }

    /// Each way a stdout can fail to be one envelope is told apart, and none of
    /// them is salvaged.
    #[test]
    fn an_unreadable_stdout_is_described_by_what_is_wrong_with_it() {
        const RULE: &str = "(an engine writes only its JSON answer to stdout; logs go to stderr)";
        let envelope = r#"{"protocol_version":"0.1.0","ok":true,"data":{"stopped":true}}"#;
        let cases: Vec<(String, String)> = vec![
            (
                String::new(),
                "the engine printed no answer on stdout".to_owned(),
            ),
            (
                " \n\t\n".to_owned(),
                "the engine printed no answer on stdout".to_owned(),
            ),
            (
                format!("INFO loading model\n{envelope}\n"),
                format!(
                    "the engine wrote other output to stdout alongside its answer, starting \
                     with `INFO loading model` {RULE}"
                ),
            ),
            (
                format!("{envelope}\ndone\n"),
                format!(
                    "the engine wrote other output to stdout alongside its answer, starting \
                     with `{envelope}` {RULE}"
                ),
            ),
            (
                "loading {model}\n  {\n    \"ok\": true,\n    \"data\": {}\n  }\n".to_owned(),
                format!(
                    "the engine wrote other output to stdout alongside its answer, starting \
                     with `loading {{model}}` {RULE}"
                ),
            ),
            (
                r#"{"status":"ok"}"#.to_owned(),
                format!(
                    "the engine's stdout is JSON but not an answer envelope (missing field `ok` \
                     at line 1 column 15), starting with `{{\"status\":\"ok\"}}` {RULE}"
                ),
            ),
            (
                "not json".to_owned(),
                format!(
                    "the engine's stdout is not JSON (expected ident at line 1 column 2), \
                     starting with `not json` {RULE}"
                ),
            ),
            (
                "\u{1b}[31mred\u{7}\n".to_owned(),
                format!(
                    "the engine's stdout is not JSON (expected value at line 1 column 1), \
                     starting with `\\u{{1b}}[31mred\\u{{7}}` {RULE}"
                ),
            ),
        ];
        for (stdout, expected) in cases {
            assert_eq!(unreadable_stdout(stdout.as_bytes()), expected, "{stdout:?}");
        }
    }

    /// The quoted part of an unreadable stdout is its first non-blank line,
    /// bounded, so a megabyte of output does not become a megabyte of error.
    #[test]
    fn the_stdout_excerpt_is_the_first_line_and_bounded() {
        assert_eq!(stdout_excerpt(b"\n\n  first  \nsecond"), "first");
        let long = "x".repeat(500);
        let excerpt = stdout_excerpt(long.as_bytes());
        assert_eq!(excerpt, format!("{}...", "x".repeat(STDOUT_EXCERPT_CHARS)));
        let exact = "y".repeat(STDOUT_EXCERPT_CHARS);
        assert_eq!(stdout_excerpt(exact.as_bytes()), exact);
    }

    /// A source fed by the test, one chunk per message, that says when it is
    /// dropped (the collecting thread has ended).
    struct Fed {
        chunks: mpsc::Receiver<Vec<u8>>,
        dropped: mpsc::Sender<()>,
    }
    impl Read for Fed {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            match self.chunks.recv() {
                Ok(chunk) => {
                    buffer[..chunk.len()].copy_from_slice(&chunk);
                    Ok(chunk.len())
                }
                Err(_) => Ok(0),
            }
        }
    }
    impl Drop for Fed {
        fn drop(&mut self) {
            let _ = self.dropped.send(());
        }
    }

    /// After a request has failed and its collector is gone, a descendant that
    /// still writes to the pipe must not make the daemon grow: what was retained
    /// is released at once, and what arrives later is drained and discarded (and
    /// not forwarded), in chunks of a fixed size. The test keeps a handle on the
    /// shared buffer to see that.
    #[test]
    fn an_abandoned_collector_releases_what_it_held_and_discards_what_arrives_later() {
        let (feed, chunks) = mpsc::channel();
        let (dropped, ended) = mpsc::channel();
        let forwarded = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let forwarded = Arc::clone(&forwarded);
            FnSink(move |bytes: &[u8]| {
                forwarded.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            })
        };
        let collector = StderrCollector::start(Fed { chunks, dropped }, Some(sink));
        let shared = Arc::clone(&collector.shared);

        feed.send(b"before the failure\n".to_vec()).unwrap();
        // Wait for the thread to have published it, not for the stderr to end.
        let deadline = Instant::now() + Duration::from_secs(10);
        while lock(&shared).bytes.is_empty() {
            assert!(Instant::now() < deadline, "the chunk was never published");
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(lock(&shared).bytes, b"before the failure\n");

        drop(collector);
        assert!(lock(&shared).bytes.is_empty(), "what was held is released");
        assert!(lock(&shared).abandoned);

        for _ in 0..50 {
            feed.send(vec![b'x'; 4096]).unwrap();
        }
        drop(feed);
        ended
            .recv_timeout(Duration::from_secs(10))
            .expect("the thread kept draining until the pipe closed");
        let retained = lock(&shared).bytes.len();
        assert_eq!(
            retained, 0,
            "bytes retained after the capture was abandoned"
        );
        assert_eq!(
            forwarded.lock().unwrap().as_slice(),
            b"before the failure\n",
            "nothing is forwarded after the request has ended"
        );
    }

    /// The multi-byte text the tests split: two, three and four byte characters.
    const MULTIBYTE: &str = "caf\u{e9} \u{2192} \u{1f4e6} done\n";

    /// A stream split at every possible boundary is forwarded whole: every chunk
    /// handed to the sink is valid UTF-8 on its own, and together they are the
    /// input. Pure, so it runs on the Windows job too.
    #[test]
    fn text_split_across_chunks_is_forwarded_whole_and_valid_at_every_boundary() {
        let bytes = MULTIBYTE.as_bytes();
        for first in 1..bytes.len() {
            for second in first..bytes.len() {
                let mut carry = Utf8Carry::default();
                let mut forwarded = Vec::new();
                for part in [&bytes[..first], &bytes[first..second], &bytes[second..]] {
                    let out = carry.push(part);
                    assert!(
                        std::str::from_utf8(&out).is_ok(),
                        "split {first}/{second}: a chunk was not valid UTF-8: {out:?}"
                    );
                    forwarded.extend(out);
                }
                forwarded.extend(carry.finish());
                assert_eq!(forwarded, bytes, "split {first}/{second}: bytes were lost");
            }
        }
    }

    /// Bytes that can never be text are replaced, and a character cut off by the
    /// end of the stream is flushed lossily, never dropped.
    #[test]
    fn bytes_that_are_not_text_are_replaced_and_a_cut_off_tail_is_flushed() {
        let mut carry = Utf8Carry::default();
        assert_eq!(
            carry.push(b"ok \xff\xfe done"),
            "ok \u{fffd}\u{fffd} done".as_bytes()
        );
        // A lead byte of a three-byte character with nothing after it.
        assert_eq!(carry.push(b"tail \xe2\x86"), b"tail ");
        assert_eq!(carry.finish(), "\u{fffd}".as_bytes());
        assert!(carry.finish().is_empty());
    }

    /// The collector forwards a character split across reads in one piece, and
    /// keeps the raw bytes for the error. The source hands out one byte at a time.
    #[test]
    fn the_collector_forwards_a_split_character_whole_and_keeps_the_raw_bytes() {
        struct OneByte(std::vec::IntoIter<u8>);
        impl Read for OneByte {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                match self.0.next() {
                    Some(byte) => {
                        buffer[0] = byte;
                        Ok(1)
                    }
                    None => Ok(0),
                }
            }
        }
        let chunks = std::cell::RefCell::new(Vec::<Vec<u8>>::new());
        let sink = FnSink(|bytes: &[u8]| {
            chunks.borrow_mut().push(bytes.to_vec());
            Ok(bytes.len())
        });
        let collected = collect_stderr(
            OneByte(MULTIBYTE.as_bytes().to_vec().into_iter()),
            Some(sink),
        );
        assert_eq!(collected, MULTIBYTE.as_bytes());
        let chunks = chunks.into_inner();
        assert!(
            chunks
                .iter()
                .all(|chunk| std::str::from_utf8(chunk).is_ok())
        );
        assert_eq!(chunks.concat(), MULTIBYTE.as_bytes());
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

    /// A source that does not answer until released, then answers once and ends.
    struct Gated {
        release: mpsc::Receiver<()>,
        released: bool,
    }
    impl Read for Gated {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.released {
                return Ok(0);
            }
            self.release.recv().unwrap();
            self.released = true;
            buffer[..5].copy_from_slice(b"late\n");
            Ok(5)
        }
    }

    /// A collector whose stderr is still open at the deadline reports what had
    /// arrived and is not waited for; the reading thread is left alone and the
    /// rest is there once the stderr does end.
    #[test]
    fn a_collector_stops_waiting_at_its_deadline_and_keeps_collecting() {
        let (release, released) = mpsc::channel();
        let collector = StderrCollector::start(
            Gated {
                release: released,
                released: false,
            },
            None::<io::Sink>,
        );
        let started = Instant::now();
        let early = collector.finish(Some(Instant::now() + Duration::from_millis(50)));
        assert!(early.is_empty());
        assert!(started.elapsed() < Duration::from_secs(5));

        release.send(()).unwrap();
        assert_eq!(collector.finish(None), b"late\n");
    }

    /// An expired deadline returns at once with what is there, however much is
    /// queued behind it.
    #[test]
    fn an_expired_deadline_returns_what_is_collected_without_waiting() {
        let (release, released) = mpsc::channel();
        let collector = StderrCollector::start(
            Gated {
                release: released,
                released: false,
            },
            None::<io::Sink>,
        );
        let started = Instant::now();
        assert!(collector.finish(Some(Instant::now())).is_empty());
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(release);
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
                "{\"protocol_version\":\"0.1.0\",\"method\":\"stop\",\
                 \"payload\":{\"service_id\":\"svc\"}}\n"
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
                    withhold_inherited_key_file: false,
                    request_progress: true,
                    forward_stderr: false,
                    note_failing_exit_after_answer: false,
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

        /// The status of a child that answered and then failed is not part of the
        /// answer, so a caller that asks for it (the daemon) gets a note on this
        /// process's stderr, and one that does not (the CLI) gets nothing. Run in
        /// a re-executed test binary so the parent can read that stderr; the
        /// decision itself is unchanged: the answer is used.
        #[test]
        fn a_failing_exit_after_an_answer_is_noted_only_when_asked_for() {
            if let Some(arm) = std::env::var_os("ROCM_TEST_CLIENT_NOTE_ARM") {
                let engine = FakeEngine::answering("client-note", STOPPED, 7);
                let options = EngineStdioOptions {
                    note_failing_exit_after_answer: arm == "on",
                    ..EngineStdioOptions::default()
                };
                let answer = run(&engine, &options).unwrap();
                assert_eq!(answer, json!({"stopped": true}));
                return;
            }
            for (arm, expected) in [("on", true), ("off", false)] {
                let output = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "stdio_client::tests::child::a_failing_exit_after_an_answer_is_noted_only_when_asked_for",
                        "--exact",
                        "--nocapture",
                    ])
                    .env("ROCM_TEST_CLIENT_NOTE_ARM", arm)
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{arm}: {:?}", output.status);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let note = "engine stdio process exited with code 7 after printing its answer; the answer was used";
                assert_eq!(stderr.contains(note), expected, "{arm}:\n{stderr}");
            }
        }

        /// The note is only made for an answer that parsed: a child that prints
        /// something unreadable and fails is reported by the error (with its
        /// status), and must not also log that "the answer was used". Both cases
        /// assert the outcome and the stderr together, in a re-executed binary.
        #[test]
        fn the_note_is_made_only_for_an_answer_that_parsed() {
            if let Some(arm) = std::env::var_os("ROCM_TEST_CLIENT_NOTE_PARSE_ARM") {
                let (body, status) = if arm == "parsed" {
                    (format!("printf '%s' '{STOPPED}'"), 7)
                } else {
                    ("printf 'not json'".to_owned(), 7)
                };
                let engine =
                    FakeEngine::new("client-note-parse", true, &format!("{body}\nexit {status}"));
                let options = EngineStdioOptions {
                    note_failing_exit_after_answer: true,
                    ..EngineStdioOptions::default()
                };
                let outcome = run(&engine, &options);
                if arm == "parsed" {
                    assert_eq!(outcome.unwrap(), json!({"stopped": true}));
                } else {
                    assert_eq!(
                        outcome.unwrap_err().to_string(),
                        "failed to parse engine response envelope \
                         (engine stdio process exited with code 7): the engine's stdout \
                         is not JSON (expected ident at line 1 column 2), starting with \
                         `not json` (an engine writes only its JSON answer to stdout; logs \
                         go to stderr)"
                    );
                }
                return;
            }
            for (arm, expected) in [("parsed", true), ("unparseable", false)] {
                let output = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "stdio_client::tests::child::the_note_is_made_only_for_an_answer_that_parsed",
                        "--exact",
                        "--nocapture",
                    ])
                    .env("ROCM_TEST_CLIENT_NOTE_PARSE_ARM", arm)
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{arm}: {:?}", output.status);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert_eq!(
                    stderr.contains("the answer was used"),
                    expected,
                    "{arm}:\n{stderr}"
                );
            }
        }

        /// With stderr a closed pipe and SIGPIPE at its default action (how `rocm`
        /// runs, and the daemon with it), the note must not end the process or
        /// panic: the answer is still returned. Re-executes this test in a process
        /// whose stderr is a pipe with no reader.
        #[test]
        #[allow(unsafe_code)] // libc::signal, to install the default action in the child
        fn the_note_to_a_closed_stderr_neither_kills_nor_panics_the_process() {
            if std::env::var_os("ROCM_TEST_CLIENT_NOTE_CLOSED_ARM").is_some() {
                // SAFETY: installing a signal disposition is always safe.
                unsafe {
                    libc::signal(libc::SIGPIPE, libc::SIG_DFL);
                }
                let engine = FakeEngine::answering("client-note-closed", STOPPED, 7);
                let options = EngineStdioOptions {
                    note_failing_exit_after_answer: true,
                    ..EngineStdioOptions::default()
                };
                let answer = run(&engine, &options).expect("the answer is still returned");
                assert_eq!(answer, json!({"stopped": true}));
                return;
            }
            let (reader, writer) = io::pipe().unwrap();
            drop(reader);
            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "stdio_client::tests::child::the_note_to_a_closed_stderr_neither_kills_nor_panics_the_process",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_CLIENT_NOTE_CLOSED_ARM", "1")
                .stderr(Stdio::from(writer))
                .status()
                .unwrap();
            assert_eq!(
                status.signal(),
                None,
                "SIGPIPE ended the process: {status:?}"
            );
            assert!(status.success(), "{status:?}");
        }

        /// A child whose stdin or stdout pipe is missing is killed and reaped, as
        /// on every other failure path, not left running. Linux-only for the
        /// `/proc` entry a zombie would keep.
        #[cfg(target_os = "linux")]
        #[test]
        fn a_missing_pipe_kills_and_reaps_the_child() {
            let mut child = Command::new("sleep")
                .arg("60")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            let proc_entry = format!("/proc/{}", child.id());
            assert!(Path::new(&proc_entry).exists());

            let error =
                require_pipe(&mut child, None::<std::process::ChildStdin>, "stdin").unwrap_err();

            assert_eq!(error.to_string(), "engine stdio child did not expose stdin");
            assert!(!Path::new(&proc_entry).exists(), "killed and reaped");
        }

        /// A pipe that is there is handed back and the child left alone.
        #[test]
        fn a_present_pipe_is_returned_and_the_child_left_running() {
            let mut child = Command::new("sleep")
                .arg("60")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            assert_eq!(require_pipe(&mut child, Some(5), "stdout").unwrap(), 5);
            assert!(child.try_wait().unwrap().is_none());
            child.kill().unwrap();
            child.wait().unwrap();
        }

        /// A failed read of the answer reports what the engine had said on
        /// stderr, with the read error, like every other failure; the message and
        /// its content are asserted together.
        #[test]
        fn a_failed_read_of_the_answer_reports_the_stderr_too() {
            let collector = StderrCollector::start(
                io::Cursor::new(b"engine crashed: out of memory\n".to_vec()),
                None::<io::Sink>,
            );
            // The thread ends at once on a cursor; wait for its text.
            let error = read_failure_with_stderr(
                anyhow::Error::new(io::Error::other("pipe went away"))
                    .context("failed to read engine stdio response"),
                Some(collector),
                Duration::from_secs(5),
            );
            assert_eq!(
                error.to_string(),
                "failed to read engine stdio response: pipe went away; stderr: engine crashed: out of memory"
            );

            // With nothing on stderr the error is the read error, unchanged.
            let quiet = read_failure_with_stderr(
                anyhow::anyhow!("failed to read engine stdio response"),
                None,
                Duration::from_secs(1),
            );
            assert_eq!(quiet.to_string(), "failed to read engine stdio response");
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
                "failed to parse engine response envelope: the engine's stdout is not JSON (expected ident at line 1 column 2), \
                 starting with `not json` (an engine writes only its JSON answer to stdout; \
                 logs go to stderr); stderr: oops"
            );

            let silent = FakeEngine::new("client-garbage-quiet", true, "printf 'not json'");
            let error = run(&silent, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope: the engine's stdout is not JSON (expected ident at line 1 column 2), \
                 starting with `not json` (an engine writes only its JSON answer to stdout; \
                 logs go to stderr)"
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
                 (engine stdio process was terminated by signal 9): the engine's stdout is \
                 not JSON (EOF while parsing an object at line 1 column 5), starting with \
                 `{\"ok\"` (an engine writes only its JSON answer to stdout; logs go to \
                 stderr)"
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
                 (engine stdio process exited with code 4): the engine's stdout is not JSON (expected ident at line 1 column 2), \
                 starting with `not json` (an engine writes only its JSON answer to stdout; \
                 logs go to stderr); stderr: oops"
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
                 (engine stdio process exited with code 5): the engine's stdout is not JSON (expected ident at line 1 column 2), \
                 starting with `not json` (an engine writes only its JSON answer to stdout; \
                 logs go to stderr)"
            );

            let clean = FakeEngine::new("client-garbage-clean", true, "printf 'not json'\nexit 0");
            let error = run(&clean, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to parse engine response envelope: the engine's stdout is not JSON (expected ident at line 1 column 2), \
                 starting with `not json` (an engine writes only its JSON answer to stdout; \
                 logs go to stderr)"
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

        /// An engine that fills its stderr before it has drained a request larger
        /// than a pipe buffer does not deadlock the client: the stderr is read from
        /// the start, so the engine's write completes while the request is still
        /// being written. (With the collector started after the write, the engine
        /// blocks on a full stderr pipe and the client on a full stdin pipe.)
        #[test]
        fn an_engine_that_writes_a_lot_of_stderr_before_reading_its_request_does_not_deadlock() {
            let engine = FakeEngine::new(
                "client-stderr-first",
                false,
                &format!(
                    "head -c 300000 /dev/zero | tr '\\0' x >&2\ncat > /dev/null\nprintf '%s' '{STOPPED}'"
                ),
            );
            let envelope =
                engine_request_envelope(EngineMethod::Stop, &json!({"blob": "y".repeat(1 << 20)}))
                    .unwrap();
            let program = engine.path().to_path_buf();
            let answer = finishes_within(Duration::from_secs(20), move || {
                run_engine_stdio(
                    &program,
                    ["stdio"],
                    &EngineStdioOptions::default(),
                    &envelope,
                )
            })
            .expect("the engine answered");
            assert!(answer.ok);
            drop(engine);
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
                let mut child = child;
                let stderr = child
                    .stderr
                    .take()
                    .map(|pipe| StderrCollector::start(pipe, None::<io::Sink>));
                abandon_unwritten_request(
                    child,
                    stderr,
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
        /// and the rest is not waited for. The diagnostic has no trailing newline,
        /// as an engine's last words often do not, and is still reported: the
        /// collector publishes what has arrived, not only complete lines.
        #[test]
        fn the_stderr_of_a_failed_write_is_not_waited_for_past_the_grace() {
            let mut child = Command::new("sh")
                .args(["-c", "printf said >&2; sleep 5 & exit 3"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            // The engine has said its piece and exited before the failure is
            // handled, as it has in the case this bounds.
            child.wait().unwrap();
            let started = Instant::now();

            let stderr = child
                .stderr
                .take()
                .map(|pipe| StderrCollector::start(pipe, None::<io::Sink>));
            let error = finishes_within(Duration::from_secs(4), move || {
                abandon_unwritten_request(
                    child,
                    stderr,
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

        /// What a well-behaved engine answers an `install` with, in this protocol.
        const INSTALLED: &str =
            r#"{"protocol_version":"0.1.0","ok":true,"data":{"installed":true}}"#;

        /// A fake engine that answers `capabilities` by running
        /// `capabilities_body` (shell; it must exit) and anything else with
        /// [`INSTALLED`].
        fn engine_answering_capabilities_with(label: &str, capabilities_body: &str) -> FakeEngine {
            FakeEngine::new(
                label,
                true,
                &format!(
                    "if grep -q '\"method\":\"capabilities\"' \"$dir/stdin\"; then\n\
                     {capabilities_body}\n\
                     fi\n\
                     printf '%s' '{INSTALLED}'"
                ),
            )
        }

        fn send(engine: &FakeEngine, method: EngineMethod) -> Result<Value> {
            let envelope = engine_request_envelope(method, &json!({"runtime_id": "r"})).unwrap();
            decode_engine_response(run_engine_stdio(
                engine.path(),
                ["stdio"],
                &EngineStdioOptions::default(),
                &envelope,
            )?)
        }

        /// The capabilities answer that lets the request through: this
        /// protocol, a newer one with only additions, or none at all (an engine
        /// older than the field). Each time the engine is run twice, the
        /// capabilities request first, and the install or launch reaches it.
        #[test]
        fn a_compatible_engine_is_checked_and_then_sent_the_request() {
            for (row, capabilities) in [
                (
                    "same",
                    r#"{"protocol_version":"0.1.0","ok":true,"data":{}}"#,
                ),
                (
                    "additive",
                    r#"{"protocol_version":"0.1.9","ok":true,"data":{}}"#,
                ),
                ("legacy", r#"{"ok":true,"data":{}}"#),
            ] {
                for method in [EngineMethod::Install, EngineMethod::Launch] {
                    let engine = engine_answering_capabilities_with(
                        &format!("handshake-ok-{row}"),
                        &format!("printf '%s' '{capabilities}'\nexit 0"),
                    );
                    let answer = send(&engine, method.clone())
                        .unwrap_or_else(|error| panic!("{row} {method:?}: {error:#}"));
                    assert_eq!(answer, json!({"installed": true}), "{row}");
                    assert_eq!(
                        engine.methods(),
                        ["capabilities", method.name()],
                        "{row}: the request must follow the check"
                    );
                    assert_eq!(
                        engine.requests()[0]["protocol_version"],
                        ENGINE_PROTOCOL_VERSION,
                        "{row}: the check is itself a versioned request"
                    );
                }
            }
        }

        /// Every way the check can fail stops the request before it is sent:
        /// the error says it was not sent and why, naming both versions where
        /// the engine named one, and the engine was only ever asked for its
        /// capabilities.
        #[test]
        fn a_request_that_changes_the_machine_is_not_sent_when_the_check_fails() {
            const NOT_SENT: &str = "the install request was not sent to the engine";
            const UNCHECKED: &str = "the install request was not sent to the engine: its \
                 protocol could not be checked against this release's (engine protocol 0.1.0) \
                 because";
            let rows: [(&str, String, String); 8] = [
                (
                    "incompatible",
                    r#"printf '%s' '{"protocol_version":"9.0.0","ok":true,"data":{}}'"#.to_owned(),
                    format!(
                        "{NOT_SENT}: the engine speaks engine protocol 9.0.0, which this release \
                         (engine protocol 0.1.0) cannot use"
                    ),
                ),
                (
                    "incompatible-refusal",
                    r#"printf '%s' '{"protocol_version":"0.2.0","ok":false,"error":{"code":"incompatible_protocol","message":"no"}}'"#
                        .to_owned(),
                    format!(
                        "{NOT_SENT}: the engine speaks engine protocol 0.2.0, which this release \
                         (engine protocol 0.1.0) cannot use"
                    ),
                ),
                (
                    "unreadable-version",
                    r#"printf '%s' '{"protocol_version":"banana","ok":true,"data":{}}'"#.to_owned(),
                    format!(
                        "{NOT_SENT}: the engine named its protocol version \"banana\", which is \
                         not a MAJOR.MINOR.PATCH version, so this release (engine protocol 0.1.0) \
                         cannot tell whether it can use it"
                    ),
                ),
                (
                    "refused",
                    r#"printf '%s' '{"protocol_version":"0.1.0","ok":false,"error":{"code":"broken","message":"no capabilities today"}}'"#
                        .to_owned(),
                    format!(
                        "{UNCHECKED} it refused the capabilities request: broken: no capabilities \
                         today"
                    ),
                ),
                (
                    "crashed",
                    "echo 'engine exploded' >&2\nexit 3".to_owned(),
                    format!(
                        "{UNCHECKED} asking for its capabilities failed: engine stdio process \
                         exited with code 3: engine exploded"
                    ),
                ),
                (
                    "silent",
                    "exit 0".to_owned(),
                    format!(
                        "{UNCHECKED} asking for its capabilities failed: failed to parse engine \
                         response envelope: the engine printed no answer on stdout"
                    ),
                ),
                (
                    "contaminated",
                    r#"echo 'INFO starting'
printf '%s' '{"protocol_version":"0.1.0","ok":true,"data":{}}'"#
                        .to_owned(),
                    format!(
                        "{UNCHECKED} asking for its capabilities failed: failed to parse engine \
                         response envelope: the engine wrote other output to stdout alongside its \
                         answer, starting with `INFO starting` (an engine writes only its JSON \
                         answer to stdout; logs go to stderr)"
                    ),
                ),
                (
                    "not-json",
                    "printf 'not json'".to_owned(),
                    format!(
                        "{UNCHECKED} asking for its capabilities failed: failed to parse engine \
                         response envelope: the engine's stdout is not JSON (expected ident at \
                         line 1 column 2), starting with `not json` (an engine writes only its \
                         JSON answer to stdout; logs go to stderr)"
                    ),
                ),
            ];
            for (row, capabilities_body, expected) in rows {
                let engine = engine_answering_capabilities_with(
                    &format!("handshake-refused-{row}"),
                    &format!("{capabilities_body}\nexit 0"),
                );
                let error = send(&engine, EngineMethod::Install).unwrap_err();
                assert_eq!(format!("{error:#}"), expected, "{row}");
                assert_eq!(
                    engine.methods(),
                    ["capabilities"],
                    "{row}: the install must never reach the engine"
                );
            }
        }

        /// The requests that do not change the machine are sent as they are,
        /// once, and their answer's version is what is checked.
        #[test]
        fn a_request_that_changes_nothing_is_sent_without_a_check() {
            for method in [
                EngineMethod::Stop,
                EngineMethod::Healthcheck,
                EngineMethod::Detect,
                EngineMethod::ResolveModel,
                EngineMethod::Endpoint,
                EngineMethod::Logs,
                EngineMethod::Capabilities,
            ] {
                let engine = FakeEngine::answering("no-handshake", INSTALLED, 0);
                send(&engine, method.clone()).unwrap();
                assert_eq!(engine.methods(), [method.name()], "{method:?}");
            }
            let engine = FakeEngine::answering(
                "no-handshake-incompatible",
                r#"{"protocol_version":"9.0.0","ok":true,"data":{"stopped":true}}"#,
                0,
            );
            let error = send(&engine, EngineMethod::Stop).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine answered in engine protocol 9.0.0, which this release \
                 (protocol 0.1.0) cannot use"
            );
            assert_eq!(engine.methods(), ["stop"]);
        }

        /// What an engine leaves on stdout besides its answer makes the answer
        /// unreadable, and the error says so; a pretty-printed answer over many
        /// lines is still one answer.
        #[test]
        fn output_besides_the_answer_on_stdout_is_refused_and_named() {
            let rule = "(an engine writes only its JSON answer to stdout; logs go to stderr)";
            let logged = FakeEngine::new(
                "contaminated-before",
                true,
                &format!("echo 'INFO loading model'\nprintf '%s' '{STOPPED}'"),
            );
            assert_eq!(
                run(&logged, &EngineStdioOptions::default())
                    .unwrap_err()
                    .to_string(),
                format!(
                    "failed to parse engine response envelope: the engine wrote other output to \
                     stdout alongside its answer, starting with `INFO loading model` {rule}"
                )
            );

            let trailing = FakeEngine::new(
                "contaminated-after",
                true,
                &format!("printf '%s\\n' '{STOPPED}'\necho done >&1\necho oops >&2"),
            );
            assert_eq!(
                run(&trailing, &EngineStdioOptions::default())
                    .unwrap_err()
                    .to_string(),
                format!(
                    "failed to parse engine response envelope: the engine wrote other output to \
                     stdout alongside its answer, starting with `{STOPPED}` {rule}; stderr: oops"
                )
            );

            let not_an_envelope = FakeEngine::answering("not-an-envelope", r#"{"status":"ok"}"#, 0);
            assert_eq!(
                run(&not_an_envelope, &EngineStdioOptions::default())
                    .unwrap_err()
                    .to_string(),
                format!(
                    "failed to parse engine response envelope: the engine's stdout is JSON but \
                     not an answer envelope (missing field `ok` at line 1 column 15), starting \
                     with `{{\"status\":\"ok\"}}` {rule}"
                )
            );

            let empty = FakeEngine::new("empty-stdout", true, "exit 0");
            assert_eq!(
                run(&empty, &EngineStdioOptions::default())
                    .unwrap_err()
                    .to_string(),
                "failed to parse engine response envelope: the engine printed no answer on stdout"
            );

            let pretty = FakeEngine::new(
                "pretty-printed",
                true,
                "printf '{\\n  \"protocol_version\": \"0.1.0\",\\n  \"ok\": true,\\n  \"data\": {\\n    \"stopped\": true\\n  }\\n}\\n'",
            );
            assert_eq!(
                run(&pretty, &EngineStdioOptions::default()).unwrap(),
                json!({"stopped": true})
            );
        }
    }
}
