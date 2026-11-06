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

use std::ffi::OsStr;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

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
/// non-zero is read as that envelope, which says more than the status does.
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
        serde_json::to_writer(&mut stdin, envelope).context("failed to write engine request")?;
        stdin.write_all(b"\n")?;
        Ok::<(), anyhow::Error>(())
    });
    if let Err(error) = written {
        // The engine will never get its request, so do not leave it running or
        // unreaped: a long-lived caller would collect one zombie per failure.
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    // End of the request: the engine reads to EOF before it answers.
    drop(stdin);

    let forward_stderr = options.forward_stderr;
    let stderr_reader = child.stderr.take().map(|stderr| {
        thread::spawn(move || collect_stderr(stderr, forward_stderr.then(io::stderr)))
    });
    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .context("engine stdio child did not expose stdout")?
        .read_to_end(&mut stdout)
        .context("failed to read engine stdio response")?;
    let status = child
        .wait()
        .context("failed waiting for engine stdio response")?;
    let stderr = stderr_reader
        .map(|reader| reader.join().unwrap_or_default())
        .unwrap_or_default();

    if !status.success() && stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {status}");
        }
        bail!("engine stdio process exited with status {status}: {stderr}");
    }
    serde_json::from_slice(&stdout).with_context(|| {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        if stderr.is_empty() {
            "failed to parse engine response envelope".to_owned()
        } else {
            format!("failed to parse engine response envelope; stderr: {stderr}")
        }
    })
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

    /// A stderr whose reads fail.
    struct Fails;
    impl Read for Fails {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("pipe went away"))
        }
    }

    /// A sink that refuses every write, like a closed terminal.
    struct Closed;
    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
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
        let collected = collect_stderr(&b"still kept\n"[..], Some(Closed));
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
                "engine stdio process exited with status exit status: 3: boom"
            );

            let silent = FakeEngine::new("client-silent-crash", true, "exit 4");
            let error = run(&silent, &EngineStdioOptions::default()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "engine stdio process exited with status exit status: 4"
            );
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
