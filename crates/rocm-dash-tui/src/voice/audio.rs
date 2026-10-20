// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Capturing a spoken turn and playing a spoken reply.
//!
//! Both stay in memory: the recorder's samples are read from its standard
//! output into a buffer, and a reply's samples are written to the player's
//! standard input. No audio is written to disk at any point.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

use super::host::{Player, Recorder};
use super::wav;

/// Where a spoken turn comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceSource {
    Microphone(Recorder),
    /// A recording given with `--voice-input`, read in place of the microphone.
    File(PathBuf),
}

/// A turn being captured. Dropping it stops the recorder.
#[derive(Debug)]
pub struct Capture {
    inner: CaptureInner,
}

#[derive(Debug)]
enum CaptureInner {
    Microphone {
        child: Child,
        samples: Arc<Mutex<Vec<u8>>>,
        reader: tokio::task::JoinHandle<()>,
    },
    File(PathBuf),
}

/// Recordings longer than this are cut off; a push-to-talk turn left running
/// must not grow without bound.
const MAX_TURN_SECONDS: usize = 120;
const MAX_TURN_BYTES: usize = MAX_TURN_SECONDS * wav::CAPTURE_SAMPLE_RATE as usize * 2;

impl Capture {
    /// Start capturing from `source`.
    pub fn start(source: &VoiceSource) -> Result<Self> {
        match source {
            VoiceSource::File(path) => Ok(Self {
                inner: CaptureInner::File(path.clone()),
            }),
            VoiceSource::Microphone(recorder) => {
                let mut child = Command::new(&recorder.program)
                    .args(recorder.args())
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .with_context(|| {
                        format!("could not start {}", recorder.program.display())
                    })?;
                let mut stdout = child.stdout.take().context("the recorder has no output")?;
                let samples = Arc::new(Mutex::new(Vec::new()));
                let sink = Arc::clone(&samples);
                let reader = tokio::spawn(async move {
                    let mut chunk = [0_u8; 8192];
                    while let Ok(read) = stdout.read(&mut chunk).await {
                        if read == 0 {
                            break;
                        }
                        let mut samples = sink.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        if samples.len() + read > MAX_TURN_BYTES {
                            break;
                        }
                        samples.extend_from_slice(&chunk[..read]);
                    }
                });
                Ok(Self {
                    inner: CaptureInner::Microphone {
                        child,
                        samples,
                        reader,
                    },
                })
            }
        }
    }

    /// Stop capturing and return the turn as a WAV file in memory.
    pub async fn finish(self) -> Result<Vec<u8>> {
        match self.inner {
            CaptureInner::File(path) => read_recording(&path).await,
            CaptureInner::Microphone {
                mut child,
                samples,
                reader,
            } => {
                // A recorder that already exited never captured anything: say
                // why, from its own error output.
                if let Ok(Some(status)) = child.try_wait() {
                    let mut stderr = String::new();
                    if let Some(mut pipe) = child.stderr.take() {
                        let _ = pipe.read_to_string(&mut stderr).await;
                    }
                    bail!(
                        "the microphone recorder stopped ({status}): {}",
                        stderr.trim()
                    );
                }
                let _ = child.kill().await;
                let _ = reader.await;
                let captured = std::mem::take(
                    &mut *samples
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                let pcm = wav::samples_of_capture(&captured);
                if pcm.is_empty() {
                    bail!("no sound was captured from the microphone");
                }
                Ok(wav::encode_s16_mono(pcm, wav::CAPTURE_SAMPLE_RATE))
            }
        }
    }
}

async fn read_recording(path: &Path) -> Result<Vec<u8>> {
    tokio::fs::read(path)
        .await
        .with_context(|| format!("cannot read the recording {}", path.display()))
}

/// Play a spoken reply (a WAV file in memory) to the end.
pub async fn play(player: &Player, reply: &[u8]) -> Result<()> {
    let parsed = wav::parse(reply)?;
    let mut child = Command::new(&player.program)
        .args(player.args(&parsed))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("could not start {}", player.program.display()))?;
    let mut stdin = child.stdin.take().context("the player has no input")?;
    let written = stdin.write_all(parsed.samples).await;
    drop(stdin);
    let output = child.wait_with_output().await?;
    if !output.status.success() {
        bail!(
            "the audio player stopped ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    written.context("could not hand the reply to the audio player")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::voice::host::AudioTool;

    /// A stand-in recorder: prints a fixed burst of samples, then waits to be
    /// stopped, like a real one would.
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("chmod script");
        path
    }

    fn temp_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rocm-voice-audio-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[tokio::test]
    async fn a_captured_turn_becomes_a_wav_file_in_memory() {
        let dir = temp_dir();
        let recorder = Recorder {
            program: script(&dir, "rec", "printf 'abcdefgh'; sleep 30"),
            kind: AudioTool::PulseAudio,
        };
        let capture = Capture::start(&VoiceSource::Microphone(recorder)).expect("start");
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let turn = capture.finish().await.expect("finish");
        let parsed = wav::parse(&turn).expect("a WAV file");
        assert_eq!(parsed.samples, b"abcdefgh");
        assert_eq!(parsed.sample_rate, 16_000);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_recorder_that_fails_to_start_reports_its_own_reason() {
        let dir = temp_dir();
        let recorder = Recorder {
            program: script(
                &dir,
                "rec",
                "echo 'pa_context_connect() failed: Connection refused' >&2; exit 1",
            ),
            kind: AudioTool::PulseAudio,
        };
        let capture = Capture::start(&VoiceSource::Microphone(recorder)).expect("start");
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let error = capture.finish().await.expect_err("nothing captured").to_string();
        assert!(error.contains("Connection refused"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_reply_is_played_as_raw_samples_with_its_format_stated() {
        let dir = temp_dir();
        let received = dir.join("received");
        let player = Player {
            program: script(
                &dir,
                "play",
                &format!("echo \"$@\" > {0}.args; cat > {0}", received.display()),
            ),
            kind: AudioTool::PulseAudio,
        };
        let reply = wav::encode_s16_mono(b"wxyz", 22_050);
        play(&player, &reply).await.expect("played");
        assert_eq!(std::fs::read(&received).expect("samples"), b"wxyz");
        let args = std::fs::read_to_string(dir.join("received.args")).expect("args");
        assert!(args.contains("--rate=22050"), "{args}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
