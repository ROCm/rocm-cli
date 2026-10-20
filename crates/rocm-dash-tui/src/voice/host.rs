// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether this host can capture and play audio, and with which tool.
//!
//! Audio goes through the system's own command-line tools rather than a linked
//! audio library, so the `rocm` binary still starts on a server with no sound
//! stack at all. Every reason voice cannot work here is turned into a message
//! that names a step which actually fixes it.

use std::path::{Path, PathBuf};

use super::wav::{SampleFormat, Wav};

/// The facts about this host that decide whether voice can work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostAudio {
    pub windows: bool,
    pub wsl: bool,
    /// The SSH session this process runs in, if any (from `SSH_CONNECTION`).
    pub ssh: bool,
    /// Whether WSLg's PulseAudio socket exists (`/mnt/wslg/PulseServer`).
    pub wslg_audio: bool,
    pub recorder: Option<Recorder>,
    pub player: Option<Player>,
}

/// A tool that streams microphone samples to its standard output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorder {
    pub program: PathBuf,
    pub kind: AudioTool,
}

/// A tool that plays samples from its standard input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Player {
    pub program: PathBuf,
    pub kind: AudioTool,
}

/// The audio stacks voice knows how to drive, in the order they are tried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTool {
    /// PipeWire's `pw-record` / `pw-play` (package `pipewire-bin`).
    PipeWire,
    /// PulseAudio's `parecord` / `paplay` (package `pulseaudio-utils`); also
    /// what WSLg's audio server speaks.
    PulseAudio,
    /// ALSA's `arecord` / `aplay` (package `alsa-utils`).
    Alsa,
}

const RECORDERS: [(&str, AudioTool); 3] = [
    ("pw-record", AudioTool::PipeWire),
    ("parecord", AudioTool::PulseAudio),
    ("arecord", AudioTool::Alsa),
];
const PLAYERS: [(&str, AudioTool); 3] = [
    ("pw-play", AudioTool::PipeWire),
    ("paplay", AudioTool::PulseAudio),
    ("aplay", AudioTool::Alsa),
];

impl HostAudio {
    /// Probe the running host. `wsl` comes from the caller, which has the
    /// project's WSL detection; an environment variable alone is not trusted.
    #[must_use]
    pub fn detect(wsl: bool) -> Self {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let find = |name: &str| {
            std::env::split_paths(&path)
                .map(|dir| dir.join(name))
                .find(|candidate| candidate.is_file())
        };
        Self {
            windows: cfg!(windows),
            wsl,
            ssh: std::env::var_os("SSH_CONNECTION").is_some(),
            wslg_audio: Path::new("/mnt/wslg/PulseServer").exists(),
            recorder: RECORDERS.iter().find_map(|(name, kind)| {
                find(name).map(|program| Recorder {
                    program,
                    kind: *kind,
                })
            }),
            player: PLAYERS.iter().find_map(|(name, kind)| {
                find(name).map(|program| Player {
                    program,
                    kind: *kind,
                })
            }),
        }
    }

    /// Why the microphone cannot be used here, with the step that fixes it;
    /// `None` when it can.
    #[must_use]
    pub fn microphone_problem(&self) -> Option<String> {
        if self.windows {
            return Some(
                "voice input from the microphone is not available on native Windows yet; \
                 run `rocm talk` inside WSL, or pass a recording with --voice-input <file.wav>"
                    .to_owned(),
            );
        }
        if self.ssh {
            return Some(
                "this session runs over SSH, so the microphone would be the remote machine's, \
                 not yours; run `rocm talk` on the machine you speak into, or pass a recording \
                 with --voice-input <file.wav>"
                    .to_owned(),
            );
        }
        if self.wsl && !self.wslg_audio {
            return Some(
                "WSL's audio server is not available (/mnt/wslg/PulseServer is missing); \
                 update WSL from Windows with `wsl --update`, then `wsl --shutdown` and \
                 open the distribution again"
                    .to_owned(),
            );
        }
        if self.recorder.is_none() {
            return Some(
                "no microphone recording tool was found; install one with \
                 `sudo apt install pulseaudio-utils` (or pipewire-bin, or alsa-utils)"
                    .to_owned(),
            );
        }
        None
    }

    /// Why spoken replies cannot be played here; `None` when they can.
    #[must_use]
    pub fn speaker_problem(&self) -> Option<String> {
        if self.windows {
            return Some(
                "spoken replies are not available on native Windows yet; replies are shown \
                 as text"
                    .to_owned(),
            );
        }
        if self.player.is_none() {
            return Some(
                "no audio playback tool was found, so replies are shown as text; install one \
                 with `sudo apt install pulseaudio-utils` (or pipewire-bin, or alsa-utils)"
                    .to_owned(),
            );
        }
        None
    }
}

impl Recorder {
    /// Arguments that stream 16 kHz mono 16-bit samples to standard output.
    #[must_use]
    pub fn args(&self) -> Vec<String> {
        let args: &[&str] = match self.kind {
            AudioTool::PipeWire => &["--rate", "16000", "--channels", "1", "--format", "s16", "-"],
            AudioTool::PulseAudio => &[
                "--raw",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
            ],
            AudioTool::Alsa => &["-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1"],
        };
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }
}

impl Player {
    /// Arguments that play `wav`'s samples, given as raw PCM on standard input.
    /// The format is stated explicitly so no tool has to guess at a container.
    #[must_use]
    pub fn args(&self, wav: &Wav<'_>) -> Vec<String> {
        let rate = wav.sample_rate.to_string();
        let channels = wav.channels.to_string();
        match (self.kind, wav.format) {
            (AudioTool::PipeWire, format) => vec![
                "--rate".to_owned(),
                rate,
                "--channels".to_owned(),
                channels,
                "--format".to_owned(),
                match format {
                    SampleFormat::S16 => "s16",
                    SampleFormat::F32 => "f32",
                }
                .to_owned(),
                "-".to_owned(),
            ],
            (AudioTool::PulseAudio, format) => vec![
                "--raw".to_owned(),
                format!(
                    "--format={}",
                    match format {
                        SampleFormat::S16 => "s16le",
                        SampleFormat::F32 => "float32le",
                    }
                ),
                format!("--rate={rate}"),
                format!("--channels={channels}"),
            ],
            (AudioTool::Alsa, format) => vec![
                "-q".to_owned(),
                "-t".to_owned(),
                "raw".to_owned(),
                "-f".to_owned(),
                match format {
                    SampleFormat::S16 => "S16_LE",
                    SampleFormat::F32 => "FLOAT_LE",
                }
                .to_owned(),
                "-r".to_owned(),
                rate,
                "-c".to_owned(),
                channels,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linux_desktop() -> HostAudio {
        HostAudio {
            recorder: Some(Recorder {
                program: PathBuf::from("/usr/bin/parecord"),
                kind: AudioTool::PulseAudio,
            }),
            player: Some(Player {
                program: PathBuf::from("/usr/bin/paplay"),
                kind: AudioTool::PulseAudio,
            }),
            ..HostAudio::default()
        }
    }

    #[test]
    fn a_linux_desktop_with_a_recorder_can_use_the_microphone() {
        assert_eq!(linux_desktop().microphone_problem(), None);
        assert_eq!(linux_desktop().speaker_problem(), None);
    }

    #[test]
    fn each_blocker_names_a_step_that_fixes_it() {
        let ssh = HostAudio {
            ssh: true,
            ..linux_desktop()
        };
        assert!(
            ssh.microphone_problem()
                .is_some_and(|m| m.contains("over SSH") && m.contains("--voice-input"))
        );
        let wsl_without_audio = HostAudio {
            wsl: true,
            ..linux_desktop()
        };
        assert!(
            wsl_without_audio
                .microphone_problem()
                .is_some_and(|m| m.contains("wsl --update"))
        );
        let wsl_with_audio = HostAudio {
            wsl: true,
            wslg_audio: true,
            ..linux_desktop()
        };
        assert_eq!(wsl_with_audio.microphone_problem(), None);
        let no_tools = HostAudio::default();
        assert!(
            no_tools
                .microphone_problem()
                .is_some_and(|m| m.contains("sudo apt install pulseaudio-utils"))
        );
        assert!(no_tools.speaker_problem().is_some());
        let windows = HostAudio {
            windows: true,
            ..linux_desktop()
        };
        assert!(
            windows
                .microphone_problem()
                .is_some_and(|m| m.contains("--voice-input"))
        );
    }

    #[test]
    fn every_recorder_streams_16khz_mono_16_bit_samples() {
        for kind in [AudioTool::PipeWire, AudioTool::PulseAudio, AudioTool::Alsa] {
            let args = Recorder {
                program: PathBuf::from("rec"),
                kind,
            }
            .args()
            .join(" ");
            assert!(args.contains("16000"), "{kind:?}: {args}");
            assert!(
                args.contains("s16") || args.contains("S16_LE"),
                "{kind:?}: {args}"
            );
        }
    }

    #[test]
    fn players_are_told_the_reply_format_instead_of_guessing() {
        let wav = Wav {
            format: SampleFormat::F32,
            sample_rate: 24_000,
            channels: 1,
            samples: &[],
        };
        let player = |kind| Player {
            program: PathBuf::from("play"),
            kind,
        };
        assert_eq!(
            player(AudioTool::PulseAudio).args(&wav),
            ["--raw", "--format=float32le", "--rate=24000", "--channels=1"]
        );
        assert!(player(AudioTool::Alsa).args(&wav).contains(&"FLOAT_LE".to_owned()));
        assert!(player(AudioTool::PipeWire).args(&wav).contains(&"f32".to_owned()));
    }
}
