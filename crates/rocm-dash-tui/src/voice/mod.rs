// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Voice mode for the chat pane (`rocm talk`).
//!
//! Voice is only another way to type a message. A push-to-talk turn is
//! captured, transcribed by the speech-to-text server `rocm serve` started, and
//! submitted through the chat pane's own submit path — the same agent loop, the
//! same tools, and the same approval card as a typed message. Nothing a voice
//! turn asks for runs without a key press on that card.
//!
//! A reply to a voice turn is shown in full and, when a text-to-speech server
//! is running, a short summary of it is spoken. No audio is written to disk.

pub mod audio;
pub mod client;
pub mod host;
pub mod speakable;
pub mod wav;

use std::sync::Arc;

pub use audio::VoiceSource;
pub use client::{SpeechService, SpeechServices};

/// Looks up the running speech servers when a turn needs one. Supplied by the
/// binary, which owns the service registry and its readiness checks.
#[derive(Clone)]
pub struct SpeechServiceResolver(Arc<dyn Fn() -> SpeechServices + Send + Sync>);

impl SpeechServiceResolver {
    pub fn new(resolve: impl Fn() -> SpeechServices + Send + Sync + 'static) -> Self {
        Self(Arc::new(resolve))
    }

    #[must_use]
    pub fn resolve(&self) -> SpeechServices {
        (self.0)()
    }
}

impl std::fmt::Debug for SpeechServiceResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SpeechServiceResolver")
    }
}

/// What voice mode starts with.
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// Where turns come from, or why none can be captured here.
    pub input: Result<VoiceSource, String>,
    /// How replies are played, or why they cannot be.
    pub player: Result<host::Player, String>,
    pub services: SpeechServiceResolver,
}

/// Where a voice turn is, for the chat pane's indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoiceState {
    #[default]
    Idle,
    Listening,
    Transcribing,
    Speaking,
}

impl VoiceState {
    /// The indicator line shown under the transcript while voice mode is on.
    #[must_use]
    pub const fn indicator(self) -> &'static str {
        match self {
            Self::Idle => "Voice: press F9 to talk",
            Self::Listening => "* Listening - press F9 to stop",
            Self::Transcribing => "Transcribing...",
            Self::Speaking => "Speaking... (F9 to talk again)",
        }
    }
}

/// The text shown when a turn needs a speech-to-text server and none runs.
pub const NO_TRANSCRIPTION_SERVER: &str =
    "No speech-to-text server is running. Start one with `rocm serve whisper`, then press F9 again.";

/// The note shown once when replies cannot be spoken because no text-to-speech
/// server runs.
pub const NO_SPEECH_SERVER: &str =
    "Replies are shown as text. To hear them, start a text-to-speech server with `rocm serve tts`.";
