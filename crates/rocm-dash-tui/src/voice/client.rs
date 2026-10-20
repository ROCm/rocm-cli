// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Talking to the speech servers `rocm serve` started: sending a spoken turn
//! for transcription, and asking for a reply to be spoken.

use std::time::Duration;

use anyhow::{Context, Result, bail};

/// A running speech server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechService {
    pub service_id: String,
    /// The OpenAI-compatible API root, already ending in `/v1`.
    pub base_url: String,
    pub model: String,
}

/// The speech servers voice can use right now.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeechServices {
    pub transcription: Option<SpeechService>,
    pub speech: Option<SpeechService>,
}

/// Transcribe a spoken turn. Returns the text, trimmed.
pub async fn transcribe(service: &SpeechService, wav: Vec<u8>) -> Result<String> {
    const BOUNDARY: &str = "rocm-voice-boundary-41c9";
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{}\r\n\
         --{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"turn.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\n",
        service.model
    )
    .into_bytes();
    body.extend_from_slice(&wav);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let response = reqwest::Client::new()
        .post(format!("{}/audio/transcriptions", service.base_url.trim_end_matches('/')))
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(body)
        .timeout(Duration::from_mins(5))
        .send()
        .await
        .with_context(|| format!("could not reach the speech-to-text server `{}`", service.service_id))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "the speech-to-text server `{}` refused the recording ({status}): {}",
            service.service_id,
            text.trim()
        );
    }
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("the speech-to-text server `{}` did not answer with JSON", service.service_id))?;
    Ok(value
        .get("text")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned())
}

/// Ask a text-to-speech server to say `text`. Returns a WAV file.
pub async fn speak(service: &SpeechService, text: &str) -> Result<Vec<u8>> {
    let response = reqwest::Client::new()
        .post(format!("{}/audio/speech", service.base_url.trim_end_matches('/')))
        .json(&serde_json::json!({
            "model": service.model,
            "input": text,
            "response_format": "wav",
        }))
        .timeout(Duration::from_mins(5))
        .send()
        .await
        .with_context(|| format!("could not reach the text-to-speech server `{}`", service.service_id))?;
    let status = response.status();
    let bytes = response.bytes().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "the text-to-speech server `{}` refused the reply ({status}): {}",
            service.service_id,
            String::from_utf8_lossy(&bytes).trim()
        );
    }
    Ok(bytes.to_vec())
}

/// Whether a transcript is worth sending to the assistant. Whisper answers
/// silence with a marker such as `[BLANK_AUDIO]`; that is not a request.
#[must_use]
pub fn is_meaningful_transcript(text: &str) -> bool {
    let text = text.trim();
    !text.is_empty()
        && !(text.starts_with('[') && text.ends_with(']'))
        && !(text.starts_with('(') && text.ends_with(')'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_markers_are_not_sent_to_the_assistant() {
        assert!(!is_meaningful_transcript(" [BLANK_AUDIO]\n"));
        assert!(!is_meaningful_transcript("(silence)"));
        assert!(!is_meaningful_transcript("  "));
        assert!(is_meaningful_transcript("Is my GPU ready?"));
    }
}
