// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! WAV in memory: wrapping captured samples for upload, and reading the
//! samples out of a server's spoken reply for playback. Nothing here touches the
//! filesystem.

use anyhow::{Result, bail};

/// The format speech-to-text servers take without resampling, and the one the
/// microphone is captured in.
pub const CAPTURE_SAMPLE_RATE: u32 = 16_000;

/// Sample encodings a spoken reply arrives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Signed 16-bit little-endian integers.
    S16,
    /// 32-bit little-endian floats (Lemonade's Kokoro answers in this).
    F32,
}

/// The parts of a WAV file a player needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wav<'a> {
    pub format: SampleFormat,
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: &'a [u8],
}

/// Wrap 16-bit mono little-endian samples captured at `sample_rate` as a WAV
/// file.
#[must_use]
pub fn encode_s16_mono(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    // Keeps the RIFF size field (`36 + data_len`) inside `u32`.
    let data_len = u32::try_from(pcm.len())
        .unwrap_or(u32::MAX - 44)
        .min(u32::MAX - 44)
        & !1;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1_u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&pcm[..data_len as usize]);
    wav
}

/// Read the format and samples out of a WAV file.
///
/// Tolerates the streaming form servers send, whose size fields are
/// `0xFFFFFFFF` because the length was not known when the header went out: the
/// data chunk then runs to the end of the file.
pub fn parse(bytes: &[u8]) -> Result<Wav<'_>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("the spoken reply is not a WAV file");
    }
    let mut at = 12;
    let mut format = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into()?) as usize;
        let body = at + 8;
        if id == b"fmt " {
            if body + 16 > bytes.len() {
                bail!("the spoken reply has a truncated format header");
            }
            let tag = u16::from_le_bytes(bytes[body..body + 2].try_into()?);
            let channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into()?);
            let sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into()?);
            let bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into()?);
            let sample_format = match (tag, bits) {
                (1, 16) => SampleFormat::S16,
                (3, 32) => SampleFormat::F32,
                _ => bail!("the spoken reply uses an unsupported sample format ({tag}, {bits} bit)"),
            };
            format = Some((sample_format, sample_rate, channels));
        } else if id == b"data" {
            let Some((format, sample_rate, channels)) = format else {
                bail!("the spoken reply has samples before its format header");
            };
            let end = body.saturating_add(size).min(bytes.len());
            return Ok(Wav {
                format,
                sample_rate,
                channels,
                samples: &bytes[body..end],
            });
        }
        at = body.saturating_add(size + (size & 1));
    }
    bail!("the spoken reply has no audio samples")
}

/// Strip a WAV header a recorder may have written ahead of its samples, so a
/// tool that writes raw PCM and one that writes a container both yield samples.
#[must_use]
pub fn samples_of_capture(captured: &[u8]) -> &[u8] {
    match parse(captured) {
        Ok(wav) if wav.format == SampleFormat::S16 => wav.samples,
        _ => captured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_samples_round_trip_through_a_wav_file() {
        let pcm: Vec<u8> = (0..200_u8).collect();
        let wav = encode_s16_mono(&pcm, CAPTURE_SAMPLE_RATE);
        let parsed = parse(&wav).expect("our own WAV parses");
        assert_eq!(parsed.format, SampleFormat::S16);
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples, &pcm[..]);
    }

    #[test]
    fn a_streaming_float_reply_with_unknown_sizes_is_read_to_the_end() {
        // The header Lemonade's Kokoro sent: RIFF and data sizes 0xFFFFFFFF,
        // 32-bit float samples (format tag 3).
        let mut wav = b"RIFF\xff\xff\xff\xffWAVEfmt ".to_vec();
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&3_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&24_000_u32.to_le_bytes());
        wav.extend_from_slice(&96_000_u32.to_le_bytes());
        wav.extend_from_slice(&4_u16.to_le_bytes());
        wav.extend_from_slice(&32_u16.to_le_bytes());
        wav.extend_from_slice(b"data\xff\xff\xff\xff");
        wav.extend_from_slice(&[0_u8; 16]);
        let parsed = parse(&wav).expect("streaming WAV parses");
        assert_eq!(parsed.format, SampleFormat::F32);
        assert_eq!(parsed.sample_rate, 24_000);
        assert_eq!(parsed.samples.len(), 16);
    }

    #[test]
    fn a_recorder_that_writes_a_container_and_one_that_writes_raw_both_give_samples() {
        let pcm = vec![7_u8; 64];
        assert_eq!(samples_of_capture(&encode_s16_mono(&pcm, 16_000)), &pcm[..]);
        assert_eq!(samples_of_capture(&pcm), &pcm[..]);
    }

    #[test]
    fn something_that_is_not_a_wav_file_is_refused() {
        assert!(parse(b"{\"error\":\"model not loaded\"}").is_err());
    }
}
