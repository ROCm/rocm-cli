// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm transcribe`: send an audio file to a running speech-to-text server
//! and print what was said.
//!
//! The file is read and uploaded; nothing is written to disk, and no audio is
//! kept. The server is one `rocm serve` started — found through the same
//! readiness check `rocm chat` uses, so only a service that actually answered a
//! transcription is offered.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord, ModelTask, MultipartFile};

/// The command that starts a speech-to-text server, named whenever none runs.
const SERVE_SPEECH_MODEL_HINT: &str = "rocm serve whisper";

/// Uploads larger than this are refused before anything is read into memory.
const MAX_AUDIO_BYTES: u64 = 512 * 1024 * 1024;

/// Transcribing is real inference over the whole recording, so the request gets
/// far longer than a readiness probe: an hour of audio on a small GPU can take
/// minutes.
const TRANSCRIBE_TIMEOUT: Duration = Duration::from_mins(15);

#[derive(Debug, Clone)]
pub(crate) struct TranscribeArgs {
    pub(crate) file: PathBuf,
    pub(crate) model: Option<String>,
    pub(crate) service: Option<String>,
    pub(crate) language: Option<String>,
    pub(crate) json: bool,
}

pub(crate) fn transcribe(args: TranscribeArgs) -> Result<()> {
    let audio = read_audio(&args.file)?;
    let paths = AppPaths::discover()?;
    let services = crate::providers::ready_local_services(&paths)?;
    let service =
        select_transcription_service(&services, args.service.as_deref(), args.model.as_deref())?;
    let model = if service.canonical_model_id.trim().is_empty() {
        service.model_ref.as_str()
    } else {
        service.canonical_model_id.as_str()
    };
    let mut fields = vec![("model", model)];
    if let Some(language) = args.language.as_deref() {
        fields.push(("language", language));
    }
    let filename = args
        .file
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio");
    let endpoint_api_key = crate::endpoint_keys::endpoint_api_key(&paths, &service.service_id);
    let response = rocm_core::http_post_multipart_with_auth(
        &service.endpoint_url,
        "/v1/audio/transcriptions",
        &fields,
        MultipartFile {
            field: "file",
            filename,
            content_type: audio_content_type(&args.file),
            bytes: &audio,
        },
        endpoint_api_key.as_deref(),
        TRANSCRIBE_TIMEOUT,
    )
    .with_context(|| {
        format!(
            "failed to reach the speech-to-text server `{}` at {}",
            service.service_id, service.endpoint_url
        )
    })?;
    print!(
        "{}",
        render_transcription(
            &service.service_id,
            response.status,
            &response.body,
            args.json
        )?
    );
    Ok(())
}

fn read_audio(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path).with_context(|| format!("cannot read {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{} is not a file", path.display());
    }
    if metadata.len() > MAX_AUDIO_BYTES {
        bail!(
            "{} is {} MiB; the limit is {} MiB - split the recording first",
            path.display(),
            metadata.len() / (1024 * 1024),
            MAX_AUDIO_BYTES / (1024 * 1024)
        );
    }
    fs::read(path).with_context(|| format!("cannot read {}", path.display()))
}

/// The upload's media type, from the file extension. Only WAV is verified
/// against Lemonade's whisper.cpp server; other types are passed through for
/// the server to accept or refuse.
fn audio_content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("flac") => "audio/flac",
        Some("ogg" | "oga") => "audio/ogg",
        Some("m4a" | "mp4") => "audio/mp4",
        Some("webm") => "audio/webm",
        _ => "application/octet-stream",
    }
}

/// Pick the speech-to-text service to send to.
///
/// `services` are the ready local services. An explicit `--service` wins and
/// must be a speech-to-text one; `--model` narrows to the speech-to-text
/// services serving it; otherwise exactly one must be running.
fn select_transcription_service<'a>(
    services: &'a [ManagedServiceRecord],
    service_id: Option<&str>,
    model: Option<&str>,
) -> Result<&'a ManagedServiceRecord> {
    if let Some(service_id) = service_id {
        let Some(service) = services
            .iter()
            .find(|service| service.service_id == service_id)
        else {
            bail!(
                "no running server is called `{service_id}`; `rocm services list` shows the \
                 ones that are"
            );
        };
        if service.task != ModelTask::Transcription {
            bail!(
                "`{service_id}` serves {} requests, not speech-to-text; start a speech-to-text \
                 server with `{SERVE_SPEECH_MODEL_HINT}`",
                service.task.display_name()
            );
        }
        return Ok(service);
    }
    let speech: Vec<&ManagedServiceRecord> = services
        .iter()
        .filter(|service| service.task == ModelTask::Transcription)
        .filter(|service| {
            model.is_none_or(|model| {
                service.canonical_model_id.eq_ignore_ascii_case(model)
                    || service.model_ref.eq_ignore_ascii_case(model)
            })
        })
        .collect();
    match (speech.as_slice(), model) {
        ([only], _) => Ok(only),
        ([], Some(model)) => bail!(
            "no running speech-to-text server is serving `{model}`; start one with \
             `rocm serve {model}`"
        ),
        ([], None) => {
            bail!("no speech-to-text server is running; start one with `{SERVE_SPEECH_MODEL_HINT}`")
        }
        (several, _) => bail!(
            "{} speech-to-text servers are running ({}); pass --service <id> to choose one",
            several.len(),
            several
                .iter()
                .map(|service| service.service_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The text printed for a server's answer: the transcript alone, or the
/// server's JSON with `--json`.
fn render_transcription(service_id: &str, status: u16, body: &str, json: bool) -> Result<String> {
    if status != 200 {
        bail!(
            "the speech-to-text server `{service_id}` refused the recording (HTTP {status}): {}",
            body.trim()
        );
    }
    let value: serde_json::Value = serde_json::from_str(body.trim()).with_context(|| {
        format!("the speech-to-text server `{service_id}` did not answer with JSON")
    })?;
    if json {
        return Ok(format!("{}\n", serde_json::to_string_pretty(&value)?));
    }
    let text = value
        .get("text")
        .and_then(serde_json::Value::as_str)
        .with_context(|| {
            format!("the speech-to-text server `{service_id}` answered without a transcript")
        })?;
    Ok(format!("{}\n", text.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service(id: &str, model: &str, task: ModelTask) -> ManagedServiceRecord {
        let paths = AppPaths {
            config_dir: PathBuf::from("/nonexistent/config"),
            data_dir: PathBuf::from("/nonexistent/data"),
            cache_dir: PathBuf::from("/nonexistent/cache"),
        };
        let mut record = ManagedServiceRecord::new(
            &paths,
            id,
            "lemonade",
            model,
            model,
            "127.0.0.1",
            11435,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.task = task;
        record
    }

    #[test]
    fn the_only_speech_to_text_service_is_chosen_over_chat_services() {
        let services = [
            service("svc-chat", "qwen", ModelTask::Chat),
            service("svc-whisper", "Whisper-Tiny", ModelTask::Transcription),
        ];
        let chosen = select_transcription_service(&services, None, None).expect("one");
        assert_eq!(chosen.service_id, "svc-whisper");
    }

    #[test]
    fn no_speech_to_text_service_names_the_command_that_starts_one() {
        let services = [service("svc-chat", "qwen", ModelTask::Chat)];
        let error = select_transcription_service(&services, None, None)
            .expect_err("a chat service cannot transcribe")
            .to_string();
        // `rocm serve whisper` resolves to a speech-to-text recipe served by
        // Lemonade; serve-27 runs the same path end to end on a GPU lane.
        assert_eq!(
            error,
            "no speech-to-text server is running; start one with `rocm serve whisper`"
        );
        let recipe = rocm_core::resolve_builtin_model_recipe("whisper").expect("whisper recipe");
        assert_eq!(recipe.model_task().expect("task"), ModelTask::Transcription);
    }

    #[test]
    fn several_speech_to_text_services_need_an_explicit_choice() {
        let services = [
            service("svc-a", "Whisper-Tiny", ModelTask::Transcription),
            service("svc-b", "Whisper-Large-v3-Turbo", ModelTask::Transcription),
        ];
        let error = select_transcription_service(&services, None, None)
            .expect_err("ambiguous")
            .to_string();
        assert!(error.contains("2 speech-to-text servers are running (svc-a, svc-b)"));
        assert_eq!(
            select_transcription_service(&services, Some("svc-b"), None)
                .expect("explicit")
                .service_id,
            "svc-b"
        );
        assert_eq!(
            select_transcription_service(&services, None, Some("whisper-tiny"))
                .expect("by model")
                .service_id,
            "svc-a"
        );
    }

    #[test]
    fn naming_a_chat_service_is_refused_with_the_reason() {
        let services = [service("svc-chat", "qwen", ModelTask::Chat)];
        let error = select_transcription_service(&services, Some("svc-chat"), None)
            .expect_err("not speech-to-text")
            .to_string();
        assert!(error.contains("`svc-chat` serves chat requests, not speech-to-text"));
    }

    #[test]
    fn the_transcript_is_printed_alone_unless_json_is_asked_for() {
        let body = r#"{"text":" And so my fellow Americans.\n"}"#;
        assert_eq!(
            render_transcription("svc", 200, body, false).expect("text"),
            "And so my fellow Americans.\n"
        );
        let json = render_transcription("svc", 200, body, true).expect("json");
        assert!(json.contains("\"text\""));
        let error = render_transcription("svc", 400, "{\"error\":\"bad audio\"}", false)
            .expect_err("refused")
            .to_string();
        assert!(
            error.contains("refused the recording (HTTP 400)"),
            "{error}"
        );
    }

    #[test]
    fn audio_type_follows_the_extension() {
        assert_eq!(audio_content_type(Path::new("a.WAV")), "audio/wav");
        assert_eq!(audio_content_type(Path::new("a.mp3")), "audio/mpeg");
        assert_eq!(
            audio_content_type(Path::new("a")),
            "application/octet-stream"
        );
    }

    #[test]
    fn a_missing_file_is_reported_before_any_server_is_looked_up() {
        let error = read_audio(Path::new("/nonexistent/recording.wav"))
            .expect_err("missing")
            .to_string();
        assert!(
            error.contains("cannot read /nonexistent/recording.wav"),
            "{error}"
        );
    }
}
