// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for serving speech-to-text models and for the commands that must not
//! mistake one for a chat model.

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{MOCK_TRANSCRIPT, MockServer, ServiceRecordOptions};

use crate::E2eWorld;

/// The speech-to-text model the scenarios use: the smallest Whisper recipe, so
/// the GPU scenario's download stays small.
const SPEECH_MODEL_REF: &str = "whisper-tiny";
const SPEECH_MODEL_ID: &str = "Whisper-Tiny";

fn cli_text(world: &E2eWorld) -> String {
    format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

fn record_cli(world: &mut E2eWorld, (stdout, stderr, rc): (String, String, i32)) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// One second of 16 kHz mono 16-bit silence, as a WAV file.
fn silent_wav() -> Vec<u8> {
    let data_len: u32 = 32_000;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&16_000_u32.to_le_bytes());
    wav.extend_from_slice(&32_000_u32.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(44 + data_len as usize, 0);
    wav
}

/// POST a recording to an OpenAI-compatible transcription URL, the way an app
/// would, and return the HTTP status and parsed body.
async fn transcribe(url: &str, model: &str) -> (u16, serde_json::Value) {
    const BOUNDARY: &str = "e2e-speech-boundary";
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
         --{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(&silent_wav());
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let response = reqwest::Client::new()
        .post(url)
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(body)
        .timeout(std::time::Duration::from_mins(2))
        .send()
        .await
        .unwrap_or_else(|error| panic!("transcription request to {url} failed: {error}"));
    let status = response.status().as_u16();
    let text = response.text().await.unwrap_or_default();
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
    (status, json)
}

/// The URL on the output line `<label>: <url>`, if the CLI printed one.
fn printed_url(output: &str, label: &str) -> Option<String> {
    output.lines().find_map(|line| {
        line.trim()
            .strip_prefix(label)
            .and_then(|rest| rest.strip_prefix(':'))
            .map(|url| url.trim().to_owned())
            .filter(|url| url.starts_with("http"))
    })
}

// ── Given ──────────────────────────────────────────────────────────

/// A speech-to-text server the CLI knows about, as `rocm serve whisper` would
/// leave behind: a Lemonade record carrying the transcription task.
#[given("a speech-to-text model is being served and registered with the CLI")]
async fn speech_model_registered(world: &mut E2eWorld) {
    let mock = MockServer::start_transcription(SPEECH_MODEL_ID).await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some(SPEECH_MODEL_ID.to_owned());
    world.mock = Some(mock);
    world.register_mock_service_with(ServiceRecordOptions {
        task: Some("transcription"),
        engine: Some("lemonade"),
        ..ServiceRecordOptions::default()
    });
}

// ── When ───────────────────────────────────────────────────────────

/// vLLM serves chat models only. The refusal comes from the engine's own
/// advertised capabilities, checked before the GPU gate, so this needs no GPU.
#[when("the user serves a speech-to-text model on an engine that only serves chat")]
async fn serve_speech_model_on_chat_only_engine(world: &mut E2eWorld) {
    let result = crate::run_rocm(world, &["serve", SPEECH_MODEL_REF, "--engine", "vllm"]);
    record_cli(world, result);
}

/// Exactly the command the chat-only refusal recommends, so this scenario is
/// also the proof that the recommendation works.
#[when("the user serves a speech-to-text model with the engine the CLI recommended")]
async fn serve_speech_model_on_lemonade(world: &mut E2eWorld) {
    crate::e2e::serving_steps::ensure_serve_port_free().await;
    let result = crate::run_rocm(world, &["serve", SPEECH_MODEL_REF, "--engine", "lemonade"]);
    record_cli(world, result);
    world.model_name = Some(SPEECH_MODEL_ID.to_owned());
}

#[when("the user asks the chat CLI to use the speech-to-text model")]
async fn chat_with_speech_model(world: &mut E2eWorld) {
    let result = crate::run_rocm(
        world,
        &[
            "chat",
            "--provider",
            "local",
            "--model",
            SPEECH_MODEL_ID,
            "--prompt",
            "Hello",
        ],
    );
    record_cli(world, result);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("serving is refused because that engine cannot serve speech-to-text models")]
async fn refused_for_engine(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no serve rc recorded");
    let text = cli_text(world);
    assert_ne!(rc, 0, "serving must be refused:\n{text}");
    assert!(
        text.contains("is a speech-to-text model, which the vllm engine cannot serve"),
        "the refusal must give the reason:\n{text}"
    );
    assert!(
        !text.to_lowercase().contains("no usable amd gpu"),
        "the reason must not be hidden behind the GPU check:\n{text}"
    );
}

#[then("the user is told the command that serves it with Lemonade")]
async fn told_lemonade_command(world: &mut E2eWorld) {
    let text = cli_text(world);
    // serve-27 runs this exact command on a GPU lane and transcribes through it.
    assert!(
        text.contains(&format!("rocm serve {SPEECH_MODEL_REF} --engine lemonade")),
        "expected the working command in:\n{text}"
    );
}

#[then("the CLI prints the speech-to-text endpoint of the served model")]
async fn prints_speech_endpoint(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no serve rc recorded");
    let text = cli_text(world);
    assert_eq!(rc, 0, "serving the speech-to-text model failed:\n{text}");
    let url = printed_url(
        world.cli_output.as_deref().unwrap_or_default(),
        "speech-to-text endpoint",
    )
    .unwrap_or_else(|| panic!("no speech-to-text endpoint printed:\n{text}"));
    assert!(
        url.ends_with("/v1/audio/transcriptions"),
        "the printed endpoint must be the transcription route, got {url}"
    );
    world.endpoint = Some(url);
}

#[then("a recording sent to that endpoint is transcribed")]
async fn recording_transcribed_at_printed_endpoint(world: &mut E2eWorld) {
    let url = world
        .endpoint
        .clone()
        .expect("no printed endpoint recorded");
    let model = world.model_name.clone().expect("no model name recorded");
    let (status, body) = transcribe(&url, &model).await;
    assert_eq!(status, 200, "transcription at {url} failed: {body}");
    assert!(
        body.get("text").is_some_and(serde_json::Value::is_string),
        "a transcription response carries its text: {body}"
    );
}

#[then("the service is listed as speech-to-text with its transcription endpoint")]
async fn listed_as_speech(world: &mut E2eWorld) {
    let output = world.cli_output.clone().expect("no services output");
    assert!(
        output.contains("task: speech-to-text"),
        "the services list must say what the server does:\n{output}"
    );
    let url =
        printed_url(&output, "endpoint").unwrap_or_else(|| panic!("no endpoint listed:\n{output}"));
    assert!(
        url.ends_with("/v1/audio/transcriptions"),
        "a speech-to-text server must be listed with the route it serves, got {url}"
    );
    world.endpoint = Some(url);
}

#[then("a recording sent to the listed endpoint is transcribed")]
async fn recording_transcribed_at_listed_endpoint(world: &mut E2eWorld) {
    let url = world.endpoint.clone().expect("no listed endpoint recorded");
    let (status, body) = transcribe(&url, SPEECH_MODEL_ID).await;
    assert_eq!(status, 200, "transcription at {url} failed: {body}");
    assert_eq!(body["text"], MOCK_TRANSCRIPT);
    let mock = world.mock.as_ref().expect("no mock server");
    assert!(
        mock.transcription_paths()
            .iter()
            .any(|path| path == "/v1/audio/transcriptions"),
        "the request must have reached the server's transcription route"
    );
}

#[then("chat is refused because the model is a speech-to-text model")]
async fn chat_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no chat rc recorded");
    let text = cli_text(world);
    assert_ne!(rc, 0, "chat must refuse a speech-to-text model:\n{text}");
    assert!(
        text.contains("as a speech-to-text model, not a chat model"),
        "the refusal must give the reason:\n{text}"
    );
    // That serving the built-in assistant makes `rocm chat` work is chat-03's
    // and chat-08's subject; here it is enough that the advice names it.
    assert!(
        text.contains("rocm serve qwen"),
        "expected the remedy in:\n{text}"
    );
}

#[then("no chat request reached the speech-to-text server")]
async fn no_chat_request_reached(world: &mut E2eWorld) {
    let mock = world.mock.as_ref().expect("no mock server");
    assert_eq!(
        mock.chat_paths(),
        Vec::<String>::new(),
        "the CLI must not send chat requests to a speech-to-text server"
    );
}

#[then("the benchmark is refused because the model is a speech-to-text model")]
async fn bench_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no bench rc recorded");
    let text = cli_text(world);
    assert_ne!(
        rc, 0,
        "benchmarking a speech-to-text model must be refused:\n{text}"
    );
    assert!(
        text.contains("is a speech-to-text model; `rocm bench load` measures chat completions"),
        "the refusal must give the reason:\n{text}"
    );
}
