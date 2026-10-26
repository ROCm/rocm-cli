// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! API docs for every running model server, served by the daemon on loopback.
//!
//! Of the servers rocm-cli runs, only vLLM generates its own docs; Lemonade and
//! the llama.cpp server behind it serve none. So the daemon describes each
//! running server from the routes rocm-cli knows it serves ([`routes_for`]),
//! renders that with Swagger UI, and forwards the page's "Try it out" requests
//! to the server through `/services/<id>/api/...`. Forwarding keeps try-it
//! same-origin, so it works on every engine whatever that engine's own
//! cross-origin settings are; Lemonade, for one, only admits browser requests
//! from its own origin.
//!
//! The server list is read from the service records on every request, so a
//! server started or stopped while the daemon runs needs no notification.

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{any, get};
use rocm_core::ModelTask;
use rocm_core::{AppPaths, DEFAULT_LOCAL_HOST, ManagedServiceRecord};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use utoipa::openapi::path::{HttpMethod, OperationBuilder, PathItem, PathsBuilder};
use utoipa::openapi::request_body::RequestBodyBuilder;
use utoipa::openapi::schema::{KnownFormat, ObjectBuilder, SchemaFormat, Type};
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityRequirement, SecurityScheme};
use utoipa::openapi::{
    ComponentsBuilder, ContentBuilder, InfoBuilder, OpenApi, OpenApiBuilder, Required,
    ResponseBuilder, ServerBuilder,
};

/// Largest request body the try-it forwarder accepts. Generous enough for an
/// audio upload; the docs server is loopback-only, so this bounds memory, not
/// abuse from elsewhere.
const FORWARD_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// How long the forwarder waits for the model server to accept a connection.
/// There is deliberately no overall timeout: a long completion may stream for
/// minutes.
const FORWARD_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The OpenAPI security scheme name used for servers started with an endpoint
/// key; Swagger UI's "Authorize" box fills it in.
const ENDPOINT_KEY_SCHEME: &str = "endpoint_key";

/// A running docs server. It runs on a thread and runtime of its own: the
/// daemon's loop runs watcher checks synchronously on its single-threaded
/// runtime, and a docs page or a streamed try-it answer must not stall for
/// the length of a check.
pub(crate) struct ApiDocsServer {
    pub(crate) base_url: String,
    shutdown: tokio::sync::oneshot::Sender<()>,
    thread: std::thread::JoinHandle<()>,
}

impl ApiDocsServer {
    /// Stop accepting requests and wait for the server thread to finish.
    pub(crate) fn stop(self) {
        let _ = self.shutdown.send(());
        let _ = self.thread.join();
    }
}

#[derive(Clone)]
struct DocsState {
    paths: Arc<AppPaths>,
    client: reqwest::Client,
}

pub(crate) fn start_api_docs_server(paths: AppPaths, port: u16) -> Result<ApiDocsServer> {
    let listener = std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, port)).with_context(|| {
        format!("failed to bind API docs server to {DEFAULT_LOCAL_HOST}:{port}")
    })?;
    let addr = listener
        .local_addr()
        .context("failed to read API docs listener address")?;
    if !addr.ip().is_loopback() {
        bail!("API docs server must bind to loopback; resolved address was {addr}");
    }
    listener
        .set_nonblocking(true)
        .context("failed to configure the API docs listener")?;
    let app = router(DocsState {
        paths: Arc::new(paths),
        client: forwarding_client()?,
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build the API docs runtime")?;
    let (shutdown, shutdown_received) = tokio::sync::oneshot::channel::<()>();
    let thread = std::thread::Builder::new()
        .name("rocmd-api-docs".to_owned())
        .spawn(move || {
            runtime.block_on(async move {
                let served = match TcpListener::from_std(listener) {
                    Ok(listener) => axum::serve(listener, app)
                        .with_graceful_shutdown(async {
                            let _ = shutdown_received.await;
                        })
                        .await
                        .map_err(anyhow::Error::from),
                    Err(error) => Err(error.into()),
                };
                if let Err(error) = served {
                    eprintln!("API docs server stopped: {error}");
                }
            });
        })
        .context("failed to start the API docs thread")?;
    Ok(ApiDocsServer {
        base_url: format!("http://{addr}"),
        shutdown,
        thread,
    })
}

/// The client the forwarder sends try-it requests with.
fn forwarding_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(FORWARD_CONNECT_TIMEOUT)
        // The model server is on this machine. A proxy configured for the
        // user's outbound traffic (`HTTP_PROXY`, `ALL_PROXY`) would carry the
        // prompt and the endpoint key off it, or fail the request outright.
        .no_proxy()
        // A redirect is not followed: the browser gets the model server's own
        // status. The routes documented here never redirect.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("failed to build the API docs forwarding client")
}

fn router(state: DocsState) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/health", get(health))
        .route("/services/{id}/openapi.json", get(service_openapi_json))
        .route("/services/{id}/docs", get(docs_redirect))
        .route("/services/{id}/docs/", get(docs_index))
        .route("/services/{id}/docs/{*file}", get(docs_file))
        .route(
            "/services/{id}/api/{*path}",
            any(forward).layer(DefaultBodyLimit::max(FORWARD_BODY_LIMIT_BYTES)),
        )
        .layer(middleware::from_fn(require_loopback_host))
        .with_state(state)
}

/// Refuse requests whose `Host` is not a loopback name, and browser requests
/// sent from a page on another origin. The listener is already loopback-only;
/// the `Host` check closes DNS rebinding, where a web page resolves its own
/// hostname to 127.0.0.1, and the `Origin` check stops any other page the user
/// has open from posting to a model server through the forwarder. Requests
/// that are not from a browser carry no `Origin` and are not affected.
async fn require_loopback_host(request: Request, next: Next) -> Response {
    let headers = request.headers();
    let host_allowed = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(host_header_is_loopback);
    let origin_allowed = headers
        .get(header::ORIGIN)
        .is_none_or(|value| value.to_str().is_ok_and(origin_is_loopback));
    if host_allowed && origin_allowed {
        next.run(request).await
    } else {
        (
            StatusCode::FORBIDDEN,
            "API docs answer only on 127.0.0.1 or localhost\n",
        )
            .into_response()
    }
}

fn host_header_is_loopback(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or_default()
    } else {
        host.rsplit_once(':').map_or(host, |(name, _)| name)
    };
    name.eq_ignore_ascii_case("localhost")
        || name.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Whether an `Origin` header names a page served from this machine.
fn origin_is_loopback(origin: &str) -> bool {
    origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
        .is_some_and(|authority| !authority.contains('/') && host_header_is_loopback(authority))
}

async fn health() -> impl IntoResponse {
    axum::Json(json!({ "status": "ok", "source": "api_docs", "bind": "loopback_only" }))
}

/// Every server the record says is up. Errors reading the records read as "no
/// servers": the docs page must never take the daemon down.
fn running_services(paths: &AppPaths) -> Vec<ManagedServiceRecord> {
    crate::persistence::load_managed_services(paths)
        .unwrap_or_default()
        .into_iter()
        .filter(service_is_running)
        .collect()
}

fn service_is_running(record: &ManagedServiceRecord) -> bool {
    matches!(
        record.status.as_str(),
        "starting" | "running" | "recovering" | "ready"
    )
}

/// The running server with this id. Looked up by matching the id against the
/// records rather than by building a path from it, so a crafted id cannot
/// reach outside the services folder.
fn find_running_service(paths: &AppPaths, service_id: &str) -> Option<ManagedServiceRecord> {
    running_services(paths)
        .into_iter()
        .find(|record| record.service_id == service_id)
}

fn not_running_page(service_id: &str) -> Response {
    let id = html_escape(service_id);
    (
        StatusCode::NOT_FOUND,
        Html(format!(
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Server not running</title></head>\
             <body><h1>Server {id} is not running</h1>\
             <p>Its API docs are available while it runs. \
             <a href=\"/\">Running servers</a></p></body></html>"
        )),
    )
        .into_response()
}

async fn index_page(State(state): State<DocsState>) -> Html<String> {
    let services = running_services(&state.paths);
    let mut rows = String::new();
    for record in &services {
        let id = html_escape(&record.service_id);
        let _ = write!(
            rows,
            "<li><a href=\"/services/{id}/docs/\">{model}</a> ({engine}, {id}, {endpoint})</li>",
            model = html_escape(&record.canonical_model_id),
            engine = html_escape(&record.engine),
            endpoint = html_escape(&record.endpoint_url),
        );
    }
    let body = if services.is_empty() {
        "<p>No model server is running. Start one with <code>rocm serve</code>.</p>".to_owned()
    } else {
        format!("<ul>{rows}</ul>")
    };
    Html(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>ROCm CLI API docs</title></head>\
         <body><h1>Running model servers</h1>{body}</body></html>"
    ))
}

async fn service_openapi_json(
    State(state): State<DocsState>,
    Path(service_id): Path<String>,
) -> Response {
    match find_running_service(&state.paths, &service_id) {
        Some(record) => axum::Json(service_openapi(&record)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({
                "error": format!("server {service_id} is not running"),
            })),
        )
            .into_response(),
    }
}

async fn docs_redirect(Path(service_id): Path<String>) -> Redirect {
    Redirect::permanent(&format!("/services/{service_id}/docs/"))
}

async fn docs_index(State(state): State<DocsState>, Path(service_id): Path<String>) -> Response {
    serve_swagger_file(&state, &service_id, "")
}

async fn docs_file(
    State(state): State<DocsState>,
    Path((service_id, file)): Path<(String, String)>,
) -> Response {
    serve_swagger_file(&state, &service_id, &file)
}

fn serve_swagger_file(state: &DocsState, service_id: &str, file: &str) -> Response {
    if find_running_service(&state.paths, service_id).is_none() {
        return not_running_page(service_id);
    }
    let config = Arc::new(
        utoipa_swagger_ui::Config::new([format!("/services/{service_id}/openapi.json")])
            // The endpoint key typed into "Authorize" lasts until the page is
            // closed; it is not saved in the browser, where it would outlive
            // the server it belongs to.
            .try_it_out_enabled(true),
    );
    match utoipa_swagger_ui::serve(file, config) {
        Ok(Some(swagger_file)) => (
            [(header::CONTENT_TYPE, swagger_file.content_type)],
            swagger_file.bytes.into_owned(),
        )
            .into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

/// Relay one try-it request to the model server and stream its answer back.
async fn forward(
    State(state): State<DocsState>,
    Path((service_id, _decoded_path)): Path<(String, String)>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(record) = find_running_service(&state.paths, &service_id) else {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(json!({ "error": format!("server {service_id} is not running") })),
        )
            .into_response();
    };
    // The path exactly as the browser sent it. The extracted one is
    // percent-decoded: an encoded `?` or `/` in it would turn into a query or
    // an extra segment, and the request would reach a different route.
    let Some(path) = raw_forwarded_path(uri.path()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut target = format!("{}/{path}", service_base_url(&record));
    if let Some(query) = uri.query() {
        target.push('?');
        target.push_str(query);
    }
    let mut request = state.client.request(method, &target).body(body);
    // Only what the routes need: the body's type, the answer the caller
    // accepts, and the endpoint key the user typed into "Authorize". Nothing
    // the browser adds on its own (cookies, origin) reaches the model server.
    for name in [header::CONTENT_TYPE, header::ACCEPT, header::AUTHORIZATION] {
        if let Some(value) = headers.get(&name) {
            request = request.header(name, value);
        }
    }
    match request.send().await {
        Ok(answer) => {
            let status = answer.status();
            let content_type = answer.headers().get(header::CONTENT_TYPE).cloned();
            let mut response = Response::new(Body::from_stream(answer.bytes_stream()));
            *response.status_mut() = status;
            if let Some(disposition) = content_type
                .as_ref()
                .and_then(|value| value.to_str().ok())
                .and_then(audio_attachment)
            {
                response
                    .headers_mut()
                    .insert(header::CONTENT_DISPOSITION, disposition);
            }
            if let Some(content_type) = content_type {
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, content_type);
            }
            response
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            )],
            json!({
                "error": format!(
                    "could not reach server {service_id} at {}: {error}",
                    service_base_url(&record)
                ),
            })
            .to_string(),
        )
            .into_response(),
    }
}

/// A `Content-Disposition` that makes Swagger UI offer an audio answer as a
/// download. Without it Swagger renders an audio player pointed back at the
/// request URL, which re-sends it as a `GET` the speech route does not answer,
/// so the audio the server made could be neither played nor saved.
fn audio_attachment(content_type: &str) -> Option<HeaderValue> {
    let media_type = content_type.split(';').next()?.trim().to_ascii_lowercase();
    let subtype = media_type.strip_prefix("audio/")?;
    let extension = match subtype {
        "wav" | "x-wav" | "wave" | "vnd.wave" => "wav",
        "mpeg" | "mp3" => "mp3",
        "x-flac" | "flac" => "flac",
        "ogg" | "opus" | "aac" => subtype,
        "l16" | "pcm" => "pcm",
        _ => "audio",
    };
    HeaderValue::from_str(&format!("attachment; filename=\"speech.{extension}\"")).ok()
}

/// The part of `/services/<id>/api/<path>` after `api/`, still percent-encoded.
fn raw_forwarded_path(request_path: &str) -> Option<&str> {
    let mut segments = request_path.splitn(5, '/');
    match (
        segments.next(),
        segments.next(),
        segments.next(),
        segments.next(),
        segments.next(),
    ) {
        (Some(""), Some("services"), Some(_), Some("api"), Some(path)) => Some(path),
        _ => None,
    }
}

/// Where the forwarder connects. A server bound to every interface is reached
/// on loopback, the same address rocm-cli's own readiness checks use.
fn service_base_url(record: &ManagedServiceRecord) -> String {
    let host = match record.host.parse::<IpAddr>() {
        Ok(ip) if ip.is_unspecified() => DEFAULT_LOCAL_HOST,
        _ => record.host.as_str(),
    };
    rocm_core::format_http_base_url(host, record.port)
}

/// One route a model server answers, as the docs describe it.
struct RouteDoc {
    method: HttpMethod,
    path: &'static str,
    summary: &'static str,
    description: &'static str,
    body: RouteBody,
    /// What the server answers with.
    answer_type: &'static str,
}

/// What a route's request carries, as "Try it out" should offer it.
enum RouteBody {
    None,
    /// JSON, with an example built for the server's real model id.
    Json(fn(&str) -> Value),
    /// A multipart audio upload, as the OpenAI transcription API takes it.
    AudioUpload,
}

/// The routes each engine's server answers: its health route plus
/// [`rocm_engine_protocol::task_routes`] for the server's task, the set its
/// plugin reports as `supported_routes`. The single place the docs learn what a
/// server serves; a new kind of server extends this.
fn routes_for(record: &ManagedServiceRecord) -> Vec<RouteDoc> {
    let health_path = if record.engine == "vllm" {
        "/health"
    } else {
        "/v1/health"
    };
    let mut routes = vec![
        RouteDoc {
            method: HttpMethod::Get,
            path: health_path,
            summary: "Check the server is up",
            description: "Answers 200 once the server accepts requests.",
            answer_type: "application/json",
            body: RouteBody::None,
        },
        RouteDoc {
            method: HttpMethod::Get,
            path: "/v1/models",
            summary: "List the served models",
            description: "OpenAI-compatible model list. The `id` of each entry is the \
                          `model` value requests use.",
            answer_type: "application/json",
            body: RouteBody::None,
        },
    ];
    match record.task {
        ModelTask::Chat => routes.extend(chat_routes()),
        ModelTask::Transcription => routes.push(RouteDoc {
            method: HttpMethod::Post,
            path: "/v1/audio/transcriptions",
            summary: "Transcribe speech",
            description: "OpenAI-compatible speech-to-text. Upload an audio file as `file` \
                          (WAV is accepted by every backend); the answer's `text` is the \
                          transcript.",
            answer_type: "application/json",
            body: RouteBody::AudioUpload,
        }),
        ModelTask::Speech => routes.push(RouteDoc {
            method: HttpMethod::Post,
            path: "/v1/audio/speech",
            summary: "Speak text",
            description: "OpenAI-compatible text-to-speech. The answer is the spoken `input` \
                          as a WAV file, which the page offers to download.",
            answer_type: "audio/wav",
            body: RouteBody::Json(speech_example),
        }),
        // A task this build does not know: only the routes every server has.
        ModelTask::Unknown => {}
    }
    routes
}

fn chat_routes() -> [RouteDoc; 2] {
    [
        RouteDoc {
            method: HttpMethod::Post,
            path: "/v1/chat/completions",
            summary: "Chat completion",
            description: "OpenAI-compatible chat completion. Set `stream` to `true` to \
                          receive server-sent events as tokens are generated.",
            answer_type: "application/json",
            body: RouteBody::Json(chat_completion_example),
        },
        RouteDoc {
            method: HttpMethod::Post,
            path: "/v1/completions",
            summary: "Text completion",
            description: "OpenAI-compatible text completion of a raw prompt.",
            answer_type: "application/json",
            body: RouteBody::Json(completion_example),
        },
    ]
}

/// The form "Try it out" shows for a transcription: a file picker for the
/// audio and the model field already filled in.
fn audio_upload_schema(model: &str) -> ObjectBuilder {
    ObjectBuilder::new()
        .property(
            "file",
            ObjectBuilder::new()
                .schema_type(Type::String)
                .format(Some(SchemaFormat::KnownFormat(KnownFormat::Binary)))
                .description(Some("The audio to transcribe, for example a WAV file.")),
        )
        .required("file")
        .property(
            "model",
            ObjectBuilder::new()
                .schema_type(Type::String)
                .default(Some(json!(model))),
        )
        .required("model")
}

fn speech_example(model: &str) -> Value {
    json!({
        "model": model,
        "input": "Hello from ROCm.",
        "response_format": "wav",
    })
}

fn chat_completion_example(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{ "role": "user", "content": "Say hello in one sentence." }],
        "max_tokens": 64,
        "stream": false,
    })
}

fn completion_example(model: &str) -> Value {
    json!({
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 16,
    })
}

/// Where each engine documents more than rocm-cli describes.
fn engine_docs_note(record: &ManagedServiceRecord) -> String {
    let base = service_base_url(record);
    if record.engine == "vllm" {
        format!(
            "vLLM also documents every route it serves, beyond the ones listed here, at \
             [{base}/docs]({base}/docs)."
        )
    } else if !record.task.is_chat() {
        String::new()
    } else {
        "On Linux, Lemonade runs chat models with the llama.cpp server, which also answers its \
         own routes; see the \
         [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)."
            .to_owned()
    }
}

fn service_openapi(record: &ManagedServiceRecord) -> OpenApi {
    let model = record.canonical_model_id.as_str();
    let mut paths = PathsBuilder::new();
    for route in routes_for(record) {
        let mut operation = OperationBuilder::new()
            .summary(Some(route.summary))
            .description(Some(route.description))
            .response(
                "200",
                ResponseBuilder::new()
                    .description("The server's answer")
                    .content(route.answer_type, ContentBuilder::new().build()),
            );
        let request_body = match route.body {
            RouteBody::None => None,
            RouteBody::Json(example_body) => Some(
                RequestBodyBuilder::new()
                    .required(Some(Required::True))
                    .content(
                        "application/json",
                        ContentBuilder::new()
                            .example(Some(example_body(model)))
                            .build(),
                    )
                    .build(),
            ),
            RouteBody::AudioUpload => Some(
                RequestBodyBuilder::new()
                    .required(Some(Required::True))
                    .content(
                        "multipart/form-data",
                        ContentBuilder::new()
                            .schema(Some(audio_upload_schema(model)))
                            .build(),
                    )
                    .build(),
            ),
        };
        operation = operation.request_body(request_body);
        paths = paths.path(route.path, PathItem::new(route.method, operation));
    }

    let description = format!(
        "`{model}` ({task}) served by {engine} at `{endpoint}`.\n\n\
         \"Try it out\" sends requests through this page to the server.\n\n{note}",
        task = record.task.display_name(),
        engine = record.engine,
        endpoint = record.endpoint_url,
        note = engine_docs_note(record),
    );
    let mut builder = OpenApiBuilder::new()
        .info(
            InfoBuilder::new()
                .title(format!("{model} ({})", record.service_id))
                .version(env!("CARGO_PKG_VERSION"))
                .description(Some(description)),
        )
        .servers(Some([ServerBuilder::new()
            .url(format!("/services/{}/api", record.service_id))
            .description(Some(format!(
                "{} through the docs page",
                record.endpoint_url
            )))
            .build()]))
        .paths(paths);
    if record.requires_api_key {
        builder = builder
            .components(Some(
                ComponentsBuilder::new()
                    .security_scheme(
                        ENDPOINT_KEY_SCHEME,
                        SecurityScheme::Http(
                            HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build(),
                        ),
                    )
                    .build(),
            ))
            .security(Some([SecurityRequirement::new(
                ENDPOINT_KEY_SCHEME,
                Vec::<String>::new(),
            )]));
    }
    builder.build()
}

fn html_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            other => escaped.push(other),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use axum::Json;
    use axum::routing::post;
    use rocm_core::RocmCliConfig;
    use std::fs;
    use std::sync::Mutex;

    /// Serializes the tests that set proxy variables. Every other test here
    /// talks to its servers through [`test_client`], which ignores them.
    static PROXY_ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The test's own HTTP client: never proxied, so a proxy variable set by
    /// the proxy test cannot reroute it.
    fn test_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    fn record(paths: &AppPaths, service_id: &str, engine: &str, port: u16) -> ManagedServiceRecord {
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            engine,
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record
    }

    fn operation_paths(openapi: &Value) -> Vec<String> {
        openapi["paths"]
            .as_object()
            .expect("paths object")
            .keys()
            .cloned()
            .collect()
    }

    #[test]
    fn chat_server_docs_list_its_routes_with_the_served_model() {
        let (root, paths) = temp_app_paths("api-docs-chat-routes");
        for (engine, health) in [("vllm", "/health"), ("lemonade", "/v1/health")] {
            let openapi =
                serde_json::to_value(service_openapi(&record(&paths, "svc-1", engine, 9))).unwrap();
            let mut listed = operation_paths(&openapi);
            listed.sort();
            let mut expected = vec![
                health.to_owned(),
                "/v1/chat/completions".to_owned(),
                "/v1/completions".to_owned(),
                "/v1/models".to_owned(),
            ];
            expected.sort();
            assert_eq!(listed, expected, "{engine}");
            assert_eq!(
                openapi["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"]
                    ["example"]["model"],
                "Qwen/Qwen3.5",
                "{engine}: try-it must start from the model the server answers to"
            );
            assert_eq!(
                openapi["servers"][0]["url"], "/services/svc-1/api",
                "{engine}"
            );
            assert!(
                openapi.get("security").is_none(),
                "{engine}: loopback needs no key"
            );
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn documented_routes_are_the_routes_the_engines_report() {
        let (root, paths) = temp_app_paths("api-docs-protocol-routes");
        for (engine, health) in [("vllm", "/health"), ("lemonade", "/v1/health")] {
            for task in [
                ModelTask::Chat,
                ModelTask::Transcription,
                ModelTask::Speech,
                ModelTask::Unknown,
            ] {
                let mut server = record(&paths, "svc-1", engine, 9);
                server.task = task;
                let documented: Vec<String> = routes_for(&server)
                    .iter()
                    .map(|route| route.path.to_owned())
                    .collect();
                assert_eq!(
                    documented,
                    rocm_engine_protocol::supported_routes(health, task),
                    "{engine} {task}"
                );
            }
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn audio_answers_are_offered_as_downloads() {
        assert_eq!(
            audio_attachment("audio/wav").unwrap(),
            "attachment; filename=\"speech.wav\""
        );
        assert_eq!(
            audio_attachment("audio/mpeg; charset=binary").unwrap(),
            "attachment; filename=\"speech.mp3\""
        );
        assert_eq!(
            audio_attachment("Audio/WAV").unwrap(),
            "attachment; filename=\"speech.wav\""
        );
        assert_eq!(
            audio_attachment("audio/L16; rate=24000").unwrap(),
            "attachment; filename=\"speech.pcm\""
        );
        assert!(audio_attachment("application/json").is_none());
        assert!(audio_attachment("text/plain").is_none());
    }

    #[test]
    fn forwarded_paths_keep_their_encoding() {
        assert_eq!(
            raw_forwarded_path("/services/svc-1/api/v1/models/Qwen%2FQwen3.5%3Fx"),
            Some("v1/models/Qwen%2FQwen3.5%3Fx")
        );
        assert_eq!(raw_forwarded_path("/services/svc-1/api/"), Some(""));
        assert_eq!(raw_forwarded_path("/services/svc-1/docs/x"), None);
    }

    #[test]
    fn only_pages_from_this_machine_may_send_browser_requests() {
        for origin in [
            "http://127.0.0.1:11430",
            "http://localhost:11430",
            "http://[::1]:11430",
        ] {
            assert!(origin_is_loopback(origin), "{origin}");
        }
        for origin in [
            "https://evil.example",
            "http://127.0.0.1.evil.example",
            "null",
            "",
        ] {
            assert!(!origin_is_loopback(origin), "{origin}");
        }
    }

    #[test]
    fn the_bundled_swagger_ui_is_the_version_named_in_the_notices() {
        let config = Arc::new(utoipa_swagger_ui::Config::new(["/openapi.json"]));
        let bundle = utoipa_swagger_ui::serve("swagger-ui-bundle.js", config)
            .unwrap()
            .expect("the vendored Swagger UI ships its bundle");
        let bundle = String::from_utf8_lossy(&bundle.bytes);
        let marker = "PACKAGE_VERSION:\"";
        let start = bundle.find(marker).expect("the bundle states its version") + marker.len();
        let version: String = bundle[start..].chars().take_while(|c| *c != '"').collect();
        // Read, not `include_str!`: the file is not ASCII, and xtask's output
        // contract rejects non-ASCII data compiled into a binary.
        let notices = fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../THIRD_PARTY_NOTICES.txt"
        ))
        .expect("THIRD_PARTY_NOTICES.txt at the repo root");
        assert!(
            notices.contains(&format!("  - Swagger UI {version} ")),
            "THIRD_PARTY_NOTICES.txt must name the Swagger UI {version} that rocmd serves; \
             update the bundled-assets section of about.hbs and run `cargo xtask tpn`"
        );
    }

    #[test]
    fn docs_stay_off_without_stopping_the_daemon() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-off");
        paths.ensure()?;
        let mut state = crate::service::build_runtime_state(&RocmCliConfig::default(), true);

        let mut disabled = RocmCliConfig::default();
        disabled.api_docs.enabled = false;
        assert!(crate::service::start_api_docs(&paths, &disabled, &mut state)?.is_none());

        let mut port_zero = RocmCliConfig::default();
        port_zero.api_docs.port = 0;
        assert!(crate::service::start_api_docs(&paths, &port_zero, &mut state)?.is_none());

        let taken = std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, 0))?;
        let mut on_taken_port = RocmCliConfig::default();
        on_taken_port.api_docs.port = taken.local_addr()?.port();
        assert!(
            crate::service::start_api_docs(&paths, &on_taken_port, &mut state)?.is_none(),
            "a taken port leaves docs off instead of failing the daemon"
        );
        let events = fs::read_to_string(paths.automation_events_path())?;
        assert!(events.contains("api_docs_unavailable"), "{events}");
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn speech_to_text_docs_offer_an_audio_upload_and_no_chat_routes() {
        let (root, paths) = temp_app_paths("api-docs-transcription");
        let mut whisper = record(&paths, "svc-asr", "lemonade", 9);
        whisper.task = ModelTask::Transcription;
        whisper.canonical_model_id = "Whisper-Large-v3-Turbo".to_owned();
        let openapi = serde_json::to_value(service_openapi(&whisper)).unwrap();
        let mut listed = operation_paths(&openapi);
        listed.sort();
        assert_eq!(
            listed,
            ["/v1/audio/transcriptions", "/v1/health", "/v1/models"]
        );
        let form = &openapi["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"]["content"]
            ["multipart/form-data"]["schema"];
        assert_eq!(form["properties"]["file"]["format"], "binary", "{form:#}");
        assert_eq!(
            form["properties"]["model"]["default"], "Whisper-Large-v3-Turbo",
            "try-it must start from the model the server answers to"
        );
        assert!(
            !openapi["info"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("llama.cpp"),
            "the llama.cpp note is about chat models"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn text_to_speech_docs_offer_the_speech_route_answering_with_audio() {
        let (root, paths) = temp_app_paths("api-docs-speech");
        let mut tts = record(&paths, "svc-tts", "lemonade", 9);
        tts.task = ModelTask::Speech;
        tts.canonical_model_id = "OpenMOSS-TTS".to_owned();
        let openapi = serde_json::to_value(service_openapi(&tts)).unwrap();
        let mut listed = operation_paths(&openapi);
        listed.sort();
        assert_eq!(listed, ["/v1/audio/speech", "/v1/health", "/v1/models"]);
        let speech = &openapi["paths"]["/v1/audio/speech"]["post"];
        assert_eq!(
            speech["requestBody"]["content"]["application/json"]["example"]["model"],
            "OpenMOSS-TTS"
        );
        assert!(
            speech["responses"]["200"]["content"]
                .get("audio/wav")
                .is_some(),
            "the answer is documented as audio: {speech:#}"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn keyed_server_docs_ask_for_the_endpoint_key() {
        let (root, paths) = temp_app_paths("api-docs-keyed");
        let mut keyed = record(&paths, "svc-1", "vllm", 9);
        keyed.requires_api_key = true;
        let openapi = serde_json::to_value(service_openapi(&keyed)).unwrap();
        assert_eq!(
            openapi["components"]["securitySchemes"][ENDPOINT_KEY_SCHEME]["scheme"],
            "bearer"
        );
        assert!(openapi["security"][0].get(ENDPOINT_KEY_SCHEME).is_some());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn engine_note_links_vllm_docs_and_llama_cpp_routes() {
        let (root, paths) = temp_app_paths("api-docs-note");
        assert!(
            engine_docs_note(&record(&paths, "a", "vllm", 8000))
                .contains("http://127.0.0.1:8000/docs")
        );
        assert!(engine_docs_note(&record(&paths, "b", "lemonade", 8000)).contains("llama.cpp"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_server_bound_to_every_interface_is_reached_on_loopback() {
        let (root, paths) = temp_app_paths("api-docs-wildcard");
        let mut wildcard = record(&paths, "a", "vllm", 8000);
        wildcard.host = "0.0.0.0".to_owned();
        assert_eq!(service_base_url(&wildcard), "http://127.0.0.1:8000");
        wildcard.host = "::".to_owned();
        assert_eq!(service_base_url(&wildcard), "http://127.0.0.1:8000");
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn only_loopback_host_names_are_accepted() {
        for host in [
            "127.0.0.1:11430",
            "localhost:11430",
            "LOCALHOST",
            "[::1]:11430",
            "127.0.0.1",
        ] {
            assert!(host_header_is_loopback(host), "{host}");
        }
        for host in [
            "evil.example:11430",
            "192.168.1.5:11430",
            "[2001:db8::1]:11430",
            "",
        ] {
            assert!(!host_header_is_loopback(host), "{host}");
        }
    }

    #[test]
    fn html_escape_neutralises_markup() {
        assert_eq!(
            html_escape("<a href=\"x\">'&'</a>"),
            "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;"
        );
    }

    /// What the stand-in text-to-speech route answers: binary, not UTF-8.
    const SPOKEN_WAV: &[u8] = b"RIFF\x24\x00\x00\x00WAVEfmt \xff\x00\x80";

    /// A stand-in model server that echoes what reached it, so the test can
    /// see exactly what the forwarder passed on.
    async fn start_echo_model_server() -> u16 {
        let listener = TcpListener::bind((DEFAULT_LOCAL_HOST, 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|headers: HeaderMap, uri: Uri, body: Bytes| async move {
                    Json(json!({
                        "query": uri.query(),
                        "authorization": headers
                            .get(header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok()),
                        "cookie": headers.get(header::COOKIE).and_then(|value| value.to_str().ok()),
                        "body": String::from_utf8_lossy(&body),
                    }))
                }),
            )
            .route(
                "/v1/audio/speech",
                post(|| async { ([(header::CONTENT_TYPE, "audio/wav")], SPOKEN_WAV) }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });
        port
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn docs_server_describes_and_forwards_to_a_running_server() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-live");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let model_port = start_echo_model_server().await;
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let mut stopped = record(&paths, "svc-stopped", "vllm", model_port);
        stopped.status = "stopped".to_owned();
        stopped.write()?;

        let server = start_api_docs_server(paths.clone(), 0)?;
        let base = server.base_url.clone();
        let client = test_client();

        let index = client.get(format!("{base}/")).send().await?.text().await?;
        assert!(index.contains("/services/svc-live/docs/"), "{index}");
        assert!(!index.contains("svc-stopped"), "{index}");

        let page = client
            .get(format!("{base}/services/svc-live/docs/"))
            .send()
            .await?;
        assert_eq!(page.status(), StatusCode::OK);
        assert!(page.text().await?.contains("swagger-ui"));
        let initializer = client
            .get(format!(
                "{base}/services/svc-live/docs/swagger-initializer.js"
            ))
            .send()
            .await?
            .text()
            .await?;
        assert!(
            initializer.contains("/services/svc-live/openapi.json"),
            "Swagger UI must load this server's description: {initializer}"
        );

        let openapi: Value = client
            .get(format!("{base}/services/svc-live/openapi.json"))
            .send()
            .await?
            .text()
            .await?
            .parse()?;
        assert_eq!(openapi["servers"][0]["url"], "/services/svc-live/api");

        let forwarded: Value = client
            .post(format!(
                "{base}/services/svc-live/api/v1/chat/completions?trace=1"
            ))
            .header(header::AUTHORIZATION, "Bearer typed-in-docs")
            .header(header::COOKIE, "session=browser-only")
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"Qwen/Qwen3.5"}"#)
            .send()
            .await?
            .text()
            .await?
            .parse()?;
        assert_eq!(forwarded["body"], r#"{"model":"Qwen/Qwen3.5"}"#);
        assert_eq!(forwarded["query"], "trace=1");
        assert_eq!(forwarded["authorization"], "Bearer typed-in-docs");
        assert_eq!(
            forwarded["cookie"],
            Value::Null,
            "browser cookies stay in the browser"
        );

        for path in ["docs/", "openapi.json", "api/v1/models"] {
            let response = client
                .get(format!("{base}/services/svc-stopped/{path}"))
                .send()
                .await?;
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
        let stopped_page = client
            .get(format!("{base}/services/svc-stopped/docs/"))
            .send()
            .await?
            .text()
            .await?;
        assert!(stopped_page.contains("is not running"), "{stopped_page}");

        let rebound = client
            .post(format!("{base}/services/svc-live/api/v1/chat/completions"))
            .header(header::HOST, "attacker.example")
            .body("{}")
            .send()
            .await?;
        assert_eq!(rebound.status(), StatusCode::FORBIDDEN, "DNS rebinding");
        let cross_site = client
            .post(format!("{base}/services/svc-live/api/v1/chat/completions"))
            .header(header::ORIGIN, "https://some-site.example")
            .header(header::CONTENT_TYPE, "text/plain")
            .body("{}")
            .send()
            .await?;
        assert_eq!(
            cross_site.status(),
            StatusCode::FORBIDDEN,
            "cross-site post"
        );
        let same_origin = client
            .post(format!("{base}/services/svc-live/api/v1/chat/completions"))
            .header(header::ORIGIN, base.as_str())
            .body("{}")
            .send()
            .await?;
        assert_eq!(same_origin.status(), StatusCode::OK, "the docs page itself");

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// Restores a proxy variable when the test ends, even by panicking.
    struct RestoredProxyVar(&'static str, Option<std::ffi::OsString>);

    impl Drop for RestoredProxyVar {
        #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
        fn drop(&mut self) {
            // SAFETY: still inside PROXY_ENV_TEST_LOCK, which the test holds
            // for longer than this guard lives.
            unsafe {
                match &self.1 {
                    Some(value) => std::env::set_var(self.0, value),
                    None => std::env::remove_var(self.0),
                }
            }
        }
    }

    // A plain test driving its own runtime, so the std lock guarding the
    // process environment is never held across an `.await`.
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    #[test]
    fn try_it_reaches_the_server_even_with_a_proxy_configured() -> Result<()> {
        let _lock = PROXY_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (root, paths) = temp_app_paths("api-docs-proxy");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let model_port = runtime.block_on(start_echo_model_server());
        record(&paths, "svc-live", "vllm", model_port).write()?;
        // A proxy that is not there: a request sent to it fails.
        let dead_proxy = format!(
            "http://127.0.0.1:{}",
            std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, 0))?
                .local_addr()?
                .port()
        );
        // Every original value is saved before any is changed: on Windows
        // `HTTP_PROXY` and `http_proxy` name one variable, so saving one after
        // setting the other would save the dead proxy as the "original".
        let proxied = ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"];
        let unproxied = ["NO_PROXY", "no_proxy"];
        let _restore: Vec<RestoredProxyVar> = proxied
            .iter()
            .chain(&unproxied)
            .map(|key| RestoredProxyVar(key, std::env::var_os(key)))
            .collect();
        for key in proxied {
            // SAFETY: PROXY_ENV_TEST_LOCK is held; `_restore` puts the old
            // values back before it is released.
            unsafe { std::env::set_var(key, &dead_proxy) };
        }
        for key in unproxied {
            // SAFETY: as above.
            unsafe { std::env::remove_var(key) };
        }

        let server = start_api_docs_server(paths, 0)?;
        let status = runtime.block_on(async {
            test_client()
                .post(format!(
                    "{}/services/svc-live/api/v1/chat/completions",
                    server.base_url
                ))
                .body("{}")
                .send()
                .await
                .map(|response| response.status())
        })?;
        assert_eq!(status, StatusCode::OK, "try-it went to the proxy");
        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spoken_audio_comes_through_the_docs_intact_and_as_a_download() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-speech-forward");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let model_port = start_echo_model_server().await;
        let mut tts = record(&paths, "svc-tts", "lemonade", model_port);
        tts.task = ModelTask::Speech;
        tts.write()?;
        let server = start_api_docs_server(paths.clone(), 0)?;

        let answer = test_client()
            .post(format!(
                "{}/services/svc-tts/api/v1/audio/speech",
                server.base_url
            ))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"Qwen/Qwen3.5","input":"hi"}"#)
            .send()
            .await?;
        assert_eq!(answer.status(), StatusCode::OK);
        assert_eq!(answer.headers()[header::CONTENT_TYPE], "audio/wav");
        assert_eq!(
            answer.headers()[header::CONTENT_DISPOSITION],
            "attachment; filename=\"speech.wav\""
        );
        assert_eq!(answer.bytes().await?.as_ref(), SPOKEN_WAV);

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn forwarding_to_a_dead_server_reports_a_bad_gateway() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-dead");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        // Bind and drop to get a port nothing listens on.
        let dead_port = std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, 0))?
            .local_addr()?
            .port();
        record(&paths, "svc-dead", "lemonade", dead_port).write()?;
        let server = start_api_docs_server(paths.clone(), 0)?;
        let response = test_client()
            .get(format!(
                "{}/services/svc-dead/api/v1/models",
                server.base_url
            ))
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(
            response
                .text()
                .await?
                .contains("could not reach server svc-dead")
        );
        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }
}
