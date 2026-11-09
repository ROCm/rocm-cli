// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! API docs for every running model server, served on loopback for as long as
//! `rocm services docs` runs in the foreground.
//!
//! Of the servers rocm-cli runs, only vLLM generates its own docs; Lemonade and
//! the llama.cpp server behind it serve none. So this server describes each
//! running server from the routes rocm-cli knows it serves ([`routes_for`]),
//! renders that with Swagger UI, and forwards the page's "Try it out" requests
//! to the server through `/services/<id>/api/...`. Forwarding keeps try-it
//! same-origin, so it works on every engine whatever that engine's own
//! cross-origin settings are; Lemonade, for one, only admits browser requests
//! from its own origin.
//!
//! It lives here, beside the daemon, rather than in `rocm`, which runs it
//! in-process: `rocm` depends on `rocmd`, so the HTTP stack it needs stays in
//! one crate. It runs only while `rocm services docs` does; nothing in the
//! background serves it.
//!
//! Which servers are running comes from the [`ServiceLoader`] the caller
//! passes, asked again on each request (at most once per
//! [`SERVICE_LIST_TTL`]). `rocm` passes the loader `rocm services list` uses,
//! which checks each server's process and endpoint, so a server that stops or
//! crashes while the docs are open drops off them.

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get};
use rocm_core::{AppPaths, DEFAULT_LOCAL_HOST, ManagedServiceRecord};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use utoipa::openapi::path::{HttpMethod, OperationBuilder, PathItem, PathsBuilder};
use utoipa::openapi::request_body::RequestBodyBuilder;
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityRequirement, SecurityScheme};
use utoipa::openapi::{
    ComponentsBuilder, ContentBuilder, InfoBuilder, OpenApi, OpenApiBuilder, Required,
    ResponseBuilder, ServerBuilder,
};

/// Largest request body the try-it forwarder accepts. Generous enough for a
/// long chat history or a large prompt; the docs server is loopback-only, so
/// this bounds memory, not abuse from elsewhere.
const FORWARD_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// How long the forwarder waits for the model server to accept a connection.
/// There is deliberately no overall timeout: a long completion may stream for
/// minutes.
const FORWARD_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The OpenAPI security scheme name used for servers started with an endpoint
/// key; Swagger UI's "Authorize" box fills it in.
const ENDPOINT_KEY_SCHEME: &str = "endpoint_key";

/// How long a stopping docs server lets in-flight requests finish. A try-it
/// answer may stream for minutes and the forwarder sets no overall timeout, so
/// without this bound an open stream would hold up Ctrl+C.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// How long one read of the running servers is reused. Opening a docs page
/// sends several requests at once, and the loader `rocm` passes probes each
/// server's endpoint; requests that arrive while a read is under way wait for
/// it, so a page load costs one read. A server that stops drops off once the
/// read after it stopped completes: at most this long after the last read,
/// plus the time the next read takes.
const SERVICE_LIST_TTL: Duration = Duration::from_secs(1);

/// `Content-Security-Policy` on every page this server serves itself: no
/// other site may frame it, so a framed docs page cannot be clicked through
/// into sending a try-it request from this origin.
const PAGE_CSP: &str = "frame-ancestors 'none'";

/// `Content-Security-Policy` on forwarded answers: as [`PAGE_CSP`], and
/// sandboxed, since the answer comes from the model server but is served from
/// the docs' origin.
const FORWARDED_CSP: &str = "sandbox; frame-ancestors 'none'";

/// Reads every service record, refreshed however the caller judges a server
/// to be running; the docs list the ones [`ManagedServiceRecord::is_live`]
/// accepts afterwards.
pub type ServiceLoader = Arc<dyn Fn(&AppPaths) -> Result<Vec<ManagedServiceRecord>> + Send + Sync>;

/// The port `rocm services docs` serves on unless told otherwise. Kept clear
/// of `DEFAULT_LOCAL_PORT` and the ports above it that consecutive
/// `rocm serve` launches pick.
pub const DEFAULT_API_DOCS_PORT: u16 = 11_430;

/// A running docs server. It runs on a thread and runtime of its own, so the
/// caller's thread stays free to wait for Ctrl+C.
pub struct ApiDocsServer {
    /// `http://127.0.0.1:<port>`, the port actually bound.
    pub base_url: String,
    shutdown: tokio::sync::oneshot::Sender<()>,
    thread: std::thread::JoinHandle<()>,
}

impl ApiDocsServer {
    /// The API docs page of one server.
    #[must_use]
    pub fn service_docs_url(&self, service_id: &str) -> String {
        format!("{}{}/docs/", self.base_url, service_path(service_id))
    }

    /// Stop accepting requests and wait for the server thread to finish.
    pub fn stop(self) {
        let _ = self.shutdown.send(());
        let _ = self.thread.join();
    }
}

/// The last read of the running servers, and when it was made.
type CachedServices = Option<(Instant, Arc<Vec<ManagedServiceRecord>>)>;

#[derive(Clone)]
struct DocsState {
    paths: Arc<AppPaths>,
    load: ServiceLoader,
    /// Held across a read, so concurrent requests share one.
    cached: Arc<tokio::sync::Mutex<CachedServices>>,
    service_list_ttl: Duration,
    client: reqwest::Client,
    /// The origins of this server's own pages, the only ones whose browser
    /// requests are admitted.
    own_origins: Arc<[String]>,
}

/// Serve API docs for the running model servers on `127.0.0.1:<port>`; port
/// 0 binds a free one, named in [`ApiDocsServer::base_url`]. `load` reads the
/// service records; see [`ServiceLoader`].
///
/// # Errors
/// Fails when the port cannot be bound or the server cannot start.
pub fn start_api_docs_server(
    paths: AppPaths,
    port: u16,
    load: ServiceLoader,
) -> Result<ApiDocsServer> {
    start_with(paths, port, load, FORWARD_CONNECT_TIMEOUT, SERVICE_LIST_TTL)
}

fn start_with(
    paths: AppPaths,
    port: u16,
    load: ServiceLoader,
    connect_timeout: Duration,
    service_list_ttl: Duration,
) -> Result<ApiDocsServer> {
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
    let bound = addr.port();
    let app = router(DocsState {
        paths: Arc::new(paths),
        load,
        cached: Arc::new(tokio::sync::Mutex::new(None)),
        service_list_ttl,
        client: forwarding_client(connect_timeout)?,
        own_origins: [
            format!("http://{DEFAULT_LOCAL_HOST}:{bound}"),
            format!("http://localhost:{bound}"),
        ]
        .into(),
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build the API docs runtime")?;
    // Registered here rather than on the server thread, so a failure is this
    // call's error instead of a thread that ends while the caller waits.
    let listener = {
        let _entered = runtime.enter();
        TcpListener::from_std(listener).context("failed to register the API docs listener")?
    };
    let (shutdown, shutdown_received) = tokio::sync::oneshot::channel::<()>();
    let thread = std::thread::Builder::new()
        .name("rocmd-api-docs".to_owned())
        .spawn(move || {
            runtime.block_on(async move {
                // `axum::serve` retries failed accepts rather than returning
                // an error, so this ends only when the caller stops it.
                if let Err(error) = serve_until_stopped(listener, app, shutdown_received).await {
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

/// Serve until `shutdown` fires, then give in-flight requests
/// [`SHUTDOWN_GRACE`] to finish before dropping them.
async fn serve_until_stopped(
    listener: TcpListener,
    app: Router,
    shutdown: tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
    let stopping = Arc::new(tokio::sync::Notify::new());
    let stop_signal = Arc::clone(&stopping);
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        let _ = shutdown.await;
        stop_signal.notify_one();
    });
    tokio::select! {
        served = serve => served.map_err(anyhow::Error::from),
        () = async {
            stopping.notified().await;
            tokio::time::sleep(SHUTDOWN_GRACE).await;
        } => Ok(()),
    }
}

/// The client the forwarder sends try-it requests with.
fn forwarding_client(connect_timeout: Duration) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
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
        .route("/services/{id}/openapi.json", get(service_openapi_json))
        .route("/services/{id}/docs", get(docs_redirect))
        .route("/services/{id}/docs/", get(docs_index))
        .route("/services/{id}/docs/{*file}", get(docs_file))
        .route(
            "/services/{id}/api/{*path}",
            any(forward).layer(DefaultBodyLimit::max(FORWARD_BODY_LIMIT_BYTES)),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_loopback_host,
        ))
        .with_state(state)
}

/// Refuse requests whose `Host` is not a loopback name, and browser requests
/// sent from any page but this server's own. The listener is already
/// loopback-only; the `Host` check closes DNS rebinding, where a web page
/// resolves its own hostname to 127.0.0.1. A browser request is recognised by
/// its `Origin`, which a page's `fetch` and form posts carry and which must
/// name this server, or, when it has none (a link, an image, a top-level
/// navigation), by `Sec-Fetch-Site`, which must say the request came from
/// this server's own pages or from the user typing the address. Requests from
/// other programs (curl) carry neither and are admitted. So no other page the
/// user has open, another local web app included, can send a request through
/// the forwarder.
///
/// Every answer is marked so no other site can frame it or have it sniffed
/// into a different type.
async fn require_loopback_host(
    State(state): State<DocsState>,
    request: Request,
    next: Next,
) -> Response {
    if !request_is_admitted(request.headers(), &state.own_origins) {
        return mark_as_not_frameable(
            (
                StatusCode::FORBIDDEN,
                "API docs answer only their own pages, on 127.0.0.1 or localhost\n",
            )
                .into_response(),
        );
    }
    mark_as_not_frameable(next.run(request).await)
}

fn request_is_admitted(headers: &HeaderMap, own_origins: &[String]) -> bool {
    let text = |name| headers.get(name).map(|value: &HeaderValue| value.to_str());
    let host_allowed =
        matches!(text(header::HOST), Some(Ok(host)) if host_header_is_loopback(host));
    let browser_allowed = match (text(header::ORIGIN), text(SEC_FETCH_SITE)) {
        (Some(origin), _) => origin.is_ok_and(|origin| origin_is_own(origin, own_origins)),
        (None, Some(site)) => site.is_ok_and(fetch_site_is_own),
        (None, None) => true,
    };
    host_allowed && browser_allowed
}

/// The fetch-metadata header browsers send with every request, `Origin` or
/// not.
const SEC_FETCH_SITE: header::HeaderName = header::HeaderName::from_static("sec-fetch-site");

/// `same-origin` is a request from this server's own pages; `none` is one the
/// user started (typing the address, a bookmark). `same-site` is refused too:
/// another port on 127.0.0.1 is the same site.
fn fetch_site_is_own(site: &str) -> bool {
    matches!(site, "same-origin" | "none")
}

fn mark_as_not_frameable(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers
        .entry(header::CONTENT_SECURITY_POLICY)
        .or_insert(HeaderValue::from_static(PAGE_CSP));
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
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

/// Whether an `Origin` header names one of this server's own pages. Browsers
/// send the scheme and host lowercased and the port only when it is not the
/// scheme's default, so an exact comparison is enough.
fn origin_is_own(origin: &str, own_origins: &[String]) -> bool {
    own_origins.iter().any(|own| own == origin)
}

/// Every running server, as the [`ServiceLoader`] judges it. A failure to
/// read the records is logged and returned, so a page can say the records
/// could not be read rather than claim that nothing is running. The loader
/// may probe each server, so it runs off the async thread. Requests that
/// arrive during a read wait for it rather than starting their own, and reuse
/// its result when it succeeds; a failed read is not remembered, so each
/// waiter then reads again in turn. The loader must therefore bound its own
/// probes, as the one `rocm` passes does: each server's liveness check and
/// inference probe have their own timeouts (750 ms and
/// `rocm_core::INFERENCE_PROBE_TIMEOUT`, 8 s), so a read takes at most about
/// nine seconds per server, servers probed in turn. A timeout here would only
/// abandon the read, not stop it.
async fn running_services(state: &DocsState) -> Result<Arc<Vec<ManagedServiceRecord>>> {
    let mut cached = state.cached.lock().await;
    if let Some((read_at, services)) = cached.as_ref()
        && read_at.elapsed() < state.service_list_ttl
    {
        return Ok(Arc::clone(services));
    }
    let load = Arc::clone(&state.load);
    let paths = Arc::clone(&state.paths);
    let loaded = tokio::task::spawn_blocking(move || load(&paths))
        .await
        .context("the service records read stopped")
        .and_then(|loaded| loaded)
        .inspect_err(|error| {
            eprintln!("API docs could not read the service records: {error:#}");
        })?;
    let services: Arc<Vec<ManagedServiceRecord>> = Arc::new(
        loaded
            .into_iter()
            .filter(ManagedServiceRecord::is_live)
            .collect(),
    );
    *cached = Some((Instant::now(), Arc::clone(&services)));
    Ok(services)
}

/// The running server with this id. Looked up by matching the id against the
/// records rather than by building a path from it, so a crafted id cannot
/// reach outside the services folder.
async fn find_running_service(
    state: &DocsState,
    service_id: &str,
) -> Result<Option<ManagedServiceRecord>> {
    Ok(running_services(state)
        .await?
        .iter()
        .find(|record| record.service_id == service_id)
        .cloned())
}

/// `/services/<id>`, with the id percent-encoded: an id may hold characters
/// (a space, `#`, `?`, `%`) that would otherwise end or change the path.
fn service_path(service_id: &str) -> String {
    let mut path = String::from("/services/");
    for byte in service_id.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            path.push(char::from(byte));
        } else {
            let _ = write!(path, "%{byte:02X}");
        }
    }
    path
}

/// The answer for a server lookup that could not read the records: 503, since
/// the server may well be running.
fn records_unreadable(error: &anyhow::Error) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(json!({
            "error": format!("could not read the service records: {error:#}"),
        })),
    )
        .into_response()
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

async fn index_page(State(state): State<DocsState>) -> Response {
    let services = match running_services(&state).await {
        Ok(services) => services,
        Err(error) => return records_unreadable(&error),
    };
    let mut rows = String::new();
    for record in services.iter() {
        let _ = write!(
            rows,
            "<li><a href=\"{href}/docs/\">{model}</a> ({engine}, {id}, {endpoint})</li>",
            // Percent-encoded, so it holds no character HTML treats specially.
            href = service_path(&record.service_id),
            id = html_escape(&record.service_id),
            model = html_escape(record.served_model_name()),
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
    .into_response()
}

async fn service_openapi_json(
    State(state): State<DocsState>,
    Path(service_id): Path<String>,
) -> Response {
    match find_running_service(&state, &service_id).await {
        Ok(Some(record)) => axum::Json(service_openapi(&record)).into_response(),
        Err(error) => records_unreadable(&error),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({
                "error": format!("server {service_id} is not running"),
            })),
        )
            .into_response(),
    }
}

/// `/services/<id>/docs` to the page itself, whose assets load relative to
/// the trailing slash. Only a running server's id is redirected, and
/// percent-encoded again, since it comes decoded from the request. The
/// redirect is temporary, since the server it names can stop.
async fn docs_redirect(State(state): State<DocsState>, Path(service_id): Path<String>) -> Response {
    match find_running_service(&state, &service_id).await {
        Ok(Some(record)) => {
            match HeaderValue::try_from(format!("{}/docs/", service_path(&record.service_id))) {
                Ok(location) => (
                    StatusCode::TEMPORARY_REDIRECT,
                    [(header::LOCATION, location)],
                )
                    .into_response(),
                Err(_) => not_running_page(&service_id),
            }
        }
        Ok(None) => not_running_page(&service_id),
        Err(error) => records_unreadable(&error),
    }
}

async fn docs_index(State(state): State<DocsState>, Path(service_id): Path<String>) -> Response {
    serve_swagger_file(&state, &service_id, "").await
}

async fn docs_file(
    State(state): State<DocsState>,
    Path((service_id, file)): Path<(String, String)>,
) -> Response {
    serve_swagger_file(&state, &service_id, &file).await
}

async fn serve_swagger_file(state: &DocsState, service_id: &str, file: &str) -> Response {
    match find_running_service(state, service_id).await {
        Ok(Some(_)) => {}
        Ok(None) => return not_running_page(service_id),
        Err(error) => return records_unreadable(&error),
    }
    let config = Arc::new(
        utoipa_swagger_ui::Config::new([format!("{}/openapi.json", service_path(service_id))])
            .try_it_out_enabled(true)
            // The endpoint key typed into "Authorize" lasts until the page is
            // closed; it is not saved in the browser, where it would outlive
            // the server it belongs to.
            .persist_authorization(false),
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
    let record = match find_running_service(&state, &service_id).await {
        Ok(Some(record)) => record,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                axum::Json(json!({ "error": format!("server {service_id} is not running") })),
            )
                .into_response();
        }
        Err(error) => return records_unreadable(&error),
    };
    // The path exactly as the browser sent it. The extracted one is
    // percent-decoded: an encoded `?` or `/` in it would turn into a query or
    // an extra segment, and the request would reach a different route.
    let Some(path) = raw_forwarded_path(uri.path()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut target = format!("{}/{path}", record.local_base_url());
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
            let headers = response.headers_mut();
            if let Some(content_type) = content_type {
                headers.insert(header::CONTENT_TYPE, content_type);
            }
            // The answer comes from the model server but is served from the
            // docs' origin. A page in it (llama.cpp's own UI, say) must not
            // run there: it is shown as data, never as a document of this
            // origin. `nosniff` is added for every answer, by
            // `require_loopback_host`.
            headers.insert(
                header::CONTENT_SECURITY_POLICY,
                HeaderValue::from_static(FORWARDED_CSP),
            );
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
                    record.local_base_url()
                ),
            })
            .to_string(),
        )
            .into_response(),
    }
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

/// One route a model server answers, as the docs describe it.
struct RouteDoc {
    method: HttpMethod,
    path: &'static str,
    summary: &'static str,
    description: &'static str,
    /// An example request body, built for the server's real model id.
    example_body: Option<fn(&str) -> Value>,
}

/// The paths the docs describe for a server of `engine`, in order. `rocm`'s
/// tests compare them with what each engine reports as `supported_routes`.
#[must_use]
pub fn documented_route_paths(engine: &str) -> Vec<String> {
    routes_for(engine)
        .iter()
        .map(|route| route.path.to_owned())
        .collect()
}

/// The routes each engine's server answers: the engine's health route from
/// [`rocm_engine_protocol::ENGINE_HEALTH_ROUTES`], when it has one there, plus
/// the chat routes. An engine the table does not know gets no health route
/// rather than another engine's.
fn routes_for(engine: &str) -> Vec<RouteDoc> {
    let health = rocm_engine_protocol::engine_health_route(engine).map(|path| RouteDoc {
        method: HttpMethod::Get,
        path,
        summary: "Check the server is up",
        description: "Answers 200 once the server accepts requests.",
        example_body: None,
    });
    health.into_iter().chain(chat_routes()).collect()
}

fn chat_routes() -> [RouteDoc; 3] {
    [
        RouteDoc {
            method: HttpMethod::Get,
            path: "/v1/models",
            summary: "List the served models",
            description: "OpenAI-compatible model list. The `id` of each entry is the \
                          `model` value requests use.",
            example_body: None,
        },
        RouteDoc {
            method: HttpMethod::Post,
            path: "/v1/chat/completions",
            summary: "Chat completion",
            description: "OpenAI-compatible chat completion. Set `stream` to `true` to \
                          receive server-sent events as tokens are generated.",
            example_body: Some(chat_completion_example),
        },
        RouteDoc {
            method: HttpMethod::Post,
            path: "/v1/completions",
            summary: "Text completion",
            description: "OpenAI-compatible text completion of a raw prompt.",
            example_body: Some(completion_example),
        },
    ]
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
    if let Some(own_docs) = record.engine_docs_url() {
        return format!(
            "vLLM also documents every route it serves, beyond the ones listed here, at \
             [{own_docs}]({own_docs})."
        );
    }
    match record.engine.as_str() {
        "lemonade" => "On Linux, Lemonade runs chat models with the llama.cpp server, which also \
                       answers its own routes; see the \
                       [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)."
            .to_owned(),
        "aim" => "This server runs in an AMD Inference Microservice (AIM) container; the routes \
                  listed here are the OpenAI-compatible ones rocm-cli uses."
            .to_owned(),
        _ => "The routes listed here are the OpenAI-compatible ones rocm-cli uses.".to_owned(),
    }
}

fn service_openapi(record: &ManagedServiceRecord) -> OpenApi {
    // The name the server answers to, which for AIM is not the image
    // reference recorded as the canonical id.
    let model = record.served_model_name();
    let mut paths = PathsBuilder::new();
    for route in routes_for(&record.engine) {
        let mut operation = OperationBuilder::new()
            .summary(Some(route.summary))
            .description(Some(route.description))
            .response(
                "200",
                ResponseBuilder::new()
                    .description("The server's answer")
                    .content("application/json", ContentBuilder::new().build()),
            );
        if let Some(example_body) = route.example_body {
            operation = operation.request_body(Some(
                RequestBodyBuilder::new()
                    .required(Some(Required::True))
                    .content(
                        "application/json",
                        ContentBuilder::new()
                            .example(Some(example_body(model)))
                            .build(),
                    )
                    .build(),
            ));
        }
        paths = paths.path(route.path, PathItem::new(route.method, operation));
    }

    let description = format!(
        "`{model}` served by {engine} at `{endpoint}`.\n\n\
         \"Try it out\" sends requests through this page to the server.\n\n{note}",
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
            .url(format!("{}/api", service_path(&record.service_id)))
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
    use std::fs;

    /// Serializes the tests that set proxy variables. Every other test here
    /// talks to its servers through [`test_client`], which ignores them.
    static PROXY_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The test's own HTTP client: never proxied, so a proxy variable set by
    /// the proxy test cannot reroute it, and never following a redirect, so a
    /// test sees exactly what the docs server answered.
    fn test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap()
    }

    /// The records exactly as written, the loader the daemon's own code uses.
    fn records_as_written() -> ServiceLoader {
        Arc::new(|paths: &AppPaths| crate::persistence::load_managed_services(paths))
    }

    fn start(paths: &AppPaths) -> Result<ApiDocsServer> {
        start_api_docs_server(paths.clone(), 0, records_as_written())
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

    fn services_dir_ready(paths: &AppPaths) -> Result<()> {
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        Ok(())
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
        for (engine, health) in [
            ("vllm", "/health"),
            ("lemonade", "/v1/health"),
            ("aim", "/health"),
        ] {
            let mut served = record(&paths, "svc-1", engine, 9);
            // As for an AIM server, whose canonical id is the image reference
            // and whose served name is adopted from the engine.
            served.canonical_model_id = "registry.example/aim/qwen:1.0".to_owned();
            served.served_model_id = Some("Qwen/Qwen3.5".to_owned());
            let openapi = serde_json::to_value(service_openapi(&served)).unwrap();
            assert!(
                openapi["info"]["title"]
                    .as_str()
                    .is_some_and(|title| title.starts_with("Qwen/Qwen3.5 ")),
                "{engine}: {}",
                openapi["info"]["title"]
            );
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

    /// The docs' own route list against the protocol's table. That each
    /// engine reports that table is `rocm`'s
    /// `documented_routes_are_the_routes_each_engine_reports`, which can call
    /// the engines.
    #[test]
    fn documented_routes_are_the_protocols_route_table() {
        for engine in ["vllm", "lemonade", "aim", "unknown"] {
            assert_eq!(
                documented_route_paths(engine),
                rocm_engine_protocol::engine_supported_routes(engine),
                "{engine}"
            );
        }
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
    fn service_paths_percent_encode_the_id() {
        assert_eq!(service_path("svc-1.a_b~"), "/services/svc-1.a_b~");
        assert_eq!(service_path("svc a#1%?/"), "/services/svc%20a%231%25%3F%2F");
        assert_eq!(service_path("é"), "/services/%C3%A9");
    }

    #[test]
    fn only_the_docs_servers_own_pages_may_send_browser_requests() {
        let own: Vec<String> = vec![
            "http://127.0.0.1:11430".to_owned(),
            "http://localhost:11430".to_owned(),
        ];
        for origin in ["http://127.0.0.1:11430", "http://localhost:11430"] {
            assert!(origin_is_own(origin, &own), "{origin}");
        }
        for origin in [
            // Another local web app, on this machine but not these docs.
            "http://localhost:3000",
            "http://127.0.0.1:11431",
            "https://127.0.0.1:11430",
            "https://evil.example",
            "http://127.0.0.1.evil.example",
            "http://127.0.0.1:11430/",
            "null",
            "",
        ] {
            assert!(!origin_is_own(origin, &own), "{origin}");
        }
        for site in ["same-origin", "none"] {
            assert!(fetch_site_is_own(site), "{site}");
        }
        // `same-site` covers every port of 127.0.0.1, other local apps too.
        for site in ["same-site", "cross-site", "", "SAME-ORIGIN"] {
            assert!(!fetch_site_is_own(site), "{site}");
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
    fn a_taken_port_is_reported_and_port_zero_binds_a_free_one() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-ports");
        paths.ensure()?;
        let taken = std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, 0))?;
        let taken_port = taken.local_addr()?.port();
        let error = start_api_docs_server(paths.clone(), taken_port, records_as_written())
            .err()
            .expect("a taken port cannot be served on");
        assert!(
            format!("{error:#}").contains(&format!("127.0.0.1:{taken_port}")),
            "{error:#}"
        );
        let server = start(&paths)?;
        assert!(!server.base_url.ends_with(":0"), "{}", server.base_url);
        assert_eq!(
            server.service_docs_url("svc-1"),
            format!("{}/services/svc-1/docs/", server.base_url)
        );
        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
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
    fn engine_note_names_each_engines_own_docs() {
        let (root, paths) = temp_app_paths("api-docs-note");
        let note = |engine| engine_docs_note(&record(&paths, "a", engine, 8000));
        assert!(note("vllm").contains("http://127.0.0.1:8000/docs"));
        assert!(note("lemonade").contains("llama.cpp"));
        for engine in ["aim", "unknown"] {
            let text = note(engine);
            assert!(
                !text.contains("llama.cpp") && !text.contains("/docs"),
                "{engine} is not Lemonade or vLLM: {text}"
            );
        }
        assert!(note("aim").contains("AIM"));
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

    fn header_text(headers: &HeaderMap, name: header::HeaderName) -> Option<String> {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    }

    /// What reached the stand-in model server, as it saw it: the path still
    /// percent-encoded, as the request line carried it.
    async fn echo(method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Json<Value> {
        Json(json!({
            "method": method.as_str(),
            "path": uri.path(),
            "query": uri.query(),
            "content_type": header_text(&headers, header::CONTENT_TYPE),
            "accept": header_text(&headers, header::ACCEPT),
            "authorization": header_text(&headers, header::AUTHORIZATION),
            "cookie": header_text(&headers, header::COOKIE),
            "body_len": body.len(),
            "body": String::from_utf8_lossy(&body),
        }))
    }

    /// A stand-in model server on `host` that echoes what reached it, so a
    /// test can see exactly what the forwarder passed on. `/redirect`
    /// answers with a redirect. `None` when `host` cannot be bound here.
    async fn start_echo_model_server_on(host: IpAddr) -> Option<u16> {
        let listener = TcpListener::bind((host, 0)).await.ok()?;
        let port = listener.local_addr().ok()?.port();
        let app = Router::new()
            .route(
                "/redirect",
                any(|| async {
                    (
                        StatusCode::TEMPORARY_REDIRECT,
                        [(header::LOCATION, "/v1/models")],
                    )
                }),
            )
            .fallback(echo)
            .layer(DefaultBodyLimit::disable());
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });
        Some(port)
    }

    async fn start_echo_model_server() -> u16 {
        start_echo_model_server_on(IpAddr::from([127, 0, 0, 1]))
            .await
            .expect("bind 127.0.0.1")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn docs_server_describes_and_forwards_to_a_running_server() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-live");
        services_dir_ready(&paths)?;
        let model_port = start_echo_model_server().await;
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let mut stopped = record(&paths, "svc-stopped", "vllm", model_port);
        stopped.status = "stopped".to_owned();
        stopped.write()?;

        let server = start(&paths)?;
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
        assert!(
            initializer.contains("\"persistAuthorization\": false"),
            "the endpoint key must not be saved in the browser: {initializer}"
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
            .header(header::ACCEPT, "text/event-stream")
            .body(r#"{"model":"Qwen/Qwen3.5"}"#)
            .send()
            .await?
            .text()
            .await?
            .parse()?;
        assert_eq!(forwarded["method"], "POST");
        assert_eq!(forwarded["path"], "/v1/chat/completions");
        assert_eq!(forwarded["body"], r#"{"model":"Qwen/Qwen3.5"}"#);
        assert_eq!(forwarded["query"], "trace=1");
        assert_eq!(forwarded["authorization"], "Bearer typed-in-docs");
        assert_eq!(
            forwarded["content_type"], "application/json",
            "the server parses the body by it"
        );
        assert_eq!(
            forwarded["accept"], "text/event-stream",
            "a streamed answer is asked for by it"
        );
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
        // Another web app on this machine is still not these docs.
        let other_local_app = client
            .post(format!("{base}/services/svc-live/api/v1/chat/completions"))
            .header(header::ORIGIN, "http://localhost:3000")
            .header(header::CONTENT_TYPE, "text/plain")
            .body("{}")
            .send()
            .await?;
        assert_eq!(
            other_local_app.status(),
            StatusCode::FORBIDDEN,
            "a page from another local port"
        );
        let localhost_alias = client
            .post(format!("{base}/services/svc-live/api/v1/chat/completions"))
            .header(
                header::ORIGIN,
                base.replace("127.0.0.1", "localhost").as_str(),
            )
            .body("{}")
            .send()
            .await?;
        assert_eq!(
            localhost_alias.status(),
            StatusCode::OK,
            "the docs page opened as localhost"
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

    /// A browser sends no `Origin` with a link, an image or a navigation, but
    /// it does send `Sec-Fetch-Site`; only this server's own pages and the
    /// user's own typing get through.
    #[tokio::test(flavor = "multi_thread")]
    async fn origin_less_browser_requests_must_come_from_the_docs_pages() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-fetch-site");
        services_dir_ready(&paths)?;
        let model_port = start_echo_model_server().await;
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let server = start(&paths)?;
        let url = format!("{}/services/svc-live/api/v1/models", server.base_url);
        let client = test_client();

        for (site, expected) in [
            (Some("cross-site"), StatusCode::FORBIDDEN),
            (Some("same-site"), StatusCode::FORBIDDEN),
            (Some("same-origin"), StatusCode::OK),
            (Some("none"), StatusCode::OK),
            // Not a browser: curl and other programs send neither header.
            (None, StatusCode::OK),
        ] {
            let mut request = client.get(&url);
            if let Some(site) = site {
                request = request.header(SEC_FETCH_SITE, site);
            }
            assert_eq!(request.send().await?.status(), expected, "{site:?}");
        }

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_other_site_can_frame_the_docs() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-frames");
        services_dir_ready(&paths)?;
        let model_port = start_echo_model_server().await;
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let server = start(&paths)?;
        let base = server.base_url.clone();
        let client = test_client();

        for (path, csp) in [
            ("/", PAGE_CSP),
            ("/services/svc-live/docs/", PAGE_CSP),
            ("/services/svc-live/docs/swagger-initializer.js", PAGE_CSP),
            ("/services/svc-live/openapi.json", PAGE_CSP),
            ("/services/not-running/docs/", PAGE_CSP),
            ("/services/svc-live/api/v1/models", FORWARDED_CSP),
        ] {
            let response = client.get(format!("{base}{path}")).send().await?;
            let headers = response.headers();
            assert_eq!(headers[header::CONTENT_SECURITY_POLICY], csp, "{path}");
            assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY", "{path}");
            assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff", "{path}");
        }
        let refused = client
            .get(format!("{base}/"))
            .header(header::HOST, "attacker.example")
            .send()
            .await?;
        assert_eq!(refused.status(), StatusCode::FORBIDDEN);
        assert_eq!(refused.headers()[header::X_FRAME_OPTIONS], "DENY");

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn try_it_keeps_the_path_encoded_and_sends_large_bodies() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-forward-path");
        services_dir_ready(&paths)?;
        let model_port = start_echo_model_server().await;
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let server = start(&paths)?;
        let base = server.base_url.clone();
        let client = test_client();

        // Decoded, `%2F` would add a segment and `%3F` would start a query.
        let forwarded: Value = client
            .get(format!(
                "{base}/services/svc-live/api/v1/models/Qwen%2FQwen3.5%3Fx"
            ))
            .send()
            .await?
            .text()
            .await?
            .parse()?;
        assert_eq!(forwarded["path"], "/v1/models/Qwen%2FQwen3.5%3Fx");
        assert_eq!(forwarded["query"], Value::Null);

        // Larger than axum's 2 MiB default, as an audio upload can be.
        let large = 3 * 1024 * 1024;
        let response = client
            .post(format!(
                "{base}/services/svc-live/api/v1/audio/transcriptions"
            ))
            .body(vec![b'a'; large])
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let forwarded: Value = response.text().await?.parse()?;
        assert_eq!(forwarded["body_len"], large);

        // A redirect is passed back, not followed by the forwarder.
        let redirect = client
            .get(format!("{base}/services/svc-live/api/redirect"))
            .send()
            .await?;
        assert_eq!(redirect.status(), StatusCode::TEMPORARY_REDIRECT);

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ids_are_escaped_in_pages_and_encoded_in_links() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-escaping");
        services_dir_ready(&paths)?;
        let model_port = start_echo_model_server().await;
        let odd_id = "svc a#1%";
        let mut odd = record(&paths, odd_id, "vllm", model_port);
        odd.canonical_model_id = "<b>model</b>".to_owned();
        odd.write()?;
        let server = start(&paths)?;
        let base = server.base_url.clone();
        let client = test_client();

        let link = server.service_docs_url(odd_id);
        assert_eq!(link, format!("{base}/services/svc%20a%231%25/docs/"));
        let page = client.get(&link).send().await?;
        assert_eq!(page.status(), StatusCode::OK, "the printed link opens");
        let initializer = client
            .get(format!("{link}swagger-initializer.js"))
            .send()
            .await?
            .text()
            .await?;
        assert!(
            initializer.contains("/services/svc%20a%231%25/openapi.json"),
            "{initializer}"
        );
        let openapi = client
            .get(format!("{base}/services/svc%20a%231%25/openapi.json"))
            .send()
            .await?;
        assert_eq!(openapi.status(), StatusCode::OK);
        let openapi: Value = openapi.text().await?.parse()?;
        assert_eq!(openapi["servers"][0]["url"], "/services/svc%20a%231%25/api");
        let redirect = client
            .get(format!("{base}/services/svc%20a%231%25/docs"))
            .send()
            .await?;
        assert_eq!(
            redirect.headers()[header::LOCATION],
            "/services/svc%20a%231%25/docs/"
        );

        let index = client.get(format!("{base}/")).send().await?.text().await?;
        assert!(
            index.contains("href=\"/services/svc%20a%231%25/docs/\""),
            "{index}"
        );
        assert!(index.contains("&lt;b&gt;model&lt;/b&gt;"), "{index}");
        assert!(!index.contains("<b>"), "{index}");

        // The id of a server that is not running is reflected into the page.
        let crafted = client
            .get(format!(
                "{base}/services/%3Cscript%3Ealert(1)%3C%2Fscript%3E/docs/"
            ))
            .send()
            .await?;
        assert_eq!(crafted.status(), StatusCode::NOT_FOUND);
        let crafted = crafted.text().await?;
        assert!(crafted.contains("&lt;script&gt;alert(1)"), "{crafted}");
        assert!(!crafted.contains("<script>"), "{crafted}");

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// A server bound to `::` may accept IPv6 only (Docker's `[::]` publish,
    /// `bindv6only=1`), so it is reached on `::1`, which works either way.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_bound_to_every_ipv6_interface_is_reached_on_ipv6_loopback() -> Result<()> {
        let Some(model_port) =
            start_echo_model_server_on(IpAddr::from([0, 0, 0, 0, 0, 0, 0, 1])).await
        else {
            eprintln!("skipped: this host cannot bind ::1");
            return Ok(());
        };
        let (root, paths) = temp_app_paths("api-docs-ipv6");
        services_dir_ready(&paths)?;
        let mut wildcard = record(&paths, "svc-v6", "vllm", model_port);
        wildcard.host = "::".to_owned();
        wildcard.write()?;
        let server = start(&paths)?;

        let response = test_client()
            .get(format!("{}/services/svc-v6/api/v1/models", server.base_url))
            .send()
            .await?;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "{}",
            response.text().await?
        );

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// The docs show the servers the loader they were given calls running,
    /// not the records as written: `rocm` relies on that to drop a server
    /// whose process died.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_docs_list_what_the_loader_reports() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-loader");
        services_dir_ready(&paths)?;
        record(&paths, "svc-on-disk", "vllm", 9).write()?;
        let mut reported = record(&paths, "svc-from-loader", "aim", 9);
        reported.canonical_model_id = "registry.example/aim/qwen:1.0".to_owned();
        reported.served_model_id = Some("served/Qwen".to_owned());
        let loader: ServiceLoader = Arc::new(move |_: &AppPaths| Ok(vec![reported.clone()]));
        let server = start_api_docs_server(paths.clone(), 0, loader)?;
        let index = test_client()
            .get(format!("{}/", server.base_url))
            .send()
            .await?
            .text()
            .await?;
        assert!(index.contains("svc-from-loader"), "{index}");
        assert!(!index.contains("svc-on-disk"), "{index}");
        assert!(
            index.contains(">served/Qwen</a>"),
            "labelled with the name the server answers to: {index}"
        );
        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// A docs page sends several requests at once and the loader probes every
    /// server, so requests within one TTL share a single read, including
    /// requests that arrive while that read is still under way; once the TTL
    /// has passed the servers are read again.
    #[tokio::test(flavor = "multi_thread")]
    async fn requests_within_the_ttl_share_one_read_of_the_servers() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (root, paths) = temp_app_paths("api-docs-service-list-ttl");
        services_dir_ready(&paths)?;
        let reads = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&reads);
        let running = record(&paths, "svc-counted", "vllm", 9);
        let loader: ServiceLoader = Arc::new(move |_: &AppPaths| {
            counted.fetch_add(1, Ordering::SeqCst);
            // Slow enough that the concurrent requests below arrive while
            // the first read is still under way.
            std::thread::sleep(Duration::from_millis(300));
            Ok(vec![running.clone()])
        });
        let ttl = Duration::from_millis(1500);
        let server = start_with(paths.clone(), 0, loader, FORWARD_CONNECT_TIMEOUT, ttl)?;
        let index = format!("{}/", server.base_url);

        let mut burst = tokio::task::JoinSet::new();
        for _ in 0..5 {
            let index = index.clone();
            burst.spawn(async move { test_client().get(index).send().await });
        }
        while let Some(answer) = burst.join_next().await {
            assert_eq!(answer??.status(), StatusCode::OK);
        }
        assert_eq!(reads.load(Ordering::SeqCst), 1, "one read for the burst");

        tokio::time::sleep(ttl + Duration::from_millis(100)).await;
        let page = test_client().get(&index).send().await?.text().await?;
        assert!(page.contains("svc-counted"), "{page}");
        assert_eq!(
            reads.load(Ordering::SeqCst),
            2,
            "read again once the TTL has passed"
        );
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
        services_dir_ready(&paths)?;
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

        let server = start(&paths)?;
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
    async fn forwarding_to_a_dead_server_reports_a_bad_gateway() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-dead");
        services_dir_ready(&paths)?;
        // Bind and drop to get a port nothing listens on.
        let dead_port = std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, 0))?
            .local_addr()?
            .port();
        record(&paths, "svc-dead", "lemonade", dead_port).write()?;
        let server = start(&paths)?;
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

    /// A model server that never accepts the connection (hung, or a firewall
    /// dropping it) gets a 502 after the connect timeout rather than holding
    /// the page forever.
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "relies on Linux dropping connections to a full accept queue; \
                  elsewhere they are refused at once, which tests nothing"
    )]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_that_never_accepts_is_a_bad_gateway_after_the_connect_timeout() -> Result<()>
    {
        // A listener that never accepts, its queue filled: Linux then drops
        // further connection attempts, so they hang instead of being refused.
        let socket = tokio::net::TcpSocket::new_v4()?;
        socket.bind("127.0.0.1:0".parse()?)?;
        let silent = socket.listen(1)?;
        let silent_addr = silent.local_addr()?;
        let mut queued = Vec::new();
        let hangs = loop {
            match std::net::TcpStream::connect_timeout(&silent_addr, Duration::from_millis(300)) {
                Ok(stream) => queued.push(stream),
                Err(error) if error.kind() == std::io::ErrorKind::TimedOut => break true,
                Err(_) => break false,
            }
            if queued.len() > 64 {
                break false;
            }
        };
        assert!(hangs, "premise: a connection to the full queue hangs");

        let (root, paths) = temp_app_paths("api-docs-connect-timeout");
        services_dir_ready(&paths)?;
        record(&paths, "svc-hung", "vllm", silent_addr.port()).write()?;
        let server = start_with(
            paths.clone(),
            0,
            records_as_written(),
            Duration::from_millis(300),
            SERVICE_LIST_TTL,
        )?;
        let answered = tokio::time::timeout(
            Duration::from_secs(10),
            test_client()
                .get(format!(
                    "{}/services/svc-hung/api/v1/models",
                    server.base_url
                ))
                .send(),
        )
        .await;
        let response = answered.expect("try-it waited on a server that never accepts")?;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        server.stop();
        drop((queued, silent));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn docs_redirect_only_names_a_running_server() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-redirect");
        services_dir_ready(&paths)?;
        record(&paths, "svc-live", "vllm", 9).write()?;
        let server = start(&paths)?;
        let base = server.base_url.clone();
        let client = test_client();

        let redirect = client
            .get(format!("{base}/services/svc-live/docs"))
            .send()
            .await?;
        assert_eq!(redirect.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            redirect.headers()[header::LOCATION],
            "/services/svc-live/docs/"
        );

        // A page the user visits can send these; they get a 404 naming no
        // server, and no `Location` built from what they sent.
        for crafted in ["%0A", "svc-live%0D%0AX-Injected:%201", "not-running"] {
            let response = client
                .get(format!("{base}/services/{crafted}/docs"))
                .send()
                .await?;
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{crafted}");
            assert!(response.headers().get(header::LOCATION).is_none());
        }

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unreadable_records_are_reported_not_shown_as_no_servers() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-unreadable");
        services_dir_ready(&paths)?;
        record(&paths, "svc-live", "vllm", 9).write()?;
        // A `.json` entry that cannot be read as a file fails the whole load.
        fs::create_dir_all(paths.services_dir().join("broken.json"))?;
        let server = start(&paths)?;
        let base = server.base_url.clone();
        let client = test_client();

        for path in [
            "/",
            "/services/svc-live/docs",
            "/services/svc-live/docs/",
            "/services/svc-live/openapi.json",
            "/services/svc-live/api/v1/models",
        ] {
            let response = client.get(format!("{base}{path}")).send().await?;
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{path}: the server may be running; the records just could not be read"
            );
            assert!(
                response
                    .text()
                    .await?
                    .contains("could not read the service records"),
                "{path}"
            );
        }

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn forwarded_answers_cannot_run_as_pages_of_the_docs_origin() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-forward-headers");
        services_dir_ready(&paths)?;
        let model_port = start_echo_model_server().await;
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let server = start(&paths)?;

        let response = test_client()
            .post(format!(
                "{}/services/svc-live/api/v1/chat/completions",
                server.base_url
            ))
            .body("{}")
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::X_CONTENT_TYPE_OPTIONS],
            "nosniff"
        );
        assert_eq!(
            response.headers()[header::CONTENT_SECURITY_POLICY],
            FORWARDED_CSP
        );
        assert!(FORWARDED_CSP.starts_with("sandbox"));
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/json",
            "the server's own content type is kept"
        );

        server.stop();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// A model server that answers with the start of a stream and then never
    /// finishes it, like a long completion.
    fn start_endless_stream_server() -> u16 {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind((DEFAULT_LOCAL_HOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().map_while(std::result::Result::ok) {
                std::thread::spawn(move || {
                    let mut request = [0_u8; 4096];
                    let _ = stream.read(&mut request);
                    let _ = stream.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                          Transfer-Encoding: chunked\r\n\r\n5\r\ndata:\r\n",
                    );
                    let _ = stream.flush();
                    // Hold the stream open, unfinished, until the peer goes.
                    while matches!(stream.read(&mut request), Ok(read) if read > 0) {}
                });
            }
        });
        port
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stopping_does_not_wait_for_an_open_try_it_stream() -> Result<()> {
        let (root, paths) = temp_app_paths("api-docs-stop-stream");
        services_dir_ready(&paths)?;
        let model_port = start_endless_stream_server();
        record(&paths, "svc-live", "vllm", model_port).write()?;
        let server = start(&paths)?;

        let mut response = test_client()
            .post(format!(
                "{}/services/svc-live/api/v1/chat/completions",
                server.base_url
            ))
            .body("{}")
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.chunk().await?.is_some(), "the stream has started");

        // On a thread of its own, not `spawn_blocking`: a `stop()` that hangs
        // must fail this test, not park a blocking-pool thread the test
        // runtime then waits for when it shuts down.
        let started = std::time::Instant::now();
        let (stopped, stop_finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            server.stop();
            let _ = stopped.send(());
        });
        assert!(
            stop_finished
                .recv_timeout(SHUTDOWN_GRACE + Duration::from_secs(10))
                .is_ok(),
            "stop() waited on the open stream for {:?}",
            started.elapsed()
        );
        drop(response);
        fs::remove_dir_all(root).ok();
        Ok(())
    }
}
