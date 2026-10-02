// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use rocm_core::{AutomationTriggerEvent, DEFAULT_LOCAL_HOST, builtin_watcher, unix_time_millis};
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug)]
pub(crate) struct LocalWebhookSource {
    pub(crate) endpoint: String,
    pub(crate) receiver: mpsc::Receiver<AutomationTriggerEvent>,
    pub(crate) task: JoinHandle<()>,
}

#[derive(Clone)]
struct LocalWebhookState {
    sender: mpsc::Sender<AutomationTriggerEvent>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LocalWebhookEventRequest {
    pub(crate) watcher_hint: String,
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) service_id: Option<String>,
    #[serde(default)]
    pub(crate) reason: Option<String>,
    #[serde(default)]
    pub(crate) payload: Value,
}

pub(crate) async fn start_local_webhook_source(port: u16) -> Result<LocalWebhookSource> {
    let listener = TcpListener::bind((DEFAULT_LOCAL_HOST, port))
        .await
        .with_context(|| {
            format!("failed to bind local webhook source to {DEFAULT_LOCAL_HOST}:{port}")
        })?;
    let addr = listener
        .local_addr()
        .context("failed to read local webhook listener address")?;
    if !addr.ip().is_loopback() {
        bail!("local webhook source must bind to loopback; resolved address was {addr}");
    }

    let (sender, receiver) = mpsc::channel(64);
    let app = Router::new()
        .route("/health", get(local_webhook_health))
        .route("/automation-events", post(local_webhook_post_event))
        .with_state(LocalWebhookState { sender });
    let task = tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("local webhook source stopped: {error}");
        }
    });

    Ok(LocalWebhookSource {
        endpoint: format!("http://{addr}/automation-events"),
        receiver,
        task,
    })
}

pub(crate) async fn receive_local_webhook_event(
    receiver: &mut Option<mpsc::Receiver<AutomationTriggerEvent>>,
) -> Option<AutomationTriggerEvent> {
    if let Some(receiver) = receiver {
        receiver.recv().await
    } else {
        std::future::pending().await
    }
}

async fn local_webhook_health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "source": "local_webhook",
        "bind": "loopback_only",
    }))
}

async fn local_webhook_post_event(
    State(state): State<LocalWebhookState>,
    Json(request): Json<LocalWebhookEventRequest>,
) -> impl IntoResponse {
    let event = match local_webhook_event_from_request(request) {
        Ok(event) => event,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status": "rejected",
                    "error": error.to_string(),
                })),
            );
        }
    };

    if state.sender.send(event.clone()).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unavailable",
                "error": "local webhook receiver is not running",
            })),
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "queued",
            "source": event.source,
            "kind": event.kind,
            "watcher_hint": event.watcher_hint,
            "message": "queued automation event; dispatch uses existing watcher policy; no new action is granted by webhook payload",
        })),
    )
}

pub(crate) fn local_webhook_event_from_request(
    request: LocalWebhookEventRequest,
) -> Result<AutomationTriggerEvent> {
    let watcher_hint = request.watcher_hint.trim();
    if watcher_hint.is_empty() {
        bail!("watcher_hint is required");
    }
    if builtin_watcher(watcher_hint).is_none() {
        bail!("unknown watcher_hint: {watcher_hint}");
    }

    let kind = request.kind.trim();
    if kind.is_empty() {
        bail!("kind is required");
    }
    if !local_webhook_kind_allowed(watcher_hint, kind) {
        bail!(
            "kind `{kind}` is not accepted for watcher `{watcher_hint}`; local webhooks can only trigger accepted watcher event kinds"
        );
    }
    let service_id = request.service_id.as_deref().map(str::trim);
    if let Some(service_id) = service_id
        && !service_id.is_empty()
    {
        // Propagate ServiceId's own message verbatim (e.g. "... must not contain
        // path separators") rather than wrapping it in context, which anyhow's
        // top-level Display would otherwise hide.
        rocm_core::ServiceId::new(service_id)?;
    }
    match watcher_hint {
        "server-recover" if service_id.unwrap_or_default().is_empty() => {
            bail!("server-recover webhook events require service_id");
        }
        "cache-warm"
            if crate::watchers::payload_string(&request.payload, "artifact_ref").is_none() =>
        {
            bail!("cache-warm webhook events require payload.artifact_ref");
        }
        "driver-upgrade"
            if crate::watchers::payload_string(&request.payload, "component").as_deref()
                != Some("driver") =>
        {
            bail!("driver-upgrade webhook events require payload.component=driver");
        }
        _ => {}
    }

    Ok(AutomationTriggerEvent {
        at_unix_ms: unix_time_millis(),
        kind: kind.to_owned(),
        source: "local_webhook".to_owned(),
        watcher_hint: Some(watcher_hint.to_owned()),
        service_id: request
            .service_id
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
        reason: request
            .reason
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
        payload: request.payload,
    })
}

fn local_webhook_kind_allowed(watcher_id: &str, kind: &str) -> bool {
    match watcher_id {
        "therock-update" => kind == "schedule.tick",
        "server-recover" => matches!(
            kind,
            "service.manifest_recoverable"
                | "service.endpoint_recoverable"
                | "service.healthcheck_recoverable"
        ),
        "gpu-metrics" => matches!(kind, "gpu.metrics" | "gpu.metrics_unavailable"),
        "cache-warm" => kind == "cache.warm",
        "driver-upgrade" => kind == "update.available",
        "gpu-thermal-protect" => {
            matches!(kind, "gpu.thermal_pressure" | "gpu.memory_pressure")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time;

    #[test]
    fn local_webhook_event_rejects_unknown_watcher() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "unknown".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("unknown watcher_hint"));
    }

    #[test]
    fn local_webhook_event_rejects_kind_outside_existing_watcher_prefix() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_therock_update_accepts_exact_schedule_tick_only() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "therock-update".to_owned(),
            kind: "schedule.tick".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })?;
        assert_eq!(event.kind, "schedule.tick");

        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "therock-update".to_owned(),
            kind: "schedule.tick.extra".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();
        assert!(error.to_string().contains("accepted watcher event kinds"));
        Ok(())
    }

    #[test]
    fn local_webhook_gpu_metrics_rejects_thermal_action_kinds() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "gpu.thermal_pressure".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_server_recover_rejects_nonrecoverable_service_kind() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.started".to_owned(),
            service_id: Some("svc-1".to_owned()),
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_cache_warm_rejects_other_cache_events() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.evict".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_event_requires_service_id_for_server_recover() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("require service_id"));
    }

    #[test]
    fn local_webhook_event_rejects_service_id_path_separators() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: Some("../svc".to_owned()),
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("path separators"));
    }

    #[test]
    fn local_webhook_event_builds_loopback_source_trigger() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: Some("manual test".to_owned()),
            payload: json!({
                "summary": "manual webhook probe",
            }),
        })?;

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.watcher_hint.as_deref(), Some("gpu-metrics"));
        assert_eq!(event.kind, "gpu.metrics");
        assert_eq!(event.reason.as_deref(), Some("manual test"));
        Ok(())
    }

    #[test]
    fn local_webhook_cache_warm_requires_artifact_ref() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.warm".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("payload.artifact_ref"));
    }

    #[test]
    fn local_webhook_cache_warm_builds_prefetch_event() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.warm".to_owned(),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
            }),
        })?;

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.watcher_hint.as_deref(), Some("cache-warm"));
        assert_eq!(event.kind, "cache.warm");
        assert_eq!(
            event.payload.get("artifact_ref").and_then(Value::as_str),
            Some("Qwen/Test-1B#hf-main")
        );
        Ok(())
    }

    #[test]
    fn local_webhook_driver_upgrade_requires_driver_component() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.available".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({
                "component": "runtime",
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("payload.component=driver"));
    }

    #[test]
    fn local_webhook_driver_upgrade_rejects_other_update_events() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.checked".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({
                "component": "driver",
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_driver_upgrade_builds_update_event() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.available".to_owned(),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
                "available_version": "test-driver",
                "tool": "restart_server",
            }),
        })?;

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.watcher_hint.as_deref(), Some("driver-upgrade"));
        assert_eq!(event.kind, "update.available");
        assert_eq!(
            event.payload.get("component").and_then(Value::as_str),
            Some("driver")
        );
        Ok(())
    }

    #[test]
    fn local_webhook_gpu_thermal_protect_accepts_only_pressure_events() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-thermal-protect".to_owned(),
            kind: "gpu.thermal_pressure".to_owned(),
            service_id: Some("svc-hot".to_owned()),
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
            }),
        })?;
        assert_eq!(event.kind, "gpu.thermal_pressure");
        assert_eq!(event.watcher_hint.as_deref(), Some("gpu-thermal-protect"));
        assert_eq!(event.service_id.as_deref(), Some("svc-hot"));

        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-thermal-protect".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("accepted watcher event kinds"));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn local_webhook_http_endpoint_queues_valid_event() -> Result<()> {
        use std::io::{Read, Write};

        let mut source = start_local_webhook_source(0).await?;
        let addr = source
            .endpoint
            .strip_prefix("http://")
            .and_then(|value| value.strip_suffix("/automation-events"))
            .expect("endpoint should include host and path")
            .to_owned();
        let body = json!({
            "watcher_hint": "gpu-metrics",
            "kind": "gpu.metrics",
            "reason": "http smoke",
            "payload": {
                "summary": "http local webhook"
            }
        })
        .to_string();
        let mut stream = std::net::TcpStream::connect(&addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        write!(
            stream,
            "POST /automation-events HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;

        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer)?;
        let response = String::from_utf8_lossy(&buffer[..read]).to_string();
        assert!(response.starts_with("HTTP/1.1 202 Accepted"), "{response}");
        assert!(response.contains("no new action is granted"));
        let event = time::timeout(Duration::from_secs(2), source.receiver.recv())
            .await?
            .expect("queued webhook event should be received");
        source.task.abort();

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.kind, "gpu.metrics");
        assert_eq!(event.watcher_hint.as_deref(), Some("gpu-metrics"));
        assert_eq!(event.reason.as_deref(), Some("http smoke"));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn local_webhook_http_endpoint_rejects_malformed_json() -> Result<()> {
        use std::io::{Read, Write};

        let source = start_local_webhook_source(0).await?;
        let addr = source
            .endpoint
            .strip_prefix("http://")
            .and_then(|value| value.strip_suffix("/automation-events"))
            .expect("endpoint should include host and path")
            .to_owned();
        let body = "{";
        let mut stream = std::net::TcpStream::connect(&addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        write!(
            stream,
            "POST /automation-events HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;

        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer)?;
        let response = String::from_utf8_lossy(&buffer[..read]).to_string();
        source.task.abort();

        let status_line = response.lines().next().unwrap_or_default();
        assert!(status_line.starts_with("HTTP/1.1 4"), "{response}");
        assert!(response.to_ascii_lowercase().contains("json"), "{response}");
        Ok(())
    }
}
