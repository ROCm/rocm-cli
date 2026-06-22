# Automations

rocm-cli automations are local watcher events routed through explicit watcher
policy. A webhook payload is an event input, not an execution API: it cannot
grant a new action, choose a tool, override watcher mode, or run shell.

## Local Webhook Source

Start the persistent watcher loop with a loopback-only webhook source:

```bash
rocmd run --automations-enabled --local-webhook-port 18080
```

The endpoint is shown as local event intake in `rocm automations` while the
background service is active:

```text
local event intake: http://127.0.0.1:18080/automation-events
```

Send a local JSON event:

```bash
curl -sS \
  -H "Content-Type: application/json" \
  -d '{"watcher_hint":"gpu-metrics","kind":"gpu.metrics","reason":"manual smoke","payload":{"summary":"manual local webhook"}}' \
  http://127.0.0.1:18080/automation-events
```

Accepted fields:

- `watcher_hint`: one of the existing watcher ids, such as `gpu-metrics`,
  `gpu-thermal-protect`, `server-recover`, `cache-warm`, `driver-upgrade`, or
  `therock-update`.
- `kind`: one exact event kind accepted by that watcher:
  `gpu.metrics` or `gpu.metrics_unavailable` for read-only metrics,
  `gpu.thermal_pressure` or `gpu.memory_pressure` for thermal protection,
  `service.manifest_recoverable`, `service.endpoint_recoverable`, or
  `service.healthcheck_recoverable` for server recovery, `cache.warm`,
  `update.available`, or `schedule.tick`.
- `service_id`: required for `server-recover` events. The service must still
  currently look recoverable before a restart review or contained restart can
  run.
- `payload.artifact_ref`: required for `cache-warm` events. Use
  `<model-ref>#<artifact-id>` from the model recipe registry.
- `payload.component`: must be `driver` for `driver-upgrade` events.
- `reason`: optional human-readable detail.
- `payload`: optional JSON object recorded with the event.

The listener only binds to `127.0.0.1`. There is no LAN or cloud webhook
listener, and local webhook events still dispatch through the configured
`observe`, `propose`, or `contained` policy for the target watcher.

## Restricted Tool API

Automation actions never receive arbitrary shell access. Reviewed requests and
contained watcher work can only call this restricted tool surface:

- `check_updates`: run a read-only `rocm update` check.
- `examine_snapshot`: capture a read-only examine summary.
- `list_servers`: list rocm-cli managed servers.
- `restart_server`: restart one explicit managed server.
- `stop_server`: stop one explicit managed server.
- `prefetch_artifact`: prepare one recipe-registered model artifact, with
  network access disabled until a reviewed source policy is present.
- `notify_user`: record a local notification/audit message.
- `driver_plan`: show a read-only Linux driver plan; no driver install runs.

For `therock-update` in contained mode, `check_updates` runs `rocm update`
read-only. If that report says a ROCm runtime update is available, rocm-cli
records a local `notify_if_newer` notification/audit event. It does not apply
the update.

Pending and recent automation review requests appear in `rocm automations` and
the TUI `/automations` view using plain action text such as `Restart a model
server`, `Prepare a model file`, or `Show a driver plan`. Pending requests show
the exact `/automations approve <id>` and `/automations reject <id>` controls.
Raw backend action and restricted-tool names are kept out of this overview so
the user-facing history stays readable.

For `cache-warm`, the webhook accepts exactly `cache.warm` and can only request
a reviewed prefetch request for an artifact ref already present in the model
recipe registry:

```powershell
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"watcher_hint":"cache-warm","kind":"cache.warm","reason":"idle window","payload":{"artifact_ref":"<model-ref>#<artifact-id>"}}' `
  http://127.0.0.1:18080/automation-events
```

Approving that review request still runs the restricted `prefetch_artifact`
tool.
Artifact bytes are not downloaded unless the prefetch command receives an
explicit source policy such as `--allow-artifact-download`.

To approve a cache-warm review request for download in the TUI, open
`/automations` or `/reviews`, choose the request, and use the rows under it:

- `Download allowed`: Left/Right turns download permission on or off.
- `Max download`: Enter types a byte limit; Left/Right chooses common limits.
- `Hugging Face`: Left/Right allows token use for gated Hugging Face files
  after Download is enabled.

Then select the review row and press Enter or Y to review/approve. Esc backs
out without rejecting; N rejects the request.

For scripts or debugging, the equivalent edit commands are:

```text
/automations edit <review-id> allow-download yes
/automations edit <review-id> artifact-max-bytes <bytes>
```

For gated Hugging Face artifacts, also allow Hugging Face token use:

```text
/automations edit <review-id> allow-huggingface-download yes
```

Webhook payloads cannot set these approval fields. They must be added during
review.

For `driver-upgrade`, the webhook accepts exactly a local `update.available`
signal with `payload.component=driver`. This is not a real AMD driver update
feed. In `propose` mode it queues a reviewed read-only driver plan request. In
`contained` mode it runs the restricted `driver_plan` tool directly. It does
not install drivers:

```powershell
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"watcher_hint":"driver-upgrade","kind":"update.available","reason":"local driver update signal","payload":{"component":"driver"}}' `
  http://127.0.0.1:18080/automation-events
```

Approving a proposal runs the restricted `driver_plan` tool. Contained mode
runs the same restricted tool without a proposal. It shows the driver plan
only; no driver will be installed.

For `gpu-thermal-protect`, rocm-cli can also create events from local
`amd-smi` monitor snapshots. The local webhook accepts only exact
`gpu.thermal_pressure` or `gpu.memory_pressure`. In `observe` mode it records
the pressure only. In `propose` and `contained` modes it queues a reviewed
`stop_server` proposal only when the target managed server is explicit or
there is exactly one running managed server. It does not stop a model server
automatically:

```powershell
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"watcher_hint":"gpu-thermal-protect","kind":"gpu.thermal_pressure","service_id":"<service-id>","reason":"hotspot_temperature_threshold","payload":{"summary":"GPU 0 hotspot temperature is 96 C (limit 95 C)"}}' `
  http://127.0.0.1:18080/automation-events
```

## Artifact Prefetch Action

The reviewed artifact-prefetch action keeps downloads disabled unless the caller
explicitly approves a source policy:

```bash
rocmd sandbox-run prefetch_artifact \
  --artifact-ref <model-ref>#<artifact-id> \
  --allow-artifact-download \
  --artifact-max-bytes <bytes> \
  --allow-native-fallback
```

Approved prefetch currently supports direct HTTP(S) artifacts that declare
`size_bytes` and `sha256` in the signed recipe metadata. rocm-cli downloads
only within the approved byte limit, verifies the hash, then writes a cache
marker under the rocm-cli model artifact cache. Missing hashes, gated sources,
and non-direct sources stay blocked unless a source-specific policy below
explicitly applies.

Signed recipes can include download rules. rocm-cli shows those rules before a
download and enforces them during prefetch. A rule can require a checked HTTPS
download, a public Hugging Face file, a Hugging Face token, a specific allowed
site, or manual download only. A normal download approval does not override
manual download only or a required Hugging Face token.

For gated Hugging Face artifacts, approve the Hugging Face source policy and
provide a token through the environment. On Windows PowerShell:

```powershell
$env:ROCM_CLI_HUGGINGFACE_TOKEN = "hf_..."
rocmd sandbox-run prefetch_artifact `
  --artifact-ref "<model-ref>#<artifact-id>" `
  --allow-artifact-download `
  --allow-huggingface-download `
  --artifact-max-bytes <bytes> `
  --allow-native-fallback
```

On Linux or WSL:

```bash
ROCM_CLI_HUGGINGFACE_TOKEN=hf_... \
rocmd sandbox-run prefetch_artifact \
  --artifact-ref <model-ref>#<artifact-id> \
  --allow-artifact-download \
  --allow-huggingface-download \
  --artifact-max-bytes <bytes> \
  --allow-native-fallback
```

`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are also accepted. The token is not put
on the rocmd command line or included in JSON output, and rocm-cli refuses to
send it to non-Hugging Face URLs or over plain HTTP.
