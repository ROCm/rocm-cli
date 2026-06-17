# rocm-cli PyTorch Engine Spec

## Current Scope Note

The early spec text below mentions CPU fallback. That is no longer an
implementation target for the current rocm-cli pass. The current product policy
is GPU-required local serving with no implicit CPU fallback: PyTorch must use a
managed TheRock ROCm runtime for serving, and `cpu_only` requests fail loudly
instead of becoming a fallback path.

Current default-engine note, audited 2026-06-06: Lemonade is now the default
local assistant/server engine. This document remains the PyTorch engine spec,
so PyTorch is still supported and tested as an explicit managed engine, but
older lines below that call PyTorch the default Windows engine are historical.

## Summary
- Build a first-party `pytorch` serving engine for `rocm-cli`.
- Keep it available as an explicit native local serving backend on Windows.
- Base it on TheRock PyTorch wheel sets installed into `pip` virtual environments managed by `rocm-cli`.
- Expose the same normalized local endpoint contract used by the other engines so the chat layer and tool layer do not care which backend is active.
- Keep scope fixed to single-node, single-model local inference in V1. This engine is not trying to compete with `vllm`, `sglang`, or `atom` on throughput or distributed scheduling.

## Goals
- Provide a native Windows local model-serving path that uses TheRock PyTorch wheels plus an installed AMD driver.
- Keep the install flow reproducible and isolated from the user’s global Python environment.
- Require ROCm GPU execution for local serving; `gpu_preferred` is normalized to
  `gpu_required`, and `cpu_only` is rejected instead of used as a fallback.
- Integrate cleanly with:
  - `rocm install sdk`
  - `rocm engines install pytorch`
  - `rocm serve ... --engine pytorch`
  - the `rocm` TUI
  - `rocmd` service supervision
- Expose an OpenAI-compatible local API for chat and text generation.

## Non-Goals
- Do not depend on TorchServe.
- Do not optimize for multi-node, tensor-parallel, or throughput-maximized serving.
- Do not promise support for every Hugging Face model.
- Do not support arbitrary pickled Python checkpoints by default.
- Do not make quantization the primary path. Quantized local models remain a better fit for `llama.cpp` in V1.

## Position in the Product
- On Windows:
  - default local engine: `lemonade`
  - explicit managed engine: `pytorch`
  - alternate local GPU engine: `llama.cpp` when a HIP-enabled llama-server and
    managed TheRock runtime are available
  - deferred engines: `vllm`, `sglang`, `atom`
- On Linux:
  - default local engine: `lemonade`
  - alternate local GPU engines: `vllm`, `llama.cpp`, `pytorch`, `sglang`, `atom`
    where their upstream ROCm support and managed TheRock runtime are present
  - `pytorch` is available as a compatibility engine for simple single-model
    serving, but it is not the default Linux GPU path.

## Engine Ownership
- This is a first-party engine owned by `rocm-cli`, not a thin wrapper around a third-party server.
- The plugin binary should be versioned and shipped with `rocm-cli`.
- The runtime environment used by the engine should be created from pinned wheel inputs and fully reconstructed when the lock changes.

## Architecture

### High-Level Model
- `rocm-cli` owns:
  - TheRock runtime selection
  - model recipe resolution
  - service lifecycle requests
  - user interaction and approvals
- `rocmd` owns:
  - long-lived service supervision
  - health checks
  - restart policy
  - log collection
- `pytorch` engine plugin owns:
  - environment materialization
  - framework import validation
  - model loading
  - request handling
  - local API process

### Process Layout
- One plugin process per launched model service.
- One model loaded per process in V1.
- No in-process multi-tenant server in V1.
- Service process layout:
  - Python interpreter from the managed engine env
  - `rocm_cli_pytorch_engine.server`
  - local loopback HTTP server
  - optional SSE streaming path

### Why This Shape
- One model per process gives predictable failure isolation.
- Windows startup and device validation failures are easier to diagnose when the service boundary is clean.
- A normalized HTTP surface makes it easy for `rocm-cli chat --provider local` to use the same adapter shape as other engines.

## Environment Model

### Core Principle
- The engine must not mutate the user’s system Python or a random active virtual environment.
- All package state lives under `~/.local/share/rocm-cli/`.

### Layout

```text
~/.local/share/rocm-cli/
  runtimes/
    therock/
      release/
      nightly/
  engines/
    pytorch/
      envs/
        <env_id>/
      locks/
        <runtime_id>.txt
      services/
        <service_id>.json
      logs/
      state/
  models/
    hf/
    local/
```

### Runtime vs Engine Env
- TheRock runtime install remains the product-level managed runtime.
- The `pytorch` engine should create a service env derived from:
  - selected TheRock runtime manifest
  - selected Python version
  - pinned engine dependency lock
- The engine env is a complete venv, not a partially mutated user env.
- The shared optimization point should be:
  - wheel cache
  - model cache
  - lockfile reuse
- Do not rely on fragile parent-venv inheritance tricks in V1.

### Environment Identifier
- `env_id` should be a stable hash of:
  - OS
  - architecture
  - runtime channel
  - TheRock runtime version
  - Python version
  - engine revision
  - engine dependency lock hash

This lets the engine rebuild deterministically and reuse envs safely.

## Install Flow

### `rocm install sdk`
- Installs the selected TheRock runtime manifest and wheel set.
- On Windows, this is always a managed `pip` venv flow in V1.

### `rocm engines install pytorch`
- Resolve the active or requested TheRock runtime.
- Resolve the engine dependency lock for that runtime.
- Create or reuse the matching `env_id`.
- Install the engine package set into the engine env.
- Validate:
  - Python import
  - `torch` import
  - tokenizer and transformers imports
  - local metadata index presence
- Record engine capabilities in the engine state store.

### Engine Dependency Set
- Mandatory V1 dependencies:
  - `rocm[libraries,devel]`
  - `torch`
  - `torchvision`
  - `torchaudio`
  - `transformers`
  - `accelerate`
  - `safetensors`
  - `tokenizers`
  - `sentencepiece` where needed
  - `jinja2`
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `huggingface_hub`
- Optional or deferred:
  - `bitsandbytes`
  - `optimum`
  - `flash-attn`
  - model-specific acceleration packages not broadly supported on Windows

### Install Policy
- Resolve TheRock runtime and PyTorch packages from the same TheRock package
  index. The install request must include `rocm[libraries,devel]`, `torch`,
  `torchvision`, and `torchaudio` together so the index can supply the matching
  ROCm dependencies instead of rocm-cli inventing fallback dependency sets.
- Use pinned versions from a generated lockfile, not loose `pip install latest`.
  When TheRock versions include build dates, match the `rocm[libraries,devel]`
  and PyTorch-family wheels by compatible dated version strings before writing
  the lockfile.
- Prefer wheel-only installs.
- Fail closed if the resolved dependency set requires building native extensions in V1.

## Plugin Protocol

### Transport
- JSON-RPC over stdio in foreground mode.
- Local control socket may be added later for long-running service introspection.

### Required Methods

#### `detect`
Input:
- optional `runtime_id`
- optional `device_filter`

Output:
- `installed`
- `env_id`
- `python_version`
- `torch_version`
- `transformers_version`
- `available_devices`
- `capabilities`
- `notes`

#### `install`
Input:
- `runtime_id`
- optional `python_version`
- `reinstall`

Output:
- `env_id`
- `installed_packages`
- `capabilities`
- `lock_hash`

#### `capabilities`
Output:
- `cpu: false`
- `rocm_gpu: true` when framework path is usable
- `openai_compatible: true`
- `tool_calling: false` in V1 unless a recipe explicitly enables it
- `quantized_models: limited`
- `distributed_serving: false`
- `reasoning_parser: false`

#### `resolve_model`
Input:
- `model_ref`
- `runtime_id`
- optional `device_policy`
- optional `recipe_override`

Output:
- `canonical_model_id`
- `task`
- `source`
- `revision`
- `loader`
- `trust_remote_code`
- `chat_template_mode`
- `dtype`
- `device_policy`
- `estimated_memory`
- `launch_defaults`
- `warnings`

#### `launch`
Input:
- `service_id`
- `env_id`
- `runtime_id`
- resolved model payload
- `host`
- `port`
- `device_policy`
- `endpoint_mode`
- `generation_defaults`

Output:
- `service_id`
- `pid`
- `endpoint_url`
- `log_path`
- `state_path`

#### `healthcheck`
Input:
- `service_id`

Output:
- `status`
- `model_loaded`
- `device`
- `uptime_sec`
- `queue_depth`
- `last_error`
- `tokens_per_sec` when available

#### `endpoint`
Input:
- `service_id`

Output:
- `endpoint_url`
- `api_style`
- `supported_routes`

#### `stop`
Input:
- `service_id`
- optional `force`

Output:
- `stopped`
- `graceful`

#### `logs`
Input:
- `service_id`
- optional `tail_lines`

Output:
- `log_path`
- `recent_lines`

## Device Model

### Windows GPU Detection
- The engine should treat Windows GPU usability as a runtime fact, not an assumption.
- Detection path:
  - validate installed driver in `rocm examine`
  - import `torch`
  - query `torch.cuda.is_available()`
  - inspect device properties through the PyTorch HIP-compatible API surface
  - map the result back to `rocm-cli` device metadata

### Device Policy
- Supported policies:
  - `gpu_required`
  - `gpu_preferred` as an alias that resolves to GPU-required execution
  - `cpu_only` is recognized only to reject it with a no-fallback error
- Windows default:
  - `gpu_required`
- CPU-only hosts:
  - fail before serving; CPU mode is not a fallback path in rocm-cli

### Device Selection Rules
- If `gpu_required` and no usable GPU is visible, fail before launch.
- If `gpu_preferred` is requested, treat it as `gpu_required`.
- If `cpu_only` is requested, reject the request with a no-fallback error.
- If GPU memory preflight fails, do not silently OOM. Return:
  - estimated required memory
  - detected memory
  - suggested smaller variants or alternate engines

## Model Contract

### Supported Sources
- Hugging Face model repo
- local filesystem path
- predeclared model aliases from the `rocm-cli` recipe index

### Required Properties for V1
- Transformers-compatible causal LM or seq2seq generation model
- tokenizer available through `transformers`
- `safetensors` weights preferred
- explicit recipe entry if `trust_remote_code` is required

### Default Safety Rules
- `trust_remote_code = false` by default
- `safetensors` preferred over pickle-based formats
- remote-code models must be allowlisted in the recipe index
- arbitrary Python package install during model load is forbidden

### Recipe Fields
- `model_id`
- `revision`
- `task`
- `loader = "transformers"`
- `trust_remote_code`
- `dtype`
- `device_policy`
- `context_window`
- `chat_template_mode`
- `tokenizer_options`
- `generation_defaults`
- `stop_sequences`
- `windows_overrides`
- `unsupported_combinations`

### Windows Recipe Overrides
- Allow per-model overrides for:
  - dtype
  - eager vs compiled execution
  - attention implementation
  - tokenizer worker settings
  - maximum batch size

## Endpoint Contract

### Default API Style
- OpenAI-compatible subset over HTTP on loopback by default.
- Bind `127.0.0.1` by default.
- Require explicit approval for non-loopback bind.

### Required Routes
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

### Deferred Routes
- `POST /v1/embeddings`
- tool-calling-specific routes beyond the normal OpenAI schema
- multi-model routing

### Request Subset
- Required request fields:
  - `model`
  - `messages` for chat
  - `prompt` for completions
- Supported generation controls in V1:
  - `max_tokens`
  - `temperature`
  - `top_p`
  - `stop`
  - `stream`
  - `seed`
- Optional fields may be accepted and ignored only if the response clearly reports unsupported behavior. Silent misbehavior is not acceptable.

### Streaming
- Streaming should use server-sent events.
- End-of-stream behavior should match the normalized local provider expectations used by `rocm-cli`.

### Error Model
- Return structured JSON errors with:
  - error code
  - message
  - retryability
  - actionable suggestion when possible

## Serving Behavior

### Load Sequence
1. Resolve runtime and engine env.
2. Validate imports and device availability.
3. Resolve model recipe.
4. Ensure weights are present or download them.
5. Load tokenizer.
6. Load model onto selected device.
7. Warm up one trivial inference pass if enabled.
8. Mark service `ready`.

### Service States
- `creating_env`
- `installing`
- `starting`
- `downloading_weights`
- `loading_tokenizer`
- `loading_model`
- `warming_up`
- `ready`
- `degraded`
- `failed`
- `stopping`
- `stopped`

### Concurrency
- V1 should prioritize correctness over throughput.
- Default behavior:
  - single loaded model per process
  - bounded in-memory request queue
  - serialized generation or very small batching
- High-throughput scheduler work is deferred.

## Commands and UX

### Expected Commands
- `rocm engines install pytorch`
- `rocm engines list`
- `rocm serve qwen --engine pytorch --device gpu_required`
- `rocm serve local/path/to/model --engine pytorch --device gpu`
- `rocm logs --service <id>`

### Expected TUI Plan
For an explicit `serve qwen on Windows with PyTorch` request:
1. detect Windows driver and TheRock runtime
2. resolve `pytorch` as the requested engine
3. validate or create the engine env
4. resolve model recipe and expected memory use
5. load `Qwen/Qwen2.5-1.5B-Instruct` using GPU-required policy
6. expose a local OpenAI-compatible endpoint

## Configuration

### Global Config Shape

```toml
[engines.pytorch]
enabled = true
default_on_windows = true
preferred_python = "3.11"
endpoint_mode = "openai"
host = "127.0.0.1"
startup_timeout_sec = 180
request_timeout_sec = 900
allow_cpu_fallback = false
max_loaded_models = 1

[engines.pytorch.model_cache]
backend = "huggingface"
path = "~/.local/share/rocm-cli/models/hf"

[engines.pytorch.windows]
device_policy = "gpu_required"
dtype = "auto"
compile = "off"
```

### Per-Service Manifest

```json
{
  "service_id": "svc_qwen25_primary",
  "engine": "pytorch",
  "runtime_id": "therock-release-7.11.0-py311-gfx1151-win64",
  "env_id": "win64-py311-therock7110-pytorch-a1b2c3",
  "model": {
    "canonical_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "main"
  },
  "device_policy": "gpu_required",
  "endpoint": {
    "host": "127.0.0.1",
    "port": 11435,
    "api_style": "openai"
  }
}
```

## Observability

### Logs
- Separate logs for:
  - env creation
  - model download
  - model load
  - request handling
  - supervisor events

### Health Data
- Report:
  - loaded model id
  - device type
  - estimated and current memory use
  - last successful request time
  - average tokens per second when measurable
  - last exception

### TUI Surface
- Sidebar status should show:
  - engine: `pytorch`
  - device: ROCm GPU
  - runtime id
  - model id
  - endpoint URL
  - health state

## Security
- Bind loopback by default.
- Require explicit user confirmation before:
  - opening a non-loopback port
  - trusting remote code for a model
  - pulling from an unapproved model source
- Keep the model cache and engine env under `rocm-cli` state directories.
- Prefer `safetensors`.
- Do not let model recipes execute arbitrary host commands.

## Failure Handling
- Common failure classes:
  - missing or incompatible driver
  - framework import failure
  - model download failure
  - unsupported model architecture
  - GPU out-of-memory
  - unsupported `trust_remote_code` requirement
- Every failure should produce:
  - machine-readable error code
  - user-readable explanation
  - suggested next step

## Acceptance Criteria

### Windows GPU
- Install TheRock runtime via `pip` venv.
- Install `pytorch` engine successfully.
- Detect usable AMD GPU through the framework path.
- Serve one supported chat model locally.
- Complete one non-streaming and one streaming request successfully.

### Windows No-Fallback Enforcement
- `cpu_only` requests fail before serving with a clear no-fallback error.
- Missing or unusable ROCm GPU state fails before serving instead of launching a
  CPU endpoint.

### Linux Compatibility
- Install `pytorch` engine on Linux.
- Serve one simple model on a managed TheRock ROCm GPU runtime.
- Prove the same normalized endpoint contract works.

### Product Integration
- `rocm` TUI can resolve `pytorch` when the user chooses or requests it.
- `rocmd` can restart a crashed `pytorch` service.
- `rocm logs` and `rocm examine` reflect engine state accurately.

## Deferred Work
- embeddings
- engine-native structured tool-calling beyond the rocm-cli local-assistant
  tool bridge
- advanced batching and scheduler work
- multi-model routing
- native distributed serving
- Windows-native `vllm`, `sglang`, or `atom` integration
- aggressive quantization and third-party acceleration plugins

## V1 Local Status

- Engine lockfile generation is implemented in the managed PyTorch engine path.
- The V1 local serving response shape is implemented through the normalized
  engine protocol and OpenAI-compatible local API surface.
- The initial Windows recipe path is implemented with verified PyTorch Qwen
  routing and GGUF alternatives routed to GPU-capable llama.cpp/llama-server
  when a managed TheRock runtime is available.
- Future changes should update the protocol/schema examples and recipe tests
  when the V1 serving contract changes.
