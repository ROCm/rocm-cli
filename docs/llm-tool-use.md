<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# LLM Tool Use

rocm-cli local assistants use structured tools, not shell commands.

## Design Rules

- The model chooses a tool call from a schema. rocm-cli validates the arguments
  before anything runs.
- Read-only ROCm checks can run immediately and return their output to the
  model.
- Installs, starts, stops, updates, deletes, settings changes, and other
  mutations must pause on the TUI approval screen.
- The model should inspect first, then request a change only when the inspection
  result shows it is useful.
- The model must use argv-style `rocm` arguments. It must not invent PowerShell,
  Bash, `cmd`, `git`, or arbitrary package-manager commands.
- CPU fallback is not a supported path. GPU-required ROCm commands must fail
  loudly when the ROCm GPU path is not ready.
- The built-in local assistant is fixed to `qwen`
  (`Qwen3-4B-Instruct-2507-GGUF`) served by Lemonade. vLLM and Lemonade are the
  general serving engines; the assistant may inspect or manage them for model
  serving, but it should not switch its own built-in chat engine away from
  Lemonade.
- On native Windows, vLLM live serving/install checks are skipped. The assistant
  should direct those requests to WSL/Linux and should not suggest CPU fallback.

This follows the same shape described by current tool-use docs: the application
defines tool schemas, the model requests a tool, the application executes the
tool, and the result is returned to the model for the next response. MCP tool
annotations such as `readOnlyHint` and `destructiveHint` are useful UI hints,
but rocm-cli still enforces approval in code.

## Local Assistant Examples

The assistant can inspect ComfyUI state:

```json
{"name":"rocm_command","arguments":{"args":["comfyui","status"]}}
```

For “is X running?” questions, inspect before answering and do not start or stop
anything. Use ComfyUI status for ComfyUI, `services list --all` for managed
model servers and serving engines, and `port_status` for loopback port
questions:

```json
{"name":"rocm_command","arguments":{"args":["services","list","--all"]}}
```

```json
{"name":"port_status","arguments":{"host":"127.0.0.1","port":8188}}
```

Treat `ready` and `running` as running, `starting` and `recovering` as starting,
`failed` and `stopped` as not running, and no matching service as unknown or not
managed by rocm-cli.

The assistant can read recent ComfyUI install/run logs without changing local
state:

```json
{"name":"rocm_command","arguments":{"args":["comfyui","logs"]}}
```

The assistant can request installing ComfyUI. rocm-cli shows a review card
before running it:

```json
{"name":"rocm_command","arguments":{"args":["comfyui","install"],"reason":"Install ComfyUI into ROCm CLI's app folder."}}
```

The assistant can request installing a serving engine through the existing
engine surface. The supported engines are `lemonade` and `vllm`:

```json
{"name":"rocm_command","arguments":{"args":["engines","install","vllm"],"reason":"Install the vLLM serving engine."}}
```

The assistant can request serving a model through vLLM by asking rocm-cli to
start the managed `vllm` engine. GPU execution is required:

```json
{"name":"rocm_command","arguments":{"args":["serve","Qwen/Qwen3.5-4B","--engine","vllm","--device","gpu_required","--managed"],"reason":"Start a local GPU vLLM server for this model."}}
```

To target a specific GPU, add `--gpu` with `auto` (default; first free GPU) or a
single index. Serving one model across multiple GPUs is not supported. CPU
fallback is never used when a GPU is busy or out of range:

```json
{"name":"rocm_command","arguments":{"args":["serve","Qwen/Qwen3.5-4B","--engine","vllm","--device","gpu_required","--gpu","1","--managed"],"reason":"Serve this model on GPU 1."}}
```

The assistant can request starting ComfyUI. rocm-cli shows the local URL and
tries to open the browser:

```json
{"name":"rocm_command","arguments":{"args":["comfyui","start"]}}
```

## Sources

- OpenAI Function Calling guide: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use guide: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Model Context Protocol tool annotations: https://modelcontextprotocol.io/specification/draft/schema#toolannotations
