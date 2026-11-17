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
- A tool call is refused if its name or any string in its arguments carries a
  terminal control or format character (an escape sequence, a carriage return,
  a bidi override, a zero-width character). Those values are printed — in the
  approval card, the request plan, tool output and error messages — and a
  terminal acts on such a character instead of showing it. This applies to
  every tool; `rocm_command` checks each `args` entry itself, after trimming it.
  The provider-assisted planner's argv is held to the same rule, and a planner
  note carrying such a character is left out and counted.
- CPU fallback is not a supported path. GPU-required ROCm commands must fail
  loudly when the ROCm GPU path is not ready. Keeping some of a model's layers
  in system RAM on a GPU that is ready (llama.cpp's `--n-gpu-layers`), which
  `rocm diagnose --model` suggests when a GGUF model does not fit in VRAM, is
  not CPU fallback, and the assistant may relay it.
- A `rocm serve` the model proposes through `rocm_command`, or the
  provider-assisted planner proposes in its tool call, is refused unless it
  asks for no public bind (`--allow-public-bind`), no CPU device, a loopback
  `--host` if it names one, and managed serving (`--managed`, never
  `--foreground`). One shared check applies those four rules to both, in that
  order, and a refused call is neither proposed for approval nor planned; the
  two refusals read alike and differ only in naming the local assistant or the
  provider planner.
- The built-in local assistant is fixed to `qwen`
  (`Qwen3-4B-Instruct-2507-GGUF`) served by Lemonade. vLLM, Lemonade and AIM (a
  Docker-based engine) are the general serving engines; the assistant may inspect or manage them for model
  serving, but it should not switch its own built-in chat engine away from
  Lemonade.
- The assistant prompt is grounded in the host. rocm-cli detects the operating
  system (including WSL), the AMD GPU and its `gfx` target, and which serving
  engines run there, and appends those facts to the system prompt on every chat
  surface. Platform statements belong in that block, not in the static prompt: a
  claim such as "on native Windows, vLLM live checks are skipped" is true only
  on native Windows, and asserting it unconditionally misinformed WSL users,
  where vLLM is the supported path. The assistant should answer platform
  questions from the injected facts and must never suggest CPU fallback.

## Tool Result Shape And Size

A tool result is an MCP envelope: a `content` array of text items, an optional
`structuredContent` object, and `isError`. For a command-backed tool the
envelope carries the command's output twice — as the `content` text and again
as `structuredContent.stdout` — because `rocm mcp-call` consumers read the
structured field. That full envelope is what `rocm mcp-call` prints and what
the approval replay path stores.

What reaches the *model* is narrower, because a tool result is not free: it is
sent back as a message and counts against the model's context on that request
and on every later turn of the same conversation. A locally served model is
commonly configured with a 32k-token context (llama.cpp's default `n_ctx`),
and the system prompt plus the tool schemas already take a few thousand of
those. So the dash shapes each result before the model sees it
(`run_rocm_read_tool` in
[tools.rs](../crates/rocm-dash-tui/src/agent/tools.rs)):

- **One copy.** `structuredContent.stdout` and `.stderr` are dropped when the
  envelope's own text already carries them. Fields that are not duplicates —
  `argv`, `exit_status`, and the data of tools whose payload lives only in
  `structuredContent` — are kept.
- **A size cap.** A result is bounded to `TOOL_RESULT_CHAR_BUDGET` characters
  of serialized JSON (16 KiB, roughly 4.5k tokens). A result within it passes
  through unchanged. One that is not is replaced by its plain-text rendering,
  cut on a line boundary, ending in
  `[output truncated: A of B characters shown, C omitted ...]`. The marker is
  part of the contract: the assistant skill tells the model to report a
  truncated result as partial rather than as a complete answer.

The CLI chat path (`rocm chat --prompt --tools`) reads only the envelope's
text, so it never sent the duplicate copy; it does not apply the cap.

Read-only commands are rendered so that the cheap form is the default:
`rocm_command` with `["model"]` returns the curated list `rocm model` prints,
and the per-recipe diagnostics need an explicit `["model", "--verbose"]`.

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
engine surface. The supported engines are `lemonade`, `vllm`, and `aim`:

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
