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
- A tool call is refused if its name or any string in its arguments carries
  something a terminal would not show as it is: a control (an escape
  sequence, a carriage return, a newline, a tab), a format character (a bidi
  override, a zero-width space), a line separator, or any other character
  Unicode marks default-ignorable, such as the invisible tag characters. This
  applies to every tool, whoever built the call; `rocm_command` checks each
  `args` entry itself, after trimming it. A provider-assisted planner note
  carrying such a character is left out and counted.
- The one exception to that rule is a single zero-width joiner or non-joiner,
  or emoji or text presentation selector (VS15, VS16), directly after a
  character that is drawn, which is how an emoji such as `❤️` is spelled; a
  run of them, or one anywhere else, is refused.
- The exception does not hold for a value the model chose on a command line
  the CLI shows (the next rule says which): there any of these characters is
  refused, a single joiner included, since after a plain character it shapes
  nothing and makes two values look alike. Model-chosen values that are not
  on such a line get only the first rule, with its exception: a
  `path_exists` path, a `port_status` host, a `natural_language_plan`
  request, and a `proposal_action` proposal id, although its approval card
  shows it as `proposal <id>`.
- The command lines shown for a model's request (the approval card, the
  `advanced manual command:` line, the install preview's `command:` line, the
  request plan's `tool_call:` and `next_tool_call:` lines) quote every
  argument for a POSIX shell, so each reads back as one word in `sh`, bash or
  WSL. Windows shells read such a line differently, so a value the model
  chose is refused if it carries a character they treat as syntax even inside
  single quotes. The set, stated here once: a single quote, straight (`'`) or
  typographic (U+2018 to U+201B), which PowerShell reads as the end of the
  string, and `& | < > ^ % " !`, which `cmd.exe` acts on because it gives
  single quotes no meaning. The check reads the argv the call is shown as,
  not the JSON it arrived in, so a field that is never printed (a
  `rocm_command` `reason`) is not refused. It covers the model's tool calls in
  `rocm chat` and in the dashboard, the provider-assisted planner, and the
  plan `natural_language_plan` returns for a request the model wrote: the
  planner turns words of that request into argv values, so the plan's
  `tool_call:` lines are checked before the plan is shown (the planner's own
  placeholder `<model>` passes as the model of a `rocm serve`, where the
  planner writes it, and nowhere else). The
  refusal names the character and tells the user to supply the value
  themselves by running the `rocm` command it belongs to, and says that
  nothing was changed and no command was offered to paste for that call; a
  read-only check the same reply asked for may already have run, and in
  `rocm chat --prompt` nothing from the turn reaches standard output. Values the
  user typed are never held to this: an install folder named in the prompt,
  or in a dashboard slash command, is rendered as typed. `cmd.exe` still splits a
  quoted value at a space, so such a line does not paste there as one
  argument, but nothing in a model-chosen value can run as a command of its
  own.
- A `rocm serve` the model requests — through `launch_server`,
  `rocm_command`, the provider-assisted planner or a planned request — may
  only use long options `rocm serve` has, must name exactly one model, and may
  only name the model and each option value with letters, digits and
  `. _ - / : @ + \`, at most 256 of them and not starting with `-`; anything
  else, such as a model file whose path contains a space, is for the user to
  run with `rocm serve` directly.
- Prose that cannot be refused — the model's reply, the model name the server
  reports, tool results, an error message quoting either — is printed by
  `rocm chat --prompt` with terminal escape sequences (CSI, OSC including
  clipboard writes, DCS, SOS, PM, APC, and their 8-bit forms) and invisible
  characters removed and line breaks kept, so text a reply tries to conceal is
  shown in plain sight instead. A sequence left unterminated ends at its own
  line, so it cannot hide the lines after it. The dashboard chat draws
  through its terminal UI library, which drops control characters but not the
  other invisible ones.
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
