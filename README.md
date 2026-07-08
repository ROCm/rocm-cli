<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI

![ROCm](https://img.shields.io/badge/ROCm-Enabled-green)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.TXT)

```
 ██████╗  ██████╗  ██████╗███╗   ███╗     ██████╗██╗     ██╗
 ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██╔════╝██║     ██║
 ██████╔╝██║   ██║██║     ██╔████╔██║    ██║     ██║     ██║
 ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║     ██║     ██║
 ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ╚██████╗███████╗██║
 ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝     ╚═════╝╚══════╝╚═╝

        Local AI on AMD GPUs — one binary, zero setup
```

ROCm CLI is a command-line tool for setting up and running local AI on AMD GPUs, with a
full-screen TUI dashboard for GPU telemetry, model serving, and chat.

A single prebuilt binary for Linux and Windows. No Python, Rust, or existing
ROCm install required. Ships with inference engine adapters for Lemonade and
vLLM.

> [!IMPORTANT]
> **Tech Preview** -- This software is provided as-is, without warranty or
> guarantee of stability. APIs, commands, and behavior may change without
> notice. Intended for experimentation and early feedback only.

## Installation

The installer downloads a prebuilt bundle, verifies its SHA-256 checksum,
installs the `rocm` and `rocmd` binaries into `~/.local/bin`, and adds that
directory to your shell `PATH`. Rerun it any time to upgrade.

> [!NOTE]
> This repository is currently private, so the `curl | sh` one-liner only works
> once the repo and its release assets are public. Until then, use the
> **GitHub CLI fallback** in each section, which downloads through your
> authenticated session.

### Linux / WSL (x86_64)

Only nightly builds are published today, so pass the `nightly` channel:

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh -s -- nightly
```

Once a stable release exists, omit the argument to track the default `release`
channel instead:

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
```

<details>
<summary>Fallback while the repo is private (GitHub CLI)</summary>

Install the [GitHub CLI](https://cli.github.com/), authenticate once, then pull
the release bundle through your authenticated session and install both binaries:

```bash
gh auth login
gh release download nightly --repo ROCm/rocm-cli \
  --pattern 'rocm-cli-nightly-linux-amd64.tar.gz' --output /tmp/rocm-cli.tar.gz --clobber
tar -xzf /tmp/rocm-cli.tar.gz -C /tmp
mkdir -p ~/.local/bin
find /tmp -path '*/bin/rocm*' -type f -exec install -m 0755 {} ~/.local/bin/ \;
```

If `~/.local/bin` is not already on your `PATH`, add it (and persist it):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
</details>

### Windows (x86_64, PowerShell)

```powershell
$env:ROCM_CLI_CHANNEL = "nightly"
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

Drop the `ROCM_CLI_CHANNEL` line to track the default `release` channel once a
stable release is published.

<details>
<summary>Fallback while the repo is private (GitHub CLI)</summary>

```powershell
gh auth login
gh release download nightly --repo ROCm/rocm-cli `
  --pattern "rocm-cli-nightly-windows-amd64.zip" --output "$env:TEMP\rocm-cli.zip" --clobber
Expand-Archive "$env:TEMP\rocm-cli.zip" "$env:TEMP\rocm-cli" -Force
New-Item -ItemType Directory -Force "$env:USERPROFILE\.local\bin" | Out-Null
Get-ChildItem "$env:TEMP\rocm-cli" -Recurse -Filter "rocm*.exe" |
  Copy-Item -Destination "$env:USERPROFILE\.local\bin"
```

Add `%USERPROFILE%\.local\bin` to your user `PATH` if it is not already there.
</details>

## Build from source

Building requires [Rust](https://rustup.rs/); the pinned toolchain in
`rust-toolchain.toml` (currently 1.96.0) installs automatically via `rustup`.

```bash
git clone https://github.com/ROCm/rocm-cli
cd rocm-cli
cargo build --release
```

This produces the two binaries under `target/release/`:

- `rocm` — the CLI and interactive interfaces
- `rocmd` — the background telemetry daemon used by the dashboard

Run without installing:

```bash
cargo run --release --bin rocm -- examine
```

Or copy the release binaries onto your `PATH`:

```bash
install -m 0755 target/release/rocm target/release/rocmd ~/.local/bin/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development setup, test
commands, and commit-signing requirements.

## First run

```
rocm
```

With no arguments on an interactive terminal, `rocm` opens the **launcher** — a
small front-door menu that gets you to the common tasks:

- **Set up this system** — install or update ROCm
- **Serve a model** — run a model on your GPU
- **Diagnose & fix** — check GPU, driver, and ROCm
- **Chat** — talk to a local or API-backed model
- **Open full dashboard →** — escalate into the live dashboard (`rocm dash`)

Pick a row with the arrow keys and `Enter`; press `q` to quit. On a
non-interactive terminal (or piped output), `rocm` prints a one-shot status
summary instead of opening the launcher.

## Interactive interfaces

`rocm` ships two terminal UIs built on [ratatui](https://ratatui.rs/):

### The launcher (`rocm`)

The lightweight hub described above. It runs the guided **Set up**, **Serve**,
**Diagnose**, and **Chat** flows in place, and hands off to the full dashboard
when you need live instruments. This is the default surface for everyday use;
the legacy full-screen setup assistant has been retired.

### The dashboard (`rocm dash`)

The full-screen telemetry dashboard — every instrument and action on one screen.
It auto-starts an embedded `rocmd` daemon when none is running, then presents
five tabs (switch with `Tab`/`Shift+Tab` or number keys `1`–`5`):

| Tab | What it shows |
|---|---|
| **Home** | At-a-glance status: GPU, active runtime, running servers |
| **ROCm** | Guided ROCm/runtime actions with inline details |
| **Serving** | Start, inspect, and manage model servers |
| **Observe** | Live GPU utilization, instances, and benchmark telemetry |
| **Chat** | Assistant chat backed by a local server or configured provider |

Live mode reads telemetry over a Unix domain socket, so it requires Linux or
WSL. Use `rocm dash --demo` for a synthetic session that runs anywhere without a
GPU or daemon.

## Getting started

Before serving a model, ensure a managed ROCm runtime is configured:

```
rocm install sdk
```

This downloads TheRock ROCm wheels and a matching PyTorch stack into a managed
environment. On machines with an existing ROCm install, `rocm examine` will
show it as `legacy_rocm_status: detected_unmanaged` — running `rocm install sdk`
creates a separate managed runtime alongside it.

Then serve a model:

```
rocm serve Qwen/Qwen2.5-1.5B-Instruct
```

Use `rocm model` to see available model recipes and their GPU memory
requirements.

## Quick reference

| Command | Description |
|---|---|
| `rocm` | Open the launcher menu (setup, serve, diagnose, chat, dashboard) |
| `rocm examine` | Check GPU, ROCm install, engines, and managed folders |
| `rocm install sdk` | Install TheRock ROCm wheels into a managed Python environment |
| `rocm install driver` | Install the AMD kernel driver on Linux |
| `rocm serve <model>` | Start a local OpenAI-compatible model server |
| `rocm dash` | Open the full-screen telemetry dashboard |
| `rocm setup status` | Show first-time setup state |
| `rocm version` | Print the rocm-cli version |
| `rocm completions <shell>` | Print a shell completion script (bash, zsh, fish, elvish, powershell) |

## Commands

### ROCm installation

```
rocm install sdk    [--channel release|nightly] [--format wheel|tarball]
                    [--version X.Y.Z | --build-date YYYY-MM-DD]
                    [--family gfx110X-all] [--prefix PATH] [--dry-run]

rocm install driver [--dkms] [--yes] [--dry-run] [--reconcile]

rocm update         [--apply] [--runtime KEY] [--activate] [--dry-run]
```

`install sdk` downloads TheRock ROCm wheels into a Python environment managed
by rocm-cli. `install driver` installs the AMD kernel driver on Linux (DKMS or
native package). `update` checks for a newer ROCm package; pass `--apply` to
install it.

### Runtime management

Manage multiple side-by-side ROCm installs:

```
rocm runtimes list
rocm runtimes activate <runtime-key>
rocm runtimes rollback
rocm runtimes uninstall <runtime-key>
rocm runtimes import <manifest-file> [--replace]
rocm runtimes adopt --python <path> [--root <path>] [--runtime-id ID]
                    [--runtime-key KEY] [--channel LABEL] [--replace]
```

`adopt` registers an existing TheRock-based Python environment as a managed
runtime. It does not work with standard ROCm package installs (e.g.
`/opt/rocm`); use `rocm install sdk` instead.

### Inference engines

```
rocm engines list
rocm engines install <engine> [--runtime-id KEY] [--python-version X.Y] [--reinstall]
rocm engines shell <engine>   [--runtime-id KEY | --env-id ID] [--shell PATH]
```

Supported engines: `lemonade`, `vllm`.

### Model serving

Start a local OpenAI-compatible model server:

```
rocm serve <model> [--engine lemonade|vllm]
                   [--device gpu_required|gpu_preferred|cpu_only]
                   [--gpu auto|<index>]
                   [--runtime-id KEY | --env-id ID]
                   [--host HOST] [--port PORT]
                   [--verbose] [--foreground | --managed]
                   [--no-smoke-test]
                   [--allow-public-bind]
```

By default the server runs in the background under rocm-cli's supervision and
prints a deployment summary — a progress indicator while it starts, then a table
with the status, the full inference endpoint, the API-qualified model name, and a
quick smoke test (time to first token and approximate tokens/sec). Control
returns to your shell with the server still running; manage it later with `rocm
services` (below).

`--verbose` (or `--foreground`) instead attaches to the server in the current
terminal and streams every engine log line — use it to debug a startup problem.
The server still runs as a managed background process, so while streaming you can
press **Ctrl-D to detach** — the log stream stops, your shell comes back, and the
server keeps running (manage it afterward with `rocm services`). Press **Ctrl-C**
to stop the server instead. `--managed` is the explicit form of the default
background behavior. `--no-smoke-test` skips the post-startup inference probe.

Use full HuggingFace model IDs (e.g., `Qwen/Qwen2.5-1.5B-Instruct`) for
reliable cross-engine compatibility. Short aliases from `rocm model` may not
resolve with all engines.

Some models (e.g., Llama) are gated and require HuggingFace authentication.
Log in with `huggingface-cli login` or set `HF_TOKEN` in your environment
before serving gated models.

`--gpu` selects which AMD GPU the server runs on. `auto` (the default) probes
per-GPU VRAM with `amd-smi` and picks the lowest-numbered GPU that is idle and
not already used by another rocm-cli server (managed or foreground), falling
back to the GPU with the most free memory. Pass a single index (`--gpu 1`) to
pin a specific device. The
selected GPU is exposed to the engine via `HIP_VISIBLE_DEVICES`. Serving one
model across multiple GPUs is not supported. `--gpu` is ignored with
`--device cpu_only` (the model runs on CPU). Because selection uses the
`amd-smi` ordinal but is applied via `HIP_VISIBLE_DEVICES`, rocm-cli warns when
`ROCR_VISIBLE_DEVICES` is set, since the two orderings can diverge.

Show recommended models and hardware compatibility:

```
rocm model [--verbose]
```

`rocm model` prints a curated catalog of popular open-weight models grouped by
the hardware path they target — Strix Halo (Lemonade / llama.cpp) and MI300X
(vLLM) — using canonical Hugging Face ids, and shows for each the quantization
that fits a single GPU. Strix Halo entries use the `owner/repo:variant` GGUF form
(e.g. `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M`) that `rocm serve` needs; MI300X
entries serve at BF16. This catalog ships inside the binary, so it is available
offline; when a recipe index is configured instead, the header names it.
`--verbose` also lists the other recipes (the default assistant and smoke/test
paths) that `rocm serve` can still resolve but that are hidden from the list.

The catalog is only a starting point: you can serve any compatible Hugging Face
model by passing its id to `rocm serve` — `owner/repo` for vLLM, or
`owner/repo:<quant>` for a Lemonade GGUF.

Manage background servers started with `--managed`:

```
rocm services list [--all]
rocm services logs <service-id>
rocm services stop <service-id> [--yes]
rocm services restart <service-id> [--yes]
```

### Dashboard

```
rocm dash [--demo] [--replay <file>]
```

Full-screen TUI with Home, ROCm, Serving, Observe, and Chat tabs — GPU
utilization graphs, active serving instances, benchmark results, guided actions,
and a chat tab backed by any configured provider. See
[Interactive interfaces](#interactive-interfaces) for the tab breakdown.

- `--demo` runs a deterministic synthetic session with no GPU or daemon needed,
  works on all platforms.
- `--replay <file>` replays a recorded NDJSON session.
- Live mode requires Unix domain sockets (Linux and WSL only).

### Chat

```
rocm chat [--provider anthropic|openai|...] [--model NAME] [--prompt TEXT] [--tools]
```

Chat with an AI provider from the terminal. Reads from stdin when `--prompt` is
omitted.

### ComfyUI

Install and manage ComfyUI for image generation (alias: `rocm comfy`):

```
rocm comfyui install    [--runtime-id KEY] [--reinstall] [--dry-run]
rocm comfyui start      [--host HOST] [--port PORT] [--no-open-browser]
rocm comfyui stop
rocm comfyui status
rocm comfyui logs       [--lines N]
rocm comfyui models-path
```

### Automations

```
rocm automations list
rocm automations enable <watcher-id>  [--mode observe|propose|contained]
rocm automations disable <watcher-id>
```

Optional background checks that can propose or apply changes automatically.

### Configuration

```
rocm config show
rocm config set-default-engine <engine>
rocm config clear-default-engine
rocm config set-default-runtime <runtime-id>
rocm config clear-default-runtime
rocm config set-engine <engine> [--runtime-id KEY | --env-id ID | --clear]
rocm config set-telemetry local|off
rocm config set-planner-provider <provider>
rocm config clear-planner-provider
rocm config enable-provider <provider>
rocm config disable-provider <provider>
rocm config set-provider-key <provider>
rocm config clear-provider-key <provider>
```

### Logs and cleanup

```
rocm logs [--service <service-id>] [--search TERM ...]

rocm uninstall [--yes] [--dry-run]
               [--keep-binaries] [--keep-config] [--keep-data] [--keep-cache]
```

### Shell completions

`rocm completions <shell>` prints a completion script for the given shell to
stdout. Supported shells are `bash`, `zsh`, `fish`, `elvish`, and `powershell`.

```
rocm completions <bash|zsh|fish|elvish|powershell>
```

Install the script for your shell:

```
# bash (per-user, no sudo; add this line to ~/.bashrc to persist)
source <(rocm completions bash)
# bash (system-wide; requires the bash-completion package)
rocm completions bash | sudo tee /etc/bash_completion.d/rocm > /dev/null

# zsh (per-user; the directory must be on $fpath and compinit must run)
mkdir -p ~/.zsh/completions
rocm completions zsh > ~/.zsh/completions/_rocm
# then in ~/.zshrc, before `compinit`:
#   fpath=(~/.zsh/completions $fpath)
#   autoload -Uz compinit && compinit

# fish
mkdir -p ~/.config/fish/completions
rocm completions fish > ~/.config/fish/completions/rocm.fish

# elvish (run once; re-running appends a duplicate block to rc.elv)
mkdir -p ~/.config/elvish
rocm completions elvish >> ~/.config/elvish/rc.elv

# powershell (current session only; to persist, append the output to $PROFILE)
rocm completions powershell | Out-String | Invoke-Expression
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## More docs

- Testing and verification: `docs/testing.md`
- Developer manual QA: `docs/manual-testing.md`
- Engine plugin policy: `docs/engine-plugins.md`
- vLLM adapter: `docs/vllm.md`
