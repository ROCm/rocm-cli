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
ROCm install required. Ships with inference engine adapters for PyTorch,
llama.cpp, Lemonade, ATOM, vLLM, and SGLang.

> [!IMPORTANT]
> **Tech Preview** -- This software is provided as-is, without warranty or
> guarantee of stability. APIs, commands, and behavior may change without
> notice. Intended for experimentation and early feedback only.

> Only nightly builds are available at this time.

## Installation

This repository is currently private. Install the [GitHub CLI](https://cli.github.com/) and authenticate once before downloading:

```bash
gh auth login
```

**Linux / WSL x86_64**:

```bash
gh release download nightly --repo ROCm/rocm-cli --pattern rocm --output ~/.local/bin/rocm
chmod +x ~/.local/bin/rocm
```

Direct link: <https://github.com/ROCm/rocm-cli/releases/download/nightly/rocm>

**Windows x86_64** (PowerShell):

```powershell
gh release download nightly --repo ROCm/rocm-cli --pattern rocm.exe --output "$env:LOCALAPPDATA\Microsoft\WindowsApps\rocm.exe"
```

Direct link: <https://github.com/ROCm/rocm-cli/releases/download/nightly/rocm.exe>

Rerun the same command to upgrade to the latest nightly.

## First run

```
rocm
```

Opens the interactive TUI. On first launch it walks through setup: choose an
install folder and let rocm-cli download and configure ROCm. After setup the
TUI shows live GPU utilization, active model servers, and a chat tab.

## Quick reference

| Command | Description |
|---|---|
| `rocm` | Open the TUI (runs setup on first launch) |
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

### Inference engines

```
rocm engines list
rocm engines install <engine> [--runtime-id KEY] [--python-version X.Y] [--reinstall]
rocm engines shell <engine>   [--runtime-id KEY | --env-id ID] [--shell PATH]
```

Supported engines: `lemonade`, `pytorch`, `llama.cpp`, `atom`, `vllm`, `sglang`.

### Model serving

Start a local OpenAI-compatible model server:

```
rocm serve <model> [--engine lemonade|pytorch|llama.cpp|atom|vllm|sglang]
                   [--device gpu_required|gpu_preferred|cpu_only]
                   [--gpu auto|<index>]
                   [--runtime-id KEY | --env-id ID]
                   [--host HOST] [--port PORT]
                   [--foreground | --managed]
                   [--allow-public-bind]
```

`--managed` runs the server in the background under rocm-cli's supervision.
`--foreground` attaches it to the current terminal.

`--gpu` selects which AMD GPU the server runs on. `auto` (the default) probes
per-GPU VRAM with `amd-smi` and picks the lowest-numbered GPU that is idle and
not already used by another rocm-cli server (managed or foreground), falling
back to the GPU with the most free memory. Pass a single index (`--gpu 1`) to
pin a specific device. The
selected GPU is exposed to the engine via `HIP_VISIBLE_DEVICES`. Serving one
model across multiple GPUs is not supported.

Show recommended models and hardware compatibility:

```
rocm model [--verbose]
```

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

Full-screen TUI with GPU utilization graphs, active serving instances,
benchmark results, and a chat tab backed by any configured provider.

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

This repo uses [prek](https://github.com/j178/prek) (a fast drop-in
replacement for `pre-commit`) to run the same checks locally that CI enforces:
`cargo fmt`, `clippy`, `cargo test`, `ruff` (Python), `shellcheck` (shell),
and PowerShell syntax.

```bash
uv tool install prek        # or: cargo install --locked prek
prek install                # fast checks on commit
prek install -t pre-push    # heavier checks on push (clippy + tests)
prek run --all-files        # run everything against the whole tree
```

Commits must be both cryptographically **signed** and carry a DCO
**`Signed-off-by`** trailer (use `git commit -s`). This is enforced by the
prek hooks above and by a blocking CI check. Enable signing once with:

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

See `docs/commit-signatures.md` for details (GPG signing, GitHub "Verified", and troubleshooting).

## More docs

- Testing and verification: `docs/testing.md`
- Developer manual QA: `docs/manual-testing.md`
- Engine plugin policy: `docs/engine-plugins.md`
- ATOM adapter: `docs/atom.md`
- vLLM adapter: `docs/vllm.md`
- SGLang adapter: `docs/sglang.md`
