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

        Local AI on AMD GPUs: one install, zero setup
```

ROCm CLI is a command-line tool for setting up and running local AI on AMD GPUs, with a
full-screen terminal user interface (TUI) dashboard for GPU telemetry, model serving, and chat.

It ships as a prebuilt bundle for Linux and Windows (x86_64) that contains the
`rocm` command-line tool and the `rocmd` background daemon. It needs no
Python, Rust, or existing ROCm install, and includes inference engine adapters
for Lemonade and vLLM.

<!-- platform-support-table-start -->
| Platform | Prebuilt binary | Notes |
|---|---|---|
| Linux (x86_64) | Yes | Ubuntu 24.04 or newer; full support, including the live dashboard and both inference engines |
| Windows (x86_64) | Yes | CLI and Lemonade serving; no live dashboard or vLLM |
| WSL2 (x86_64) | Yes (Linux binary) | Ubuntu 24.04 or newer; supports the live dashboard and both inference engines, but `rocm diagnose` does not inspect the GPU yet; see [docs/wsl.md](https://github.com/ROCm/rocm-cli/blob/main/docs/wsl.md) for setup |
| macOS | No | No official installer, release, CI, or QA coverage |
<!-- platform-support-table-end -->

Live dashboard telemetry requires Linux or WSL2 (see
[Interactive interfaces](#interactive-interfaces)). vLLM serving is Linux or WSL2
only (see [docs/vllm.md](docs/vllm.md)).
<!-- docs-site: platform-notes-end -->

The minimum supported Linux release, native or under WSL2, is Ubuntu 24.04. On
other distributions the equivalent requirement is glibc 2.38 with
`GLIBCXX_3.4.32`: that is what the Lemonade engine is linked against, and every
published build of it needs those versions, so there is no older release to fall
back to. Ubuntu 22.04 ships glibc 2.35 and cannot run it; Ubuntu 24.04 provides
glibc 2.39 and `GLIBCXX_3.4.33`.

> [!IMPORTANT]
> **Tech Preview:** This software is provided as-is, without warranty or
> guarantee of stability. APIs, commands, and behavior might change without
> notice. Intended for experimentation and early feedback only.

## Demos

### ROCm CLI

Inspect the environment, discover engines and models, find a running service,
and chat with a locally served model:

![ROCm CLI demo](https://raw.githubusercontent.com/ROCm/rocm-cli/media/cli.gif)

### ROCm Console

Explore simulated GPU telemetry, model serving, and offline chat in the
full-screen Console:

![ROCm Console demo](https://raw.githubusercontent.com/ROCm/rocm-cli/media/console.gif)

<!--
The GIFs above are generated in CI and served from the orphan `media` branch;
they are never committed to source branches. See docs/demos.md to regenerate or
add a demo. Until the demo-gifs workflow has run once, these links 404.
-->

## Installation

The installer downloads a prebuilt bundle, verifies its SHA-256 checksum,
installs the `rocm` and `rocmd` binaries into `~/.local/bin`, and adds that
directory to your shell `PATH`. Rerun it any time to upgrade.

### Linux and WSL2 (x86_64)

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh
```

This tracks the default `release` channel. For nightly builds, pass the
`nightly` channel instead:

```bash
curl -fsSL https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh | sh -s -- nightly
```

### Windows (x86_64, PowerShell)

```powershell
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

This tracks the default `release` channel. For nightly builds, set
`ROCM_CLI_CHANNEL` to `nightly` first:

```powershell
$env:ROCM_CLI_CHANNEL = "nightly"
irm https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.ps1 | iex
```

### Verify the installation

Open a new terminal so the updated `PATH` takes effect, then run:

```
rocm version
```

The output shows the ROCm CLI version, release tag or branch, and commit hash.

<!-- docs-site: verify-next-start -->
Continue with [First run](#first-run).
<!-- docs-site: verify-next-end -->

## Build from source

Building requires [Rust](https://rustup.rs/); the pinned toolchain in
`rust-toolchain.toml` (currently 1.96.0) installs automatically through `rustup`.

```bash
git clone https://github.com/ROCm/rocm-cli
cd rocm-cli
cargo build --release
```

This produces the two binaries under `target/release/`:

- `rocm`: the CLI and interactive interfaces
- `rocmd`: the background telemetry daemon used by the dashboard

Run without installing:

```bash
cargo run --release --bin rocm -- examine
```

Or copy the release binaries onto your `PATH`:

```bash
install -m 0755 target/release/rocm target/release/rocmd ~/.local/bin/
```

See [CONTRIBUTING.md](https://github.com/ROCm/rocm-cli/blob/main/CONTRIBUTING.md) for the full development setup, test
commands, and commit-signing requirements.

## First run

Launch ROCm CLI with no arguments:

```
rocm
```

With no arguments on an interactive terminal, `rocm` opens the **launcher**, a
small front-door menu that gets you to the common tasks:

- **Set up this system:** install or update ROCm
- **Serve a model:** run a model on your GPU
- **Diagnose & fix:** check GPU, driver, and ROCm
- **Chat:** talk to a local or API-backed model
- **Open full dashboard →:** escalate into the live dashboard (`rocm dash`)

Pick a row with the arrow keys and `Enter`. Press `q` or `Ctrl-C` to quit;
`Ctrl-C` quits from the launcher and the dashboard alike and restores your
terminal.

The one exception is the dashboard's console for a **running** job. There,
`Ctrl-C` keeps its existing meaning of "cancel this job" and does not quit. Once
that job finishes, `Ctrl-C` quits there too.

On a non-interactive terminal (or piped output), `rocm` prints a one-shot status
summary instead of opening the launcher.

## Interactive interfaces

`rocm` ships two terminal user interfaces (UIs) built on [ratatui](https://ratatui.rs/):

### The launcher (`rocm`)

The lightweight hub described above. It runs the guided **Set up**, **Serve**,
**Diagnose**, and **Chat** flows in place, and hands off to the full dashboard
when you need live instruments. This is the default surface for everyday use;
the legacy full-screen setup assistant has been retired.

### The dashboard (`rocm dash`)

The full-screen telemetry dashboard shows every instrument and action on one screen.
It auto-starts an embedded `rocmd` daemon when none is running, then presents
five tabs (switch with `Tab`/`Shift+Tab` or number keys `1`–`5`):

| Tab | What it shows |
|---|---|
| **Home** | At-a-glance status: GPU, active runtime, running servers |
| **ROCm** | Guided ROCm and runtime actions with inline details |
| **Serving** | Start, inspect, and manage model servers |
| **Observe** | Live GPU utilization, instances, and benchmark telemetry |
| **Chat** | Assistant chat backed by a local server or configured provider |

Live mode reads telemetry over a Unix domain socket, so it requires Linux or
WSL2. Use `rocm dash --demo` for a synthetic session that runs anywhere without a
GPU or daemon.

## Configure ROCm and serve a model

Before serving a model, ensure a managed ROCm runtime is configured:

```
rocm install sdk
```

This downloads TheRock ROCm wheels and a matching PyTorch stack into a managed
environment. On machines with an existing ROCm install, `rocm examine` will
show it as `legacy_rocm_status: detected_unmanaged`. Running `rocm install sdk`
creates a separate managed runtime alongside it.

By default, `rocm install sdk` installs ROCm 7.14. To install ROCm 10.0 or
10.1, add `--version` and `--family` with the exact GPU arch from
`rocm examine`, for example `--version 10.1.0 --family gfx1200`.

<!-- docs-site: install-gate-start -->
Running the command when a managed runtime is already the active default asks
first, because the new install takes over as the active default. See
[ROCm installation](#rocm-installation) for that gate, the flags that approve it
without a prompt, and the ROCm 10 and newer requirements.
<!-- docs-site: install-gate-end -->

Then serve a model:

```
rocm serve qwen
```

`qwen` is a built-in alias for a small assistant model that serves out of the
box.

<!-- docs-site: serve-note-start -->
You can also serve any compatible Hugging Face model directly. See
[Model serving](#model-serving) for the GGUF versus safetensors rule, because
which form works depends on the engine your GPU selects.
<!-- docs-site: serve-note-end -->

## Quick reference

The most common commands at a glance.

| Command | Description |
|---|---|
| `rocm` | Open the launcher menu (setup, serve, diagnose, chat, dashboard) |
| `rocm examine` | Check GPU, ROCm install, engines, and managed folders |
| `rocm diagnose` | Match this machine against known failure modes in ROCm, PyTorch, and llama.cpp |
| `rocm diagnose --model <model>` | Say whether a model will run here, before downloading it |
| `rocm diagnose --report` | Show what this machine would contribute to a problem report, and send nothing |
| `rocm fix [<fix-id>]` | Apply a fix reported by `rocm diagnose` |
| `rocm install sdk` | Install TheRock ROCm wheels into a managed Python environment |
| `rocm install driver` | Install the AMD kernel driver on Linux |
| `rocm serve <model>` | Start a local OpenAI-compatible model server |
| `rocm remote serve <machine> <model>` | Serve a model on another GPU machine over your private network |
| `rocm dash` | Open the full-screen telemetry dashboard |
| `rocm bench load --endpoint <url>` | Load-test a local OpenAI-compatible endpoint |
| `rocm setup status` | Show first-time setup state |
| `rocm version` | Print the ROCm CLI version, release tag or branch, and commit hash, plus the ROCm SDK and GPU driver in use |
| `rocm completions <shell>` | Print a shell completion script (bash, zsh, fish, elvish, powershell) |

## Commands

Each section describes a command or command group, its options, and what it
does.

<!-- docs-site: commands-start -->
### Examine

```
rocm examine [--json] [--framework auto|pytorch|llama-cpp|skip]
```

Checks this computer's GPU, ROCm install, engines, and managed setup folders.
Run it first to see whether a system is ready and what `rocm install sdk` and
`rocm serve` will see.

- `--json` emits a machine-readable report for diagnosis tooling instead of the
  human-readable summary.
- `--framework` controls which ML framework the `--json` report probes for its
  ROCm build and compiled GPU architectures. It does not affect the
  human-readable report. The values are:
  - `auto` (the default) tries PyTorch, then falls back to llama.cpp.
  - `pytorch` or `llama-cpp` probes only that framework.
  - `skip` runs no framework probe, which is fastest and still enough to answer
    GPU and driver questions.

### Diagnose and fix

```
rocm diagnose [--symptom TEXT] [--top N] [--json] [--distro [NAME]]
              [--report [--send]]
rocm diagnose --model <model> [--json]
rocm fix [<fix-id>] [--yes] [--dry-run] [--device-index N]
```

`diagnose` matches this machine against a fixed catalog of known
misconfigurations in ROCm, PyTorch, and llama.cpp, and ranks what it finds. It
can only recognize failure modes that are in the catalog. No match means "not
recognized", not "nothing is wrong", and in that case it points you at where to
report the symptom. Each result prints an `id:` and an `apply with:` command.
The leading `#1`, `#2` are ranking positions for reading order only; `rocm fix`
takes the id, not the position.

`diagnose` accepts these options:

- `--symptom` takes raw error text to sharpen keyword scoring.
- `--top` caps how many matches are shown in the human-readable output
  (default 5). `--json` always emits the full, untruncated report.
- `--distro` diagnoses a WSL distribution from the Windows host instead of
  this machine. Nothing needs to be installed inside the distribution. Name it
  only when more than one is installed. Inspecting remotely this way skips
  checks that need to read the distribution's own environment
  (`HSA_OVERRIDE_GFX_VERSION`, `PATH`, and the framework and ROCm pairing), so
  run `rocm diagnose` inside the distribution for those.
- `--report` shows exactly what this machine would contribute to a problem
  report, and sends nothing. There is no transport yet, and there will be no
  automatic one: a report leaves a machine only by its owner's own action. See
  [What a report contains](#what-a-report-contains).
- `--send` requires `--report`. It additionally offers a prefilled mail that
  carries that report. See [How `--send` behaves](#how---send-behaves).

#### What a report contains

A report is deliberately narrow. It contains:

- A schema version.
- The matched catalog entry, and whether a fix was offered for it.
- The GPU architecture, and which compatibility matrix snapshot it was checked
  against.
- The OS family, distribution, and major version.
- The ROCm release, and the inference engine and its release.
- The CLI version.

It carries no host name, user name, file path, or error text. The ROCm release
and the inference engine's release are each cut back to a release, so a build
number that would narrow toward one machine never appears. The CLI version is
the exception, because it names the tool that wrote the report rather than
something read off the machine. The distribution is checked against a list of
known names rather than repeated from the machine.

No report is produced in two cases:

- The hardware is not on AMD's published compatibility matrix. The CLI says why.
- The machine is WSL2. This CLI does not inspect the GPU on WSL2 yet, so it
  cannot confirm that the hardware is on the compatibility matrix. It says that
  rather than claiming the architecture could not be read.

#### How `--send` behaves

`--send` still sends nothing. The mail opens already filled in with the content
`--report` just printed, addressed to `ROCmCLI@amd.com`, and it leaves the
machine only when you send it yourself. Requiring `--report` guarantees that the
content is shown before the mail is offered.

- A mail client opens only when you asked and the machine looks like a desktop
  you are at.
- Over SSH, with no display, or with `ROCM_NO_BROWSER` set, the address and the
  link are printed instead. The same happens on a machine with no mail client.
- `--send` cannot be combined with `--json`. `--json` exists for scripts, and a
  script cannot read a mail before sending it.
- The mail carries your address, which the report itself does not.

#### Applying fixes

`fix` applies a known fix by the `id:` that `diagnose` reported, not by the
ranking position, which isn't a stable name. Run it with no id to list the whole
catalog. Each fix carries a marker that says what happens on this machine:

- AUTO: this command carries out the change.
- NEEDS-ARG: this command carries out the change once you give it the argument
  the marker names.
- PRINT-ONLY: it prints the steps for you to run yourself. This is usually
  because the right command depends on a choice only you can make, and
  sometimes because it also needs `sudo` or a reboot.
- DIAGNOSE-ONLY: no reliable fix exists, so nothing is changed. No catalog entry
  carries this marker today. It is reserved for a future detect-but-cannot-repair
  failure.

`fix` accepts these options:

- `--dry-run` shows any fix's plan without changing anything.
- `--yes` skips the interactive confirmation after you have reviewed it.
- `--device-index` pins the discrete GPU index for `fix-9-igpu-dgpu`, which is
  marked NEEDS-ARG. Without it, that fix only prints the `rocminfo` (Linux) or
  `hipInfo.exe` (Windows) query needed to find the index, and makes no change.

### ROCm installation

```
rocm install sdk    [--channel release|nightly] [--format wheel|tarball]
                    [--version x.y.z | --build-date YYYY-MM-DD]
                    [--family gfx110X-all] [--prefix PATH] [--devel] [--dry-run]
                    [--approve-replacing-active-default] [--yes]

rocm install driver [--dkms] [--yes] [--dry-run] [--reconcile]

rocm update         [--apply] [--runtime KEY] [--activate] [--dry-run]
                    [--json] [--timeout-secs SECS] [--yes]
```

`install sdk` downloads TheRock ROCm wheels into a Python environment managed
by ROCm CLI. It can install ROCm 7.14 (the default), 10.0, or 10.1. To install
10.0 or 10.1, follow the steps under [ROCm 10 and newer](#rocm-10-and-newer).

To remove ROCm CLI and what it manages, see
[Logs and cleanup](#logs-and-cleanup).

#### Compiler toolchain

Pass `--devel` to also install the compiler and headers needed to build GPU
code. This roughly doubles the download.

`--devel` isn't an addition to an existing runtime. A runtime is identified by
the packages it was installed from, so running `rocm install sdk` and later
`rocm install sdk --devel` at the same version leaves you with **two**
side-by-side runtimes. The second is a fresh full install, not a toolchain
bolted onto the first, and it becomes active.

To tell the two apart:

- `rocm runtimes list` marks each runtime `toolchain=included` or
  `toolchain=excluded`.
- `rocm examine` reports the active runtime's toolchain as
  `active_runtime_toolchain`.
- `rocm storage remove-old-installs` counts the two kinds separately, so
  neither evicts the other.

To reclaim the space, uninstall the one you don't want with
`rocm runtimes uninstall <runtime-key>`.

#### Approval prompt

If no managed runtime is the active default, `install sdk` doesn't prompt.
Otherwise it asks first, because the new install becomes the active default.
The prompt applies to any install, including a `--family` or `--channel` you
haven't installed before, as it does for a same-family upgrade.

To approve without a prompt, for example in scripts or CI, where the prompt
would otherwise refuse:

- `--approve-replacing-active-default` approves the change of active default.
  The refusal message recommends it, and ROCm CLI's own non-interactive
  surfaces (chat, MCP, and the dashboard) pass it.
- `--yes` gives the same approval and also approves installing required system
  packages, such as OpenMPI for vLLM. That requires `sudo`, so use it only where
  something can answer a sudo password prompt. An unattended job can't, unless
  it has passwordless sudo configured.

#### Install location

In the default managed install root, the root and its manifest are keyed by
version. An upgrade or downgrade keeps the previous install on disk. Only a
same-version reinstall reuses the same root.

`--prefix` changes this. The folder you name is used as-is for every version, so
successive installs into one prefix replace each other in place. If the venv
already there no longer runs its own Python, it is removed outright and rebuilt.
The approval prompt doesn't cover this, because it asks only about changing the
active default runtime, not about what a named prefix loses.

#### ROCm 10 and newer

ROCm 10 and newer ship from a different source layout. You opt in by passing two
things together: pin the version with `--version`, and name the exact GPU arch,
using the raw `gfx` code rather than a family label:

```
rocm install sdk --version 10.1.0 --family gfx1200 --dry-run
```

A family label such as `--family gfx120X-all` is rejected for those versions
rather than resolved to a guess, because the ROCm 10 packages publish one
payload per exact arch and there is no bucket payload to fall back to. Run
`rocm examine` to see the arch this machine reports.

For ROCm 10, `install sdk` asks `uv` to resolve Torch, torchvision, and
torchaudio from their published dependency metadata, then validates that every
selected framework package carries the same ROCm build identifier before it
creates or changes a managed runtime.

Nothing about this happens on its own. Without a `--version` of 10 or newer,
`install sdk` resolves the same release and nightly sources as before. It doesn't
quietly retry against the ROCm 10 sources when a lookup finds nothing; it tells
you what it couldn't find instead.

#### Driver installation

`install driver` installs the AMD kernel driver on Linux, using DKMS or a native
package.

#### Updates

`update` checks for a newer ROCm package.

| Flag | Description |
| --- | --- |
| `--apply` | Installs the update. Never prompts and needs no approval flag, because selecting a runtime to update is the approval. Leaves the active default alone unless you add `--activate`. |
| `--dry-run` | Previews what `--apply` would do without changing anything. Doesn't require `--apply`. |
| `--runtime`, `--activate` | Require `--apply` or `--dry-run`. |
| `--json` | Prints the check result as a single line of JSON instead of text. Conflicts with `--apply` and `--dry-run`. |
| `--timeout-secs` | Bounds the network calls of the check. Requires `--json`. Conflicts with `--apply`. |
| `--yes` | Accepted for consistency with other mutating commands, but grants nothing on `update`. The approval line the update path prints never credits it. |

### Runtime management

Manage multiple side-by-side ROCm runtimes:

```
rocm runtimes list
rocm runtimes activate <runtime-key>
rocm runtimes rollback
rocm runtimes uninstall <runtime-key> [--yes] [--dry-run]
rocm runtimes import <manifest-file> [--replace]
rocm runtimes adopt --python <path> [--root <path>] [--runtime-id ID]
                    [--runtime-key KEY] [--channel LABEL] [--replace]
```

`list` shows each runtime's version, GPU family, mode, and whether it carries
the compiler toolchain (`toolchain=included|excluded`). Runtimes are told apart
by the packages they were installed from, not by version alone, so one version
can appear more than once: a `--devel` install and a plain one of the same
version are two separate runtimes, as are two installs that resolved different
GPU device payloads.

`uninstall` prompts for confirmation unless you pass `--yes`. Outside an
interactive terminal, `--yes` is required. `--dry-run` prints the plan and exits
without prompting or making changes.

`adopt` registers an existing TheRock-based Python environment as a managed
runtime. It does not work with standard ROCm package installs (for example,
`/opt/rocm`); use `rocm install sdk` instead.

### Disk space

Each ROCm runtime keeps its own multi-gigabyte folder, so installing or
updating a few times adds up. `rocm storage` shows where the space went and
frees the parts that are safe to remove:

```
rocm storage [report] [--json]
rocm storage remove-old-installs [--keep N] [--dry-run] [--yes]
rocm storage remove-downloads [--dry-run] [--yes]
```

`remove-old-installs` keeps the two most recent installs for each channel,
format, GPU family, and toolchain choice. It never touches the install in use,
the rollback target, or a folder ROCm CLI did not create. Anything it declines
to remove is listed with the reason, and `--dry-run` shows the whole plan
without changing anything.

- "Most recent" means most recently installed, not highest version, so after a
  deliberate downgrade the older version counts as the newer install.
- The count applies per channel, format, GPU family, and toolchain choice. A
  machine that has tried several channels keeps `--keep` installs for each of
  them.
- A runtime-only install never evicts a `--devel` one. The two are separate
  runtimes that serve different purposes, not newer and older versions of the
  same thing.

`remove-downloads` clears cached archives that ROCm CLI can download again. A
cache folder that is a link to somewhere else is left alone rather than
followed. The two archive rows in the report are tagged
`note: can be downloaded again; safe to remove`, so you can see which rows
`remove-downloads` acts on.

The report also lists these items, which `rocm storage` doesn't remove:

- `local server records`: one JSON record plus the engine's log for each
  `rocm serve --managed` launch, kept after the server stops. `rocm services
  list --all` lists them, and `rocm services prune` removes them.
- The `uv` package cache, the Hugging Face model cache, and downloaded models.
  Other tools share these, so ROCm CLI never removes them.

### Inference engines

```
rocm engines list
rocm engines install <engine> [--runtime-id KEY] [--python-version X.Y] [--reinstall]
rocm engines shell <engine>   [--runtime-id KEY | --env-id ID] [--shell PATH]
```

Supported engines: `lemonade`, `vllm`.

### Will a model run here?

Ask before downloading anything:

```
rocm diagnose --model <model> [--json]
```

This answers in seconds from the curated recipe and this machine's GPU. It
fetches no weights and makes no network call. The verdict is `ready`,
`degraded`, `blocked`, or `undetermined`. A `ready` answer also names the
engine `rocm serve` would use; a `blocked` one names curated models that would
run here instead.

`undetermined` is a real answer, not a failure. You get it when any of these is
true:

- The recipe catalog could not be read.
- This machine's GPU memory could not be measured.
- The model is not one of the curated recipes. [`rocm model`](#curated-models)
  lists those.

None of these says anything about whether the model fits, so none of them is
reported as though it did. `rocm serve` still accepts a model outside the
catalog, but `diagnose` cannot tell you in advance how it will go.

### Curated models

```
rocm model [--verbose]
```

`rocm model` (alias `rocm models`) lists the recommended local models and shows
which of them this machine can run. Pass `--verbose` for more detail about each
model.

### Model serving

Start a local OpenAI-compatible model server:

```
rocm serve <model> [--engine lemonade|vllm]
                   [--device gpu_required|gpu_preferred]
                   [--gpu auto|<index>]
                   [--runtime-id KEY | --env-id ID]
                   [--host HOST] [--port PORT]
                   [--verbose] [--foreground | --managed]
                   [--no-smoke-test]
                   [--allow-public-bind]
                   [--temperature TEMP] [--top-p PROB] [--max-tokens N]
```

`--temperature` (>= 0.0), `--top-p` (0.0-1.0), and `--max-tokens` (> 0) set
server-wide sampling defaults for the launched engine. They apply only to
`vllm` and `lemonade`, and other engines reject them. Each control is optional
and independent. Omit any of them to keep the engine's own default.

- For vLLM, the controls are folded into a single `--override-generation-config`
  JSON object, and `--max-tokens` maps to vLLM's `max_new_tokens`.
- For Lemonade, they pass straight through as llama.cpp's `--temperature`,
  `--top-p`, and `--n-predict` flags.

`rocm serve` reuses an already-running service for the same engine and model
only if its sampling controls and other recipe settings match the ones you
request this time. Otherwise it errors out instead of silently serving with
different settings. If you previously started a service with `--temperature`
(or another sampling flag) and now run `rocm serve` for the same model without
flags, or with different ones, stop the existing service first with
`rocm services stop`, or match the original flags.

By default the server runs in the background under ROCm CLI's supervision.
`--managed` is the explicit form of this default. `rocm serve` prints a
deployment summary: a progress indicator while the server starts, then a table
with the status, the full inference endpoint, the API-qualified model name, and
a quick smoke test (time to first token and approximate tokens per second).
Control returns to your shell with the server still running. Manage it later
with [`rocm services`](#managing-background-servers).
`--no-smoke-test` skips the post-startup inference probe.

`--verbose` (or `--foreground`) instead attaches to the server in the current
terminal and streams every engine log line. Use it to debug a startup problem.
The server still runs as a managed background process, so you can leave the
stream without stopping it:

- Press **Ctrl-D** to detach. The log stream stops, your shell comes back, and
  the server keeps running.
- Press **Ctrl-C** to stop the server instead.

Which model form to pass depends on the engine your GPU selects. The Lemonade
engine (Ryzen AI or Radeon) serves llama.cpp **GGUF** models. Pass a GGUF repo
with an explicit quantization variant, for example
`rocm serve unsloth/Qwen3-0.6B-GGUF:Q4_0`. The vLLM engine (Instinct) serves
**safetensors** repos, such as `rocm serve Qwen/Qwen2.5-1.5B-Instruct`. A
safetensors-only id has no GGUF build, so serving it through Lemonade fails
rather than silently substituting a different model.

Some models (such as Llama) are gated and require Hugging Face authentication.
Log in with the Hugging Face command-line tool, or set `HF_TOKEN` in your
environment, before serving gated models.

`--gpu` selects which AMD GPU the server runs on. `auto` (the default) probes
per-GPU VRAM (through `amd-smi`, or the amdgpu DRM sysfs counters when
`amd-smi` is not installed). It picks the lowest-numbered GPU that is idle and
not already used by another ROCm CLI server (managed or foreground), falling
back to the GPU with the most free memory. Pass a single index (`--gpu 1`) to
pin a specific device.

The selected GPU is exposed to the engine through `HIP_VISIBLE_DEVICES`. Serving
one model across multiple GPUs is not supported. Selection uses the `amd-smi`
ordinal but is applied through `HIP_VISIBLE_DEVICES`, so ROCm CLI warns when
`ROCR_VISIBLE_DEVICES` is set, because the two orderings can diverge.

#### Managing background servers

Manage background servers started with `--managed`:

```
rocm services list [--all] [--json]
rocm services logs <service-id>
rocm services stop <service-id> [--yes]
rocm services restart <service-id> [--yes]
rocm services remove <service-id> --yes
rocm services prune [--older-than-hours <n> | --any-age] [--dry-run] [--yes]
```

`remove` deletes one record that is no longer running, together with its log,
its engine state file, and its endpoint key file. A running server is refused,
so stop it first.

`prune` does the same in bulk. It always leaves running servers alone, and it
also clears leftover files whose record is already gone. Removal destroys both
the log and the `restart` option for the records it takes, so `prune` only
considers records untouched for 24 hours.

- Age is measured from when the record file was last written, so a stop, a
  restart, or a status correction all count as touching it.
- Pass `--older-than-hours <n>` for a different threshold, or `--any-age` to
  take every record that is not running, however recent. `prune` names
  `--any-age` in its own summary when it reports how many records it kept for
  being too recent. The two flags cannot be combined.

A file whose record has not been written yet belongs to a server that is still
starting, not to something left behind. So `prune` waits for any managed launch
already under way to finish publishing its record before it looks at the
directory.

- The wait lasts as long as the launch does and has no timeout. It is usually
  imperceptible but is not bounded.
- On an interactive terminal, `prune` prints
  `Waiting for a launch already under way…` while it waits, including under
  `--dry-run`. That notice goes to stderr and is suppressed when stderr is not a
  terminal, so a piped or scripted prune waits silently.
- The same lock runs in the other direction: a `rocm serve` started while a
  `prune` is scanning waits for the prune.

`--json` prints the service records verbatim, for scripting and for the remote
orchestration below.

### Remote machines (preview)

Run a model on a different GPU machine and reach it from your own. Both machines
join a [Tailscale](https://tailscale.com) network (a "tailnet"). The GPU machine
serves the model on its own loopback address and publishes that port onto the
network. The endpoint keeps working after the command exits and answers from any
of your machines, not only the one that started it.

Before you start, install Tailscale on both machines and run `tailscale up` on
each. Communication with the GPU machine uses your existing `ssh` setup.

```console
rocm remote targets [--tag <tag>]
rocm remote doctor <machine> [--symptom <text>]
rocm remote serve <machine> <model> [--engine <engine>] [--gpu <index>]
                                    [--tailnet-port <port>] [--install-rocm]
rocm remote status [<session>]
rocm remote attach <session>
rocm remote stop <session> [--force]
```

- `targets` lists machines on your network. It does not check whether they can
  actually serve, which is what `doctor` is for. `--tag` filters the list by
  Tailscale tag.
- `serve` prepares the machine, starts the model, publishes the endpoint, and
  prints the address together with an API key. `--engine` and `--gpu` work as
  they do for [`rocm serve`](#model-serving).
  - `--tailnet-port` sets the port that the endpoint is published on. The
    default is 8000.
  - `--install-rocm` also installs ROCm on the GPU machine. It is off by
    default.
  - ROCm CLI is installed on the GPU machine if it is missing.
- `status` reports the model and the endpoint separately, because either can fail
  alone. A healthy model with no endpoint needs `attach`, not a restart.
- `stop` withdraws the endpoint and stops the model. It keeps the session listed
  if it cannot confirm both. Use `--force` to forget a session whose machine is
  gone.

**The endpoint is reachable by every machine on your network that your network's
access rules allow**, not just yours. The API key is what stops anyone else from
using it, so `rocm remote` always sets one. Local serving doesn't need a key,
because only your own machine can reach it.

Set `ROCM_REMOTE_SSH_CONFIG` to point at an `ssh` configuration file other than
the default.

### Dashboard

```
rocm dash [--demo] [--replay <file>]
```

The dashboard is a full-screen terminal user interface (TUI) with Home, ROCm,
Serving, Observe, and Chat tabs. It shows GPU utilization graphs, active serving
instances, benchmark results, and guided actions, and its Chat tab uses any
configured provider.

<!-- docs-site: interactive-link-start -->
See [Interactive interfaces](#interactive-interfaces) for the tab breakdown.
<!-- docs-site: interactive-link-end -->

- `--demo` runs a deterministic synthetic session with no GPU or daemon. It
  works on all platforms.
- `--replay <file>` replays a recorded NDJSON session.
- Live mode requires Unix domain sockets, so it runs on Linux and WSL2 only.

### Bench

```
rocm bench load --endpoint URL [--model NAME] [--concurrency N,N,...]
                [--isl N] [--osl N] [--requests N] [--out FILE] [--auto-ramp]
```

Saturates a local OpenAI-compatible endpoint and reports rough client-side
throughput. This is a local smoke test, **not** an official ROCm or AMD
benchmark. `load` measures raw serving throughput with synthetic single-shot
requests (the vLLM `benchmark_serving` shape). It does not reproduce
agent-shaped traffic (multi-turn, long-context, with tool calls), so its results
aren't comparable to agent benchmark suites that measure answer quality.

- `--endpoint` is the OpenAI-compatible URL shown by `rocm services list`. A
  plain host address without `/v1` also works. Only `http://` is accepted.
  `https://` endpoints are rejected outright, because the load generator has no
  TLS backend compiled in.
- `--concurrency` sweeps a comma-separated list of levels (default `1,8,32,64`,
  each 1-128). `--auto-ramp` ignores `--concurrency` and ramps
  `1,2,4,8,16,32,64,128` automatically, stopping early once generation
  throughput plateaus or the request queue backs up.
- `--isl` and `--osl` set the input and output sequence length (default 1024
  each) and accept 1-32768. `--requests` (default 128) accepts 1-10000.
- Results are written to `--out` (default `<data-dir>/bench/results.csv`, where
  `<data-dir>` is `~/.rocm` unless overridden). The default is intended to match
  the path the daemon tails to feed the dashboard's **Observe** tab.

The CLI's default output path and the daemon's tailed path are computed
independently. If you customized either the CLI's data directory or the daemon's
`bench_results_dir` setting (the folder where the daemon looks for benchmark
results), confirm that they still point at the same file.

### Chat

```
rocm chat [--provider local|openai|anthropic] [--model NAME] [--prompt TEXT] [--tools]
          [--temperature TEMP] [--top-p PROB] [--max-tokens N]
```

Chat with an AI provider from the terminal. The command reads from stdin when
you omit `--prompt`.

- `--provider` selects `local` (a model served on this machine), `openai`, or
  `anthropic`. A cloud provider needs setup first: run
  `rocm config enable-provider <provider>` and
  `rocm config set-provider-key <provider>`. See [Configuration](#configuration).
- `--tools` lets an OpenAI-compatible provider request ROCm tool calls.
- `--temperature`, `--top-p`, and `--max-tokens` are optional sampling controls
  forwarded to the request. Each is independent, so omit any of them to use the
  provider's default.

### ComfyUI

Install and manage ComfyUI for image generation (alias: `rocm comfy`):

```
rocm comfyui install    [--runtime-id KEY] [--reinstall] [--dry-run] [--yes]
rocm comfyui start      [--host HOST] [--port PORT] [--no-open-browser] [--yes]
rocm comfyui stop       [--yes]
rocm comfyui status
rocm comfyui logs       [--lines N]
rocm comfyui models-path
```

The `install`, `start`, and `stop` commands never prompt for confirmation. They
accept `--yes` for consistency with other mutating commands, but it has no
effect today.

### Automations

```
rocm automations list
rocm automations enable <watcher-id>  [--mode observe|propose|contained]
rocm automations disable <watcher-id>
```

Automations are optional background checks, called watchers, that can propose or
apply changes automatically. `list` shows the available watchers and their IDs.
`enable` turns one on, and `--mode` sets how far it can act:

- `observe` reports what it finds and changes nothing.
- `propose` suggests a change for you to approve.
- `contained` applies a change on its own, within the limits of that watcher.

Each watcher has a default mode:

| Watcher ID | Default mode |
| --- | --- |
| `therock-update` | `observe` (checks every 6 hours) |
| `server-recover` | `contained` |
| `gpu-metrics` | `observe` |
| `cache-warm` | `propose` |
| `driver-upgrade` | `propose` |
| `gpu-thermal-protect` | `propose` |

### Configuration

Show or change ROCm CLI's saved settings: the default engine and runtime, which
runtime each engine prefers, local GPU telemetry opt-in, and the provider used
for chat, automations, and ambiguous natural-language plans. You also enable
providers and store their API keys here.

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

### Setup

```
rocm setup status
rocm setup reset
```

Manage the state of the first-time setup flow.

- `status` shows whether first-time setup has completed.
- `reset` clears the recorded completed or dismissed state. Resetting doesn't
  start the setup flow again by itself. To open it, run `rocm dash`, switch to
  the **Observe** tab, and press `n`. ROCm installs, API keys, and provider
  settings are left untouched.

### Logs and cleanup

```
rocm logs [--service <service-id>] [--search TERM ...] [QUERY ...]

rocm uninstall [--yes] [--dry-run]
               [--keep-binaries] [--keep-config] [--keep-data] [--keep-cache]
               [--force-dev-binaries]
```

`rocm logs` shows ROCm CLI logs. `--service` limits the output to one service,
and `--search` or a plain query filters it by search words.

`rocm uninstall` removes ROCm CLI and the data it manages. It warns you about
managed services that are still running, and about remote sessions: run
`rocm remote stop <session>` first. Use `--dry-run` to preview what it removes.
`--yes` skips the confirmation. Each `--keep-*` flag leaves one category in
place:

- `--keep-binaries` keeps the installed `rocm` and `rocmd` binaries.
- `--keep-config` keeps your saved settings.
- `--keep-data` keeps downloaded data, including the `uv` package cache.
- `--keep-cache` keeps cached downloads.
- `--force-dev-binaries` also removes binaries that were built from source,
  which `uninstall` otherwise leaves alone.

### Shell completions

`rocm completions <shell>` prints a completion script to stdout. The supported
shells are `bash`, `zsh`, `fish`, `elvish`, and `powershell`. Install the script
for your shell:

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

- [Architecture and module map](https://github.com/ROCm/rocm-cli/blob/main/docs/architecture.md)
- [Testing and verification](https://github.com/ROCm/rocm-cli/blob/main/docs/testing.md)
- [Developer manual QA](https://github.com/ROCm/rocm-cli/blob/main/docs/manual-testing.md)
- [Engine plugin policy](https://github.com/ROCm/rocm-cli/blob/main/docs/engine-plugins.md)
- [vLLM adapter](https://github.com/ROCm/rocm-cli/blob/main/docs/vllm.md)
