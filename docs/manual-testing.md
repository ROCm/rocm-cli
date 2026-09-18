<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Developer Manual QA

Use this checklist to verify rocm-cli behavior as a developer or release
tester. On Windows, run these commands from PowerShell. On Linux or WSL, use
the same commands in a shell.

For normal users, keep the flow simple: install rocm-cli, run `rocm`, choose a
ROCm folder, approve setup, and then use the main TUI.

When validating release behavior, build and use the native per-OS binary for
the target you are testing:

```bash
cargo build --workspace --release
```

This writes `target/release/rocm.exe` on Windows and `target/release/rocm` on
Linux/WSL. Run the binary directly on each platform. On WSL/Linux the examine
output must report `os: linux` and `wsl: true`.

Do not set `ROCM_CLI_THEROCK_FAMILY` during normal setup tests. rocm-cli should
detect the right TheRock package family or tell the user what is missing.

When testing this branch and you want all generated state inside the workspace,
set these environment variables before running commands:

```powershell
$repo = (Get-Location).Path
$env:ROCM_CLI_CONFIG_DIR = "$repo\.rocm-work\config"
$env:ROCM_CLI_DATA_DIR = "$repo\.rocm-work\data"
$env:ROCM_CLI_CACHE_DIR = "$repo\.rocm-work\cache"
```

Do not create those folders by hand. rocm-cli should create what it needs.
If you choose an explicit ROCm folder during setup or pass `--prefix`, the
TheRock pip cache should live inside that folder at
`<install-folder>\pip-cache`; rocm-cli should pass that location to pip and let
pip create it when downloads start. If you omit `--prefix`, rocm-cli should
choose a managed runtime folder and still place the pip cache inside that
runtime folder at `<managed-runtime-folder>\pip-cache`.

The `uv` package cache is separate from that pip cache and does **not** follow
`--prefix`. It lives at `<data-dir>\uv-cache` so it is reachable from the managed
environments without crossing a mount point and `uv` can hardlink into them. With
`--prefix` pointing somewhere that is a separate mount from `ROCM_CLI_DATA_DIR`,
`uv` falls back to copying packages; set `ROCM_CLI_UV_CACHE_DIR` to a folder on
the prefix mount to restore hardlinking. Note it is the mount, not the
filesystem: a bind mount is enough to trigger the fallback even when both paths
resolve to one underlying filesystem. Making `--prefix` do this automatically is
tracked separately.

## 1. First-Time Setup

Start rocm-cli:

```powershell
rocm
```

Expected result:

- The launcher opens; choosing "Set up this system" there opens the setup
  screen. It does not open automatically before the main TUI, and the user
  does not need to type `/setup`.
- The setup shows a recommended ROCm folder.
- The setup shows `downloads stay inside: <ROCm folder>\pip-cache` so the user
  can see that pip downloads stay inside the chosen ROCm folder.
- The install-folder row opens an interactive folder picker. Arrow keys and the
  mouse can choose folders; Enter opens or selects; Esc returns without losing
  the current setup screen.
- The setup asks for approval before installing anything.
- The setup shows what is being installed and shows progress.
- Install logs show only in the foreground progress card, with PageUp/PageDown
  and mouse-wheel scrolling.
- The setup installs ROCm into the folder chosen by the user.
- After a successful install, setup shows a simple success card and then
  continues to the main TUI.
- The setup saves choices so the next `rocm` run opens the main TUI.

If the first run opens the main TUI and expects the user to type `/setup`, this
test fails.

Quiet UI rule:

- First-view setup, engine, service, assistant, and ComfyUI screens should stay
  terse.
- Live pip/install logs belong only in the foreground progress card.
- Finished screens should not show raw command dumps, repeated `Output:`
  prefixes, hidden log paths, or stale background panes unless the user opens a
  log/details card.
- One Esc closes the focused card or asks to quit from the main menu.

After setup, check the machine state:

```powershell
rocm examine
rocm runtimes list
```

Expected result:

- `rocm examine` shows a managed runtime.
- `rocm runtimes list` shows the runtime key for the installed TheRock venv.
- The active runtime is ready, or the output gives one clear next command.

## 2. TheRock SDK Command-Line Install

This tests the command-line install path without using the TUI:

```powershell
rocm install sdk --channel release --format wheel --prefix .\.rocm-work\data\envs\default
rocm runtimes list
rocm runtimes activate <runtime_key>
rocm examine
```

Replace `<runtime_key>` with the exact key printed by `rocm runtimes list`.
Omit `--prefix` if you want rocm-cli to choose its standard managed folder.

Section 1 has already made a managed runtime the active default, so the
`install sdk` above will **ask for confirmation before it installs**: the new
install takes over as the active default. That is the expected behaviour, not a
regression. Answer the prompt to continue. The gate is not scoped to the family
or channel, so it asks even when this install targets a family this machine has
never held. To take the same step without a prompt — and this is required in a
non-interactive shell, where the command refuses instead of asking — re-run it
with `--approve-replacing-active-default`:

```powershell
rocm install sdk --channel release --format wheel --prefix .\.rocm-work\data\envs\default --approve-replacing-active-default
```

On Linux and WSL, use `--yes` only if you also want to approve installing
required system packages with `sudo`, which needs a terminal to answer a
password prompt. On native Windows that second consent buys nothing — the
system-package step returns early there — so `--approve-replacing-active-default`
is the whole approval this gate needs either way.

Expected result:

- On a machine with an active default runtime (the state section 1 leaves
  behind), the install prompts first and names what would be displaced;
  declining leaves the existing runtime untouched.
- In a non-interactive shell with neither approval flag, the install refuses
  rather than silently displacing the active default, and the error names
  `--approve-replacing-active-default` as the flag to add.
- With `--approve-replacing-active-default`, the install proceeds without
  asking and prints a line crediting that flag by name — not `--yes`.
- `rocm install sdk ... --dry-run` never prompts or refuses, whatever the
  active default is: the preview stops before the gate.
- rocm-cli creates or reuses a rocm-cli managed Python venv.
- pip installs pinned `rocm`, `torch`, and `torchvision` requirements with
  exactly one `device-<detected-gfx-target>` extra (`rocm` also requests
  `libraries,devel`), alongside pinned `torchaudio` from the TheRock index. On a host
  with no detectable AMD GPU the preview reports `device_target: undetermined`
  and a real install refuses rather than pulling every published device payload.
- rocm-cli chooses the newest exact ROCm build suffix common to the SDK package
  and the PyTorch stack for the current Python/platform wheel tags, then pins
  all four packages in one pip transaction.
- The install does not ask for an external Python venv.
- Runtime validation uses TheRock's runtime/devel package roots and
  `rocm_sdk.find_libraries`; `rocm-sdk path --root` is expected after the
  pinned `rocm[libraries,devel,device-…]` install succeeds.
- `rocm examine` reports the active runtime as ready.

Developer-only deterministic override:

```powershell
python scripts\therock_sdk_install_test.py --dry-run --family gfx120X-all
```

Use `--family` only when a test needs a fixed package family. Do not use it for
normal user setup.

## 3. Lemonade GPU Verification

Lemonade is the default local assistant/server engine. Serve a small assistant
model with the managed runtime:

```powershell
rocm serve qwen --engine lemonade --device gpu_required --foreground --port 11435
```

Expected result:

- The engine uses ROCm GPU execution.
- The run does not fall back to CPU or Vulkan.
- If the GPU cannot be used, the command fails with a clear error.
- The OpenAI-compatible endpoint answers a simple chat request.

While the log stream is attached, verify detach and stop behave differently:

- Press `Ctrl+D`. The stream ends and the shell prompt returns, printing a
  "detached — server still running" note with the service id. Confirm the server
  is still up: `rocm services` lists it and the endpoint still answers a chat
  request. Stop it afterward with `rocm services stop <service-id> --yes`.
- Re-run the serve command and press `Ctrl+C` instead. The server shuts down and
  `rocm services` no longer lists it as running.

## 4. Local Server Records

After a managed or foreground serve attempt, inspect local server records:

```powershell
rocm services
rocm services list --all
rocm services logs <service-id>
```

Expected result:

- `rocm services` lists only living local servers, but its `Status:` line
  counts the whole registry, so a failed or stopped record is counted as
  `N not running` even though no row for it is shown.
- When such records exist, `rocm services` ends with a `Past attempts:` block
  giving their count and two runnable commands: `See them: rocm services list
  --all` and `Read the newest: rocm services logs <service-id>`, with the id of
  the newest record that is no longer running. Paste that second line as-is and
  confirm it opens the logs for that attempt.
- On a machine that has never served, `rocm services` prints no `Past attempts:`
  block at all.
- `rocm services list --all` shows saved history, including the failed or
  stopped attempts the default view hides.
- The logs command shows the exact service failure or startup output.
- `rocm storage report` lists these records under `local server records` with
  the folder that holds them; no `rocm storage` command deletes them.
- Stop and restart require explicit approval:

```powershell
rocm services stop <service-id> --yes
rocm services restart <service-id> --yes
```

Then delete a record you no longer want. Removal is destructive and cannot be
undone, so read the log first:

```powershell
rocm services logs <service-id>
rocm services remove <service-id> --yes
rocm services prune --dry-run
rocm services prune --dry-run --any-age
rocm services prune --yes --any-age
rocm services prune --any-age --older-than-hours 0
```

Expected result:

- `rocm services remove` on a *running* server fails and tells you to run
  `rocm services stop <service-id> --yes` first; nothing is deleted.
- Without `--yes` both commands fail and print the command to repeat.
- After a successful removal, `rocm services list --all` no longer lists the
  record, `rocm services logs <service-id>` fails, and all of
  `<data>/services/<service-id>.json`, `<data>/services/<service-id>.log`,
  `<data>/engines/<engine>/state/<service-id>.json` and any
  `<data>/services/<service-id>.endpoint-key` are gone.
- `<data>/services/launch.lock` is untouched — it is shared by every launch.
- `rocm services prune --dry-run` prints the plan and removes nothing. (It still
  refreshes each record against the real processes, so a record whose server has
  since died can have its status corrected on disk; nothing is deleted.)
- `rocm services prune` leaves running servers alone and says how many it
  skipped, and also removes leftover engine state files whose record is gone.
- With no age argument, a record written in the last 24 hours is left alone, and
  the output both counts it ("too recent, kept: 1") and names the flag to
  include it: `rocm services prune --any-age --yes`.
- `--any-age` then removes that same record. `--older-than-hours 0` is the
  equivalent long form.
- `--any-age` together with `--older-than-hours` is rejected by the argument
  parser rather than one of them silently winning.
- A record whose server died long ago but has not been listed since is still
  removed by a plain `rocm services prune --yes`: age is read from the record
  file as it was before the command refreshed it, not after.

## 5. ComfyUI Verification

ComfyUI is managed as an app surface. It should start a local web server and
show the URL to open:

```powershell
rocm comfyui install --yes
rocm comfyui start --yes --port 18188
rocm comfyui status
rocm comfyui stop --yes
```

Expected result:

- ComfyUI installs without replacing the managed ROCm GPU package stack.
- The server starts on `http://127.0.0.1:18188`.
- Status shows the local URL and current state.
- Stop shuts down the saved process.

For the stricter developer GPU test:

```powershell
python scripts\comfyui_therock_gpu_test.py
```

This test may download a small checkpoint and submit a cat image workflow
through the ComfyUI HTTP API.

## 6. Optional Cloud Provider Key

Local ROCm use does not need a cloud provider key. If you want to test OpenAI or
Anthropic provider setup, save the key through stdin so it does not land in
shell history:

```powershell
$env:OPENAI_API_KEY | rocm config set-provider-key openai
rocm config enable-provider openai
rocm config show
```

Expected result:

- `rocm config show` says the key is saved in the OS secure store, or that the
  current session is using `OPENAI_API_KEY`.
- The key value itself is never printed.
- `config.json`, logs, and examine output do not contain the key.

To remove the saved key:

```powershell
rocm config clear-provider-key openai
```

## 7. Optional Provider-Assisted Planning

Most users should leave this off. To test ambiguity resolution with an already
running local provider service:

```powershell
rocm config set-planner-provider local
rocm "start a local model"
```

Expected result:

- The rendered plan says `planner: hybrid-parser-v1 + provider:local` only if
  the local provider returned a valid structured `rocm` tool call.
- The plan still asks for review before mutating actions.
- `rocm --yes "start a local model"` refuses to auto-run a provider-assisted
  plan; run the displayed structured command directly after reviewing it.

To turn provider-assisted planning back off:

```powershell
rocm config clear-planner-provider
```
