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
- On the "Install ROCm SDK (pip)" step's Configure screen, the Folder row reads
  `(default managed folder · Tab to browse)` until a folder is chosen. `Tab`
  opens the folder picker; arrow keys choose, Enter opens or selects, Esc
  returns without losing the Configure screen or an already-chosen folder.
  Leaving the row unset installs to the default managed folder.
- On that same screen `←`/`→` toggle Release/Nightly, and `Tab` opens the
  folder picker rather than toggling the channel — the same Tab-to-browse
  binding the Install and Serve forms use.
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

Optional: an unreadable settings file. Copy `config.json` from the config
directory aside, cut the original off part-way through, and run `rocm version`
without `ROCM_CLI_DATA_DIR` set. Expected result:

- The command fails and names `config.json`, saying it cannot be read, and
  explains the three ways out: repair it; set `ROCM_CLI_DATA_DIR` (it names the
  folder if the damaged file still shows it), after which commands that change
  settings still stop and others either stop or run on default settings; or
  move it aside, which resets every setting.
- `config.json` is byte-for-byte what you left; nothing was saved over it.
- Restoring the copy makes `rocm runtimes list` show the same runtimes as before.

## 2. TheRock SDK Command-Line Install

This tests the command-line install path without using the TUI:

```powershell
rocm install sdk --channel release --format wheel --prefix .\.rocm-work\data\envs\default
rocm runtimes list
rocm runtimes activate <runtime_key>
rocm examine
```

Replace `<runtime_key>` with the exact key printed by `rocm runtimes list`. Keys
are matched exactly first; a selector that matches two installed keys only by
letter case is refused with both keys listed, so copy the key as printed.
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
  `libraries`, and `devel` only when `--devel` is passed), alongside pinned
  `torchaudio` from the TheRock index. On a host with no detectable AMD GPU the
  preview reports `device_target: undetermined` and a real install refuses
  rather than pulling every published device payload.
- rocm-cli chooses the newest exact ROCm build suffix common to the SDK package
  and the PyTorch stack for the current Python/platform wheel tags, then pins
  all four packages in one pip transaction.
- The install does not ask for an external Python venv.
- Runtime validation uses TheRock's runtime package roots and
  `rocm_sdk.find_libraries`; `rocm-sdk path --root` is expected after the
  pinned `rocm[libraries,device-…]` install succeeds. The compiler toolchain is
  not required for validation to pass; `--devel` adds `devel` to those extras.
- `rocm examine` reports the active runtime as ready.
- `rocm runtimes activate <runtime_key>` ends its report with a count of the
  running local servers left on the previous runtime. Nothing is serving yet at
  this point, so it must read `services_on_previous_runtime: 0`. A fixed note
  about running services in place of that count means this test fails.
- `rocm runtimes activate <runtime_key> --restart-services` without `--yes`
  refuses, names `--restart-services --yes` as the way to repeat it, and leaves
  the active runtime alone. Section 4 exercises the same report and the same
  flags with a server actually running.

Developer-only deterministic override:

```powershell
python scripts\therock_sdk_install_test.py --dry-run --family gfx120X-all
```

Use `--family` only when a test needs a fixed package family. Do not use it for
normal user setup. The script checks the default install; add `--devel` to check
the opt-in compiler toolchain path instead.

### 2a. A GPU target with no ROCm build

Any machine can run this one; it needs no particular GPU. `gfx90c` is the Vega
iGPU in Renoir, Lucienne, Cezanne and Barcelo APUs, and ROCm publishes no build
for it:

```powershell
rocm install sdk --family gfx90c --dry-run
```

Expected result:

- The command fails. It says no ROCm SDK build exists for `gfx90c`, names
  `--family` as where the target came from, and states that nothing was
  installed and no other family was substituted.
- It offers **no** package family to try instead. A message listing
  `gfx110X-all`, `gfx90X-dcgpu` or any other family here is a regression: it
  invites installing a build for different silicon. In particular
  `gfx90X-dcgpu` appearing means the data-center catch-all has been reopened.
- It does not fall back to this machine's own GPU. On a host with a supported
  card, a run that succeeds and installs for that card is a regression — the
  named target must not be silently replaced.

On a Ryzen 4000 or 5000 machine whose Vega iGPU is the only GPU, four further
checks, which cannot be made anywhere else:

- `rocm examine` shows `detected_gfx_target: gfx90c` with an empty
  `compatible_therock_family`, plus a `rocm_support` line saying the GPU is
  detected but not supported by ROCm and that the Vulkan backend can still be
  used. A `<unknown>` target, or a missing `rocm_support` line, means detection
  regressed.
- A plain `rocm install sdk`, with no `--family`, gives the same refusal with
  the wording for a *detected* GPU rather than a requested one.
- `rocm diagnose` reports `info-20-gpu-no-rocm-support`, naming `gfx90c` and
  saying Vulkan-based engines still work, with a `DIAGNOSE-ONLY` /
  `no reliable fix` flag and an `apply with: rocm fix info-20-gpu-no-rocm-support`
  line. **`fix-13-hip-sdk-missing` ("HIP SDK not installed") must not be the
  top-ranked cause, and on a host with no HIP-built framework must not appear
  at all** — telling this user to download the HIP SDK is the regression
  ROCMAI-546 removed, and the download would not have helped. On **Windows**
  this is the only place the check can be made: `check_13_hip_sdk_missing`
  returns early on every non-Windows host, so the Linux e2e lane (`diagnose-40`)
  cannot distinguish a working gate from a deleted one.
- `rocm fix info-20-gpu-no-rocm-support --yes` exits 0, prints "This problem has
  no reliable fix, so this command changes nothing", and does **not** print
  "Copy the commands above". Anything installed, downloaded or written by that
  command is a regression.

On Windows, a hybrid laptop with both the Vega iGPU and a discrete card must
still report the discrete card from `rocm examine`; the iGPU is listed first by
WMI and must not hide it. With a HIP SDK installed, check the per-GPU targets
rather than only the list: each card must carry its own `gfx_target`, and a
discrete card reported as `gfx1035` with `is_apu: true` means the `hipInfo`
fallback paired an architecture line with the wrong GPU. On that machine
`rocm diagnose` must **not** report `info-20-gpu-no-rocm-support` at all — the
discrete card does have ROCm support, so the whole-machine claim is false.

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

If the llama.cpp server exits at start with an access violation (`0xC0000005`
in `MSVCP140.dll`), the Visual C++ runtime on the machine is probably older than
14.40. Run `rocm diagnose` and expect `fix-21-msvc-redist-outdated`. `rocm fix
fix-21-msvc-redist-outdated` prints the update plan and changes nothing. After
the update, restart Lemonade and repeat the serve command above.

While the log stream is attached, verify detach and stop behave differently:

- Press `Ctrl+D`. The stream ends and the shell prompt returns, printing a
  "detached - server still running" note with the service id. Confirm the server
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
  giving their count and three runnable commands: `See them: rocm services list
  --all`, `Read the newest: rocm services logs <service-id>` (with the id of the
  newest record that is no longer running) and `Reclaim the space: rocm services
  prune`. Paste the `Read the newest:` line as-is and confirm it opens the logs
  for that attempt.
- On a machine that has never served, `rocm services` prints no `Past attempts:`
  block at all.
- `rocm services list --all` shows saved history, including the failed or
  stopped attempts the default view hides.
- The logs command shows the exact service failure or startup output.
- `rocm storage report` lists these records under `local server records` with
  the folder that holds them; no `rocm storage` command deletes them, and the
  row's note points at `rocm services prune` to reclaim the space.
- Stop and restart require explicit approval:

```powershell
rocm services stop <service-id> --yes
rocm services restart <service-id> --yes
```

`restart` starts the server again only once its stop has confirmed every
recorded process gone; otherwise it reports that and starts nothing.

The rest of this section needs a **vLLM** server running, on Linux or WSL, and
two runtimes to move it between. Nothing earlier in this document sets either
up: section 3 serves Lemonade and then stops it on both of its branches, the
block just above stops whatever was left, and section 2 leaves one runtime.

**Get two distinct runtime keys.** A wheel runtime key is built from the
channel, the version and a fingerprint of the exact packages installed, in the
shape `release-wheel-multi-arch-<version>-<hash>`; `--prefix` is not part of it.
A second `rocm install sdk` with the same options therefore resolves to the same
key and replaces the first runtime instead of adding one. Change something that
is part of the package set. `--devel` does that: per its help text, passing it
over a plain install of the same version creates a second side-by-side runtime
and activates it. Leave `--prefix` off so the new runtime gets its own managed
folder rather than reusing section 2's:

```powershell
rocm install sdk --channel release --format wheel --devel --approve-replacing-active-default
rocm runtimes list
```

If section 2's runtime was itself installed with `--devel`, drop `--devel` here
and pass `--version <other-version>` instead, using a version the release
channel publishes. `rocm runtimes list` must now show two keys. Call the one
section 2 left `<runtime_key_a>` and the new one `<runtime_key_b>`, copied as
printed.

**Install vLLM into each runtime.** vLLM is installed per runtime, into that
runtime's own `engines` folder, so a runtime without it can neither serve nor
receive a moved server; skipping this makes the steps below fail early:

```bash
rocm engines install vllm --runtime-id <runtime_key_a> --yes
rocm engines install vllm --runtime-id <runtime_key_b> --yes
```

`--yes` approves required system-package installs such as OpenMPI, which can
call `sudo`. If the `install sdk` report above already printed an
`engine auto-install` block for vLLM, that runtime has it; `--reinstall` is the
flag that forces a reinstall when the engine already exists.

**Start from runtime A.** Make it the active runtime and start a server on it.
`--runtime-id` names the runtime explicitly, so the server's recorded runtime
does not depend on any saved engine preference:

```bash
rocm runtimes activate <runtime_key_a>
rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed --runtime-id <runtime_key_a>
rocm services
```

`rocm services` must list it as running before you go on; note its service id.
Re-running section 3's Lemonade command instead does not work here: Lemonade
brings its own runtime and is deliberately never counted by the reconciler, so
every step below would report `services_on_previous_runtime: 0`, print no
`note:` line to read the next command from, and verify nothing.

Switching runtimes does not move a server that is already running. Activate
runtime B without the restart flag and read what the report says about the
server you just started:

```powershell
rocm services
rocm runtimes activate <runtime_key_b>
```

The server keeps serving on the runtime it started with, so the report must
read `services_on_previous_runtime: 1` and name it with
`recorded_runtime=<runtime_key_a>`. Move it with the command the report's own
note gives you — which is a second `activate` of the key that is *already*
active:

```powershell
rocm runtimes activate <runtime_key_b> --restart-services --yes
rocm runtimes list
```

That report must read `services_restarted: 1` (or, if the server had not
answered an inference request within the 45-second wait,
`services_restarted_not_yet_serving: 1`; see `rocm services logs <service-id>`)
and `changed_from_runtime_key: <unchanged>`.

Run these in this order. Each step sets up the next, and a step run against a
runtime the server is already recorded on has nothing to move: the reconciler
only restarts servers whose recorded runtime differs from the one being
activated, so it would report `services_on_previous_runtime: 0`, print no
`services_restarted:` line, and verify nothing.

That second command is the no-op-activation case on purpose: it moves the
servers without moving the runtime, and it must leave the rollback target
alone. `rocm runtimes list` must still show one afterwards — an activation that
cleared it would break the `next step: if this causes problems, run rocm
runtimes rollback` line printed directly beneath the note.

Now the server is recorded on `<runtime_key_b>`, so rolling back is a real
switch away from it and exercises the same restart path from the other command.
It restarts the server onto `<runtime_key_a>`, which is why that runtime needed
vLLM too:

```powershell
rocm runtimes rollback --restart-services --yes
```

Expected result:

- The activation report counts the running servers left on the previous runtime
  under `services_on_previous_runtime: <n>` and names each one as
  `- <service-id> engine=<engine> recorded_runtime=<runtime_key>`, then says
  they keep serving on their recorded runtime until they are restarted, naming
  `rocm runtimes activate <runtime_key> --restart-services --yes` as the way to
  move them.
- A Lemonade server is never counted, because that engine brings its own
  runtime: with only Lemonade serving, the count stays
  `services_on_previous_runtime: 0`. That is why the steps above start a vLLM
  server — it is what makes the count non-zero. A server started with `--env-id`
  is counted just like any
  other: the `env_id` recorded in the service record is not a pin and does not
  exempt it from `services_on_previous_runtime`.
- A running server whose record names no runtime is counted apart, under
  `services_with_unrecorded_runtime:`.
- Re-activating the already-active runtime with `--restart-services --yes`
  stops and respawns each counted server onto it, listing them under
  `services_restarted: <n>` and dropping them from
  `services_on_previous_runtime`, while `rocm runtimes list` still marks the
  same rollback target as before.
- The rollback switches back and restarts each counted server onto the runtime
  it restored, listing them under `services_restarted: <n>`; `rocm services`
  shows them running again.
- A server that fails to restart is named under
  `services_restart_failed: <n>`, its record is put back on the runtime it last
  ran on, and the command exits with an error instead of reporting success. The
  error names the servers read back as stopped separately from the ones read
  back as still live: a stopped one shows as stopped in `rocm services` and
  drops out of `services_on_previous_runtime`, while a live one keeps serving
  and stays counted there. These steps can only produce the stopped branch —
  the live branch needs a running server bound to a public host whose endpoint
  key is gone, which nothing here creates, and it is covered by the unit tests.
  Verify the stopped branch against `rocm services`.

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
- A prune running during a launch does not delete the starting server's endpoint
  key. `rocm serve` writes `<data>/services/<id>.endpoint-key` before it writes
  `<id>.json`, and `prune` waits on the shared `<data>/services/launch.lock` that
  `serve` holds across both writes. Start `rocm serve --managed ...` and, while
  it is still coming up, run `rocm services prune --any-age --yes` from a second
  shell: the prune shows `Waiting for a launch already under way…` on a terminal
  and blocks until the launch has published its record — as long as that takes,
  with no timeout — then leaves that service's key and record alone (reporting it
  under "still running, left alone") while still removing every other non-running
  record. `--dry-run` waits in exactly the same way. Run in the other order, a
  `serve` started while a prune is scanning waits for the prune instead — the
  same lock, the same cost, in the other direction.
- `rocm services prune --dry-run` on a machine that has never served creates
  `<data>/services/` and an empty `launch.lock` (the lock the wait above needs),
  and reports that nothing would be removed. That is the only thing a preview
  writes.

With a model still managed-serving (`rocm serve --managed ...`), open the
dashboard or the pre-dashboard launcher and press `q` (or the Esc menu's
`Quit` row, or `/quit` from chat):

```powershell
rocm dash
rocm
```

Expected result:

- A prompt appears naming the serving model and warning that it may
  still be running in the background after you quit, with the real
  `rocm services stop <service-id> --yes` command (the actual id, not a
  placeholder) to stop it first. From the dashboard (not the launcher) on a
  host where its own daemon client never connects (Windows, for the whole
  session), the prompt still shows that same stop command and the service
  id, but not the model's name — the dashboard's own live view is where the
  name comes from, and it never fills there. The launcher's own copy of the
  prompt always names the model, Windows included — it reads the
  managed-service registry directly rather than through the dashboard's
  daemon connection.
- From the dashboard (not the launcher), the prompt also offers `s`: pressing
  it closes the prompt and opens the Services overlay, pre-focused on the
  serving model, so you can stop it there through the overlay's own
  approval+job flow. The launcher's copy of the prompt does not show this
  hint or respond to `s` — it has no live Services overlay to jump to — and
  neither does the dashboard on Windows, for the same reason the model's name
  is missing there.
- Declining (`n`/Esc) returns to where you were — the dashboard, or the
  launcher's own front door, whichever you pressed `q` from. Any plain modal
  that was open when you pressed `q` (the Esc menu, Options, Detail, Theme
  Picker, Help, …) is closed along with the prompt, not restored. A manager
  overlay (Services, the Serve wizard, …) can never be the thing that was
  open in this state at all: its own `q` closes that overlay first, so the
  prompt only ever opens with none of them open.
- Confirming (`y`/Enter) exits immediately, and `rocm services list --all`
  afterward still shows the model as `ready`/`running` — quitting never
  stops it.
- With nothing being served, `q` exits immediately with no prompt, same as
  before this behavior existed.

Now reach the same dashboard FROM the launcher instead (`rocm` → `Open full
dashboard →`), with the same model still managed-serving, and press `q` inside
that escalated session:

Expected result:

- No prompt appears here — this inner session's own `q` returns straight to
  the launcher's front-door menu. The launcher's own `q` (pressed again, at
  that menu) is what asks "May still be serving — quit anyway?", with a fresh,
  on-disk read of the registry backing that decision. Approving it there is
  what actually exits the process. Seeing the prompt twice for one quit
  decision — once inside the dashboard, once more at the launcher, with the
  first one's "yes" not actually exiting anything — is the bug this fixes
  (issue #145).

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

Reinstall over a used install. Put a file in the folder `rocm comfyui
models-path` prints, save a workflow in the UI, generate one image, stop
ComfyUI, then:

```powershell
rocm comfyui install --reinstall --dry-run
rocm comfyui install --reinstall
```

Expected result:

- The dry run prints `reinstall: replaces the ComfyUI code in <folder>`,
  `keeps:` naming the preserved folders the install has (from `models, user,
  output, input, custom_nodes, datasets, extra_model_paths.yaml`),
  `leaves in place:`, and either `sets aside:` or, for an install made before
  rocm-cli recorded what its release shipped, `replaces:`. Its
  `install command:` ends in `--reinstall`. The ComfyUI folder is not changed.
- The reinstall prints `kept:` naming the folders the install already had,
  and the model file, the saved workflow and the image are all still there
  afterwards, unchanged. Put a file such as `styles.csv` at the top of that
  folder first: it is listed under `left in place:` and still there after.
  For an install made before the record, the reinstall also prints
  `replaced:` naming the code entries it deleted.
- With the network disconnected (and no archive cached under the ComfyUI
  folder's `downloads/`), the reinstall fails and the existing install and
  your files are untouched.
- With ComfyUI started (`rocm comfyui start`), the dry run prints a `note:`
  that the reinstall is refused until `rocm comfyui stop`. The reinstall is
  then refused and names `rocm comfyui stop`; after running that, the
  reinstall goes ahead.
- With no archive cached, start ComfyUI (`rocm comfyui start`) from another
  terminal while the reinstall is still downloading: once the download
  finishes, the reinstall is refused before it replaces any code, names
  `rocm comfyui stop`, and the ComfyUI folder is unchanged.
- A download that is not a ComfyUI release (no `main.py` in its top folder)
  is hard to produce by hand; `comfyui-13` serves one and checks that the
  reinstall fails saying so, leaves the install as it was and does not keep
  the download.
- With ComfyUI stopped (which removes rocm-cli's saved record of it), create
  `apps/comfyui/state/running.json` under rocm-cli's data folder containing
  only `{`: the reinstall prints a `Note:` that it cannot tell whether ComfyUI
  is running, and goes ahead.
- If the reinstall fails after the new code is in place (on a host where the
  AMD GPU check fails, say), the error names what was replaced or set aside
  and `rocm comfyui install --runtime-id <key>`; once the cause is fixed, that
  command finishes the install and `rocm comfyui status` shows
  `installed: yes`.

Interrupting the code swap itself is hard to time by hand; the unit test
`interrupted_swap_converges_when_run_again` stops it at every step instead.

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

## 8. Uninstall

Run this last, on a machine installed with the release installer (not a
`cargo` build, whose binaries uninstall leaves alone):

```powershell
rocm uninstall --dry-run
rocm uninstall
```

Expected result:

- The dry run lists the binaries (on Windows, `rocm.exe` is marked as deleted
  as the command exits) and, under
  `PATH changes from the installer that would be undone:`, the shell profile
  (Linux) or the user `PATH` entry (Windows). It removes nothing.
- After confirming, the install folder holds no `rocm`/`rocmd` files, and a new
  terminal no longer gets the install folder on `PATH` from the installer's
  change.
- If another program lives in the install folder, the review instead says
  `PATH change left in place` and names it, and the `PATH` change stays.
- With `rocm uninstall --keep-binaries`, the review says the `PATH` change
  stays too, and it does.
