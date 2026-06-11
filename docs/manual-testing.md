# Developer Manual QA

Use this checklist to verify rocm-cli behavior as a developer or release
tester. On Windows, run these commands from PowerShell. On Linux or WSL, use
the same commands in a shell.

For normal users, keep the flow simple: install rocm-cli, run `rocm`, choose a
ROCm folder, approve setup, and then use the main TUI.

When validating release behavior, use the single universal binary:

```text
.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe
```

On Windows, run it directly. On WSL/Linux, copy or rename the same file to
`rocm` and run it through `sh` so the Linux path is used:

```bash
sh ./rocm doctor
```

Current caveat: direct `./rocm` inside WSL can still be intercepted by
WSLInterop as a Windows executable before rocm-cli starts. The WSL release
smoke must report `os: linux` through `sh ./rocm`.

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

## 1. First-Time Setup

Start rocm-cli:

```powershell
rocm
```

Expected result:

- A setup screen opens automatically before the main TUI.
- The user does not need to type `/setup`.
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
rocm doctor
rocm runtimes list
```

Expected result:

- `rocm doctor` shows a managed runtime.
- `rocm runtimes list` shows the runtime key for the installed TheRock venv.
- The active runtime is ready, or the output gives one clear next command.

## 2. TheRock SDK Command-Line Install

This tests the command-line install path without using the TUI:

```powershell
rocm install sdk --channel release --format pip --prefix .\.rocm-work\data\envs\default
rocm runtimes list
rocm runtimes activate <runtime_key>
rocm doctor
```

Replace `<runtime_key>` with the exact key printed by `rocm runtimes list`.
Omit `--prefix` if you want rocm-cli to choose its standard managed folder.

Expected result:

- rocm-cli creates or reuses a rocm-cli managed Python venv.
- pip installs pinned `rocm[libraries,devel]`, `torch`, `torchvision`, and
  `torchaudio` versions from the TheRock index.
- rocm-cli chooses the newest exact ROCm build suffix common to the SDK package
  and the PyTorch stack for the current Python/platform wheel tags, then pins
  all four packages in one pip transaction.
- The install does not ask for an external Python venv.
- Runtime validation uses TheRock's runtime/devel package roots and
  `rocm_sdk.find_libraries`; `rocm-sdk path --root` is expected after the
  pinned `rocm[libraries,devel]` install succeeds.
- `rocm doctor` reports the active runtime as ready.

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
rocm serve qwen --engine lemonade --device gpu_required --managed --foreground --port 11435
```

Expected result:

- The engine uses ROCm GPU execution.
- The run does not fall back to CPU or Vulkan.
- If the GPU cannot be used, the command fails with a clear error.
- The OpenAI-compatible endpoint answers a simple chat request.

Stop the foreground server with `Ctrl+C`.

## 4. PyTorch GPU Verification

Install or refresh the PyTorch engine:

```powershell
rocm engines install pytorch
```

Serve a small model with the managed runtime:

```powershell
rocm serve qwen --engine pytorch --device gpu_required --managed --foreground --port 11435
```

Expected result:

- The engine uses the active rocm-cli managed TheRock runtime.
- PyTorch detects the AMD GPU.
- The run does not fall back to CPU.
- If the GPU cannot be used, the command fails with a clear error.

Current known caveat: explicit `Qwen/Qwen3.5-4B` PyTorch requests are gated
before launch because the installed Transformers package does not recognize
the checkpoint's `qwen3_5` architecture. The short alias `qwen` resolves to
the recommended low-VRAM `Qwen/Qwen2.5-1.5B-Instruct` assistant recipe.

Stop the foreground server with `Ctrl+C`.

## 5. Local Server Records

After a managed or foreground serve attempt, inspect local server records:

```powershell
rocm services
rocm services list --all
rocm services logs <service-id>
```

Expected result:

- `rocm services` shows only living local servers.
- `rocm services list --all` shows saved history, including failed or stopped
  attempts.
- The logs command shows the exact service failure or startup output.
- Stop and restart require explicit approval:

```powershell
rocm services stop <service-id> --yes
rocm services restart <service-id> --yes
```

## 6. llama.cpp GPU Verification

Install or refresh the llama.cpp engine:

```powershell
rocm engines install llama.cpp
```

Run a GGUF model through llama.cpp with the managed runtime:

```powershell
rocm serve target\models\stories260K.gguf --engine llama.cpp --managed --foreground --port 11450
```

Expected result:

- llama.cpp starts with HIP enabled.
- ROCm libraries load from the active rocm-cli managed TheRock runtime.
- The run does not fall back to CPU.
- If `llama-server` or the model file is missing, the command says what is
  missing.

Stop the foreground server with `Ctrl+C`.

For the stricter developer GPU test:

```powershell
python scripts\llama_cpp_therock_gpu_test.py --timeout 120
```

This test downloads or reuses a tiny GGUF model, launches llama.cpp with GPU
required, checks the HTTP endpoint, and verifies that the loaded ROCm libraries
come from the managed TheRock runtime.

## 7. ComfyUI Verification

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

## 8. Optional Cloud Provider Key

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
- `config.json`, logs, and doctor output do not contain the key.

To remove the saved key:

```powershell
rocm config clear-provider-key openai
```

## 9. Optional Provider-Assisted Planning

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
