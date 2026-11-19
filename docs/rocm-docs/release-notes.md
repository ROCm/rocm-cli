---
myst:
  html_meta:
    "description": "ROCm CLI release notes"
    "keywords": "ROCm CLI, release notes, ROCm"
---

<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI 0.3.0 release notes

These release notes describe notable changes since the previous ROCm CLI
release. For the full list, see the
[ROCm CLI releases](https://github.com/ROCm/rocm-cli/releases) on GitHub.

## Release highlights

ROCm CLI 0.3.0 adds a third serving engine and an Anthropic-compatible
endpoint, makes serving and runtime changes safer, and makes driver
installation consistent across Linux distributions.

- **AIM serving engine.** Serve AMD Enterprise AI AIM pre-built Docker
  containers with `--engine aim`, alongside vLLM and Lemonade.
- **Anthropic-compatible endpoint.** `rocm serve` exposes a `/v1/messages`
  endpoint, including tool use and streaming.
- **Follow a starting server.** Stream a server's output, including download
  progress bars, with `rocm logs --service <id> --follow`.
- **Safer serving and runtime changes.** `rocm serve` picks a free port without
  collisions, runtime activation and rollback are transactional, and native
  Windows refuses vLLM up front with the reason.
- **Newest stable ROCm by default.** `rocm install sdk` now installs the newest
  stable ROCm release, and `--python-version` chooses the Python version for a
  ROCm 10 install.
- **Repo-native driver installs.** `rocm install driver` uses the same
  repository-based flow on apt, dnf, and zypper hosts, and adds Ubuntu 26.04.
  `amdgpu-install` is removed.
- **Better diagnosis.** `rocm diagnose` takes `--symptom-file`, estimates the
  memory that a GGUF model needs, and flags an outdated Visual C++ runtime. vLLM
  startup failures now name the failing line.
- **Runtime visibility.** `rocm runtimes list` and `rocm examine` show where each
  runtime came from, and `rocm runtimes list --json` prints JSON.
- **Lemonade on Linux.** Lemonade is pinned to 2026.44.1 so that a ROCm 10.1 SDK
  gets a ROCm backend, and installs check glibc and OpenSSL up front.
- **Security.** Endpoint API key files are accessible only to their owner on
  every platform, and vLLM requires the endpoint key on every server route.

## Breaking changes

- `amdgpu-install` is removed as an install method, with no deprecation period.
  `rocm install driver` on dnf and zypper hosts now uses the same repo-native
  flow as apt: it imports the AMD key, writes a package-manager repository file,
  and installs `amdgpu-dkms`. The default driver source moves from a pinned
  `7.2.4` to `latest`.
- The `ROCM_CLI_AMDGPU_VERSION` and `ROCM_CLI_AMDGPU_PACKAGE_RELEASE`
  environment variables are no longer read. Set `ROCM_CLI_AMDGPU_DRIVER_VERSION`
  to pin a specific amdgpu release. `rocm install driver` refuses to plan while
  either old variable is still set, so an old pin is never silently turned into
  `latest`.

## Added

- `--engine aim` serves AMD Enterprise AI AIM pre-built Docker containers as a
  third serving engine, alongside vLLM and Lemonade. The engine is opt-in. The
  catalog gains the AIM images, which resolve through `rocm serve` and appear in
  `rocm model --verbose`.
- `rocm logs --service <id> --follow` (also `rocm services logs <id> --follow`)
  streams a starting server's output, including in-place download progress
  bars, until the server is no longer starting or running. Press `Ctrl-C` to
  stop following and leave the server running.
- `rocm serve` without `--port` picks the first free loopback port in
  `11435..=11535`. The CLI and the `rocmd` background daemon now claim ports one
  at a time, so two launches can no longer take the same port. An explicit
  `--port` behaves as before.
- `rocm serve` exposes an Anthropic-compatible `/v1/messages` endpoint,
  including tool use and streaming requests.
- `rocm runtimes activate`, `rocm runtimes rollback`, and the activation step of
  `rocm update --activate` and `rocm install sdk` are now transactional. They
  name the running servers that keep the old runtime. If the second of the two
  writes (the configuration and the active-runtime marker) fails, the two no
  longer disagree.
- `rocm runtimes list` and `rocm examine` show where each runtime came from, and
  `rocm runtimes list --json` prints the list as JSON.
- `rocm install sdk` reads the current stable and nightly TheRock hosts by
  default (`stable.repo.amd.com` and `nightly.repo.amd.com`) and keeps earlier
  version pins on the old hosts. Without `--version`, it now installs the newest
  stable release.
- `rocm install sdk --python-version X.Y` chooses the Python version for a
  ROCm 10 SDK install. An install that doesn't need vLLM can use Python 3.10 to
  3.14. The default stays 3.14.
- `rocm install driver` plans the Ubuntu 26.04 (`resolute`) DKMS flow and warns
  about `amdgpu-dkms` left by a previous install, without removing anything.
- `rocm examine` shows the AMD display driver version, date, and age on Windows,
  and flags drivers older than the first HIP SDK for Windows.
- `rocm diagnose` flags a Visual C++ runtime older than 14.40, which can crash
  programs such as the Lemonade llama.cpp server. `rocm examine --json` gains
  `msvc_redist_version`, and a new fix (`fix-21-msvc-redist-outdated`) covers
  the condition. That fix only prints its update plan and never runs the
  install.
- `rocm examine` detects `gfx90c` (Renoir and Cezanne Vega integrated GPUs)
  instead of mapping it to `gfx90X-dcgpu`.
- vLLM startup failures are easier to diagnose. Serve summaries give
  out-of-memory diagnostics and guidance. When vLLM aborts at startup because it
  can't import its engine (for example, a missing `libtorch_cuda.so`), the
  failure names the failing line and the `rocm diagnose --symptom '<line>'`
  command that diagnoses it. The `rocm diagnose` fix `fix-17-torch-dlpack` now
  covers Linux and WSL.
- `rocm diagnose --symptom-file <path>` reads the symptom text from a file or
  from standard input, and each match says whether the symptom text contributed
  to it.
- `rocm diagnose` estimates the context size and memory that a GGUF model needs.
- `rocm fix --json` describes each published exit code.
- vLLM discovery recognizes ROCm 10.2 builds and local version tags of the form
  `.rocmNNN`.
- Lemonade installs announce themselves and show their progress, and
  `rocm storage` reports them.
- Lemonade managed-serve failures capture the identity of the backend.
- On Linux, `rocm engines install lemonade` and starting Lemonade check the
  host's glibc and OpenSSL first, and say what falls short instead of failing at
  launch. Set `ROCM_CLI_DISABLE_LEMONADE_HOST_LIBRARY_CHECK=1` to turn the check
  into a warning on hosts that provide the libraries another way.
- The dashboard shows recent serving throughput in the instance detail view,
  shows starting services on the **Home** and **Serving** tabs, and collects GPU
  and memory metrics in-process on Windows.
- The dashboard shows one node-efficiency figure everywhere it appears, with a
  held flag. It also attributes VRAM and power to managed `rocm serve` services,
  and lists running managed services on Windows.
- The dashboard chat waits two minutes for a reply by default, and the wait is
  configurable.
- The dashboard **Diagnose** screen can prepare a Doctor report. Press `p` to
  preview what the report would carry (nothing is sent). Press `s` to send it
  only after approval. Deny is the default.
- The Doctor skill (`rocm-doctor`), which diagnoses broken ROCm, HIP SDK, and
  PyTorch setups, stops when the CLI predates the failure catalog that the skill
  requires. It also covers the two vLLM startup failures above: out of memory,
  and a `libtorch_cuda.so` import error.
- Before offering the mail, `rocm diagnose --report` now says that the mail goes
  from your own account and carries your address, and that the report itself
  names no user, host, or file path.
- The dashboard asks for confirmation before quitting while a model is still
  being served.
- `rocm install` prints cleaner progress and a completion banner.
- The engine stdio protocol is now versioned. An engine built for an
  incompatible protocol is refused with a clear message instead of failing
  later.
- An engine caller can withhold an inherited endpoint key file.

## Changed

- `rocm serve <model>` on a host with no usable AMD GPU now succeeds when the
  model matches an already-running managed service for the same engine, and
  prints the reuse summary. Every other case is still refused under the default
  `gpu_required` policy.
- The dashboard serve wizard is now model selection plus inline **Advanced**
  settings, validated the same way as the CLI before approval.
- The dashboard assistant no longer advertises the `list_skills` and
  `skill_plan` tools, whose install recipes could not run.
- `rocm install sdk` without `--version` now installs the newest stable ROCm
  release instead of ROCm 7.14. To keep the earlier release, pin it with
  `--version 7.14.1`.

## Resolved issues

### Install and uninstall

- Fixed `rocm install sdk --dry-run` downloading `uv` and installing a managed
  Python on a host without one. The preview now says that a real install would
  prepare Python (`python_note:`) and changes nothing.
- Fixed `rocm uninstall --dry-run` creating the folders that it then listed for
  removal.
- Fixed `rocm uninstall` leaving behind the `PATH` change that the installer
  made, and, on Windows, leaving the running `rocm.exe` in the install folder.
  The installers now record their `PATH` change so that uninstall can undo
  exactly that change.
- Fixed `rocm storage` and `rocm uninstall` acting on the default `~/.rocm/cache`
  instead of `<data>/cache` when `ROCM_CLI_CONFIG_DIR` and `ROCM_CLI_DATA_DIR`
  are set but `ROCM_CLI_CACHE_DIR` is not.
- Fixed vLLM installs on ROCm 10.x failing with an unsatisfiable `amd-aiter`
  pin. The `flash-attn` and `amd-aiter` pins now come from what the discovered
  vLLM depends on.
- Fixed `rocm engines install lemonade` failing on every run when a kept ROCm
  alignment pin names a download that is no longer served. The packaged backend
  pins are now restored.
- Fixed a ROCm 10.1 SDK getting no usable Lemonade ROCm backend. Lemonade is now
  pinned to 2026.44.1, which has one.
- Fixed a vLLM install being reported as usable before its pinned `tensorizer`
  package imports.
- Fixed `rocm install sdk --dry-run` fetching `uv` for a ROCm 10 SDK. The
  preview no longer downloads anything.
- Fixed the update check preparing a managed Python.
- Fixed the installer not configuring fish when fish isn't the login shell.
- Fixed installer download progress drawing outside the terminal.
- Fixed `rocm uninstall` printing a wrong dry-run footer and answering `--force`
  with `--force-dev-binaries`.

### Configuration and runtimes

- Fixed an unreadable `config.json` being replaced with defaults.
- Fixed running ROCm 7 and ROCm 10 runtimes side by side, and a removed rollback
  target staying in the active-runtime marker.
- Fixed the SDK torch stack being dropped or replaced when torch alignment is
  disabled, and the ROCm 10 SDK torch and torchvision not being bounded to the
  series that vLLM pairs.
- Fixed a second `amd-smi` library ending up in the runtime's torch.
- Fixed a managed runtime on WSL not loading its own `librocdxg`, and not saying
  when it doesn't.

### Serving and services

- Fixed `rocm serve --engine vllm` on native Windows printing
  `managed service launched` and exiting with code 0 while the engine died with
  no output. It, and `rocm engines install vllm`, now refuse up front and give
  the platform reason.
- Fixed `rocm serve` on Windows reporting a managed server as started when it
  exited immediately. It now fails fast and shows the startup log.
- Fixed `rocm services` reporting an unconfirmed stop as stopped.
- Fixed `rocm serve` hanging, and surviving `Ctrl-C`, when `amd-smi` ran against
  a kernel driver in a bad state. The `/dev/kfd` pre-flight check now runs every
  time `amd-smi` is launched, and an `amd-smi` child can no longer outlive its
  timeout.
- Fixed `--gpu` accepting an index that Windows does not have.
- Fixed an unknown service ID surfacing the raw operating system error instead
  of saying that the service was not found.
- Fixed managed-service recovery from the embedded and the standalone daemon.
  Managed services are now stopped through one routine shared by `rocm` and
  `rocmd`, with hardened identity, liveness, and tree-kill handling on Windows.
- Fixed Lemonade and vLLM teardown not escalating to `SIGKILL` when an engine
  ignores the polite stop request.
- Fixed Lemonade service state being written non-atomically, so that concurrent
  writers could corrupt it.
- Fixed serving a Lemonade model registered by name on Linux, and discovering
  serving models in the dashboard.
- Fixed a Hugging Face repository with no variant being passed to Lemonade. It's
  now refused up front.
- Fixed the small assistant in the catalog pointing at the wrong GGUF build for
  Lemonade.
- Fixed `rocm serve` not saying in its summary when the post-launch smoke test
  fails.
- Fixed log tails being reported as empty when the log had content.
- Fixed the service ID being dropped from inline `key=value` service rows.

### Detection and diagnosis

- Fixed `rocm examine` reporting `wsl_rocdxg_missing` and
  `rocm serve --gpu auto` refusing with `no usable AMD GPU detected` on WSL when
  `librocdxg.so` is reachable only through `LD_LIBRARY_PATH`.
- Fixed `rocm examine --json` not reporting an AMD GPU on WSL when the Windows
  host has one.
- Fixed `rocm diagnose` listing `fix-13-hip-sdk-missing` as the top result on
  hosts with no HIP-based setup.
- Fixed commands that `rocm diagnose` prints for Windows not running in
  PowerShell.
- Fixed a host ROCm version with a non-numeric build suffix, such as
  `7.2.4-98rc1`, being silently read as `7.2.4`. It is now treated as unknown.
- Fixed `/etc/os-release` being parsed differently by the driver plan,
  `rocm examine`, and the OpenMPI hint, so that a quoted `ID='ubuntu'` read as an
  unsupported distribution. One parser now reads it, and `/usr/lib/os-release` is
  used when `/etc/os-release` is absent.
- Fixed `rocm diagnose --report --send` and the dashboard send action never
  opening a mail client on Windows, because the link was cut off at the `&`
  character.
- Fixed `rocm examine` not reporting `amdkfd_loaded` and `amdgpu_loaded` when the
  drivers are built into the kernel.
- Fixed a timed-out `rocm fix` step not being reported as a timeout.
- Fixed wording in several places in the Doctor skill and `rocm fix`.

### Daemon and automation

- Fixed automation stores (events, audit, and proposals) losing records when
  several writers wrote at once. A store whose last record was incomplete and
  couldn't be repaired by replacing the file, for example on a full disk or when
  Windows security software held the file open, could stop `rocmd` from
  starting. It is now repaired in place.
- Fixed automation stores growing without bound. They are now capped by rotation
  and compaction.
- Fixed a detached `rocm daemon` staying alive on native Windows after
  `rocm serve` and `rocm services stop`. It now runs only while an automation
  needs it, and `rocm services stop` and the uninstall plan say what remains.
- Fixed `rocmd` reporting a timeout when a child process wrote more output than
  a pipe can hold, because it did not read the output until the child exited.
- Fixed `rocmd` leaving an engine child process running when the supervisor
  couldn't record it.

### Dashboard and chat

- Fixed the dashboard daemon dropping managed services when the registry could
  not be read, and keeping a stale Lemonade instance after its host or port
  changed.
- Fixed the dashboard **Hardware** tab labeling binary VRAM figures as GB instead
  of GiB.
- Fixed the dashboard header warning badge (`⚠ N`) giving no way to read the
  warnings. It now opens their text.
- Fixed the dashboard **Services** manager showing every service's status as
  healthy, including stopped and failed services.
- Fixed the dashboard showing an idle server's all-time time to first token
  (TTFT) and time per output token (TPOT) indefinitely. Values are now held,
  marked with `*`, and then cleared.
- Fixed dashboard chat tool calls stalling other dashboard work while a slow
  probe ran.
- Fixed chat with the ChatGPT sign-in silently ignoring `--temperature`,
  `--top-p`, and `--max-tokens`. It now warns.
- Fixed the dashboard running a different tool call from the one its approval
  card shows. It now runs exactly the call you approved.
- Fixed local Qwen3 chat answers in the dashboard being lost to unbounded
  reasoning, and dashboard chat tool output swamping small-context models.
- Fixed the dashboard **Home** hero showing a `-0` throughput, the two
  **Observe** heroes disagreeing on throughput, ADLX initialization on Windows,
  and CPU graphs not rendering continuously.

### Command output

- Fixed command previews that did not read back as the exact arguments that run.
  Arguments are now quoted for a POSIX shell, including in copyable lines.
- Fixed CLI output such as `rocm model` showing an em dash as `ΓÇö` on Windows
  consoles that use the OEM code page. Output text is now ASCII.
- Fixed `rocm version` reporting the frozen 0.1.0 instead of the release tag.
- Fixed a `[y/N]` answer that isn't understood being silently treated as no. The
  prompt now says so, and ends the line on EOF.
- Fixed the natural-language plan shown to you being built separately from the
  plan that runs, so that a cloud planner's non-deterministic answer could make
  them differ. The plan is now built once per request.

## Known issues

- Instinct serving on RHEL 9.6 can still need manual vLLM, OpenMPI, and
  `amdsmi` workarounds.
- On ROCm SDK 10.2.0 with Lemonade 2026.44.1 (observed on Strix Halo in WSL2),
  copies between ordinary host memory and the GPU run at roughly 20 MiB/s, more
  than 100 times slower than on 10.1.0, so loading a large model takes many
  minutes. `rocm serve` with Lemonade stops the server at its 2-minute readiness
  timeout and marks the service as failed. As a workaround, install ROCm SDK
  10.1.0 with `rocm install sdk --version 10.1.0`, or run
  `rocm runtimes rollback` if a 10.1 runtime was the previous one. Then start
  serving again with `rocm serve <model>`.
