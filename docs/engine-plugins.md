<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Engine Plugins

rocm-cli discovers serving engine adapters as executable files.

Search order:

1. `<data_dir>/engines/plugins`
2. `<data_dir>/engines`
3. Packaged sibling binaries installed beside `rocm`

The first directory is the preferred location for external adapters. Use a
binary name in the form `rocm-engine-<engine>` on Linux/WSL and
`rocm-engine-<engine>.exe` on Windows.

Packaged first-party adapters are `lemonade` and `vllm`. Linux/WSL-only ROCm GPU
adapters (such as `vllm`) fail explicitly on native Windows instead of selecting
a CPU fallback.

The engine-selecting commands (`rocm serve --engine`, `rocm engines
install`/`shell`, `rocm config set-engine`/`set-default-engine`) currently accept
only the built-in `lemonade` and `vllm` engines. Discovery still lists external
plugins under the search directories above, but selecting one by name from the
CLI is not supported while the engine set is limited to the two built-ins.

The `lemonade` adapter uses Lemonade embeddable and prefers Lemonade's
`llamacpp:rocm` backend, falling back to `llamacpp:vulkan` when ROCm is
unsupported. rocm-cli does not use a CPU fallback for this path.

## Model tasks

An engine advertises which kinds of model it can serve in the `tasks` field of
its capabilities: `chat` (`POST /v1/chat/completions`), `transcription`
(speech-to-text, `POST /v1/audio/transcriptions`), and `speech` (text-to-speech,
`POST /v1/audio/speech`). An engine that does not
send the field — an external plugin written before it existed — is treated as
serving `chat` only. `rocm serve` reads each model's task from its catalog
recipe (a model outside the catalog is a chat model) and refuses, before any
GPU check or engine preparation, to hand a model to an engine that does not
advertise its task. The task is passed to the engine as `task` in the
`ResolveModel` and `Launch` requests, omitted for chat so a request for a chat
model is unchanged.

`lemonade` advertises `chat`, `transcription`, and `speech`; `vllm` advertises
`chat`.

Lemonade serves a speech-to-text model through its router with the
`whispercpp` recipe, and a text-to-speech model with the `openmoss` recipe. The
adapter names the backend explicitly — `rocm`, or `vulkan` when Lemonade
reports ROCm unsupported, never `cpu` — because Lemonade's `auto` picks
whichever whisper.cpp build is installed, which can be the CPU one (OpenMOSS
has no CPU build). Before reporting ready it checks that Lemonade loaded the
model on that backend and that a test request for the task (a transcription,
or one spoken word) returns HTTP 200. It does not trust
the per-model `device` field of Lemonade's health report: Lemonade 11.5.1
derives it from the backend name, labelling a ROCm Whisper model `cpu`, and a
Vulkan one `gpu` even where whisper.cpp found no Vulkan device. `rocm serve`'s
usable-AMD-GPU check, which runs before the engine starts, ensures an AMD GPU
is present, but not that a Vulkan build of whisper.cpp or OpenMOSS can use it:
where the Vulkan driver exposes no device, that build may run on the CPU
without reporting it. This is a known gap, shared with the chat path's Vulkan
llama.cpp; the ROCm backends are not affected.

## Lemonade backend alignment on engine install

Lemonade's `resources/backend_versions.json` pins which ROCm SDK version its
`llamacpp:rocm` backend downloads. On Linux/WSL and Windows, `rocm engines
install lemonade` attempts to rewrite that pin to match the active ROCm SDK so
the backend it installs is paired with the SDK actually in use, rather than
whatever version Lemonade shipped pinned to. The attempt is skipped outright
on any other OS, where no mechanism exists to verify whether a backend
actually resolves its ROCm libraries, left alone when the active SDK reports
a nightly version rather than a plain `X.Y.Z` the alignment can match against
(or no active SDK version can be determined at all, or Lemonade's packaged
pin can't be read), skipped when the pin already matches the active version,
and skipped on a host where Lemonade would select `llamacpp:vulkan` instead of
`llamacpp:rocm` regardless (WSL2 being the documented case, and also the
fallback if that probe itself fails), since no rewrite can make an
unsupported ROCm build
resolve there.

The rewrite tries two builds in order. The first keeps Lemonade's own pinned
llama.cpp release and just repoints it at the active ROCm version — a
deliberate, reproducible pin. If that attempt fails to resolve against the
installed backend's shared libraries — for example because that release never
shipped an asset for the active ROCm version — the second queries GitHub's
`releases/latest` for `lemonade-sdk/llama.cpp` (an unauthenticated
`api.github.com` call, subject to GitHub's public rate limit) and installs
whatever build is newest at that moment. That second build is a moving
target, not a pin: which llama.cpp commit actually gets installed depends on
when the install ran, and the version installed is not recorded anywhere
`rocm examine`/`rocm version` report. An attempt that cannot be verified
against the installed backend's shared libraries reverts to whichever version
was pinned before the attempt — the packaged pin on a fresh install or after
`--reinstall`, but whatever alignment last wrote otherwise — rather than
being kept.

Set `ROCM_CLI_DISABLE_LEMONADE_BACKEND_ALIGNMENT` to keep whatever
`backend_versions.json` already pins and skip the rewrite — any value works,
including an empty one, since the variable being set is the signal:

```bash
ROCM_CLI_DISABLE_LEMONADE_BACKEND_ALIGNMENT=1 rocm engines install lemonade --yes
```

Use it if you have hand-edited `backend_versions.json` to pin a specific
version, matching that file's own documented use as a first-party
customization point.

## Pinned runtime versions

The versions of the third-party runtimes rocm-cli downloads are pinned in
`runtime-deps.toml` at the repository root — one `[runtime.<name>]` table per
runtime. That file is the only place a runtime version is written down:
archive names, download URLs, and the dashboard's offline fallback are all
derived from it, so a bump is a one-line edit and the tree cannot end up
holding two different versions of the same runtime.

`rocm engines list` shows the exact plugin directories for the current host.
The same output is available in the TUI with `/engine`.

Installer policy:

- `install.sh` and `install.ps1` update only the rocm-cli binary install
  directory (its binaries, `.rocm-cli-manifest`, and the `.rocm-cli-path-edit`
  record of the `PATH` change they made) and that `PATH` change itself: a
  marked block in the shell profile, or the user `PATH` on Windows.
- External plugins under the rocm-cli data directory are not touched by
  install or upgrade.
- `rocm uninstall` removes the data directory by default. Use
  `rocm uninstall --keep-data --keep-config` when external plugins, managed
  runtimes, service records, or model cache entries should be preserved. On the
  default layout the configuration and data directories are the same
  directory, so both flags are needed.

No fallback engine is selected automatically. If an engine adapter is missing
or cannot satisfy the requested device policy, the command must fail until the
requested engine is installed or the user explicitly selects a different one.
