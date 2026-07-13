<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Demo screencasts

The README embeds short terminal screencasts (GIFs) of common `rocm` commands.
They are authored as [VHS](https://github.com/charmbracelet/vhs) tapes, rendered
in CI, and published to the orphan **`media`** branch. GIFs are **never committed
to source branches** — the README references them by absolute raw URL, so the
binaries stay out of `main`'s history.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/tapes/*.tape` | One VHS tape per demo — a **pure command sequence**, no setup. |
| `docs/tapes/render.sh` | Sets up the env + mock server, then runs `vhs <tape>`. |
| `docs/tapes/lib/demo-env.sh` | Sourced by `render.sh`; starts the mock server and points `rocm` at an isolated config. |
| `.github/workflows/demo-gifs.yml` | Builds the binaries, renders every tape, publishes GIFs to the `media` branch. |
| `docs/media/` | Render output directory (git-ignored; present only so `vhs` has a target). |

## Why setup lives in render.sh, not the tapes

VHS types on a fixed clock and **never waits for a command to finish**. If a tape
does its own setup (e.g. `source`-ing a helper that blocks while a server starts),
VHS keeps typing into a busy shell — the following commands never execute cleanly
and the "hidden" setup leaks into the recording. So all setup happens in
`render.sh` *before* `vhs` starts, and every tape is a pure command sequence that
runs against an already-ready shell.

## How the server-backed demos work

`chat` and `services` need a running model endpoint. Instead of a GPU or a real
model, they reuse the end-to-end test harness's mock OpenAI server via the
standalone `rocm-demo-env` binary (`tests/e2e-cucumber/src/bin/rocm-demo-env.rs`).
`render.sh` sources `demo-env.sh`, which:

1. Starts `rocm-demo-env` — boots the axum mock server on a loopback port and
   plants a managed-service record into an isolated config dir.
2. Exports `ROCM_CLI_CONFIG_DIR` / `ROCM_CLI_DATA_DIR` / `ROCM_CLI_CACHE_DIR` so
   `rocm` discovers the mock, and `ROCM_MOCK_CHAT_REPLY` for a realistic answer.
3. Installs an EXIT trap that stops the server when rendering finishes.

Output is deterministic and needs no hardware, so renders are reproducible.

## Regenerate locally

Install [VHS](https://github.com/charmbracelet/vhs) (plus `ttyd` + `ffmpeg`),
then from the repo root:

```bash
# Build the binaries the tapes invoke.
cargo build --release -p rocm --bin rocm
cargo build --release -p e2e-cucumber --bin rocm-demo-env

# Render one tape (writes docs/media/<name>.gif).
docs/tapes/render.sh docs/tapes/chat.tape
```

`render.sh` resolves binaries from `target/release` by default; override with
`ROCM_BIN_DIR=/path/to/bin`. Change the demo model with `ROCM_DEMO_MODEL`.

## Add a demo

1. Add `docs/tapes/<name>.tape` with `Output docs/media/<name>.gif` — a pure
   command sequence (no `Hide`/`source`; `render.sh` provides the env).
2. Add `<name>` to the render loop in `.github/workflows/demo-gifs.yml`.
3. Reference the published GIF from the README:
   `https://raw.githubusercontent.com/ROCm/rocm-cli/media/<name>.gif`.

## Triggers

The workflow runs on `workflow_dispatch` and on published releases — never on
pull requests (rendering is slow and would churn the `media` branch). GitHub
proxies README images through a cache; when a regenerated GIF looks stale, append
a cache-buster to the URL (e.g. `...chat.gif?v=<release-tag>`).
