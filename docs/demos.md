<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Demo screencasts

The README embeds two complete terminal demos:

- **ROCm CLI** follows a first-use journey from environment inspection through
  local-model chat.
- **ROCm Console** tours the real full-screen TUI with deterministic simulated
  telemetry and its offline chat backend.

Both are authored as [VHS](https://github.com/charmbracelet/vhs) tapes, rendered
in CI, and published to the orphan **`media`** branch. GIFs are never committed
to source branches; the README references them by absolute raw URL.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/tapes/cli.tape` | CLI first-use journey. |
| `docs/tapes/console.tape` | Interactive Console tour using `--demo --chat-mock`. |
| `xtask/src/demos.rs` | Builds binaries, starts the isolated mock service, runs VHS, and cleans up. |
| `tests/e2e-cucumber/src/bin/rocm-demo-env.rs` | Mock-service helper: reuses the e2e harness's loopback OpenAI server and plants a service record under isolated config. |
| `.github/workflows/demo-gifs.yml` | Installs VHS and publishes generated GIFs to `media`. |
| `docs/media/` | Git-ignored render output directory. |

## Deterministic environment

`cargo xtask demos` performs all setup before VHS starts:

1. Builds release versions of `rocm` and `rocm-demo-env`.
2. Starts `rocm-demo-env`, which reuses the end-to-end harness's loopback
   OpenAI-compatible server and plants a managed-service record under isolated
   config, data, and cache directories.
3. Waits for the helper to publish all required environment paths and its
   readiness marker.
4. Prepends the release binary directory to `PATH`, renders the selected tapes,
   then stops the helper and removes its temporary state.

Setup deliberately does not live inside a tape. VHS advances on a fixed clock
and does not wait for commands to finish, so hidden setup can race subsequent
keystrokes. The tapes remain pure visible command/key sequences against an
already-ready environment.

The Console needs no ROCm hardware: `rocm dash --demo` replays the project's
seeded synthetic telemetry through the same UI as a live daemon, and visibly
marks it **SIMULATED DATA**. `--chat-mock` provides the deterministic offline
chat response. The CLI's service and chat commands use only the loopback mock.
Neither demo downloads a model or calls a cloud provider.

## Storyboards

### CLI

`cli.tape` demonstrates:

1. `rocm examine`
2. `rocm engines list`
3. `rocm model`
4. `rocm services list`
5. one-shot `rocm chat` against the discovered mock service

### Console

`console.tape` launches `rocm dash --demo --chat-mock`, visits Home, ROCm,
Serving, Observe, and Chat, asks which simulated GPU needs attention, opens the
built-in help, and exits cleanly.

## Regenerate

Install VHS and its runtime dependencies (`ttyd`, `ffmpeg`, and Chrome), then
run from the repository root:

```bash
# Render both demos, building the binaries first.
cargo xtask demos

# Render one demo.
cargo xtask demos cli
cargo xtask demos console

# Reuse release binaries already present in target/release.
cargo xtask demos --skip-build
```

Set `ROCM_BIN_DIR` to use binaries from another directory. `ROCM_DEMO_MODEL`
changes the model the mock service advertises; the CLI tape's `--model`
argument must match it (both default to the same identifier), so override
them together if you change one.

Running `rocm-demo-env` directly (instead of through `cargo xtask demos`) skips
the wrapper's `ROCM_MOCK_CHAT_REPLY` export, so the mock server's chat replies
fall back to a generic test stub; set `ROCM_MOCK_CHAT_REPLY` yourself for a
realistic chat response.

> [!NOTE]
> Browser-backed VHS GIF rendering can hang under WSL2. On WSL2, run the Rust
> tests and command-level checks locally, then use the `demo-gifs` GitHub Actions
> workflow for the actual GIF render.

## Add or change a demo

1. Keep each tape a pure command/key sequence; put environment orchestration in
   `xtask/src/demos.rs`.
2. When adding a new tape, add its name to the `DEMOS` list in that module.
3. Reference the published GIF as
   `https://raw.githubusercontent.com/ROCm/rocm-cli/media/<name>.gif`.
4. Verify the story without relying on GPU hardware, model downloads, or
   non-loopback services.

## Workflow triggers

The workflow runs on `workflow_dispatch` and on published releases, never on
pull requests. Rendering is slow and would churn the generated-media branch.
GitHub proxies README images through a cache; append a release-based query
parameter if a regenerated image appears stale.
