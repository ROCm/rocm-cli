<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# E2E Tests (cucumber-rs)

Behavioral end-to-end tests using [cucumber-rs](https://github.com/cucumber-rs/cucumber) — Gherkin `.feature` files backed by Rust step functions.

## Architecture

```
.feature files (Gherkin)  →  step functions (Rust)  →  rocm binary / mock server
```

Tests exercise the `rocm` binary as a black box — no imports from the rocm-cli codebase. Mock-tier tests use an in-process axum server; GPU-tier tests run real model serving.

## Prerequisites

- Rust toolchain (cargo)
- For GPU-tier tests: an AMD GPU with ROCm drivers and `rocm` binary on PATH

## Directory layout

```
tests/e2e-cucumber/
├── Cargo.toml                    # crate definition, dependencies
├── README.md
├── expectations.toml             # per-scenario known-bug (xfail) matrix, keyed by @id
│
├── features/                     # .feature files — one per feature area
│   ├── chat.feature
│   ├── examine.feature
│   ├── model_serving.feature
│   └── runtime_setup.feature
│
├── tests/                        # test binary + step modules
│   ├── e2e.rs                    # World struct, runner, expectation reconciliation, Drop cleanup
│   └── e2e/                      # step functions — one file per feature area
│       ├── chat_steps.rs
│       ├── examine_steps.rs
│       ├── runtime_steps.rs
│       └── serving_steps.rs
│
└── src/                          # shared test infrastructure
    ├── lib.rs
    ├── capability.rs             # host capability probe (OS / GPU / effective engine)
    ├── expectation.rs            # tag parsing + pass/xfail/skip resolution
    └── mock_server.rs            # axum mock OpenAI server
```

## Running tests

The `cargo xtask e2e` task builds the release `rocm` binary and runs the suite.
Extra arguments after `--` are forwarded to the cucumber CLI.

```bash
# All features:
cargo xtask e2e

# Filter by scenario name:
cargo xtask e2e -- -n "model name"

# Stop on first failure:
cargo xtask e2e -- --fail-fast

# With a pre-built binary (skips the build step):
ROCM_CLI_BINARY=./target/release/rocm cargo xtask e2e
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ROCM_CLI_BINARY` | `rocm` (on PATH) | Path to the rocm binary under test |
| `ROCM_CLI_CONFIG_DIR` | (temp dir) | Isolated config directory per scenario |
| `ROCM_CLI_DATA_DIR` | (temp dir) | Isolated data directory per scenario |
| `ROCM_CLI_CACHE_DIR` | (temp dir) | Isolated cache directory per scenario |

## Tags and per-scenario expectations

There is no tag-filter tiering. Each CI job runs the **whole** suite
(`cargo xtask e2e`, no `-t` filter); the harness resolves every scenario to
**pass / xfail / skip** at runtime from its capability tags plus the known-bug
matrix, then reconciles the actual result against that expectation.

Scenarios carry stable-id and capability tags:

| Tag | Meaning |
|---|---|
| `@id:<slug>` | Stable scenario id. Keys the expectation matrix and the report grid; every scenario has one. |
| `@requires-gpu` | Needs a real AMD GPU. Resolves to **skip** (n/a) on a host with none (e.g. the mock or hosted WSL job). |
| `@requires-wsl` | Needs a real WSL host. Resolves to **skip** on native Linux, native Windows, and other environments. |
| `@requires-no-wsl` | Needs native host behavior that WSL deliberately routes around. Resolves to **skip** on WSL. |
| `@requires-engine:<vllm\|lemonade>` | Pins the serve engine. Resolves to skip where that engine can't start (e.g. vLLM on a lemonade-only Strix host). |
| `@requires-os:<linux\|windows>` | Premise is OS-specific; skip on other OSes. |
| `@serve-timeout:<secs>` | Lengthen the serve-readiness wait for a genuinely slow serve (e.g. a large model). |
| `@nightly` | Expensive scenario skipped by default; included when `E2E_INCLUDE_NIGHTLY=1`. |

Known bugs are **not** tagged in the `.feature` files — they live in
`expectations.toml`, keyed by `@id`, each with a `when = { ... }` condition (e.g.
`effective_engine = "vllm"`), a `bug` reference, and a `reason`. A scenario that
matches a condition is expected to fail (xfail); if it then passes, that is an
**XPASS**. Deterministic XPASS is stale and must be removed; entries marked
`flaky = true` tolerate either outcome while still reporting the intermittent
bug. See `src/expectation.rs` for the resolver and `expectations.toml`'s header
for the condition grammar.

CI runs one job per platform, each executing the full suite:

| Job | Platform | Blocking |
|---|---|---|
| `e2e` | Mock (no GPU, GitHub-hosted Linux) | yes |
| `e2e-wsl` | WSL2 / Ubuntu (no GPU, GitHub-hosted Windows) | no |
| `e2e-gpu` | MI300X (self-hosted) | no |
| `e2e-gpu-strix-ubuntu` | Strix Halo / Ubuntu (self-hosted) | no |
| `e2e-gpu-strix-windows` | Strix Halo / Windows (self-hosted) | no |

The blocking mock job passes when every applicable scenario is pass-or-xfail with
no XPASS or unexpected failure. The hosted WSL and GPU jobs are non-blocking.
The WSL job validates the real WSL2 host boundary, Linux build, and CLI behavior;
it has no AMD GPU and does not replace GPU-on-WSL hardware coverage. The
`e2e-report` job consolidates all platforms' results into one cross-platform
report.

The nightly workflow runs three non-blocking jobs — the existing MI300X job and
new Strix Halo jobs on Ubuntu and Windows — with `E2E_INCLUDE_NIGHTLY=1`. The
shared large-model scenario serves `Qwen/Qwen3.6-27B` through vLLM on MI300X and
the hardware-verified `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL` checkpoint
through Lemonade on Strix Halo.

Use the CI workflow dispatch to run either model independently on a ref:

```bash
# MI300X / vLLM / Qwen3.6-27B
gh workflow run ci.yml --ref <ref> \
  -f platform=app-dev-gpu \
  -f include_nightly=true \
  -f name_filter='large platform-specific model'

# Strix Halo Linux / Lemonade / Qwen3.6-35B-A3B-GGUF (UD-Q4_K_XL)
gh workflow run ci.yml --ref <ref> \
  -f platform=strix-ubuntu \
  -f include_nightly=true \
  -f name_filter='large platform-specific model'

# Strix Halo Windows / Lemonade / Qwen3.6-35B-A3B-GGUF (UD-Q4_K_XL)
gh workflow run ci.yml --ref <ref> \
  -f platform=strix-windows \
  -f include_nightly=true \
  -f name_filter='large platform-specific model'
```

## From scenarios to tests

1. Write the `.feature` file with Gherkin scenarios (same words a user would use).
2. Add step functions in a `_steps.rs` file under `tests/e2e/`.
3. Add `pub mod <name>_steps;` to `tests/e2e.rs`.
4. Run with `cargo xtask e2e`.

The `.feature` file is both the spec and the test input — cucumber reads it at runtime and matches each step to a Rust function via `#[given]`/`#[when]`/`#[then]` annotations.

## Writing new tests

1. Write the Gherkin scenario in the appropriate `.feature` file (or create a new one for a new feature area).
2. Create a `_steps.rs` file if the feature area is new.
3. Implement step functions: Given = setup, When = action, Then = assertion.
4. Register the module in `tests/e2e.rs` with `pub mod <name>_steps;`.
5. Run to verify.

## Design principles

- **Black-box only.** Step functions interact with `rocm` through its CLI and HTTP endpoints. No imports from the rocm-cli codebase. Where a scenario needs the CLI to know about a running server, it plants a managed-service record as plain JSON matching the on-disk schema (see `register_mock_service`) — the same file `rocm serve --managed` would write — rather than importing the record type.
- **Isolated state.** Each scenario uses isolated config, data, and cache directories. Tests never touch `~/.rocm`.
- **Behavioral language.** Feature files describe what users care about, not implementation details. How steps are implemented (mock vs real, which port, which API) stays in the step functions.
- **OS-assigned ports.** The mock server binds to `127.0.0.1:0` to avoid port conflicts between tests.
