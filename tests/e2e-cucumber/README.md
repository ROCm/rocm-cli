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
│
├── features/                     # .feature files — one per feature area
│   ├── chat.feature
│   └── model_serving.feature
│
├── tests/                        # test binary + step modules
│   ├── e2e.rs                    # World struct, runner, Drop cleanup
│   └── e2e/                      # step functions — one file per feature area
│       ├── chat_steps.rs
│       └── serving_steps.rs
│
└── src/                          # shared test infrastructure
    ├── lib.rs
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

# Skip known-bug scenarios:
cargo xtask e2e -- -t "not @expected-failure-EAI-*"

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

## Tags

Cucumber tag filters match exact tag names (no globbing), so scenarios carry a
bare category tag for filtering plus an optional specific tag for traceability.

| Tag | Meaning |
|---|---|
| `@gpu` | Requires real AMD GPU / ROCm hardware. Runs only in the non-blocking `e2e-gpu` CI job; excluded from the mock-tier job via `-t "not @gpu"`. |
| `@expected-failure` | Known bug — excluded from the blocking mock job (`-t "not @expected-failure"`) and run in the non-blocking known-bugs job. |
| `@expected-failure-EAI-NNNN` | Traceability to the specific tracked bug. Always paired with the bare `@expected-failure`. Remove both when the bug is fixed. |

CI runs three selections:

| Job | Filter | Blocking |
|---|---|---|
| `e2e` | `not @gpu and not @expected-failure` | yes |
| `e2e-known-bugs` | `@expected-failure and not @gpu` | no |
| `e2e-gpu` | `@gpu` | no |

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
