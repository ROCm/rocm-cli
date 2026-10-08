<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Contributing to ROCm CLI

Thank you for your interest in contributing. This page covers how to get started, how work is tracked, and what to expect from the review process.

## Code of Conduct

By participating in this project you agree to abide by the [Code of Conduct](https://github.com/ROCm/rocm-cli/blob/main/CODE_OF_CONDUCT.md).

## Issue tracking

Bug reports and feature requests are tracked as [GitHub Issues](https://github.com/ROCm/rocm-cli/issues). Before opening a new issue, search existing ones to avoid duplicates.

## Branch and commit naming

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
feat: add support for vLLM multi-GPU serving
fix: correct VRAM probe when amd-smi is absent
docs: clarify GPU selection flag behavior
chore: bump rust-toolchain to 1.96
```

Branch names should be short and descriptive:

```
feat/vllm-multi-gpu
fix/vram-probe-fallback
```

## Development setup

Prerequisites: Rust (see `rust-toolchain.toml` for the pinned version) and [uv](https://github.com/astral-sh/uv), which installs `prek` and runs the scripts.

```bash
git clone https://github.com/ROCm/rocm-cli
cd rocm-cli
uv tool install prek        # alternative: cargo install --locked prek
prek install                # fast checks on every commit
prek install -t pre-push    # heavier checks on push (clippy + tests)
```

`prek` runs the same checks locally that CI enforces: `cargo fmt`, `clippy`, `cargo test`, `ruff` (Python), `shellcheck` (shell), PowerShell syntax, and the generated manifests (`MANIFEST.md`, `THIRD_PARTY_NOTICES.txt`).

The manifest hooks only run when you change the dependency graph. They *rewrite* the generated file instead of only reporting that it is stale. When that happens, the commit stops so you can re-stage the refreshed file.

The `THIRD_PARTY_NOTICES.txt` hook also needs the pinned generator. Without it, the hook skips and CI remains the gate:

```bash
cargo install cargo-about@0.9.1 --locked --features cli   # optional, for THIRD_PARTY_NOTICES.txt
```

The license-headers hook (`hawkeye`) runs on both commit and push for matching files. `prek` runs it like the other hooks but doesn't install the `hawkeye` binary. Unlike the manifest hooks, this hook fails instead of skipping when the binary is missing. Install it with:

```bash
cargo install hawkeye@7.0.0 --locked   # pinned to match the CI license-headers job
```

### Workspace layout

| Path | Description |
| --- | --- |
| `apps/rocm` | Main CLI binary |
| `apps/rocmd` | Background daemon |
| `crates/rocm-core` | Core library |
| `crates/rocm-dash-*` | Dashboard terminal user interface (TUI) libraries |
| `crates/rocm-engine-protocol` | Engine IPC protocol |
| `engines/` | Inference engine adapters (lemonade, vllm) |

### Module organization

Put new subcommands and subsystems in their own file from the start. Don't let them grow inside `main.rs` or `lib.rs` while waiting for a later extraction pass.

See [docs/architecture.md](https://github.com/ROCm/rocm-cli/blob/main/docs/architecture.md) for the two extraction patterns in use, the current module map, and the full module-organization convention. Its path citations are relative Markdown links. The `docs-links` job described below checks that they resolve, but not that the prose is accurate.

Crate-layering invariants, such as `rocmd` never depending on `rocm`, are enforced by `cargo xtask check-crate-edges` (`xtask/src/crate_edges.rs`).

The `docs-links` CI job checks every local (relative-path) Markdown link and `#anchor` fragment in the repository. It skips `docs/rocm-docs/`, which the `docs-build` job covers instead. The job runs offline, so it doesn't catch broken external `https://` links. See `lychee.toml` for the current exclusions.

### Test commands

| Component | Command |
| --- | --- |
| Rust (all crates) | `cargo test` |
| Lint + format check | `prek run --all-files` |

See the [test guide](https://github.com/ROCm/rocm-cli/blob/main/docs/testing.md) for the full set of tests and the [manual testing guide](https://github.com/ROCm/rocm-cli/blob/main/docs/manual-testing.md) for manual QA steps.

## Making changes

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes.** Keep commits focused, with one logical change per commit.
3. **Add or update tests** for any new behavior.
4. **Run the relevant test suite** before opening a PR.
5. **Open a pull request** against `main` with a clear title and description following the Conventional Commits format.

### Commit signing and sign-off

Commits must be both cryptographically **signed** and carry a Developer Certificate of Origin (DCO) **`Signed-off-by`** trailer. Use `git commit -s` to add the trailer automatically. This is enforced by the prek hooks and by a blocking CI check.

Enable SSH signing once with:

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

See [Commit signatures](https://github.com/ROCm/rocm-cli/blob/main/docs/commit-signatures.md) for GPG signing, GitHub "Verified" status, and troubleshooting.

### What reviewers look for

- Tests cover the new behavior
- No secrets, credentials, or internal hostnames in committed files
- Third-party dependencies recorded in `Cargo.lock`
- License headers on new source files (see `licenserc.toml`)

## Reporting security issues

Do **not** open a public GitHub Issue for security vulnerabilities. See [SECURITY.md](https://github.com/ROCm/rocm-cli/blob/main/SECURITY.md) for the responsible disclosure process.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](https://github.com/ROCm/rocm-cli/blob/main/LICENSE.TXT).
