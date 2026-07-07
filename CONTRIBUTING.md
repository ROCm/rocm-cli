<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Contributing to ROCm CLI

Thank you for your interest in contributing. This document explains how to get started, how work is tracked, and what to expect from the review process.

## Code of Conduct

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Issue Tracking

Bug reports and feature requests are tracked as [GitHub Issues](../../issues). Before opening a new issue, search existing ones to avoid duplicates.

## Branch and Commit Naming

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

## Development Setup

Prerequisites: Rust (see `rust-toolchain.toml` for the pinned version) and [uv](https://github.com/astral-sh/uv) (for prek and scripts).

```bash
git clone https://github.com/ROCm/rocm-cli
cd rocm-cli
uv tool install prek        # or: cargo install --locked prek
prek install                # fast checks on every commit
prek install -t pre-push    # heavier checks on push (clippy + tests)
```

`prek` runs the same checks locally that CI enforces: `cargo fmt`, `clippy`, `cargo test`, `ruff` (Python), `shellcheck` (shell), and PowerShell syntax.

### Workspace layout

| Path | Description |
| --- | --- |
| `apps/rocm` | Main CLI binary |
| `apps/rocmd` | Background daemon |
| `crates/rocm-core` | Core library |
| `crates/rocm-dash-*` | Dashboard TUI libraries |
| `crates/rocm-engine-protocol` | Engine IPC protocol |
| `engines/` | Inference engine adapters (lemonade, vllm) |

### Test commands

| Component | Command |
| --- | --- |
| Rust (all crates) | `cargo test` |
| Lint + format check | `prek run --all-files` |

See `docs/testing.md` for the full test guide and `docs/manual-testing.md` for manual QA steps.

## Making Changes

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes.** Keep commits focused — one logical change per commit.
3. **Add or update tests** for any new behavior.
4. **Run the relevant test suite** before opening a PR.
5. **Open a pull request** against `main` with a clear title and description following the Conventional Commits format.

### Commit signing and sign-off

Commits must be both cryptographically **signed** and carry a DCO **`Signed-off-by`** trailer. Use `git commit -s` to add the trailer automatically. This is enforced by the prek hooks and by a blocking CI check.

Enable SSH signing once with:

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

See `docs/commit-signatures.md` for GPG signing, GitHub "Verified" status, and troubleshooting.

### What reviewers look for

- Tests cover the new behavior
- No secrets, credentials, or internal hostnames in committed files
- Third-party dependencies declared in `Cargo.lock`; license headers present on new source files (see `licenserc.toml`)

## Reporting Security Issues

Do **not** open a public GitHub Issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE.TXT).
