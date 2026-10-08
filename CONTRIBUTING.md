<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Contributing to ROCm CLI

Thank you for your interest in contributing. This document explains how to get started, how work is tracked, and what to expect from the review process.

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

Prerequisites: Rust (see `rust-toolchain.toml` for the pinned version) and [uv](https://github.com/astral-sh/uv) (for prek and scripts).

```bash
git clone https://github.com/ROCm/rocm-cli
cd rocm-cli
uv tool install prek        # or: cargo install --locked prek
prek install                # fast checks on every commit
prek install -t pre-push    # heavier checks on push (clippy + tests)
```

`prek` runs the same checks locally that CI enforces: `cargo fmt`, `clippy`, `cargo test`, `ruff` (Python), `shellcheck` (shell), PowerShell syntax, license headers (`hawkeye`), markdown links (`lychee`), and the generated manifests (`MANIFEST.md`, `THIRD_PARTY_NOTICES.txt`).

The manifest hooks only run when you change the dependency graph, and they *rewrite* the generated file rather than just reporting it stale — when that happens the commit stops so you can re-stage the refreshed file. `THIRD_PARTY_NOTICES.txt` additionally needs the pinned generator; without it that hook skips and CI remains the gate:

```bash
cargo install cargo-about@0.9.1 --locked --features cli   # optional, for THIRD_PARTY_NOTICES.txt
```

The license-headers hook (`hawkeye`) runs on every commit and push, whatever is staged — it has no file-type filter, because hawkeye always scans the whole working tree per `licenserc.toml` regardless of which files changed or are staged, including untracked files that aren't gitignored, so an unrelated unheadered file (even one not yet committed) can block an otherwise-unrelated commit. A clean CI checkout never has untracked files, so this hook can be stricter locally than CI is. prek runs it like the others, but doesn't provision the `hawkeye` binary — and, unlike the manifest hooks above, it fails hard (not just skips) when the binary is missing. It does not check that an installed binary is the pinned version; a mismatched version may run whatever that version's own license-header rules happen to be, which may disagree with CI, or may reject `licenserc.toml` outright — for example, hawkeye 6.x rejects this repo's config with `unknown field 'files'`, which blocks every commit and push, not just ones touching code:

```bash
cargo install hawkeye@7.0.0 --locked   # pinned to match the CI license-headers job
```

To commit or push without it (for example, while iterating without the binary installed, or on a version it rejects), skip it explicitly: `SKIP=license-headers git commit ...` or `SKIP=license-headers git push`. CI's `license-headers` job still enforces the check either way.

The markdown-links hook (`lychee`) runs on every commit, over the whole repo rather than just the changed files, since moving or deleting a file can break a link in a markdown file you didn't touch. It checks your working tree, so a link to a new file you haven't `git add`ed passes locally and still fails in CI. Like `hawkeye`, prek doesn't provision the binary and the hook fails hard when it's missing:

```bash
cargo install lychee@0.24.2 --locked   # pinned to match the CI docs-links job (lycheeVersion)
```

### Workspace layout

| Path | Description |
| --- | --- |
| `apps/rocm` | Main CLI binary |
| `apps/rocmd` | Background daemon |
| `crates/rocm-core` | Core library |
| `crates/rocm-dash-*` | Dashboard TUI libraries |
| `crates/rocm-engine-protocol` | Engine IPC protocol |
| `engines/` | Inference engine adapters (lemonade, vllm) |

### Module organization

New subcommands and subsystems default to their own file from day one — don't let them grow inside `main.rs`/`lib.rs` waiting for a future extraction pass. See `docs/architecture.md` for the two extraction patterns in use, the current module map, and the module-organization convention in full — its path citations are relative markdown links, checked for resolution (not prose accuracy) by the `docs-links` job below.

Crate-layering invariants (e.g. `rocmd` must never depend on `rocm`) are enforced by `cargo xtask check-crate-edges` (`xtask/src/crate_edges.rs`).

Every local (relative-path) markdown link and `#anchor` fragment in the repo — outside `docs/rocm-docs/`, which `docs-build` covers instead, and hidden directories such as `.github/`, which neither check walks — is checked by the `docs-links` CI job and by the `lychee` prek hook above, both configured by `lychee.toml` (which lists the current exclusions). Both run offline only, so they don't catch broken external `https://` links.

Since only links are checked, cite a specific repo file as a link relative to the citing file — `[fix.rs](../crates/rocm-core/src/fix.rs)` from `docs/` — rather than a bare backtick path, which nothing checks. The full rule, including the link forms that fail the check and the places it doesn't apply (`.github/`, `docs/rocm-docs/` and the files it includes — this one and README.md — and links leaving a `skills/` folder), is in AGENTS.md §5 ("Investigate rocm-cli Before Editing"); it applies to citations you add or edit.

### Test commands

| Component | Command |
| --- | --- |
| Rust (all crates) | `cargo test` |
| Lint + format check | `prek run --all-files` |

See `docs/testing.md` for the full test guide and `docs/manual-testing.md` for manual QA steps.

## Making changes

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes.** Keep commits focused — one logical change per commit.
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

See `docs/commit-signatures.md` for GPG signing, GitHub "Verified" status, and troubleshooting.

### What reviewers look for

- Tests cover the new behavior
- No secrets, credentials, or internal hostnames in committed files
- Third-party dependencies declared in `Cargo.lock`; license headers present on new source files (see `licenserc.toml`)

## Reporting security issues

Do **not** open a public GitHub Issue for security vulnerabilities. See [SECURITY.md](https://github.com/ROCm/rocm-cli/blob/main/SECURITY.md) for the responsible disclosure process.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](https://github.com/ROCm/rocm-cli/blob/main/LICENSE.TXT).
