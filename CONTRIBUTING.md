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

```text
feat: add support for vLLM multi-GPU serving
fix: correct VRAM probe when amd-smi is absent
docs: clarify GPU selection flag behavior
chore: bump rust-toolchain to 1.96
```

Branch names should be short and descriptive:

```text
feat/vllm-multi-gpu
fix/vram-probe-fallback
```

## Development setup

Prerequisites: Rust (see `rust-toolchain.toml` for the pinned version) and [uv](https://github.com/astral-sh/uv) (for `prek` and the scripts).

```bash
git clone https://github.com/ROCm/rocm-cli
cd rocm-cli
uv tool install prek        # alternative: cargo install --locked prek
prek install                # fast checks on every commit
prek install -t pre-push    # heavier checks on push (clippy + tests of the crates the push affects)
```

`prek` runs the same checks locally that CI enforces: `cargo fmt`, `clippy`, `cargo test`, `ruff` (Python), `shellcheck` (shell), PowerShell syntax, license headers (`hawkeye`), markdown links (`lychee`), the module map (`cargo xtask check-architecture-doc`), and the generated files (`MANIFEST.md`, `THIRD_PARTY_NOTICES.txt`, and the published Doctor catalog, `doctor-catalog.json`).

The pre-push hooks (`cargo clippy`, the e2e harness clippy, and `cargo test`) check only the crates your push affects: the crates whose files changed plus every crate that depends on them, selected the way CI's `cargo xtask affected` does. Each step prints the scope it chose and why.

The hooks check the whole workspace when:

- A change cannot be confined to specific crates (the lockfile, the root manifest, the toolchain file, `clippy.toml`, workflow files, a doc an xtask test reads, or a path no crate owns).
- The affected crates cannot be determined at all, for example because the remote branch tip is not fetched locally or the pushed range lists no changed files.

The hooks also add the crates whose tests read files from outside their own crate, as CI's selection does:

- Any Rust file adds `xtask`, whose contract tests scan every source file.
- `README.md`, `docs/`, and `skills/` add `rocm` and `xtask`.
- `skills/rocm-doctor/` adds `e2e-cucumber`, which the hooks only lint. The hook's `cargo test` never includes `e2e-cucumber`, so the hook never runs its `skill_reference` test. CI runs it in the Linux test job and in the Windows debug job, whose build and test steps run for any change under `skills/`.

A push that changes no Rust file runs none of the cargo steps, whatever else it changes (a lockfile-only push included), except that a change to `clippy.toml` runs the two clippy steps.

A step that already passed on this exact content is skipped, and says so. After a green run, the step records a stamp under the clone's `.git` directory, where every worktree shares it. The stamp is keyed on:

- The git tree hash of `HEAD`
- The exact cargo command, so a pass for a narrow selection never covers a wider one
- The toolchain
- The `CARGO_*` and `RUST*` environment

A retried push, a manual run before the push, or an amend that only rewords the message therefore does not pay for the same checks twice.

Stamps are used and written only when the working tree is clean, because cargo checks the working tree, not the pushed commit, and an untracked file can hide a missing `git add`. Stamps expire after seven days. They do not cover your user-level cargo configuration, installed tools, files git ignores, or flaky tests.

A skip shows that this tree passed those checks earlier, not that they ran during this push. The `cargo test --workspace` and `cargo clippy --workspace` commands you run by hand are unaffected. When you want proof that the checks ran for this push, run `cargo xtask pre-push-check <step> --rerun`, or set `ROCM_PRE_PUSH_RERUN=1` for the hook. Either one ignores the stamps, drops this step's earlier pass so that a failed rerun cannot leave it to skip the next run, and records a fresh pass when the step succeeds. `SKIP=<hook id>` still skips a hook outright, and the commit-signature hook runs on every push.

A green hook is an early warning, not the gate. On pull requests, CI lints the whole workspace, but its Linux test job runs only the affected crates. The whole workspace's tests run in the Windows debug job when the change matches CI's `heavy` category, in the merge queue, and on `main`. Any Rust file matches `heavy`. A change only to prose docs such as `README.md` or `docs/*.md`, outside `docs/keys/`, does not. To see what a push would check without compiling anything, run the following command (the steps are `clippy`, `clippy-e2e`, and `test`):

```bash
cargo xtask pre-push-check test --dry-run --base origin/main
```

The manifest hooks only run when you change the dependency graph. They *rewrite* the generated file instead of only reporting that it is stale. When that happens, the commit stops so you can re-stage the refreshed file.

The `THIRD_PARTY_NOTICES.txt` hook also needs the pinned generator. Without it, the hook skips and CI remains the gate for that hook, but the AGENTS.md section 8 gate needs it:

```bash
cargo install cargo-about@0.9.1 --locked --features cli   # for THIRD_PARTY_NOTICES.txt and `cargo xtask tpn --check`
```

The license-headers hook (`hawkeye`) runs on every commit and push, whatever is staged. It has no file-type filter, because `hawkeye` always scans the whole working tree per `licenserc.toml`, including untracked files that aren't gitignored. An unrelated unheadered file, even one not yet committed, can therefore block an otherwise unrelated commit. A clean CI checkout never has untracked files, so this hook can be stricter locally than CI is. Both the hook and the CI job run `hawkeye check --fail-on-unknown`, so an included file whose type has no comment-style rule, built into hawkeye or in `licenserc.toml`, is still listed as `unsupported` but now fails the check, where without the flag it would pass with its header unchecked; an included file that is not UTF-8, such as a UTF-16 `.ps1` saved by Windows PowerShell 5.1, is listed and fails the same way, and locally that includes untracked files.

`prek` runs the hook like the others but doesn't provision the `hawkeye` binary. Unlike the manifest hooks above, it fails hard (not just skips) when the binary is missing. It also doesn't check that an installed binary is the pinned version. A mismatched version might apply different license-header rules than CI, or might reject `licenserc.toml` outright. For example, `hawkeye` 6.5.1, the last 6.x release, cannot parse this repository's config, because it rejects unknown fields and the `[header]`, `[files]` and `[[rules]]` tables are all unknown to it, which blocks every commit and push, not just ones touching code. Install the pinned version:

```bash
cargo install hawkeye@7.0.0 --locked   # pinned to match the CI license-headers job
```

To commit or push without it (for example, while iterating without the binary installed, or on a version it rejects), skip it explicitly: `SKIP=license-headers git commit ...` or `SKIP=license-headers git push`. CI's `license-headers` job still enforces the check either way.

The Markdown-links hook (`lychee`) runs on every commit, over the whole repository instead of just the changed files, because moving or deleting a file can break a link in a Markdown file you didn't touch. It checks your working tree, so a link to a new file you haven't `git add`ed passes locally and still fails in CI. The hook runs `cargo xtask lychee --if-available`. `prek` doesn't provision the binary; without it the hook prints a warning and passes, leaving the `docs-links` CI job as the gate, and a version other than the pinned one runs with a warning:

```bash
cargo install lychee@0.24.2 --locked   # pinned to match the CI docs-links job (lycheeVersion)
```

### Workspace layout

The workspace contains these top-level components:

| Path | Description |
| --- | --- |
| `apps/rocm` | Main CLI binary |
| `apps/rocmd` | Background daemon |
| `crates/rocm-core` | Core library |
| `crates/rocm-dash-*` | Dashboard terminal user interface (TUI) libraries |
| `crates/rocm-engine-protocol` | Engine inter-process communication (IPC) protocol |
| `engines/` | Inference engine adapters (lemonade, vllm, aim) |

### Module organization

New subcommands and subsystems default to their own file from day one. Don't let them grow inside `main.rs`/`lib.rs` waiting for a future extraction pass. See [`docs/architecture.md`](https://github.com/ROCm/rocm-cli/blob/main/docs/architecture.md) for the two extraction patterns in use, the current module map, and the module-organization convention in full.

The `docs-links` job below checks the file links in that document, including every concrete file citation in the per-crate inventories. It checks that they resolve, not that the prose is accurate. Every production source file must also appear in it (see below). Its section headings and directory names stay bare and unchecked.

Crate-layering invariants, such as `rocmd` never depending on `rocm`, are enforced by `cargo xtask check-crate-edges` (`xtask/src/crate_edges.rs`). File size is checked by `cargo xtask check-module-size` (`xtask/src/module_size.rs`): no source file may exceed 1,500 production lines, test code excluded. A change that would cross the limit splits the file in the same change instead. The `clippy` CI job fails on a file over the limit. The `cargo-module-size` pre-commit hook is the local mirror of that CI step: it runs the same check and refuses the commit (it only runs if you installed the hooks).

The hook measures the working tree, so it also fails on a Rust file under a crate's `src/` that no `mod` declares, even an untracked scratch file. It also refuses every intermediate commit of a split spread over several commits; commit those steps with `SKIP=cargo-module-size git commit ...`. `docs/architecture.md` has the details.

Every production source file, meaning a `.rs` file under `apps/*/src`, `crates/*/src` or `engines/*/src`, must have an entry in `docs/architecture.md`. So must every `.rs` file under `xtask/src`, `tests/e2e-cucumber/src` and `tests/e2e-cucumber/tests`. Out of scope are `build.rs` files and the `tests/` and `examples/` trees of the product crates. Test-only helpers (`test_support.rs`, `test_env.rs`, `*_proptests.rs`, `*_props.rs`, `*_properties.rs`) of those product crates need no entry; the `xtask` and `tests/e2e-cucumber` trees have no such exemption.

An entry is a relative markdown link to the file, and it has to render as a link. A path inside an HTML comment, a code block, or a code span does not count, and neither does a malformed link (such as one missing its `)`) or an image. Use an inline `[text](path)` link with its text on one line; reference-style links, a `?query` suffix, and root-relative paths are not recognized.

`cargo xtask check-architecture-doc` (`xtask/src/architecture_doc.rs`) enforces this in the `clippy` CI job and in the `architecture-doc` prek hook, which runs on every commit and every push, over the whole tree. A file with no entry fails it, and there is no exemption list. This is the opposite direction to the `lychee` check below, which only confirms that links already in the doc still resolve. The check does not read the prose of an entry, so keeping a description true when a file's role changes is still up to you.

Every local (relative-path) Markdown link and `#anchor` fragment in the repository (outside `docs/rocm-docs/`, which `docs-build` covers instead, and hidden directories such as `.github/`, which neither check walks) is checked by the `docs-links` CI job and by the `lychee` prek hook above, both configured by `lychee.toml` (which lists the current exclusions). Both run offline only, so they don't catch broken external `https://` links.

Because only links are checked, cite a specific repository file as a link relative to the citing file, `[fix.rs](../crates/rocm-core/src/fix.rs)` from `docs/`, instead of a bare backtick path, which nothing checks. The full rule, including the link forms that fail the check and the places it doesn't apply (`.github/`, links leaving a `skills/` folder, and `docs/rocm-docs/` along with the files it includes, which are this file and README.md), is in [AGENTS.md](https://github.com/ROCm/rocm-cli/blob/main/AGENTS.md) section 5 ("Investigate rocm-cli Before Editing"); it applies to citations you add or edit.

### Updating the AIM catalog

AIM model entries are hand-maintained in `crates/rocm-core/src/model_catalog.json`; see [`docs/aim-catalog.md`](https://github.com/ROCm/rocm-cli/blob/main/docs/aim-catalog.md) for the refresh process and staleness warning.

### Test commands

Run the tests with these commands:

| Component | Command |
| --- | --- |
| Rust (all crates) | `cargo test` |
| Lint and format check | `prek run --all-files` |

See the [test guide](https://github.com/ROCm/rocm-cli/blob/main/docs/testing.md) for the full set of tests and the [manual testing guide](https://github.com/ROCm/rocm-cli/blob/main/docs/manual-testing.md) for manual QA steps. [AGENTS.md section 8](https://github.com/ROCm/rocm-cli/blob/main/AGENTS.md) has the minimum quality gate; this table lists only some of its commands.

## Making changes

To contribute a change:

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes.** Keep commits focused, with one logical change per commit.
3. **Add or update tests** for any new behavior.
4. **Run at least the minimum quality gate** ([AGENTS.md section 8](https://github.com/ROCm/rocm-cli/blob/main/AGENTS.md), "Verification Matrix For This Repo"), plus the suites relevant to your change, before opening a PR.
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

### Commit messages are public history

Pull requests are normally squash-merged, and by default the squash commit's message is built from the messages of the branch's commits; rebase-merge is also enabled, and lands each commit with its own message. Either way, once merged the messages are public history that is not rewritten, so commit messages are reviewed for accuracy on the same footing as the diff: each describes what its own commit does in words still true at the branch head, and one whose words a later commit on the branch made misleading is corrected before the pull request merges. AGENTS.md section 5 states the rule for contributors and section 11 for reviewers.

### What reviewers look for

Reviewers look for the following:

- Tests cover the new behavior
- No secrets, credentials, or internal hostnames in committed files
- Third-party dependencies recorded in `Cargo.lock`
- License headers on new source files (see `licenserc.toml`)
- Comments and documentation a pull request adds or changes that state how the code behaves are checked against that code, and one that is the reason for a decision names the file and the function, constant or test that makes it true, as AGENTS.md section 3 states
- A fix, guard or hardening applied at one place is checked against the other sites that share its defect, as AGENTS.md section 3 states
- A blocking objection resting on a timing, ordering, concurrency or "this cannot happen" argument names the mechanism it assumes and where that mechanism is defined, and one whose mechanism cannot be located is raised as a question instead, as AGENTS.md section 11 states
- A revision answering review feedback is re-reviewed as a whole change, and a commit in it that answers the feedback widens no documented user-facing option's meaning or authority, as AGENTS.md section 11 states
- An existing end-to-end scenario moved off the blocking mock `e2e` job (for example by a tag) is reviewed like deleting the scenario, as AGENTS.md section 8 states

## Reporting security issues

Do **not** open a public GitHub Issue for security vulnerabilities. See [SECURITY.md](https://github.com/ROCm/rocm-cli/blob/main/SECURITY.md) for the responsible disclosure process.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](https://github.com/ROCm/rocm-cli/blob/main/LICENSE.TXT).
