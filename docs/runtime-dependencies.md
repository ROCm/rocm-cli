<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Runtime dependencies

rocm-cli installs third-party runtimes: the ROCm SDK and its PyTorch stack
from AMD's wheel indexes, vLLM, and Lemonade. This page describes how their
versions are pinned, locked, checked and kept current, and why it is built
the way it is.

## Layout

Everything lives under [`runtimes/`](../runtimes/):

| File | What it holds | Written by |
|---|---|---|
| [`mise.toml`](../runtimes/mise.toml) | Binary pins (Lemonade; uv as lock tooling), lock platforms, cooldown, task entry points | People, `mise upgrade --bump` |
| [`mise.lock`](../runtimes/mise.lock) | Per platform: version, download URL, SHA-256 | `mise lock` |
| [`profiles.toml`](../runtimes/profiles.toml) | Python runtime profiles and the policy their locks must meet | People |
| [`uv.toml`](../runtimes/uv.toml) | Which indexes uv resolves from | People |
| `locks/pylock.<profile>-<os>.toml` | One PEP 751 lock per profile and OS: every package, exact URL, SHA-256 | `cargo xtask deps lock` |
| [`bin/mise`](../runtimes/bin/mise), [`bin/mise.cmd`](../runtimes/bin/mise.cmd) | Wrappers (POSIX, Windows) that download a pinned, checksum-verified mise | `mise generate install-script` (see [MANIFEST.md](../MANIFEST.md)) |

No hand-written non-test source spells Lemonade's version or archive
digests outside `runtimes/`; their build-time constants are generated from
`mise.lock`: [`rocm-deps`](../crates/rocm-deps/src/pins_gen.rs) turns
`mise.lock` into constants at build time (`LEMONADE_VERSION`,
`LEMONADE_LINUX_X64_URL`, `LEMONADE_LINUX_X64_SHA256`, …), and the Lemonade
adapter checks its download against that digest, so a version and its digests
can only move together. uv is pinned only as lock tooling, so no constants are
generated for it. The Python stacks are locked here but not yet read by an
installer: until they are, the versions vLLM installs are still also spelled
in [`wheel_builds.rs`](../engines/vllm/src/wheel_builds.rs) (see the last
section).

## Commands

All from the repository root; `runtimes/bin/mise` (on Windows,
`runtimes\bin\mise.cmd`) needs no prior install. mise finds the tasks only
with `--cd runtimes`.

| Command | Does |
|---|---|
| `runtimes/bin/mise run --cd runtimes lock` | Re-lock binaries at the current pins and every Python profile at the cutoff its lock records (network; see below) |
| `runtimes/bin/mise run --cd runtimes update` | Move pins to the newest releases past the cooldown, then re-lock |
| `cargo xtask deps check` | Verify every lock, offline (also run by `cargo test -p xtask`) |
| `cargo xtask deps diff --base <rev>` | Markdown table of versions changed since `<rev>`. Every row carries a short digest after the version (of the asset for a binary, of all its artifacts' hashes for a Python package), so an artifact re-uploaded at the same version shows up too |

The mise tasks install the pinned tools first, Lemonade included, so even
`runtimes/bin/mise run --cd runtimes check` needs the network on a fresh
machine; `cargo xtask deps check` does not.

To bump one binary by hand, edit its version in `mise.toml` and run `lock`.
To change a Python stack, edit its requirements in `profiles.toml` and run
`lock`. Commit everything under `runtimes/` together.

## Two locks, two tools

### Binaries: mise

Lemonade's embeddable build and uv are GitHub release archives, not Python
packages. mise's `github`/`aqua` backends lock them per platform with a URL
and SHA-256, apply a release-age cooldown, and come with a bootstrap wrapper,
so no binary has to be repackaged or re-hosted.

Where each digest came from is recorded in the lock, and differs by tool. For
uv, the lock records `provenance = "github-attestations"`: mise verified the
release's build attestation when it locked it. For Lemonade, which publishes
no attestation, the lock records `checksum_source = "mise-versions"`: the
digest is the one mise's own version index lists for that release asset, not
one GitHub signed. That digest is what
[`install.rs`](../engines/lemonade/src/install.rs) checks a download against,
so trusting it is a decision this repository makes: a bump of Lemonade is
reviewed in its pull request like any other pin change, and
`cargo xtask deps check` (below) ties the URL to Lemonade's own release for
that version. `deps check` does not read `checksum_source`; it does require
uv's `provenance = "github-attestations"`, ties `url_api` (which mise downloads
through when set) to the same repository as `url`, and refuses any lock field
it does not know.

Two `mise lock` behaviours are guarded by `cargo xtask deps check`, because
both produced a lock that looked fine while this was set up:

- when it cannot resolve a platform it warns, exits 0 and writes a
  version-only entry with no URL or digest;
- an `asset_pattern` shared by both platforms locked the Windows `.zip` for
  `linux-x64`. Lemonade's patterns are therefore set per platform, and the
  check refuses an asset whose file name names the wrong OS.

It also requires every locked asset to be exactly
`https://github.com/<owner>/<repo>/releases/download/<tag>/<file>` for the
repository the tool's mise backend names, with no `..`, `.` or `%`-escape in
any segment and a `<tag>` that is the locked version (with or without a `v`),
since the weekly update commits a lock produced by a job that ran upstream
code; requires `lockfile_platforms` to be exactly Linux and Windows x64 and
refuses a lock entry for any other platform; checks each pin in `mise.toml`
against the version locked for it;
and runs the `rocm-deps` generator ([`pins_gen.rs`](../crates/rocm-deps/src/pins_gen.rs))
over the lock, so a lock it accepts is one the build accepts too.

### Python stacks: uv, one lock per OS

The SDK, torch and vLLM stacks are locked with `uv pip compile` into PEP 751
`pylock.toml` files — a system description, not a Python project, so there is
no `pyproject.toml`. Each profile is locked once per OS because AMD's wheels
for one release do not declare the same dependencies everywhere: the Linux
`torch 2.13.0+rocm10.1.0` wheel requires `triton`, the Windows wheel does not,
and `triton` ships no Windows wheel. A single cross-platform lock (uv's
`uv.lock`) records `triton` unconditionally and cannot be installed on Windows.

mise's own `pypi` backend was evaluated for these stacks and cannot lock them
(mise 2026.10.25):

- it locks one cross-platform graph per tool — the `triton` problem above;
- it always resolves and installs with `--no-build`, and AMD publishes `rocm`
  only as an sdist; such graphs fall back to a version-only entry by design
  ([jdx/mise#15029](https://github.com/jdx/mise/issues/15029));
- dependencies without an upload time are dropped by its cooldown by design
  ([jdx/mise#15146](https://github.com/jdx/mise/issues/15146));
- it takes one registry plus PyPI per tool, where the vLLM stack needs two
  AMD indexes ([jdx/mise#15145](https://github.com/jdx/mise/issues/15145));
- it derives a registry's index URL as `<root>/simple/`, which AMD's indexes
  do not serve.

### What `deps check` enforces on a Python lock

- it starts with the header `cargo xtask deps lock` writes (no hand edits);
- that header's inputs digest matches the profile's `python`, platform and
  requirements and the text of `uv.toml` as they are now, so a lock left stale
  by an edit to any of them is caught offline, even where it still happens to
  satisfy the requirements;
- that header records the `exclude-newer` cutoff the lock was resolved at,
  which is what lets a re-lock reproduce it (AMD-only packages aside, see
  *Cooldown*);
- every requirement in its profile is locked at a version that satisfies it;
- every artifact is an `https` URL with a SHA-256, served by PyPI
  (`files.pythonhosted.org`) or a `vendor-hosts` index, so a download the
  update repoints elsewhere is refused even though the summary shows no URLs
  or hosts;
- every package in `vendor-packages` comes from a `vendor-hosts` index, so a
  ROCm build missing from AMD's index is never replaced by a same-named
  package from PyPI;
- every `+rocm…` build belongs to the profile's ROCm line (vLLM's own range
  otherwise pulls a `+rocm10.0.0` torchvision into a 10.1 stack);
- every wheel's platform tag fits the lock's OS (x86-64 Linux, or `win_amd64`),
  since the Windows locks are resolved on a Linux runner;
- each package is locked once, with a version and at least one artifact;
- every lock a profile declares exists, and `runtimes/locks/` holds nothing
  else: the directory (fixed, not configurable) belongs to the tooling, so
  `deps lock` removes every entry there no profile produces — dotfiles and
  subdirectories included — `deps check` reports the same set, and the weekly
  update commits only locks.

[`runtimes/uv.toml`](../runtimes/uv.toml) resolves with
`unsafe-first-match`: AMD's indexes mirror a few common packages (setuptools,
numpy) at one version each, and uv's default `first-index` would fail on a
requirement those mirrors cannot meet instead of falling through to PyPI. The
vendor-host rule above closes the substitution risk that strategy opens.

## AMD's indexes publish no hashes

`stable.repo.amd.com` and `rocm.frameworks.amd.com` list wheels without
`#sha256=` fragments; TheRock's index generator
([`generate_s3_index.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/packaging/python/generate_s3_index.py))
writes bare links. To record a
hash uv downloads each wheel. For the all-arch ROCm 10.1 SDK profile that is
about 8 GiB of per-GPU device wheels per OS, on every re-lock. Consequences:

- locks are generated on GitHub-hosted runners: on a pull request that
  changes `runtimes/` or the `deps` tooling, the `runtime-locks` CI job runs
  `runtimes/bin/mise run --cd runtimes lock`, and when the result differs from
  what is committed it fails and uploads the re-locked files as the
  `runtime-locks` artifact, so a new or stale lock is committed from that
  artifact rather than re-downloaded locally. It does not run on pushes to
  `main` or in the merge queue. There the gate is the offline `cargo xtask deps
  check`, which also catches a pin edited in `mise.toml` without re-locking, and
  the weekly update below regenerates the locks;
- the cooldown cannot judge AMD's packages (see below).

Publishing PEP 503 hash fragments in those indexes would remove the
downloads entirely.

## Cooldown

Like [`.github/dependabot.yml`](../.github/dependabot.yml), updates wait 7
days so a bad release can be found and pulled first. mise applies it to
binaries (`minimum_release_age`). For Python locks, every lock is resolved at
an `--exclude-newer` cutoff and records it in its header
(`# exclude-newer: <time>`). `update` runs `deps lock --upgrade
--cooldown-days 7`, which sets a new cutoff 7 days before now; `lock` resolves
each lock again at the cutoff it records, so it reproduces the lock unless an
input changed or AMD published a new build of an AMD-only package (those are
exempt from the cutoff, below, and several are pinned as ranges such as
`torch==2.13.*`), and a lock that does not exist yet gets a cutoff 7 days
before now. A resolution that differs from the committed lock only in its
cutoff keeps the committed file unchanged, so a weekly update with nothing
new proposes nothing. The previous lock is not handed to uv as version preferences: with these
indexes that does not converge, as successive runs alternate between AMD's
mirrored version of a package and PyPI's. AMD's indexes publish no
upload times and uv treats an undated file as too new, so the AMD-only
projects they serve — those matching `vendor-packages` in `profiles.toml` — are
exempted (`--exclude-newer-package <name>=false`). Everything else, including
the common packages AMD's indexes mirror from PyPI, stays under the cooldown.
An AMD-only dependency missing from `vendor-packages` makes the update fail
rather than slip past the cooldown; add it to the policy, which `deps check`
then also enforces.

## Keeping current

[`runtime-deps-update.yml`](../.github/workflows/runtime-deps-update.yml) runs
weekly and on demand. It runs the mise `update` task, and if anything moved opens
or refreshes one pull request on `runtime-deps/update` with a table from
`deps diff`. Resolving runs upstream code (uv builds the `rocm` sdist for its
metadata), so that job holds a read-only token. A second read-only job runs
the base commit's validator and `cargo xtask deps check` against the
artifact, and recomputes the pull request's table with `deps diff`, refusing
an artifact whose summary differs; only then does a third, holding the write token, validate it again
and commit through `createCommitOnBranch`, which GitHub signs, as the
`commit-signatures` gate requires. A pull request opened with
`GITHUB_TOKEN` does not trigger CI: close and reopen it to run the checks.

**Prerequisite:** the repository setting "Allow GitHub Actions to create and
approve pull requests" (Settings → Actions → General) must be on. While it is
off, the update commits to `runtime-deps/update` and then fails at
`gh pr create`; open the pull request from that branch by hand, or turn the
setting on.

The update keeps each Python requirement within its range (`torch==2.13.*`
moves to a newer 2.13 build, not to 2.14). Widening a range — a new torch
minor, a new ROCm line — is a reviewed edit to `profiles.toml`, as a major
version bump is under Dependabot.

Dependabot keeps Cargo, Actions and pre-commit current. It cannot take this
over: it does not read `mise.toml`
([dependabot-core#12320](https://github.com/dependabot/dependabot-core/issues/12320)),
its `uv` support reads only `pyproject.toml` + `uv.lock`, and it has open bugs
with explicit and per-package indexes
([dependabot-core#13094](https://github.com/dependabot/dependabot-core/issues/13094),
[#14397](https://github.com/dependabot/dependabot-core/issues/14397)).
Renovate reads `mise.toml` but needs an organisation-level app install.

## Bootstrapping a profile's first lock

A new profile has no committed lock until one is generated, and `cargo test -p xtask` (which runs `deps check`) fails
with "missing or unreadable" until then. Generate it with
`runtimes/bin/mise run --cd runtimes lock`, or take it from the
`runtime-locks` artifact the CI job of the same name uploads when it fails,
and commit it in the same pull request as the profile.

## What reads the locks today

Lemonade only: the Lemonade adapter takes its version and the archive digest
it checks from the constants generated from `mise.lock`, and derives its
download URLs from that version (a `rocm-deps` test ties the derived URLs to
the locked ones); `rocm engines` takes the version from them too. The Python
locks are not yet read by any installer. They are the target for these
follow-ups, each its own change with its own scenarios:

1. `rocm install sdk` (release channel) installs from the SDK lock for the
   host's arch instead of resolving at install time.
2. vLLM installs from its lock, replacing the tables in
   [`wheel_builds.rs`](../engines/vllm/src/wheel_builds.rs).
3. uv itself is downloaded at the version and digest in `mise.lock` instead
   of `latest`.
4. ComfyUI and the llama.cpp backend Lemonade aligns move into `mise.toml`.
5. Nightly channels stay unlocked: their indexes delete old builds, so they
   resolve at install time against the profile's ranges, and the resolved
   lock is kept with the install.

GPU drivers stay outside: kernel modules and the Windows driver installer are
system state that needs root, not a per-user runtime. The WSL user-space
library `librocdxg` may not need a pin of its own: `rocm-sdk-core` 7.14.1
ships `librocdxg.so.1`, but the runtime opens `librocdxg.so`, so whether that
copy is used without the separate package is unverified.
