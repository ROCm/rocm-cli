# WIP checkpoint: download/extraction spinner PTY coverage (issue #368)

This file is a checkpoint of the approved implementation plan for finishing
this worktree's WIP (paced download server, tarball PTY scenario) and adding
the missing ComfyUI half, per issue #368. It is not meant to ship as part of
the final PR — delete it once the work lands on a proper branch off `main`
(see "Implementation approach" step 1 below).

---

# Review of issue #368's plan + refined implementation plan

## Context

Issue #368 proposes a 4-step plan to make the download/extraction spinner
(`apps/rocm/src/cli_progress.rs`) observable in E2E tests: (1) a URL-override
test hook gated behind the `e2e-test-hooks` Cargo feature, (2) a local fixture
HTTP server, (3) driving the real install paths under a PTY, (4) new Gherkin
scenarios. The user asked me to review that plan specifically for dead code
and test-coverage sufficiency, and to fold BDD scenarios into the result.

Investigation turned up a directly relevant fact: an **uncommitted worktree**
(`.claude/worktrees/e2e-download-spinner-pty`, branch
`worktree-e2e-download-spinner-pty`) already implements ~90% of the tarball
half of this exact plan — a paced fixture HTTP server, a real gzip tarball
fixture that defeats compression, the PTY scenario itself, and the necessary
test-hook plumbing. It is unregistered (fails `feature_naming.rs`'s key
check) and incomplete (no ComfyUI half), but its design is sound and directly
reusable. The plan below is built around finishing and reconciling this WIP
rather than re-deriving it from scratch.

(Aside, not part of this plan: two other worktrees —
`progress-indication-downloads`, `progress-indicator-gaps` — were confirmed
to be stale, unmerged, pre-#347 drafts. They predate the spinner feature
that's already on `main` and are superseded/irrelevant. No action needed on
them.)

## Answering the two review questions

### Does issue #368's plan leave dead code?

Partially, but not for the reason the issue assumed, and the WIP's actual
choices avoid it. Concretely:

- The issue's plan says **neither** download path has an existing override
  hook. That's true for ComfyUI but **false** for the tarball path: the
  tarball catalog base URL is already overridable via `env_override_base` +
  `ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE`/`ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE`
  (`therock.rs:214-260`), a runtime-gated (not feature-gated) mechanism
  already documented in `docs/release-trust.md` and already exercised by
  `therock_steps.rs`'s `tarball_index_fixtures` for dry-run scenarios. Adding
  a second, `e2e-test-hooks`-gated override for the same URL would be
  redundant dead weight. **The WIP correctly reuses the existing mechanism
  as-is for the tarball path** — no new override code needed there at all.
- For ComfyUI, there genuinely is no existing override (confirmed:
  `COMFYUI_SOURCE_ARCHIVE_URL` is a hardcoded const, zero indirection). The
  WIP adds a new `ROCM_CLI_COMFYUI_SOURCE_ARCHIVE_URL_OVERRIDE`, gated behind
  `e2e-test-hooks`. This is the right call, not dead code: it mirrors the
  established pattern in `dash.rs` (`dash_test_clock_offset_path`, feature-gated
  accessor with a `#[cfg(not(...))]` `None` twin so it compiles away entirely
  in release builds) and in `engines/lemonade/src/lib.rs`. Unlike the tarball
  base, ComfyUI's archive URL has no legitimate production override use case
  (no documented proxy/mirror story), so a test-only, compile-time-gated hook
  is the correct fit — not a third inconsistent mechanism, but the second
  instance of an existing one.
- Net conclusion: **use the right existing pattern per surface** — reuse
  `env_override_base` for tarball, add a `dash.rs`-style `e2e-test-hooks`
  accessor for ComfyUI. Do not introduce a uniform new mechanism across both,
  which is what the issue's plan implicitly proposed and which would have
  been the actual source of inconsistency/dead code.
- One real (small, mechanical) defect in the WIP as it stands: the new
  `download_progress_pty.feature` isn't registered in
  `tests/e2e-cucumber/tests/feature_naming.rs`'s `FEATURE_KEYS`, so
  `cargo test` fails immediately on the naming-convention check. This isn't
  dead code, just an incomplete registration — fixed as part of this plan
  (see Implementation approach).

### Is the test coverage sufficient?

Mostly, once one gap is closed:

- **Good**: the WIP's tarball scenario already asserts an intermediate
  `%)`-bearing frame (proving live progress renders, not just start/end),
  then a clean exit, then that no `"Downloading …"`/`"Extracting …"` text
  remains on screen. This is genuinely new coverage — nothing today drives
  either install path under a real PTY.
- **Gap**: the final "neither string remains" assertion doesn't prove the
  *extraction* spinner ever appeared — it's equally true if the extraction
  spinner never rendered at all. Since `therock.rs`'s install has two
  independent spinners (download-with-progress, then extraction-without-progress,
  each explicitly `drop()`ped — see `therock.rs:2695-2713`), a regression that
  silently dropped the extraction spinner would not be caught. Fix: assert an
  intermediate frame containing `"Extracting …"` is observed before the final
  cleared state (see acceptance scenario below). To make this reliably
  observable (extraction of a tiny fixture completes near-instantly, which
  would race the PTY polling interval), size the fixture tarball's filler
  content large enough (tens of MB) that extraction takes a measurable amount
  of wall time — reusing the same xorshift-filler trick the WIP already uses
  to defeat gzip compression, just scaled up. This avoids adding a new
  artificial-delay test hook for a `therock.rs`-internal step.
- **Correctly scoped out (no new coverage needed here)**: the monotonic
  progress clamp on retry (`set_progress_never_displays_fewer_bytes_than_already_shown`)
  and all formatting/truncation logic are already thoroughly unit-tested in
  `cli_progress.rs`. E2E scenarios should not re-assert these — only true
  end-to-end TTY rendering belongs at this level.
- **Gap, straightforward to close**: no ComfyUI-side scenario exists yet in
  the WIP (the URL-override plumbing was added but no `.feature`/steps). This
  plan adds one, extending the existing `comfyui.feature` (`comfyui-01..03`)
  rather than creating a new file, since it belongs with the other ComfyUI
  install scenarios.
- **Not a gap**: non-interactive (piped-stdio) install behavior for both
  paths is already exercised by existing non-PTY scenarios today (e.g.
  `comfyui_steps.rs`'s `cli_succeeds_and_shows_progress`, explicitly testing
  the non-TTY fallback). This plan only adds the missing PTY-observed half;
  it doesn't touch or duplicate that coverage.
- **Not a gap**: Linux-only scoping (`@requires-os:linux`) for both new
  scenarios matches the suite's existing convention — the only other PTY
  scenario in the whole suite (`install_lifecycle.feature`'s `lifecycle-08`)
  is also Linux-only. This is a deliberate, already-established choice, not
  an oversight to flag.

## Acceptance scenarios

Following the `bdd-scenarios` skill's quality rules and this repo's existing
Gherkin conventions (`@id:`, `@requires-os:`, `<key>-NN - ` naming from
`feature_naming.rs`).

New file `tests/e2e-cucumber/features/download_progress_pty.feature`
(key: `download-progress`):

```gherkin
@id:download-progress-linux-tarball-install-shows-live-progress @requires-os:linux
Scenario: download-progress-01 - Linux - installing an SDK tarball over an interactive terminal shows live progress
  Given a paced fixture tarball is served as the release tarball
  When the user installs the SDK tarball through a pseudo-terminal
  Then the interactive terminal shows download progress advancing before the download completes
  And the interactive terminal shows the archive being extracted
  And the install completes successfully
  And no download or extraction progress text remains on screen
```

Extend existing `tests/e2e-cucumber/features/comfyui.feature` (key: `comfyui`)
with a fourth scenario:

```gherkin
@id:comfyui-linux-source-download-shows-live-progress @requires-os:linux
Scenario: comfyui-04 - Linux - installing ComfyUI over an interactive terminal shows live download progress
  Given a paced fixture archive is served as the ComfyUI source archive
  When the user installs ComfyUI through a pseudo-terminal
  Then the interactive terminal shows download progress advancing before the download completes
  And the install completes successfully
  And no download progress text remains on screen
```

(No extraction-spinner assertion here — ComfyUI's install has no separate
extraction spinner, unlike the tarball path; the two surfaces are not
symmetric.)

## Implementation approach

1. **Reconcile the WIP worktree** (`.claude/worktrees/e2e-download-spinner-pty`)
   onto a proper branch off current `main` rather than restarting: cherry-pick
   or manually port `tests/e2e-cucumber/src/paced_download.rs` (the
   `PacedDownloadServer`, already unit-tested), the `therock_steps.rs` fixture
   setup (real gzip tarball over xorshift filler bytes, `PacedDownloadServer`
   wiring, existing `ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE` +
   `ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE` reuse), and the
   `ROCM_CLI_DISABLE_TORCH_RUNTIME_DEP_CHECKS` test hook in `main.rs`.
2. **Fix the feature registration**: add `download_progress_pty.feature` to
   `FEATURE_KEYS` in `tests/e2e-cucumber/tests/feature_naming.rs` with key
   `download-progress`, and rename the scenario name/id to
   `download-progress-01 - ...` / `@id:download-progress-...` (the WIP
   currently uses `download-progress-pty` as if it were the registered key,
   which it isn't).
3. **Add the missing extraction-visibility assertion** to the tarball
   scenario, sizing the fixture tarball's filler payload large enough that
   extraction is reliably observable by the PTY driver's polling.
4. **Add the ComfyUI half from scratch**: port the WIP's
   `ROCM_CLI_COMFYUI_SOURCE_ARCHIVE_URL_OVERRIDE` test hook in `comfyui.rs`
   (already drafted in the WIP), add a `PacedDownloadServer`-backed fixture
   in `comfyui_steps.rs` mirroring the tarball fixture pattern, and add
   `comfyui-04` to `comfyui.feature` plus its step glue.
5. **Reuse `TuiSession`** (`tests/e2e-cucumber/tests/e2e/tui_driver.rs`)
   unmodified for both scenarios — confirmed to be a correct fit as-is: it
   spawns a real PTY (satisfying the spinner's `stderr().is_terminal()` gate)
   and asserts via `vt100`'s terminal emulation, which correctly interprets
   single-line CR/clear-line repaints without any special-casing needed.
6. **Document** the new ComfyUI override env var in `docs/release-trust.md`
   (the WIP already drafted this section) alongside the existing TheRock
   override documentation, but explicitly note it as test-only (unlike the
   TheRock overrides, which are real operator-facing knobs).

**Verification**: run `cargo test -p e2e-cucumber` (or the equivalent
naming-convention test target) to confirm `feature_naming.rs`'s checks pass
after registration. Run the two new scenarios directly against a real build
via the repo's E2E harness (`cargo xtask e2e` or equivalent, per
`install_lifecycle.feature`'s documented invocation) — this requires a live
build and real fixture servers, so it cannot be verified via unit tests or
dry runs alone; both scenarios must be run to a real pass/fail before this
work is considered done.

## Tradeoffs

- **Reuse vs. rewrite the WIP worktree**: the WIP is uncommitted, ~30 commits
  behind `main`, and has one known defect (unregistered feature key). Given
  its design (paced server, xorshift fixture trick, correct reuse of
  `env_override_base`) is already sound and matches this plan's conclusions
  independently, reconciling it is far cheaper than rewriting equivalent code
  from scratch. Recommendation: reconcile, don't rewrite.
- **Extraction observability via a bigger fixture vs. a new artificial-delay
  hook**: a delay hook (à la `ROCM_CLI_DISABLE_TORCH_RUNTIME_DEP_CHECKS`)
  would be more deterministic but adds another test-only code path inside
  `therock.rs`'s extraction step. Sizing the fixture tarball larger achieves
  the same observability with zero production-code changes. Recommendation:
  bigger fixture first; only add a delay hook if timing proves flaky in
  practice.

## Open questions / Risks

- `E2eWorld`'s expectation-matrix loading (`load_expectations()`/
  `expectation::Expectations::parse`) may require a per-scenario entry in
  `expectations.toml` — not confirmed as a blocker, should be checked while
  implementing rather than assumed away.
- Extraction-timing observability (the fixture-size approach) is an
  assumption, not yet empirically verified against the PTY driver's actual
  polling cadence — flagged above as a tradeoff with a fallback (delay hook)
  if it proves unreliable.

---

## Reconciliation notes (added after the plan was approved)

Investigation into porting this WIP onto a fresh branch off `main` confirmed:

- `git log --oneline main..HEAD` in this worktree is **empty** — this
  branch has zero commits ahead of `main`. Every change described above is a
  purely uncommitted working-tree modification sitting on a branch tip that
  is not ahead of current `main`.
- `tests/e2e-cucumber/tests/e2e.rs`'s WIP diff against current `main` is
  exactly 4 surgical additions: the `PacedDownloadServer` import, the
  `paced_download_server: Option<PacedDownloadServer>` field + doc comment on
  `E2eWorld`, one `Default` line, and one `Drop` line. All `run_rocm*`
  invocation helpers are byte-for-byte unmodified.
- **Flagged risk, not yet confirmed via `git diff`**: the WIP's `mod e2e { }`
  block in `e2e.rs` appears to be missing `pub mod service_cleanup_steps;`,
  which current `main` has (backing `service_record_cleanup.feature`). Since
  the branch has no commits ahead of `main`, this must be sitting inside the
  uncommitted diff itself — likely accidental. **Do not port this removal.**
  When reconciling `e2e.rs`, apply only the 4 confirmed additions above to
  the current `main` version of the file; never replace `main`'s file with
  the WIP's version wholesale.
- Remaining exact hunks for `apps/rocm/src/main.rs`, `apps/rocm/src/comfyui.rs`,
  `docs/release-trust.md`, `tests/e2e-cucumber/src/lib.rs`, and
  `tests/e2e-cucumber/tests/e2e/therock_steps.rs` still need a `git diff`
  pull (only `git status` file-level modification flags were confirmed
  before this checkpoint) before porting them onto the fresh branch.
- `download_progress_pty.feature`'s current WIP text still needs the rename
  called out in Implementation approach step 2 (`download-progress-pty-01`
  → `download-progress-01`, id prefix fixed) and the extraction-visibility
  step added.
- `PACED_TARBALL_PAYLOAD_BYTES` in the WIP's `therock_steps.rs` is currently
  `400_000` (390KB) — must be enlarged to "tens of MB" per the plan's step 3
  before extraction timing is reliably observable.
