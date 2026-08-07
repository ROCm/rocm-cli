<!-- WIP file. Personal notes on the orphan progress branch — never merged into main. -->

# WIP: Pre-seed Qwen3.6-27B weights for nightly E2E (EAI-7477)

**Stage:** 7-decision (obsolete — scenario 10 now passes on recent nightlies; pre-seed unproven)
**Pipeline:** standard
**Branch:** fix-nightly-27b-preseed
**Jira:** EAI-7477 (Bug, component rocm-cli) — https://amd.atlassian.net/browse/EAI-7477
**Last Updated:** 2026-08-07
**Pre-PR-check:** passed — pre-PR reviewer (fix-nightly-27b-preseed worktree), 2026-07-23

**Token Usage:** in=72 out=22056 cache_create=253791 cache_read=4225838 calls=39

---

## Problem

The nightly E2E scenario `serve-large-model-inference` (scenario 10 in `model_serving.feature`;
`Qwen/Qwen3.6-27B` on vLLM, tagged `@nightly @serve-timeout:2400`) fails on **every** nightly run
(5 in a row, 2026-07-16 → 07-19). It waits the full 40 min then fails with
`did not serve model Qwen3.6-27B within 2400s`; the reconciler flags it as an unexpected regression
(expect-pass).

Not flaky, not GPU/engine: a small vLLM model (scenario 5) serves fine on the same host/port minutes
earlier, proving the machinery is healthy.

**Root cause:** the serve step (`serving_steps.rs:355`) assumes weights are pre-seeded in the shared
HF cache so serving is load-only. Nothing seeds them. The nightly pre-warm (`nightly.yml:373-381`)
runs only `rocm install sdk` (seeds the ROCm *runtime*, not weights); there is no `hf download` of
the 27B anywhere in `nightly.yml`/`ci.yml`. So every run does a cold ~54 GiB pull at serve time,
worsened by unauthenticated HF Hub rate-limiting, which can't finish in 2400s.

## Solution

Reuse the existing built-in on-demand HF pull, just move it OUT of the timed serve step: add a
one-time-per-runner pre-seed of the 27B weights into `$E2E_SHARED_CACHE_DIR/huggingface` in the
nightly pre-warm block, guarded like the existing runtime pre-warm (skip if already present),
ideally with an authenticated `HF_TOKEN` to dodge rate limits. Then the timed serve is genuinely
load-only and fits the window. NOT a timeout bump.

Confirmed plumbing:
- Harness points every scenario command's `HF_HOME` at the shared cache: `isolate_cmd` sets
  `HF_HOME = $E2E_SHARED_CACHE_DIR/huggingface` (`tests/e2e-cucumber/tests/e2e.rs:176`).
- That dir lives under `$RUNNER_WORKSPACE`, which persists across nightlies (same guarantee the
  runtime pre-warm relies on — skipped on run 2+). Weights are content-addressed/immutable → re-fetch
  is a genuine no-op once present.
- Pre-warm block currently sets `HF_HOME=$E2E_SHARED_CACHE_DIR/huggingface` only for `install sdk`
  (`nightly.yml:379`); the new pre-seed must set the same `HF_HOME`.

Secondary: same scenario also runs on the per-run `e2e-gpu` job (90-min cap, `ci.yml:672`) — a cold
54 GiB pull is a cap risk there too; apply the same pre-seed.

## Scenarios

_TODO — activate bdd-scenarios skill before writing. Likely a CI/harness precondition rather than a
new Gherkin scenario; the existing scenario 10 is the behavioral assertion. Decide whether any new
scenario is warranted or this is pure test-infra hardening._

## Implementation Steps

### Todo 📋
- ✅ Decided pre-seed mechanism: `uv run --with huggingface_hub … snapshot_download("Qwen/Qwen3.6-27B")` with HF_HOME=shared cache; uv resolved from system PATH else newest managed uv under `$prewarm/data/managed-tools/uv/*/uv` (rocm's uv isn't on PATH). HF_TOKEN optional (`${{ secrets.HF_TOKEN }}`), unauth fallback — no secret exists in CI today.
- ✅ Added guarded pre-seed to `nightly.yml` pre-warm (unconditional; nightly always runs 27B) + `ci.yml` e2e-gpu (gated on `include_nightly`). Guard = snapshot marker `hub/models--Qwen--Qwen3.6-27B`. Committed 12721fa (signed, DCO).
- ⏳ Prove via scratch dispatch: branch `scratch-27b-preseed` pushed to origin; dispatched ci.yml (platform=app-dev-gpu, include_nightly=true, name_filter="large vLLM model serves") → run 29756582721, scenario 10 only. **BLOCKED**: E2E-GPU job stuck `queued` ~15h — both `app-dev-gpu` runners `offline` + app-dev cluster auth (`kc.app-dev.silogen.ai`) returning EOF. No CI signal yet. Leaving run queued so it auto-picks-up when runners return (user says soon). Resume: `gh api repos/ROCm/rocm-cli/actions/jobs/88400569474 --jq .status` → once `in_progress`, watch to green; confirm marker dir + load-only serve.
- 📋 On green: open PR from feature branch `fix-nightly-27b-preseed`; delete scratch branch.
- 📋 Verify local gates (cargo test/clippy, smoke) — YAML-only change but run.

## Decisions (2026-07-20, stage 4→6)
- Mechanism: uv-run hf_hub snapshot_download (NOT throwaway `rocm serve`, NOT huggingface-cli).
- Auth: optional HF_TOKEN secret, unauth fallback (pre-seed is off the timed path, one-time-per-runner).
- Marker path (to confirm on runner): `$HF_HOME/hub/models--Qwen--Qwen3.6-27B`.
- Scratch-branch dispatch pattern reused from prior work; real change lives on feature branch, scratch only for scoped dispatch.
- Reused existing ci.yml `name_filter` + `include_nightly` workflow_dispatch inputs (already on main) to scope to scenario 10.

## Next Steps

**Waiting for decision**: Close as won't-fix (obsolete) or rebase + re-prove as insurance against cache wipe? See Work Log 2026-08-07.

## Blockers / Open Questions

- **RESOLVED**: scenario 10 now passes on Aug 3–6 nightlies (was failing 07-16 → 07-19). Pre-seed not in main; recovery happened on its own.
- **Proof gap**: branch pre-seed (commit 12721fa) was never verified (dispatch run 29756582721 cancelled, not run).
- **Decision point**: close as obsolete (timeout was transient cold-cache condition) or land as insurance (rebase + dispatch needed).

## Notes

- Built-in pull already exists (that's why it times out rather than erroring "not found") — this is a pre-seeding gap, not a missing downloader.
- Inbox origin: `~/Developer/work-ledger/inbox.md` rocm-cli item 13 (from test-e2e-diagnose nightly investigation, 2026-07-20).

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-nightly-27b-preseed`
- Recreate with: `create_worktree.sh fix-nightly-27b-preseed`
- Base: origin/main @ 73e0fd1

## Work Log

### 2026-07-20

- Created EAI-7477 (Bug, component rocm-cli) from inbox item; root-caused the cold 54 GiB pull at serve time.
- Verified harness `HF_HOME` shared-cache plumbing and cross-run persistence assumption in code.
- Set up worktree off fresh origin/main and this WIP at stage 4-design.
- Next: pin pre-seed mechanism + HF_TOKEN sourcing, then scenarios/impl.

### 2026-08-07

- **Key finding**: scenario 10 (27B serve) passes on recent nightlies (Aug 3–6, all green); Aug 6 took only 126s (load-only) vs. 2400s timeout previously.
- **Root resolution**: 27B weights now resident in shared cache — cold 54 GiB pull is gone. Pre-seed not in main; recovery self-resolved.
- **Status**: branch is 20 commits behind main; pre-seed commit unproven (dispatch run cancelled); no open PR. Options: (1) close as obsolete (timeout transient), or (2) rebase + dispatch to prove as insurance. Recommending option 1 — awaiting user decision.
