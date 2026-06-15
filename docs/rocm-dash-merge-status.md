# rocm-dash → rocm-cli merge — status & handoff (EAI-6871)

> Branch: `our-spec` (current working branch; supersedes the earlier
> `integration/rocm-dash-merge`).
> Base: `837067f` ("Import rocm-cli"). Source: `~/git/rocm-dash/app` @ `main`.
> Scope of this branch: **Phase 1 (foundation) + Phase 2 (fold) COMPLETE** of
> `wiki/plans/rocm-cli-unification.md` (that plan lives in the rocm-dash repo). The
> standalone rocm-dash has been folded into the `rocm` binary: `rocm dash` launches the
> unified dashboard TUI (tabs: Overview / Hardware / Instances / Bench / Chat) backed by
> an embedded daemon. Pure-engineering; governance/legal gates (G1 license, G3/G4/G5) are
> tracked out-of-band.

## What landed (Phase 1 — green)

The four rocm-dash **library** crates are first-class workspace members under `crates/`:

| Crate | Tests | Notes |
|---|---|---|
| `rocm-dash-core` | 46 | pure-reducer types; no tokio/ratatui at the boundary (invariant LRN-20260405-004) |
| `rocm-dash-collectors` | 54 (+2 ign) +2 integ | amd-smi / Docker / vLLM-Prom / Lemonade / per-proc VRAM; reqwest **0.12** |
| `rocm-dash-daemon` | 19 +3 integ | telemetry runner + snapshot ring |
| `rocm-dash-tui` | 201 (+2 ign) | ratatui **0.30** dashboard base; rig-core 0.38.1 chat; reqwest **0.13** |

- **Edition 2024 / rust 1.88 / Apache-2.0** (inherited from the workspace). `cargo fix --edition`
  needed **zero** source changes; clippy `--fix` applied `collapsible_if`→let-chains; fmt clean.
- **ratatui majors coexist, confined per crate:** `apps/rocm` `tui.rs` keeps **0.29**; `rocm-dash-tui`
  uses **0.30**. Each crate `use`s only its own major.
- **HTTP partition:** ureq stays the rocm-cli sync default everywhere; reqwest confined to
  `rocm-dash-collectors` (0.12) + `rocm-dash-tui` (0.13, via Rig).
- **Vendor not promoted:** the rocm-dash `crates/vendor/*` (stale March snapshot) and the rocm-dash
  `rocm` bin were intentionally **not** copied. `apps/rocm` remains the sole `rocm` bin. The bin's
  vendor→real-crate reconcile + verb fold (now **Phase 2, complete**): `rocm dash` launches the
  unified telemetry dashboard (tabs Overview / Hardware / Instances / Bench / Chat) backed by an
  embedded daemon. Note: bare `rocm` and `rocm chat` still launch the legacy chat-first assistant
  (`apps/rocm/src/tui.rs`); its retirement is deferred.
- Verified: `cargo build --workspace --all-targets` green; the 4 crates at **exact parity** with rocm-dash.

## Pre-existing `apps/rocm` test failures (NOT caused by this merge)

`cargo test --workspace` shows 3 failures, all in `apps/rocm` (code this merge never touched).
Diagnosed 2026-06-11; none are GPU-related, none are merge regressions:

1. `tui::tests::served_model_chat_accepts_typed_messages_and_uses_selected_model`
2. `tui::tests::assistant_tui_support_prompts_reach_validated_local_model`
   → both use an in-process **fake HTTP chat server** with a 5 s `recv_timeout`; they are
   **parallelism/timing flakes** — they **pass when run single-threaded** (`--test-threads=1`).
3. `therock::tests::python_launcher_prefers_path_python_before_saved_managed_python`
   **and** `therock::tests::python_launcher_skips_path_python_without_pip_ready_venv`
   → **environmental** (two tests, same root cause): `resolve_python_launcher` executes a generated
   fake-python stub to check it can build a pip venv; on a host where the stub can't run, the
   resolver skips it. Both fail deterministically on this machine **independent of the merge**
   (guarded by `PYTHON_RESOLVER_TEST_ENV_LOCK`, i.e. the authors already treat these as
   env-sensitive). (Originally only the first was named here; verified 2026-06-11 that both fail on
   the pre-merge HEAD, so the honest pre-existing count is **4**: 2 flaky chat + 2 env python-launcher.)

These belong to the rocm-cli side and predate the merge. Do **not** treat them as merge regressions.

## Deferred (intentionally NOT in Phase 1)

- **crossterm 0.28→0.29 unification** + port `app.rs` off the crossterm-0.28 `EventStream`. The two
  versions coexist cleanly today; unification is a nicety, sequenced with the TUI work.
- **collectors reqwest 0.12→0.13 unification** (cosmetic; both coexist).
- **Phase 2** (config/engines/daemon/dispatch + bin fold + vendor→real reconcile) is now
  **complete** — `rocm dash` is live (Unix-only for the live socket transport; `--demo`/`--replay`
  also run on Windows). **Phase 3** (unified-TUI screen merge + chat split) has also landed (no-key
  ChatGPT OAuth chat backend; config unified into `config.json` with legacy `config.toml`
  auto-migration). See `wiki/plans/rocm-cli-unification*.md`.
