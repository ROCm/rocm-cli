<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Plan: `rocm bench load` — a minimal GPU-saturating load generator

> Status: ready (adversarially vetted against `main`) · Owner: TBD · Date: 2026-07-08
> Supersedes the 2026-06-22 draft. Consumer pipeline is already shipping; this
> plan is Phase 1 only: **generate rows + append to the tailed CSV**.

## 0. Why this is a small change

The **consumer half already ships on `main`** (verified). Rows flow:

```
CSV file ──CsvBenchTailer.drain()──▶ runner drain ──▶ StateEvent::BenchmarkRows
        (collectors/bench_tail.rs)   (daemon/runner.rs:576)  + BenchRing.push
                                                             + broadcast BenchmarkRowsAppended
                                                                     │
                     TUI AppState.bench_rows ◀── client subscribe ◀──┘
                     bench tab (rollup + rows table + sparkline)
```

The bench tab in the Observe pane is inert only because **nothing produces
rows**. The single missing piece is a load generator. MVP = a generator that
appends normalized rows to the file a running daemon already tails
(`RunnerOptions.bench_csv` ← `config.dashboard.daemon.bench_results_dir`), so
the whole consumer pipeline lights up with **zero daemon/protocol/UI changes**.

## 1. Final scope

**Phase 1 IS:** one concurrency-sweep load generator that POSTs synthetic
`/chat/completions` requests to a **local `http://` OpenAI-compatible endpoint**,
computes client-side `gen_tps` **and** `prompt_tps` from a wall makespan, writes
**one aggregate `BenchmarkRow` per concurrency cell** as append-only CSV to a
**distinct self-labeled file**, exposed as `rocm bench load`. All quality fields
default → verdict `Unknown`. Every run prints a fixed "not an official
benchmark" note.

### Cut from the original draft (with why)

| Cut | Why |
|---|---|
| Second file `daemon/src/bench_load.rs` | Daemon has **no `csv` dep**; writer lives in collectors (owns csv/reqwest/tokio). Deletes a file, a manifest edit, a dep hop. |
| `run_sweep` public fn | One-caller 3-line loop; inline it. YAGNI. |
| `LoadSpec.prom_host` + server-side Prometheus peaks | Plan's own Phase 1.5. Leave `max_running/max_waiting = None`. |
| One-row-per-request | Only Pass^N needs per-request rows, and it's `Unknown` here. Rollup+table+sparkline render fine with 1 row/cell. |
| "record client + server gen_tps distinctly" | Schema has **one** `gen_tps` field + no second UI column. Not representable in MVP. |
| reqwest `json`/TLS feature adds | Hand-serialize with `serde_json` + `.text()` (`lemonade.rs` pattern); localhost `http` needs zero manifest change. |
| Folding bench into `Dash` struct-variant | Would make Dash the only variant mixing flags + `#[command(subcommand)]` — untested clap precedence. Use top-level `Command::Bench`. |
| `anyhow` in writer signature | Collectors has no `anyhow`; use `thiserror` (already present). |
| Default `--out` → shared `bench_results_dir` | Silently pollutes users' real agent-bench comparison CSVs with Unknown-quality rows. Opt-in only. |
| "byte-compatible with external tools" claim | **False** if we auto-serialize the struct (~60 cols ≠ upstream 30-col order). We write our own fixed minimal header; drop the compat claim. |
| Streaming TTFT/TPOT, auto-ramp, `Command::RunBench` | Phase 1.5 / Phase 2. |

## 2. Files touched

| File | Change | ~LOC |
|---|---|---|
| `crates/rocm-dash-collectors/src/bench_load.rs` | **NEW.** `LoadSpec`, `run_cell` (JoinSet + owned-permit Semaphore, hand-serialized POST, client `gen_tps`+`prompt_tps`), `run_and_append_csv` (inlined sweep loop, `csv::Writer`→String, `O_APPEND`, header-if-empty), `thiserror` error enum. | ~230 |
| `crates/rocm-dash-collectors/src/lib.rs` | `pub mod bench_load;` | 1 |
| `crates/rocm-dash-daemon/src/lib.rs` | `pub use rocm_dash_collectors::bench_load;` — `apps/rocm` reaches the writer with no new dep edge. | 1 |
| `apps/rocm/src/main.rs` | New `Command::Bench { #[command(subcommand)] command: BenchCommand }` + `BenchCommand::Load {..}`; one dispatch arm; add `"bench"` to the two structured-command lists (runtime slice ~15230, test list ~19168). | ~45 |
| `apps/rocm/src/dash.rs` | New sync `run_bench(...)`: resolve config, default `--out` to a distinct file, default `--model` via `pick_first_model`, validate `http://`, `build_dashboard_runtime()` + `block_on`, print summary + note. | ~40 |
| `crates/rocm-dash-tui/src/ui/tabs/bench.rs:28` | Update empty-hint string to mention `rocm bench load`. | 1 |
| `crates/rocm-core/src/lib.rs:~3945` | Fix `bench_results_dir` doc comment: "file", not "directory". | 1 |

Net: **~360 LOC, one new file** (vs the draft's two files across two crates).

**Daemon file dropped (reviews unanimous):** placing `csv::Writer` in daemon
requires adding `csv` to daemon's manifest and splitting writer-from-reader
across crates, which the non-flexible reader makes fragile. Collectors already
owns `csv`, so writer and reader share one dialect/quoting.

## 3. CLI shape — top-level `Command::Bench`, verb `rocm bench load`

Top-level parse shape keeps `rocm dash` / launcher / focused-host /
`launch_default` byte-identical (all verified independent of the `Dash`
variant). Decision D1 resolved: **`rocm bench load`**.

```rust
// main.rs — new variant beside Dash (Dash arm untouched):
/// Saturate a local OpenAI-compatible endpoint and report rough client-side
/// throughput (local smoke-test, NOT an official ROCm/AMD benchmark).
Bench {
    #[command(subcommand)]
    command: BenchCommand,
},

#[derive(Subcommand, Debug)]
enum BenchCommand {
    /// Run a concurrency sweep (RAW serving throughput, synthetic single-shot
    /// requests — not agent-shaped, not comparable to *-agent-bench harnesses).
    Load {
        #[arg(long, value_name = "URL")] endpoint: String,
        #[arg(long)] model: Option<String>,
        #[arg(long, value_delimiter = ',', default_value = "1,8,32,64")] concurrency: Vec<u32>,
        #[arg(long, default_value_t = 1024)] isl: u32,
        #[arg(long, default_value_t = 1024)] osl: u32,
        #[arg(long, default_value_t = 128)] requests: u32,
        #[arg(long, value_name = "FILE")] out: Option<PathBuf>,
    },
}
```

```rust
// main.rs dispatch — ONE new arm, Dash untouched:
Some(Command::Bench { command }) => match command {
    BenchCommand::Load { endpoint, model, concurrency, isl, osl, requests, out } =>
        dash::run_bench(endpoint, model, concurrency, isl, osl, requests, out),
},
```

Guardrails (verified): `value_delimiter = ','` is **mandatory** or clap rejects
`1,8,32,64`. Do **not** mark bench `hide = true` (completion-leak tests require
it visible). Add `"bench"` to both structured-command lists or bare `rocm bench`
routes to the NL planner.

## 4. Data / CSV contract

Deserialization is **by header name** (`rec.deserialize(Some(&header))`); reader
is `has_headers(true)` + non-flexible (`flexible=false`). Header names = struct
field names verbatim (no rename/flatten/deny_unknown_fields). A stable subset is
safe (`#[serde(default)]` on `BenchmarkRow`; proven by
`missing_30col_fields_default_safely`).

**Fixed minimal header (written once, for the file's lifetime):**

```
cell,run,concurrency,model,engine,input_len,output_len,n_requests,prompt_tokens,completion_tokens,prompt_tps,gen_tps,wall_s,launcher
```

**Populated per aggregate row (one per concurrency cell):**

| Field | Value |
|---|---|
| `cell` | distinct label per sweep point (e.g. `bench-c{N}`) — non-Option grouping key |
| `run` | `1` (non-Option u32) |
| `concurrency` | the cell's N (RollupKey member; separates groups) |
| `model`, `engine` | resolved model; `engine` stamped **once per sweep** (or `None` consistently — never varying, it's a RollupKey member) |
| `input_len`/`output_len` | isl/osl |
| `n_requests` | count of **successful token-bearing** responses |
| `prompt_tokens`/`completion_tokens` | Σ over successes |
| `gen_tps` | Σcompletion_tokens / makespan_s (client-side, authoritative) |
| `prompt_tps` | Σprompt_tokens / makespan_s — **required**: the sparkline plots `prompt_tps` only, so a gen_tps-only row is a flat zero line |
| `wall_s` | makespan |
| `launcher` | `"rocm bench load (local smoke)"` — carries the caveat in-band |

Quality fields omitted → `PassFail::Unknown` → `row_verdict` = Unknown →
rollup never counts as pass (confirmed by `unknown_verdict_does_not_count_as_pass`).

**Append-race + header-once (CRITICAL — a newline-less complete row parses as
valid, so this is the real hazard):**

- Open with `OpenOptions::new().append(true)` (`O_APPEND`).
- Serialize each row via `csv::Writer` into a `Vec<u8>` that **ends in `\n`**,
  then **one `write_all`** of the whole line. No long-lived buffered writer; no
  deferred trailing newline. On a regular file, `O_APPEND` + a single small
  `write_all` keeps each row's bytes contiguous.
- Write the header **only when the file is new/empty**; append-only thereafter
  (matches `skip(rows_seen)` drain). **Never truncate** a file a daemon tails.

**Makespan:** `let t0 = Instant::now()` **before** the spawn loop; drain
`while js.join_next().await …`; `makespan = t0.elapsed().as_secs_f64()`. If
`makespan <= 0.0` → `gen_tps = None`.

**Per-request isolation:** each task returns `Result<Outcome, _>` and never uses
`?` to escape; check `resp.status().is_success()` before reading usage (non-2xx =
recorded failure, not parsed); missing `usage.completion_tokens` = failure
(excluded from sums and count, not a silent 0); `JoinError` → failure count. One
refusal must not zero a cell.

**Semaphore:** `let _permit = Arc::clone(&sem).acquire_owned().await?;` bound to a
**named** variable held for the whole request.

## 5. Public-landing specifics

- **Command:** `rocm bench load` (top-level parse shape; Dash untouched).
- **`--help` long doc (on `Load`, un-croppable at point of use):**
  > Measures RAW serving throughput (synthetic single-shot requests, vLLM
  > benchmark_serving shape). It does NOT reproduce agent-shaped, multi-turn,
  > long-context tool traffic and is not comparable to the *-agent-bench quality
  > harnesses.
- **Stdout — one summary line per cell + one fixed note per run:**
  ```
  cell=bench-c32 concurrency=32 gen_tps=4213.5 prompt_tps=812.0 wall=6.41s n=128
  note: local saturation smoke-test — client-measured throughput, not an official ROCm/AMD benchmark.
  ```
  Emit once at start: `mode: raw serving throughput (synthetic prompts) — not agent-workload.`
- **Empty-hint** (`bench.rs:28`): `no rows · run \`rocm bench load --endpoint <url>\` or start the daemon with --bench-csv <path>`
- **`--out` default:** distinct self-labeled file
  `~/.rocm/bench/rocm-bench-<timestamp>.csv`, **not** the shared
  `bench_results_dir`. Appending to `bench_results_dir` is opt-in via explicit
  `--out`; on an existing file, refuse if the header differs.
- **https:** validate `--endpoint`; reject `https://` with
  `error: rocm bench load supports http:// endpoints only (no TLS backend compiled in)`.

## 6. Dependencies

| Manifest | Change | Ponytail justification |
|---|---|---|
| `collectors/Cargo.toml` `[dependencies]` | **none** | tokio `full` (JoinSet+Semaphore), serde_json, csv, reqwest (http POST) all present. POST body hand-serialized + `.text()` parsed like `lemonade.rs` — no feature needed. |
| `collectors/Cargo.toml` `[dev-dependencies]` | `wiremock = "0.6"` | No HTTP mock anywhere in the workspace. Async-native; built on http/hyper/tower already in `Cargo.lock`. Hand-rolling a hyper stub = more code. |
| daemon / apps/rocm | **none** | Re-export reaches the writer; no new dep edge. |

Shipped runtime deps added: **zero**. New dev-dep: **one** (`wiremock`).

## 7. Tests (minimal, deterministic, no GPU)

Run under `cargo test -p rocm-dash-collectors -- --test-threads=1` (repo
deterministic gate).

| Test | Guards | Assert |
|---|---|---|
| **T1** `run_cell` vs wiremock stub | POST shape, token sum, tps math | row fields correct, `completion_tokens` sum, `gen_tps > 0.0` (never exact — divides by measured makespan) |
| **T2** concurrency cap | Semaphore truthfulness of the `concurrency` column | mock `Respond` with `Arc<{cur,max: AtomicUsize}>` + short barrier; `fetch_max`; assert `max <= N` and `max == N` for N<requests. Structural, not timed. |
| **T3** CSV round-trip | writer↔non-flexible reader dialect, header-once, append-only | `run_and_append_csv(A)` → `CsvBenchTailer::drain()` returns A; 2nd drain empty; append B → drain returns only B; `pass_fail` deserializes to `Unknown`. Reuse in-crate tempdir helper (`bench_tail.rs`) — **no `tempfile` dep**. |
| **T4** Unknown-verdict | throughput rows never false-pass | generator row (cell/run/gen_tps/concurrency only) → `row_verdict == Unknown` and `rollup_pass_n → n_passed == 0`. |
| **T5** clap parse smoke | shape wiring | `Cli::try_parse_from(["rocm","bench","load","--endpoint","http://x","--concurrency","1,8,32,64"])` → `concurrency == [1,8,32,64]`. |

Non-goals asserted: no exact tps/wall asserts; no wall-clock math; live-endpoint
smoke is `#[ignore]`.

## 8. Ordered task list

1. **Manifest + module:** add `wiremock = "0.6"` to collectors dev-deps;
   `pub mod bench_load;` in collectors lib. → `cargo build -p rocm-dash-collectors`.
2. **`LoadSpec` + `run_cell`:** struct (no `prom_host`), hand-serialized POST,
   named owned-permit semaphore, pinned makespan, per-request `Result`
   isolation, `status().is_success()` check, client `gen_tps`+`prompt_tps`.
   → **T1** green.
3. **Concurrency cap:** verify permit lifetime. → **T2** green.
4. **`run_and_append_csv`:** inlined sweep loop, one aggregate row/cell,
   `thiserror` error, `O_APPEND` single-`write_all` per line incl. `\n`,
   header-if-empty. → **T3** green.
5. **Unknown-verdict guard.** → **T4** green.
6. **daemon re-export:** `pub use rocm_dash_collectors::bench_load;`.
   → `cargo build -p rocm-dash-daemon`.
7. **CLI:** `Command::Bench`/`BenchCommand::Load` (+ `value_delimiter`), dispatch
   arm, `"bench"` in both structured lists. → **T5** green + `cargo build -p rocm`.
8. **`dash.rs::run_bench`:** config, distinct `--out` default, `pick_first_model`,
   http-only validation, `block_on`, summary + note.
9. **Public polish:** `--help` caveat, stdout note/mode lines, empty-hint string,
   `bench_results_dir` comment fix.
10. **Gate:** `cargo test -p rocm-dash-collectors -- --test-threads=1` +
    `cargo clippy --all-targets -- -D warnings`. Manual live smoke against a real
    `rocm serve`d model (`#[ignore]`).

## 9. Residual decisions

| # | Item | Status |
|---|---|---|
| D1 | Surfaced verb name | **Resolved: `rocm bench load`** (top-level parse shape, Dash untouched). |
| D2 | `--out` default location | **Resolved:** distinct self-labeled file; shared `bench_results_dir` opt-in only. |
| D3 | "byte-compatible with external tools" | **Resolved:** dropped — we write our own fixed 14-col header. |
| D4 | https support | **Deferred:** rejected with a clear error in MVP. Add `features = ["rustls-tls"]` (0.12 spelling) only when remote https is a real requirement. |
| D5 | `engine` column consistency | Stamp once per sweep or leave `None` consistently — never vary within a sweep (RollupKey member). |
| D6 | Prometheus peaks (1.5), streaming TTFT/TPOT (1.5), `Command::RunBench` (2), auto-ramp (2) | Deferred. |

## 10. Definition of done (MVP)

- `rocm bench load --endpoint … --concurrency 1,8,32,64` runs a sweep against a
  local endpoint and appends normalized rows to a distinct CSV.
- A `rocm dash` already running shows those rows in the bench tab within one tick
  (no restart), with `gen_tps`, `prompt_tps`, `concurrency` populated.
- T1–T5 pass under `--test-threads=1`; clippy clean.
- `--help` and stdout carry the raw-throughput-vs-agent-workload caveat.
