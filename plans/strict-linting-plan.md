# Enable strictest (practical) linting for ROCm CLI

## Motivation

Linting today is minimal and partly inert:

- The workspace declares `[workspace.lints.rust]` with `unsafe_code = "forbid"`, but **no member crate sets `lints.workspace = true`** — so that lint (and any we add) is currently applied to *zero* crates.
- There is no `[workspace.lints.clippy]`; CI runs `cargo clippy ... -D warnings` against only the *default* lint set.
- Python (ruff), Shell (shellcheck), and PowerShell (PSScriptAnalyzer) run at default rule sets.

Raising lint levels catches a much broader class of correctness and quality issues. Builds on the existing prek hooks (which *run* the linters) by raising *what* they enforce.

## Description

Adopt a **pragmatic-strict** configuration across all languages:

- **Rust:** enable `clippy::all + pedantic + nursery` (warn locally, deny in CI), wired into every crate via `lints.workspace = true`. Auto-fix everything `cargo clippy --fix` handles, then hand-fix the valuable remainder. A small, **documented** allow-list suppresses high-noise/low-value lints rather than churning the codebase for them.
- **Python:** broaden ruff to a strict, mostly-auto-fixable rule selection.
- **Shell / PowerShell:** enable shellcheck optional checks and PSScriptAnalyzer (verified via prek + CI, since neither tool is installed locally).
- **Integration:** prek hooks and CI enforce the new levels identically.

No runtime behavior changes — this is tooling/quality-gate work, so no acceptance scenarios.

## Implementation approach

### 1. Rust lint config (`Cargo.toml`)
- Add `[workspace.lints.clippy]`:
  - `all = "warn"`, `pedantic = "warn"`, `nursery = "warn"` (priority `-1` so specific allows win).
  - Documented allow-list (each with a one-line reason):
    `must_use_candidate`, `missing_errors_doc`, `missing_panics_doc`, `too_many_lines`,
    `module_name_repetitions`, `doc_markdown`, `multiple_crate_versions`, `cargo_common_metadata`.
    (These are the dominant noise sources; `doc_markdown` may instead be fixed if cheap.)
- Keep `[workspace.lints.rust]` `unsafe_code = "forbid"`; add `rust_2018_idioms = "warn"` and `unreachable_pub = "warn"` (skip `missing_debug_implementations` if it proves noisy).
- Add `lints.workspace = true` to all 14 crate manifests: `apps/{rocm,rocmd}`, `crates/{rocm-core,rocm-engine-protocol,rocm-dash-core,rocm-dash-collectors,rocm-dash-daemon,rocm-dash-tui}`, `engines/{llama-cpp,atom,pytorch,lemonade,sglang,vllm}`.

### 2. Burn down Rust findings
- Run `cargo clippy --workspace --all-targets --fix --allow-dirty` iteratively (resolves the bulk: `uninlined_format_args`, `semicolon_if_nothing_returned`, `assigning_clones`, `redundant_closure_for_method_calls`, `use_self`, …).
- Hand-fix the valuable manual tail not covered by the allow-list, e.g. `missing_const_for_fn`, `needless_pass_by_value`, `map_unwrap_or`, `match_same_arms`, `redundant_pub_crate`, `redundant_clone`, `cast_*` (annotate or restructure).
- Use targeted in-code `#[allow(...)]` **with justification** only where a finding is a deliberate false positive — prefer global allow-list for whole-lint decisions.

### 3. Python (`ruff.toml`)
- Expand `[lint] select`/`extend-select` to a strict set: `E, F, W, I, B, UP, SIM, C4, RUF` (consider `RET`, `ARG`, `PTH`).
- `ruff check --fix` for the ~16 auto-fixable; hand-fix the small tail (`B904` raise-from, a few `RET`/`SIM`). Decide `E501`: rely on `ruff-format` (88 col) and `# noqa`/`per-file-ignores` for the comment/string long lines rather than reflowing.

### 4. Shell & PowerShell
- shellcheck: enable a curated optional set (e.g. `enable=all` or `add-default-case,quote-safe-variables,require-variable-braces`) via prek args + CI; fix `install.sh` + `scripts/*.sh` findings.
- PowerShell: add PSScriptAnalyzer (`Invoke-ScriptAnalyzer`) to the prek `powershell` hook and the CI PowerShell job; fix `install.ps1` + `scripts/*.ps1` findings.

### 5. Wire prek + CI
- `.pre-commit-config.yaml`: pass strict args to ruff/shellcheck; add PSScriptAnalyzer; clippy entry already `-D warnings` (now stricter via workspace lints).
- `.github/workflows/ci.yml`: keep `cargo clippy ... -D warnings`; update ruff/shellcheck/PowerShell steps to match prek.

### Verification
- `cargo clippy --workspace --all-targets -- -D warnings` passes clean.
- `cargo build --workspace --all-targets` + `cargo test --workspace --all-targets` pass.
- `cargo fmt --all --check` clean.
- `ruff check` + `ruff format --check` clean on `scripts/` + `engines/`.
- `prek run --all-files` green (covers shellcheck/PSScriptAnalyzer, which aren't installed in this sandbox).

## Tradeoffs

- **Pragmatic allow-list vs maximal strictness** (chosen: pragmatic). The allow-listed lints (`must_use_candidate`, `missing_errors_doc`, `missing_panics_doc`, `too_many_lines`, `doc_markdown`, `cargo_common_metadata`, `multiple_crate_versions`) would each require large, low-value churn (doc comments on every fallible fn, metadata on internal-only crates, unactionable transitive-dep dedup). Allowing them keeps the signal-to-noise high. They're centralized in `Cargo.toml`, so any can be re-enabled later (e.g. a follow-up to burn down `missing_errors_doc`).

## Risks / notes

- **CI uses `rust-toolchain@stable`, not a pin** (a separate effort will pin the toolchain). `nursery` lints can shift between compiler versions; mitigated by `warn`-level workspace lints + `-D warnings` in CI, and by that future toolchain pin. Worth a heads-up that a future stable bump may surface new nursery lints.
- shellcheck/pwsh not installed in this dev sandbox → those fixes are validated via prek/CI rather than locally. Stated assumption: the pinned prek `shellcheck-py` / a PSScriptAnalyzer image in CI is the source of truth.
- Expect the diff to be large but mostly mechanical (clippy `--fix`). Will keep formatting-only churn out of logic commits where practical.
