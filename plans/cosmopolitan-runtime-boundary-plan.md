# Cosmopolitan Runtime Boundary Plan

Status: in progress

This plan turns the current universal-binary runtime handling into a small,
named boundary inside rocm-cli. The goal is not to build a new compatibility
layer. Cosmopolitan remains the portability layer. rocm-cli should only keep
the host-specific decisions that are genuinely product facts: user-visible path
style, config location, process launch intent, GPU/driver probing, and
TheRock/runtime storage layout.

## Current Progress

Updated on 2026-06-07:

- Done: added `crates/rocm-core/src/runtime.rs`.
- Done: moved `RuntimePlatform`/`RuntimeHost`, runtime OS detection,
  Cosmopolitan host detection, executable suffix/name helpers, `current_exe`
  recovery, Windows/Linux path normalization, absolute-path checks, and
  config/data/cache default-root helpers into `rocm_core::runtime`.
- Done: re-exported the existing public helper names from `rocm_core` so
  existing callers keep behavior while the implementation has one source.
- Done: moved focused runtime path/current-executable tests into the runtime
  module.
- Done: routed `AppPaths::discover()`, setup pip-cache display, TheRock
  managed pip cache/tools root, ComfyUI pip cache, and Doctor pip-cache
  fallback through runtime storage helpers.
- Done: added runtime helpers for Python venv layout, activation hints,
  runtime library filenames, child-process path text, PATH splitting/joining,
  shell command selection, drive roots, directory labels, path comparison,
  protected install roots, and TCP timeout support.
- Done: migrated TheRock installer curl/PATH/Python-venv helpers, engine shell
  setup, ComfyUI runtime PATH/LD_LIBRARY_PATH assembly, setup/runtimes folder
  protection, folder-picker drive/path labels, and runtime-manager Python path
  lookups to the runtime helpers.
- Done: added `scripts/tui_e2e_smoke.py`, a PTY-driven TUI smoke test that
  launches the real TUI in isolated temp config/data/cache roots and validates
  fresh setup, main-menu help, slash-popup behavior, and setup-specific `?`
  handling. The single-exe release gate now runs this smoke on Windows.
- Done: audited the remaining `cfg!(windows)` hits in app code. The remaining
  occurrences are in tests/fixtures or are platform diagnostics, not shipped
  feature-level path decisions.
- Remaining: add higher-level `spawn_self` / `find_on_path` helpers, migrate
  daemon/service process launch call sites to those intent APIs, and add an
  audit guard against new scattered runtime decisions.

## Why This Exists

The current code works, but runtime details are scattered across feature code:

- `runtime_is_windows()` / `runtime_is_linux()` checks appear in unrelated
  flows.
- Windows path normalization, Cosmopolitan path display, `.exe` suffixes,
  `current_exe` recovery, and child-process launch decisions are mixed into
  setup, services, engines, tests, and TUI code.
- The universal APE compile target is Linux-shaped, so `cfg!(windows)`,
  `std::env::consts::OS`, and `std::path::MAIN_SEPARATOR` are not reliable
  product decisions for runtime UX.
- Bugs have shown up as repeated one-off fixes: wrong path style in WSL,
  broken selected-folder validation, daemon/helper path lookup failures, Python
  probing issues, and Windows path leakage into Linux runs.

The fix should happen at the source: one runtime boundary, one storage policy,
one process-launch policy, one scroll/UI boundary for path display, and no
feature-level path guessing.

## Upstream Guidance

The design follows the pattern used by mature Cosmopolitan applications:

- Cosmopolitan provides the portable libc and APE runtime.
- Applications still make explicit runtime decisions when behavior is
  inherently host-specific.
- llamafile centralizes host checks and file/runtime discovery behind helpers
  such as `IsWindows()` and `GetProgramExecutableName()` rather than scattering
  raw checks everywhere.
- Cosmopolitan documents that it is not a POSIX-emulation layer that erases all
  OS differences. It makes one binary possible; it does not make Windows paths,
  Linux paths, driver discovery, or user expectations identical.

rocm-cli does not need llamafile's embedded-zip or app-directory extraction
pattern. The universal rocm-cli binary should not unpack its own program files;
TheRock wheels, Python environments, ComfyUI, engines, caches, logs, and models
are installed only as explicit user-managed runtime content under the selected
install root.

References used for the plan:

- `docs/cosmopolitan-universal-binary-plan.md`
- `llamafile/docs/technical_details.md`
- `llamafile/docs/support.md`
- `llamafile/llamafile/llamafile.c`
- `llamafile/llamafile/gpu_backend.c`
- `cosmopolitan/libc/README.md`
- `cosmopolitan/ape/loader.c`
- `cosmopolitan/libc/proc/execve.c`
- `cosmopolitan/libc/proc/posix_spawn.c`

## Non-Negotiable Product Rules

- The release target is one no-extract universal APE. No sibling
  `rocm-engine-*`, `rocmd`, or OS-specific payload binaries.
- The daemon/helper process is the same executable re-entered with internal
  arguments.
- On WSL/Linux, the same file must run the Linux path. It must not silently run
  as the Windows program.
- `~/.rocm` stores only persistent user config and lightweight state needed to
  rediscover the selected install root. Tools, logs, pip cache, app installs,
  engine environments, TheRock venvs, and model/app data live under the
  user-selected install root.
- User-facing paths are native to the running host:
  - Windows shows `D:\jam\temp\therock_venvs`.
  - Linux/WSL shows `/home/...`, `/mnt/...`, or the selected Linux path.
  - Internal `/D/...`-style paths must not leak into normal UI.
- Feature code must not decide separators, executable suffixes, config roots,
  current executable paths, or runtime OS by itself.
- Setup/bootstrap must remain foolproof and non-verbose. Logs are shown only in
  install/progress cards or explicit log modals.

## Architecture

Create a `rocm_core::runtime` module and move all cross-host runtime decisions
behind it. Feature code should receive an immutable `RuntimeHost` or call a
small public API from this module.

Suggested module layout:

```text
crates/rocm-core/src/runtime.rs
crates/rocm-core/src/runtime/paths.rs
crates/rocm-core/src/runtime/process.rs
crates/rocm-core/src/runtime/storage.rs
crates/rocm-core/src/runtime/tools.rs
```

Keep the module small. If a function is not about host/runtime behavior, it
does not belong here.

## Runtime API Shape

The exact names can change during implementation, but feature code should end
up using operations like these:

```rust
pub enum RuntimeOs {
    Windows,
    Linux,
    Macos,
    Other(String),
}

pub struct RuntimeHost {
    os: RuntimeOs,
    is_wsl: bool,
    is_cosmopolitan: bool,
}

impl RuntimeHost {
    pub fn current() -> Self;
    pub fn os(&self) -> RuntimeOs;
    pub fn is_windows(&self) -> bool;
    pub fn is_linux(&self) -> bool;
    pub fn is_wsl(&self) -> bool;
    pub fn is_cosmopolitan(&self) -> bool;
}

pub struct RuntimePaths {
    pub config_dir: PathBuf,
    pub selected_install_root: Option<PathBuf>,
    pub default_install_root: PathBuf,
}

pub fn config_dir(host: &RuntimeHost) -> Result<PathBuf>;
pub fn read_selected_install_root(host: &RuntimeHost) -> Result<Option<PathBuf>>;
pub fn default_install_root(host: &RuntimeHost) -> Result<PathBuf>;

pub fn normalize_user_path(host: &RuntimeHost, text: &str) -> Result<PathBuf>;
pub fn normalize_storage_path(host: &RuntimeHost, path: &Path) -> PathBuf;
pub fn display_path(host: &RuntimeHost, path: &Path) -> String;
pub fn is_absolute_user_path(host: &RuntimeHost, text: &str) -> bool;

pub fn current_exe(host: &RuntimeHost) -> Result<PathBuf>;
pub fn spawn_self(host: &RuntimeHost, args: &[String]) -> Result<Child>;
pub fn find_on_path(host: &RuntimeHost, tool: &str) -> Option<PathBuf>;
pub fn executable_name(host: &RuntimeHost, base: &str) -> String;
```

Important rule: `normalize_user_path` is for user input. It should preserve
the running host's path model. It should not invent cross-host conversions
unless the user explicitly supplied a path that the current host can open.

## Storage Policy

Move all default root decisions into `runtime::storage`:

- `config_dir`: `~/.rocm` or env override for tests.
- `selected_install_root`: read from config/registry, not guessed from
  `~/.rocm`.
- `default_install_root`: a friendly default under the user's home only when no
  install root has been selected yet.
- `pip_cache_dir(root)`: `<selected-root>/pip-cache`.
- `logs_dir(root)`: `<selected-root>/logs`.
- `tools_dir(root)`: `<selected-root>/tools`.
- `engines_dir(root)`: `<selected-root>/engines`.
- `apps_dir(root)`: `<selected-root>/apps`.

After this migration, new writes under `~/.rocm` should fail tests unless they
are config/state files explicitly allowlisted.

## Process Policy

Move process decisions into `runtime::process`:

- `current_exe()` owns Cosmopolitan `argv0` recovery and APE loader edge cases.
- `spawn_self()` starts helper/daemon/service actions from the same executable.
- `find_on_path("python")` uses normal PATH lookup first. No hard-coded Python
  paths before PATH is checked.
- `executable_name("python")` is only a last-mile host display/lookup helper.
- Feature code must not append `.exe` manually.

The process module should use Rust `std::process` and Cosmopolitan libc through
Rust std where possible. Any required `_COSMO_SOURCE` or Cosmopolitan-specific
build patch remains in the build scripts/toolchain setup, not feature code.

## Path Policy

Path handling should become boring and predictable:

- User input is parsed once through `runtime::paths`.
- UI display calls `display_path`.
- Config/storage writes call `normalize_storage_path`.
- Command output should not expose internal temporary or APE-loader paths.
- Windows drive paths and Linux absolute paths are separate user-facing forms.
- WSL `/mnt/...` paths are Linux paths. They should remain Linux paths unless
  a Windows-host interop command explicitly requests conversion.

Tests should cover:

- `D:\jam\temp\therock_venvs`
- `D:/jam/temp/therock_venvs`
- UNC paths
- `/home/jam/rocm_venvs/default`
- `/mnt/d/jam/temp/therock_venvs`
- relative paths rejected where setup requires full paths
- storage path formatting for config JSON
- display path formatting in setup, doctor, services, ComfyUI, and logs

## Migration Plan

### Phase 1: Introduce the Boundary Without Behavior Changes

- Add `crates/rocm-core/src/runtime.rs`.
- Move the existing runtime helpers from `lib.rs` into the new module.
- Re-export only the public surface needed by existing code.
- Keep the old function names as thin wrappers temporarily if needed to reduce
  risk.
- Add focused unit tests for `RuntimeHost`, path normalization, storage roots,
  current executable resolution, and executable suffix behavior.

Exit criteria:

- No user-facing behavior changes.
- Existing tests pass.
- New runtime tests document the current Windows, WSL, and Linux behavior.

### Phase 2: Remove Feature-Level Runtime Decisions

Replace direct calls in setup, services, engines, ComfyUI, assistant, Doctor,
and TUI with the runtime module.

Searches that should trend toward zero outside `runtime` and build/test files:

```text
runtime_is_windows
runtime_is_linux
runtime_is_cosmopolitan_windows
std::env::consts::OS
std::path::MAIN_SEPARATOR
cfg!(windows)
".exe"
"/D/"
"/mnt/"
```

Some exceptions are valid in tests and in code that deliberately documents a
host-specific external tool, but they must be explicit.

Exit criteria:

- Feature code asks for intent-level operations: `spawn_self`, `display_path`,
  `config_dir`, `selected_install_root`, `find_on_path`.
- No setup/TUI/engine feature code builds platform paths by string surgery.

### Phase 3: Enforce the Storage Split

- Add tests that run setup with temporary config and selected install roots.
- Assert `~/.rocm` contains only config/state allowlist files.
- Assert pip cache, logs, tools, engines, apps, and TheRock venvs land under
  the selected install root.
- Update setup, doctor, and F1 help text to report the selected install root,
  not `~/.rocm`, for install-owned content.

Exit criteria:

- Clean first-run setup on Windows and WSL uses the selected root.
- Re-run detects existing setup from config and does not recreate `.rocm`
  clutter.

### Phase 4: Universal-Binary Process Hardening

- Route daemon/helper/engine foreground launches through `spawn_self`.
- Remove sibling-binary assumptions from feature code.
- Add universal-binary smoke tests for:
  - `doctor`
  - setup dry-run
  - service list
  - daemon/helper re-entry
  - ComfyUI status/models-path
  - assistant status
- Run the same artifact on Windows and WSL/Linux path.

Exit criteria:

- Same APE file works without sidecars.
- No feature path depends on native-only `target/debug` layout.

### Phase 5: Cleanup and Guardrails

- Add a small static audit script or test that rejects new runtime leakage in
  feature code.
- Document approved exceptions.
- Update `docs/cosmopolitan-universal-binary-plan.md` and `docs/testing.md`
  once the boundary is implemented.
- Keep the plan file as the checklist until all phases are complete.

Exit criteria:

- New platform behavior has one obvious home.
- Future path/process fixes happen in `runtime`, not in TUI/setup/engine code.

## Testing Plan

Use the normal Rust tests for unit coverage:

```powershell
cargo test --workspace --all-targets -- --test-threads=64
```

Use universal-binary acceptance for release-relevant behavior:

```powershell
python scripts\rust_cosmopolitan_spike.py build-rocm --release --clean --jobs 96
python scripts\single_exe_release_gate.py --skip-build
```

WSL/Linux path checks must use the same artifact:

```bash
sh .rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe doctor
```

E2E tests must use temporary state roots and selected install roots so the
user's real `.rocm` and TheRock folders are not tainted.

## Acceptance Checklist

- The universal binary remains a single no-extract APE.
- No `rocm-engine-*`, `rocmd`, or OS payload sidecars are required.
- Direct product code no longer scatters OS/path/process decisions.
- Windows setup accepts and displays native full folder paths.
- WSL/Linux setup displays Linux paths and does not leak Windows paths.
- `~/.rocm` contains config/state only.
- Selected install root owns tools, logs, pip cache, engines, apps, models, and
  TheRock venvs.
- Python lookup checks PATH before preparing portable Python.
- Daemon/helper launches come from the same executable.
- Universal-binary Windows and WSL smoke tests pass.
- Setup, assistant, ComfyUI, services, and logs still meet the current UX rule:
  minimal screens, verbose output only in install/progress/log modals.
