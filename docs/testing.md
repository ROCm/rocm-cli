<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Testing

This page collects developer verification commands. The README stays focused on quick-start usage.

## Local Verification

Run the Rust test suite:

```bash
cargo test --workspace --all-targets
```

Note that `cargo test` runs every test as a thread in **one process**, which is
also what the required `windows-build-and-test` lane does. See
[Tests that touch the environment](#tests-that-touch-the-environment) before
writing a test that sets an environment variable.

Run clippy with warnings as errors:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Tests that touch the environment

`std::env::set_var` / `remove_var` change state shared by every thread in the
process. Under a threaded harness two tests touching the same key race, and one
reads the other's value and fails an assertion unrelated to what it tests. This
is enforced by a test, not by an `xtask` subcommand:
`a_test_mutating_the_environment_serializes_itself` in `xtask` scans the whole
tree and **fails the build** on an unguarded mutation inside a `#[test]` or
`#[tokio::test]`. Run it on its own with:

```bash
cargo test -p xtask a_test_mutating_the_environment_serializes_itself
```

Best is not to touch the environment at all — pass the value in through a test
seam, as `newest_rocm_install_dir_in` and `engine_envs_root_from` do. A seam
cannot test the wiring it bypasses, though, so when exercising the real
env-reading path *is* the point, take a process-wide lock for the duration of
the test and restore the previous value before releasing it:

```rust
/// Stands in for the production code under test. It reads `KEY` itself, which
/// is why the test cannot use a seam and has to set a real variable.
fn production_function() -> String {
    std::env::var("KEY").unwrap_or_default()
}

/// Serializes every test in this process that replaces `KEY`.
static SOMETHING_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
#[test]
fn reads_its_setting_from_the_environment() {
    // Poison is not a failure here: a panicking sibling leaves the value
    // restored or not, and either way this test still wants the lock.
    let _guard = SOMETHING_ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = std::env::var_os("KEY");
    unsafe { std::env::set_var("KEY", "value") };
    let observed = production_function();
    match &previous {
        Some(value) => unsafe { std::env::set_var("KEY", value) },
        None => unsafe { std::env::remove_var("KEY") },
    }
    assert_eq!(observed, "value");
}
```

The guard accepts `ScopedTestEnv` or **any** static named `*_TEST_LOCK` taken
inside the test body. The suffix is the rule, so a new lock
is recognised the day it is declared rather than when someone remembers to add
it to a list. The exemption is per test function, not per file: one test taking
a lock does not cover its neighbours.

Two things the shape above gets right, both of which the guard checks only
partly:

- **Take the lock before the mutation.** A lock protects from where it is
  acquired, not retroactively. The guard enforces this ordering.
- **Use the same lock as every other test that touches that key.** Two tests
  replacing one key under two different mutexes race each other while both
  satisfy the guard — it cannot see which key a call names, because the keys
  are string literals it strips before scanning. This one is on you.

Restoring on the next line, as above, leaks the variable if the code under test
panics — and because the lock is taken with `into_inner`, the next test to hold
it reads what the panicking one left behind. `rocm-core`'s `RestoredEnvVar`
(`crates/rocm-core/src/test_env.rs`) moves the restore into `Drop`, where
unwinding runs it; prefer it where one exists. It restores but does not
serialize, so the lock is still yours to take.

Keep the mutation and the lock in the same test body. Three known blind spots,
all of which the scan reports nothing for:

- A `#[test]` that delegates its mutation to a helper — the helper's body is a
  different scope, so the scan cannot see the two together. It matches call
  text, so a wrapper hides the mutation until the wrapper itself is named in the
  scanner's `MUTATIONS` list, as `RestoredEnvVar::set(` is. Add a new restoring
  helper to that list, or it is outside the guard.
- A harness attribute that does not end in `test`, such as `#[test_case(..)]` or
  `#[rstest]`. Neither is used in this tree.
- `#[cfg_attr(unix, test)]`, for the same reason: the attribute's path reads as
  `cfg_attr`. Write `#[cfg(unix)]` above `#[test]` instead, which the scan
  arms on.

### Tests that run a fake executable

Plant a stand-in `python`, `uv`, `amd-smi`, ... with
`rocm_core::fake_executable::write_fake_executable`, not `std::fs::write`
followed by `set_permissions(0o755)`. On Linux the plain write makes the first
run of the fake fail intermittently with `Text file busy (os error 26)`: while
the write fd is open, a test on another thread can spawn a child, and that child
keeps a copy of the fd until its own `exec` (`O_CLOEXEC` closes it only then).
The kernel refuses to execute a file any process still has open for writing, so
the fake's first run can land in that window. The helper has a short-lived
child write the file instead, so this process never holds a write fd on it. The
same applies to E2E steps: scenarios run concurrently in one process, and the
`rocm` child that runs the shim can be the one refused.

Run the cross-platform smoke test:

```bash
python scripts/smoke_local.py
```

If the workspace is already built:

```bash
python scripts/smoke_local.py --skip-build
```

## Coverage floors

Every workspace crate (except the `e2e-cucumber` harness) has a committed line-coverage
floor in `coverage-floors.toml`. CI fails when a crate drops more than a small tolerance
below its floor, so deleting a test is a check failure rather than a silent loss.

Check the floors locally. This needs both the instrumentation tooling and the LLVM
tools the toolchain ships separately — CI installs the same two:

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
cargo xtask coverage
```

After adding tests, ratchet the floors up to the new measurement:

```bash
cargo xtask coverage --bless
```

`--bless` rewrites `coverage-floors.toml`; commit the result. Lowering a floor is
allowed but deliberate — it shows up as a diff a reviewer has to approve, so say in the
commit why coverage legitimately dropped (a crate shrank, tests moved elsewhere) rather
than re-blessing to make a red check go away.

Adding a workspace crate fails the check until that crate has a floor, so a new crate
cannot land outside the gate. A crate that no test binary compiles at all never shows up
in the coverage report; both the check and `--bless` fail on it by name, and the way out
is tests or an entry in `EXCLUDED` in `xtask/src/coverage.rs` with the reason — there is
no measurement to bless a floor from.

These are `cargo llvm-cov` line percentages, which count `#[cfg(test)]` modules as
covered source. That inflates the numbers and damps the gate — see the module comment in
`xtask/src/coverage.rs` for what the measurement is and is not good for.

## Remote control-channel checks

`rocm remote` drives `ssh`, `scp`, and the remote machine's own tooling. Its unit
tests use a scripted stand-in, which proves control flow but assumes those tools
behave a certain way. Two scripts check that assumption against a real OpenSSH
server in a container — no GPU, no ROCm, no tailnet, since the remote's `rocm`
and `tailscale` are stand-ins:

```bash
tests/remote-ssh/run.sh       # the tools and their contracts
tests/remote-ssh/run-e2e.sh   # the real binary, end to end
```

`run.sh` covers argument handling, exit-code propagation, delivering a
credential on stdin, file copy, batch-mode refusal, that withdrawing a
published endpoint actually removes it, and the shape Tailscale Funnel takes in
the serve config — including that a port Funnel cannot serve is refused.

`run-e2e.sh` runs the built `rocm` through the whole flow — discover, probe,
serve, publish, reconcile status, re-publish after an out-of-band withdrawal,
tear down, and refuse to publish over a Funnel-exposed port — and needs
`cargo build -p rocm` first. It does not prove the endpoint carries traffic;
that needs a real two-node tailnet.

The `rocm remote` cucumber scenarios that need a second machine carry
`@requires-docker` and run only when `E2E_INCLUDE_DOCKER=1`, which the
GitHub-hosted `E2E tests` lane sets. A working daemon alone is not enough — the
self-hosted GPU runners have one but cannot reach the package mirror the fixture
image builds from, so they skip those scenarios. To run them locally:

```bash
E2E_INCLUDE_DOCKER=1 cargo xtask e2e
```

On a network that intercepts TLS, point the container's package manager at
plain-HTTP mirrors:

```bash
export ROCM_TEST_APK_REPOS="--repository http://dl-cdn.alpinelinux.org/alpine/v3.20/main \
  --repository http://dl-cdn.alpinelinux.org/alpine/v3.20/community"
```

## CI test selection

On pull requests, CI does not test the whole workspace. The `test` job runs
`cargo nextest run` over only the crates a change can reach — the changed crates
plus their transitive dependents, derived from the workspace dependency graph by:

```bash
cargo xtask affected --base origin/main
```

which prints the cargo package flags to test (`--workspace`, or `-p <crate> …`,
or nothing when no crate is affected). It falls back to `--workspace` for any
change that can't be confined to specific crates (the lockfile, the toolchain
file, the workspace root manifest, CI config, or a doc an xtask test reads —
`MANIFEST.md`, `CONTRIBUTING.md`, `docs/ci-hardware-testing.md`,
`tests/e2e-cucumber/README.md`). On `main`
and in the merge queue the full workspace always runs. For local verification, keep using
`cargo test --workspace --all-targets` above — `affected` is a CI optimization.

The smoke path is the cross-platform local no-fallback acceptance surface. It
uses an isolated config/data/cache root, does not install TheRock wheels, does
not require a managed runtime, and verifies:

- first-run examine and engine inventory state
- telemetry-off config behavior
- GPU-required recipe planning for `tiny-gpt2`
- exact-runtime rejection for `rocm engines install` before any runtime exists
- direct vLLM adapter and `rocm serve` CPU policy rejection
- GPU-required paths fail loudly instead of silently falling back
- no first-run pip cache or runtime registry is created

Build the platform-native release artifact:

```bash
python scripts/build_single_exe_release.py standalone
```

On Windows this writes `.rocm-work/standalone-release/rocm.exe`; on Linux it
writes `.rocm-work/standalone-release/rocm`. The artifact is the rocm-cli binary
itself, not a self-extracting launcher and not a model bundle. Running it with
no arguments opens the normal rocm-cli launcher; choose "Set up this system"
there to run first-time setup (it does not open automatically).

rocm-cli ships native per-OS binaries; there is no cross-OS universal binary.
Build and test the binary natively on each supported target (native Windows,
Ubuntu/Fedora on WSL, and Linux on server/bare-metal):

```bash
cargo build --workspace --release
cargo test --workspace
```

On WSL/Linux, build and run the native Linux binary directly. The examine
output on WSL must include `os: linux` and `wsl: true`, and must not include
`os: windows`.

Use isolated `ROCM_CLI_CONFIG_DIR`, `ROCM_CLI_DATA_DIR`, and
`ROCM_CLI_CACHE_DIR` roots for smoke tests, then delete those roots after the
test so the user's real `.rocm` state stays clean.

Focused examine guidance coverage:

```bash
cargo test -p rocm-core examine_render
cargo test -p rocm-core managed_sdk_probe
cargo test -p rocm --bin rocm examine_runtime_state_reports_ambiguous_default_runtime_id
cargo test -p rocm --bin rocm engine_runtime_selection_rejects_ambiguous_default_runtime_id
cargo test -p rocm --bin rocm permissions_
```

Self-hosted GPU CI smoke is intentionally non-mutating. The MI300X job builds
the workspace, runs `rocm examine`, then runs `detect` and `capabilities` for
all first-party engine adapters: Lemonade and vLLM.
Live serving acceptance remains separate because it needs engine-specific
runtime installs, model artifacts, and supported upstream GPU targets.

WSL live examine sanity after building with a Linux target directory:

```bash
export CARGO_TARGET_DIR=/home/user/.cache/rocm-cli-target
cargo build --workspace
rocm examine
```

Expected WSL fields when a managed TheRock runtime is registered and ROCDXG is
ready:

- `detected_gfx_target: gfxNNNN` from sysfs, PATH ROCm tools, or the managed
  TheRock SDK tool path recorded in the runtime registry
- `compatible_therock_family: ...`
- `active_runtime_status: ready`, `unset`, `missing_manifest`, or
  `ambiguous_runtime_id`
- if ambiguous, `active_runtime_matches` plus the exact
  `rocm runtimes activate <runtime_key>` action

Ambiguous runtime selectors should also fail before engine install or serve
launches:

```bash
rocm engines install vllm
rocm serve qwen --engine vllm --managed
```

On a host where `default_runtime_id` matches multiple registered runtime keys,
both commands should report the matching runtime keys and ask for
`rocm runtimes activate <runtime_key>` or an explicit runtime key.

Focused TheRock metadata cache coverage:

```bash
cargo test -p rocm --bin rocm metadata_cache
cargo test -p rocm --bin rocm metadata_signature_verification_accepts_generated_key_and_rejects_tamper
cargo test -p rocm --bin rocm http_header_value
rocm install sdk --channel release --format wheel --dry-run
```

## TheRock SDK Install Test

The live SDK acceptance test creates an isolated test root under `target/`, creates a local bootstrap Python venv, runs:

```bash
rocm install sdk --channel release --format wheel --yes
```

`--yes` approves replacing whatever managed runtime is currently the active
default without prompting, which keeps the command non-interactive when the test
root is reused across runs (a root with no active default runtime never
prompts). The gate is not scoped to the family or channel being installed, so
`--yes` is needed on a reused root even when the install targets a family that
root has never held. It matches the invocation in
`scripts/therock_sdk_install_test.py`.

`--yes` is used here because this test also wants the second approval it
carries: installing required system packages with `sudo`. When all you need is
to clear the active-default gate — the usual case for a script or a CI job —
pass the narrower `--approve-replacing-active-default` instead. That is the flag
the refusal message itself recommends, and the only one ROCm CLI's own
terminal-less surfaces pass. Without either flag, the same command on a reused
root prompts when a terminal is attached and fails outright when one is not; the
failure names the flag to add, so read the message before treating it as a
regression. Check both routes by hand after changing the gate:

```bash
rocm install sdk --channel release --format wheel --approve-replacing-active-default
rocm install sdk --channel release --format wheel < /dev/null   # expect the refusal
```

The preview path is unaffected: `--dry-run` returns before the gate is
consulted, so `rocm install sdk --channel release --format wheel --dry-run`
never prompts and never refuses, whatever the active default is.

Then it verifies:

- runtime manifest metadata
- localized pip cache inside the selected ROCm runtime folder at
  `<install-root>/pip-cache`, including both generated managed folders and
  explicit `--prefix` folders
- the installer does not pre-create that pip cache during dry-run or setup;
  pip creates it inside the ROCm folder when packages are downloaded
- a single TheRock-index pip install plan for pinned `rocm`, `torch`, and
  `torchvision` requirements with exactly one `device-<detected-gfx-target>`
  extra (`rocm` also requests `libraries`), plus pinned `torchaudio`, and that
  the toolchain is not planned for unless asked (pass the script `--devel` to
  check the opt-in path instead, which adds `devel` to the `rocm` extras)
- on a host with no detectable AMD GPU the preview reports `device_target:
  undetermined` and renders the device extra as a placeholder; a real install
  refuses rather than falling back to every published device payload
- package selection uses the newest exact ROCm build suffix common to the SDK
  package and the PyTorch stack for the current Python/platform wheel tags
- `python -m rocm_sdk version`
- `python -m rocm_sdk targets`
- runtime-only TheRock wheel discovery through
  `rocm_sdk.find_libraries("amdhip64", "hipblas")`
- vLLM adapter discovery of the managed TheRock runtime environment

Run the full live test with auto-detected GPU family:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --fresh
```

Run only the resolver/harness path without downloading SDK wheels:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --dry-run --check-windows-tools
```

Verify the first-time-setup style explicit folder path without downloading SDK
wheels:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --dry-run --prefix .rocm-work/tests/rocm-folder
```

Use a deterministic family only for developer tests that need fixed package
selection:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --fresh --family gfx120X-all
```

### GPU targets with no ROCm build

Some recognized targets have no TheRock package family at all. Today that is
`gfx90c`, the Vega iGPU in Renoir, Lucienne, Cezanne and Barcelo APUs. For these
`rocm install sdk` refuses instead of resolving a family, and `rocm examine`
adds a `rocm_support` line saying the GPU is detected but unsupported.

Two things make this testable without a Ryzen 4000/5000 laptop, and both matter
when changing it:

- `gfx_target_has_no_rocm_build` in
  [host_gpu.rs](../crates/rocm-core/src/host_gpu.rs) is deliberately conjoined
  with `normalize_therock_family` returning `None`. Do not simplify it to a bare
  target-name check. Every test that drives the refusal does so by naming the
  target explicitly, and the conjunct is the only reason any of them is also
  evidence about the mapping: it is what makes the refusal stop firing once the
  mapping comes back. Were the predicate to carry its own copy of the target
  name alone, the refusal would keep firing from that copy and those tests
  would stay green across a reopened catch-all.
  `no_rocm_build_implies_no_therock_family` pins the conjunct.
- detection itself has no override — `detect_host_gfx_target` reads real sysfs
  or WMI — so no lane can fake the host half. The end-to-end coverage is
  `@id:runtime-install-sdk-refuses-unsupported-gpu-target` (`runtime-19` in
  [runtime_setup.feature](../tests/e2e-cucumber/features/runtime_setup.feature)),
  which goes through `--family gfx90c`. It runs on the ordinary E2E lane, with
  no GPU tag. The detected-GPU wording itself is covered at unit level by
  `unsupported_gpu_error_names_the_gpu_and_does_not_suggest_families` in
  [therock.rs](../apps/rocm/src/therock.rs).

The detected-host door — the one a real Ryzen 4000/5000 laptop opens — has no
end-to-end lane at all, and that gap is stated here rather than papered over.
`runtime-19` opens the `--family` door. Do not read a green `runtime-19` as
coverage of host detection.

#### Which mutants each test actually kills

Measured on Linux with `cargo test -p rocm-core -p rocm --all-targets
--no-fail-fast` followed by `cargo xtask e2e -- -n "runtime-19"`. Re-run them
rather than trusting this list if you change either function, and confirm the
edited file really differs first: a mutant that fails to apply is
indistinguishable from one that survived.

1. **Delete the `gfx90c` arm from `normalize_therock_family`.** `runtime-19`
   goes **red** — the first `Then` fails because the dry run now succeeds,
   resolving `gfx90X-dcgpu`. Also red:
   `gfx_target_has_no_rocm_build_names_only_gfx90c`,
   `normalize_therock_family_has_no_family_for_gfx90c`,
   `examine_render_says_a_gpu_without_a_rocm_family_is_unsupported`,
   `windows_display_parser_prefers_a_supported_gpu_over_a_vega_igpu`,
   `therock::gfx90c_resolves_to_no_family_instead_of_the_datacenter_one` and
   `therock::requesting_gfx90c_by_name_fails_instead_of_installing_another_family`.
   Note `no_rocm_build_implies_no_therock_family` does *not* fail here: the
   predicate answers `false`, so the implication it asserts holds vacuously.
2. **Reduce the predicate to `normalize_therock_family(gfx).is_none()`.**
   `runtime-19` stays **green** — gfx90c still has no family, so the refusal
   still fires. Caught at unit level only, by
   `gfx_target_has_no_rocm_build_names_only_gfx90c` and
   `examine_render_omits_the_support_note_for_supported_or_missing_gpus`, which
   pin that the refusal is about gfx90c and not about every unmapped target.
3. **Reduce the predicate to the bare `starts_with("gfx90c")` check and delete
   the mapping arm.** `runtime-19` stays **green** through exactly the
   regression it exists to catch. This is the drift the conjunct forbids, and
   the only mutant `no_rocm_build_implies_no_therock_family` fires against;
   `normalize_therock_family_has_no_family_for_gfx90c`,
   `examine_render_says_a_gpu_without_a_rocm_family_is_unsupported` and
   `therock::gfx90c_resolves_to_no_family_instead_of_the_datacenter_one` go red
   with it.

Read 1 and 3 together. `runtime-19` is evidence about the mapping, but only
transitively, through the conjunct in the predicate — remove the conjunct and
the scenario stops being evidence about anything but itself. The direct
evidence about the mapping is the unit tests.

## ComfyUI TheRock GPU Acceptance

This opt-in test verifies that rocm-cli can install or reuse its managed
ComfyUI app, start it through the active TheRock ROCm runtime, reach the local
ComfyUI HTTP endpoint, report status/logs, and stop the launched process
without CPU fallback.

Run the offline harness self-test:

```bash
python scripts/comfyui_therock_gpu_test.py --self-test
```

Run the live test after first-time setup has installed and selected a managed
ROCm runtime:

```bash
python scripts/comfyui_therock_gpu_test.py
```

Run the live test with isolated rocm-cli state while reusing only the existing
managed TheRock runtime records:

```bash
python scripts/comfyui_therock_gpu_test.py --temp-state --copy-runtime-state-from %USERPROFILE%\.rocm
```

That command copies `config.json` and `runtimes/` into a temporary
`ROCM_CLI_CONFIG_DIR`, `ROCM_CLI_DATA_DIR`, and `ROCM_CLI_CACHE_DIR`, installs
ComfyUI in that temporary app state, then removes the temporary state after it
stops the process it started.

To reuse an already installed ComfyUI app without reinstalling dependencies:

```bash
python scripts/comfyui_therock_gpu_test.py --skip-install
```

To run a real image-generation smoke test, add `--generate-cat`. The harness
places a safetensors checkpoint in ComfyUI's checkpoint folder when needed,
submits a cat text-to-image workflow through the ComfyUI HTTP API, waits for
history completion, downloads the PNG, and rejects CPU-only device reports:

```bash
python scripts/comfyui_therock_gpu_test.py --generate-cat --output-dir .rocm-work/tests/comfyui-cat
```

The live test uses `rocm comfyui install`, `rocm comfyui start`, `rocm comfyui
status`, `rocm comfyui logs`, and `rocm comfyui stop`. It starts ComfyUI with
`--no-open-browser`, waits for `/system_stats`, verifies that
`rocm comfyui status` says `status: running`, and stops the process it started
unless `--keep-running` is set.

Manual non-TUI launch:

```bash
rocm comfyui install
rocm comfyui start
```

`rocm comfyui start` prints the local browser URL, normally
`http://127.0.0.1:8188`. It must also print `AMD GPU check: ready`; if that
check fails, rocm-cli stops instead of launching ComfyUI in CPU mode.

For WSL, run from the WSL filesystem instead of `/mnt`:

```bash
cd /home/$USER/rocm-cli-work/rocm-cli
CARGO_TARGET_DIR=/home/$USER/rocm-cli-work/target-linux cargo build -p rocm
python3 scripts/comfyui_therock_gpu_test.py \
  --rocm /home/$USER/rocm-cli-work/target-linux/debug/rocm \
  --temp-state \
  --copy-runtime-state-from /home/$USER/.rocm \
  --generate-cat
```

## ComfyUI Reinstall

`rocm comfyui install --reinstall` deletes only the top-level entries the
installed ComfyUI release shipped in its `source/` folder (as recorded when it
was installed), keeps the preserved ones (`models/`, `user/`, `output/`,
`input/`, `custom_nodes/`, `datasets/`, `extra_model_paths.yaml`), renames an
entry of the user's that the new release starts shipping (`set aside:`) and
leaves anything else in place. An install from before that record replaces
the entries named like the new release's code and lists them (`replaced:`).
It leaves the install untouched when the new source cannot be obtained or is
not a ComfyUI release, or when ComfyUI is found running just before the swap;
`start` refuses a folder whose swap was interrupted or stopped part-way, and
the command it names finishes it. The swap is unit-tested directly, without a
fake Python. Most of those tests also run on Windows: stopping the swap before
each filesystem change in turn and checking that a re-run converges (with and
without an entry to set aside), unknown entries and `datasets/` left in place,
an unreadable or malformed release record or swap marker, an entry that
cannot be removed, the copy taken when the release cannot be renamed into
place across filesystems, `main.py` removed first and moved in last, a
release without `main.py` refused before an existing install changes, and the
`replaced:` list of an install with no record staying complete when its swap
is interrupted at any step. The symlink tests (symlinked model folders, links
inside them, loops, dangling links, copying links without following them), the
inode/ctime test and a copy that fails part-way are Unix-only; the retry while
another program holds a file and names that differ only in case are
Windows-only:

```bash
cargo test -p rocm --bin rocm comfyui::source_swap
cargo test -p rocm --bin rocm comfyui::tests
```

The second filter covers the install-level tests: `reinstall_*`,
`install_after_an_interrupted_reinstall_finishes_it`,
`start_refuses_an_interrupted_reinstall_until_install_finishes_it`,
`a_reinstall_that_*`, `a_reinstall_proceeds_over_an_unreadable_saved_state`,
`a_failure_after_the_swap_names_what_it_replaced`,
`only_an_install_that_put_new_code_in_place_names_the_command_that_finishes_it`,
`swap_changes_summary_names_what_was_replaced_and_set_aside`,
`a_plain_install_is_not_refused_while_comfyui_runs`,
`plain_install_dry_run_previews_no_reinstall`,
`a_refusal_just_before_the_swap_changes_nothing` and `chat_status_answer_*`.
The running guard is tested against a ComfyUI answering on its port
(`running`) and one whose process is up but whose port is not answering yet
(`starting`).

The e2e scenarios `comfyui-05` to `comfyui-13`
([comfyui.feature](../tests/e2e-cucumber/features/comfyui.feature), Linux, no
GPU) cover the same through the built binary: what is kept, replaced and left
in place (also when the saved run state is unreadable), a failed download, the
dry run, a reinstall refused while ComfyUI runs (which the dry run says
first, and which is refused before anything is downloaded), `start` refusing a half-replaced folder until the command it names is
run, the answer to a plain-words status question
(`rocm "what is the comfyui status"`) while ComfyUI starts and while a
reinstall is unfinished, a folder of the user's set aside when the new release
ships one of the same name, a reinstall that fails after the swap naming
what it replaced and the command that finishes it, a ComfyUI started while
the reinstall downloads, which the check just before the swap refuses, and a
download that is not a ComfyUI release, which is neither installed nor kept:

```bash
cargo xtask e2e -- -n "comfyui-(0[5-9]|1[0-3])"
```

## Runtime Selection And Activation

List registered managed or read-only runtimes and their exact side-by-side keys:

```bash
rocm runtimes list
```

Activate one validated runtime:

```bash
rocm runtimes activate <runtime_key>
rocm runtimes activate <runtime_key> --restart-services --yes
```

New engine installs and new serve launches use this active runtime by default.
Already-running managed servers do not follow the activation: each one keeps
the runtime recorded in its service record until it is restarted. The
activation report says so from that recorded state rather than as a fixed
note — it prints `services_on_previous_runtime: <n>` with one line per stale
service (`- <service_id> engine=<engine> recorded_runtime=<key>`), and
`services_on_previous_runtime: 0` when nothing is left behind. Live services
whose record names no runtime are counted separately under
`services_with_unrecorded_runtime:`. Only services on an engine that
manages its own runtime (`lemonade`) are never counted as stale; an `env_id`
recorded in a service record is not a pin and does not exempt the service.

`--restart-services` moves those services: it rewrites each service record onto
the newly active runtime and then restarts it. Moving a service also clears any
`env_id` stored in its record; the engine writes its own back on the next
launch. It requires `--yes` and never prompts, the same idiom as
`rocm services restart <id> --yes` and `rocm runtimes uninstall --yes`. Restart
is best effort per service — every failure is named, its record is put back on
the runtime it actually ran on, and the command exits non-zero. After a failure
the service is read back: one that is still live stays counted under
`services_on_previous_runtime`, and one that is not is reported as stopped by
the attempt. The error says which is which. `services_restarted` lists only
servers that answered an inference request after the restart; one that was
respawned but had not answered when the restart stopped waiting (45 seconds) is
listed under `services_restarted_not_yet_serving` instead, and does not fail the
command.

Activating the runtime that is already active changes nothing: the report prints
`changed_from_runtime_key: <unchanged>` and the existing rollback target is kept,
so the activation command in the report's own note can be run safely.

If no runtime is active, pass `--runtime-id` explicitly or activate one first;
the CLI does not fall back to a built-in TheRock selector.

Normal user testing should switch versions by activating the exact runtime key
printed by `rocm runtimes list`. The TUI equivalent is `/runtimes`, then arrow
to an installed ROCm entry and press Enter.

A runtime selector is matched against keys exactly first. If it matches no key
exactly but more than one key ignoring letter case — possible on a
case-sensitive filesystem, where `Foo` and `foo` are separate registry entries —
every command that resolves a runtime selector through the shared resolver
refuses and lists the matching keys instead of picking one. Two exceptions
remain: ComfyUI's installer (`comfyui install --runtime-id`) keeps its own lookup
and still takes the first case-insensitive match, and `config set-default-runtime`
and `config set-engine --runtime-id` store a selector without resolving it. For
`runtimes activate`, `@id:runtime-lifecycle-case-twin-selector-refused` also
checks that nothing became active and that naming one key exactly then works.

Developer-only previous-runtime regression check:

```bash
rocm runtimes rollback
rocm runtimes rollback --restart-services --yes
```

This command validates the saved previous-runtime marker, but it is not the
primary user-facing way to switch ROCm installs. It reports and restarts
running services exactly as `activate` does.

Import an existing rocm-cli/TheRock runtime manifest in read-only mode:

```bash
rocm runtimes import /path/to/.rocm-cli-runtime.json
```

Adopt an existing TheRock Python environment in read-only mode without writing
into that environment:

```bash
rocm runtimes adopt \
  --python /path/to/venv/bin/python \
  --root /path/to/venv \
  --runtime-id therock-release:gfx120X-all \
  --runtime-key adopted-release-pip-gfx120x-all-7-13-0
```

Focused unit coverage:

```bash
cargo test -p rocm --bin rocm runtime_activation
cargo test -p rocm --bin rocm runtime_import
cargo test -p rocm --bin rocm runtime_adopt
cargo test -p rocm --bin rocm runtime_key_includes_version_for_side_by_side_installs
cargo test -p rocm --bin rocm engine_plugin
cargo test -p rocm --bin rocm missing_packaged_engine_reason
cargo test -p rocm --bin rocm model_registry
cargo test -p rocm --bin rocm render_model_registry_text_reports_host_ram_fit
cargo test -p rocm --bin rocm model_recipe
cargo test -p rocm --bin rocm logs
cargo test -p rocm-core model_recipe
cargo test -p rocm-core load_model_recipe_index
cargo test -p rocm --bin rocm update_report_policy
cargo test -p rocm-engine-lemonade stdio_protocol_routes_all_methods_without_side_effects
cargo test -p rocm-engine-vllm stdio_protocol_routes_all_methods_without_side_effects
cargo test -p rocm-engine-vllm
cargo test -p rocmd event_collector
cargo test -p rocmd event_dispatcher
```

## Uninstall And Download Removal

`rocm uninstall` and `rocm storage remove-downloads` both end in the same
removal helper. Their property tests build a real sandbox tree with planted
symlinks (live, dangling, and spelled with a trailing `/` or `/.`), sibling
folders whose names are prefixes of a removed root, and sentinel files outside
it. Each case runs the real plan and removal code, then compares the tree by
`(dev, ino)` against what the review listed:

```bash
cargo test -p rocm --bin rocm deletion_properties
ROCM_DELETION_PROP_CASES=4096 cargo test -p rocm --bin rocm deletion_properties
```

The default is 256 cases; `ROCM_DELETION_PROP_CASES` raises it for a deeper
local run. They are Unix-only because they plant symlinks.

The `uninstall-NN` E2E scenarios (`features/uninstall.feature`) check the same
behaviour through the built binary, confined to the scenario's own folder,
including a folder that cannot be checked stopping the uninstall before
anything is removed:

```bash
cargo xtask e2e -- -n "uninstall-"
```

## Provider-Assisted Planning

The deterministic planner remains the default. Optional LLM/provider ambiguity
resolution is only used after a user configures a planner provider:

```bash
rocm config set-planner-provider local
rocm "start a local model"
```

For OpenAI or Anthropic, prompt sending must also be enabled and a key must be
available through the OS secure store or session environment:

```bash
rocm config set-planner-provider openai
rocm config enable-provider openai
rocm config set-provider-key openai
```

Provider-assisted output is reduced to a validated `rocm` tool call. It is
shown for review in the TUI, but non-interactive `rocm --yes ...` does not
execute provider-assisted plans automatically.

Focused coverage:

```bash
cargo test -p rocm --bin rocm provider_planner
cargo test -p rocm --bin rocm freeform_execution_validation_rejects_provider_assisted_plans
```

Log browser smoke:

```bash
rocm logs
rocm logs --search server
rocm logs --service <service-id>
```

The no-log first-run path should report `no CLI logs found yet` without
creating the configured `data/logs` directory.

TUI log pagination focused tests:

```bash
cargo test -p rocm --bin rocm logs_browser
```

After opening `/logs` or `/logs --search <query>` in the TUI, use the arrow-key
menu instead of typing paging subcommands. `Left`/`Right` page the log browser,
`PageUp`/`PageDown` scroll the details, `R` refreshes, and the `Show file
locations` row reveals advanced paths only when needed.

Engine inventory smoke on native Windows:

```powershell
rocm engines list
```

The packaged Linux/WSL-only vLLM adapter should render
`runtime: unsupported_native_windows`, not `runtime: not found`.
The vLLM live GPU acceptance script should return a clean skip on native
Windows. It remains a strict GPU-required test on Linux/WSL.

Serve resolver focused tests:

```bash
cargo test -p rocm --bin rocm serve_engine_selection
```

These tests verify that explicit/configured engines still win and shared model
recipes only choose a preferred engine when no user/default engine is set.

Public-endpoint authentication focused tests:

```bash
cargo test -p rocm --bin rocm resolve_endpoint_auth
cargo test -p rocm --bin rocm endpoint_client_config
cargo test -p rocm --bin rocm api_key_client_config
cargo test -p rocm --bin rocm public_bind_fails_closed
cargo test -p rocm --bin rocm respawn_fails_closed
cargo test -p rocm --bin rocm respawn_allows_loopback
cargo test -p rocm --bin rocm restart_refuses_a_public_service
cargo test -p rocm-core generate_endpoint_api_key
cargo test -p rocm-engine-protocol endpoint_api_key
cargo test -p rocm-engine-protocol is_public_bind_host
cargo test -p rocmd recovery_respawn_fails_closed
cargo test -p rocmd apply_endpoint_key_env
```

A loopback bind (`rocm serve <model>`, default `127.0.0.1`) stays
credential-free. A public bind (`--host 0.0.0.0 --allow-public-bind`) requires an
API key: pass one with `--api-key` / the `ROCM_SERVE_API_KEY` environment
variable, or let one be generated. The key is handed to the engine via a 0600
key file (never argv), persisted as a 0600 per-service file (not the OS keychain,
which is unavailable on headless serving hosts), and printed once as client
configuration — it must never appear in `rocm services`, `rocm logs`, or the
audit log. Verifying that the running server actually *rejects* an
unauthenticated request requires a live engine; that end-to-end assertion is a
deferred follow-up and is **not yet** exercised by the GPU acceptance scripts
(`scripts/vllm_therock_gpu_test.py`). The unit tests cover key policy,
generation, the 0600 key file, and key-file resolution.

The same requirement holds across respawns, because the invariant belongs to the
recorded *host*, not to the key file: a service recorded on a public host refuses
to respawn once its key is gone, rather than coming back unauthenticated. That
covers `rocm services restart` and `rocmd`'s recovery supervisor, and the refusal
happens before anything is stopped, so it cannot take down a running service. A
stop therefore drops the key only once it has confirmed the engine is gone —
otherwise the CLI would lock itself out of a service that is still running and
still enforcing the key — and the deferred cleanup lands on the liveness refresh
that later observes the process dead. `service-stop-01`
(`features/service_stop.feature`) covers `services stop` succeeding against a
running record on the mock lane and the record being persisted as stopped;
there is still no e2e coverage of the key being dropped or `services restart`,
which remain unit-tested only. Reusing a running service for a
`--require-api-key` request it never had is refused rather than silently
upgraded — `serve-24` (`features/model_serving.feature`, Lemonade GPU lane,
Linux-only since the Windows managed Lemonade spawn can't enforce the key)
covers the local refusal and that its printed stop hint actually clears it;
`remote-15` (`features/remote.feature`, `@requires-docker`) covers the same
refusal on `rocm remote serve`, reusing the fake remote's own deterministic
service id.

`rocm services remove` / `rocm services prune` are the one place a key file is
dropped for a service that was never stopped: the record itself is being
deleted, so keeping its key would strand a 0600 secret belonging to a service
that can no longer be restarted. Neither will touch a record the liveness
refresh still reads as running, and `prune` re-checks liveness immediately
before each delete — but the check and the delete are not atomic, so that
narrows the race against a concurrent `restart` rather than eliminating it.
`features/service_record_cleanup.feature` covers both commands on the mock lane,
asserting on the files left on disk rather than on the summary line the command
prints — the defect they exist to fix is a file being left behind, which a
summary claiming success cannot reveal.

`prune`'s leftover sweep carries a second, unrelated race: a file with no record
beside it is also what a *launch in progress* looks like, because `rocm serve`
writes the 0600 endpoint key before it writes the record, and `--any-age` leaves
no age rule to hide that window behind. `serve` already holds the shared
managed-launch lock across both writes, so `prune` acquires the same lock before
it reads the directory.

Testing that is awkward, because `FileLock::acquire` blocks and has no `try_`
variant: a test that took the lock and then called `prune` on the same thread
would simply deadlock. Both tests therefore stage the launch from a second
thread and let the code under test block on it.

- `services_prune_waits_for_the_managed_launch_lock_before_sweeping`
  (`apps/rocm/src/main.rs`) holds the lock on the main thread with only the key
  written, runs `prune` on a worker, asserts the worker does *not* report
  completion while the lock is held, then publishes the record and releases.
  Two assertions fail independently if the acquire is removed: that negative
  wait, and the endpoint key still being on disk with its original value. The
  second only works because the record is published with
  `plant_service_record_without_its_key` — the full planter rewrites the key, and
  a rewritten key is present at the end whether or not the sweep deleted it,
  which would leave only the timing assertion doing real work. The negative wait
  in turn cannot pass vacuously: the worker physically cannot report anything
  without first holding a lock the main thread has.
- `service-cleanup-07` does the cross-process half, which the unit test cannot:
  a thread holds the real `launch.lock` with `rocm_core::FileLock` while a
  separate real `rocm services prune --any-age --yes` process runs, and only
  publishes the record after a fixed delay sized to outlast that process's
  startup. A fixed delay against a variable startup fails one-sidedly in the
  unhelpful direction — a slow runner lets the prune arrive after the record is
  already published, where the key survives for a reason unrelated to the lock.
  The scenario's discriminating assertion is the key surviving, and it only
  discriminates while the prune's startup is shorter than the hold. The wall
  clock it also asserts (`the prune blocked until the launch published its
  record`) is a one-sided sanity bound, not a cure for that: `cli_elapsed`
  brackets the whole child process, so a long elapsed time does not establish
  that the prune blocked on the lock. It catches a prune that returned *too
  fast* to have waited; it cannot catch one that was merely slow to start.

Both go red if the `FileLock::acquire` is removed from
`prune_managed_service_records`, and `services_prune_sweeps_engine_state_left_behind_by_a_deleted_record`
still pins that real leftovers are swept at any age, so a fix that simply
stopped sweeping could not pass either.

What neither pins is the lock's *scope*. `prune_managed_service_records` argues
for holding the guard across the apply phase as well as the scan, and both tests
still pass if it is released after the plan is built. That is not an oversight to
fix with another test: the only behaviour the wider span changes needs a service
id to repeat across runs (a backwards clock step), and a test that instead tried
to slip a launch in between the two phases would be racing a microsecond-wide
window and would pass on a timing-lucky run rather than flake. The scope is a
documented conservative choice, not a covered property.

Windows + Lemonade note: the Windows *managed* native-Lemonade server is launched
via `spawn_hidden_console_with_log`, whose env-override API is path-valued only,
so it cannot receive the value-typed `LEMONADE_API_KEY` that Lemonade's server
reads. Rather than launch an unauthenticated public server, `rocm serve` **fails
closed** on that exact combination (public bind + `--engine lemonade` + Windows)
and directs the user to `--engine vllm` or a loopback host. vLLM (all platforms,
via `VLLM_API_KEY`), the packaged llama-server fallback (via `--api-key-file`),
and Lemonade on non-Windows are all fully supported. Teaching the Windows
detached-spawn primitive to carry a value-typed override (so Windows Lemonade can
serve public with auth) is tracked as a separate follow-up.

Engine recipe adapter contract focused tests:

```bash
cargo test -p rocm-engine-protocol engine_recipe_hint_roundtrips_through_resolve_request
cargo test -p rocm --bin rocm engine_recipe
cargo test -p rocm-engine-lemonade engine_recipe
cargo test -p rocm-engine-vllm engine_recipe
```

These tests verify that signed-index engine-specific recipe metadata is mapped
into the versioned engine protocol hint, and that each first-party adapter
accepts matching hints, rejects mismatched engine ids or contract versions, and
echoes accepted hints from `resolve_model`. First-party launch commands also
apply accepted engine recipe `required_flags`; mismatched hints are rejected
instead of being silently ignored.

Contained update-check automation tests:

```bash
cargo test -p rocmd therock_update_contained_mode_runs_read_only_check_without_queueing
cargo test -p rocmd therock_update_contained_mode_records_update_available_without_applying
cargo test -p rocmd therock_update_contained_mode_uses_restricted_check_updates_tool
cargo test -p rocmd therock_update_notify_if_newer_uses_restricted_notification_contract
cargo test -p rocmd sandbox_check_updates_value_is_read_only_and_preserves_output
cargo test -p rocmd sandbox_check_updates_value_marks_runtime_update_available
cargo test -p rocmd sandbox_driver_plan_value_is_read_only_and_preserves_output
cargo test -p rocmd watcher_policy_maps_modes_to_decisions
```

Manual Windows smoke:

```powershell
rocmd sandbox-run check_updates --allow-native-fallback
rocmd sandbox-run driver_plan --allow-native-fallback
```

The JSON output should include `mutating: false` and the captured
`rocm update` stdout for `check_updates`. Its `status` is `checked` when no
runtime update is reported, `update_available` when the read-only report
contains `status=update_available`, or `error` when the check fails.
Contained `therock-update` consumes the restricted `check_updates` tool result
and records a `notify_if_newer` notification/audit event only for the
`update_available` case, while still leaving proposal history empty.
`driver_plan` should include `status: planned`, `mutating: false`, and captured
`rocm install driver --dkms --dry-run` stdout. Neither tool must apply updates
or execute driver commands.

Restricted sandbox tool API tests:

```bash
cargo test -p rocmd sandbox_tool_cli_values_cover_restricted_plan_api
cargo test -p rocmd sandbox_tool_examine_snapshot_is_read_only
cargo test -p rocmd sandbox_tool_list_servers_returns_records
cargo test -p rocmd sandbox_tool_list_servers_first_run_returns_empty_list
cargo test -p rocmd sandbox_tool_requires_service_id_for_restart
cargo test -p rocmd sandbox_tool_restart_server_reports_missing_service
cargo test -p rocmd sandbox_tool_requires_service_id_for_stop
cargo test -p rocmd sandbox_tool_stop_server_reports_missing_service
cargo test -p rocmd sandbox_tool_stop_server_updates_manifest_and_skips_current_pid
cargo test -p rocmd sandbox_tool_stop_server_reports_an_unconfirmed_stop
cargo test -p rocmd mcp_stop_server_reports_an_unconfirmed_stop_as_an_error
cargo test -p rocmd mcp_stop_server_reports_a_confirmed_stop_as_success
cargo test -p rocmd the_stop_request_is_recorded_before_any_process_is_signalled
cargo test -p rocmd a_stop_keeps_what_another_writer_persisted_while_it_ran
cargo test -p rocmd a_stop_does_not_clear_processes_recorded_while_it_ran
cargo test -p rocmd sandbox_tool_notify_user_is_read_only
cargo test -p rocmd sandbox_runner_native_fallback_records_audit
```

These cover the plan-listed restricted internal tool API: `check_updates`,
`examine_snapshot`, `list_servers`, `restart_server`, `stop_server`,
`prefetch_artifact`, and `notify_user`, plus the plan-derived `driver_plan`
read-only extension. Read-only tools must report `mutating: false`;
`notify_user` must record a local notification audit event; server restart/stop
must require an explicit service id; sandbox-run native fallback or Linux
bubblewrap isolation must still execute only this internal tool API and record
a sandbox audit event. `stop_server` — the sandbox tool and the MCP tool alike —
reports a stop only once every recorded process is confirmed gone: otherwise
the sandbox tool reports `status: stop_unconfirmed`, the MCP tool answers with
`isError: true`, and the manifest keeps its PIDs and endpoint key. The stop
records the request in the manifest before it signals anything, and applies its
result to the manifest as it is once the stop ends, so a write made meanwhile is
kept — and a restart that recorded new processes meanwhile makes the stop
unconfirmed rather than having its PIDs cleared.

`features/service_stop.feature` (`service-stop-02`, Linux, mock lane)
drives the real `rocmd sandbox-tool stop_server` against a live recorded
process and checks the confirmed report against the process, the record and
the endpoint key file. Run it with
`ROCM_CLI_BINARY=target/debug/rocm ROCM_CLI_ROCMD_BINARY=target/debug/rocmd
cargo test -p e2e-cucumber --test e2e -- -n "service-stop-02"`.

Manual restricted-tool smoke:

```bash
rocmd sandbox-run examine_snapshot --allow-native-fallback
rocmd sandbox-run list_servers --allow-native-fallback
rocmd sandbox-run notify_user --message "ROCm check complete" --allow-native-fallback
```

On Linux/WSL with `bubblewrap` installed, these commands should report
`isolation: bubblewrap`; on native Windows they should report
`isolation: native_restricted`.

Direct MCP helper safety smoke:

```bash
cargo test -p rocmd direct_mcp_call -- --nocapture
cargo test -p rocmd rocm_mcp_tools -- --nocapture
rocmd mcp-call examine --arguments-json '{}'
rocmd mcp-call install_sdk --arguments-json '{}'
```

The read-only `examine` helper call should run. The `install_sdk` helper call
must fail unless `--allow-mutation` is supplied after an explicit user
approval. User-facing TUI/chat flows should still route mutating tool calls
through their normal approval cards rather than relying on this hidden helper
flag.

Artifact cache and prefetch policy tests:

```bash
cargo test -p rocm-core model_artifact_cache
cargo test -p rocm --bin rocm model_recipe_artifact_lines_surface_signed_index_metadata
cargo test -p rocmd sandbox_prefetch_downloads_direct_artifact_when_approved
cargo test -p rocmd sandbox_prefetch_blocks_approved_artifact_without_sha256
cargo test -p rocmd sandbox_prefetch_blocks_artifact_larger_than_approved_limit
cargo test -p rocmd sandbox_prefetch_cached_marker_skips_network
cargo test -p rocmd sandbox_prefetch_blocks_non_direct_non_huggingface_source
cargo test -p rocmd sandbox_prefetch_unknown_artifact_ref_errors
cargo test -p rocmd sandbox_prefetch_reports_policy_required_without_network
cargo test -p rocmd sandbox_prefetch_blocks_gated_huggingface_without_source_policy
cargo test -p rocmd sandbox_prefetch_blocks_huggingface_source_policy_without_token
cargo test -p rocmd sandbox_prefetch_respects_manual_only_source_policy
cargo test -p rocmd sandbox_prefetch_respects_declared_huggingface_auth_policy
cargo test -p rocmd sandbox_prefetch_never_sends_huggingface_token_to_non_huggingface_url
cargo test -p rocmd sandbox_prefetch_never_sends_huggingface_token_over_plain_http
cargo test -p rocmd sandbox_prefetch_requires_artifact_ref
```

`prefetch_artifact` is no-network by default. Without approval it should report
`source_policy_required`, `mutating: false`, and `network_used: false`.
Approved live prefetch uses:

```bash
rocmd sandbox-run prefetch_artifact \
  --artifact-ref <model-ref>#<artifact-id> \
  --allow-artifact-download \
  --artifact-max-bytes <bytes> \
  --allow-native-fallback
```

The approved path supports direct HTTP(S) artifact URIs that include
`size_bytes` and `sha256` recipe metadata. It downloads the bytes, enforces the
byte limit, verifies sha256, writes an atomic cache marker, and reports
`status: prefetched`. Gated artifacts, missing hashes, and non-direct sources
remain blocked instead of falling back to an unverified download unless a
source-specific policy explicitly applies.

Signed recipe artifacts can also declare `source_policy` metadata. The
signed-index loader validates policy names, host constraints, HTTPS
requirements, and required integrity metadata. `/model` shows the policy, and
`prefetch_artifact` enforces `manual_only` and authenticated Hugging Face
requirements even after a generic artifact download approval.

For gated Hugging Face artifacts, the approved path also requires
`--allow-huggingface-download` plus `ROCM_CLI_HUGGINGFACE_TOKEN`, `HF_TOKEN`,
or `HUGGING_FACE_HUB_TOKEN`. rocm-cli should still refuse to send that token to
anything other than an HTTPS Hugging Face URL, and tokens must not appear in
JSON output. On Windows, pass `--allow-native-fallback` when manually running
`rocmd sandbox-run` because bubblewrap isolation is Linux-only.

Driver reconciliation focused tests:

```bash
cargo test -p rocm --bin rocm driver_plan_ -- --nocapture
cargo test -p rocm --bin rocm driver_reconcile_updates_state_after_reboot
cargo test -p rocm --bin rocm driver_passive_check_summary_counts_non_present_as_missing
```

The driver-plan tests are non-privileged. They verify AMD-documented
package-manager plans for Ubuntu, Debian, RHEL, Oracle Linux, Rocky, and SLES,
while unsupported Linux IDs remain non-mutating instead of guessed. The
reconcile path is also non-privileged. It refreshes passive checks and persists
the `total`/`present`/`missing` check summary without running DKMS commands.

Provider audit focused test:

```bash
cargo test -p rocm --bin rocm local_provider_stream_chat_posts_sse_request
cargo test -p rocm --bin rocm remote_openai_stream_chat_uses_live_sse
cargo test -p rocm --bin rocm remote_anthropic_stream_chat_uses_live_sse
```

This verifies the streamed local provider path writes a `provider` audit event
with action `stream_chat` after the SSE stream completes. The remote provider
tests use local HTTP fixtures to prove OpenAI and Anthropic send `stream: true`
requests and deliver the first SSE delta to the callback before the server
closes the connection.

Proposal history rendering:

```bash
cargo test -p rocm --bin rocm render_automations_text_uses_plain_proposal_history
```

The plain proposal history test verifies that `/automations` and
`rocm automations` show recent review requests with friendly action, reason,
server/artifact details, and approve/reject controls instead of raw backend
action or restricted-tool names.

Natural-language plan handoff tests:

```bash
cargo test -p rocm --bin rocm hybrid_planner_normalizes_model_alias_and_structured_serve_call
cargo test -p rocm --bin rocm freeform_plan_next_action_surfaces_approval_action
cargo test -p rocm --bin rocm freeform_invocation_supports_leading_yes_for_natural_language_only
cargo test -p rocm --bin rocm freeform_invocation_rejects_unquoted_structured_command_names_after_yes
cargo test -p rocm --bin rocm freeform_invocation_rejects_flag_shaped_yes_request
cargo test -p rocm --bin rocm freeform_execution_validation_rejects_placeholder_tool_calls
cargo test -p rocm --bin rocm freeform_execution_header_surfaces_explicit_approval_and_tool_call
cargo test -p rocm --bin rocm hybrid_planner_driver_action_includes_yes_for_approved_execution
cargo test -p rocm --bin rocm render_freeform_plan_exposes_structured_tool_calls
cargo test -p rocm --bin rocm natural_serve_with_missing_model_does_not_focus_approval
```

In the TUI, recognized mutating plain-language requests render a plan and then
focus the normal approval card. Missing placeholders remain plan-only.
Outside the TUI, `rocm --yes <natural language request>` renders the same plan
and executes only the final structured `rocm` tool call after placeholder
validation. Requests beginning with flags or unquoted structured command names
are rejected; quote the natural-language request or phrase it as natural
language, and run structured commands directly with their own approval flag
when they define one.

Cloud provider opt-in tests:

```bash
cargo test -p rocm-core provider_config_defaults_to_local_only
cargo test -p rocm --bin rocm provider
cargo test -p rocm --bin rocm render_config_text_includes_telemetry_policy
```

OpenAI and Anthropic prompt sending must fail before env-key lookup or network
access until the provider is explicitly enabled with `rocm config
enable-provider <provider>`.

Cloud API keys are stored outside `config.json`:

```bash
printf '%s' "$OPENAI_API_KEY" | rocm config set-provider-key openai
rocm config enable-provider openai
rocm config show
rocm config clear-provider-key openai
```

`rocm config show` should report whether the key is saved in the OS secure
store, missing, or coming from `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` for the
current session. It must never print the key. Remote provider HTTP requests are
sent in-process so API keys are not passed through `curl` command arguments.
In the TUI, clearing a saved API key defaults to `Cancel`; users must move the
highlight to `Clear API key` before Enter removes anything.

Update-surface report test:

```bash
cargo test -p rocm --bin rocm render_update_text_reports_all_update_surfaces
```

`rocm update` must report runtime checks plus CLI, engine, and model-recipe
surfaces honestly. `rocm update --apply` remains runtime-only until production
metadata feeds exist for the other surfaces.

Model recipe engine metadata checks:

```bash
cargo test -p rocm-core model_recipe_index
cargo test -p rocm-core model_recipe_index_signature_accepts_generated_key_and_rejects_tamper
cargo test -p rocm model_recipe_artifact_lines_surface_signed_index_metadata
cargo test -p rocm render_freeform_plan_exposes_structured_tool_calls
```

Engine-specific recipe metadata uses a versioned adapter/protocol hint for
`resolve_model` and launch. `/model` and `rocm serve` should show selected
engine recipe metadata clearly. First-party adapters must apply accepted
launch-time `required_flags`, reject mismatched engine ids or contract
versions, and fail loudly for unsupported flags instead of silently ignoring
them.

Release trust checks:

```bash
cargo test -p rocm --bin rocm metadata_signature_verification_accepts_generated_key_and_rejects_tamper
cargo test -p rocm-core model_recipe_index_signature_accepts_generated_key_and_rejects_tamper
cargo test -p rocm --bin rocm wsl_rocdxg_generated_digest_step_accepts_only_the_matching_file
python scripts/release_readiness.py --self-test
bash scripts/setup-wsl-portable-build-deps.sh --self-test
bash scripts/reclaim-gpu.sh --self-test
```

The ROCDXG digest check pins the shell fragment the WSL driver plan generates
as a whole string — so a change to its quoting, spacing or field order fails
the test rather than slipping past a substring assertion — and then runs that
same generated fragment against a real file, with only the digest and the path
redirected at a test payload: a matching digest, a mismatched one, a malformed
one, and an empty one. That step is the only thing authenticating a package
that is then installed as root, and this replaces the self-test that shipped
with the removed `scripts/wsl_setup_rocdxg.sh`. It needs a POSIX shell and
`sha256sum`, so it is Unix-only.

The eight native Linux GPU preflights are one script,
[`gpu-preflight.sh`](../.github/scripts/gpu-preflight.sh), which each lane runs
with its own `GPU_PREFLIGHT_MIN_FREE_GIB` (16 on the discrete lanes, 8 on
Strix). It applies two gates, because neither sees what the other does: a
bounded wait on free memory, and then `reclaim-gpu.sh --assert-clear`. The
memory floor is a fixed number of GiB against whatever pool `rocm-smi` reports,
so its sensitivity is set by the pool's size rather than by what a serve holds —
against a ~62 GiB Strix GTT aperture it trips only past ~54 GiB held, and
against a 192 GiB MI300X only past ~176 GiB, so a leftover `llama-server`
holding 10–20 GiB clears every lane. The process assertion catches that by
name. Every external call the script makes is bounded by `timeout 15`, the
`--assert-clear` scan included: that scan reads `/proc/<pid>/cmdline` for every
process, and a holder wedged in uninterruptible sleep — the state this gate
fires on — can block the read. A scan that does not finish **fails** the
preflight, with its own message, rather than being taken as a clear host; so
does a scan that could not be run at all — a missing script, a flag the two
scripts disagree about — which gets a third message rather than being reported
as a surviving process nobody can then find. The
floor is kept rather than replaced because the converse is just as
real and the process check cannot see it: a non-E2E tenant on a shared host can
exhaust the pool while running nothing the reclaim rules match. The two
Windows-native blocks remain PowerShell copies with the memory floor only; the
two WSL blocks remain advisory, warning where these fail.

`reclaim-gpu.sh` frees the GPU from engine processes a killed prior run leaked.
The self-hosted Linux GPU lanes run it before their GPU preflight, and again
with `--assert-clear` from inside the preflight once the memory floor is
satisfied — that mode kills nothing, exits non-zero naming any leaked process
still running, and reuses the same selection the reclaim itself acts on. The two
native Windows lanes have no bash and restate the rule in PowerShell instead,
with their root list pinned to the script's by an `xtask` contract test. The
WSL lanes skip the proactive reclaim before preflight — they provision a fresh
guest per job and unregister it afterwards, so no prior run's processes survive
into them — but they still call the script for `--report-holders` when their
preflight fails, as the other bash lanes do. The two native Windows lanes call
it at neither point: their preflight failure path has no PowerShell equivalent
of the diagnostic. Its `--self-test`
needs no GPU — it spawns decoy processes and asserts which ones the matching
rule selects — so it runs on the GitHub-hosted lane rather than only where GPU
hardware is held, under the same `heavy` path filter as the other checks there
(which any change under `scripts/` reaches).

The release-readiness self-test is cross-platform and uses only workspace-local
temporary files under `.rocm-work/tests/release-readiness`. It also checks exact
release asset sets, so stale archives and orphan checksum/signature sidecars in
`dist` fail before upload, and it validates both missing and explicitly
configured production trust inputs. Normal Linux and Windows CI run this
self-test before the install-lifecycle E2E scenarios.

### Packaging

`cargo xtask package <dist-name> [output-dir] [--target <triple>]` builds the
release distribution bundle in pure Rust — a `.tar.gz` on Unix and a `.zip` on
Windows — with a `sha256` sidecar and, when a signing key is configured, a
detached `.sig`. It replaces the former `scripts/package-{linux,windows}-release`
scripts and produces the same external artifact contract (bundle layout,
checksum syntax, signature presence, and installer compatibility). Signing
inputs come from the environment for parity with those scripts:
`ROCM_CLI_SIGNING_PRIVATE_KEY_PATH` (a PEM file) or
`ROCM_CLI_SIGNING_PRIVATE_KEY_PEM` (an inline PEM); set
`ROCM_CLI_REQUIRE_SIGNATURE=1` to fail unless a signature is produced.

### Install lifecycle (opt-in `@lifecycle` E2E)

The full install lifecycle — packaging, signature-verified install, tamper
rejection, reinstall, PATH handling, and uninstall — is expressed as
`@lifecycle` scenarios in the [E2E cucumber suite](../tests/e2e-cucumber/README.md)
(`features/install_lifecycle.feature`). They drive `cargo xtask package`, the
real root installer (`install.sh` / `install.ps1`), and the installed binaries
end to end, replacing the former
`scripts/acceptance-install-upgrade-tui-uninstall.{sh,ps1}` scripts and
preserving the union of their assertions.

These scenarios are expensive and OS-mutating, so they are **skipped by
default** to keep `cargo xtask e2e` fast. Run only the lifecycle scenarios for
the current host by opting in and selecting that set through the harness:

```bash
E2E_INCLUDE_LIFECYCLE=1 E2E_ONLY_LIFECYCLE=1 cargo xtask e2e
```

The harness applies the matching OS requirement itself; do not select these
with cucumber's `--tags`/`-n` flags, because those bypass the suite's custom
expectation and applicability resolver.

Heavy CI runs the matching set per platform. The scenarios reject untrusted
signers, bad checksums, bad detached signatures, and missing required signature
sidecars before activation; cover generated private-key PEM packaging and
generated public-key PEM installer verification; verify first-install PATH setup
(Windows persists the install folder to the user PATH and restores it afterward;
Linux writes the shell profile); reinstall stale-manifest purge and config
preservation; a Windows loopback-HTTP install; isolated installed-binary
directory smoke checks (`rocm examine` must read only the isolated
config/data/cache, never the real user `.rocm` state); and uninstall: a full
purge that also restores the shell profile, the Windows user `PATH` taken back
off with its other entries and its registry type kept, the `PATH` change kept while another program shares the install folder or
with `--keep-binaries`, a `PATH` change that cannot be undone removing nothing
so it can be retried, a profile and install-folder spelling found only through
the installer's record, and the running Windows `rocm.exe` removed. The two
Windows scenarios that change the user `PATH` are `@serial`, since it is one
registry value for the whole machine.
Each scenario generates its own local keys, package, and install root under a
per-scenario temp directory, so they are independent and use generated local
keys only — project-owned production signing keys remain an owner-controlled
release step.

If the host does not have `libcap-dev` or `libssl-dev`, run
`bash scripts/setup-wsl-portable-build-deps.sh` first: it downloads the Ubuntu
development packages into `.rocm-work/tools/wsl-build-deps`, extracts only the
headers, libraries, and pkg-config metadata there, and points
`PKG_CONFIG_PATH`/`PKG_CONFIG_SYSROOT_DIR` at that local copy — no sudo required.
Run `bash scripts/setup-wsl-portable-build-deps.sh --self-test` to verify the
portable sysroot normalization path without apt, network access, or a real WSL
package download.

Automation endpoint-health event tests:

```bash
cargo test -p rocmd event_collector
cargo test -p rocmd event_collector_emits_endpoint_recoverable_service_event
cargo test -p rocmd server_recover_local_webhook_does_not_restart_healthy_service
cargo test -p rocmd recovery_reason_display_avoids_raw_status_tokens
cargo test -p rocmd recovery_does_not_restart_a_service_whose_stop_is_unconfirmed
cargo test -p rocmd recovery_skips_a_failed_service_only_while_its_stop_request_stands
cargo test -p rocmd restart_managed_service_drops_the_replaced_runs_engine_pid
cargo test -p rocmd supervisor_exit_write_keeps_a_standing_stop_marker
cargo test -p rocmd supervisor_exit_write_leaves_a_record_it_no_longer_owns
cargo test -p rocmd supervise_service_first_write_keeps_a_standing_stop_marker
cargo test -p rocmd restart_failure_write_keeps_a_stop_marker_set_meanwhile
cargo test -p rocmd restart_records_its_supervisor_without_erasing_a_stop_marker
cargo test -p rocmd restart_leaves_a_record_naming_another_supervisor_or_none
cargo test -p rocm --bin rocm stop_records_its_marker_on_disk_before_any_process_is_signalled
cargo test -p rocm --bin rocm stop_does_not_confirm_when_a_restart_recorded_new_processes_meanwhile
cargo test -p rocm --bin rocm stop_final_write_keeps_what_others_wrote_while_it_ran
cargo test -p rocm --bin rocm restart_does_not_start_an_engine_over_an_unconfirmed_stop
cargo test -p rocm --bin rocm restart_readiness_write_keeps_a_stop_asked_for_while_it_waited
cargo test -p rocm --bin rocm restart_readiness_write_leaves_a_stop_that_finished_meanwhile
cargo test -p rocm --bin rocm restart_reports_a_child_it_could_not_record_and_writes_nothing
cargo test -p rocm --bin rocm restarted_child_record_names_the_runtime_the_restart_was_pinned_to
```

Server recovery never restarts a service with a standing stop request
(`stop_requested_unix_ms`): one whose stop is in flight, or could not be
confirmed. Without that, an unconfirmed stop — which leaves the status at
`ready` — would be undone by the `server-recover` watcher. So both stops
(`rocmd`'s and `rocm services stop`) persist the marker before signalling
anything, and the supervisor's and a restart's later writes re-read the
record and never clear it. Each stop also re-reads the record before its own
final write, and confirms the stop only if that record still names the
processes it handled. `rocm services restart` (and the `restart_server`
sandbox tool) starts a new engine only after its stop is confirmed; otherwise
it reports the unconfirmed stop and starts nothing. Its own later writes are
applied to the record as it is on disk, and only while it still reads as the
restart left it, so a stop asked for during the restart keeps its marker.

Automation GPU-metrics event tests:

```bash
cargo test -p rocmd gpu_metrics
cargo test -p rocmd event_collector_emits_gpu_metrics_event_when_enabled
```

The `gpu-metrics` watcher is read-only in all modes. It records bounded local
`amd-smi` telemetry availability or unavailability and must not queue proposals
or take thermal/serving actions until a policy is explicitly defined.

Automation cache-warm event tests:

```bash
cargo test -p rocmd local_webhook_cache_warm_builds_prefetch_event
cargo test -p rocmd local_webhook_cache_warm_rejects_other_cache_events
cargo test -p rocmd cache_warm_propose_mode_queues_prefetch_proposal
cargo test -p rocmd cache_warm_unknown_artifact_does_not_queue_proposal
cargo test -p rocmd cache_warm_contained_mode_still_requires_reviewed_source_policy
```

The `cache-warm` watcher accepts exactly `cache.warm`, requires
`payload.artifact_ref`, verifies that the ref exists in the model recipe
registry before queueing, queues a reviewed `prefetch_artifact` proposal, and
does not grant webhook payloads arbitrary tool choice or automatic download
permission. The TUI can add reviewed source-policy fields with
`/automations edit <id> allow-download yes` and
`/automations edit <id> artifact-max-bytes <bytes>`; approved downloads require
that byte limit.

Automation driver-upgrade event tests:

```bash
cargo test -p rocmd local_webhook_driver_upgrade
cargo test -p rocmd driver_upgrade
cargo test -p rocmd driver_upgrade_contained_mode_runs_restricted_driver_plan
cargo test -p rocmd driver_upgrade_contained_mode_requires_restricted_driver_plan_tool
cargo test -p rocmd sandbox_driver_plan_value_is_read_only_and_preserves_output
cargo test -p rocm-core builtin_watchers_include_reviewed_driver_upgrade
```

The `driver-upgrade` watcher accepts exactly `update.available` with
`payload.component=driver`. In `propose` mode it queues a reviewed
`driver_plan` proposal. In `contained` mode it consumes the restricted
`driver_plan` tool result directly. Both paths stay read-only; the user-facing
approval should say it shows the driver plan only and that no driver will be
installed.

Automation local-webhook source tests:

```bash
cargo test -p rocmd local_webhook
cargo test -p rocmd local_webhook_therock_update_accepts_exact_schedule_tick_only
cargo test -p rocmd local_webhook_server_recover_rejects_nonrecoverable_service_kind
cargo test -p rocmd local_webhook_gpu_metrics_rejects_thermal_action_kinds
cargo test -p rocm render_automations_text_surfaces_local_webhook_endpoint
```

The local webhook source is loopback-only. It is enabled only with
`rocmd run --automations-enabled --local-webhook-port <port>`, accepts JSON
`POST /automation-events`, validates `watcher_hint` and `kind` against the
exact event kinds accepted by that watcher, and then dispatches through the
same watcher policy paths as scheduler, managed-service recovery, GPU
telemetry, and cache-warm proposal events. See `docs/automations.md` for the
local curl smoke and accepted fields.

vLLM TheRock GPU acceptance:

```bash
cargo test -p rocm-engine-vllm write_running_state_records_managed_therock_env_for_gpu_verification
cargo test -p rocm-engine-vllm managed_env_reflects_managed_runtime_manifest_source
cargo test -p rocm-engine-vllm resolve_model_omits_gpu_memory_utilization_default
python -m py_compile scripts/vllm_therock_gpu_test.py
python scripts/vllm_therock_gpu_test.py --self-test

# On Linux/WSL with vLLM installed in the active rocm-cli managed TheRock venv:
python3 scripts/vllm_therock_gpu_test.py \
  --engine /home/user/.cache/rocm-cli-target/debug/rocm-engine-vllm \
  --model facebook/opt-125m
```

The script defaults to the active exact runtime key. If `--runtime-id` is used,
it must be an exact runtime key or an unambiguous runtime id. It requires
`gpu_required`, rejects external vLLM command overrides, checks `/health` and
`/v1/completions`, and verifies loaded ROCm libraries came from the managed
TheRock SDK wheel directories.
For TheRock 7.13, patch vLLM's GPTQ ROCm compatibility guard to include HIP
7.13 before building from source; otherwise `q_gemm.hip` can fail on missing
`half`/`half2` `atomicAdd` overloads. Building from source needs the compiler
toolchain, so install the runtime with `rocm install sdk --devel` (see
[vllm.md](vllm.md)).
On native Windows this script prints a JSON skip result; run it from WSL/Linux
for live ROCm GPU acceptance.

## Serve OOM Guidance

The `rocm serve` interactive deployment summary's out-of-memory note is checked
here. See [the README](../README.md#when-a-serve-runs-out-of-gpu-memory)
for the user-facing contract and [vllm.md](vllm.md#shared-or-busy-gpus) for the
flag and the quoting rule.

Focused unit coverage — what counts as a vLLM OOM, and which log line is routed
into the printed `rocm diagnose --symptom` command:

```bash
cargo test -p rocm-core vllm_oom_gate_matches_allocator_signatures_but_not_generic_failures
cargo test -p rocm-core vllm_oom_diagnose_symptom_selects_the_failing_line_or_reports_no_oom
cargo test -p rocm-core every_rendered_line_break_keeps_the_gate_in_step_with_the_scorer
cargo test -p rocm-core the_engine_hint_is_never_quoted_back_as_the_users_own_line
cargo test -p rocm-engine-vllm the_hint_never_quotes_itself_back_on_a_second_read_of_the_shared_log
cargo test -p rocm-core quotable_in_single_quotes_rejects_quote_and_control_bearing_symptoms
cargo test -p rocm-core the_stripper_follows_the_escape_grammar_not_just_the_colour_case
```

The note itself — its conditions, the de-duplication of the hint the pre-launch
low-VRAM warning already printed, and the quoting guard on the command it hands
the user to paste:

```bash
cargo test -p rocm --bin rocm oom_note_
cargo test -p rocm --bin rocm oom_signatures_are_detected_case_insensitively
cargo test -p rocm --bin rocm unrelated_failures_are_not_flagged_as_oom
cargo test -p rocm --bin rocm the_shared_hint_is_de_duplicated_without_losing_the_rest_of_the_oom_note
cargo test -p rocm --bin rocm an_apostrophe_in_the_failing_line_cannot_break_out_of_the_printed_command
cargo test -p rocm --bin rocm control_bytes_from_the_log_never_reach_the_printed_command
```

The gate that decides whether the note is appended at all:

```bash
cargo test -p rocm --bin rocm append_oom_serve_note_
```

`append_oom_serve_note_ignores_an_already_running_services_log` is the only
check that discriminates the `already_running` clause of that gate: it passes a
log path that *does* contain an OOM signature together with
`already_running: true`. No E2E scenario can, because the real reuse path
reports no log path at all and the note is withheld on that ground alone.

The fault-injection feature must never reach a shipped binary:

```bash
cargo test -p xtask the_release_workflow_builds_no_feature_gated_binaries
cargo test -p xtask prebuilt_e2e_lanes_enable_test_hooks
```

The E2E scenario lives in
[`tests/e2e-cucumber/features/model_serving.feature`](../tests/e2e-cucumber/features/model_serving.feature).
Run the suite unfiltered and let the harness select:

```bash
cargo xtask e2e
```

There is deliberately no focused command for it. Cucumber 0.23 honours
*either* its own CLI filter *or* the harness closure, never both, so `-n` or
`--tags` bypasses the OS, capability and expectation resolution that decides
whether a scenario may run at all (see the comment at
[`tests/e2e-cucumber/tests/e2e.rs`](../tests/e2e-cucumber/tests/e2e.rs), above
`only_lifecycle`). That matters most
for serve-26, which is gated on the `@requires-oom-fault-injection` capability:
selected from the CLI it would run on a binary built without the hook and fail
for the wrong reason. Note also that `-n` matches the scenario *name*
(`serve-26 - A launch that runs out of GPU memory ...`), not the `@id:` tag, so
passing an id to it silently selects nothing.

- serve-26, `@id:serve-oom-launch-memory-guidance` — a launch that runs out of
  GPU memory blames this launch, names the knobs, and quotes the engine's own
  failing line. It carries `@requires-no-gpu` (the fabricated launch stands in
  for the GPU the mock lane does not have) and `@requires-os:linux`.

### The OOM fault-injection capability

serve-26 needs a managed launch that really failed on VRAM, which a GPU-less CI
host cannot produce. It is therefore driven by a test-only fault-injection hook:
the `rocm` crate feature `e2e-oom-fault-injection`, armed at run time by
`ROCM_E2E_SIMULATE_OOM_LAUNCH=1`, makes `rocm serve` fabricate the on-disk state
of a managed vLLM launch that spawned, wrote an allocator OOM traceback to the
log it owns, and never became ready. No process is spawned, and the seam
compiles out entirely without the feature, so a build that does not enable it
behaves exactly as the shipped binary does.

Only `cargo xtask e2e` enables it, by building the binary under test with
`--features "rocm/e2e-test-hooks rocm/e2e-oom-fault-injection"`, and it is the
only producer of the `ROCM_E2E_OOM_FAULT_INJECTION=1` signal the harness reads.
This is the one capability the harness **cannot probe for** — no product command
exposes a compile-time feature — so `@requires-oom-fault-injection` resolves
from that variable alone, and xtask clears it rather than inheriting it when it
was handed a prebuilt binary.

Consequently, every lane that runs a **prebuilt** `ROCM_CLI_BINARY` — the
self-hosted lanes in [`e2e-selfhosted.yml`](../.github/workflows/e2e-selfhosted.yml)
and the [`nightly.yml`](../.github/workflows/nightly.yml) lanes — has the hook
compiled out and resolves serve-26 to a reported **skip**, not a silent pass. It
gates PRs on the GitHub-hosted mock lane, which lets `cargo xtask e2e` build the
binary itself. The prebuilt lanes' `--features` floor is asserted as a superset
check rather than an exact match for this reason; see the test notes on
`assert_prebuilt_e2e_lanes_enable_test_hooks` in
[`xtask/src/workflow_contract.rs`](../xtask/src/workflow_contract.rs).

Be precise about *which* gate produces the skip, because the hook is not always
the one that decides. On the Windows lifecycle lane in
[`ci.yml`](../.github/workflows/ci.yml) the hook is compiled out too, but
serve-26 is never selected there at all: that lane runs `E2E_ONLY_LIFECYCLE` and
the scenario carries `@requires-os:linux`. On the GPU lanes it is
`@requires-no-gpu` that excludes it. The missing hook is the deciding gate only
where a Linux, GPU-less lane runs a prebuilt binary.

## Windows Tool Notes

The TheRock SDK wheel install path should not require users to install global
Python or curl. rocm-cli uses Rust-native HTTP downloads and can bootstrap a
managed Python under its data directory when no usable Python is available.

TheRock's Windows source-build guide documents `winget` and manual installs for
MSVC, Git, CMake, Ninja, Python, Strawberry Perl, DVC, and related tools. The
SDK install test reports these when `--check-windows-tools` is set, but normal
SDK wheel setup should avoid global source-build tools.

Reference: [TheRock Windows install tools](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md#install-tools)

## WSL Preflight

Read-only WSL/ROCDXG preflight:

```bash
rocm diagnose --json
```

The WSL catalog covers `/dev/dxg`, the DXCore handoff, ROCDXG and its linker
entry, the distro release floor, the Windows host driver, and WSL 1. A clean run
reports no findings; anything it does report carries a `fix-wsl-*` id and a plan.

From the Windows host, to inspect a distro without installing anything in it:

```powershell
rocm diagnose --distro          # the only distro installed
rocm diagnose --distro Ubuntu   # a named one
```

Both forms run the same catalog. The host-side one collects its facts over
`wsl.exe` with a POSIX shell, so the target distro needs neither `rocm-cli` nor
Python.

ROCDXG install inside WSL:

```bash
rocm install driver            # review the plan
rocm install driver --yes      # run it
rocm examine                   # expect driver_status: wsl_rocdxg_ready
rocm diagnose
```

The last two are the plan's own `post_install_checks`, so running them is what
confirms the install took rather than merely that the commands exited zero.

The `.deb` is verified against a digest pinned per ROCDXG release, so the
default path needs nothing set. To exercise a release rocm-cli has no digest
for, supply one — or opt out explicitly, which is the only way to reach an
unverified install:

```bash
ROCM_CLI_ROCDXG_VERSION=<version> \
ROCM_CLI_ROCDXG_SHA256=<64-hex-sha256> rocm install driver --yes

ROCM_CLI_ROCDXG_VERSION=<version> \
ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED=1 rocm install driver --yes
```

The install is covered by unit tests over the generated plan (`cargo test -p
rocm --bin rocm wsl_rocdxg`). Running it end to end needs a WSL2 host with
`/dev/dxg` and dxcore present, since the plan refuses before installing
otherwise.

## /etc/os-release

Every Rust reader of `/etc/os-release` parses it with
[`os_release.rs`](../crates/rocm-core/src/os_release.rs), which returns the
value `sh` would assign or nothing: the driver plan
([`driver_install.rs`](../apps/rocm/src/driver_install.rs)), the OpenMPI,
libatomic and libnuma install plans `rocm` runs before serving and the hints
naming them ([`openmpi.rs`](../crates/rocm-core/src/openmpi.rs)), `rocm
examine`'s distro fields ([`examine.rs`](../crates/rocm-core/src/examine.rs)),
and the distro name it prints
([`host_gpu.rs`](../crates/rocm-core/src/host_gpu.rs)). The one exception is
`rocm diagnose --distro` run from Windows, which has the WSL distribution's own
`sh` source the file and reports what it assigned.

Every one of those readers except the WSL remote probe above goes through
[`hardware_root::read_os_release`](../crates/rocm-core/src/hardware_root.rs),
which reads `/etc/os-release` and, if that file does not exist, falls back to
`/usr/lib/os-release` per `os-release(5)` — a file present but unreadable (the
line-reporting cases below, and the file-level case further down) does not
fall back, since a stale fallback could describe the wrong host entirely.
The two reads it covers: a probe routed through a simulated test root
(`rocm examine`'s distro fields and distro name) resolves both candidates
through `host_path`; a package-manager plan (the driver install, and the
OpenMPI/libatomic/libnuma installs and hints) always resolves both against the
real machine, for the same reason it never follows a simulated root at all.
Each production call site is a one-line delegation to this shared reader, so
it names the candidate it actually read — `/etc/os-release` or
`/usr/lib/os-release` — rather than always the former, in every message below.

A line that is not a plain assignment makes the whole file unreadable, and
each reader names that file and line instead of reporting an unknown distro.
Where the message asks for a fix, it also says what to change on the line: LF
line endings, single quotes around the value, or a `NAME=value` line in place
of anything else.

- `rocm install driver`: policy `unreadable_os_release`, no commands, and a
  reason naming the file and line.
- The OpenMPI, libatomic and libnuma installs: no commands, and the file and
  line ahead of the manual-install advice in the plan's reason.
- The hints naming those installs in vLLM's warnings and errors: the file and
  line after the manual-install advice, and no offer of `rocm engines install
  vllm --yes`, which would plan nothing. That command is offered only when its
  plan has commands, so it is not offered for an unknown distribution either.
- `rocm examine`: `distro: Linux (/etc/os-release line N is not a plain
  assignment: "…")`; with `--json`, also an empty `distro_id` and
  `distro_version`, the same file and line in `probe_failures`, and `status:
  degraded` on an otherwise healthy GPU host.

A CRLF file is unreadable too: `sh` would assign `ubuntu\r`.

A candidate that exists but cannot be read at all (permission denied, a
directory in its place) is reported distinctly from both of the above: not as
an unknown distribution (which would send the user looking for a support
matrix for what is a filesystem problem) and not folded into "absent, so fall
back to the other candidate" (which could silently plan from a stale file that
may not describe this host). This is `hardware_root::OsRelease`, the fallback
reader's return type: `Found { path, text }` for a successful read (naming
whichever candidate it was), `Absent` when neither candidate exists (not a
failure; a missing file declares nothing), or `Unreadable { path, error }`
when a candidate exists but could not be read. Every production call site
above matches on all three: `rocm install driver` via
`build_driver_install_plan_read`, the OpenMPI/libatomic/libnuma installs and
hints via `plan_for_os_release_read`/`install_hint_for_read`, and `rocm
examine`'s distro fields and distro name via `record_distro` and
`distro_name_from_os_release`, which both take the whole `OsRelease` directly.
The three `_read`-suffixed functions are the one seam the real call site
hands its result to, and each delegates to the matching `_at` function with
the path it actually read. `install_hint_for`/`build_driver_install_plan`
are bare-text siblings of `_at` — not something `_read` delegates through —
kept only for the unit tests below, which fixture text directly and have no
`OsRelease` of their own to construct.

```bash
cargo test -p rocm-core --lib os_release
cargo test -p rocm-core --lib -- pretty_name_reads_as_sh_reads_it an_unreadable_os_release
cargo test -p rocm-core --lib openmpi::tests
cargo test -p rocm --bin rocm os_release
cargo test -p rocm --bin rocm -- a_fallback_read_plans an_unreadable_file_plans_nothing_and_names_itself
cargo test -p rocm-core --lib hardware_root::tests::read_os_release
cargo test -p rocm-core --lib -- a_fallback_read_plans an_unreadable_file_is_reported_as_itself distro_name_names_an_unreadable_file an_unreadable_file_names_itself
cargo test -p rocm-core --lib --features e2e-test-hooks -- os_release_reads_follow_the_simulated_host_root os_release_reads_fall_back_to_usr_lib_when_etc_is_absent try_os_release_does_not_source_the_fallback_when_the_primary_is_not_sourceable
```

The parser tests that compare against `/bin/sh` itself are Linux-only. The
`hardware_root::tests::read_os_release*` tests check the fallback itself
(primary found, primary absent, neither found, and each candidate existing but
erroring for a reason other than absence) against an injected temp directory
rather than the real host. The `a_fallback_read_plans*`/
`an_unreadable_file*`/`distro_name_names_an_unreadable_file` tests in
`rocm-core` and `rocm` construct an `OsRelease` directly — bypassing the real
read entirely, since the production functions that do it
(`openmpi::real_os_release`, `runtime_install_finalize::real_os_release`)
read the real machine
unconditionally and cannot honour a simulated root — to pin that a
fallback-sourced read plans and hints identically to a primary-sourced one
with the same text, and that a file which exists but cannot be read is named
as itself rather than silently treated as an unknown distribution. The two
`--features e2e-test-hooks` os-release tests plant a file under the simulated
host root and check that both `rocm examine` reads answer from it through the
shared parser: the first at `/etc/os-release`, the second with only
`/usr/lib/os-release` present, so the fallback is exercised through the real
`host_path`-routed call sites, not just the helper in isolation; the feature
they need is enabled only by its own CI step. The
`try_os_release_does_not_source_the_fallback_when_the_primary_is_not_sourceable`
test runs the WSL remote probe's own shell fallback for real under `/bin/sh`
against a primary that exists but cannot be sourced (a directory in its
place) — proving existence is not readability, the same claim the Rust-side
`hardware_root` tests make for the two in-process readers, but **not** the
same failure mode: a genuinely permission-denied primary aborts the whole
shell probe instead of merely leaving `$ID` unset, which this test does not
exercise — see its own doc comment for why, and for the known limitation
this leaves.

`rocm examine` reads the file through the simulated host root (see Simulated
Host Root in [release-trust.md](release-trust.md)), so scenarios
`examine-20` to `examine-23` and `examine-27` to `examine-28` in
[examine.feature](../tests/e2e-cucumber/features/examine.feature) plant one
(`examine-24` to `examine-26` are unrelated amd-smi telemetry scenarios that
share this file's numbering). `examine-23` also plants a GPU in the kernel
topology and puts an `lspci` listing an empty bus on `PATH`, so `status:
degraded` is checked on every Linux lane, with or without a GPU. `examine-27`
plants only `/usr/lib/os-release`, with no `/etc/os-release` present, to check
the `os-release(5)` fallback. `examine-28` plants a directory in place of
`/etc/os-release` (reusing `examine-23`'s GPU fixtures) to check the degraded
verdict and failure message a present-but-unreadable release file now gets,
distinctly from both a malformed line and an unknown distribution:

```bash
RUSTC_WRAPPER= cargo xtask e2e -- -n 'examine-2[0-3] |examine-2[7-8] '
```

The install plans deliberately read the real file, so no scenario plants one
for them, including for the fallback; their parsing is covered by the unit
tests above, and the fallback itself by `hardware_root`'s own unit tests,
since both now read through the same helper with an identity resolver.

## Windows Runtime Preflight

On Windows, `rocm diagnose` compares the Visual C++ runtime in `System32`
with the floor programs built with a recent toolset need, 14.40. The version
comes from the file version of `msvcp140.dll` and `vcruntime140.dll`, and
shows in `rocm examine --json` as `msvc_redist_version` (the older of the two,
empty when neither could be read).

```powershell
rocm examine --json | ConvertFrom-Json | Select-Object msvc_redist_version
rocm diagnose
rocm fix fix-21-msvc-redist-outdated
```

On a machine whose runtime is older than 14.40, `diagnose` reports
`fix-21-msvc-redist-outdated` with no symptom text. `rocm fix` prints the plan
and runs nothing: the update needs the Windows administrator prompt. After the
update, the plan's verify command prints the new version.

The version comparison and the symptom keywords are covered by unit tests in
`diagnose.rs`. No end-to-end scenario drives this check, because the e2e suite
cannot give a Linux host a Windows runtime: the check is gated to Windows. The
catalog scenarios (`diagnose-12`, `diagnose-34`) and the reference-doc
scenarios (`skill-catalog-ids-match-cli` and its neighbours) do list and
compare the new entry on every lane. A
scenario that runs the check itself would need the Windows lane and a host
with an old runtime installed.

### A GPU with no ROCm build

`info-20-gpu-no-rocm-support` is the catalog's one `DIAGNOSE-ONLY` entry: every
AMD GPU on the host identifies as a target ROCm has never shipped a runtime
build for (today only `gfx90c`, the Vega iGPU in Renoir / Lucienne / Cezanne /
Barcelo). `rocm fix` still answers for the id — by naming the problem, changing
nothing and exiting 0 — because `diagnose` prints `apply with: rocm fix <id>`
for every cause it reports.

`diagnose-40` covers it end to end. The host is simulated rather than required:
a KFD node planted under the scenario's own root reports `gfx_target_version
90012`, so it runs on **every** Linux lane, WSL2 included — `@requires-os:linux`
matches a host whose `os_family` is `linux`, and WSL2's is. Every `rocm` call in
that scenario therefore has to carry `ROCM_CLI_TEST_HOST_ROOT`: `rocm fix` gates
on the OS, and `fix::current_os` resolves WSL *through* the host root, so an
unrooted call on a WSL2 runner is read as `wsl` — an OS this entry deliberately
does not apply on — and exits 3.

```bash
cargo xtask e2e -- -n '(diagnose-40)'
```

The other half of the same change — `fix-13-hip-sdk-missing` no longer firing
just because no HIP SDK is installed — has **no** end-to-end coverage and cannot
have any. `check_13_hip_sdk_missing` returns early on every non-Windows host, so
on a Linux lane that early return alone satisfies "no cause tells this user to
install the HIP SDK"; the assertion would still hold with `check_20` deleted.
The gate itself is held by unit tests in
[diagnose.rs](../crates/rocm-core/src/diagnose.rs)
(`fix_13_does_not_fire_on_a_lemonade_vulkan_host_with_no_hip_evidence` and its
neighbours) and by the Windows manual check in
[manual-testing.md](manual-testing.md), in the Ryzen 4000/5000 section.

## Model Fit Preflight

`rocm diagnose --model <ref>` answers whether a curated model will run on this
host before anything downloads. It reports one of four verdicts: `ready`,
`degraded` (runs, but under the recipe's recommended system RAM),
`blocked` (will not run here, with alternatives that would), or
`undetermined` (the CLI could not judge it — an unreachable catalog, an
unmeasured GPU, or a ref outside the curated set).

```bash
rocm diagnose --model qwen-smoke --json   # smallest curated recipe
rocm diagnose --model glm5 --json         # largest curated recipe
```

On a host with a measured GPU, the smallest recipe should report `ready` (or
`degraded`, if this host's system RAM is below its recommendation) and name
the engine `rocm serve` would pick; the largest should report `blocked` with
at least one alternative that fits. On a host with no GPU visible to ROCm,
both report `blocked` with no fitting alternative. Either way, the command
must exit 0 — it is a query, not a check that only passes on a compatible
host — and must not populate the model-weight cache.

The e2e suite (`cargo xtask e2e -- -n diagnose-2`) exercises all four verdicts,
including the two that need a synthetic signed catalog to trigger
deterministically (`ModelNotCurated`, `Degraded`) since no built-in recipe can
produce them on an arbitrary real host.

## Doctor Report Preflight

Preview the content a problem report would carry, without sending anything:

```bash
rocm diagnose --report
rocm diagnose --report --json
```

The command refuses rather than prepares a report on three hosts, and the
three reasons are not interchangeable. One holds an architecture the ROCm
compatibility matrix does not list as supported (`unreleased-hardware`). One
has an AMD GPU architecture that could not be read (`architecture-unreadable`).
The third is any WSL host (`platform-not-probed`): `examine` returns before
any GPU probe runs there, so nothing has looked, and saying the architecture
could not be read would state a finding about hardware nothing inspected. All
three exit 0 and are told apart by `--json`'s `refused` field, and the refusal
envelope also carries `architecture_matrix`, the same compatibility-matrix
snapshot stamp a genuine report carries, so a refusal is just as traceable to
a matrix revision as a report is.

Offer a prefilled mail carrying that report, which still sends nothing:

```bash
rocm diagnose --report --send
```

Two argument rules are worth checking by hand, because both are the kind that
only break when somebody reorders a declaration. `--send` without `--report`
must be refused, since showing the content first is the guarantee `--send`
makes. `--send` with `--json` must also be refused: that combination is for
scripts, and starting a browser from a scripted invocation is not wanted.

Whether `--send` opens a mail client or prints the address and link depends on
the machine, and the printed line says which happened. It prints rather than
opens over SSH, with no `DISPLAY` or `WAYLAND_DISPLAY` on Linux, or with
`ROCM_NO_BROWSER` set to a non-empty value. A machine with no mail client
configured reaches the same printed form, which is why the address appears on
its own and not only inside the `mailto:` link. That is the common case on
servers and in containers. The opt-out is the easiest to check on a desktop:

```bash
ROCM_NO_BROWSER=1 rocm diagnose --report --send
```

The destination is `ROCmCLI@amd.com`, fixed in code. A report also carries its
classification in the mail subject, because a mailbox has no labels: the
subject names the matched catalog entry, or `unrecognised`, then the
architecture and the distribution. Check that the subject carries no field the
report body does not.

On a host with an approved architecture (see `APPROVED_ARCHITECTURES` in
`crates/rocm-core/src/report.rs`), the command prints the full `Report`:
`schema`, `architecture`, `architecture_matrix`, `entry`, `os_family`,
`os_major`, `distro`, `rocm`, `engine`, `engine_version`, `cli_version`, and
`fix_offered`. This path has not been exercised against real hardware in CI;
verifying it needs a lane whose GPU architecture is on the allowlist.

Four of those fields — `distro`, `rocm`, `engine`, and `engine_version` — can
answer with a word rather than a value, and the words are not interchangeable.
`none` means the thing is absent, `unknown` means this build looked and could
not tell, and `other` means a distribution was named but is not one this build
recognises. A host with no ROCm installed reports `"rocm": "none"`, while a
host whose install exists but whose version could not be read reports `"rocm":
"unknown"` — worth checking by hand on a machine with a partial install, since
the two are easy to merge by accident and a counter cannot tell them apart
afterwards. `engine`/`engine_version` carry the same distinction: a host a
probe found no engine on reports `"engine": "none"`, while a host whose engine
probe never ran (skipped rather than completed) reports `"engine": "unknown"`,
since a probe that never ran cannot say an engine is absent.
