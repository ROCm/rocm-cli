// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `vllm_tensorizer.feature`: `rocm engines install vllm` against a
//! managed runtime whose vLLM pins a tensorizer that cannot be imported.
//!
//! The runtime's Python and `uv` are shell stand-ins that share state files in
//! the runtime folder, so the scenario can read what the "installed" tensorizer is
//! rather than trusting the CLI's account of it:
//!
//! - `tensorizer-version`: written by the stand-in `uv` when it installs
//!   tensorizer (and reset to vLLM's pin when it installs vLLM), and seeded by
//!   the scenario itself, as the already upgraded release, for the two scenarios
//!   that plant a violation beside the upgraded tensorizer;
//! - `uv.log`: every `uv` invocation;
//! - `probe-ran`: written by the stand-in Python whenever the tensorizer import
//!   probe runs;
//! - `repaired`: written by the stand-in `uv` when a vLLM install clears the
//!   planted violation;
//! - marker files written when the scenario plants the runtime, which pick how
//!   the stand-ins behave: `library-failure`, `library-failure-in-tensorizer`,
//!   `key-error-in-torch`, `needs-pythonpath`, `needs-repair` and
//!   `repair-needs-reinstall-flag`;
//! - `bin/vllm`, which the stand-in `uv` writes on every vLLM install (creating
//!   it in a runtime that did not have it).
//!
//! The stand-ins use only shell builtins, because the scenario runs the CLI with
//! a PATH that holds nothing but its own shims.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

const RUNTIME_KEY: &str = "release-wheel-gfx1151-e2e-tensorizer";

/// The fixed release. Deliberately a literal rather than
/// `TENSORIZER_PYTHON314_FIX`: a bump of the constant must fail these scenarios
/// loudly so their stand-ins, which model that one release, are revisited.
const TENSORIZER_FIXED: &str = "2.12.1";

/// The line the CLI prints once it has replaced the unimportable pin.
const UPGRADE_WARNING: &str = "upgraded it to 2.12.1";

fn runtime_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("tensorizer-runtime")
}

fn write_shim(path: &Path, body: &str) {
    rocm_core::fake_executable::write_fake_executable(path, body)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

/// What installing tensorizer 2.12.1 does in the planted runtime.
#[derive(Clone, Copy)]
enum Upgrade {
    /// The stand-in Python can import tensorizer afterwards.
    Works,
    /// `uv` succeeds but the stand-in Python still cannot import it.
    NoEffect,
    /// `uv` fails, as on an offline host.
    Fails,
}

/// What else is different about the planted runtime.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Extra {
    /// Nothing: a runtime on a ROCm line without a discover build.
    None,
    /// A ROCm 10.1.0 discover runtime whose stand-in Python needs the `PYTHONPATH`
    /// the server launches with, and, like a real one, does not see it under `-I`.
    NeedsPythonPath,
    /// The import fails inside torch, below a tensorizer frame, with a missing
    /// library: not tensorizer's fault.
    LibraryFailure,
    /// The same missing library, raised by tensorizer's own code (the last frame
    /// is in the tensorizer package): still not a Python-version failure.
    LibraryFailureInTensorizer,
    /// A `KeyError` raised in torch, below tensorizer frames: not a library-load
    /// error, so only the last-frame rule can tell it is not tensorizer's.
    KeyErrorInTorch,
    /// A genuine unmet requirement beside the already upgraded tensorizer. A vLLM
    /// install clears it and, like the real one, puts back vLLM's tensorizer pin.
    /// Any `vllm==` install clears it here: the automatic repair does not pass
    /// `--reinstall` (only the user's `--reinstall` does), and
    /// [`Extra::ViolationClearedOnlyByReinstall`] models the violation a plain
    /// install leaves.
    ViolationBesideUpgrade,
    /// As [`Extra::ViolationBesideUpgrade`], but the stand-in `uv` only clears the
    /// violation when vLLM is installed with `--reinstall`, so the plain repair
    /// leaves it and the CLI advises the reinstall.
    ViolationClearedOnlyByReinstall,
    /// vLLM is not installed in the runtime yet (no `vllm` command beside the
    /// interpreter), so the install pass is a first install; the stand-in `uv`
    /// creates the command when it is asked to install vLLM.
    VllmNotInstalled,
}

/// Plants the registry entry, the stand-in Python and `vllm` command, and the
/// stand-in `uv` and OpenMPI shims. `coloured` makes the stand-in Python write
/// its traceback with ANSI escapes around the quoted path, as Python 3.13+ does
/// under `FORCE_COLOR`.
fn plant(world: &E2eWorld, upgrade: Upgrade, coloured: bool, extra: Extra) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let rt = runtime_dir(world);
    let bin = rt.join("bin");
    let shim = rt.join("shim");
    std::fs::create_dir_all(&bin).expect("create runtime bin");
    std::fs::create_dir_all(&shim).expect("create shim dir");
    let state = rt.display().to_string();

    // vLLM's own probe answers `present`. The tensorizer import fails unless the
    // version file says 2.12.1: with the real traceback's frame and exception when
    // the server's launch environment is present, and with a torch-style library
    // error when it is not (so a probe run without that environment would see a
    // failure that is not tensorizer's and leave the runtime alone).
    let traceback = if coloured {
        r#"esc=$(printf '\033')
      echo "${esc}[1;35mTraceback (most recent call last):${esc}[0m" >&2
      echo "  File ${esc}[35m\"/venv/lib/python3.14/site-packages/tensorizer/_crypt/_encryption.py\"${esc}[0m, line 367, in __new__" >&2
      echo "${esc}[1;35mKeyError${esc}[0m: ${esc}[35m'crypto_stream_salsa20_NONCEBYTES'${esc}[0m" >&2"#
    } else {
        r#"echo 'File "/venv/lib/python3.14/site-packages/tensorizer/_crypt/_encryption.py", line 367, in __new__' >&2
      echo "KeyError: 'crypto_stream_salsa20_NONCEBYTES'" >&2"#
    };
    write_shim(
        &bin.join("python"),
        &format!(
            r#"#!/bin/sh
# Like a real interpreter, ignore PYTHONPATH under -E and under -I (which implies
# -E), alone or combined with other short flags (`-sE`), before the `-c` script;
# an option's value (`-Wignore::ImportWarning`) is not a flag.
for arg in "$@"; do
  case "$arg" in
    -c) break ;;
    --*) ;;
    # `-W` and `-X` take a value that may contain a capital E or I
    # (`-Wignore::ImportWarning`, `-X importtime`): not the flags.
    -W*|-X*) ;;
    -*E*|-*I*) unset PYTHONPATH ;;
  esac
done
case "$*" in
  *tensorizer.serialization*)
    : > "{state}/probe-ran"
    if [ -e "{state}/library-failure" ] || [ -e "{state}/library-failure-in-tensorizer" ]; then
      echo 'File "/venv/lib/python3.14/site-packages/tensorizer/serialization.py", line 12, in <module>' >&2
      if [ -e "{state}/library-failure" ]; then
        echo 'File "/venv/lib/python3.14/site-packages/torch/__init__.py", line 300, in <module>' >&2
      fi
      echo 'OSError: libamdhip64.so.7: cannot open shared object file: No such file or directory' >&2
      exit 1
    fi
    if [ -e "{state}/key-error-in-torch" ]; then
      echo 'File "/venv/lib/python3.14/site-packages/tensorizer/serialization.py", line 12, in <module>' >&2
      echo 'File "/venv/lib/python3.14/site-packages/torch/_utils.py", line 80, in helper' >&2
      echo "KeyError: 'gfx1151'" >&2
      exit 1
    fi
    if [ -e "{state}/needs-pythonpath" ]; then
      case ":$PYTHONPATH:" in
        *":{state}/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi:"*) ;;
        *)
          echo "ModuleNotFoundError: No module named 'amdsmi'" >&2
          exit 1 ;;
      esac
    fi
    version=""
    [ -e "{state}/tensorizer-version" ] && read -r version < "{state}/tensorizer-version"
    if [ "$version" != "2.12.1" ]; then
      if [ "$VLLM_TARGET_DEVICE" != rocm ]; then
        echo 'OSError: libamdhip64.so.7: cannot open shared object file' >&2
        exit 1
      fi
      {traceback}
      exit 1
    fi ;;
  *find_spec*) echo '{{"present": true, "version": "0.29.1"}}' ;;
esac
exit 0
"#
        ),
    );
    if extra != Extra::VllmNotInstalled {
        write_shim(&bin.join("vllm"), "#!/bin/sh\nexit 0\n");
    }
    match extra {
        Extra::None => {}
        Extra::NeedsPythonPath => {
            std::fs::create_dir_all(rt.join("lib").join("python3.14").join("site-packages"))
                .expect("create site-packages");
            std::fs::write(rt.join("needs-pythonpath"), "").expect("write marker");
        }
        Extra::LibraryFailure => {
            std::fs::write(rt.join("library-failure"), "").expect("write marker");
        }
        Extra::KeyErrorInTorch => {
            std::fs::write(rt.join("key-error-in-torch"), "").expect("write marker");
        }
        Extra::ViolationBesideUpgrade | Extra::ViolationClearedOnlyByReinstall => {
            if extra == Extra::ViolationClearedOnlyByReinstall {
                std::fs::write(rt.join("repair-needs-reinstall-flag"), "").expect("write marker");
            }
            std::fs::write(rt.join("needs-repair"), "").expect("write marker");
            std::fs::write(rt.join("tensorizer-version"), "2.12.1\n").expect("write version");
        }
        Extra::VllmNotInstalled => {}
        Extra::LibraryFailureInTensorizer => {
            std::fs::write(rt.join("library-failure-in-tensorizer"), "").expect("write marker");
        }
    }

    // `pip check` reports vLLM's exact pin as unmet once 2.12.1 is installed,
    // as the real `uv` does; installing tensorizer records the version only
    // when the scenario says the upgrade helps.
    let record = match upgrade {
        Upgrade::Works => format!("echo 2.12.1 > \"{state}/tensorizer-version\""),
        Upgrade::NoEffect => "true".to_owned(),
        Upgrade::Fails => "echo 'error: network unreachable' >&2; exit 1".to_owned(),
    };
    write_shim(
        &shim.join("uv"),
        &format!(
            r#"#!/bin/sh
echo "$*" >> "{state}/uv.log"
case "$*" in
  *"pip check"*)
    failed=""
    if [ -e "{state}/needs-repair" ] && [ ! -e "{state}/repaired" ]; then
      echo "The package \`vllm\` requires \`numpy==2.0.0\`, but \`1.26.0\` is installed" >&2
      failed=1
    fi
    version=""
    [ -e "{state}/tensorizer-version" ] && read -r version < "{state}/tensorizer-version"
    if [ "$version" = "2.12.1" ]; then
      echo "The package \`vllm\` requires \`tensorizer==2.10.1\`, but \`2.12.1\` is installed" >&2
      failed=1
    fi
    [ -z "$failed" ] || exit 1 ;;
  *vllm==*)
    printf '#!/bin/sh\nexit 0\n' > "{state}/bin/vllm"
    echo 2.10.1 > "{state}/tensorizer-version"
    if [ ! -e "{state}/repair-needs-reinstall-flag" ]; then
      : > "{state}/repaired"
    else
      case "$*" in *--reinstall*) : > "{state}/repaired" ;; esac
    fi ;;
  *tensorizer==2.12.1*) {record} ;;
esac
exit 0
"#
        ),
    );
    // OpenMPI reads as present, so the install does not reach for a package manager.
    write_shim(
        &shim.join("ldconfig"),
        "#!/bin/sh\necho \"libmpi.so.40 (libc6,x86-64) => /usr/lib/libmpi.so.40\"\n",
    );
    write_shim(&shim.join("mpirun"), "#!/bin/sh\nexit 0\n");

    let registry = root.path().join("data").join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("create runtime registry");
    let manifest = serde_json::json!({
        "runtime_key": RUNTIME_KEY,
        "runtime_id": RUNTIME_KEY,
        "channel": "release",
        "format": "wheel",
        "family": "gfx1151",
        "family_source": "e2e",
        "version": if extra == Extra::NeedsPythonPath { "10.1.0" } else { "7.14.0" },
        "install_root": rt.display().to_string(),
        "python_executable": bin.join("python").display().to_string(),
        "selected_artifact_url": "https://example.invalid/e2e.whl",
        "installed_at_unix_ms": 1,
    });
    std::fs::write(
        registry.join(format!("{RUNTIME_KEY}.json")),
        serde_json::to_vec_pretty(&manifest).expect("manifest serialises"),
    )
    .expect("write runtime manifest");
}

fn install_vllm(world: &mut E2eWorld, extra: &[&str]) {
    run_engines_install(world, &["install", "vllm"], extra);
}

/// Runs `rocm engines <subcommand...> --runtime-id <planted> --yes <extra...>`.
fn run_engines_install(world: &mut E2eWorld, subcommand: &[&str], extra: &[&str]) {
    let shim = runtime_dir(world).join("shim");
    let path = shim.display().to_string();
    let uv = shim.join("uv").display().to_string();
    let mut args = vec!["engines"];
    args.extend_from_slice(subcommand);
    args.extend_from_slice(&["--runtime-id", RUNTIME_KEY, "--yes"]);
    args.extend_from_slice(extra);
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &args,
        &[
            ("PATH", path.as_str()),
            ("ROCM_CLI_UV_BINARY", uv.as_str()),
            // A decoy that never satisfies the stand-in, and that replaces whatever
            // PYTHONPATH the host exports: only the entry the server's launch
            // environment prepends to it does, so the result does not depend on
            // the host.
            ("PYTHONPATH", "/nonexistent/decoy-pythonpath"),
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined(world: &E2eWorld) -> String {
    format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

fn uv_log(world: &E2eWorld) -> String {
    std::fs::read_to_string(runtime_dir(world).join("uv.log")).unwrap_or_default()
}

#[given("a managed vLLM runtime whose pinned tensorizer cannot be imported")]
async fn plant_fixable(world: &mut E2eWorld) {
    plant(world, Upgrade::Works, false, Extra::None);
}

#[given("a managed vLLM runtime whose Python writes its tracebacks in colour")]
async fn plant_coloured(world: &mut E2eWorld) {
    plant(world, Upgrade::Works, true, Extra::None);
}

#[given("a managed vLLM runtime whose tensorizer cannot be imported even after an upgrade")]
async fn plant_unfixable(world: &mut E2eWorld) {
    plant(world, Upgrade::NoEffect, false, Extra::None);
}

#[given("a managed vLLM runtime whose pinned tensorizer cannot be imported and cannot be upgraded")]
async fn plant_unupgradable(world: &mut E2eWorld) {
    plant(world, Upgrade::Fails, false, Extra::None);
}

#[given("a managed vLLM runtime that only imports with the server's PYTHONPATH")]
async fn plant_needs_pythonpath(world: &mut E2eWorld) {
    plant(world, Upgrade::Works, false, Extra::NeedsPythonPath);
}

#[given("a managed vLLM runtime whose import fails on a missing library below tensorizer")]
async fn plant_library_failure(world: &mut E2eWorld) {
    plant(world, Upgrade::Works, false, Extra::LibraryFailure);
}

#[given("a managed vLLM runtime whose import raises a non-library error in torch below tensorizer")]
async fn plant_key_error_in_torch(world: &mut E2eWorld) {
    plant(world, Upgrade::Works, false, Extra::KeyErrorInTorch);
}

#[given(
    "a managed vLLM runtime whose violation beside the upgraded tensorizer only a reinstall clears"
)]
async fn plant_reinstall_only_violation(world: &mut E2eWorld) {
    plant(
        world,
        Upgrade::Works,
        false,
        Extra::ViolationClearedOnlyByReinstall,
    );
}

#[given("a managed vLLM runtime with a genuine violation beside the upgraded tensorizer")]
async fn plant_violation_beside_upgrade(world: &mut E2eWorld) {
    plant(world, Upgrade::Works, false, Extra::ViolationBesideUpgrade);
}

#[given(
    "a managed runtime without vLLM whose pinned tensorizer cannot be imported and cannot be upgraded"
)]
async fn plant_first_install_unupgradable(world: &mut E2eWorld) {
    plant(world, Upgrade::Fails, false, Extra::VllmNotInstalled);
}

#[given("a managed vLLM runtime whose tensorizer code raises a missing-library error")]
async fn plant_library_failure_in_tensorizer(world: &mut E2eWorld) {
    plant(
        world,
        Upgrade::Works,
        false,
        Extra::LibraryFailureInTensorizer,
    );
}

#[given("the user has installed the vLLM engine from that runtime")]
#[when("the user installs the vLLM engine from that runtime")]
async fn user_installs_vllm(world: &mut E2eWorld) {
    install_vllm(world, &[]);
}

#[when("the user reinstalls the vLLM engine from that runtime")]
async fn user_reinstalls_vllm(world: &mut E2eWorld) {
    install_vllm(world, &["--reinstall"]);
}

#[then("the install succeeds")]
async fn install_succeeds(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "expected the install to succeed:\n{}",
        combined(world)
    );
}

#[then("the user is told tensorizer was upgraded to 2.12.1")]
async fn told_upgraded(world: &mut E2eWorld) {
    let output = combined(world);
    assert!(
        output.contains(UPGRADE_WARNING) && output.contains("tensorizer"),
        "expected the upgrade warning:\n{output}"
    );
}

#[then("the dependency check reports the tensorizer pin as an expected divergence")]
async fn expected_divergence(world: &mut E2eWorld) {
    let output = combined(world);
    assert!(
        output.contains("dependency_check: expected_divergence")
            && output.contains("requires `tensorizer==2.10.1`, but `2.12.1` is installed"),
        "expected the tensorizer pin as an expected divergence:\n{output}"
    );
    // The block explains itself, so a repeat install (which prints no upgrade
    // warning) still says why, and the action line points at that note.
    // The note names the release the runtime holds, read from the state the
    // stand-in `uv` recorded rather than from a constant.
    let held =
        std::fs::read_to_string(runtime_dir(world).join("tensorizer-version")).unwrap_or_default();
    assert_eq!(held.trim(), "2.12.1", "the runtime must hold the upgrade");
    assert!(
        output.contains(&format!("this runtime has tensorizer {},", held.trim()))
            && output
                .contains("action: none; this tensorizer is kept on purpose (see the note above)"),
        "expected the divergence block to carry its own reason and action:\n{output}"
    );
    assert!(
        !output.contains("dependency_check: violated")
            && !output.contains("rocm engines install vllm --reinstall"),
        "the deliberate divergence must not read as a violation or advise a reinstall:\n{output}"
    );
}

#[then("tensorizer 2.12.1 is what the runtime now holds")]
async fn holds_fixed_tensorizer(world: &mut E2eWorld) {
    let version =
        std::fs::read_to_string(runtime_dir(world).join("tensorizer-version")).unwrap_or_default();
    assert_eq!(version.trim(), "2.12.1", "uv log:\n{}", uv_log(world));
    assert!(
        uv_log(world).contains("--no-deps tensorizer==2.12.1"),
        "the upgrade must leave vLLM's other dependencies alone:\n{}",
        uv_log(world)
    );
}

#[then("tensorizer was upgraded only once and vLLM was not reinstalled")]
async fn upgraded_once(world: &mut E2eWorld) {
    let log = uv_log(world);
    let upgrades = log
        .lines()
        .filter(|line| line.contains("tensorizer==2.12.1"))
        .count();
    assert_eq!(upgrades, 1, "uv log:\n{log}");
    assert!(
        !log.contains("vllm") && !log.contains("--reinstall"),
        "no vLLM install or reinstall may run on the second install:\n{log}"
    );
    assert!(
        !combined(world).contains(UPGRADE_WARNING),
        "the second install must not report a second upgrade:\n{}",
        combined(world)
    );
}

#[then("the install warns that tensorizer still cannot be imported and shows the KeyError")]
async fn warns_still_broken(world: &mut E2eWorld) {
    let output = combined(world);
    assert!(
        output.contains("tensorizer still cannot be imported after upgrading")
            && output.contains("KeyError: 'crypto_stream_salsa20_NONCEBYTES'"),
        "expected the warning to name the failure:\n{output}"
    );
    // No vLLM install ran on this runtime, so the warning must not claim one did,
    // and the upgrade was really attempted.
    assert!(
        !output.contains("was installed"),
        "an existing runtime's warning must not claim an install ran:\n{output}"
    );
    let log = uv_log(world);
    assert!(
        log.contains("--no-deps tensorizer==2.12.1") && !log.contains("vllm=="),
        "the upgrade is attempted and no vLLM install runs, uv log:\n{log}"
    );
}

#[then("the install warns that tensorizer could not be upgraded")]
async fn warns_not_upgraded(world: &mut E2eWorld) {
    let output = combined(world);
    assert!(
        output.contains("upgrading it to tensorizer==2.12.1 failed")
            && output.contains("network unreachable")
            && output.contains("vLLM would fail at startup"),
        "expected the failed-upgrade warning:\n{output}"
    );
    assert!(
        !runtime_dir(world).join("tensorizer-version").exists(),
        "a failed upgrade must leave the runtime's tensorizer as it was"
    );
}

#[then("the install is refused because tensorizer could not be upgraded")]
async fn refused_not_upgraded(world: &mut E2eWorld) {
    assert_ne!(world.cli_rc, Some(0), "{}", combined(world));
    let output = combined(world);
    assert!(
        output.contains("upgrading it to tensorizer==2.12.1 failed")
            && output.contains("network unreachable"),
        "expected the refusal to name the failed upgrade:\n{output}"
    );
    // This run installed vLLM (a `--reinstall` or a first install) before the check,
    // so the refusal may say so.
    assert!(
        output.contains("vLLM was installed, but"),
        "a refusal after this run installed vLLM should say the install ran:\n{output}"
    );
    assert!(
        uv_log(world).contains("vllm"),
        "this run must have installed vLLM, uv log:\n{}",
        uv_log(world)
    );
}

#[then("the install succeeds without changing tensorizer")]
async fn succeeds_without_change(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "{}", combined(world));
    let output = combined(world);
    assert!(
        !output.contains(UPGRADE_WARNING) && !uv_log(world).contains("tensorizer=="),
        "a failure that is not tensorizer's must leave it alone:\n{output}\n{}",
        uv_log(world)
    );
    assert!(
        !runtime_dir(world).join("tensorizer-version").exists(),
        "tensorizer must not have been touched"
    );
    assert!(
        runtime_dir(world).join("probe-ran").exists(),
        "the probe must have looked at tensorizer, or nothing was left alone on purpose"
    );
}

#[then(
    "the refusal says the install ran and tensorizer still cannot be imported after the upgrade"
)]
async fn refusal_after_install_and_upgrade(world: &mut E2eWorld) {
    assert_ne!(world.cli_rc, Some(0), "{}", combined(world));
    let output = combined(world);
    assert!(
        output.contains(
            "vLLM was installed, but tensorizer still cannot be imported after upgrading"
        ),
        "expected the fatal message:\n{output}"
    );
    let log = uv_log(world);
    let install = log.find("vllm==").expect("the reinstall ran");
    let upgrade = log
        .find("--no-deps tensorizer==2.12.1")
        .expect("the upgrade was attempted");
    assert!(install < upgrade, "the upgrade follows the install:\n{log}");
}

#[then("the violation is cleared and tensorizer is upgraded again after vLLM was reinstalled")]
async fn violation_cleared_and_tensorizer_kept(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "{}", combined(world));
    let output = combined(world);
    assert!(
        output.contains("dependency_check: expected_divergence"),
        "the repair must leave only the divergence:\n{output}"
    );
    // The repair warning above the check names the violation it found; the check
    // itself, after the install, must no longer.
    let (before, after) = output
        .split_once("dependency_check:")
        .expect("the dependency check ran");
    assert!(
        before.contains("violation: The package `vllm` requires `numpy")
            && !after.contains("numpy"),
        "the violation must be found by the repair and gone from the check:\n{output}"
    );
    let held =
        std::fs::read_to_string(runtime_dir(world).join("tensorizer-version")).unwrap_or_default();
    assert_eq!(
        held.trim(),
        TENSORIZER_FIXED,
        "the reinstall put vLLM's pin back, so tensorizer must have been upgraded again"
    );
    let log = uv_log(world);
    let install = log.find("vllm==").expect("vLLM was reinstalled");
    let upgrade = log
        .find("--no-deps tensorizer==2.12.1")
        .expect("tensorizer was upgraded");
    assert!(install < upgrade, "the upgrade follows the install:\n{log}");
}

/// The command the CLI printed beside the violation, as the `action:` line gives it.
fn advised_command(world: &E2eWorld) -> Vec<String> {
    let output = combined(world);
    let line = output
        .lines()
        .find_map(|line| line.trim().strip_prefix("action: rocm "))
        .unwrap_or_else(|| panic!("the CLI printed no `action: rocm ...` advice:\n{output}"));
    line.split_whitespace().map(str::to_owned).collect()
}

#[then("the install reports the violation and advises reinstalling vLLM")]
async fn reports_violation_and_advises_reinstall(world: &mut E2eWorld) {
    let output = combined(world);
    assert!(
        output.contains("dependency_check: violated")
            && output.contains("violation: The package `vllm` requires `numpy")
            && output.contains("divergence: The package `vllm` requires `tensorizer"),
        "expected the violation beside the divergence:\n{output}"
    );
    let advice = advised_command(world);
    assert_eq!(
        advice.first().map(String::as_str),
        Some("engines"),
        "{advice:?}"
    );
    // The plain install left the violation: the repair did not clear it.
    assert!(
        !runtime_dir(world).join("repaired").exists(),
        "the stand-in clears the violation only on --reinstall"
    );
}

#[when("the user runs the advised command")]
async fn runs_advised_command(world: &mut E2eWorld) {
    let advice = advised_command(world);
    let words: Vec<&str> = advice.iter().map(String::as_str).collect();
    // The advice names the subcommand and flags; the planted runtime id and the
    // approval are what a user would add for this machine.
    let split = words
        .iter()
        .position(|part| part.starts_with("--"))
        .unwrap_or(words.len());
    let (subcommand, flags) = words.split_at(split);
    run_engines_install(world, &subcommand[1..], flags);
}

#[then("the advised reinstall cleared the violation and tensorizer was upgraded after it")]
async fn advised_reinstall_worked(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "{}", combined(world));
    let output = combined(world);
    let (_, after) = output
        .split_once("dependency_check:")
        .expect("the dependency check ran");
    assert!(
        output.contains("dependency_check: expected_divergence") && !after.contains("numpy"),
        "the reinstall must clear the violation and leave only the divergence:\n{output}"
    );
    assert!(runtime_dir(world).join("repaired").exists());
    let held =
        std::fs::read_to_string(runtime_dir(world).join("tensorizer-version")).unwrap_or_default();
    assert_eq!(held.trim(), TENSORIZER_FIXED);
    let log = uv_log(world);
    let reinstall = log
        .lines()
        .position(|line| line.contains("vllm==") && line.contains("--reinstall"))
        .unwrap_or_else(|| panic!("vLLM was never reinstalled:\n{log}"));
    let upgrade = log
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("--no-deps tensorizer==2.12.1"))
        .map(|(index, _)| index)
        .last()
        .expect("tensorizer was upgraded");
    assert!(
        upgrade > reinstall,
        "tensorizer must be upgraded after the reinstall put vLLM's pin back:\n{log}"
    );
}
