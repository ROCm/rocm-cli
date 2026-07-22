// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Black-box steps for the release install lifecycle (`@lifecycle`).
//!
//! These drive the real release surface end to end: `cargo xtask package` to
//! build a signed bundle, the root installer (`install.sh` / `install.ps1`) to
//! verify and activate it, and the installed binaries for the smoke and
//! uninstall phases. They replace the former
//! `scripts/acceptance-install-upgrade-tui-uninstall.{sh,ps1}` scripts, keeping
//! the union of both scripts' assertions.
//!
//! Everything a scenario creates lives under its `E2eWorld` temp root, so
//! teardown removes it. The one machine-global side effect — the Windows user
//! PATH, which `install.ps1`'s default install mutates — is captured and restored
//! by [`LifecycleState`]'s `Drop`, even when a step fails.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use cucumber::{given, then, when};

use crate::E2eWorld;
use crate::e2e::tui_driver::{DEFAULT_TIMEOUT, TuiSession};

/// Per-scenario release-lifecycle state. All paths are rooted in the scenario's
/// isolated temp dir; `Drop` restores the captured Windows user PATH.
#[derive(Debug, Default)]
pub struct LifecycleState {
    /// Distribution name (archive stem + bundle directory name).
    dist_name: String,
    /// Directory holding the bundle directory and archive (`<root>/dist`).
    dist_dir: PathBuf,
    /// Signing key paths.
    private_key: PathBuf,
    public_key: PathBuf,
    /// Where the installer places binaries (`<root>/install/bin`).
    install_dir: PathBuf,
    /// Isolated HOME for the installer's shell-profile writes (Linux).
    home_dir: PathBuf,
    /// Download base the installer fetches from (usually `file://<dist_dir>`).
    download_base: String,
    /// The most recent installer/CLI output captured for assertions.
    last_output: String,
    /// The most recent installed-binary exit code captured for assertions.
    last_rc: i32,
    /// A generated installer fixture with controlled pinned trust roots.
    installer_fixture: Option<PathBuf>,
    /// A captured Windows user PATH to restore on drop (Windows only).
    captured_user_path: Option<String>,
    /// Isolated config/data/cache directories for installed-binary smoke tests.
    smoke_config: Option<PathBuf>,
    smoke_data: Option<PathBuf>,
    smoke_cache: Option<PathBuf>,
}

impl Drop for LifecycleState {
    fn drop(&mut self) {
        // Restore the machine user PATH if a scenario mutated it, even on panic.
        #[cfg(windows)]
        if let Some(previous) = self.captured_user_path.take() {
            restore_user_path(&previous);
        }
        // A no-op reference on non-Windows so the field is always "used".
        let _ = &self.captured_user_path;
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Absolute workspace root, derived from this crate's manifest dir
/// (`<root>/tests/e2e-cucumber`).
fn workspace_root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR")))
        .join("../..")
        .canonicalize()
        .expect("failed to resolve workspace root")
}

/// The isolated temp root for this scenario.
fn root(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .to_path_buf()
}

/// Mutable access to the scenario's lifecycle state, panicking with a clear
/// message if a step ran out of order without initializing it.
const fn state_mut(world: &mut E2eWorld) -> &mut LifecycleState {
    world
        .lifecycle
        .as_mut()
        .expect("lifecycle state not initialized; a Given step must run first")
}

const fn state(world: &E2eWorld) -> &LifecycleState {
    world
        .lifecycle
        .as_ref()
        .expect("lifecycle state not initialized; a Given step must run first")
}

/// The archive file name for the current platform.
fn archive_name(dist_name: &str) -> String {
    if cfg!(windows) {
        format!("{dist_name}.zip")
    } else {
        format!("{dist_name}.tar.gz")
    }
}

/// Run `cargo xtask package <dist> <dist_dir>` from the workspace root, with the
/// given signing environment, and return combined stdout+stderr.
fn run_package(world: &E2eWorld, sign_env: &[(&str, String)]) -> (String, bool) {
    let st = state(world);
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.args(["xtask", "package", &st.dist_name])
        .arg(&st.dist_dir)
        .current_dir(workspace_root());
    // Point packaging at the already-built release binaries so a scenario does
    // not rebuild them; the harness's `cargo xtask e2e` built them first.
    cmd.env("ROCM_BIN_DIR", release_bin_dir());
    cmd.env("ROCM_CLI_REQUIRE_SIGNATURE", "1");
    for (key, value) in sign_env {
        cmd.env(key, value);
    }
    let output = cmd.output().expect("failed to run cargo xtask package");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (combined, output.status.success())
}

/// Directory holding the built release binaries the harness produced. Derived
/// from the `rocm` binary path the harness exports via `ROCM_CLI_BINARY`, so
/// packaging bundles exactly the binaries under test.
fn release_bin_dir() -> PathBuf {
    let binary = crate::rocm_binary();
    let path = PathBuf::from(&binary);
    path.parent().map_or_else(
        || workspace_root().join("target/release"),
        Path::to_path_buf,
    )
}

/// Run the platform installer (`install.sh` / `install.ps1`) with the given
/// extra environment, returning combined output and success.
fn run_installer(
    world: &E2eWorld,
    install_dir: &Path,
    download_base: &str,
    public_key_env: &[(&str, String)],
    extra_env: &[(&str, String)],
) -> (String, bool) {
    let root = workspace_root();
    let cmd = installer_command(&root);
    run_installer_command(
        world,
        cmd,
        &root,
        install_dir,
        download_base,
        public_key_env,
        extra_env,
    )
}

fn run_installer_command(
    world: &E2eWorld,
    mut cmd: Command,
    current_dir: &Path,
    install_dir: &Path,
    download_base: &str,
    public_key_env: &[(&str, String)],
    extra_env: &[(&str, String)],
) -> (String, bool) {
    let st = state(world);
    cmd.current_dir(current_dir);
    cmd.env("ROCM_CLI_DOWNLOAD_BASE", download_base);
    cmd.env("ROCM_CLI_INSTALL_DIR", install_dir);
    cmd.env("ROCM_CLI_REQUIRE_SIGNATURE", "1");
    // Isolate the installer's HOME so shell-profile writes and the seeded
    // ~/.rocm/config.json land in the scenario's tree, never the real home.
    cmd.env("HOME", &st.home_dir);
    for (key, value) in public_key_env {
        cmd.env(key, value);
    }
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    let output = cmd.output().expect("failed to run the installer");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (combined, output.status.success())
}

/// Build the platform-appropriate installer command (`sh install.sh release` on
/// Unix; `pwsh -File install.ps1 release` on Windows).
#[cfg(not(windows))]
fn installer_command(root: &Path) -> Command {
    let mut cmd = Command::new("sh");
    cmd.arg(root.join("install.sh")).arg("release");
    cmd
}

#[cfg(windows)]
fn installer_command(root: &Path) -> Command {
    powershell_installer_command(&root.join("install.ps1"))
}

#[cfg(windows)]
fn powershell_installer_command(installer: &Path) -> Command {
    // Prefer pwsh; fall back to Windows PowerShell.
    let shell = if which_ok("pwsh") {
        "pwsh"
    } else {
        "powershell"
    };
    let mut cmd = Command::new(shell);
    cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(installer)
        .arg("release");
    cmd
}

#[cfg(windows)]
fn which_ok(name: &str) -> bool {
    Command::new(name)
        .arg("-NoProfile")
        .arg("-Command")
        .arg("exit 0")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Capture the current Windows user PATH (registry `HKCU\Environment\Path`).
#[cfg(windows)]
fn capture_user_path() -> String {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "[Environment]::GetEnvironmentVariable('Path','User')",
        ])
        .output();
    output
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Restore the Windows user PATH to a previously captured value.
///
/// Runs during `Drop`, so it must not panic; a failed restore is loud on stderr
/// instead, because a botched restore leaves the runner's real user PATH
/// polluted with the test's temp install dir and that must be diagnosable.
#[cfg(windows)]
fn restore_user_path(value: &str) {
    let script = "[Environment]::SetEnvironmentVariable('Path', $env:ROCM_E2E_PREV_PATH, 'User')";
    let status = Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .env("ROCM_E2E_PREV_PATH", value)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => eprintln!(
            "warning: failed to restore Windows user PATH (exit {status}); the runner PATH may still reference the test install dir"
        ),
        Err(error) => eprintln!(
            "warning: could not run PowerShell to restore Windows user PATH ({error}); the runner PATH may still reference the test install dir"
        ),
    }
}

/// The real (non-isolated) user rocm state dir, `~/.rocm`, if a home dir is
/// resolvable. Used only for the negative isolation assertion: the installed
/// binary under isolated `ROCM_CLI_*_DIR` env must never read this.
fn real_user_rocm_dir() -> Option<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    let home = PathBuf::from(home);
    if home.as_os_str().is_empty() {
        return None;
    }
    Some(home.join(".rocm"))
}

/// The installer's per-user config file under the isolated HOME.
fn installed_config_file(world: &E2eWorld) -> PathBuf {
    state(world).home_dir.join(".rocm").join("config.json")
}

/// The installed binary's platform file name.
fn installed_binary(install_dir: &Path, name: &str) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    install_dir.join(file)
}

/// Assert a path exists.
fn assert_exists(path: &Path) {
    assert!(path.exists(), "expected path to exist: {}", path.display());
}

/// Assert a path does not exist.
fn assert_missing(path: &Path) {
    assert!(
        !path.exists(),
        "expected path to be removed: {}",
        path.display()
    );
}

// ── Given: setup ───────────────────────────────────────────────────────

#[given("a freshly built release tree")]
async fn given_release_tree(world: &mut E2eWorld) {
    // The harness already built the release `rocm`/`rocmd` (see `cargo xtask
    // e2e`); confirm they are present so a packaging failure is diagnosable.
    let bin_dir = release_bin_dir();
    for name in ["rocm", "rocmd"] {
        let path = installed_binary(&bin_dir, name);
        assert!(
            path.exists(),
            "release binary missing: {} (build with `cargo build --release -p rocm -p rocmd`)",
            path.display()
        );
    }

    let root = root(world);
    let dist_name = if cfg!(windows) {
        "rocm-cli-windows-amd64"
    } else {
        "rocm-cli-linux-amd64"
    }
    .to_string();
    let dist_dir = root.join("dist");
    let home_dir = root.join("home");
    std::fs::create_dir_all(&home_dir).expect("failed to create isolated HOME");

    world.lifecycle = Some(LifecycleState {
        dist_name,
        dist_dir,
        private_key: PathBuf::new(),
        public_key: PathBuf::new(),
        install_dir: root.join("install").join("bin"),
        home_dir,
        download_base: String::new(),
        last_output: String::new(),
        last_rc: 0,
        installer_fixture: None,
        captured_user_path: None,
        smoke_config: None,
        smoke_data: None,
        smoke_cache: None,
    });
}

#[given("a generated signing keypair")]
async fn given_keypair(world: &mut E2eWorld) {
    let root = root(world);
    let private_key = root.join("signing-private.pem");
    let public_key = root.join("signing-public.pem");
    let status = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["xtask", "keygen"])
        .arg("--private-out")
        .arg(&private_key)
        .arg("--public-out")
        .arg(&public_key)
        .current_dir(workspace_root())
        .status()
        .expect("failed to run cargo xtask keygen");
    assert!(status.success(), "keygen failed");
    let st = state_mut(world);
    st.private_key = private_key;
    st.public_key = public_key;
}

#[given("a signed bundle installed with the shell profile updated")]
async fn given_installed_shell_profile(world: &mut E2eWorld) {
    package_with_key_file(world).await;
    install_with_key_file_shell_profile(world).await;
}

#[given("a signed bundle installed with the public key file")]
async fn given_installed_key_file(world: &mut E2eWorld) {
    package_with_key_file(world).await;
    install_with_key_file(world).await;
}

#[given("the installed binary has isolated XDG directories with state")]
async fn given_isolated_xdg(world: &mut E2eWorld) {
    seed_isolated_dirs(world);
}

#[given("the installed binary has isolated directories with state")]
async fn given_isolated_dirs(world: &mut E2eWorld) {
    seed_isolated_dirs(world);
}

#[given("the user PATH is captured for restoration")]
async fn given_capture_user_path(world: &mut E2eWorld) {
    #[cfg(windows)]
    {
        state_mut(world).captured_user_path = Some(capture_user_path());
    }
    #[cfg(not(windows))]
    {
        let _ = world;
    }
}

/// Seed isolated config/data/cache dirs with a marker so the installed-binary
/// smoke can prove the CLI reads only them, not the real user state.
fn seed_isolated_dirs(world: &mut E2eWorld) {
    let root = root(world);
    let config = root.join("smoke-config");
    let data = root.join("smoke-data");
    let cache = root.join("smoke-cache");
    for dir in [&config, &data, &cache] {
        std::fs::create_dir_all(dir).expect("failed to create isolated smoke dir");
    }
    let st = state_mut(world);
    st.smoke_config = Some(config);
    st.smoke_data = Some(data);
    st.smoke_cache = Some(cache);
}

// ── When: package ──────────────────────────────────────────────────────

async fn package_with_key_file(world: &mut E2eWorld) {
    let st = state(world);
    let sign_env = [(
        "ROCM_CLI_SIGNING_PRIVATE_KEY_PATH",
        st.private_key.to_string_lossy().into_owned(),
    )];
    let (out, ok) = run_package(world, &sign_env);
    assert!(ok, "packaging with key file failed:\n{out}");
    finalize_package(world);
}

#[when("the release is packaged and signed with the private key file")]
async fn when_package_key_file(world: &mut E2eWorld) {
    package_with_key_file(world).await;
}

#[when("the release is packaged and signed with the private key PEM")]
async fn when_package_pem(world: &mut E2eWorld) {
    let st = state(world);
    let pem = std::fs::read_to_string(&st.private_key).expect("failed to read private key");
    let sign_env = [("ROCM_CLI_SIGNING_PRIVATE_KEY_PEM", pem)];
    let (out, ok) = run_package(world, &sign_env);
    assert!(ok, "packaging with PEM failed:\n{out}");
    finalize_package(world);
}

/// Record the default `file://` download base after a successful package, and
/// assert the archive and its sidecars exist.
fn finalize_package(world: &mut E2eWorld) {
    let st = state_mut(world);
    let archive = st.dist_dir.join(archive_name(&st.dist_name));
    assert!(
        archive.exists(),
        "packaged archive missing: {}",
        archive.display()
    );
    assert!(
        archive.with_extension("").exists() || archive.exists(),
        "archive missing"
    );
    let sig = with_suffix(&archive, ".sig");
    assert!(sig.exists(), "signature sidecar missing: {}", sig.display());
    let sha = with_suffix(&archive, ".sha256");
    assert!(sha.exists(), "checksum sidecar missing: {}", sha.display());
    st.download_base = file_url(&st.dist_dir);
}

/// Append a suffix to a path's file name (e.g. `.sig`, `.sha256`).
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    name.push(suffix);
    path.with_file_name(name)
}

/// A `file://` URL for a directory the installer downloads from.
fn file_url(dir: &Path) -> String {
    // The installer accepts a directory `file://` base and appends the archive
    // name. On Unix this is `file://<abs>`; keep it simple and absolute.
    format!("file://{}", dir.display())
}

// ── When: tamper ───────────────────────────────────────────────────────

#[when("the bundle checksum sidecar is corrupted")]
async fn when_corrupt_checksum(world: &mut E2eWorld) {
    let st = state(world);
    let archive = st.dist_dir.join(archive_name(&st.dist_name));
    let sha = with_suffix(&archive, ".sha256");
    let bad = format!("{}  {}\n", "0".repeat(64), archive_name(&st.dist_name));
    std::fs::write(&sha, bad).expect("failed to corrupt checksum sidecar");
}

#[when("the bundle signature sidecar is corrupted")]
async fn when_corrupt_signature(world: &mut E2eWorld) {
    let st = state(world);
    let archive = st.dist_dir.join(archive_name(&st.dist_name));
    let sig = with_suffix(&archive, ".sig");
    std::fs::write(&sig, b"not a real signature\n").expect("failed to corrupt signature sidecar");
}

#[when("the bundle signature sidecar is removed")]
async fn when_remove_signature(world: &mut E2eWorld) {
    let st = state(world);
    let archive = st.dist_dir.join(archive_name(&st.dist_name));
    let sig = with_suffix(&archive, ".sig");
    std::fs::remove_file(&sig).expect("failed to remove signature sidecar");
}

// ── When: install ──────────────────────────────────────────────────────

async fn install_with_key_file(world: &mut E2eWorld) {
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let download_base = st.download_base.clone();
    let key_env = [(
        "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH",
        st.public_key.to_string_lossy().into_owned(),
    )];
    let no_path = [("ROCM_CLI_UPDATE_SHELL_PATH", "0".to_string())];
    let (out, _ok) = run_installer(world, &install_dir, &download_base, &key_env, &no_path);
    state_mut(world).last_output = out;
}

async fn install_with_key_file_shell_profile(world: &mut E2eWorld) {
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let download_base = st.download_base.clone();
    let key_env = [(
        "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH",
        st.public_key.to_string_lossy().into_owned(),
    )];
    let shell_on = [("ROCM_CLI_UPDATE_SHELL_PATH", "1".to_string())];
    let (out, _ok) = run_installer(world, &install_dir, &download_base, &key_env, &shell_on);
    state_mut(world).last_output = out;
}

const MALFORMED_PUBLIC_KEY: &str =
    "-----BEGIN PUBLIC KEY-----\nnot-valid-base64\n-----END PUBLIC KEY-----";

fn write_pinned_key_installer_fixture(world: &mut E2eWorld, current_key: &str, next_key: &str) {
    let source_path = workspace_root().join("install.ps1");
    let source = std::fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", source_path.display()));
    let fixture =
        e2e_cucumber::installer_fixture::with_pinned_release_keys(&source, current_key, next_key)
            .unwrap_or_else(|e| panic!("failed to generate pinned-key installer fixture: {e}"));
    let fixture_path = root(world).join("install-pinned-keys.ps1");
    std::fs::write(&fixture_path, fixture)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fixture_path.display()));
    state_mut(world).installer_fixture = Some(fixture_path);
}

#[when("an installer fixture has a malformed current key and the generated public key next")]
async fn when_fixture_has_valid_rotation_key(world: &mut E2eWorld) {
    let public_key = std::fs::read_to_string(&state(world).public_key)
        .expect("failed to read generated public key");
    write_pinned_key_installer_fixture(world, MALFORMED_PUBLIC_KEY, &public_key);
}

#[when("an installer fixture has malformed current and next keys")]
async fn when_fixture_has_all_malformed_keys(world: &mut E2eWorld) {
    write_pinned_key_installer_fixture(world, MALFORMED_PUBLIC_KEY, MALFORMED_PUBLIC_KEY);
}

#[when("the signed bundle is installed through the pinned-key fixture")]
async fn when_install_pinned_key_fixture(world: &mut E2eWorld) {
    #[cfg(windows)]
    {
        let st = state(world);
        let fixture = st
            .installer_fixture
            .as_ref()
            .expect("pinned-key installer fixture was not generated")
            .clone();
        let install_dir = st.install_dir.clone();
        let download_base = st.download_base.clone();
        let no_path = [("ROCM_CLI_UPDATE_USER_PATH", "0".to_string())];
        let cmd = powershell_installer_command(&fixture);
        let (out, _ok) = run_installer_command(
            world,
            cmd,
            &workspace_root(),
            &install_dir,
            &download_base,
            &[],
            &no_path,
        );
        state_mut(world).last_output = out;
    }
    #[cfg(not(windows))]
    {
        let _ = world;
        panic!("the pinned-key installer fixture is Windows-only");
    }
}

#[when("the signed bundle is installed with the public key file")]
async fn when_install_key_file(world: &mut E2eWorld) {
    install_with_key_file(world).await;
}

#[when("the signed bundle is installed with the public key file updating the shell profile")]
async fn when_install_key_file_shell(world: &mut E2eWorld) {
    install_with_key_file_shell_profile(world).await;
}

#[when("the signed bundle is reinstalled with the public key file")]
async fn when_reinstall_key_file(world: &mut E2eWorld) {
    install_with_key_file(world).await;
}

#[when("the signed bundle is installed with the public key file and openssl removed from PATH")]
async fn when_install_no_openssl(world: &mut E2eWorld) {
    // Prove the installer verifies with native crypto: run the install with every
    // PATH entry that contains an `openssl` executable stripped out, so the
    // installer cannot shell out to openssl. Non-Windows hosts skip the openssl
    // scrubbing behaviour (this scenario is @requires-os:windows).
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let download_base = st.download_base.clone();
    let key_env = [(
        "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH",
        st.public_key.to_string_lossy().into_owned(),
    )];
    let scrubbed = path_without_openssl();
    let extra = [
        ("ROCM_CLI_UPDATE_SHELL_PATH", "0".to_string()),
        ("PATH", scrubbed),
    ];
    let (out, _ok) = run_installer(world, &install_dir, &download_base, &key_env, &extra);
    state_mut(world).last_output = out;
}

/// The current `PATH` with every directory that contains an `openssl` executable
/// removed, so a child installer cannot resolve openssl even if the host has it.
fn path_without_openssl() -> String {
    let sep = if cfg!(windows) { ';' } else { ':' };
    let candidates = if cfg!(windows) {
        &["openssl.exe", "openssl"][..]
    } else {
        &["openssl"][..]
    };
    let current = std::env::var("PATH").unwrap_or_default();
    current
        .split(sep)
        .filter(|entry| {
            !entry.is_empty()
                && !candidates
                    .iter()
                    .any(|exe| Path::new(entry).join(exe).is_file())
        })
        .collect::<Vec<_>>()
        .join(&sep.to_string())
}

#[when("the signed bundle is installed with the public key PEM")]
async fn when_install_key_pem(world: &mut E2eWorld) {
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let download_base = st.download_base.clone();
    let pem = std::fs::read_to_string(&st.public_key).expect("failed to read public key");
    let key_env = [("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM", pem)];
    let no_path = [("ROCM_CLI_UPDATE_SHELL_PATH", "0".to_string())];
    let (out, _ok) = run_installer(world, &install_dir, &download_base, &key_env, &no_path);
    state_mut(world).last_output = out;
}

#[when("the signed bundle is installed with no public key supplied")]
async fn when_install_no_key(world: &mut E2eWorld) {
    // No public key → the installer falls back to the production pinned trust
    // root, which the acceptance key is not, so verification must fail.
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let download_base = st.download_base.clone();
    let no_path = [("ROCM_CLI_UPDATE_SHELL_PATH", "0".to_string())];
    let (out, _ok) = run_installer(world, &install_dir, &download_base, &[], &no_path);
    state_mut(world).last_output = out;
}

#[when("the signed bundle is installed updating the user PATH")]
async fn when_install_update_user_path(world: &mut E2eWorld) {
    // Windows-only: a default install (no -NoPathUpdate) mutates the user PATH.
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let download_base = st.download_base.clone();
    let key_env = [(
        "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH",
        st.public_key.to_string_lossy().into_owned(),
    )];
    // Do NOT pass ROCM_CLI_UPDATE_SHELL_PATH=0 so the user PATH is updated.
    let (out, _ok) = run_installer(world, &install_dir, &download_base, &key_env, &[]);
    state_mut(world).last_output = out;
}

#[when("the signed bundle is reinstalled with the shell profile updated")]
async fn when_reinstall_shell_profile(world: &mut E2eWorld) {
    install_with_key_file_shell_profile(world).await;
}

#[when("a stale engine entry is recorded in the prior install")]
async fn when_record_stale(world: &mut E2eWorld) {
    let st = state(world);
    let install_dir = st.install_dir.clone();
    let stale = installed_binary(&install_dir, "rocm-engine-stale");
    std::fs::write(&stale, b"stale").expect("failed to write stale engine");
    let manifest = install_dir.join(".rocm-cli-manifest");
    let mut contents = std::fs::read_to_string(&manifest).unwrap_or_default();
    let _ = writeln!(contents, "{}", stale.display());
    std::fs::write(&manifest, contents).expect("failed to append to manifest");
}

#[when("the user changes the default engine to vllm in the installed config")]
async fn when_change_engine(world: &mut E2eWorld) {
    let config = installed_config_file(world);
    std::fs::write(&config, "{\"default_engine\":\"vllm\"}\n")
        .expect("failed to write installed config");
}

#[when("the signed bundle is installed over a loopback HTTP server with the public key file")]
async fn when_install_http(world: &mut E2eWorld) {
    // Serve the signed dist over loopback HTTP so the installer exercises its
    // native HTTP download path, then install and verify.
    let st = state(world);
    let dist_dir = st.dist_dir.clone();
    let install_dir = st.install_dir.clone();
    let public_key = st.public_key.clone();
    let server = crate::e2e::lifecycle_steps::http::LoopbackServer::start(&dist_dir);
    let base = server.base_url();
    let key_env = [(
        "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH",
        public_key.to_string_lossy().into_owned(),
    )];
    let no_path = [("ROCM_CLI_UPDATE_SHELL_PATH", "0".to_string())];
    let (out, _ok) = run_installer(world, &install_dir, &base, &key_env, &no_path);
    state_mut(world).last_output = out;
    drop(server);
}

/// Run an installed binary with the isolated smoke env and record rc + output.
fn run_installed(world: &mut E2eWorld, binary: &str, args: &[&str]) {
    let st = state(world);
    let exe = installed_binary(&st.install_dir, binary);
    let mut cmd = Command::new(&exe);
    cmd.args(args);
    apply_smoke_env(&mut cmd, st);
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run installed {binary}: {e}"));
    let rc = output.status.code().unwrap_or(-1);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let st = state_mut(world);
    st.last_output = combined;
    st.last_rc = rc;
}

#[when("the installed rocm opens interactive chat through a pseudo-terminal")]
async fn when_installed_chat_pty(world: &mut E2eWorld) {
    let rocm = installed_binary(&state(world).install_dir, "rocm");
    let session = TuiSession::spawn_binary(world, &rocm, &["chat", "--chat-mock"])
        .unwrap_or_else(|e| panic!("failed to open installed interactive chat: {e}"));
    world.tui = Some(session);
}

#[when("the default engine is set to vllm in the installed config")]
async fn when_set_engine_installed(world: &mut E2eWorld) {
    let rocm = installed_binary(&state(world).install_dir, "rocm");
    let mut cmd = Command::new(&rocm);
    cmd.args(["config", "set-default-engine", "vllm"]);
    // Use the same isolated config dir the PTY chat inherits, so setting the
    // engine here and reading it back after the session exercises one config.
    for (key, value) in world.isolate_env() {
        cmd.env(key, value);
    }
    let output = cmd
        .output()
        .expect("failed to run installed rocm config set-default-engine");
    assert!(
        output.status.success(),
        "setting the default engine failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[then("the installed config still selects the vllm default engine")]
async fn then_installed_config_survives(world: &mut E2eWorld) {
    // The PTY chat inherits isolate_env, whose ROCM_CLI_CONFIG_DIR is
    // `<isolated_root>/config`; config.json lives directly under it.
    let config_dir = world
        .isolate_env()
        .into_iter()
        .find(|(key, _)| *key == "ROCM_CLI_CONFIG_DIR")
        .map(|(_, value)| PathBuf::from(value))
        .expect("isolate_env did not set ROCM_CLI_CONFIG_DIR");
    let config = config_dir.join("config.json");
    let contents = std::fs::read_to_string(&config)
        .unwrap_or_else(|e| panic!("installed config {} not readable: {e}", config.display()));
    assert!(
        contents.contains("\"vllm\""),
        "interactive chat clobbered the default engine config:\n{contents}"
    );
}

#[when("the user quits the installed interactive chat")]
async fn when_quit_installed_chat(world: &mut E2eWorld) {
    world
        .tui
        .as_mut()
        .expect("no installed interactive chat session is open")
        .quit_and_wait(DEFAULT_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("installed interactive chat did not exit cleanly: {e}"));
}

#[when("the installed rocm version runs")]
async fn when_installed_version(world: &mut E2eWorld) {
    run_installed(world, "rocm", &["version"]);
}

#[when("the installed rocm engines list runs")]
async fn when_installed_engines_list(world: &mut E2eWorld) {
    run_installed(world, "rocm", &["engines", "list"]);
}

#[when("the installed rocmd status runs")]
async fn when_installed_rocmd_status(world: &mut E2eWorld) {
    run_installed(world, "rocmd", &["status"]);
}

#[when("the installed rocm examine runs with the isolated environment")]
async fn when_examine_isolated(world: &mut E2eWorld) {
    let st = state(world);
    let rocm = installed_binary(&st.install_dir, "rocm");
    let mut cmd = Command::new(&rocm);
    cmd.arg("examine");
    if let Some(dir) = &st.smoke_config {
        cmd.env("ROCM_CLI_CONFIG_DIR", dir);
    }
    if let Some(dir) = &st.smoke_data {
        cmd.env("ROCM_CLI_DATA_DIR", dir);
    }
    if let Some(dir) = &st.smoke_cache {
        cmd.env("ROCM_CLI_CACHE_DIR", dir);
    }
    let output = cmd.output().expect("failed to run installed rocm examine");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    state_mut(world).last_output = combined;
}

#[when("the user uninstalls from the installed binary")]
async fn when_uninstall(world: &mut E2eWorld) {
    let st = state(world);
    let rocm = installed_binary(&st.install_dir, "rocm");
    let mut cmd = Command::new(&rocm);
    cmd.arg("uninstall").arg("--yes");
    apply_smoke_env(&mut cmd, st);
    let output = cmd.output().expect("failed to run uninstall");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    state_mut(world).last_output = combined;
}

#[when("the user uninstalls from the installed binary keeping config data and cache")]
async fn when_uninstall_keep(world: &mut E2eWorld) {
    let st = state(world);
    let rocm = installed_binary(&st.install_dir, "rocm");
    let mut cmd = Command::new(&rocm);
    cmd.args([
        "uninstall",
        "--yes",
        "--keep-config",
        "--keep-data",
        "--keep-cache",
    ]);
    apply_smoke_env(&mut cmd, st);
    let output = cmd.output().expect("failed to run uninstall");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    state_mut(world).last_output = combined;
}

/// Apply the isolated config/data/cache env for installed-binary invocations.
fn apply_smoke_env(cmd: &mut Command, st: &LifecycleState) {
    if let Some(dir) = &st.smoke_config {
        cmd.env("ROCM_CLI_CONFIG_DIR", dir);
    }
    if let Some(dir) = &st.smoke_data {
        cmd.env("ROCM_CLI_DATA_DIR", dir);
    }
    if let Some(dir) = &st.smoke_cache {
        cmd.env("ROCM_CLI_CACHE_DIR", dir);
    }
}

// ── Then: assertions ───────────────────────────────────────────────────

#[then("the installer reports the signature verified")]
async fn then_signature_verified(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("signature verified"),
        "installer did not report signature verification:\n{out}"
    );
}

#[then("the installer reports the user PATH updated")]
async fn then_user_path_updated(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("user PATH updated"),
        "installer did not report the user PATH update:\n{out}"
    );
}

#[then("the installer reports the installer process PATH updated")]
async fn then_installer_path_updated(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("installer PATH updated"),
        "installer did not report updating its own process PATH:\n{out}"
    );
}

#[then("the installer reports the shell profile updated")]
async fn then_shell_profile_updated(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("shell profile updated"),
        "installer did not report the shell profile update:\n{out}"
    );
}

#[then("the shell profile lists the install directory")]
async fn then_shell_profile_lists_dir(world: &mut E2eWorld) {
    let st = state(world);
    let profile = st.home_dir.join(".bashrc");
    let contents = std::fs::read_to_string(&profile).unwrap_or_default();
    let install_dir = st.install_dir.to_string_lossy();
    assert!(
        contents.contains(install_dir.as_ref()),
        "shell profile did not list the install directory {install_dir}:\n{contents}"
    );
}

#[then("the installed interactive chat surface is displayed")]
async fn then_installed_chat_displayed(world: &mut E2eWorld) {
    world
        .tui
        .as_mut()
        .expect("no installed interactive chat session is open")
        .wait_for_screen("No messages yet.", DEFAULT_TIMEOUT)
        .await
        .unwrap_or_else(|e| panic!("installed interactive chat did not become ready: {e}"));
}

#[then("the installed interactive chat exits successfully")]
async fn then_installed_chat_exited(world: &mut E2eWorld) {
    assert!(
        world.tui.as_ref().is_some_and(TuiSession::is_finished),
        "installed interactive chat was not reaped after a clean exit"
    );
}

#[then("the installed command exits successfully")]
async fn then_installed_command_ok(world: &mut E2eWorld) {
    let st = state(world);
    assert_eq!(
        st.last_rc, 0,
        "installed command exited with {}:\n{}",
        st.last_rc, st.last_output
    );
}

#[then("the install directory is on the user PATH")]
async fn then_install_dir_on_user_path(world: &mut E2eWorld) {
    #[cfg(windows)]
    {
        let install_dir = state(world).install_dir.clone();
        let current = capture_user_path();
        let target = install_dir.to_string_lossy().to_lowercase();
        let on_path = current.split(';').any(|entry| {
            entry.trim().to_lowercase().trim_end_matches('\\') == target.trim_end_matches('\\')
        });
        assert!(on_path, "install dir not on user PATH: {current}");
    }
    #[cfg(not(windows))]
    {
        let _ = world;
    }
}

#[then("the installed rocm and rocmd binaries are present")]
async fn then_binaries_present(world: &mut E2eWorld) {
    let st = state(world);
    assert_exists(&installed_binary(&st.install_dir, "rocm"));
    assert_exists(&installed_binary(&st.install_dir, "rocmd"));
}

#[then("the installed rocm binary is present")]
async fn then_rocm_present(world: &mut E2eWorld) {
    let st = state(world);
    assert_exists(&installed_binary(&st.install_dir, "rocm"));
}

#[then("the install manifest is present")]
async fn then_manifest_present(world: &mut E2eWorld) {
    let st = state(world);
    assert_exists(&st.install_dir.join(".rocm-cli-manifest"));
}

#[then("a minimal config is seeded with the lemonade default engine")]
async fn then_config_seeded(world: &mut E2eWorld) {
    let config = installed_config_file(world);
    let contents = std::fs::read_to_string(&config)
        .unwrap_or_else(|_| panic!("installed config not found: {}", config.display()));
    assert!(
        contents.contains("\"default_engine\"") && contents.contains("\"lemonade\""),
        "config did not seed the lemonade default engine:\n{contents}"
    );
}

#[then("the install fails reporting signature verification failed")]
async fn then_fail_signature(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("signature verification failed"),
        "expected signature verification failure:\n{out}"
    );
}

#[then("the install fails reporting checksum verification failed")]
async fn then_fail_checksum(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("checksum verification failed"),
        "expected checksum verification failure:\n{out}"
    );
}

#[then("the install fails reporting the required signature is missing")]
async fn then_fail_missing_signature(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("required signature sidecar is missing or unavailable"),
        "expected missing-signature failure:\n{out}"
    );
}

#[then("no binaries are activated in the target directory")]
async fn then_no_binaries(world: &mut E2eWorld) {
    let st = state(world);
    assert_missing(&installed_binary(&st.install_dir, "rocm"));
    assert_missing(&st.install_dir.join(".rocm-cli-manifest"));
}

#[then("the installer reports removing the previous install")]
async fn then_removing_previous(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("removing previous rocm-cli install"),
        "installer did not report removing the previous install:\n{out}"
    );
}

#[then("the stale engine entry is gone")]
async fn then_stale_gone(world: &mut E2eWorld) {
    let st = state(world);
    assert_missing(&installed_binary(&st.install_dir, "rocm-engine-stale"));
}

#[then("the preserved config still selects the vllm default engine")]
async fn then_config_preserved(world: &mut E2eWorld) {
    let config = installed_config_file(world);
    let contents = std::fs::read_to_string(&config).expect("installed config not found");
    assert!(
        contents.contains("\"vllm\""),
        "installer overwrote the preserved config:\n{contents}"
    );
}

#[then("the shell profile has exactly one rocm-cli PATH marker")]
async fn then_single_path_marker(world: &mut E2eWorld) {
    let profile = state(world).home_dir.join(".bashrc");
    let contents = std::fs::read_to_string(&profile).unwrap_or_default();
    let count = contents.matches("# >>> rocm-cli path >>>").count();
    assert_eq!(
        count, 1,
        "expected exactly one rocm-cli PATH marker, found {count}:\n{contents}"
    );
}

#[then("examine reads only the isolated config, data, and cache directories")]
async fn then_examine_isolated(world: &mut E2eWorld) {
    let st = state(world);
    let out = &st.last_output;
    for (label, dir) in [
        ("config", &st.smoke_config),
        ("data", &st.smoke_data),
        ("cache", &st.smoke_cache),
    ] {
        let dir = dir.as_ref().expect("isolated smoke dir not seeded");
        let needle = dir.to_string_lossy();
        assert!(
            out.contains(needle.as_ref()),
            "examine did not reference the isolated {label} dir {needle}:\n{out}"
        );
    }
    // The negative half is what actually proves isolation: examine must not read
    // the real user's rocm state. Mirrors the old acceptance script's check.
    if let Some(real) = real_user_rocm_dir() {
        let real = real.to_string_lossy();
        assert!(
            !out.contains(real.as_ref()),
            "examine referenced the real user rocm dir {real}:\n{out}"
        );
    }
}

#[then("uninstall reports skipping the running executable on Windows")]
async fn then_skip_running_exe(world: &mut E2eWorld) {
    let out = &state(world).last_output;
    assert!(
        out.contains("skipping running executable on Windows"),
        "uninstall did not report the running-executable skip:\n{out}"
    );
}

#[then("the installed rocm and rocmd binaries are gone")]
async fn then_binaries_gone(world: &mut E2eWorld) {
    let st = state(world);
    assert_missing(&installed_binary(&st.install_dir, "rocm"));
    assert_missing(&installed_binary(&st.install_dir, "rocmd"));
}

#[then("the install manifest is gone")]
async fn then_manifest_gone(world: &mut E2eWorld) {
    let st = state(world);
    assert_missing(&st.install_dir.join(".rocm-cli-manifest"));
}

#[then("the non-running installed rocmd binary is gone")]
async fn then_rocmd_gone(world: &mut E2eWorld) {
    // The keep-config uninstall skips the running rocm executable on Windows but
    // must still remove the non-running rocmd binary (old acceptance-script check).
    let st = state(world);
    assert_missing(&installed_binary(&st.install_dir, "rocmd"));
}

#[then("the isolated XDG config, data, and cache state is gone")]
async fn then_xdg_gone(world: &mut E2eWorld) {
    let st = state(world);
    for dir in [&st.smoke_config, &st.smoke_data, &st.smoke_cache]
        .into_iter()
        .flatten()
    {
        // Uninstall purges the CLI's own state; the top-level isolated dir may
        // remain but its rocm-cli subtree must be gone.
        let rocm_cli = dir.join("rocm-cli");
        assert_missing(&rocm_cli);
    }
}

// ── Loopback HTTP server (Windows HTTP install) ─────────────────────────

pub mod http {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;

    /// A minimal loopback HTTP file server serving one directory, for the Windows
    /// native-HTTP install scenario. Shuts down on drop.
    pub struct LoopbackServer {
        port: u16,
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl LoopbackServer {
        pub fn start(root: &Path) -> Self {
            let listener =
                TcpListener::bind("127.0.0.1:0").expect("failed to bind loopback listener");
            listener
                .set_nonblocking(true)
                .expect("failed to set non-blocking");
            let port = listener.local_addr().expect("no local addr").port();
            let stop = Arc::new(AtomicBool::new(false));
            let root = root.to_path_buf();
            let stop_thread = Arc::clone(&stop);
            let handle = std::thread::spawn(move || serve(&listener, &root, &stop_thread));
            Self {
                port,
                stop,
                handle: Some(handle),
            }
        }

        pub fn base_url(&self) -> String {
            format!("http://127.0.0.1:{}", self.port)
        }
    }

    impl Drop for LoopbackServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn serve(listener: &TcpListener, root: &Path, stop: &Arc<AtomicBool>) {
        while !stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let _ = handle_conn(stream, root);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Err(_) => break,
            }
        }
    }

    fn handle_conn(mut stream: TcpStream, root: &Path) -> std::io::Result<()> {
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf)?;
        let request = String::from_utf8_lossy(&buf[..n]);
        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");
        let relative = path.trim_start_matches('/');
        let file = safe_join(root, relative);
        match file.and_then(|f| std::fs::read(&f).ok()) {
            Some(bytes) => {
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                );
                stream.write_all(header.as_bytes())?;
                stream.write_all(&bytes)?;
            }
            None => {
                stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )?;
            }
        }
        stream.flush()
    }

    /// Join a request path under `root`, rejecting any traversal so a malformed
    /// request cannot read outside the served directory.
    fn safe_join(root: &Path, relative: &str) -> Option<PathBuf> {
        if relative.is_empty() {
            return None;
        }
        let candidate = root.join(relative);
        let root = root.canonicalize().ok()?;
        let candidate = candidate.canonicalize().ok()?;
        candidate.starts_with(&root).then_some(candidate)
    }
}
