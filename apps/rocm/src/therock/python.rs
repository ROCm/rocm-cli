// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Managed Python for TheRock installs: finding an interpreter that can serve
//! the install (an explicit `ROCM_CLI_PYTHON`, one on `PATH`, or the managed one
//! `uv` provisions), the interpreter the layout requires (`python_requirement`),
//! the wheel tags that interpreter reports, and creating the runtime's virtual
//! environment.
//!
//! Builds on `io`, `process` and `source` (`python_requirement` takes a
//! `SourceLayout`); nothing in those depends on this module. The one item it
//! reads from `therock.rs` is `venv_python_path`, which stays there for good
//! with the other keys and roots helpers.

use super::io::{linux_temp_dir, windows_temp_dir};
use super::process::{
    capture_python_stdout, progress_line, progress_output_suppressed, run_command,
    run_command_with_env,
};
use super::source::SourceLayout;
use super::venv_python_path;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, ManagedToolConfig, RocmCliConfig, ensure_uv_binary, managed_tools_dir,
    normalize_runtime_path_for_host, runtime_is_windows, runtime_os_name, runtime_path_list_split,
    unix_time_millis, uv_command_env, uv_venv_args,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_MANAGED_PYTHON_VERSION: &str = "3.12";

/// Interpreter `uv` provisions for a [`SourceLayout::Next`] install. See
/// [`python_requirement`] for why this differs from the canonical default.
const NEXT_MANAGED_PYTHON_VERSION: &str = "3.14";

#[derive(Debug, Clone)]
pub(super) struct WheelCompatibility {
    pub(super) python_tag: String,
    pub(super) platform_tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct PythonLauncher {
    pub(super) executable: PathBuf,
    pub(super) source: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedPythonManifest {
    executable: PathBuf,
    version: String,
    installed_at_unix_ms: u128,
}

pub(super) fn wheel_compatibility_for_python(
    python_executable: &Path,
) -> Result<WheelCompatibility> {
    let python_tag = capture_python_stdout(
        python_executable,
        "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
        "inspect Python wheel tag",
    )
    .with_context(|| {
        format!(
            "failed to inspect Python wheel tag via {}",
            python_executable.display()
        )
    })?;
    let python_tag = python_tag.trim().to_owned();
    if python_tag.is_empty() {
        bail!("Python did not report a wheel tag");
    }
    Ok(WheelCompatibility {
        python_tag,
        platform_tags: current_platform_wheel_tags()?,
    })
}

fn current_platform_wheel_tags() -> Result<Vec<String>> {
    let platform_tag = match (runtime_os_name(), std::env::consts::ARCH) {
        ("windows", "x86_64") => "win_amd64",
        ("linux", "x86_64") => "linux_x86_64",
        ("linux", "aarch64") => "linux_aarch64",
        (os, arch) => bail!("TheRock wheel filtering is not implemented for {os}/{arch}"),
    };
    Ok(vec![platform_tag.to_owned(), "any".to_owned()])
}

pub(super) fn ensure_uv_venv(
    paths: &AppPaths,
    uv: &Path,
    python_launcher: &Path,
    install_root: &Path,
) -> Result<()> {
    let env_python = venv_python_path(install_root);
    if env_python.is_file() {
        let reusable = run_command(
            &env_python,
            &["--version"],
            "verify existing managed TheRock runtime Python",
        )
        .is_ok()
            && wheel_compatibility_for_python(&env_python).is_ok_and(|existing| {
                wheel_compatibility_for_python(python_launcher)
                    .is_ok_and(|required| existing.python_tag == required.python_tag)
            });
        if reusable {
            return Ok(());
        }
        // Same `install_root` can be reused across reinstalls (it's keyed on
        // resolved version/build date, not on the interpreter), so an existing
        // venv left over from a lower-tag requirement must not be mistaken for
        // a compatible one just because it still runs.
        progress_line(
            "Existing Python environment does not match the required interpreter; recreating it.",
        );
        fs::remove_dir_all(install_root).with_context(|| {
            format!(
                "failed to remove incompatible Python environment at {}",
                install_root.display()
            )
        })?;
    }
    let args = uv_venv_args(python_launcher, install_root);
    run_command_with_env(
        uv,
        args.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        &uv_command_env(paths),
        "create managed TheRock runtime virtual environment",
    )?;
    if !env_python.is_file() {
        bail!(
            "managed Python environment did not create expected executable: {}",
            env_python.display()
        );
    }
    Ok(())
}

fn python_venv_args(install_root: &Path) -> Vec<String> {
    vec![
        "-m".to_owned(),
        "venv".to_owned(),
        install_root.to_string_lossy().to_string(),
    ]
}

fn managed_tools_root(paths: &AppPaths) -> PathBuf {
    managed_tools_dir(&paths.data_dir)
}

fn managed_python_manifest_path(paths: &AppPaths) -> PathBuf {
    managed_tools_root(paths)
        .join("registry")
        .join("python.json")
}

fn load_managed_python_manifest(paths: &AppPaths) -> Result<Option<ManagedPythonManifest>> {
    let path = managed_python_manifest_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn save_managed_python_manifest(paths: &AppPaths, manifest: &ManagedPythonManifest) -> Result<()> {
    let path = managed_python_manifest_path(paths);
    let parent = path
        .parent()
        .context("managed Python manifest path has no parent directory")?;
    fs::create_dir_all(parent)?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(manifest)
            .context("failed to serialize managed Python manifest")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn record_managed_python_config(paths: &AppPaths, python: &Path) -> Result<()> {
    let mut config = RocmCliConfig::load(paths).unwrap_or_default();
    config.tools.insert(
        "python".to_owned(),
        ManagedToolConfig {
            path: Some(python.to_path_buf()),
            managed: true,
        },
    );
    config.save(paths)
}

fn managed_python_bootstrap_disabled() -> bool {
    std::env::var("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP")
        .ok()
        .is_some_and(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
}

/// The interpreter a runtime's wheels require.
///
/// ROCm >= [`THEROCK_NEXT_MIN_MAJOR`] publishes its framework wheels (notably
/// vLLM, flash-attn and amd-aiter) for one CPython version only, and it is not
/// the one the canonical stream uses. Both fields move together, so they travel
/// together: `version` is what `uv` provisions, `tag` is what an already-present
/// interpreter is checked against.
///
/// [`THEROCK_NEXT_MIN_MAJOR`]: super::source::THEROCK_NEXT_MIN_MAJOR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PythonRequirement {
    tag: &'static str,
    version: &'static str,
}

/// The interpreter required to install from `layout`.
///
/// Keyed on the layout rather than on a resolved ROCm version because the
/// interpreter has to be chosen *before* resolution: `resolve_pip_runtime` picks
/// wheels using the interpreter's own tags, so the resolved version is not
/// available yet. The layout is an honest stand-in only when it was decided
/// from an explicit stable pin (`next_layout_requested`); a fresh install with
/// no pin, or a nightly build-date/prerelease pin, needs
/// [`canonical_auto_select_needs_next_python`] to check whether the channel's
/// own "latest" has rotated past [`THEROCK_NEXT_MIN_MAJOR`] on its own.
///
/// [`canonical_auto_select_needs_next_python`]: super::canonical_auto_select_needs_next_python
/// [`THEROCK_NEXT_MIN_MAJOR`]: super::source::THEROCK_NEXT_MIN_MAJOR
pub(super) const fn python_requirement(layout: SourceLayout) -> PythonRequirement {
    match layout {
        SourceLayout::Next => PythonRequirement {
            tag: "cp314",
            version: NEXT_MANAGED_PYTHON_VERSION,
        },
        SourceLayout::Canonical => PythonRequirement {
            tag: "cp312",
            version: DEFAULT_MANAGED_PYTHON_VERSION,
        },
    }
}

pub(super) fn managed_python_version(requirement: PythonRequirement) -> String {
    std::env::var("ROCM_CLI_MANAGED_PYTHON_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| requirement.version.to_owned())
}

fn ensure_managed_python(
    paths: &AppPaths,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    let version = managed_python_version(requirement);
    progress_line(format!("Preparing Python {version}..."));

    let uv = ensure_uv_binary(paths)?;

    // Check the manifest first — if the recorded executable is still usable, skip the install.
    // ponytail: this manifest holds one version at a time, so alternating between a
    // canonical (3.12) and a `Next` (3.14) install re-runs `uv python install` on every
    // switch instead of keeping both cached — `uv` itself still short-circuits the
    // re-download, so this costs a subprocess round trip, not a real reinstall. Upgrade
    // to a version-keyed manifest if that alternation turns out to be a common pattern.
    if let Ok(Some(manifest)) = load_managed_python_manifest(paths)
        && manifest.version == version
        && manifest.executable.is_file()
        && python_launcher_install_ready(&manifest.executable, requirement).is_ok()
    {
        progress_line(format!(
            "Using existing Python {version} at {}.",
            manifest.executable.display()
        ));
        let _ = record_managed_python_config(paths, &manifest.executable);
        return Ok(PythonLauncher {
            executable: manifest.executable,
            source: "managed",
        });
    }

    progress_line(format!("Installing Python {version} via uv..."));
    let install_stdio = || {
        if progress_output_suppressed() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        }
    };
    let install_output = Command::new(&uv)
        .args(["python", "install", &version])
        .envs(uv_command_env(paths))
        .stdin(Stdio::null())
        .stdout(install_stdio())
        .stderr(install_stdio())
        .output()
        .context("failed to launch uv python install")?;
    if !install_output.status.success() {
        let status = install_output.status;
        let stderr = String::from_utf8_lossy(&install_output.stderr);
        let stderr = stderr.trim();
        if stderr.is_empty() {
            bail!("uv python install {version} failed with {status}");
        }
        bail!("uv python install {version} failed with {status}: {stderr}");
    }

    progress_line(format!("Finding Python {version}..."));
    let output = Command::new(&uv)
        .args(["python", "find", &version])
        .envs(uv_command_env(paths))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("failed to launch uv python find")?;
    if !output.status.success() {
        bail!("uv python find {version} failed after install");
    }
    let executable = PathBuf::from(
        String::from_utf8(output.stdout)
            .context("uv python find output was not valid UTF-8")?
            .trim(),
    );
    if !executable.is_file() {
        bail!(
            "uv python find returned a path that does not exist: {}",
            executable.display()
        );
    }

    python_launcher_install_ready(&executable, requirement).with_context(|| {
        format!(
            "Python {version} at {} is not usable for this ROCm install",
            executable.display()
        )
    })?;

    let manifest = ManagedPythonManifest {
        executable: executable.clone(),
        version: version.clone(),
        installed_at_unix_ms: unix_time_millis(),
    };
    save_managed_python_manifest(paths, &manifest)?;
    let _ = record_managed_python_config(paths, &executable);
    progress_line(format!(
        "Python {version} is ready at {}.",
        executable.display()
    ));
    Ok(PythonLauncher {
        executable,
        source: "managed",
    })
}

/// The environment inputs [`resolve_python_launcher`] reads.
///
/// Passed in rather than read at each use site so a caller can point the
/// resolver somewhere else without touching the process environment. Tests need
/// that: `cargo test` runs every test as a thread in one process, so a test that
/// overwrote `PATH` to steer this resolver also hid every other PATH-resolved
/// binary from unrelated tests running at the same moment. In test builds, a
/// test can also install one as a thread-local override (see
/// `OverridePythonResolverEnv`) instead of constructing one to pass directly
/// to [`resolve_python_launcher_in`].
#[cfg_attr(test, derive(Clone))]
pub(super) struct PythonResolverEnv {
    /// `ROCM_CLI_PYTHON`: an explicit interpreter that wins over any search.
    pub(super) python_override: Option<String>,
    /// The directories to search for an interpreter, in `PATH` order.
    pub(super) search_dirs: Vec<PathBuf>,
}

impl PythonResolverEnv {
    fn from_process_env() -> Self {
        Self {
            python_override: std::env::var("ROCM_CLI_PYTHON").ok(),
            search_dirs: std::env::var_os("PATH")
                .map(|value| split_runtime_path(&value))
                .unwrap_or_default(),
        }
    }
}

// Lets a test steer `resolve_python_launcher` at a fixed `PythonResolverEnv`
// instead of the real process environment, scoped to the calling thread only.
//
// Unlike mutating process-wide `PATH`, a thread-local cannot race a sibling
// test thread that spawns a PATH-resolved binary (e.g. bare-name `tar`)
// during the window a test is steering resolution (see issue #650, where
// that mutation intermittently failed unrelated tests under `cargo test`'s
// one-process-many-threads model).
//
// Because the override is thread-local, it only applies when
// `resolve_python_launcher` runs on the thread that installed it. If the code
// under test ever resolves from a spawned thread (or `thread::scope`), that
// thread silently falls back to the real process environment.
#[cfg(test)]
thread_local! {
    static PYTHON_RESOLVER_ENV_OVERRIDE: std::cell::RefCell<Option<PythonResolverEnv>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard installing a fixed [`PythonResolverEnv`] for [`resolve_python_launcher`]
/// on the calling thread only, restoring the prior override on drop (so nested
/// use composes correctly, mirroring [`SuppressProgressOutput`]; the
/// `nested_python_resolver_env_override_restores_the_outer_one_on_drop` test
/// covers the property this claims).
///
/// [`SuppressProgressOutput`]: super::process::SuppressProgressOutput
#[cfg(test)]
pub(super) struct OverridePythonResolverEnv {
    previous: Option<PythonResolverEnv>,
}

#[cfg(test)]
impl OverridePythonResolverEnv {
    pub(super) fn new(env: PythonResolverEnv) -> Self {
        let previous = PYTHON_RESOLVER_ENV_OVERRIDE.with(|cell| cell.replace(Some(env)));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for OverridePythonResolverEnv {
    fn drop(&mut self) {
        let previous = self.previous.take();
        PYTHON_RESOLVER_ENV_OVERRIDE.with(|cell| *cell.borrow_mut() = previous);
    }
}

#[cfg(test)]
fn python_resolver_env_override() -> Option<PythonResolverEnv> {
    PYTHON_RESOLVER_ENV_OVERRIDE.with(|cell| cell.borrow().clone())
}

#[cfg(not(test))]
const fn python_resolver_env_override() -> Option<PythonResolverEnv> {
    None
}

pub(super) fn resolve_python_launcher(
    paths: &AppPaths,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    let env = python_resolver_env_override().unwrap_or_else(PythonResolverEnv::from_process_env);
    resolve_python_launcher_in(paths, &env, requirement)
}

/// The interpreter an install of the wheel runtime goes on with, or, for a dry
/// run, the one it would.
///
/// The single place the install decides between the two, so a dry run cannot
/// reach [`resolve_python_launcher`]'s preparation by a path of its own.
pub(super) fn python_for_install(
    paths: &AppPaths,
    requirement: PythonRequirement,
    dry_run: bool,
) -> Result<PlannedPython> {
    if dry_run {
        plan_python_launcher(paths, requirement)
    } else {
        resolve_python_launcher(paths, requirement).map(PlannedPython::Ready)
    }
}

/// The interpreter a *preview* of the install reports, found without preparing
/// one.
///
/// [`resolve_python_launcher`] ends by downloading a managed `uv`, having it
/// install Python, and saving that Python's path in the configuration when no
/// interpreter on the host will do. A dry run promises to change nothing, so it
/// asks this instead: an interpreter that is usable now comes back as
/// [`PlannedPython::Ready`], exactly as the real install would pick it, and a
/// host that has none comes back as [`PlannedPython::WouldPrepareManaged`], the
/// step a real install would take, left untaken. Every error the real
/// resolution raises before that step (an unusable `ROCM_CLI_PYTHON`, a
/// disabled managed bootstrap) is raised here too, so a preview never succeeds
/// on a host where the install would refuse.
fn plan_python_launcher(paths: &AppPaths, requirement: PythonRequirement) -> Result<PlannedPython> {
    let env = python_resolver_env_override().unwrap_or_else(PythonResolverEnv::from_process_env);
    plan_python_launcher_in(paths, &env, requirement)
}

fn plan_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
) -> Result<PlannedPython> {
    let lookup = find_python_launcher_in(paths, env, requirement, LookupPurpose::Preview)?;
    Ok(match lookup {
        PythonLookup::Found(launcher) => PlannedPython::Ready(launcher),
        PythonLookup::NeedsManagedBootstrap => PlannedPython::WouldPrepareManaged {
            version: managed_python_version(requirement),
            requirement,
        },
    })
}

/// What [`plan_python_launcher`] found.
#[derive(Debug, Clone)]
pub(super) enum PlannedPython {
    /// An interpreter that is usable as it stands.
    Ready(PythonLauncher),
    /// No interpreter is usable. A real install would prepare managed Python
    /// `version`; this preview has not.
    WouldPrepareManaged {
        version: String,
        requirement: PythonRequirement,
    },
}

impl PlannedPython {
    /// The `python_launcher:` line's value: the interpreter's path, or, when
    /// none exists yet, a statement that none was prepared rather than a path
    /// that would be a guess.
    pub(super) fn launcher_display(&self) -> String {
        match self {
            Self::Ready(launcher) => launcher.executable.display().to_string(),
            Self::WouldPrepareManaged { .. } => "not prepared by this preview".to_owned(),
        }
    }

    /// The `python_source:` line's value.
    pub(super) const fn source(&self) -> &'static str {
        match self {
            Self::Ready(launcher) => launcher.source,
            Self::WouldPrepareManaged { .. } => "managed",
        }
    }

    /// What `uv venv --python` is handed in the printed command: the
    /// interpreter's path, or the version that would be prepared, which `uv`
    /// resolves to the managed Python the real install prepares.
    pub(super) fn venv_python_arg(&self) -> PathBuf {
        match self {
            Self::Ready(launcher) => launcher.executable.clone(),
            Self::WouldPrepareManaged { version, .. } => PathBuf::from(version),
        }
    }

    /// The `python_note:` line, present only when something was left undone.
    /// Written from the same value the branch was taken on, so it cannot name a
    /// version the install will not prepare.
    pub(super) fn note(&self) -> Option<String> {
        match self {
            Self::Ready(_) => None,
            Self::WouldPrepareManaged { version, .. } => Some(format!(
                "no usable Python was found; a real install would use uv to install managed Python {version} and save its path in the configuration, which this preview did not do"
            )),
        }
    }

    /// The wheel tags the install will select packages by. An interpreter that
    /// exists is asked; one that does not yet is the one the layout requires,
    /// because a real install refuses to go on with any other (see
    /// `python_launcher_install_ready`).
    pub(super) fn wheel_compatibility(&self) -> Result<WheelCompatibility> {
        match self {
            Self::Ready(launcher) => wheel_compatibility_for_python(&launcher.executable),
            Self::WouldPrepareManaged { requirement, .. } => Ok(WheelCompatibility {
                python_tag: requirement.tag.to_owned(),
                platform_tags: current_platform_wheel_tags()?,
            }),
        }
    }

    /// The interpreter a real install runs with. A preview never reaches here
    /// with an interpreter it has not prepared.
    pub(super) fn into_launcher(self) -> Result<PythonLauncher> {
        match self {
            Self::Ready(launcher) => Ok(launcher),
            Self::WouldPrepareManaged { version, .. } => bail!(
                "internal error: Python {version} was not prepared before the install needed it"
            ),
        }
    }
}

/// Whether a search for an interpreter is for an install or a preview of one.
/// It changes only the wording of the progress line about a saved Python that
/// turned out unusable: a preview must not say it is preparing one.
#[derive(Clone, Copy)]
enum LookupPurpose {
    Install,
    Preview,
}

/// Where the search for an interpreter ended.
enum PythonLookup {
    /// An interpreter that is usable as it stands.
    Found(PythonLauncher),
    /// Nothing is usable and the managed bootstrap is allowed: the next step is
    /// [`ensure_managed_python`], which downloads and writes.
    NeedsManagedBootstrap,
}

fn resolve_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    match find_python_launcher_in(paths, env, requirement, LookupPurpose::Install)? {
        PythonLookup::Found(launcher) => Ok(launcher),
        PythonLookup::NeedsManagedBootstrap => ensure_managed_python(paths, requirement),
    }
}

/// Everything [`resolve_python_launcher_in`] does short of preparing a managed
/// Python: it reads the host and the saved manifest and writes nothing.
fn find_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
    purpose: LookupPurpose,
) -> Result<PythonLookup> {
    if let Some(value) = env.python_override.as_deref() {
        python_launcher_install_ready(Path::new(value), requirement)
            .with_context(|| format!("ROCM_CLI_PYTHON is not usable for ROCm setup: {value}"))?;
        return Ok(PythonLookup::Found(PythonLauncher {
            executable: PathBuf::from(value),
            source: "env",
        }));
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates(&env.search_dirs) {
        match python_launcher_install_ready(&candidate, requirement) {
            Ok(()) => {
                return Ok(PythonLookup::Found(PythonLauncher {
                    executable: candidate,
                    source: "path",
                }));
            }
            Err(_) => {
                skipped_path_python = true;
            }
        }
    }
    if skipped_path_python {
        progress_line(format!(
            "Python from PATH is not usable for this ROCm install ({} is required); {} ROCm CLI's managed Python.",
            requirement.tag,
            match purpose {
                LookupPurpose::Install => "using",
                LookupPurpose::Preview => "a real install would use",
            }
        ));
    }

    if let Some(manifest) = load_managed_python_manifest(paths)?
        && manifest.executable.is_file()
    {
        if python_launcher_install_ready(&manifest.executable, requirement).is_ok() {
            return Ok(PythonLookup::Found(PythonLauncher {
                executable: manifest.executable,
                source: "managed",
            }));
        }
        progress_line(format!(
            "Saved managed Python is not usable for this ROCm install ({} is required); {}.",
            requirement.tag,
            match purpose {
                LookupPurpose::Install => "preparing Python again",
                LookupPurpose::Preview => "a real install would prepare Python again",
            }
        ));
    }

    if managed_python_bootstrap_disabled() {
        bail!(
            "unable to locate Python, and managed Python bootstrap is disabled by ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP"
        );
    }
    Ok(PythonLookup::NeedsManagedBootstrap)
}

fn python_path_candidates(search_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let program_names: &[&str] = if runtime_is_windows() {
        &["python", "python3", "py"]
    } else {
        &["python3", "python"]
    };
    program_names
        .iter()
        .flat_map(|program| resolve_program_on_path(program, search_dirs))
        .collect()
}

fn resolve_program_on_path(program: &str, search_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let candidates = program_path_candidates(program);
    search_dirs
        .iter()
        .flat_map(|dir| candidates.iter().map(move |candidate| dir.join(candidate)))
        .filter(|path| path.is_file())
        .map(|path| normalize_runtime_path_for_host(&path))
        .collect()
}

pub(super) fn split_runtime_path(value: &std::ffi::OsStr) -> Vec<PathBuf> {
    runtime_path_list_split(value)
}

fn program_path_candidates(program: &str) -> Vec<String> {
    let path = Path::new(program);
    if !runtime_is_windows() || path.extension().is_some() {
        return vec![program.to_owned()];
    }
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
    let mut names = vec![program.to_owned()];
    for ext in pathext
        .split(';')
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
    {
        names.push(format!("{program}{ext}"));
        names.push(format!("{program}{}", ext.to_ascii_lowercase()));
    }
    names.sort();
    names.dedup();
    names
}

pub(super) fn python_launcher_install_ready(
    program: &Path,
    requirement: PythonRequirement,
) -> Result<()> {
    let compatibility = wheel_compatibility_for_python(program)?;
    if compatibility.python_tag != requirement.tag {
        bail!(
            "Python wheel tag {} is not supported by this ROCm install; {} is required",
            compatibility.python_tag,
            requirement.tag
        );
    }
    verify_python_can_create_venv(program).context("Python could not create a virtual environment")
}

fn verify_python_can_create_venv(program: &Path) -> Result<()> {
    let probe_root = python_venv_probe_temp_root()?;
    let probe_dir = probe_root.join("env");
    let args = python_venv_args(&probe_dir);
    let result = run_command(
        program,
        args.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        "probe Python virtual environment support",
    );
    let _ = fs::remove_dir_all(&probe_root);
    result
}

fn python_venv_probe_temp_root() -> Result<PathBuf> {
    if runtime_is_windows() {
        windows_temp_dir("rocm-cli-python-venv-probe")
    } else {
        linux_temp_dir("rocm-cli-python-venv-probe")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::test_paths;

    #[cfg(unix)]
    fn write_fake_python_with_venv(dir: &Path, name: &str) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join(name);
        let script = r#"#!/bin/sh
if [ "$1" = "-c" ]; then
  echo cp312
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  /bin/mkdir -p "$3/bin"
  /bin/cat > "$3/bin/python" <<'PY'
#!/bin/sh
echo Python 3.12.10
PY
  /bin/chmod +x "$3/bin/python"
  exit 0
fi
echo Python 3.12.10
"#;
        write_fake_executable(&path, script)?;
        Ok(path)
    }

    #[cfg(unix)]
    fn write_fake_python_reporting(
        dir: &Path,
        name: &str,
        tag: &str,
        version: &str,
    ) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join(name);
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then\n  echo {tag}\n  exit 0\nfi\necho {version}\n"
        );
        write_fake_executable(&path, script)?;
        Ok(path)
    }

    #[cfg(unix)]
    fn write_fake_uv_creating_venv(
        dir: &Path,
        name: &str,
        tag: &str,
        version: &str,
    ) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join(name);
        let script = format!(
            r#"#!/bin/sh
if [ "$1" = "venv" ] && [ "$2" = "--python" ]; then
  /bin/mkdir -p "$4/bin"
  /bin/cat > "$4/bin/python" <<PY
#!/bin/sh
if [ "\$1" = "-c" ]; then
  echo {tag}
  exit 0
fi
echo {version}
PY
  /bin/chmod +x "$4/bin/python"
  exit 0
fi
exit 1
"#
        );
        write_fake_executable(&path, script)?;
        Ok(path)
    }

    #[test]
    fn python_venv_args_use_python_default_linking() {
        let args = python_venv_args(Path::new("/mnt/d/path/to/rocm"));

        assert!(!args.iter().any(|arg| arg == "--copies"));
        assert_eq!(args.last().map(String::as_str), Some("/mnt/d/path/to/rocm"));
    }

    #[test]
    fn python_venv_args_target_install_root() {
        let args = python_venv_args(Path::new("/mnt/envs/my-env"));
        assert_eq!(args, vec!["-m", "venv", "/mnt/envs/my-env"]);
    }

    #[test]
    fn managed_python_defaults_to_312() {
        assert_eq!(DEFAULT_MANAGED_PYTHON_VERSION, "3.12");
    }

    /// ROCm 10.x framework wheels (vLLM, flash-attn, amd-aiter) are published
    /// cp314-only, so the next layout must provision 3.14 while everything else
    /// stays on 3.12.
    #[test]
    fn python_requirement_follows_the_source_layout() {
        let canonical = python_requirement(SourceLayout::Canonical);
        assert_eq!(canonical.tag, "cp312");
        assert_eq!(canonical.version, "3.12");

        let next = python_requirement(SourceLayout::Next);
        assert_eq!(next.tag, "cp314");
        assert_eq!(next.version, "3.14");
    }

    /// A tag mismatch and a broken venv are different failures with different
    /// fixes, so the gate must not report one as the other.
    #[cfg(unix)]
    #[test]
    fn python_gate_rejects_the_wrong_tag_without_blaming_the_venv() -> Result<()> {
        let (root, _paths) = test_paths("python-gate-tag-mismatch");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        // A real, working cp312 interpreter: the venv probe would succeed.
        let python = write_fake_python_with_venv(&bin_dir, "python")?;

        python_launcher_install_ready(&python, python_requirement(SourceLayout::Canonical))
            .expect("a cp312 python satisfies a canonical install");

        let err = python_launcher_install_ready(&python, python_requirement(SourceLayout::Next))
            .expect_err("a cp312 python cannot serve a next-layout install");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("cp314"), "{rendered}");
        assert!(
            !rendered.contains("virtual environment"),
            "a tag mismatch must not be reported as a venv failure: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn managed_python_manifest_round_trips() -> Result<()> {
        let (root, paths) = test_paths("managed-python-manifest");
        let manifest = ManagedPythonManifest {
            executable: paths
                .data_dir
                .join("tools")
                .join("python")
                .join("python.exe"),
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };

        save_managed_python_manifest(&paths, &manifest)?;
        let loaded = load_managed_python_manifest(&paths)?.expect("manifest should load");

        fs::remove_dir_all(root).ok();
        assert_eq!(loaded.executable, manifest.executable);
        assert_eq!(loaded.version, "3.12");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn python_launcher_prefers_path_python_before_saved_managed_python() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            // No wheel platform tag for this host (e.g. macOS): every python fails
            // the wheel-compatibility check, so resolution always falls through to
            // the managed/uv path regardless of PATH. Nothing to assert here.
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-prefers-path");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        let path_python = write_fake_python_with_venv(&bin_dir, "python")?;
        let managed_python = paths.data_dir.join("tools").join("python").join("python");
        fs::create_dir_all(managed_python.parent().expect("managed python parent"))?;
        fs::write(&managed_python, "not used")?;
        let manifest = ManagedPythonManifest {
            executable: managed_python,
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };
        save_managed_python_manifest(&paths, &manifest)?;
        // Keep the search hermetic: including the real PATH lets a genuine cp312
        // python (present on CI) win over the fake one and breaks the
        // executable assertion. The fake alone is all this test needs.
        let launcher = resolve_python_launcher_in(
            &paths,
            &PythonResolverEnv {
                python_override: None,
                search_dirs: vec![bin_dir],
            },
            python_requirement(SourceLayout::Canonical),
        )?;
        assert_eq!(launcher.source, "path");
        assert!(
            launcher.executable.is_absolute(),
            "PATH launcher should resolve to an absolute executable: {}",
            launcher.executable.display()
        );
        let launcher_path = launcher
            .executable
            .to_string_lossy()
            .replace('\\', "/")
            .to_ascii_lowercase();
        let expected_path = path_python
            .to_string_lossy()
            .replace('\\', "/")
            .to_ascii_lowercase();
        assert_eq!(launcher_path, expected_path);
        assert!(path_python.exists());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn python_venv_probe_temp_root_uses_windows_temp_env() -> Result<()> {
        if !runtime_is_windows() {
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, _paths) = test_paths("python-probe-temp-root");
        let temp_root = root.join("Temp");
        fs::create_dir_all(&temp_root)?;
        let old_temp = std::env::var_os("TEMP");
        let old_tmp = std::env::var_os("TMP");
        let old_localappdata = std::env::var_os("LOCALAPPDATA");
        unsafe {
            std::env::set_var("TEMP", &temp_root);
            std::env::remove_var("TMP");
            std::env::remove_var("LOCALAPPDATA");
        }
        let probe_root = python_venv_probe_temp_root();
        unsafe {
            match old_temp {
                Some(value) => std::env::set_var("TEMP", value),
                None => std::env::remove_var("TEMP"),
            }
            match old_tmp {
                Some(value) => std::env::set_var("TMP", value),
                None => std::env::remove_var("TMP"),
            }
            match old_localappdata {
                Some(value) => std::env::set_var("LOCALAPPDATA", value),
                None => std::env::remove_var("LOCALAPPDATA"),
            }
        }
        let probe_root = probe_root?;
        assert!(
            probe_root.starts_with(&temp_root),
            "probe root should stay under TEMP: {} not under {}",
            probe_root.display(),
            temp_root.display()
        );
        assert!(
            !probe_root.to_string_lossy().starts_with("/tmp/"),
            "Windows probe root must not use Unix /tmp: {}",
            probe_root.display()
        );
        fs::remove_dir_all(&probe_root).ok();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn python_launcher_prefers_path_python_over_managed_when_venv_capable() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            // No wheel platform tag for this host (e.g. macOS): every python fails
            // the wheel-compatibility check, so resolution always falls through to
            // the managed/uv path regardless of PATH. Nothing to assert here.
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-path-over-managed");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        let path_python = write_fake_python_with_venv(&bin_dir, "python3")?;
        let managed_dir = paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let managed_python = write_fake_python_with_venv(&managed_dir, "python")?;
        let manifest = ManagedPythonManifest {
            executable: managed_python,
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };
        save_managed_python_manifest(&paths, &manifest)?;
        let launcher = resolve_python_launcher_in(
            &paths,
            &PythonResolverEnv {
                python_override: None,
                search_dirs: vec![bin_dir],
            },
            python_requirement(SourceLayout::Canonical),
        )?;

        assert_eq!(launcher.source, "path");
        assert!(path_python.exists());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// The interpreter search must look only where it is told to look.
    ///
    /// This is the property that keeps the resolver out of the process
    /// environment. While it read `PATH` itself, the only way to steer it was to
    /// overwrite `PATH` for the whole process — which, under `cargo test`, also
    /// hid `tar` and every other PATH-resolved binary from the unrelated tests
    /// sharing that process.
    #[cfg(unix)]
    #[test]
    fn python_path_search_only_uses_the_given_directories() -> Result<()> {
        let (root, _paths) = test_paths("python-search-scope");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        write_fake_python_with_venv(&bin_dir, "python3")?;

        assert!(
            python_path_candidates(&[]).is_empty(),
            "an empty search list must yield no candidates even though the real PATH has a python"
        );
        let candidates = python_path_candidates(std::slice::from_ref(&bin_dir));
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.starts_with(&bin_dir)),
            "the search must stay inside the given directories: {candidates:?}"
        );
        assert_eq!(candidates.len(), 1, "expected exactly the fixture python");

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// `resolve_python_launcher` must consult an installed
    /// `OverridePythonResolverEnv` instead of the real process environment.
    /// That is what lets a test steer resolution without overwriting
    /// process-wide `PATH`, which raced sibling test threads spawning
    /// PATH-resolved binaries (issue #650).
    #[cfg(unix)]
    #[test]
    fn python_resolver_env_override_is_consulted_by_resolve_python_launcher() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            // No wheel platform tag for this host (e.g. macOS): every python fails
            // the wheel-compatibility check, so resolution always falls through to
            // the managed/uv path regardless of PATH. Nothing to assert here.
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-resolver-env-override-hermetic");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        write_fake_python_with_venv(&bin_dir, "python3")?;

        let launcher = {
            let _guard = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: vec![bin_dir.clone()],
            });
            resolve_python_launcher(&paths, python_requirement(SourceLayout::Canonical))?
        };

        assert_eq!(
            launcher.source, "path",
            "resolve_python_launcher must consult the test override, not the real environment"
        );
        assert!(launcher.executable.starts_with(&bin_dir));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn nested_python_resolver_env_override_restores_the_outer_one_on_drop() {
        // Mirrors the bookkeeping check `suppress_progress_output_contract` (in
        // `therock::process`) makes for `SuppressProgressOutput`: proves the
        // same restore-previous composition for `OverridePythonResolverEnv`,
        // which its doc comment claims but no other test exercises.
        // Thread-local, so this doesn't race with other tests' threads.
        assert!(python_resolver_env_override().is_none());
        {
            let _outer = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: Some("outer".to_owned()),
                search_dirs: vec![],
            });
            assert_eq!(
                python_resolver_env_override()
                    .unwrap()
                    .python_override
                    .as_deref(),
                Some("outer")
            );
            {
                let _inner = OverridePythonResolverEnv::new(PythonResolverEnv {
                    python_override: Some("inner".to_owned()),
                    search_dirs: vec![],
                });
                assert_eq!(
                    python_resolver_env_override()
                        .unwrap()
                        .python_override
                        .as_deref(),
                    Some("inner")
                );
            }
            assert_eq!(
                python_resolver_env_override()
                    .unwrap()
                    .python_override
                    .as_deref(),
                Some("outer"),
                "dropping the inner override must restore the outer one, not clear it"
            );
        }
        assert!(
            python_resolver_env_override().is_none(),
            "dropping the outer override must restore the pre-override state"
        );
    }

    // ---- a dry run prepares no Python ----------------------------------------

    /// Process environment variables set for one test and restored on drop.
    /// Every caller holds `PROCESS_ENV_TEST_LOCK` for as long as it holds this.
    #[cfg(unix)]
    struct RestoredEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

    #[cfg(unix)]
    impl RestoredEnv {
        #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
        fn set(vars: &[(&'static str, Option<&Path>)]) -> Self {
            let saved = vars
                .iter()
                .map(|(key, _)| (*key, std::env::var_os(key)))
                .collect();
            for (key, value) in vars {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
            Self(saved)
        }
    }

    #[cfg(unix)]
    impl Drop for RestoredEnv {
        #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
        fn drop(&mut self) {
            for (key, value) in self.0.drain(..) {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    /// A `uv` that records its arguments in `marker` and then fails. A host that
    /// reaches it has gone on to prepare a managed Python; failing keeps a test
    /// that does reach it from going on to the network.
    #[cfg(unix)]
    fn write_recording_uv(dir: &Path, marker: &Path) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join("uv");
        write_fake_executable(
            &path,
            format!("#!/bin/sh\necho \"$@\" >> '{}'\nexit 1\n", marker.display()),
        )?;
        Ok(path)
    }

    /// Every path under the three folders a command may write to, sorted. Empty
    /// when none of them exists, so "nothing was created" and "something empty
    /// was created" are different answers.
    #[cfg(unix)]
    fn snapshot(paths: &AppPaths) -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                out.push(entry.path());
                walk(&entry.path(), out);
            }
        }
        let mut out = Vec::new();
        for dir in [&paths.config_dir, &paths.data_dir, &paths.cache_dir] {
            if dir.exists() {
                out.push(dir.clone());
            }
            walk(dir, &mut out);
        }
        out.sort();
        out
    }

    /// A host with no usable Python: an empty search folder and no saved Python,
    /// unless the test adds one. `uv` is a recording stand-in, so a code path that
    /// prepares Python is visible as a marker instead of a download.
    #[cfg(unix)]
    struct PythonHost {
        root: PathBuf,
        paths: AppPaths,
        search_dir: PathBuf,
        uv_marker: PathBuf,
        _env: RestoredEnv,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    #[cfg(unix)]
    impl PythonHost {
        fn new(name: &str) -> Result<Self> {
            let lock = PROCESS_ENV_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let (root, paths) = test_paths(name);
            let search_dir = root.join("search");
            fs::create_dir_all(&search_dir)?;
            let uv_marker = root.join("uv-was-run");
            let tools = root.join("tools");
            fs::create_dir_all(&tools)?;
            let uv = write_recording_uv(&tools, &uv_marker)?;
            let env = RestoredEnv::set(&[
                ("ROCM_CLI_UV_BINARY", Some(&uv)),
                ("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP", None),
                ("ROCM_CLI_MANAGED_PYTHON_VERSION", None),
            ]);
            Ok(Self {
                root,
                paths,
                search_dir,
                uv_marker,
                _env: env,
                _lock: lock,
            })
        }

        /// What an install (`dry_run: false`) or its preview (`true`) goes on
        /// with, searching only this host's folder.
        fn python_for_install(&self, dry_run: bool) -> Result<PlannedPython> {
            let _resolver = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: vec![self.search_dir.clone()],
            });
            python_for_install(
                &self.paths,
                python_requirement(SourceLayout::Canonical),
                dry_run,
            )
        }

        fn uv_was_run(&self) -> bool {
            self.uv_marker.exists()
        }

        fn save_managed_python(&self, executable: PathBuf) -> Result<Vec<u8>> {
            save_managed_python_manifest(
                &self.paths,
                &ManagedPythonManifest {
                    executable,
                    version: DEFAULT_MANAGED_PYTHON_VERSION.to_owned(),
                    installed_at_unix_ms: 123,
                },
            )?;
            Ok(fs::read(managed_python_manifest_path(&self.paths))?)
        }
    }

    #[cfg(unix)]
    impl Drop for PythonHost {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).ok();
        }
    }

    /// The bug: a dry run on a host with no usable Python went on to prepare one.
    /// Preparing means a managed `uv` and Python are downloaded and the Python's
    /// path is saved in the configuration, all before a plan is printed.
    ///
    /// Asserts the preview's answer and the state it leaves together (the message
    /// claims nothing was prepared, so the folders have to agree), and that the
    /// same host DOES reach `uv` when the install is real, so an empty marker
    /// means the preview stayed out of it rather than that the stand-in is not
    /// wired up. The Gherkin scenario `runtime-22` drives the same case through
    /// the built binary.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_with_no_usable_python_prepares_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-none")?;
        assert!(
            snapshot(&host.paths).is_empty(),
            "the fixture must start with the config, data and cache folders absent"
        );

        let planned = host.python_for_install(true)?;

        let PlannedPython::WouldPrepareManaged {
            ref version,
            requirement,
        } = planned
        else {
            panic!("a host with no Python must be planned as needing one, got {planned:?}");
        };
        assert_eq!(version, DEFAULT_MANAGED_PYTHON_VERSION);
        assert_eq!(requirement, python_requirement(SourceLayout::Canonical));
        let note = planned
            .note()
            .expect("a preview that skipped a step says so");
        assert!(
            note.contains("would use uv to install managed Python 3.12")
                && note.ends_with("which this preview did not do"),
            "{note}"
        );
        // What the note says was not done, was not done.
        assert!(!host.uv_was_run(), "the dry run ran uv");
        assert!(
            !host.paths.config_path().exists(),
            "the dry run saved a Python path in the configuration"
        );
        assert!(
            !managed_python_manifest_path(&host.paths).exists(),
            "the dry run saved a managed Python manifest"
        );
        assert_eq!(
            snapshot(&host.paths),
            Vec::<PathBuf>::new(),
            "the dry run created files or folders under the config, data or cache folder"
        );

        // Control: the real install on the same host does reach `uv`.
        assert!(host.python_for_install(false).is_err());
        assert!(
            host.uv_was_run(),
            "the stand-in uv is not reachable, so the checks above prove nothing"
        );
        Ok(())
    }

    /// A Python on the search path is used by the preview exactly as by the
    /// install, and costs nothing: no `uv`, nothing written.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_reports_a_usable_path_python_and_prepares_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-path")?;
        let python = write_fake_python_with_venv(&host.search_dir, "python3")?;

        let planned = host.python_for_install(true)?;

        let PlannedPython::Ready(launcher) = planned else {
            panic!("a usable PATH Python must be reported as ready, got {planned:?}");
        };
        assert_eq!(launcher.source, "path");
        assert_eq!(launcher.executable, python);
        assert!(!host.uv_was_run());
        assert_eq!(snapshot(&host.paths), Vec::<PathBuf>::new());
        Ok(())
    }

    /// A saved managed Python that still works is reported, and left as it was.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_reports_a_usable_saved_managed_python_and_changes_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-saved")?;
        let managed_dir = host.paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let managed = write_fake_python_with_venv(&managed_dir, "python")?;
        let manifest = host.save_managed_python(managed.clone())?;
        let before = snapshot(&host.paths);

        let planned = host.python_for_install(true)?;

        let PlannedPython::Ready(launcher) = planned else {
            panic!("a usable saved Python must be reported as ready, got {planned:?}");
        };
        assert_eq!(launcher.source, "managed");
        assert_eq!(launcher.executable, managed);
        assert!(!host.uv_was_run());
        assert!(
            !host.paths.config_path().exists(),
            "reporting a saved Python must not re-save it in the configuration"
        );
        assert_eq!(
            fs::read(managed_python_manifest_path(&host.paths))?,
            manifest
        );
        assert_eq!(snapshot(&host.paths), before);
        Ok(())
    }

    /// A saved managed Python that no longer works is a host with none: the
    /// preview says one would be prepared, and leaves the stale record alone
    /// rather than clearing or rewriting it.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_with_an_unusable_saved_python_prepares_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-stale")?;
        let managed_dir = host.paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let broken = managed_dir.join("python");
        fs::write(&broken, "not an interpreter")?;
        let manifest = host.save_managed_python(broken)?;
        let before = snapshot(&host.paths);

        let planned = host.python_for_install(true)?;

        assert!(
            matches!(planned, PlannedPython::WouldPrepareManaged { .. }),
            "{planned:?}"
        );
        assert!(!host.uv_was_run());
        assert!(!host.paths.config_path().exists());
        assert_eq!(
            fs::read(managed_python_manifest_path(&host.paths))?,
            manifest
        );
        assert_eq!(snapshot(&host.paths), before);
        Ok(())
    }

    /// Where an install refuses, the preview refuses too, with the same message:
    /// a preview that printed a plan for an install that cannot happen would be
    /// a false promise.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_refuses_where_the_install_does_when_managed_python_is_disabled() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-disabled")?;
        let _disabled = RestoredEnv::set(&[(
            "ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP",
            Some(Path::new("1")),
        )]);

        let preview = host.python_for_install(true).unwrap_err().to_string();
        let install = host.python_for_install(false).unwrap_err().to_string();

        assert!(
            preview.contains("managed Python bootstrap is disabled"),
            "{preview}"
        );
        assert_eq!(preview, install);
        assert!(!host.uv_was_run());
        Ok(())
    }

    /// An interpreter that has not been prepared yet is described by the one the
    /// layout requires: that is what a real install insists on (see
    /// `python_launcher_install_ready`), so it is what wheels must be chosen for.
    #[test]
    fn an_unprepared_python_is_described_by_the_interpreter_the_layout_requires() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        for (layout, tag, version) in [
            (SourceLayout::Canonical, "cp312", "3.12"),
            (SourceLayout::Next, "cp314", "3.14"),
        ] {
            let requirement = python_requirement(layout);
            let planned = PlannedPython::WouldPrepareManaged {
                version: requirement.version.to_owned(),
                requirement,
            };

            let compatibility = planned.wheel_compatibility()?;
            assert_eq!(compatibility.python_tag, tag);
            assert!(compatibility.platform_tags.contains(&"any".to_owned()));
            assert_eq!(planned.source(), "managed");
            assert_eq!(planned.launcher_display(), "not prepared by this preview");
            assert_eq!(planned.venv_python_arg(), PathBuf::from(version));
            assert!(
                planned
                    .note()
                    .is_some_and(|note| note.contains(&format!("managed Python {version}"))),
                "{planned:?}"
            );
            assert!(
                planned.into_launcher().is_err(),
                "an install must never proceed on a Python that was only planned"
            );
        }
        Ok(())
    }

    /// A ready interpreter prints as it always did: its path and its source, no
    /// note.
    #[test]
    fn a_ready_python_prints_its_path_and_source() {
        let planned = PlannedPython::Ready(PythonLauncher {
            executable: PathBuf::from("/usr/bin/python3"),
            source: "path",
        });
        assert_eq!(planned.launcher_display(), "/usr/bin/python3");
        assert_eq!(planned.source(), "path");
        assert_eq!(planned.venv_python_arg(), PathBuf::from("/usr/bin/python3"));
        assert!(planned.note().is_none());
        assert_eq!(
            planned.into_launcher().unwrap().executable,
            PathBuf::from("/usr/bin/python3")
        );
    }

    #[test]
    #[cfg(unix)]
    fn ensure_uv_venv_recreates_on_interpreter_tag_mismatch() {
        // `install_root` is keyed on resolved version/build date, not on the
        // interpreter, so a reinstall that now requires cp314 must not reuse a
        // stale cp312 venv left over from before this tag check existed.
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-ensure-uv-venv-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        fs::create_dir_all(root.join("cache")).unwrap();
        let root = root.canonicalize().unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };

        let install_root = root.join("install");
        let stale_bin = install_root.join("bin");
        fs::create_dir_all(&stale_bin).unwrap();
        write_fake_python_reporting(&stale_bin, "python", "cp312", "Python 3.12.10").unwrap();

        let python_launcher =
            write_fake_python_reporting(&root, "python3.14", "cp314", "Python 3.14.5").unwrap();
        let uv = write_fake_uv_creating_venv(&root, "uv", "cp314", "Python 3.14.5").unwrap();

        ensure_uv_venv(&paths, &uv, &python_launcher, &install_root).unwrap();

        let env_python = venv_python_path(&install_root);
        let compat = wheel_compatibility_for_python(&env_python).unwrap();
        assert_eq!(
            compat.python_tag, "cp314",
            "stale cp312 venv must be recreated, not silently reused"
        );

        fs::remove_dir_all(root).ok();
    }
}
