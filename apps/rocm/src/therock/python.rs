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
pub(crate) const NEXT_MANAGED_PYTHON_VERSION: &str = "3.14";

#[derive(Debug, Clone)]
pub(super) struct WheelCompatibility {
    pub(super) python_tag: String,
    pub(super) platform_tags: Vec<String>,
}

/// Where an interpreter came from. `as_str` is what the install plan prints on
/// its `python_source:` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PythonSource {
    /// `ROCM_CLI_PYTHON`.
    Env,
    /// Found on `PATH`.
    Path,
    /// The managed Python ROCm CLI prepares.
    Managed,
    /// The interpreter an installed runtime's manifest recorded.
    Manifest,
}

impl PythonSource {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Path => "path",
            Self::Managed => "managed",
            Self::Manifest => "manifest",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PythonLauncher {
    pub(super) executable: PathBuf,
    pub(super) source: PythonSource,
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

/// `config.json` for the managed-Python step, failing with what was left
/// alone and how to recover (the advice comes from [`RocmCliConfig::load`]).
fn load_config_for_managed_python(paths: &AppPaths) -> Result<RocmCliConfig> {
    RocmCliConfig::load(paths).with_context(|| {
        format!(
            "cannot record the managed Python in {} because that file cannot be read; it was \
             left unchanged",
            paths.config_path().display()
        )
    })
}

/// Record the managed interpreter under `tools.python` in `config.json`.
///
/// A `config.json` that exists but cannot be read is an error here, as it is
/// for [`RocmCliConfig::load`] itself: it is not "no config yet", and saving
/// over it would replace the user's active runtime, TheRock folder, providers
/// and preferences with defaults. The callers fail rather than skip the
/// record: the install cannot finish against that file anyway (activating the
/// new runtime loads it strictly). `ensure_managed_python` checks the file
/// before it downloads anything, so this only fails there when the file
/// changed meanwhile.
fn record_managed_python_config(paths: &AppPaths, python: &Path) -> Result<()> {
    let mut config = load_config_for_managed_python(paths)?;
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
    pub(super) tag: &'static str,
    pub(super) version: &'static str,
}

/// The interpreter required to install from `layout`.
///
/// Keyed on the layout rather than on a resolved ROCm version because the
/// interpreter has to be chosen *before* resolution: `resolve_pip_runtime` picks
/// wheels using the interpreter's own tags, so the resolved version is not
/// available yet. For a request routed to the next layout (`layout_serving`:
/// no pin, a stable ROCm 10+ pin, or a nightly built since the next layout began)
/// the layout is an honest stand-in. A pin routed to the canonical stream can
/// still resolve a ROCm 10+ build there (the earlier nightly stream carries ROCm
/// 10 alphas), so [`canonical_auto_select_needs_next_python`] checks that feed's
/// own listing for a build at or past [`THEROCK_NEXT_MIN_MAJOR`].
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

/// The CPython versions a ROCm 10+ install can run on when nothing needs vLLM:
/// the SDK device wheels are `py3-none` and the torch stack is published for
/// each of these. [`NEXT_MANAGED_PYTHON_VERSION`] stays the default because
/// vLLM, flash-attn and amd-aiter for ROCm 10 exist for it alone.
const NEXT_SDK_PYTHONS: [PythonRequirement; 5] = [
    PythonRequirement {
        tag: "cp310",
        version: "3.10",
    },
    PythonRequirement {
        tag: "cp311",
        version: "3.11",
    },
    PythonRequirement {
        tag: "cp312",
        version: "3.12",
    },
    PythonRequirement {
        tag: "cp313",
        version: "3.13",
    },
    PythonRequirement {
        tag: "cp314",
        version: NEXT_MANAGED_PYTHON_VERSION,
    },
];

/// The interpreter `layout` requires when the user asked for `version`
/// (`rocm install sdk --python-version`).
///
/// Only a [`SourceLayout::Next`] install has a choice; the canonical stream is
/// published for one interpreter, so naming anything else for it is refused
/// instead of ignored.
pub(super) fn python_requirement_for_version(
    layout: SourceLayout,
    version: &str,
) -> Result<PythonRequirement> {
    let version = version.trim();
    match layout {
        SourceLayout::Next => NEXT_SDK_PYTHONS
            .iter()
            .copied()
            .find(|candidate| candidate.version == version)
            .with_context(|| {
                let supported = NEXT_SDK_PYTHONS
                    .iter()
                    .map(|candidate| candidate.version)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "--python-version {version} is not one ROCm 10 and newer is published for; use one of {supported}"
                )
            }),
        SourceLayout::Canonical => {
            let required = python_requirement(layout);
            if version == required.version {
                return Ok(required);
            }
            bail!(
                "--python-version applies only to ROCm 10 and newer installs; this install uses Python {}",
                required.version
            )
        }
    }
}

/// The requirement a runtime installed from `layout` was built for, read from
/// the wheel `tag` of the interpreter it recorded. `None` for a tag the layout
/// is not published for, so the caller falls back to the layout's default.
pub(super) fn python_requirement_for_tag(
    layout: SourceLayout,
    tag: &str,
) -> Option<PythonRequirement> {
    match layout {
        SourceLayout::Next => NEXT_SDK_PYTHONS
            .iter()
            .copied()
            .find(|candidate| candidate.tag == tag),
        SourceLayout::Canonical => None,
    }
}

fn managed_python_version_override() -> Option<String> {
    std::env::var("ROCM_CLI_MANAGED_PYTHON_VERSION").ok()
}

/// The managed Python version a bootstrap prepares under the process
/// environment. A decision made from an injected [`PythonResolverEnv`] asks
/// [`PythonResolverEnv::managed_python_version`] instead.
pub(super) fn managed_python_version(requirement: PythonRequirement) -> String {
    PythonResolverEnv::from_process_env().managed_python_version(requirement)
}

fn ensure_managed_python(
    paths: &AppPaths,
    requirement: PythonRequirement,
    version: String,
) -> Result<PythonLauncher> {
    // The step ends by recording the interpreter in config.json. Check that
    // the file can be read before downloading uv or Python, so an unreadable
    // one stops the step before it fetches anything.
    load_config_for_managed_python(paths)?;
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
        record_managed_python_config(paths, &manifest.executable)?;
        return Ok(PythonLauncher {
            executable: manifest.executable,
            source: PythonSource::Managed,
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
    record_managed_python_config(paths, &executable)?;
    progress_line(format!(
        "Python {version} is ready at {}.",
        executable.display()
    ));
    Ok(PythonLauncher {
        executable,
        source: PythonSource::Managed,
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
    /// `ROCM_CLI_MANAGED_PYTHON_VERSION`: the version a managed bootstrap
    /// prepares in place of the layout's own, when set and not blank.
    pub(super) managed_python_version: Option<String>,
    /// `ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP`: whether a host with no usable
    /// interpreter is refused instead of bootstrapped.
    pub(super) managed_bootstrap_disabled: bool,
}

impl PythonResolverEnv {
    pub(super) fn from_process_env() -> Self {
        Self {
            python_override: std::env::var("ROCM_CLI_PYTHON").ok(),
            search_dirs: std::env::var_os("PATH")
                .map(|value| split_runtime_path(&value))
                .unwrap_or_default(),
            managed_python_version: managed_python_version_override(),
            managed_bootstrap_disabled: managed_python_bootstrap_disabled(),
        }
    }

    /// The managed Python version a bootstrap under this environment prepares.
    fn managed_python_version(&self, requirement: PythonRequirement) -> String {
        self.managed_python_version
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| requirement.version.to_owned())
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
///
/// It also prints the line that says which interpreter was chosen, once the
/// choice is made. A preview's line is built from the same value the decision
/// returned ([`PlannedPython::progress_text`]), so a line that says what a real
/// install would do cannot be printed on a path that ends in a refusal: the
/// refusal returns before any such line exists.
pub(super) fn python_for_install(
    paths: &AppPaths,
    requirement: PythonRequirement,
    dry_run: bool,
) -> Result<PlannedPython> {
    let python = if dry_run {
        plan_python_launcher(paths, requirement)?
    } else {
        PlannedPython::Ready(resolve_python_launcher(paths, requirement)?)
    };
    progress_line(python.progress_text());
    Ok(python)
}

/// The interpreter a *preview* of the install, or an update check, reports,
/// found without preparing one.
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
pub(super) fn plan_python_launcher(
    paths: &AppPaths,
    requirement: PythonRequirement,
) -> Result<PlannedPython> {
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
        PythonLookup::NeedsManagedBootstrap => {
            let version = env.managed_python_version(requirement);
            // `ROCM_CLI_MANAGED_PYTHON_VERSION` can name a version other than the
            // one the wheels need. A real install prepares it, finds it reports
            // the wrong tag and refuses (see `ensure_managed_python`), so the
            // preview refuses by the same rule rather than announce a plan the
            // install would not carry out. A version whose tag cannot be read
            // off its name is announced as unchecked: only installing it would
            // tell.
            let tag = python_tag_for_version(&version);
            if let Some(tag) = tag.as_deref() {
                ensure_python_tag_matches(tag, requirement).with_context(|| {
                    format!(
                        "a real install would prepare managed Python {version}, whose wheel tag it would then refuse"
                    )
                })?;
            }
            PlannedPython::WouldPrepareManaged {
                version,
                requirement,
                tag_check: if tag.is_some() {
                    TagCheck::Checked
                } else {
                    TagCheck::Unchecked
                },
            }
        }
    })
}

/// What [`plan_python_launcher`] found.
#[derive(Debug, Clone)]
pub(super) enum PlannedPython {
    /// An interpreter that is usable as it stands.
    Ready(PythonLauncher),
    /// No interpreter is usable. A real install would prepare managed Python
    /// `version`; this preview has not. `tag_check` says whether the wheel tag
    /// `version` reports could be read off its name and matched against the
    /// requirement; when it could not, only installing it would say.
    WouldPrepareManaged {
        version: String,
        requirement: PythonRequirement,
        tag_check: TagCheck,
    },
}

/// Whether a version a preview would prepare could be matched against the
/// wheel tag the layout requires before it is installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TagCheck {
    /// The tag was read off the version's name and matches the requirement.
    Checked,
    /// The tag cannot be read off the version's name (another implementation, a
    /// range, a wildcard that leaves the minor open): only installing it would
    /// tell.
    Unchecked,
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
    pub(super) const fn source(&self) -> PythonSource {
        match self {
            Self::Ready(launcher) => launcher.source,
            Self::WouldPrepareManaged { .. } => PythonSource::Managed,
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
            Self::WouldPrepareManaged {
                version,
                requirement,
                tag_check,
            } => Some(format!(
                "no usable Python was found; a real install would use uv to install managed Python {version}{} and save its path in the configuration, which this preview did not do",
                unchecked_tag_caveat(*tag_check, *requirement)
            )),
        }
    }

    /// The `python_note:` line of an update check, in the same terms as
    /// [`Self::note`] but for the step an update would take. An update check
    /// needs the interpreter's wheel tag to read the index, and a host with no
    /// interpreter gets it from the one the layout requires, so this line is how
    /// the check says it did not go and prepare one. Built from the decision, as
    /// the preview's is.
    pub(super) fn update_note(&self) -> Option<String> {
        match self {
            Self::Ready(_) => None,
            Self::WouldPrepareManaged {
                version,
                requirement,
                tag_check,
            } => {
                let unchecked = unchecked_tag_caveat(*tag_check, *requirement);
                Some(format!(
                    "no usable Python was found; updating would use uv to install managed Python {version}{unchecked} and save its path in the configuration, which this check did not do"
                ))
            }
        }
    }

    /// The progress line that names the chosen interpreter: the one an install
    /// goes on with, or, for a preview on a host with no usable Python, what a
    /// real install would prepare. It is the only line that predicts the
    /// preparation, and it exists only for a decision that has already passed the
    /// bootstrap policy and the version check, so the two cannot disagree. Built
    /// from the same fields as [`Self::note`], so the two cannot differ about the
    /// version or whether its tag was checked.
    pub(super) fn progress_text(&self) -> String {
        match self {
            Self::Ready(launcher) => match launcher.source {
                PythonSource::Path => {
                    format!("Using Python from PATH: {}.", launcher.executable.display())
                }
                PythonSource::Env => format!(
                    "Using Python from ROCM_CLI_PYTHON: {}.",
                    launcher.executable.display()
                ),
                PythonSource::Managed => format!(
                    "Using ROCm CLI's portable Python: {}.",
                    launcher.executable.display()
                ),
                PythonSource::Manifest => {
                    format!("Using Python from {}.", launcher.executable.display())
                }
            },
            Self::WouldPrepareManaged {
                version,
                requirement,
                tag_check,
            } => format!(
                "No usable Python found; a real install would prepare Python {version}{}. This preview does not.",
                unchecked_tag_caveat(*tag_check, *requirement)
            ),
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

/// The parenthetical a preview adds after the Python it would prepare when that
/// version's wheel tag could not be read off its name, and nothing when it
/// could. One function for [`PlannedPython::note`] and
/// [`PlannedPython::progress_text`], so the two cannot word the same caveat
/// differently.
fn unchecked_tag_caveat(tag_check: TagCheck, requirement: PythonRequirement) -> String {
    match tag_check {
        TagCheck::Checked => String::new(),
        TagCheck::Unchecked => format!(
            " (its wheel tag cannot be checked until it is installed; {} is required)",
            requirement.tag
        ),
    }
}

/// Whether a search for an interpreter is for an install or a preview of one.
/// It changes only the wording of the two progress lines about a Python that
/// turned out unusable: an install says what it goes on to do, a preview states
/// the finding and leaves the prediction to the decision's own line.
#[derive(Clone, Copy, Debug)]
enum LookupPurpose {
    Install,
    Preview,
}

impl LookupPurpose {
    /// The progress line for an interpreter (`subject`) that exists but cannot
    /// serve the install. Both purposes state the finding; an install then says
    /// what it goes on to do (`install_next_step`) and a preview stops.
    fn unusable_python_line(
        self,
        subject: &str,
        requirement: PythonRequirement,
        install_next_step: &str,
    ) -> String {
        let finding = format!(
            "{subject} is not usable for this ROCm install ({} is required)",
            requirement.tag
        );
        match self {
            Self::Install => format!("{finding}; {install_next_step}."),
            Self::Preview => format!("{finding}."),
        }
    }
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
        PythonLookup::NeedsManagedBootstrap => {
            ensure_managed_python(paths, requirement, env.managed_python_version(requirement))
        }
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
            source: PythonSource::Env,
        }));
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates(&env.search_dirs) {
        match python_launcher_install_ready(&candidate, requirement) {
            Ok(()) => {
                return Ok(PythonLookup::Found(PythonLauncher {
                    executable: candidate,
                    source: PythonSource::Path,
                }));
            }
            Err(_) => {
                skipped_path_python = true;
            }
        }
    }
    // A preview states only what it found here. What a real install would then
    // do depends on the saved Python, the bootstrap policy and the version, none
    // of which is known yet, so the prediction waits for
    // `PlannedPython::progress_text`, after the decision. An install's wording is
    // unchanged.
    if skipped_path_python {
        progress_line(purpose.unusable_python_line(
            "Python from PATH",
            requirement,
            "using ROCm CLI's managed Python",
        ));
    }

    if let Some(manifest) = load_managed_python_manifest(paths)?
        && manifest.executable.is_file()
    {
        if python_launcher_install_ready(&manifest.executable, requirement).is_ok() {
            return Ok(PythonLookup::Found(PythonLauncher {
                executable: manifest.executable,
                source: PythonSource::Managed,
            }));
        }
        progress_line(purpose.unusable_python_line(
            "Saved managed Python",
            requirement,
            "preparing Python again",
        ));
    }

    if env.managed_bootstrap_disabled {
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
    ensure_python_tag_matches(&compatibility.python_tag, requirement)?;
    verify_python_can_create_venv(program).context("Python could not create a virtual environment")
}

/// The one rule an interpreter must meet to serve an install: its wheel tag is
/// the one the layout requires. [`python_launcher_install_ready`] applies it to
/// an interpreter that exists, and a preview applies it to the managed Python it
/// would prepare, so the two cannot disagree about what the install accepts.
fn ensure_python_tag_matches(python_tag: &str, requirement: PythonRequirement) -> Result<()> {
    if python_tag != requirement.tag {
        bail!(
            "Python wheel tag {python_tag} is not supported by this ROCm install; {} is required",
            requirement.tag
        );
    }
    Ok(())
}

/// The wheel tag a managed Python requested as `version` reports, or `None`
/// when it cannot be told without installing it.
///
/// `uv` is what reads `version` (it is passed through to `uv python install`),
/// and this repo has no parser of its own to share, so this reads the spellings
/// `uv` accepts for a CPython release that name a major and a minor: `3.13`,
/// `3.13.1`, `3.13.*`, `3.13t` (free-threaded, which still reports `cp313`), and
/// any of those after `cpython-` or `cpython@`. Whatever follows the minor is
/// not validated, because only the major and minor decide the wheel tag.
/// Anything that leaves the minor open or names something else (another
/// implementation, a range, a wildcard such as `3.*`, a pre-release spelled on
/// the minor) yields `None`.
///
/// This reader is a hand-rolled approximation of what `uv` accepts and can drift
/// from it in either direction: a spelling `uv` takes may read as `None`, and one
/// `uv` would reject may read as a tag. Both are benign. A `None` is announced
/// as unchecked (the preview still prints a plan and says the tag cannot be
/// checked until the Python is installed), never refused, and a tag read off a
/// spelling `uv` then rejects is only a preview that predicted an install that
/// fails at `uv python install`, with `uv`'s own error, as it always did.
fn python_tag_for_version(version: &str) -> Option<String> {
    let version = version.trim();
    let version = ["cpython-", "cpython@"]
        .iter()
        .find_map(|prefix| version.strip_prefix(prefix))
        .unwrap_or(version);
    let mut parts = version.split('.');
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next()?;
    // Only a free-threaded marker may follow the minor's digits.
    let minor = minor
        .strip_suffix('t')
        .unwrap_or(minor)
        .parse::<u32>()
        .ok()?;
    Some(format!("cp{major}{minor}"))
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
#[path = "python_config_props.rs"]
mod config_props;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::{RestoredEnv, test_paths};

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

    /// `--python-version` is a choice among the interpreters the layout is
    /// published for. ROCm 10 takes 3.10 to 3.14 (each with its own wheel tag and
    /// the version `uv` is asked for); the canonical layout takes only its own 3.12;
    /// everything else is refused with the allowed set named, not ignored.
    #[test]
    fn python_version_is_a_choice_among_the_layouts_published_pythons() {
        for (version, tag) in [
            ("3.10", "cp310"),
            ("3.11", "cp311"),
            ("3.12", "cp312"),
            ("3.13", "cp313"),
            ("3.14", "cp314"),
            (" 3.12 ", "cp312"),
        ] {
            let requirement = python_requirement_for_version(SourceLayout::Next, version)
                .unwrap_or_else(|error| panic!("{version} must be accepted: {error:#}"));
            assert_eq!(requirement.tag, tag, "{version}");
            assert_eq!(requirement.version, version.trim(), "{version}");
        }
        for version in ["3.9", "3.15", "3", "3.12.1", "cpython-3.12", ""] {
            let error = python_requirement_for_version(SourceLayout::Next, version)
                .expect_err("an unpublished Python must be refused");
            let rendered = format!("{error:#}");
            assert!(
                rendered.contains("3.10, 3.11, 3.12, 3.13, 3.14"),
                "{version}: {rendered}"
            );
        }

        assert_eq!(
            python_requirement_for_version(SourceLayout::Canonical, "3.12")
                .expect("the canonical layout's own Python is accepted"),
            python_requirement(SourceLayout::Canonical)
        );
        let error = python_requirement_for_version(SourceLayout::Canonical, "3.13")
            .expect_err("the canonical layout has one Python");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("applies only to ROCm 10 and newer installs")
                && rendered.contains("this install uses Python 3.12"),
            "{rendered}"
        );
    }

    /// An update keeps the interpreter a ROCm 10 runtime was built on. The tag
    /// recorded in the runtime maps back to the requirement it came from for every
    /// published Python, a tag the layout is not published for maps to none (so the
    /// update falls back to the layout's default), and the canonical layout never
    /// reads one.
    #[test]
    fn a_recorded_wheel_tag_maps_back_to_the_requirement_it_came_from() {
        for requirement in NEXT_SDK_PYTHONS {
            assert_eq!(
                python_requirement_for_tag(SourceLayout::Next, requirement.tag),
                Some(requirement)
            );
        }
        assert_eq!(python_requirement_for_tag(SourceLayout::Next, "cp39"), None);
        assert_eq!(
            python_requirement_for_tag(SourceLayout::Canonical, "cp313"),
            None
        );
        // The default stays the one vLLM needs, so a flag-less install is unchanged.
        assert_eq!(
            python_requirement(SourceLayout::Next),
            python_requirement_for_tag(SourceLayout::Next, "cp314").expect("cp314 is published")
        );
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
                ..PythonResolverEnv::from_process_env()
            },
            python_requirement(SourceLayout::Canonical),
        )?;
        assert_eq!(launcher.source, PythonSource::Path);
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
                ..PythonResolverEnv::from_process_env()
            },
            python_requirement(SourceLayout::Canonical),
        )?;

        assert_eq!(launcher.source, PythonSource::Path);
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
                ..PythonResolverEnv::from_process_env()
            });
            resolve_python_launcher(&paths, python_requirement(SourceLayout::Canonical))?
        };

        assert_eq!(
            launcher.source,
            PythonSource::Path,
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
                ..PythonResolverEnv::from_process_env()
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
                    ..PythonResolverEnv::from_process_env()
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
                ..PythonResolverEnv::from_process_env()
            });
            python_for_install(
                &self.paths,
                python_requirement(SourceLayout::Canonical),
                dry_run,
            )
        }

        /// What an update check goes on with, searching only this host's folder.
        fn plan_for_update_check(&self) -> Result<PlannedPython> {
            let _resolver = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: vec![self.search_dir.clone()],
                ..PythonResolverEnv::from_process_env()
            });
            plan_python_launcher(&self.paths, python_requirement(SourceLayout::Canonical))
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
    /// wired up. The Gherkin scenario `runtime-23` drives the same case through
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
            tag_check,
        } = planned
        else {
            panic!("a host with no Python must be planned as needing one, got {planned:?}");
        };
        assert_eq!(version, DEFAULT_MANAGED_PYTHON_VERSION);
        assert_eq!(requirement, python_requirement(SourceLayout::Canonical));
        assert_eq!(
            tag_check,
            TagCheck::Checked,
            "the default version's tag is readable"
        );
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

    /// The preview's decision, made from an injected environment, on every
    /// platform.
    ///
    /// The tests around it drive the decision with shell-script interpreters and
    /// run on Unix only. The decision itself (what a host with no usable Python
    /// is told, which managed version is announced or refused, what the bootstrap
    /// policy does) needs no interpreter: a [`PythonResolverEnv`] with nothing to
    /// find and the policy written into it exercises `python_for_install`
    /// end to end, so this one runs on the Windows CI job as well. The search
    /// order and the interpreter probes stay Unix-only, as the PR text says.
    ///
    /// Each row asserts the whole progress output and the state left behind
    /// together. The control at the end is the install on the same host: it
    /// reaches the managed `uv` (a file that cannot be launched, so nothing is
    /// downloaded or run), which shows the preview's silence about `uv` means it
    /// stayed out of it, not that the stand-in was never wired up.
    #[test]
    fn a_dry_run_decides_from_an_injected_environment_on_every_platform() -> Result<()> {
        use super::super::process::PROGRESS_LINE_SINK;

        let _lock = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, paths) = test_paths("python-dry-run-injected");
        let tools = root.join("tools");
        fs::create_dir_all(&tools)?;
        // Present but not executable, on every platform: `ensure_uv_binary`
        // returns it as the override, and launching it fails.
        let uv = tools.join("uv");
        fs::write(&uv, "not an executable")?;
        let _uv = RestoredEnv::set(&[("ROCM_CLI_UV_BINARY", Some(&uv))]);
        assert!(snapshot(&paths).is_empty());

        let requirement = python_requirement(SourceLayout::Canonical);
        let run = |version: Option<&str>, disabled: bool, dry_run: bool| {
            let _resolver = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: Vec::new(),
                managed_python_version: version.map(str::to_owned),
                managed_bootstrap_disabled: disabled,
            });
            PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
            let result = python_for_install(&paths, requirement, dry_run);
            let lines = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());
            (result, lines)
        };
        let promise = |version: &str, caveat: &str| {
            format!(
                "No usable Python found; a real install would prepare Python {version}{caveat}. This preview does not."
            )
        };

        // Announced: the default, a patch release of the required minor, a
        // blank override (read as unset) and a version whose tag cannot be read.
        for (version, announced, tag_check) in [
            (None, "3.12", TagCheck::Checked),
            (Some("3.12.4"), "3.12.4", TagCheck::Checked),
            (Some("cpython@3.12"), "cpython@3.12", TagCheck::Checked),
            (Some("  "), "3.12", TagCheck::Checked),
            (Some("pypy@3.10"), "pypy@3.10", TagCheck::Unchecked),
        ] {
            let (result, lines) = run(version, false, true);
            let planned = result.unwrap_or_else(|error| panic!("{version:?}: {error:#}"));
            let PlannedPython::WouldPrepareManaged {
                version: planned_version,
                tag_check: planned_check,
                ..
            } = &planned
            else {
                panic!(
                    "{version:?}: a host with no Python must be planned as needing one, got {planned:?}"
                );
            };
            assert_eq!(planned_version, announced, "{version:?}");
            assert_eq!(*planned_check, tag_check, "{version:?}");
            let caveat = match tag_check {
                TagCheck::Checked => "",
                TagCheck::Unchecked => {
                    " (its wheel tag cannot be checked until it is installed; cp312 is required)"
                }
            };
            assert_eq!(lines, vec![promise(announced, caveat)], "{version:?}");
            assert!(
                planned.note().is_some_and(|note| note.contains(&format!(
                    "install managed Python {announced}{caveat} and save"
                ))),
                "{version:?}: {planned:?}"
            );
        }

        // Refused: a version whose wheel tag is not the layout's, in every
        // spelling that reads as one, and a host where the bootstrap is
        // disabled. Nothing is promised, so no line predicts the install.
        for version in ["3.13", "3.13t", "cpython-3.13", "cpython@3.13"] {
            let (result, lines) = run(Some(version), false, true);
            let error = format!("{:#}", result.unwrap_err());
            assert!(
                error.contains(&format!("would prepare managed Python {version}"))
                    && error.contains("Python wheel tag cp313 is not supported")
                    && error.contains("cp312 is required"),
                "{version}: {error}"
            );
            assert_eq!(lines, Vec::<String>::new(), "{version}");
        }
        let (preview, lines) = run(None, true, true);
        let preview = preview.unwrap_err().to_string();
        let (install, _) = run(None, true, false);
        assert!(
            preview.contains("managed Python bootstrap is disabled"),
            "{preview}"
        );
        assert_eq!(preview, install.unwrap_err().to_string());
        assert_eq!(lines, Vec::<String>::new());

        // State: every preview above left the config, data and cache folders as
        // they were, which for this host is absent.
        assert!(!paths.config_path().exists());
        assert!(!managed_python_manifest_path(&paths).exists());
        assert_eq!(snapshot(&paths), Vec::<PathBuf>::new());

        // Control: the real install on the same host goes on to prepare Python,
        // which reaches the managed `uv`.
        let (install, lines) = run(None, false, false);
        let error = format!("{:#}", install.unwrap_err());
        assert!(
            error.contains("failed to launch uv python install"),
            "the install did not reach uv, so the previews above prove nothing: {error}"
        );
        assert_eq!(
            lines,
            vec![
                "Preparing Python 3.12...".to_owned(),
                "Installing Python 3.12 via uv...".to_owned()
            ]
        );
        fs::remove_dir_all(&root).ok();
        Ok(())
    }

    /// The two lines about a Python that exists but cannot serve the install are
    /// worded for the purpose of the search: an install says what it goes on to
    /// do, a preview states only the finding (what a real install would then do
    /// is decided afterwards, and said by the decision's own line). Pinned here
    /// as they are printed, so that the two wordings stay what they are as the
    /// code that prints them changes; the decisions those lines sit in are
    /// covered by `a_dry_runs_whole_python_output_follows_one_decision_in_every_state`
    /// and `a_dry_run_decides_from_an_injected_environment_on_every_platform`.
    ///
    /// A text file stands for the unusable interpreters (neither the PATH one nor
    /// the saved one can be launched), so this runs on every platform. Bootstrap is
    /// disabled in the injected environment so both searches end in a refusal
    /// before anything could be prepared.
    #[test]
    fn the_findings_about_an_unusable_python_are_worded_for_the_purpose_of_the_search() -> Result<()>
    {
        use super::super::process::PROGRESS_LINE_SINK;

        let (root, paths) = test_paths("python-unusable-findings");
        let search_dir = root.join("search");
        fs::create_dir_all(&search_dir)?;
        fs::write(search_dir.join("python3"), "not an interpreter")?;
        let saved_dir = paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&saved_dir)?;
        let saved = saved_dir.join("python");
        fs::write(&saved, "not an interpreter")?;
        save_managed_python_manifest(
            &paths,
            &ManagedPythonManifest {
                executable: saved,
                version: DEFAULT_MANAGED_PYTHON_VERSION.to_owned(),
                installed_at_unix_ms: 123,
            },
        )?;
        let env = PythonResolverEnv {
            python_override: None,
            search_dirs: vec![search_dir],
            managed_python_version: None,
            managed_bootstrap_disabled: true,
        };
        let requirement = python_requirement(SourceLayout::Canonical);

        for (purpose, expected) in [
            (
                LookupPurpose::Install,
                [
                    "Python from PATH is not usable for this ROCm install (cp312 is required); using ROCm CLI's managed Python.",
                    "Saved managed Python is not usable for this ROCm install (cp312 is required); preparing Python again.",
                ],
            ),
            (
                LookupPurpose::Preview,
                [
                    "Python from PATH is not usable for this ROCm install (cp312 is required).",
                    "Saved managed Python is not usable for this ROCm install (cp312 is required).",
                ],
            ),
        ] {
            PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
            let outcome = find_python_launcher_in(&paths, &env, requirement, purpose);
            let lines = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());
            assert_eq!(lines, expected, "{purpose:?}");
            let error = outcome
                .err()
                .expect("bootstrap is disabled, so the search refuses");
            assert!(
                error
                    .to_string()
                    .contains("managed Python bootstrap is disabled"),
                "{purpose:?}: {error:#}"
            );
        }
        fs::remove_dir_all(&root).ok();
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
        assert_eq!(launcher.source, PythonSource::Path);
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
        assert_eq!(launcher.source, PythonSource::Managed);
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

    /// `ROCM_CLI_MANAGED_PYTHON_VERSION` naming a version whose wheel tag is not
    /// the one the layout needs: a real install prepares it and then refuses it,
    /// so the preview refuses too, by the same rule, and prepares nothing. A
    /// version the layout does accept (a patch release of its own minor) is still
    /// announced as before.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_refuses_a_managed_python_version_the_install_would_refuse() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-version-override")?;
        let overridden =
            RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new("3.13")))]);

        let error = format!("{:#}", host.python_for_install(true).unwrap_err());

        assert!(
            error.contains("would prepare managed Python 3.13")
                && error.contains("Python wheel tag cp313 is not supported")
                && error.contains("cp312 is required"),
            "{error}"
        );
        assert!(!host.uv_was_run());
        assert_eq!(snapshot(&host.paths), Vec::<PathBuf>::new());
        drop(overridden);

        // A patch release of the required minor is what the install accepts.
        let _patch =
            RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new("3.12.4")))]);
        let planned = host.python_for_install(true)?;
        assert!(
            matches!(&planned, PlannedPython::WouldPrepareManaged { version, .. } if version == "3.12.4"),
            "{planned:?}"
        );
        assert!(!host.uv_was_run());
        Ok(())
    }

    // ---- the preview's output, row by row -------------------------------------
    //
    // State matrix for what a dry run prints about Python and what it leaves
    // behind. A preview line that says what a real install would do must exist
    // only where the install would do it, so every row below is the whole output
    // of the decision, not a spot check of one line.
    //
    //   PATH Python   usable | unusable | absent
    //   saved Python  usable | unusable | absent
    //   bootstrap     enabled | disabled (ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP)
    //   version       default | accepted patch release | unreadable | refused tag
    //
    // Expected output by row (cp312 required; `P`, `S` are the PATH and saved
    // interpreters):
    //   PATH usable                      -> `Using Python from PATH: P.`; nothing else
    //   PATH unusable, saved usable      -> PATH finding, `Using ROCm CLI's portable Python: S.`
    //   neither usable, bootstrap off    -> findings only, then the refusal; no promise
    //   neither usable, version refused  -> findings only, then the refusal; no promise
    //   neither usable, otherwise        -> findings, then the one promise
    //                                       `No usable Python found; a real install
    //                                       would prepare Python V.` (V checked or not)
    // A finding is `Python from PATH is not usable ... (cp312 is required).` or
    // `Saved managed Python is not usable ... (cp312 is required).`, printed only
    // for a PATH or saved Python that exists and fails. Every row leaves the
    // config, data and cache folders as they were and never runs `uv`.

    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum PathPython {
        Usable,
        Unusable,
        Absent,
    }

    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum SavedPython {
        Usable,
        Unusable,
        Absent,
    }

    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum VersionOverride {
        Default,
        Accepted,
        Unreadable,
        RefusedTag,
    }

    /// Runs one row of the matrix and checks the preview's complete output and
    /// the state it leaves against what the row's description above says.
    #[cfg(unix)]
    fn check_preview_matrix_row(
        path: PathPython,
        saved: SavedPython,
        bootstrap_disabled: bool,
        version: VersionOverride,
    ) -> Result<()> {
        use super::super::process::PROGRESS_LINE_SINK;

        let row = format!(
            "{path:?} PATH / {saved:?} saved / disabled={bootstrap_disabled} / {version:?}"
        );
        let host = PythonHost::new("python-dry-run-matrix")?;
        let path_python = match path {
            PathPython::Usable => Some(write_fake_python_with_venv(&host.search_dir, "python3")?),
            PathPython::Unusable => {
                write_fake_python_reporting(&host.search_dir, "python3", "cp311", "Python 3.11.0")?;
                None
            }
            PathPython::Absent => None,
        };
        let managed_dir = host.paths.data_dir.join("tools").join("python");
        let saved_python = match saved {
            SavedPython::Usable => {
                fs::create_dir_all(&managed_dir)?;
                let python = write_fake_python_with_venv(&managed_dir, "python")?;
                host.save_managed_python(python.clone())?;
                Some(python)
            }
            SavedPython::Unusable => {
                fs::create_dir_all(&managed_dir)?;
                let broken = managed_dir.join("python");
                fs::write(&broken, "not an interpreter")?;
                host.save_managed_python(broken)?;
                None
            }
            SavedPython::Absent => None,
        };
        let (version_value, version_name) = match version {
            VersionOverride::Default => (None, "3.12"),
            VersionOverride::Accepted => (Some("3.12.4"), "3.12.4"),
            VersionOverride::Unreadable => (Some("pypy@3.10"), "pypy@3.10"),
            VersionOverride::RefusedTag => (Some("3.13"), "3.13"),
        };
        let _env = RestoredEnv::set(&[
            (
                "ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP",
                bootstrap_disabled.then_some(Path::new("1")),
            ),
            (
                "ROCM_CLI_MANAGED_PYTHON_VERSION",
                version_value.map(Path::new),
            ),
        ]);
        let before = snapshot(&host.paths);

        // What the row's description says, written out independently of the code.
        let path_finding =
            "Python from PATH is not usable for this ROCm install (cp312 is required).";
        let saved_finding =
            "Saved managed Python is not usable for this ROCm install (cp312 is required).";
        let mut expected: Vec<String> = Vec::new();
        let mut expected_error: Option<&str> = None;
        let mut expected_promise = false;
        if let Some(python) = &path_python {
            expected.push(format!("Using Python from PATH: {}.", python.display()));
        } else {
            if matches!(path, PathPython::Unusable) {
                expected.push(path_finding.to_owned());
            }
            if let Some(python) = &saved_python {
                expected.push(format!(
                    "Using ROCm CLI's portable Python: {}.",
                    python.display()
                ));
            } else {
                if matches!(saved, SavedPython::Unusable) {
                    expected.push(saved_finding.to_owned());
                }
                if bootstrap_disabled {
                    expected_error = Some("managed Python bootstrap is disabled");
                } else if matches!(version, VersionOverride::RefusedTag) {
                    expected_error = Some("Python wheel tag cp313 is not supported");
                } else {
                    expected_promise = true;
                    expected.push(if matches!(version, VersionOverride::Unreadable) {
                        format!("No usable Python found; a real install would prepare Python {version_name} (its wheel tag cannot be checked until it is installed; cp312 is required). This preview does not.")
                    } else {
                        format!("No usable Python found; a real install would prepare Python {version_name}. This preview does not.")
                    });
                }
            }
        }

        PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
        let result = host.python_for_install(true);
        let lines = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());

        assert_eq!(lines, expected, "{row}: the preview's progress lines");
        // No line that predicts the install may exist unless the row ends in
        // the promise: a refusal is known before any such line is built.
        let predictions = lines
            .iter()
            .filter(|line| line.contains("a real install would"))
            .count();
        assert_eq!(
            predictions,
            usize::from(expected_promise),
            "{row}: lines that predict the install: {lines:?}"
        );
        match (&result, expected_error) {
            (Err(error), Some(expected)) => {
                assert!(format!("{error:#}").contains(expected), "{row}: {error:#}");
            }
            (Ok(planned), None) => {
                assert_eq!(
                    planned.note().is_some(),
                    expected_promise,
                    "{row}: the note goes with the promise"
                );
                if expected_promise {
                    assert!(
                        planned.note().is_some_and(
                            |note| note.contains(&format!("managed Python {version_name}"))
                        ),
                        "{row}: {planned:?}"
                    );
                }
            }
            (result, expected) => panic!("{row}: got {result:?}, expected error {expected:?}"),
        }
        // State: nothing was prepared and nothing was written.
        assert!(!host.uv_was_run(), "{row}: the preview ran uv");
        assert!(
            !host.paths.config_path().exists(),
            "{row}: a config was written"
        );
        assert_eq!(snapshot(&host.paths), before, "{row}: files changed");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_dry_runs_whole_python_output_follows_one_decision_in_every_state() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        for path in [PathPython::Usable, PathPython::Unusable, PathPython::Absent] {
            for saved in [
                SavedPython::Usable,
                SavedPython::Unusable,
                SavedPython::Absent,
            ] {
                for bootstrap_disabled in [false, true] {
                    for version in [
                        VersionOverride::Default,
                        VersionOverride::Accepted,
                        VersionOverride::Unreadable,
                        VersionOverride::RefusedTag,
                    ] {
                        check_preview_matrix_row(path, saved, bootstrap_disabled, version)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The same refusal for every spelling `uv` accepts for that release: a
    /// free-threaded `3.13t` and the `cpython-` and `cpython@` forms report
    /// `cp313` once installed just as `3.13` does, so a preview that did not read
    /// them would promise a plan the install refuses. A spelling whose tag
    /// cannot be read is announced as unchecked, not as a promise.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_reads_every_spelling_of_a_managed_python_version_the_same_way() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-version-spellings")?;
        for spelling in ["3.13t", "cpython-3.13", "cpython@3.13"] {
            let _version =
                RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new(spelling)))]);
            let error = format!("{:#}", host.python_for_install(true).unwrap_err());
            assert!(
                error.contains(&format!("would prepare managed Python {spelling}"))
                    && error.contains("Python wheel tag cp313 is not supported")
                    && error.contains("cp312 is required"),
                "{spelling}: {error}"
            );
        }
        for accepted in ["cpython-3.12", "cpython@3.12.4"] {
            let _version =
                RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new(accepted)))]);
            let planned = host.python_for_install(true)?;
            assert!(
                matches!(
                    &planned,
                    PlannedPython::WouldPrepareManaged {
                        tag_check: TagCheck::Checked,
                        ..
                    }
                ),
                "{accepted}: {planned:?}"
            );
        }

        let _unreadable = RestoredEnv::set(&[(
            "ROCM_CLI_MANAGED_PYTHON_VERSION",
            Some(Path::new("pypy@3.10")),
        )]);
        let planned = host.python_for_install(true)?;
        assert!(
            matches!(
                &planned,
                PlannedPython::WouldPrepareManaged {
                    tag_check: TagCheck::Unchecked,
                    ..
                }
            ),
            "{planned:?}"
        );
        let note = planned
            .note()
            .expect("a preview that skipped a step says so");
        assert!(
            note.contains("managed Python pypy@3.10 (its wheel tag cannot be checked until it is installed; cp312 is required)")
                && note.ends_with("which this preview did not do"),
            "{note}"
        );
        assert_eq!(
            planned.progress_text(),
            "No usable Python found; a real install would prepare Python pypy@3.10 (its wheel tag cannot be checked until it is installed; cp312 is required). This preview does not."
        );
        assert!(!host.uv_was_run());
        assert_eq!(snapshot(&host.paths), Vec::<PathBuf>::new());
        Ok(())
    }

    /// An update check reads the index by the interpreter's wheel tag. On a host
    /// with no interpreter that tag is the one the layout requires, which a real
    /// update would prepare an interpreter to match, so the check needs nothing
    /// prepared. The note it prints says so, and the folders agree; the same host
    /// DOES reach `uv` for the install, so an unrun stand-in means the check
    /// stayed out of it. The Gherkin scenario `update-09` drives the same case
    /// through the built binary.
    #[cfg(unix)]
    #[test]
    fn an_update_check_with_no_usable_python_reads_the_required_tag_and_prepares_nothing()
    -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-update-check-none")?;
        let requirement = python_requirement(SourceLayout::Canonical);

        let planned = host.plan_for_update_check()?;

        let compatibility = planned.wheel_compatibility()?;
        assert_eq!(compatibility.python_tag, requirement.tag);
        assert!(!compatibility.platform_tags.is_empty());
        assert_eq!(
            planned.update_note().as_deref(),
            Some(
                "no usable Python was found; updating would use uv to install managed Python 3.12 and save its path in the configuration, which this check did not do"
            )
        );
        assert!(!host.uv_was_run(), "the update check ran uv");
        assert_eq!(
            snapshot(&host.paths),
            Vec::<PathBuf>::new(),
            "the update check created files or folders under the config, data or cache folder"
        );

        // Control: the install on the same host does reach `uv`.
        assert!(host.python_for_install(false).is_err());
        assert!(
            host.uv_was_run(),
            "the stand-in uv is not reachable, so the checks above prove nothing"
        );
        Ok(())
    }

    /// A usable interpreter needs no note, and a managed Python whose tag cannot
    /// be read off its version says so in the update note as in the preview's.
    #[cfg(unix)]
    #[test]
    fn an_update_note_is_absent_for_a_usable_python_and_flags_an_unreadable_tag() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-update-note")?;
        {
            let _unreadable = RestoredEnv::set(&[(
                "ROCM_CLI_MANAGED_PYTHON_VERSION",
                Some(Path::new("pypy@3.10")),
            )]);
            let planned = host.plan_for_update_check()?;
            let note = planned.update_note().expect("nothing was prepared");
            assert!(
                note.contains("managed Python pypy@3.10 (its wheel tag cannot be checked until it is installed; cp312 is required)")
                    && note.ends_with("which this check did not do"),
                "{note}"
            );
        }

        write_fake_python_with_venv(&host.search_dir, "python3")?;
        let planned = host.plan_for_update_check()?;
        assert!(matches!(planned, PlannedPython::Ready(_)), "{planned:?}");
        assert_eq!(planned.update_note(), None);
        assert!(!host.uv_was_run());
        Ok(())
    }

    /// The version-to-tag reading the preview's refusal rests on, and that it is
    /// the same rule `python_launcher_install_ready` applies (same message).
    #[test]
    fn a_managed_python_version_is_read_as_the_wheel_tag_it_reports() {
        assert_eq!(python_tag_for_version("3.12").as_deref(), Some("cp312"));
        assert_eq!(python_tag_for_version(" 3.14.1 ").as_deref(), Some("cp314"));
        assert_eq!(python_tag_for_version("3.9").as_deref(), Some("cp39"));
        // Spellings `uv` accepts for the same CPython release read the same.
        assert_eq!(python_tag_for_version("3.13t").as_deref(), Some("cp313"));
        assert_eq!(python_tag_for_version("3.13.1").as_deref(), Some("cp313"));
        assert_eq!(python_tag_for_version("3.13.*").as_deref(), Some("cp313"));
        assert_eq!(
            python_tag_for_version("cpython-3.13").as_deref(),
            Some("cp313")
        );
        assert_eq!(
            python_tag_for_version("cpython@3.13").as_deref(),
            Some("cp313")
        );
        assert_eq!(
            python_tag_for_version("cpython-3.12.4").as_deref(),
            Some("cp312")
        );
        // Anything else: only installing it would say what it reports.
        for unreadable in [
            "",
            "3",
            "3.x",
            "3.*",
            "*",
            "latest",
            "pypy@3.10",
            "3.13rc1",
            ">=3.12",
            "cpython-",
        ] {
            assert_eq!(python_tag_for_version(unreadable), None, "{unreadable:?}");
        }

        let required = python_requirement(SourceLayout::Canonical);
        assert!(ensure_python_tag_matches("cp312", required).is_ok());
        let error = ensure_python_tag_matches("cp313", required)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "Python wheel tag cp313 is not supported by this ROCm install; cp312 is required"
        );
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
                tag_check: TagCheck::Checked,
            };

            let compatibility = planned.wheel_compatibility()?;
            assert_eq!(compatibility.python_tag, tag);
            assert!(compatibility.platform_tags.contains(&"any".to_owned()));
            assert_eq!(planned.source(), PythonSource::Managed);
            assert_eq!(planned.source().as_str(), "managed");
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
            source: PythonSource::Path,
        });
        assert_eq!(planned.launcher_display(), "/usr/bin/python3");
        assert_eq!(planned.source().as_str(), "path");
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

    /// A stand-in `uv` that records every invocation and answers
    /// `python install` / `python find` without touching the network.
    /// Planted where `ensure_uv_binary` looks for the managed copy, so no
    /// download and no environment override is needed.
    #[cfg(unix)]
    fn plant_fake_uv(paths: &AppPaths, python: &Path, log: &Path) -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let dir = managed_tools_dir(&paths.data_dir).join("uv").join("latest");
        fs::create_dir_all(&dir)?;
        let uv = dir.join(rocm_core::uv_binary_name());
        let script = format!(
            "#!/bin/sh\necho \"$*\" >> '{log}'\nif [ \"$1 $2\" = \"python find\" ]; then echo '{python}'; fi\nexit 0\n",
            log = log.display(),
            python = python.display(),
        );
        write_fake_executable(&uv, script)?;
        Ok(())
    }

    /// A config.json `load` rejects, holding settings that must survive.
    #[cfg(unix)]
    fn plant_unreadable_config(paths: &AppPaths) -> Result<Vec<u8>> {
        fs::create_dir_all(&paths.config_dir)?;
        let damaged = br#"{"active_runtime_key": "kept-runtime", "onboarding_dismissed": "yes"}"#;
        fs::write(paths.config_path(), damaged)?;
        assert!(RocmCliConfig::load(paths).is_err());
        Ok(damaged.to_vec())
    }

    /// The error names the exact file to rename and what to rename it to.
    /// The advice text itself is `AppPaths::config_aside_advice`; that
    /// following it keeps the data dir is
    /// `config_rename_advice_keeps_a_recorded_data_dir_when_followed` in
    /// rocm-core's paths.rs.
    #[cfg(unix)]
    fn assert_names_the_file_to_rename(message: &str, paths: &AppPaths) {
        let config = paths.config_path();
        assert!(
            message.contains(&format!(
                "rename \"{}\" to \"{}.broken\"",
                config.display(),
                config.display()
            )) && message.contains(&paths.config_aside_advice()),
            "the error does not name the file to rename: {message}"
        );
    }

    /// The reuse path: a usable managed Python is already recorded, so
    /// `ensure_managed_python` returns early. Meeting an unreadable config
    /// there must fail — not report success with the record silently skipped
    /// — before `uv` is run at all, and leave the file as it was.
    #[cfg(unix)]
    #[test]
    fn ensure_managed_python_reuse_path_fails_on_an_unreadable_config() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        // Held because the uv and Python versions are read from the
        // environment, which other tests in this module set.
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("ensure-python-reuse-unreadable");
        let python_dir = managed_tools_dir(&paths.data_dir).join("python");
        fs::create_dir_all(&python_dir)?;
        let python = write_fake_python_with_venv(&python_dir, "python")?;
        save_managed_python_manifest(
            &paths,
            &ManagedPythonManifest {
                executable: python.clone(),
                version: managed_python_version(python_requirement(SourceLayout::Canonical)),
                installed_at_unix_ms: 1,
            },
        )?;
        let uv_log = root.join("uv.log");
        plant_fake_uv(&paths, &python, &uv_log)?;
        let damaged = plant_unreadable_config(&paths)?;

        // `PythonLauncher` is not `Debug`, so `expect_err` is unavailable.
        let requirement = python_requirement(SourceLayout::Canonical);
        let Err(error) =
            ensure_managed_python(&paths, requirement, managed_python_version(requirement))
        else {
            panic!("an unreadable config must fail the reuse path");
        };
        let uv_calls = fs::read_to_string(&uv_log).unwrap_or_default();
        let after = fs::read(paths.config_path())?;
        fs::remove_dir_all(root).ok();

        let message = format!("{error:#}");
        assert!(
            message.contains("cannot record the managed Python")
                && message.contains("it was left unchanged"),
            "unexpected error: {message}"
        );
        assert_names_the_file_to_rename(&message, &paths);
        assert_eq!(after, damaged, "config.json was rewritten");
        assert!(
            uv_calls.is_empty(),
            "uv ran before the unreadable config stopped the step: {uv_calls}"
        );
        Ok(())
    }

    /// The install path: no managed Python yet, so `uv python install` runs
    /// and the new interpreter is recorded at the end. An unreadable config
    /// fails the bootstrap before `uv` is run at all, with the file left as it
    /// was.
    #[cfg(unix)]
    #[test]
    fn ensure_managed_python_install_path_fails_on_an_unreadable_config() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("ensure-python-install-unreadable");
        let python_dir = root.join("uv-python");
        fs::create_dir_all(&python_dir)?;
        let python = write_fake_python_with_venv(&python_dir, "python")?;
        let uv_log = root.join("uv.log");
        plant_fake_uv(&paths, &python, &uv_log)?;
        let damaged = plant_unreadable_config(&paths)?;

        // `PythonLauncher` is not `Debug`, so `expect_err` is unavailable.
        let requirement = python_requirement(SourceLayout::Canonical);
        let Err(error) =
            ensure_managed_python(&paths, requirement, managed_python_version(requirement))
        else {
            panic!("an unreadable config must fail the install path");
        };
        let uv_calls = fs::read_to_string(&uv_log).unwrap_or_default();
        let after = fs::read(paths.config_path())?;
        fs::remove_dir_all(root).ok();

        let message = format!("{error:#}");
        assert!(
            message.contains("cannot record the managed Python")
                && message.contains("it was left unchanged"),
            "unexpected error: {message}"
        );
        assert_names_the_file_to_rename(&message, &paths);
        assert_eq!(after, damaged, "config.json was rewritten");
        assert!(
            uv_calls.is_empty(),
            "uv ran before the unreadable config stopped the step: {uv_calls}"
        );
        Ok(())
    }

    /// A config.json that reads fine but cannot be saved: its folder is made
    /// read-only, so the save cannot create its temp file. Both paths must
    /// report that the interpreter was not recorded, rather than finish as if
    /// it had been. Skipped where the folder stays writable (running as root).
    #[cfg(unix)]
    #[test]
    fn ensure_managed_python_fails_when_the_record_cannot_be_saved() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        for reuse in [true, false] {
            let (root, paths) = test_paths("ensure-python-record-unsaved");
            let python_dir = if reuse {
                managed_tools_dir(&paths.data_dir).join("python")
            } else {
                root.join("uv-python")
            };
            fs::create_dir_all(&python_dir)?;
            let python = write_fake_python_with_venv(&python_dir, "python")?;
            if reuse {
                save_managed_python_manifest(
                    &paths,
                    &ManagedPythonManifest {
                        executable: python.clone(),
                        version: managed_python_version(python_requirement(
                            SourceLayout::Canonical,
                        )),
                        installed_at_unix_ms: 1,
                    },
                )?;
            }
            plant_fake_uv(&paths, &python, &root.join("uv.log"))?;
            RocmCliConfig::default().save(&paths)?;
            fs::set_permissions(&paths.config_dir, fs::Permissions::from_mode(0o555))?;
            let probe = paths.config_dir.join("probe");
            if fs::write(&probe, b"").is_ok() {
                fs::remove_file(&probe).ok();
                fs::set_permissions(&paths.config_dir, fs::Permissions::from_mode(0o755))?;
                fs::remove_dir_all(root).ok();
                return Ok(());
            }

            let requirement = python_requirement(SourceLayout::Canonical);
            let result =
                ensure_managed_python(&paths, requirement, managed_python_version(requirement));
            fs::set_permissions(&paths.config_dir, fs::Permissions::from_mode(0o755))?;
            fs::remove_dir_all(root).ok();
            let Err(error) = result else {
                panic!("reuse={reuse}: an unsaved record must fail the step");
            };
            assert!(
                format!("{error:#}").contains("config.json"),
                "reuse={reuse}: {error:#}"
            );
        }
        Ok(())
    }
}
