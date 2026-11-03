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

fn resolve_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    if let Some(value) = env.python_override.as_deref() {
        python_launcher_install_ready(Path::new(value), requirement)
            .with_context(|| format!("ROCM_CLI_PYTHON is not usable for ROCm setup: {value}"))?;
        return Ok(PythonLauncher {
            executable: PathBuf::from(value),
            source: "env",
        });
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates(&env.search_dirs) {
        match python_launcher_install_ready(&candidate, requirement) {
            Ok(()) => {
                return Ok(PythonLauncher {
                    executable: candidate,
                    source: "path",
                });
            }
            Err(_) => {
                skipped_path_python = true;
            }
        }
    }
    if skipped_path_python {
        progress_line(format!(
            "Python from PATH is not usable for this ROCm install ({} is required); using ROCm CLI's managed Python.",
            requirement.tag
        ));
    }

    if let Some(manifest) = load_managed_python_manifest(paths)?
        && manifest.executable.is_file()
    {
        if python_launcher_install_ready(&manifest.executable, requirement).is_ok() {
            return Ok(PythonLauncher {
                executable: manifest.executable,
                source: "managed",
            });
        }
        progress_line(format!(
            "Saved managed Python is not usable for this ROCm install ({} is required); preparing Python again.",
            requirement.tag
        ));
    }

    if managed_python_bootstrap_disabled() {
        bail!(
            "unable to locate Python, and managed Python bootstrap is disabled by ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP"
        );
    }
    ensure_managed_python(paths, requirement)
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
