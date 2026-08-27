// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use directories::{BaseDirs, ProjectDirs};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RuntimePlatform {
    Windows,
    Linux,
    Other(&'static str),
}

impl RuntimePlatform {
    pub const fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else if cfg!(target_os = "linux") {
            Self::Linux
        } else {
            Self::Other(std::env::consts::OS)
        }
    }

    pub const fn os_name(self) -> &'static str {
        match self {
            Self::Windows => "windows",
            Self::Linux => "linux",
            Self::Other(os) => os,
        }
    }

    pub const fn is_windows(self) -> bool {
        matches!(self, Self::Windows)
    }

    pub const fn is_linux(self) -> bool {
        matches!(self, Self::Linux)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RuntimeHost {
    platform: RuntimePlatform,
}

impl RuntimeHost {
    pub const fn current() -> Self {
        Self {
            platform: RuntimePlatform::current(),
        }
    }

    pub const fn platform(self) -> RuntimePlatform {
        self.platform
    }

    pub const fn os_name(self) -> &'static str {
        self.platform.os_name()
    }

    pub const fn is_windows(self) -> bool {
        self.platform.is_windows()
    }

    pub const fn is_linux(self) -> bool {
        self.platform.is_linux()
    }
}

pub const fn runtime_is_windows() -> bool {
    RuntimeHost::current().is_windows()
}

pub const fn runtime_is_linux() -> bool {
    RuntimeHost::current().is_linux()
}

pub const fn runtime_os_name() -> &'static str {
    RuntimeHost::current().os_name()
}

pub const fn runtime_exe_suffix() -> &'static str {
    if runtime_is_windows() { ".exe" } else { "" }
}

pub const fn runtime_python_bin_dir_name() -> &'static str {
    if runtime_is_windows() {
        "Scripts"
    } else {
        "bin"
    }
}

pub const fn runtime_python_executable_name() -> &'static str {
    if runtime_is_windows() {
        "python.exe"
    } else {
        "python"
    }
}

pub fn runtime_python_env_bin_dir(env_root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(env_root).join(runtime_python_bin_dir_name())
}

pub fn runtime_python_executable_in_env(env_root: &Path) -> PathBuf {
    runtime_python_env_bin_dir(env_root).join(runtime_python_executable_name())
}

pub fn runtime_python_activation_script(env_root: &Path) -> PathBuf {
    let script = if runtime_is_windows() {
        "activate.bat"
    } else {
        "activate"
    };
    runtime_python_env_bin_dir(env_root).join(script)
}

pub fn runtime_python_activation_hint(env_root: &Path) -> String {
    let script = runtime_python_activation_script(env_root);
    if runtime_is_windows() {
        script.display().to_string()
    } else {
        format!("source {}", script.display())
    }
}

// `shortname` is always an internal, lowercase ROCm library name (never user
// input), so the `.dll`/`.so` suffix checks are intentionally case-sensitive.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
pub fn runtime_rocm_library_filename(shortname: &str) -> String {
    if runtime_is_windows() {
        match shortname {
            "amdhip64" => "amdhip64.dll".to_owned(),
            other if other.ends_with(".dll") => other.to_owned(),
            other => format!("{other}.dll"),
        }
    } else {
        match shortname {
            other if other.starts_with("lib") && other.ends_with(".so") => other.to_owned(),
            other if other.ends_with(".so") => other.to_owned(),
            other => format!("lib{other}.so"),
        }
    }
}

pub fn default_interactive_shell_program() -> Option<String> {
    if runtime_is_windows() {
        std::env::var("COMSPEC")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| Some("cmd".to_owned()))
    } else {
        std::env::var("SHELL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| Some("sh".to_owned()))
    }
}

pub fn shell_command_for_host(command: &str) -> (String, Vec<String>) {
    if runtime_is_windows() {
        ("cmd".to_owned(), vec!["/C".to_owned(), command.to_owned()])
    } else {
        ("sh".to_owned(), vec!["-c".to_owned(), command.to_owned()])
    }
}

pub fn runtime_home_dir() -> Option<PathBuf> {
    if runtime_is_windows() {
        if let Some(profile) = std::env::var_os("USERPROFILE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            return Some(profile);
        }
        if let (Some(drive), Some(path)) = (
            std::env::var_os("HOMEDRIVE").filter(|value| !value.is_empty()),
            std::env::var_os("HOMEPATH").filter(|value| !value.is_empty()),
        ) {
            let mut home = PathBuf::from(drive);
            home.push(path);
            return Some(home);
        }
    }
    BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
}

pub(crate) fn env_path_override(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

pub(crate) fn home_rocm_dir() -> Option<PathBuf> {
    runtime_home_dir().map(|dir| dir.join(".rocm"))
}

fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("org", "ROCm", "rocm-cli")
}

pub fn default_config_dir() -> Option<PathBuf> {
    home_rocm_dir().or_else(|| project_dirs().map(|dirs| dirs.config_dir().to_path_buf()))
}

pub fn default_data_dir() -> Option<PathBuf> {
    home_rocm_dir().or_else(|| project_dirs().map(|dirs| dirs.data_dir().to_path_buf()))
}

pub fn default_cache_dir() -> Option<PathBuf> {
    home_rocm_dir()
        .map(|dir| dir.join("cache"))
        .or_else(|| project_dirs().map(|dirs| dirs.cache_dir().to_path_buf()))
}

/// Resolve a writable, user-owned directory for per-user runtime state from
/// explicit environment inputs.
///
/// Runtime state (sockets, lock files, an engine's scratch area) needs a
/// directory whose *parent* is user-owned, so it can be tightened to mode
/// `0o700` without the `EPERM` that results from trying to `chmod` a shared,
/// root-owned `/tmp` (mode `1777`). Precedence:
///
/// 1. `$XDG_RUNTIME_DIR` — already mode `0700` on systemd systems, ideal.
/// 2. `$HOME/.rocm/data/<home_subdir>` — standard per-user data dir.
/// 3. `temp_dir()/rocm-<user>/<temp_subdir>` — user-named subdir so the parent
///    is something we create and own, not `/tmp` itself.
///
/// Tiers 2 and 3 are not exotic: `$XDG_RUNTIME_DIR` is populated by
/// `pam_systemd` at login, so it is absent for every non-login process (cron
/// jobs, CI runners, `systemd-run`, a bare container exec).
///
/// The environment is taken as arguments rather than read here so the
/// precedence is testable without mutating process-global env vars, which is
/// `unsafe` and racy under parallel tests in edition 2024.
///
/// `home_subdir` and `temp_subdir` name the caller's own leaf directory in
/// tiers 2 and 3; either may be empty when the caller owns that whole tier.
/// They are separate parameters because tier 2 lands inside the shared
/// `.rocm/data` tree, where every component needs its own leaf, whereas tier 3
/// is already under a rocm-private `rocm-<user>` directory.
pub fn user_runtime_dir(
    xdg_runtime_dir: Option<OsString>,
    home: Option<OsString>,
    user: Option<String>,
    temp_dir: PathBuf,
    home_subdir: &str,
    temp_subdir: &str,
) -> PathBuf {
    if let Some(runtime_dir) = xdg_runtime_dir.filter(|value| !value.is_empty()) {
        return PathBuf::from(runtime_dir);
    }
    if let Some(home) = home.filter(|value| !value.is_empty()) {
        return join_subdir(PathBuf::from(home).join(".rocm").join("data"), home_subdir);
    }
    let user_dir = format!("rocm-{}", sanitized_user_path_component(user));
    join_subdir(temp_dir.join(user_dir), temp_subdir)
}

fn join_subdir(mut base: PathBuf, subdir: &str) -> PathBuf {
    if !subdir.is_empty() {
        base.push(subdir);
    }
    base
}

/// Reduce a user name to a single safe path component: keep only alphanumeric,
/// hyphen, and underscore so a path separator or `..` in the env var cannot
/// escape the intended subdirectory. An absent or fully-stripped name yields
/// `user`.
fn sanitized_user_path_component(user: Option<String>) -> String {
    let sanitized: String = user
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "user".to_owned()
    } else {
        sanitized
    }
}

pub fn managed_runtime_cache_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("cache")
}

pub fn managed_pip_cache_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("pip-cache")
}

/// `uv`'s content-addressed cache, kept under the managed root so it shares a filesystem
/// with the environments `uv` populates and hardlinking keeps working (see issue #160).
pub fn managed_uv_cache_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("uv-cache")
}

pub fn managed_logs_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("logs")
}

pub fn managed_tools_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("tools")
}

pub fn runtime_path_list_split(value: &OsStr) -> Vec<PathBuf> {
    if !runtime_is_windows() {
        return std::env::split_paths(value).collect();
    }
    value
        .to_string_lossy()
        .split(';')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| normalize_runtime_path_for_host(Path::new(entry)))
        .collect()
}

pub fn runtime_path_list_join<I, P>(entries: I) -> Result<OsString>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let entries = entries
        .into_iter()
        .map(|entry| normalize_runtime_path_for_host(entry.as_ref()))
        .collect::<Vec<_>>();
    if runtime_is_windows() {
        let joined = entries
            .iter()
            .map(|entry| runtime_path_for_windows_child(entry))
            .collect::<Vec<_>>()
            .join(";");
        return Ok(OsString::from(joined));
    }
    std::env::join_paths(entries).context("failed to join PATH entries")
}

pub fn prepend_runtime_path(prefix: &Path, current_path: Option<&OsStr>) -> Result<OsString> {
    let mut parts = vec![normalize_runtime_path_for_host(prefix)];
    if let Some(current_path) = current_path {
        parts.extend(runtime_path_list_split(current_path));
    }
    runtime_path_list_join(parts)
}

pub(crate) fn runtime_path_for_child_process(path: &Path) -> String {
    if runtime_is_windows() {
        runtime_path_for_windows_child(path)
    } else {
        path.display().to_string()
    }
}

pub fn runtime_path_for_windows_child(path: &Path) -> String {
    normalize_windows_storage_path_text(&path.display().to_string())
}

pub fn runtime_path_for_child(path: &Path) -> String {
    runtime_path_for_child_process(path)
}

pub fn runtime_directory_label(path: &Path) -> String {
    let mut label = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.display().to_string());
    let separator = if runtime_is_windows() { '\\' } else { '/' };
    if !label.ends_with(['/', '\\']) {
        label.push(separator);
    }
    label
}

pub fn runtime_path_sort_key(path: &Path) -> String {
    let key = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.display().to_string());
    if runtime_is_windows() {
        key.to_ascii_lowercase()
    } else {
        key
    }
}

pub fn runtime_drive_roots() -> Vec<PathBuf> {
    if !runtime_is_windows() {
        return Vec::new();
    }
    ('A'..='Z')
        .map(|letter| PathBuf::from(format!("{letter}:/")))
        .filter(|path| path.is_dir())
        .collect()
}

pub fn runtime_drive_root_for_key(ch: char) -> Option<PathBuf> {
    if !runtime_is_windows() || !ch.is_ascii_alphabetic() {
        return None;
    }
    let path = PathBuf::from(format!("{}:/", ch.to_ascii_uppercase()));
    path.is_dir().then_some(path)
}

pub fn runtime_paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = normalize_runtime_path_for_host(left)
        .display()
        .to_string()
        .replace('\\', "/");
    let right = normalize_runtime_path_for_host(right)
        .display()
        .to_string()
        .replace('\\', "/");
    if runtime_is_windows() {
        left.eq_ignore_ascii_case(&right)
    } else {
        left == right
    }
}

pub fn runtime_path_is_same_or_inside(path: &Path, base: &Path) -> bool {
    let path = normalize_runtime_path_for_host(path);
    let base = normalize_runtime_path_for_host(base);
    if runtime_paths_equivalent(&path, &base) {
        return true;
    }
    path.ancestors()
        .skip(1)
        .any(|ancestor| runtime_paths_equivalent(ancestor, &base))
}

const MANAGED_RUNTIME_FORMATS: [&str; 2] = ["wheel", "tarball"];

/// Strip trailing managed runtime leaves to recover the canonical data root.
pub fn managed_runtime_data_root(root: &Path) -> PathBuf {
    let mut root = normalize_runtime_path_for_host(root);
    while let Some(parent) = strip_managed_runtime_leaf(&root) {
        root = parent;
    }
    root
}

fn strip_managed_runtime_leaf(path: &Path) -> Option<PathBuf> {
    let format_dir = path.parent()?;
    let format = format_dir.file_name()?.to_str()?;
    if !MANAGED_RUNTIME_FORMATS.contains(&format) {
        return None;
    }
    let runtimes_dir = format_dir.parent()?;
    if runtimes_dir.file_name()? != std::ffi::OsStr::new("runtimes") {
        return None;
    }
    runtimes_dir.parent().map(Path::to_path_buf)
}

pub fn runtime_install_root_is_protected(path: &Path) -> bool {
    let path = normalize_runtime_path_for_host(path);
    if let Some(home) = runtime_home_dir() {
        let home = normalize_runtime_path_for_host(&home);
        if runtime_path_is_same_or_inside(&path, &home) && !runtime_paths_equivalent(&path, &home) {
            return false;
        }
    }

    if runtime_is_windows() {
        let system_roots = ["C:/Windows", "C:/Program Files", "C:/Program Files (x86)"];
        return system_roots
            .iter()
            .map(Path::new)
            .any(|root| runtime_path_is_same_or_inside(&path, root));
    }

    if runtime_paths_equivalent(&path, Path::new("/")) {
        return true;
    }

    [
        "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/opt", "/proc", "/root", "/sbin",
        "/sys", "/usr", "/var",
    ]
    .iter()
    .map(Path::new)
    .any(|root| runtime_path_is_same_or_inside(&path, root))
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RuntimePathSeparator {
    Native,
    Slash,
    Backslash,
}

impl RuntimePathSeparator {
    const fn separator(self) -> char {
        match self {
            Self::Native if std::path::MAIN_SEPARATOR == '\\' => '\\',
            Self::Native | Self::Slash => '/',
            Self::Backslash => '\\',
        }
    }
}

pub fn current_executable_path() -> Result<PathBuf> {
    match std::env::current_exe() {
        Ok(path) => Ok(path),
        Err(current_exe_error) => current_executable_path_from_argv0()
            .with_context(|| format!("failed to discover current executable: {current_exe_error}")),
    }
}

fn current_executable_path_from_argv0() -> Result<PathBuf> {
    let argv0 = std::env::args_os()
        .next()
        .context("current process argv[0] is unavailable")?;
    let current_dir = std::env::current_dir().ok();
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    current_executable_path_from_argv0_value(
        argv0.as_os_str(),
        current_dir.as_deref(),
        Some(path_var.as_os_str()),
        false,
    )
}

fn current_executable_path_from_argv0_value(
    argv0: &OsStr,
    current_dir: Option<&Path>,
    path_var: Option<&OsStr>,
    prefer_current_dir_file: bool,
) -> Result<PathBuf> {
    let argv0_text = argv0.to_string_lossy().trim().to_owned();
    if argv0_text.is_empty() {
        bail!("current process argv[0] is empty");
    }
    if runtime_path_text_is_absolute(&argv0_text) {
        return Ok(PathBuf::from(normalize_runtime_path_text(&argv0_text)));
    }

    let looks_path_like = argv0_text.contains('/')
        || argv0_text.contains('\\')
        || argv0_text.starts_with('.')
        || argv0_text.starts_with('~');
    if looks_path_like && let Some(current_dir) = current_dir {
        return Ok(normalize_runtime_join_path(current_dir, &argv0_text));
    }

    if prefer_current_dir_file
        && let Some(current_dir) = current_dir
        && let Some(candidate) = runtime_executable_search_candidates(current_dir, &argv0_text)
            .into_iter()
            .find(|candidate| candidate.is_file())
    {
        return Ok(candidate);
    }

    let path_var = path_var.map(OsString::from).unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        for candidate in runtime_executable_search_candidates(&dir, &argv0_text) {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    if let Some(current_dir) = current_dir {
        return Ok(normalize_runtime_join_path(current_dir, &argv0_text));
    }

    bail!("unable to resolve current executable from argv[0]: {argv0_text}");
}

fn runtime_executable_search_candidates(dir: &Path, argv0: &str) -> Vec<PathBuf> {
    let normalized = normalize_runtime_path_text(argv0);
    let mut candidates = vec![normalize_runtime_join_path(dir, &normalized)];
    if runtime_is_windows()
        && Path::new(argv0).extension().is_none()
        && !normalized.to_ascii_lowercase().ends_with(".exe")
    {
        candidates.push(normalize_runtime_join_path(
            dir,
            &format!("{normalized}.exe"),
        ));
    }
    candidates
}

fn normalize_runtime_join_path(base: &Path, child: &str) -> PathBuf {
    let child = normalize_runtime_path_text(child);
    if runtime_path_text_is_absolute(&child) {
        return PathBuf::from(child);
    }
    if runtime_is_windows() && std::path::MAIN_SEPARATOR == '/' {
        let base = normalize_runtime_path_text(&base.display().to_string());
        return PathBuf::from(format!(
            "{}/{}",
            base.trim_end_matches('/'),
            child.trim_start_matches('/')
        ));
    }
    base.join(child)
}

fn normalize_runtime_path_text(value: &str) -> String {
    normalize_runtime_path_text_for_platform(value, RuntimePlatform::current())
}

pub fn normalize_runtime_path_for_host(path: &Path) -> PathBuf {
    PathBuf::from(normalize_runtime_path_text(&path.display().to_string()))
}

pub fn normalize_runtime_path_text_for_host(value: &str) -> String {
    normalize_runtime_path_text(value)
}

/// Resolve `path` to the real location on disk, keeping any trailing components
/// that do not exist yet.
///
/// For a path that is about to be *recorded* rather than merely used. The
/// filesystem happily reaches a folder through a symlink, so an absolute path
/// built by joining onto a linked ancestor names a route rather than a place:
/// correct now, and wrong the moment the link goes. Persisting it — in a manifest,
/// or in the shebang of a console script — outlives whatever made the route valid.
///
/// This is the resolve half of the pipeline the runtime manifest goes through:
/// resolve, then [`normalize_runtime_path_for_storage`], then persist.
///
/// The leaf usually does not exist yet — it is where something is about to be
/// installed — so this walks up to the deepest ancestor that does, canonicalizes
/// that, and re-attaches the rest. Same shape as `disk_space::nearest_existing_ancestor`,
/// except that one answers "which filesystem is this on?" and so discards the
/// walked components, while this needs them back.
///
/// Best-effort by design: a path that cannot be resolved comes back absolutized
/// but otherwise unchanged, which is what the caller would have used anyway.
/// Cases that give up deliberately rather than guess:
///
/// * a `..` in the not-yet-existing tail, where re-attaching after resolving an
///   ancestor could name a different place than the caller meant;
/// * a relative path with no readable current directory.
///
/// On Windows this also strips the verbatim `\\?\` prefix that `canonicalize`
/// returns, since a stored path is later compared against ordinary ones. Windows
/// directory junctions resolve through the same call, so no separate branch.
pub fn resolve_path_through_symlinks(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(path),
            Err(_) => return path.to_path_buf(),
        }
    };

    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut candidate = absolute.as_path();
    loop {
        if candidate.exists() {
            let Ok(resolved) = candidate.canonicalize() else {
                return absolute;
            };
            let mut resolved = crate::disk_space::strip_verbatim_prefix(&resolved);
            for component in tail.iter().rev() {
                resolved.push(component);
            }
            return resolved;
        }
        // `file_name` is None for the root and for a `..` component. Either way
        // there is nothing safe to re-attach, so keep the caller's path.
        let (Some(name), Some(parent)) = (candidate.file_name(), candidate.parent()) else {
            return absolute;
        };
        tail.push(name.to_os_string());
        candidate = parent;
    }
}

pub fn normalize_runtime_path_for_storage(path: &Path) -> PathBuf {
    PathBuf::from(normalize_runtime_path_text_for_storage(
        &path.display().to_string(),
    ))
}

pub fn normalize_runtime_path_text_for_storage(value: &str) -> String {
    if runtime_is_windows() {
        normalize_windows_storage_path_text(value)
    } else {
        value.to_owned()
    }
}

fn normalize_windows_runtime_path_text(value: &str) -> String {
    normalize_windows_runtime_path_text_with_separator(value, RuntimePathSeparator::Native)
}

pub fn normalize_runtime_path_text_for_platform(value: &str, platform: RuntimePlatform) -> String {
    if platform.is_windows() {
        normalize_windows_runtime_path_text(value)
    } else {
        value.to_owned()
    }
}

fn normalize_windows_runtime_path_text_with_separator(
    value: &str,
    separator: RuntimePathSeparator,
) -> String {
    let value = value.trim();
    let forward = value.replace('\\', "/");
    let forward_bytes = forward.as_bytes();
    if forward_bytes.len() >= 2
        && forward_bytes[0] == b'/'
        && forward_bytes[1].is_ascii_alphabetic()
        && (forward_bytes.len() == 2 || forward_bytes[2] == b'/')
    {
        let drive = (forward_bytes[1] as char).to_ascii_uppercase();
        let rest = if forward_bytes.len() > 3 {
            forward[3..].trim_start_matches('/')
        } else {
            ""
        };
        return format_windows_drive_path(drive, rest, separator);
    }
    let bytes = value.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let drive = (bytes[0] as char).to_ascii_uppercase();
        let rest = value[2..].replace('\\', "/");
        let rest = rest.trim_start_matches('/');
        return format_windows_drive_path(drive, rest, separator);
    }
    if let Some(rest) = forward.strip_prefix("//") {
        return format_windows_unc_path(rest, separator);
    }
    match separator.separator() {
        '/' => value.replace('\\', "/"),
        '\\' => value.replace('/', "\\"),
        _ => value.to_owned(),
    }
}

fn normalize_windows_storage_path_text(value: &str) -> String {
    normalize_windows_runtime_path_text_with_separator(value, RuntimePathSeparator::Backslash)
}

fn format_windows_drive_path(drive: char, rest: &str, separator: RuntimePathSeparator) -> String {
    let separator = separator.separator();
    if rest.is_empty() {
        return format!("{drive}:{separator}");
    }
    let rest = match separator {
        '\\' => rest.replace('/', "\\"),
        _ => rest.to_owned(),
    };
    format!("{drive}:{separator}{rest}")
}

fn format_windows_unc_path(rest: &str, separator: RuntimePathSeparator) -> String {
    match separator.separator() {
        '\\' => format!(r"\\{}", rest.replace('/', "\\")),
        _ => format!("//{rest}"),
    }
}

fn runtime_path_text_is_absolute(value: &str) -> bool {
    runtime_path_text_is_absolute_for_platform(value, RuntimePlatform::current())
}

pub fn runtime_path_text_is_absolute_for_host(value: &str) -> bool {
    runtime_path_text_is_absolute(value)
}

pub fn runtime_path_text_is_absolute_for_platform(value: &str, platform: RuntimePlatform) -> bool {
    if platform.is_windows() {
        windows_runtime_path_text_is_absolute(value)
    } else {
        value.trim().starts_with('/')
    }
}

fn windows_runtime_path_text_is_absolute(value: &str) -> bool {
    let normalized = normalize_windows_runtime_path_text(value);
    let bytes = normalized.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\')
    {
        return true;
    }
    let unc = normalized.replace('\\', "/");
    if !unc.starts_with("//") {
        return false;
    }
    let mut parts = unc.split('/').filter(|part| !part.is_empty());
    parts.next().is_some() && parts.next().is_some()
}

pub fn platform_binary_name(binary_name: &str) -> String {
    format!("{binary_name}{}", runtime_exe_suffix())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch_dir(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rocm-core-resolve-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::remove_dir_all(&root).ok();
        fs::create_dir_all(&root).unwrap();
        // Resolved up front, so an assertion cannot turn on whether the platform's
        // temp dir is itself reached through a link.
        //
        // Deliberately the function under test rather than a bare `canonicalize`:
        // on Windows `canonicalize` hands back a verbatim `\\?\C:\…` path, which
        // would make every expectation here verbatim while the function correctly
        // returns a plain one — the tests would then fail on the prefix rather than
        // on the behaviour they are about. That the prefix really is stripped is
        // asserted separately, in `resolving_never_yields_a_windows_verbatim_prefix`.
        resolve_path_through_symlinks(&root)
    }

    #[test]
    fn resolving_keeps_components_that_do_not_exist_yet() {
        // The common case: the leaf is where something is about to be installed.
        let root = scratch_dir("missing-leaf");
        assert_eq!(
            resolve_path_through_symlinks(&root.join("a").join("b").join("c")),
            root.join("a").join("b").join("c")
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn resolving_follows_a_linked_ancestor_and_reattaches_the_rest() {
        let root = scratch_dir("linked-ancestor");
        let real = root.join("real");
        fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.join("link")).unwrap();

        assert_eq!(
            resolve_path_through_symlinks(&root.join("link").join("wheel").join("key")),
            real.join("wheel").join("key")
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn resolving_leaves_a_dangling_link_alone() {
        // Nothing to resolve to, so the caller's path is the best answer available.
        let root = scratch_dir("dangling");
        std::os::unix::fs::symlink(root.join("nowhere"), root.join("link")).unwrap();
        let requested = root.join("link").join("child");

        assert_eq!(resolve_path_through_symlinks(&requested), requested);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn resolving_is_idempotent() {
        let root = scratch_dir("idempotent");
        let real = root.join("real");
        fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.join("link")).unwrap();
        let once = resolve_path_through_symlinks(&root.join("link").join("leaf"));

        assert_eq!(resolve_path_through_symlinks(&once), once);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolving_gives_up_rather_than_guess_at_a_parent_component() {
        // Re-attaching `..` after resolving an ancestor could name a different
        // place than the caller meant, so the path comes back untouched.
        let root = scratch_dir("parent-component");
        let requested = root.join("missing").join("..").join("sibling");

        assert_eq!(resolve_path_through_symlinks(&requested), requested);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolving_never_yields_a_windows_verbatim_prefix() {
        // A stored path is later compared against ordinary ones, and `\\?\C:\…`
        // never `starts_with`-matches `C:\…`. Trivially true off Windows; the
        // prefix stripping itself is covered in `disk_space`.
        let root = scratch_dir("verbatim");
        let resolved = resolve_path_through_symlinks(&root.join("leaf"));

        assert!(
            !resolved.to_string_lossy().starts_with(r"\\?\"),
            "{}",
            resolved.display()
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn platform_binary_name_follows_runtime_host() {
        let name = platform_binary_name("rocm");
        if runtime_is_windows() {
            assert_eq!(name, "rocm.exe");
        } else {
            assert_eq!(name, "rocm");
        }
    }

    #[test]
    fn runtime_windows_paths_accept_mixed_drive_separators() {
        if !runtime_is_windows() {
            return;
        }

        let path = normalize_windows_runtime_path_text(r"D:\/path/to/therock_venvs");

        assert!(windows_runtime_path_text_is_absolute(&path));
        if std::path::MAIN_SEPARATOR == '\\' {
            assert_eq!(path, r"D:\path\to\therock_venvs");
        } else {
            assert_eq!(path, "D:/path/to/therock_venvs");
        }
    }

    #[test]
    fn runtime_windows_paths_accept_universal_drive_prefixes() {
        if !runtime_is_windows() {
            return;
        }

        let path = normalize_windows_runtime_path_text("/D/path/to/therock_venvs");

        assert!(windows_runtime_path_text_is_absolute(&path));
        if std::path::MAIN_SEPARATOR == '\\' {
            assert_eq!(path, r"D:\path\to\therock_venvs");
        } else {
            assert_eq!(path, "D:/path/to/therock_venvs");
        }
    }

    #[test]
    fn runtime_windows_storage_paths_use_native_drive_syntax() {
        if !runtime_is_windows() {
            return;
        }

        assert_eq!(
            normalize_runtime_path_text_for_storage("/D/path/to/therock_venvs"),
            r"D:\path\to\therock_venvs"
        );
        assert_eq!(
            normalize_runtime_path_text_for_storage("D:/path/to/therock_venvs"),
            r"D:\path\to\therock_venvs"
        );
    }

    #[test]
    fn runtime_windows_paths_normalize_relative_backslashes_for_unix_separator_runtime() {
        if !runtime_is_windows() || std::path::MAIN_SEPARATOR != '/' {
            return;
        }

        assert_eq!(
            normalize_windows_runtime_path_text(r".\rocm.exe"),
            "./rocm.exe"
        );
    }

    #[test]
    fn runtime_path_normalization_accepts_windows_drive_forms() {
        let cases = [
            (r"D:\path\to\therock_venvs", "D:/path/to/therock_venvs"),
            ("D:/path/to/therock_venvs", "D:/path/to/therock_venvs"),
            (r"D:\/path/to/therock_venvs", "D:/path/to/therock_venvs"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_windows_runtime_path_text_with_separator(
                    input,
                    RuntimePathSeparator::Slash
                ),
                expected
            );
            assert!(runtime_path_text_is_absolute_for_platform(
                input,
                RuntimePlatform::Windows
            ));
        }
    }

    #[test]
    fn runtime_path_normalization_accepts_windows_unc_forms() {
        assert_eq!(
            normalize_windows_runtime_path_text_with_separator(
                r"\\server\share\rocm",
                RuntimePathSeparator::Slash
            ),
            "//server/share/rocm"
        );
        assert_eq!(
            normalize_windows_runtime_path_text_with_separator(
                "//server/share/rocm",
                RuntimePathSeparator::Backslash
            ),
            r"\\server\share\rocm"
        );
        assert!(runtime_path_text_is_absolute_for_platform(
            r"\\server\share\rocm",
            RuntimePlatform::Windows
        ));
    }

    #[test]
    fn managed_runtime_data_root_strips_runtime_leaf_nesting() {
        let data_root = PathBuf::from("/tmp/rocm-cli");
        let release_root = data_root
            .join("runtimes")
            .join("wheel")
            .join("release-wheel-gfx942-7-0");
        let nested_root = release_root
            .join("runtimes")
            .join("wheel")
            .join("nightly-wheel-gfx942-7-1");

        assert_eq!(managed_runtime_data_root(&release_root), data_root);
        assert_eq!(managed_runtime_data_root(&nested_root), data_root);
    }

    #[test]
    fn managed_runtime_data_root_preserves_custom_prefix() {
        let custom_prefix = PathBuf::from("/tmp/therock_venvs");

        assert_eq!(managed_runtime_data_root(&custom_prefix), custom_prefix);
    }

    #[test]
    fn runtime_path_normalization_keeps_wsl_paths_linux_native() {
        let wsl_path = "/mnt/d/path/to/therock_venvs";
        assert_eq!(
            normalize_runtime_path_text_for_platform(wsl_path, RuntimePlatform::Linux),
            wsl_path
        );
        assert!(runtime_path_text_is_absolute_for_platform(
            wsl_path,
            RuntimePlatform::Linux
        ));
    }

    #[test]
    fn runtime_path_normalization_does_not_treat_windows_drive_as_linux_absolute() {
        let windows_path = r"D:\path\to\therock_venvs";
        assert_eq!(
            normalize_runtime_path_text_for_platform(windows_path, RuntimePlatform::Linux),
            windows_path
        );
        assert!(!runtime_path_text_is_absolute_for_platform(
            windows_path,
            RuntimePlatform::Linux
        ));
    }

    #[test]
    fn argv0_resolution_uses_path_by_default() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-current-exe-path-resolution-{}",
            crate::unix_time_millis()
        ));
        let current_dir = root.join("cwd");
        let path_dir = root.join("bin");
        fs::create_dir_all(&current_dir)?;
        fs::create_dir_all(&path_dir)?;
        let local_binary = current_dir.join("install");
        let path_binary = path_dir.join("install");
        fs::write(&local_binary, b"local")?;
        fs::write(&path_binary, b"path")?;
        let path_var = std::env::join_paths([path_dir.as_os_str()])?;

        let resolved = current_executable_path_from_argv0_value(
            std::ffi::OsStr::new("install"),
            Some(&current_dir),
            Some(path_var.as_os_str()),
            false,
        )?;

        assert_eq!(resolved, path_binary);
        fs::remove_dir_all(root).ok();
        Ok(())
    }
}
