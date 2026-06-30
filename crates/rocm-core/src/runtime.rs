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

pub fn managed_runtime_cache_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("cache")
}

pub fn managed_pip_cache_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("pip-cache")
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
