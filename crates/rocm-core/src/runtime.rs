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

/// The loader search-path variable used to expose a runtime's ROCm libraries to
/// a child process.
pub const RUNTIME_LIBRARY_PATH_ENV: &str = if cfg!(windows) {
    "PATH"
} else {
    "LD_LIBRARY_PATH"
};

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

/// `uv`'s content-addressed cache, kept under the managed root (see issue #160).
///
/// Colocating it keeps the cache reachable from the environments `uv` populates without
/// crossing a mount point, which is what lets `uv` hardlink into them instead of copying.
/// It is the mount, not the filesystem: a bind mount or `subPath` volume is enough to make
/// Linux refuse the hardlink and send `uv` back to copying.
pub fn managed_uv_cache_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("uv-cache")
}

pub fn managed_logs_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("logs")
}

pub fn managed_tools_dir(root: &Path) -> PathBuf {
    normalize_runtime_path_for_host(root).join("tools")
}

/// Split a search-path list into entries, host-normalising each one.
///
/// Safe on an inherited OS `PATH` as well as on a list this tool recorded
/// itself: the Windows branch implements the same quoting rules as
/// [`std::env::split_paths`] before it normalises. It adds trimming and the
/// dropping of empty entries on top, which a recorded list wants and an
/// inherited one does not mind.
///
/// That quote handling is load-bearing rather than incidental. `c:\some;dir` is
/// a legal Windows path, so a list containing one has to quote it, and a naive
/// `split(';')` both tears the entry in two and leaves the `"` characters in
/// the result — where [`std::env::join_paths`] rejects them outright and the
/// caller loses the whole list, not the one bad entry. Keep this in step with
/// [`runtime_path_list_join`], which re-quotes on the way back out.
pub fn runtime_path_list_split(value: &OsStr) -> Vec<PathBuf> {
    if !runtime_is_windows() {
        return std::env::split_paths(value).collect();
    }
    split_windows_path_list_text(&value.to_string_lossy())
        .iter()
        .map(|entry| normalize_runtime_path_for_host(Path::new(entry)))
        .collect()
}

/// Split a Windows `;`-separated path list, honouring the quoting rules
/// [`std::env::split_paths`] uses: a `"` opens a run in which `;` is an ordinary
/// character, and the quotes are removed rather than kept in the entry.
///
/// Entries are trimmed and empty ones dropped, so a trailing separator or a
/// stray `;;` does not yield a path that resolves to the current directory.
///
/// Free of any host dependency so the Windows shape stays testable on Linux.
fn split_windows_path_list_text(value: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for ch in value.chars() {
        match ch {
            '"' => quoted = !quoted,
            ';' if !quoted => {
                entries.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    entries.push(current);
    entries
        .into_iter()
        .map(|entry| entry.trim().to_owned())
        .filter(|entry| !entry.is_empty())
        .collect()
}

/// Render one entry for a Windows path list, quoting it when it contains the
/// `;` separator so that [`runtime_path_list_split`] can recover it whole.
///
/// A `"` in the entry goes out raw, where [`std::env::join_paths`] rejects the
/// list outright. That rests on a precondition rather than on a check, so name
/// it: `"` is reserved in a Windows path, and the splitter above consumes quotes
/// rather than emitting them, so neither a path from disk nor an entry recovered
/// from a list can hold one. Rejecting the character would trade that
/// unreachable case for the failure this pair exists to avoid -- one bad entry
/// costing the caller every entry.
fn windows_path_list_entry_text(path: &Path) -> String {
    let text = runtime_path_for_windows_child(path);
    if text.contains(';') {
        return format!("\"{text}\"");
    }
    text
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
            .map(|entry| windows_path_list_entry_text(entry))
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

/// Split a forward-slash path into the prefix that `..` must never climb past
/// and the components that follow it.
///
/// An absolute path's root, a Windows drive, and a UNC share are all anchors:
/// `/..` is `/` on every POSIX system, and no amount of `..` leaves `C:\`. A
/// relative path has no anchor, so a leading `..` there is meaningful and is
/// kept.
fn split_runtime_path_anchor(value: &str, platform: RuntimePlatform) -> (&str, &str) {
    // A drive letter and a UNC share are anchors only where they mean anything.
    // On Linux `C:/x` is an ordinary relative name and `//etc` is just `/etc`,
    // so reading either as a root would invent a path that is not there.
    //
    // The platform is a parameter rather than `runtime_is_windows()` so that the
    // Windows side of a guard against recursive deletion can be tested from a
    // Linux host, which is the only host this repository runs clippy and most
    // of its unit tests on.
    if platform.is_windows() {
        if let Some(rest) = value.strip_prefix("//") {
            // `//server/share/...`: the share itself is the anchor.
            let share_end = rest
                .match_indices('/')
                .nth(1)
                .map_or(rest.len(), |(index, _)| index);
            let (anchor_tail, rest) = rest.split_at(share_end);
            return (&value[..2 + anchor_tail.len()], rest);
        }
        let bytes = value.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            let drive_end = if bytes.len() > 2 && bytes[2] == b'/' {
                3
            } else {
                2
            };
            return value.split_at(drive_end);
        }
    }
    if value.starts_with('/') {
        return value.split_at(1);
    }
    ("", value)
}

/// Collapse `.`, `..`, repeated separators and a trailing separator, so that
/// every spelling of one folder reduces to one string.
///
/// Purely lexical, and deliberately so: this feeds the comparisons that decide
/// whether a folder may be recursively deleted, and those run against paths
/// that often do not exist yet (or exist only in a registry entry), where
/// `canonicalize` has nothing to resolve.
///
/// This is NOT symlink-safe, and the gap is not merely theoretical. Because
/// [`runtime_install_root_is_protected`] exempts anything inside the user's
/// home before it consults the protected-root list, a symlink the user owns is
/// enough: with `$HOME/link -> /etc`, the path `$HOME/link/child` is lexically
/// inside home, so the guard reports it removable while the kernel lands in
/// `/etc`. Resolving lexically is strictly better than the raw-text comparison
/// it replaces — it closes every *spelling* of a protected path — but it does
/// not close that hole, and closing it needs a decision about canonicalizing
/// the part of the path that does exist, which is a separate change.
fn lexically_resolved_runtime_path_text(value: &str, platform: RuntimePlatform) -> String {
    let (anchor, rest) = split_runtime_path_anchor(value, platform);
    let anchored = !anchor.is_empty();
    let mut parts: Vec<&str> = Vec::new();
    for part in rest.split('/') {
        match part {
            "" | "." => {}
            ".." => match parts.last() {
                // A relative path keeps the `..` it cannot resolve: `../a` names
                // a real place, just not one this function can name differently.
                None if !anchored => parts.push(".."),
                Some(&"..") => parts.push(".."),
                // Anchored and already at the top: `/..` is `/`.
                None => {}
                Some(_) => {
                    parts.pop();
                }
            },
            other => parts.push(other),
        }
    }
    let joined = parts.join("/");
    if anchored {
        format!("{}{joined}", anchor.trim_end_matches('/').to_owned() + "/")
    } else if joined.is_empty() {
        ".".to_owned()
    } else {
        joined
    }
}

/// Reduce a path to the single text that names its folder, whatever spelling it
/// arrived in.
fn comparable_runtime_path_text(path: &Path, platform: RuntimePlatform) -> String {
    let text = normalize_runtime_path_text_for_platform(&path.display().to_string(), platform);
    // Folding `\` to `/` is a WINDOWS rule and must stay gated on the platform.
    // Off Windows a backslash is an ordinary filename byte, so a directory
    // genuinely named `..\tmp` inside `/usr` would otherwise be rewritten to
    // `/usr/../tmp` and the `..` resolution below would walk it straight out of
    // the protected root — turning a guard into a bypass. Raw-text comparison
    // tolerated the unconditional fold because equality never *removed*
    // components; resolving `..` does.
    let text = if platform.is_windows() {
        text.replace('\\', "/")
    } else {
        text
    };
    lexically_resolved_runtime_path_text(&text, platform)
}

pub fn runtime_paths_equivalent(left: &Path, right: &Path) -> bool {
    runtime_paths_equivalent_on(left, right, RuntimePlatform::current())
}

fn runtime_paths_equivalent_on(left: &Path, right: &Path, platform: RuntimePlatform) -> bool {
    let left = comparable_runtime_path_text(left, platform);
    let right = comparable_runtime_path_text(right, platform);
    if platform.is_windows() {
        left.eq_ignore_ascii_case(&right)
    } else {
        left == right
    }
}

pub fn runtime_path_is_same_or_inside(path: &Path, base: &Path) -> bool {
    runtime_path_is_same_or_inside_on(path, base, RuntimePlatform::current())
}

fn runtime_path_is_same_or_inside_on(path: &Path, base: &Path, platform: RuntimePlatform) -> bool {
    // Compared after resolution rather than by walking `Path::ancestors`, which
    // treats `..` as an ordinary component and so counts a path that climbs
    // back OUT of `base` as still inside it.
    let path = comparable_runtime_path_text(path, platform);
    let base = comparable_runtime_path_text(base, platform);
    let inside_prefix = format!("{}/", base.trim_end_matches('/'));
    if path.len() <= inside_prefix.len() {
        return runtime_paths_equivalent_on(Path::new(&path), Path::new(&base), platform);
    }
    // Compared as bytes: a path may hold any UTF-8, and slicing a `str` at a
    // byte offset that lands mid-character panics.
    let head = &path.as_bytes()[..inside_prefix.len()];
    if platform.is_windows() {
        head.eq_ignore_ascii_case(inside_prefix.as_bytes())
    } else {
        head == inside_prefix.as_bytes()
    }
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

/// Why a folder must never be recursively deleted by ROCm CLI.
///
/// Returned by [`runtime_protected_location`] so a caller that refuses can say
/// *why* from the same value it decided on, instead of restating the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectedLocation {
    /// The top of a filesystem: `/`, or a drive root such as `C:\` on Windows.
    FilesystemRoot,
    /// The user's home folder itself.
    Home,
    /// A folder that contains the user's home folder, such as `/home`.
    ContainsHome,
    /// A system location such as `/usr` or `C:\Windows`, or a folder inside one.
    System,
    /// A real directory whose location could not be resolved, so it cannot be
    /// shown not to be one of the above. Only `rocm uninstall` reports this.
    Unresolvable,
}

impl ProtectedLocation {
    /// What the folder is, phrased to follow "it is".
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::FilesystemRoot => "the top of the filesystem",
            Self::Home => "your home folder",
            Self::ContainsHome => "a folder that contains your home folder",
            Self::System => "a protected system location",
            Self::Unresolvable => "a folder whose real location could not be resolved",
        }
    }
}

/// Whether ROCm CLI must refuse to recursively delete the runtime folder `path`.
///
/// The predicate `rocm runtimes uninstall`, `rocm storage remove-old-installs`
/// and the local assistant's install-folder check share. See
/// [`runtime_protected_location`]; `rocm uninstall`'s own roots use the
/// narrower [`uninstall_root_protected_location_for_home`].
pub fn runtime_install_root_is_protected(path: &Path) -> bool {
    runtime_protected_location(path).is_some()
}

/// Why `path` is protected, if it is, judged against the current user's home.
pub fn runtime_protected_location(path: &Path) -> Option<ProtectedLocation> {
    runtime_protected_location_for_home(path, runtime_home_dir().as_deref())
}

/// [`runtime_protected_location`] with the home folder passed in, so a caller
/// (or a test) can judge a path against a home other than the process's own.
///
/// Anything strictly inside `home` is never protected, so a user install under
/// `~/.rocm` stays removable even when home itself sits under a protected root
/// (`/root` for a root user). Home itself, and every folder that contains it,
/// is protected: removing either deletes everything the user owns. A `home`
/// that is itself a filesystem root grants no exemption, or it would exempt
/// every folder on the machine. Anything else inside a system location is
/// protected too.
pub fn runtime_protected_location_for_home(
    path: &Path,
    home: Option<&Path>,
) -> Option<ProtectedLocation> {
    protected_location_on(
        path,
        home,
        RuntimePlatform::current(),
        SystemRule::AndInside,
    )
}

/// Why `rocm uninstall` must refuse to remove its config, data or cache root
/// `path`, if it must, judged against `home`.
///
/// The same rule as [`runtime_protected_location_for_home`] except for system
/// locations: only the location ITSELF (`/var`, `/opt`, `C:\Windows`, ...) is
/// refused, not a folder inside one. A dedicated `/var/cache/rocm-cli` or
/// `/opt/rocm-cli` is a normal place to point these roots, and removing it is
/// exactly what the user asked for — the same reason a folder inside home is
/// allowed.
pub fn uninstall_root_protected_location_for_home(
    path: &Path,
    home: Option<&Path>,
) -> Option<ProtectedLocation> {
    protected_location_on(path, home, RuntimePlatform::current(), SystemRule::Itself)
}

/// How a system location protects the folders under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystemRule {
    /// The location and everything inside it.
    AndInside,
    /// Only the location itself.
    Itself,
}

const fn system_roots(platform: RuntimePlatform) -> &'static [&'static str] {
    if platform.is_windows() {
        &["C:/Windows", "C:/Program Files", "C:/Program Files (x86)"]
    } else {
        &[
            "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/opt", "/proc", "/root", "/sbin",
            "/sys", "/usr", "/var",
        ]
    }
}

fn protected_location_on(
    path: &Path,
    home: Option<&Path>,
    platform: RuntimePlatform,
    system_rule: SystemRule,
) -> Option<ProtectedLocation> {
    if is_filesystem_root_on(path, platform) {
        return Some(ProtectedLocation::FilesystemRoot);
    }

    if let Some(home) = home.filter(|home| !is_filesystem_root_on(home, platform)) {
        if runtime_paths_equivalent_on(path, home, platform) {
            return Some(ProtectedLocation::Home);
        }
        if runtime_path_is_same_or_inside_on(path, home, platform) {
            return None;
        }
        if runtime_path_is_same_or_inside_on(home, path, platform) {
            return Some(ProtectedLocation::ContainsHome);
        }
    }

    system_roots(platform)
        .iter()
        .map(Path::new)
        .any(|root| match system_rule {
            SystemRule::AndInside => runtime_path_is_same_or_inside_on(path, root, platform),
            SystemRule::Itself => runtime_paths_equivalent_on(path, root, platform),
        })
        .then_some(ProtectedLocation::System)
}

/// `/`, a Windows drive root such as `C:/`, or a UNC share root: anchored,
/// with nothing above it once `.`, `..` and repeated separators are resolved
/// (`//..` is `/`).
fn is_filesystem_root_on(path: &Path, platform: RuntimePlatform) -> bool {
    let resolved = comparable_runtime_path_text(path, platform);
    let (anchor, rest) = split_runtime_path_anchor(&resolved, platform);
    !anchor.is_empty() && rest.trim_matches('/').is_empty()
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
/// `fs::canonicalize`, minus the verbatim `\\?\` prefix it adds on Windows, so
/// the result compares with ordinary paths. For a path that exists; unlike
/// [`resolve_path_through_symlinks`] it resolves `..` through the filesystem.
pub fn canonicalize_for_compare(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize()
        .map(|resolved| crate::disk_space::strip_verbatim_prefix(&resolved))
}

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
        // Checked BEFORE `exists`, because the two platforms disagree about what a
        // `..` past a missing directory even means. Windows collapses it lexically,
        // so `<root>/missing/..` "exists" and canonicalizes to `<root>`; Unix walks
        // the path through the filesystem, so it does not exist at all. Resolving
        // here would therefore record a different folder depending on the host —
        // and the whole point of this function is that the folder it returns is the
        // one the files land in. Give up instead.
        if candidate.components().next_back() == Some(std::path::Component::ParentDir) {
            return absolute;
        }
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
        // `file_name` is None at the root. Nothing left to walk up to, so keep the
        // caller's path.
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
        // A `..` reached past a missing directory means different things on
        // different hosts: Windows collapses it lexically, so `<root>/missing/..`
        // resolves to `<root>`, while Unix walks the filesystem and finds nothing.
        // Resolving it would record a different folder depending on the host, so
        // the path comes back untouched on both.
        let root = scratch_dir("parent-component");
        let requested = root.join("missing").join("..").join("sibling");

        assert_eq!(resolve_path_through_symlinks(&requested), requested);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolving_still_handles_a_parent_component_the_filesystem_can_walk() {
        // The give-up above must not spread to a `..` whose parent is really
        // there — that one is unambiguous on every host, and refusing it would
        // leave an ordinary path unresolved.
        let root = scratch_dir("parent-that-exists");
        let nested = root.join("a").join("b");
        fs::create_dir_all(&nested).unwrap();
        let requested = nested.join("..").join("b").join("leaf");

        assert_eq!(
            resolve_path_through_symlinks(&requested),
            nested.join("leaf")
        );
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

    /// The Windows path-list tests below drive the text helpers directly rather
    /// than `runtime_path_list_split`, which takes the `std::env::split_paths`
    /// branch on a Linux host. The helpers carry the whole Windows shape and no
    /// host dependency, so the behaviour is pinned on every lane; what a Windows
    /// lane adds is that `runtime_path_list_split` really routes into them.
    #[test]
    fn a_quoted_windows_path_entry_survives_the_separator_inside_it() {
        // `c:\some;dir` is a legal Windows path, so a list carrying one quotes
        // it. Splitting on every `;` would tear it in two and invent a `dir`
        // entry relative to wherever the child happens to start.
        assert_eq!(
            split_windows_path_list_text(r#"C:\rocm\bin;"C:\some;dir";C:\windows"#),
            vec![
                r"C:\rocm\bin".to_owned(),
                r"C:\some;dir".to_owned(),
                r"C:\windows".to_owned(),
            ]
        );
    }

    #[test]
    fn splitting_a_windows_path_list_strips_the_quotes_it_split_on() {
        // The regression this guards: quotes left in an entry make
        // `std::env::join_paths` fail on Windows, and the caller composing a
        // loader path then loses every entry rather than the one bad one.
        let entries = split_windows_path_list_text(r#""C:\quoted\bin";C:\plain\bin"#);
        assert_eq!(
            entries,
            vec![r"C:\quoted\bin".to_owned(), r"C:\plain\bin".to_owned()]
        );
        assert!(
            entries.iter().all(|entry| !entry.contains('"')),
            "a quote reaching join_paths costs the caller the whole list: {entries:?}"
        );
    }

    #[test]
    fn splitting_a_windows_path_list_trims_and_drops_empty_entries() {
        // A trailing separator or a stray `;;` must not yield an entry that
        // resolves against the child's current directory.
        assert_eq!(
            split_windows_path_list_text(r"C:\rocm\bin; ;;  C:\windows  ;"),
            vec![r"C:\rocm\bin".to_owned(), r"C:\windows".to_owned()]
        );
    }

    #[test]
    fn joining_a_windows_path_list_requotes_an_entry_holding_the_separator() {
        // The other half of the same rule: an entry that goes out unquoted comes
        // back as two, so join has to restore what split consumed.
        assert_eq!(
            windows_path_list_entry_text(Path::new(r"C:\some;dir")),
            r#""C:\some;dir""#
        );
        assert_eq!(
            windows_path_list_entry_text(Path::new(r"C:\rocm\bin")),
            r"C:\rocm\bin",
            "an ordinary entry must not gain quotes it never had"
        );
    }

    #[test]
    fn a_windows_path_list_round_trips_through_split_and_join() {
        let entries = [r"C:\rocm\bin", r"C:\some;dir", r"C:\windows"];
        let joined = entries
            .iter()
            .map(|entry| windows_path_list_entry_text(Path::new(entry)))
            .collect::<Vec<_>>()
            .join(";");

        assert_eq!(split_windows_path_list_text(&joined), entries.to_vec());
    }

    /// The Windows side of the same comparisons, exercised from whatever host
    /// runs the tests. `runtime_is_windows()` is decided at compile time, so
    /// without a platform parameter this branch would be checked only by the
    /// one CI lane that runs on Windows — and it is the branch that decides
    /// whether `C:\Windows` may be recursively deleted.
    #[test]
    fn windows_containment_sees_through_spelling_and_case() {
        let windows = RuntimePlatform::Windows;
        let system_root = Path::new("C:/Windows");

        for inside in [
            r"C:\Windows",
            "C:/Windows/",
            "c:/windows",
            "C:/Windows/./System32",
            "C:/Program Files/../Windows/System32",
            "C:/Windows//System32",
        ] {
            assert!(
                runtime_path_is_same_or_inside_on(Path::new(inside), system_root, windows),
                "{inside} names C:/Windows or something under it"
            );
        }

        for outside in [
            "C:/Users/dev/.rocm",
            "C:/Windows/../Users/dev",
            "C:/WindowsApps",
            "D:/Windows",
        ] {
            assert!(
                !runtime_path_is_same_or_inside_on(Path::new(outside), system_root, windows),
                "{outside} is not inside C:/Windows"
            );
        }

        // A drive is an anchor: `..` can never climb off it onto another one.
        // (The resolver is handed forward slashes; the conversion happens in
        // `comparable_runtime_path_text`, which the assertions above go through.)
        assert_eq!(
            lexically_resolved_runtime_path_text("C:/../../Windows", windows),
            "C:/Windows"
        );
        // A UNC share is an anchor too.
        assert_eq!(
            lexically_resolved_runtime_path_text("//server/share/../../rocm", windows),
            "//server/share/rocm"
        );
    }

    /// The same text means different things on the two platforms, so the
    /// resolution must not borrow Windows' reading on Linux: `//etc` is `/etc`
    /// there, and `C:/x` is an ordinary relative name, not a drive.
    #[test]
    fn linux_resolution_does_not_borrow_windows_anchors() {
        let linux = RuntimePlatform::Linux;

        assert_eq!(lexically_resolved_runtime_path_text("//etc", linux), "/etc");
        assert_eq!(
            lexically_resolved_runtime_path_text("/etc/./../etc/", linux),
            "/etc"
        );
        // `/..` is `/` on every POSIX system.
        assert_eq!(lexically_resolved_runtime_path_text("/../..", linux), "/");
        // Relative: a leading `..` names a real place this cannot rename.
        assert_eq!(
            lexically_resolved_runtime_path_text("../a/../b", linux),
            "../b"
        );
        assert_eq!(lexically_resolved_runtime_path_text("a/..", linux), ".");
        assert_eq!(
            lexically_resolved_runtime_path_text("C:/x", linux),
            "C:/x",
            "a drive letter is not a root on Linux"
        );
    }

    /// `windows_containment_sees_through_spelling_and_case` above exercises the
    /// Windows path-resolution rules from a Linux host, but only through the
    /// platform-parameterised helpers. The public gate itself,
    /// [`runtime_install_root_is_protected`], still decides which protected-root
    /// list to consult by reading [`runtime_is_windows()`] — a compile-time
    /// answer — so its Windows arm is reachable only when this binary is
    /// actually running on Windows. Not `#[cfg(unix)]`-gated, so it compiles
    /// and runs on every lane: it follows the same `cfg!(windows)`-at-runtime
    /// shape `ensure_runtime_install_root_rejects_protected_system_path` (in
    /// `apps/rocm`) already uses to pick host-appropriate fixtures, so a lane
    /// that is not Windows still exercises the gate end-to-end against the
    /// Unix answer it already owns.
    #[test]
    fn the_public_gate_refuses_every_spelling_of_a_protected_root_on_its_own_host() {
        let spellings: Vec<PathBuf> = if cfg!(windows) {
            vec![
                PathBuf::from("C:/Windows/"),            // trailing separator
                PathBuf::from("c:/windows"),             // mixed case
                PathBuf::from("C:/Windows//System32"),   // doubled separator
                PathBuf::from("C:/Program Files//"),     // doubled separator
                PathBuf::from("C:/PROGRAM FILES (X86)"), // mixed case
            ]
        } else {
            vec![PathBuf::from("/etc/"), PathBuf::from("//etc")]
        };

        let accepted: Vec<String> = spellings
            .into_iter()
            .filter(|path| !runtime_install_root_is_protected(path))
            .map(|path| path.display().to_string())
            .collect();

        assert!(accepted.is_empty(), "accepted as removable: {accepted:?}");
    }

    // ── Properties: the recursive-delete guard ─────────────────────
    //
    // `runtime_install_root_is_protected` is the single source of truth for
    // "may ROCm CLI `remove_dir_all` this folder?". Everything below states a
    // contract it must satisfy for EVERY spelling of a path, because the whole
    // point of the guard is that it is handed a path somebody (or something)
    // else wrote down — a registry entry, an `--prefix` argument, a tool call
    // from the local assistant. Such a path is text, and text has many
    // spellings for one folder.
    //
    // Unix-only: the protected-root list the guard consults is the Unix one,
    // and `runtime_is_windows()` is decided at compile time, so the Windows
    // branch cannot be exercised from here. All pure and in-process.
    #[cfg(unix)]
    mod delete_guard_properties {
        use super::*;
        use proptest::prelude::*;

        /// The Unix roots `runtime_install_root_is_protected` refuses, restated
        /// here so a property compares the guard against the policy rather than
        /// against itself.
        const PROTECTED_ROOTS: [&str; 13] = [
            "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/opt", "/proc", "/root", "/sbin",
            "/sys", "/usr", "/var",
        ];

        /// Resolve `.`, `..`, repeated and trailing separators lexically — what
        /// `realpath --no-symlinks`, every shell, and `Path::components` on
        /// Windows all do, and what the kernel does for a path with no symlinks
        /// in it.
        ///
        /// This is the reference the guard is measured against, and it is
        /// derived from POSIX semantics rather than from the implementation on
        /// purpose: an oracle restated from the code under test shares that
        /// code's mistakes and proves nothing. Keep it that way — note it does
        /// NOT treat `\` specially, which is exactly what catches an
        /// unconditional backslash fold leaking into `..` resolution.
        ///
        /// It says nothing about symlinks. The guard is not symlink-safe (see
        /// `lexically_resolved_runtime_path_text`); this reference only pins
        /// that every *spelling* of one folder resolves alike.
        ///
        /// Absolute input only: every generator seed in this module
        /// (`PROTECTED_ROOTS`, `prefix()`, `home_text()`) is already rooted at
        /// `/`, so a relative path never reaches this oracle today. That is a
        /// property of the generators, not of this function's logic, so it is
        /// asserted rather than merely assumed — a generator change that starts
        /// seeding relative text would otherwise desync the oracle from
        /// `runtime_install_root_is_protected` (which does handle relative
        /// paths) without a single property failing to say so.
        fn lexically_resolved(path: &str) -> String {
            assert!(
                path.starts_with('/'),
                "lexically_resolved is a POSIX-absolute-path oracle; got relative input {path:?}"
            );
            let mut parts: Vec<&str> = Vec::new();
            for part in path.split('/') {
                match part {
                    "" | "." => {}
                    // POSIX: `/..` is `/`, so popping an empty stack is a no-op.
                    ".." => {
                        parts.pop();
                    }
                    other => parts.push(other),
                }
            }
            format!("/{}", parts.join("/"))
        }

        /// Does `path` really name `base` or something under it, once both are
        /// resolved?
        fn lexically_same_or_inside(path: &str, base: &str) -> bool {
            let path = lexically_resolved(path);
            let base = lexically_resolved(base);
            path == base || path.starts_with(&format!("{}/", base.trim_end_matches('/')))
        }

        /// Does `path` really resolve to a protected system location?
        fn lexically_protected(path: &str) -> bool {
            let resolved = lexically_resolved(path);
            resolved == "/"
                || PROTECTED_ROOTS
                    .iter()
                    .any(|root| lexically_same_or_inside(&resolved, root))
        }

        fn home_text() -> Option<String> {
            runtime_home_dir().map(|home| home.display().to_string())
        }

        /// The guard deliberately exempts anything STRICTLY inside the user's
        /// own home directory, so a user install under `~/.rocm` stays
        /// removable even when home itself sits under a protected root (`/root`
        /// for a root user). A property about protection has to grant the same
        /// exemption, or it would just be re-litigating that decision.
        fn exempt_as_user_owned(path: &str) -> bool {
            home_text().is_some_and(|home| {
                lexically_same_or_inside(path, &home)
                    && lexically_resolved(path) != lexically_resolved(&home)
            })
        }

        /// The spellings [`a_protected_location_is_refused_however_it_is_spelled`]
        /// shrank to, pinned as examples so each stays named even if the
        /// generator is retuned. Every one of these names `/etc` (or `/`), and
        /// every one of them is a plausible way for a path to be written down
        /// by hand, assembled by a script, or produced by joining.
        #[test]
        fn the_delete_guard_refuses_every_spelling_of_a_protected_root() {
            let home = runtime_home_dir().expect("a home directory");
            let home = home.display().to_string();
            let spellings = [
                "/etc".to_owned(),
                "/etc/".to_owned(),
                "//etc".to_owned(),
                "/./etc".to_owned(),
                "/etc/.".to_owned(),
                "/etc//".to_owned(),
                "/usr/../etc".to_owned(),
                "/etc/..".to_owned(),
                format!("{home}/../../etc"),
                format!("{home}/.rocm/../../../etc"),
            ];
            let reported: Vec<(String, bool)> = spellings
                .into_iter()
                .map(|text| {
                    let protected = runtime_install_root_is_protected(Path::new(&text));
                    (text, protected)
                })
                .collect();
            let removable: Vec<&str> = reported
                .iter()
                .filter(|(_, protected)| !protected)
                .map(|(text, _)| text.as_str())
                .collect();

            assert!(
                removable.is_empty(),
                "reported removable: {removable:?}, full result: {reported:?}"
            );
        }

        /// A naive generator is worthless here. Drawing arbitrary strings would
        /// spend every draw on paths that resolve nowhere near a protected
        /// root, and would pass against a guard that is wide open. So the
        /// alphabet is tiny and entirely made of the components that matter:
        /// `..` and `.` (the ones nothing in the guard resolves), the empty
        /// string (which produces a doubled separator once joined), and the
        /// names of real protected roots.
        fn component() -> impl Strategy<Value = &'static str> {
            prop_oneof![
                6 => Just(".."),
                3 => Just("."),
                2 => Just(""),
                3 => Just("etc"),
                2 => Just("usr"),
                2 => Just("rocm"),
                2 => Just("runtimes"),
                // Off Windows a backslash is an ordinary filename byte, so this
                // is ONE legitimate directory name, not two components. It is in
                // the alphabet because folding `\` to `/` unconditionally and
                // then resolving `..` silently walks out of a protected root;
                // without this component every property below still passes.
                2 => Just(r"..\tmp"),
                // The leaf names a real managed `install_root` ends in, so the
                // generated paths look like the ones the guard actually sees.
                2 => Just("wheel"),
                2 => Just("tarball"),
            ]
        }

        /// Seeded with the places a real `install_root` is written down: the
        /// protected roots themselves, the user's home, and an ordinary
        /// unprotected folder.
        fn prefix() -> impl Strategy<Value = String> {
            let mut seeds: Vec<String> = PROTECTED_ROOTS
                .iter()
                .map(|&root| root.to_owned())
                .collect();
            seeds.push("/".to_owned());
            seeds.push("/tmp".to_owned());
            seeds.push("/home".to_owned());
            if let Some(home) = home_text() {
                seeds.push(format!("{home}/.rocm"));
                seeds.push(home);
            }
            proptest::sample::select(seeds)
        }

        /// Paths rooted at the user's own home, so the exemption that keeps
        /// `~/.rocm/...` removable is sampled densely instead of being drowned
        /// out by the thirteen protected prefixes.
        fn user_owned_text() -> impl Strategy<Value = String> {
            let home = home_text().unwrap_or_else(|| "/home/rocm".to_owned());
            let seeds = vec![
                home.clone(),
                format!("{home}/.rocm"),
                format!("{home}/.rocm/data"),
            ];
            path_text(proptest::sample::select(seeds))
        }

        /// A path is text, and the guard must answer for the folder that text
        /// names, not for the characters it happens to be spelled with.
        fn install_root_text() -> impl Strategy<Value = String> {
            path_text(prefix())
        }

        fn path_text(prefix: impl Strategy<Value = String>) -> impl Strategy<Value = String> {
            (
                prefix,
                proptest::collection::vec(component(), 0..4),
                prop_oneof![4 => Just(""), 2 => Just("/"), 1 => Just("/."), 1 => Just("/..")],
            )
                .prop_map(|(prefix, parts, trailing)| {
                    let mut text = prefix;
                    for part in parts {
                        text.push('/');
                        text.push_str(part);
                    }
                    text.push_str(trailing);
                    text
                })
        }

        proptest::proptest! {
            /// The contract the guard exists for: no path that really resolves
            /// into a protected system location may be reported removable,
            /// however it is spelled. A counterexample here is `remove_dir_all`
            /// on a system directory.
            #[test]
            fn a_protected_location_is_refused_however_it_is_spelled(
                text in install_root_text(),
            ) {
                prop_assume!(!exempt_as_user_owned(&text));
                prop_assume!(lexically_protected(&text));

                prop_assert!(
                    runtime_install_root_is_protected(Path::new(&text)),
                    "{text} resolves to {} but was reported removable",
                    lexically_resolved(&text)
                );
            }

            /// The guard must not swing the other way either: a folder that
            /// really is the user's own stays removable, or `runtimes uninstall`
            /// refuses the very folder it created.
            #[test]
            fn a_user_owned_location_stays_removable(text in user_owned_text()) {
                // Strictly-inside-home is the whole contract: the exemption is
                // unconditional, so this must hold even when home itself sits
                // under a protected root (`/root` for a root user). Filtering
                // those draws out with `!lexically_protected` would both assume
                // away the interesting case AND reject every draw on such a
                // host, which proptest reports as a hard abort rather than a
                // skip.
                prop_assume!(exempt_as_user_owned(&text));

                prop_assert!(
                    !runtime_install_root_is_protected(Path::new(&text)),
                    "{text} resolves to {} but was refused",
                    lexically_resolved(&text)
                );
            }

            /// Containment is what both the home exemption and the protected
            /// -root check are built out of, so it has to agree with where the
            /// paths actually resolve — in both directions. Answering "inside"
            /// for a path that escaped lets a caller out of the guard;
            /// answering "outside" for one that did not hides a protected root.
            #[test]
            fn containment_agrees_with_where_the_paths_resolve(
                path in install_root_text(),
                base in prefix(),
            ) {
                prop_assert_eq!(
                    runtime_path_is_same_or_inside(Path::new(&path), Path::new(&base)),
                    lexically_same_or_inside(&path, &base),
                    "containment of {} in {} disagrees with {} in {}",
                    path.clone(),
                    base.clone(),
                    lexically_resolved(&path),
                    lexically_resolved(&base)
                );
            }

            /// Two spellings of one folder are one folder.
            #[test]
            fn equivalence_sees_through_spelling(text in install_root_text()) {
                let resolved = lexically_resolved(&text);
                prop_assert!(
                    runtime_paths_equivalent(Path::new(&text), Path::new(&resolved)),
                    "{text} and {resolved} name the same folder but compared unequal"
                );
            }

            /// Both protected-location rules against an oracle restated from the
            /// policy, never from the code: `/` is the filesystem root; home is
            /// home; anything strictly inside home is free; a folder containing
            /// home protects it; otherwise the system list decides — the
            /// locations AND what is inside them for runtime folders, only the
            /// locations themselves for `rocm uninstall`'s roots. Home is drawn
            /// too, including `/` (which must exempt nothing) and `/root`.
            #[test]
            fn both_rules_agree_with_the_policy_for_any_home(
                text in path_text(proptest::sample::select(policy_seeds())),
                home in proptest::sample::select(vec![
                    FIXED_HOME.to_owned(),
                    "/".to_owned(),
                    "/root".to_owned(),
                ]),
            ) {
                let resolved = lexically_resolved(&text);
                let path = Path::new(&text);
                let home_path = Some(Path::new(home.as_str()));
                prop_assert_eq!(
                    runtime_protected_location_for_home(path, home_path),
                    policy_location(&resolved, &home, true),
                    "runtime rule: {} resolves to {}, home {}", text, resolved, home
                );
                prop_assert_eq!(
                    uninstall_root_protected_location_for_home(path, home_path),
                    policy_location(&resolved, &home, false),
                    "uninstall rule: {} resolves to {}, home {}", text, resolved, home
                );
            }
        }

        const FIXED_HOME: &str = "/home/alice";

        fn policy_seeds() -> Vec<String> {
            let mut seeds: Vec<String> = PROTECTED_ROOTS
                .iter()
                .map(|root| (*root).to_owned())
                .collect();
            seeds.extend(
                [
                    "/",
                    "/home",
                    FIXED_HOME,
                    "/home/alice/.rocm",
                    "/home/alicex",
                    "/tmp",
                    "/var/cache",
                    "/opt/rocm-cli",
                ]
                .map(str::to_owned),
            );
            seeds
        }

        /// The policy, for an already-resolved absolute path.
        fn policy_location(
            resolved: &str,
            home: &str,
            and_inside: bool,
        ) -> Option<ProtectedLocation> {
            if resolved == "/" {
                return Some(ProtectedLocation::FilesystemRoot);
            }
            if home != "/" {
                if resolved == home {
                    return Some(ProtectedLocation::Home);
                }
                if lexically_same_or_inside(resolved, home) {
                    return None;
                }
                if home.starts_with(&format!("{resolved}/")) {
                    return Some(ProtectedLocation::ContainsHome);
                }
            }
            let system = if and_inside {
                PROTECTED_ROOTS
                    .iter()
                    .any(|root| lexically_same_or_inside(resolved, root))
            } else {
                PROTECTED_ROOTS.contains(&resolved)
            };
            system.then_some(ProtectedLocation::System)
        }

        /// The drive root is the filesystem root on Windows, judged from any
        /// host, in every spelling; on Linux `C:/` is an ordinary relative name.
        #[test]
        fn a_windows_drive_root_is_the_filesystem_root() {
            let windows = RuntimePlatform::Windows;
            for root in ["C:/", "C:\\", "c:/", "C:/Windows/..", "//server/share/"] {
                assert!(is_filesystem_root_on(Path::new(root), windows), "{root}");
            }
            for not_root in ["C:/Windows", "//server/share/dir"] {
                assert!(
                    !is_filesystem_root_on(Path::new(not_root), windows),
                    "{not_root}"
                );
            }
            assert!(!is_filesystem_root_on(
                Path::new("C:/"),
                RuntimePlatform::Linux
            ));
            assert_eq!(
                protected_location_on(Path::new("C:\\"), None, windows, SystemRule::Itself),
                Some(ProtectedLocation::FilesystemRoot)
            );
            assert_eq!(
                protected_location_on(
                    Path::new("C:/Program Files/rocm-cli"),
                    None,
                    windows,
                    SystemRule::Itself
                ),
                None
            );
            assert_eq!(
                protected_location_on(Path::new("C:/Windows/"), None, windows, SystemRule::Itself),
                Some(ProtectedLocation::System)
            );
        }

        /// The cases the issue reported, plus the root-user and root-home
        /// shapes, pinned as examples.
        #[test]
        fn home_and_its_ancestors_are_named_for_what_they_are() {
            let home = Some(Path::new(FIXED_HOME));
            let cases: [(&str, Option<ProtectedLocation>); 11] = [
                ("/", Some(ProtectedLocation::FilesystemRoot)),
                ("//", Some(ProtectedLocation::FilesystemRoot)),
                ("//..", Some(ProtectedLocation::FilesystemRoot)),
                ("/home/alice", Some(ProtectedLocation::Home)),
                ("/home/alice/", Some(ProtectedLocation::Home)),
                ("/home/alice/.rocm/..", Some(ProtectedLocation::Home)),
                ("/home", Some(ProtectedLocation::ContainsHome)),
                ("/home/alice/.cache", None),
                ("/home/alicex", None),
                ("/usr/local", Some(ProtectedLocation::System)),
                ("/tmp/rocm", None),
            ];
            for (path, expected) in cases {
                assert_eq!(
                    runtime_protected_location_for_home(Path::new(path), home),
                    expected,
                    "{path}"
                );
            }
        }

        /// Root's home sits under a protected root; it is refused as a home,
        /// and what is inside it stays removable.
        #[test]
        fn a_root_users_home_is_refused_as_a_home() {
            let home = Some(Path::new("/root"));
            assert_eq!(
                runtime_protected_location_for_home(Path::new("/root"), home),
                Some(ProtectedLocation::Home)
            );
            assert_eq!(
                runtime_protected_location_for_home(Path::new("/root/.rocm"), home),
                None
            );
        }

        /// A home of `/` must not exempt the whole machine.
        #[test]
        fn a_home_at_the_filesystem_root_exempts_nothing() {
            let home = Some(Path::new("/"));
            assert_eq!(
                runtime_protected_location_for_home(Path::new("/etc"), home),
                Some(ProtectedLocation::System)
            );
            assert_eq!(
                runtime_protected_location_for_home(Path::new("/"), home),
                Some(ProtectedLocation::FilesystemRoot)
            );
        }

        /// The public gate refuses the real home, which it used to report as
        /// removable.
        #[test]
        fn the_public_gate_refuses_the_users_own_home() {
            let home = runtime_home_dir().expect("a home directory");
            assert!(
                runtime_install_root_is_protected(&home),
                "{}",
                home.display()
            );
        }
    }
}
