// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::runtime_is_linux;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) struct ParentRuntimeEnvironment {
    xdg_runtime_dir: Option<OsString>,
    home: Option<OsString>,
    user: Option<String>,
    temp_dir: PathBuf,
}

impl ParentRuntimeEnvironment {
    pub(crate) fn current() -> Self {
        Self {
            xdg_runtime_dir: std::env::var_os("XDG_RUNTIME_DIR"),
            home: std::env::var_os("HOME"),
            // An empty `USER` must fall through to `LOGNAME`, not short-circuit it.
            user: std::env::var("USER")
                .ok()
                .filter(|v| !v.is_empty())
                .or_else(|| std::env::var("LOGNAME").ok().filter(|v| !v.is_empty())),
            temp_dir: std::env::temp_dir(),
        }
    }
}

/// Leaf directory the engine gets inside a fallback runtime root, so a crashing
/// or misbehaving server cannot disturb the telemetry socket that shares the
/// same root.
const RUNTIME_DIR_LEAF: &str = "lemonade";

/// Guarantee the child a usable `XDG_RUNTIME_DIR`.
///
/// `lemond` refuses to start unless it can resolve a writable runtime directory
/// from `XDG_RUNTIME_DIR` or `RUNTIME_DIRECTORY`. Neither exists on a headless
/// host: `XDG_RUNTIME_DIR` is populated by `pam_systemd` at login, so it is
/// absent for every non-login process (cron jobs, CI runners, `systemd-run`, a
/// bare container exec), and `RUNTIME_DIRECTORY` only exists for a systemd unit
/// that declares `RuntimeDirectory=`. There, a managed serve failed
/// deterministically before the server ever came up.
///
/// A value already present in the parent environment is passed through
/// unchanged — an operator's choice always wins, and it is the tier the rest of
/// this precedence exists to fall back from.
pub(crate) fn child_runtime_dir_var(
    parent: &ParentRuntimeEnvironment,
) -> Result<Option<(&'static str, OsString)>> {
    if !runtime_is_linux() {
        return Ok(None);
    }
    if let Some(existing) = parent
        .xdg_runtime_dir
        .as_ref()
        .filter(|value| !value.is_empty())
    {
        return Ok(Some(("XDG_RUNTIME_DIR", existing.clone())));
    }
    let has_home = parent.home.as_ref().is_some_and(|value| !value.is_empty());
    let dir = rocm_core::user_runtime_dir(
        None,
        parent.home.clone(),
        parent.user.clone(),
        parent.temp_dir.clone(),
        RUNTIME_DIR_LEAF,
        RUNTIME_DIR_LEAF,
    );
    let prepare = if has_home {
        create_private_runtime_dir(&dir)
    } else {
        create_private_tier3_runtime_dir(&dir)
    };
    prepare.with_context(|| {
        format!(
            "preparing a runtime directory for the Lemonade server at {} \
             (XDG_RUNTIME_DIR is unset in this environment)",
            dir.display()
        )
    })?;
    Ok(Some(("XDG_RUNTIME_DIR", dir.into_os_string())))
}

#[cfg(unix)]
fn create_private_tier3_runtime_dir(dir: &Path) -> Result<()> {
    let user_parent = dir
        .parent()
        .ok_or_else(|| anyhow!("{} has no tier-3 user parent", dir.display()))?;
    let temp_root = user_parent.parent().ok_or_else(|| {
        anyhow!(
            "{} has no temporary-directory parent",
            user_parent.display()
        )
    })?;
    let temp_root = open_safe_tier3_root(temp_root)?;
    let user_component = single_directory_component(user_parent)?;
    let user_parent = create_private_owned_directory_at(&temp_root, user_component, user_parent)?;
    let leaf_component = single_directory_component(dir)?;
    create_private_owned_directory_at(&user_parent, leaf_component, dir)?;
    Ok(())
}

#[cfg(unix)]
fn single_directory_component(path: &Path) -> Result<&OsStr> {
    path.file_name()
        .filter(|component| Path::new(component).components().count() == 1)
        .ok_or_else(|| anyhow!("{} has no safe final directory component", path.display()))
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn open_safe_tier3_root(temp_root: &Path) -> Result<fs::File> {
    use std::ffi::CString;
    use std::io::ErrorKind;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Component;

    if !temp_root.is_absolute() {
        bail!(
            "tier-3 temporary root {} is relative or otherwise unstable",
            temp_root.display()
        );
    }
    let mut components = temp_root.components();
    if components.next() != Some(Component::RootDir) {
        bail!(
            "tier-3 temporary root {} has no filesystem-root anchor",
            temp_root.display()
        );
    }

    let mut directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(Path::new("/"))
        .context("opening filesystem root without following symlinks")?;
    let current_euid = effective_user_id();
    let mut traversed = PathBuf::from("/");
    let mut components = components.peekable();
    while let Some(component) = components.next() {
        let Component::Normal(component) = component else {
            bail!(
                "tier-3 temporary root {} contains a relative or unstable component",
                temp_root.display()
            );
        };
        validate_runtime_ancestry_container(&traversed, &directory, current_euid)?;
        let child_path = traversed.join(component);
        let component = CString::new(component.as_bytes()).with_context(|| {
            format!(
                "tier-3 temporary root {} contains a NUL byte",
                temp_root.display()
            )
        })?;
        let is_final = components.peek().is_none();
        let raw_fd = match unsafe_openat_directory(directory.as_raw_fd(), &component) {
            Ok(fd) => fd,
            Err(error) if is_final && error.kind() == ErrorKind::NotFound => {
                match unsafe_mkdirat(directory.as_raw_fd(), &component, 0o700) {
                    Ok(()) => {}
                    Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("creating tier-3 temporary root {}", temp_root.display())
                        });
                    }
                }
                unsafe_openat_directory(directory.as_raw_fd(), &component).with_context(|| {
                    format!(
                        "opening newly created tier-3 temporary root {}",
                        temp_root.display()
                    )
                })?
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "opening {} while securely traversing tier-3 temporary-root ancestry",
                        child_path.display()
                    )
                });
            }
        };
        // SAFETY: `unsafe_openat_directory` returned a new owned descriptor,
        // transferred exactly once into a `File` that closes it on drop.
        directory = unsafe { fs::File::from_raw_fd(raw_fd) };
        validate_runtime_directory_type(&child_path, &unsafe_fstat(directory.as_raw_fd())?)?;
        traversed = child_path;
    }

    let metadata = unsafe_fstat(directory.as_raw_fd())?;
    if !directory_prevents_cross_uid_entry_replacement(
        metadata.st_uid,
        metadata.st_mode,
        current_euid,
        filesystem_is_read_only(directory.as_raw_fd())?,
    ) {
        bail!(
            "tier-3 temporary root {} is unsafe: trusted owners require no group/other write or sticky protection; foreign owners require a read-only filesystem",
            temp_root.display()
        );
    }
    Ok(directory)
}

#[cfg(unix)]
const fn directory_prevents_cross_uid_entry_replacement(
    uid: u32,
    mode: u32,
    current_euid: u32,
    filesystem_read_only: bool,
) -> bool {
    if uid == 0 || uid == current_euid {
        return mode & 0o022 == 0 || mode & libc::S_ISVTX != 0;
    }
    filesystem_read_only
}

#[cfg(unix)]
fn validate_runtime_ancestry_container(
    path: &Path,
    directory: &fs::File,
    current_euid: u32,
) -> Result<()> {
    use std::os::fd::AsRawFd;

    let metadata = unsafe_fstat(directory.as_raw_fd())?;
    validate_runtime_directory_type(path, &metadata)?;
    if !directory_prevents_cross_uid_entry_replacement(
        metadata.st_uid,
        metadata.st_mode,
        current_euid,
        filesystem_is_read_only(directory.as_raw_fd())?,
    ) {
        bail!(
            "tier-3 temporary-root ancestor {} permits cross-UID rename or replacement",
            path.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn filesystem_is_read_only(fd: std::os::fd::RawFd) -> Result<bool> {
    Ok(unsafe_fstatvfs(fd)?.f_flag & libc::ST_RDONLY != 0)
}

#[cfg(unix)]
fn validate_runtime_directory_type(path: &Path, metadata: &libc::stat) -> Result<()> {
    if metadata.st_mode & libc::S_IFMT != libc::S_IFDIR {
        bail!("{} is not a directory", path.display());
    }
    Ok(())
}

#[cfg(not(unix))]
fn create_private_tier3_runtime_dir(dir: &Path) -> Result<()> {
    create_private_runtime_dir(dir)
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn create_private_owned_directory_at(
    parent: &fs::File,
    component: &OsStr,
    display_path: &Path,
) -> Result<fs::File> {
    use std::ffi::CString;
    use std::io::ErrorKind;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let component = CString::new(component.as_bytes()).with_context(|| {
        format!(
            "{} contains a NUL byte in its final component",
            display_path.display()
        )
    })?;
    let mkdir_result = unsafe_mkdirat(parent.as_raw_fd(), &component, 0o700);
    if let Err(error) = mkdir_result
        && error.kind() != ErrorKind::AlreadyExists
    {
        return Err(error).with_context(|| format!("creating {}", display_path.display()));
    }

    let raw_fd = unsafe_openat_directory(parent.as_raw_fd(), &component).with_context(|| {
        format!(
            "opening {} relative to its validated parent without following symlinks",
            display_path.display()
        )
    })?;
    // SAFETY: `unsafe_openat_directory` returned a new owned descriptor, and
    // this is its single transfer into a type that closes it on drop.
    let directory = unsafe { fs::File::from_raw_fd(raw_fd) };
    let current_euid = effective_user_id();
    validate_runtime_directory_owner(
        display_path,
        &unsafe_fstat(directory.as_raw_fd())?,
        current_euid,
    )?;
    unsafe_fchmod(directory.as_raw_fd(), 0o700)
        .with_context(|| format!("restricting {} to mode 0700", display_path.display()))?;

    let secured = unsafe_fstat(directory.as_raw_fd())?;
    validate_runtime_directory_owner(display_path, &secured, current_euid)?;
    if secured.st_mode & 0o777 != 0o700 {
        bail!(
            "{} is not private after chmod; expected mode 0700",
            display_path.display()
        );
    }
    Ok(directory)
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn unsafe_fstat(fd: std::os::fd::RawFd) -> std::io::Result<libc::stat> {
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `metadata` points to writable storage for one `libc::stat`, and
    // `fd` is borrowed from a live `File`. On success fstat initializes it.
    if unsafe { libc::fstat(fd, metadata.as_mut_ptr()) } == 0 {
        // SAFETY: successful fstat initialized the full `libc::stat` value.
        Ok(unsafe { metadata.assume_init() })
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn unsafe_fstatvfs(fd: std::os::fd::RawFd) -> std::io::Result<libc::statvfs> {
    let mut metadata = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `metadata` points to writable storage for one `libc::statvfs`,
    // and `fd` is borrowed from a live directory `File`. On success fstatvfs
    // initializes it.
    if unsafe { libc::fstatvfs(fd, metadata.as_mut_ptr()) } == 0 {
        // SAFETY: successful fstatvfs initialized the full value.
        Ok(unsafe { metadata.assume_init() })
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn unsafe_fchmod(fd: std::os::fd::RawFd, mode: libc::mode_t) -> std::io::Result<()> {
    // SAFETY: `fd` is borrowed from a live directory `File`; fchmod does not
    // retain it, and the supplied mode is a valid permission bitmask.
    if unsafe { libc::fchmod(fd, mode) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn unsafe_mkdirat(
    parent_fd: std::os::fd::RawFd,
    component: &std::ffi::CStr,
    mode: libc::mode_t,
) -> std::io::Result<()> {
    // SAFETY: `component` is NUL-terminated and lives for the call; `parent_fd`
    // is borrowed from an open directory descriptor; mkdirat does not retain
    // either argument.
    if unsafe { libc::mkdirat(parent_fd, component.as_ptr(), mode) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn unsafe_openat_directory(
    parent_fd: std::os::fd::RawFd,
    component: &std::ffi::CStr,
) -> std::io::Result<std::os::fd::RawFd> {
    // SAFETY: `component` is NUL-terminated and lives for the call; `parent_fd`
    // is an open directory descriptor. On success openat returns a new owned fd.
    let fd = unsafe {
        libc::openat(
            parent_fd,
            component.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd >= 0 {
        Ok(fd)
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn validate_runtime_directory_owner(
    dir: &Path,
    metadata: &libc::stat,
    current_euid: u32,
) -> Result<()> {
    if metadata.st_mode & libc::S_IFMT != libc::S_IFDIR {
        bail!("{} is not a directory", dir.display());
    }
    if metadata.st_uid != current_euid {
        bail!(
            "{} is owned by uid {}, not current euid {}",
            dir.display(),
            metadata.st_uid,
            current_euid
        );
    }
    Ok(())
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn effective_user_id() -> u32 {
    // SAFETY: `geteuid` takes no arguments, has no caller-side preconditions,
    // and only returns process identity state.
    unsafe { libc::geteuid() }
}

/// Create `dir` (and its parents) restricted to the current user.
///
/// This is the existing recursive behavior used for the HOME-based fallback,
/// where legitimate symlinked HOME or `.rocm` paths retain normal filesystem
/// semantics. Tier 3 uses [`create_private_tier3_runtime_dir`] instead so its
/// predictable parent is validated separately without following symlinks.
///
/// The leaf itself is still refused when it is a symlink, and an existing leaf
/// is tightened explicitly because `DirBuilder::mode` only affects directories
/// it creates.
fn create_private_runtime_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        if fs::symlink_metadata(dir)?.file_type().is_symlink() {
            bail!(
                "{} is a symlink, so it may point somewhere another user controls",
                dir.display()
            );
        }
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).with_context(|| {
            format!(
                "restricting {} to mode 0700 — it is most likely owned by another user",
                dir.display()
            )
        })?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::{LemonadeProcessEnvironment, lemonade_process_environment_vars};

    // `XDG_RUNTIME_DIR` is a Linux/freedesktop concept and the Lemonade server
    // only consults it there, so these are Linux-only by construction.
    #[cfg(target_os = "linux")]
    mod child_runtime_dir {
        use super::*;

        fn value_of(vars: &[(&'static str, OsString)], key: &str) -> Option<OsString> {
            vars.iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.clone())
        }

        fn runtime_scratch() -> tempfile::TempDir {
            tempfile::Builder::new()
                .prefix(".lemonade-runtime-test-")
                .tempdir_in(std::env::current_dir().expect("current directory"))
                .expect("private-ancestry scratch")
        }
        fn vars_for(parent: &ParentRuntimeEnvironment) -> Vec<(&'static str, OsString)> {
            lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), parent)
                .expect("building the child environment must succeed")
        }

        fn assert_private_dir(dir: &Path) {
            use std::os::unix::fs::PermissionsExt;
            assert!(
                dir.is_dir(),
                "{} must exist before the server is spawned",
                dir.display()
            );
            let mode = fs::metadata(dir).expect("metadata").permissions().mode() & 0o777;
            assert_eq!(
                mode,
                0o700,
                "{} must be private to this user",
                dir.display()
            );
        }

        /// Regression: on a headless host `pam_systemd` never runs, so
        /// `XDG_RUNTIME_DIR` is unset and the Lemonade server exits during
        /// startup with "Unable to resolve writable runtime directory". The
        /// child must be handed a directory that already exists.
        #[test]
        fn is_synthesized_when_the_parent_has_none() {
            let scratch = runtime_scratch();
            let home = scratch.path().join("home");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: Some(home.clone().into_os_string()),
                user: Some("alice".to_owned()),
                temp_dir: scratch.path().join("tmp"),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("the child must be given a runtime directory");

            // Tier 2 of the shared precedence, with a `lemonade` leaf so the
            // engine never shares a directory with the telemetry socket.
            let dir = PathBuf::from(value);
            assert_eq!(dir, home.join(".rocm").join("data").join("lemonade"));
            assert_private_dir(&dir);
        }

        /// An exported-but-empty variable is as good as unset, and must not be
        /// mistaken for an operator's choice.
        #[test]
        fn is_synthesized_when_the_parent_value_is_empty() {
            let scratch = runtime_scratch();
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: Some(OsString::new()),
                home: Some(scratch.path().join("home").into_os_string()),
                user: Some("alice".to_owned()),
                temp_dir: scratch.path().join("tmp"),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("an empty value must fall through, not be passed on");
            assert_eq!(
                PathBuf::from(value),
                scratch
                    .path()
                    .join("home")
                    .join(".rocm")
                    .join("data")
                    .join("lemonade")
            );
        }

        /// Tier 2 lives under the operator's HOME and must retain normal path
        /// semantics; homes (or `.rocm`) are legitimately symlinked on some
        /// systems. The no-follow hardening is intentionally tier-3-only.
        #[test]
        fn permits_a_symlinked_home_path_for_tier2() {
            let scratch = runtime_scratch();
            let real_home = scratch.path().join("real-home");
            let linked_home = scratch.path().join("linked-home");
            fs::create_dir_all(&real_home).expect("real home");
            std::os::unix::fs::symlink(&real_home, &linked_home).expect("linked home");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: Some(linked_home.clone().into_os_string()),
                user: Some("alice".to_owned()),
                temp_dir: scratch.path().join("tmp"),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("the child must be given a runtime directory");

            let dir = PathBuf::from(value);
            assert_eq!(dir, linked_home.join(".rocm").join("data").join("lemonade"));
            assert_private_dir(&dir);
            assert!(real_home.join(".rocm/data/lemonade").is_dir());
        }
        /// Tier 3: neither `XDG_RUNTIME_DIR` nor `HOME` — a user-named subdir of
        /// the temp dir, so the parent is one we create and own, not `/tmp`.
        #[test]
        fn falls_back_to_a_user_owned_temp_subdir_without_home() {
            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("tmp");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: temp_dir.clone(),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("the child must be given a runtime directory");

            let dir = PathBuf::from(value);
            assert_eq!(dir, temp_dir.join("rocm-alice").join("lemonade"));
            assert_private_dir(&dir);
        }

        /// The tier-3 fallback lands under a shared temp dir, so the target may
        /// already exist as a symlink someone else planted. Chmod would follow
        /// it, so the resolver refuses rather than loosening another directory.
        #[test]
        fn refuses_a_symlinked_runtime_dir() {
            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("tmp");
            let elsewhere = scratch.path().join("elsewhere");
            fs::create_dir_all(temp_dir.join("rocm-alice")).expect("runtime root");
            fs::create_dir_all(&elsewhere).expect("elsewhere");
            std::os::unix::fs::symlink(&elsewhere, temp_dir.join("rocm-alice").join("lemonade"))
                .expect("symlink");

            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir,
            };
            let error = format!(
                "{:#}",
                lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), &parent)
                    .expect_err("a symlinked runtime dir must be refused")
            );
            assert!(error.contains("symlink"), "{error}");
        }

        /// Tier 3 uses a predictable user parent under a shared temp directory.
        /// Refuse that parent when it is a symlink: recursively creating the
        /// leaf would otherwise follow it into an attacker-controlled tree.
        #[test]
        fn refuses_a_symlinked_tier3_user_parent() {
            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("tmp");
            let elsewhere = scratch.path().join("elsewhere");
            fs::create_dir_all(&temp_dir).expect("temp root");
            fs::create_dir_all(&elsewhere).expect("elsewhere");
            std::os::unix::fs::symlink(&elsewhere, temp_dir.join("rocm-alice")).expect("symlink");

            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir,
            };
            let error = format!(
                "{:#}",
                lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), &parent)
                    .expect_err("a symlinked tier-3 user parent must be refused")
            );

            assert!(error.contains("symlink"), "{error}");
            assert!(
                !elsewhere.join("lemonade").exists(),
                "the resolver followed the tier-3 user-parent symlink"
            );
        }

        /// A pre-existing tier-3 user parent may have been created with a
        /// permissive umask. The resolver must secure that predictable parent,
        /// not merely the Lemonade leaf beneath it.
        #[test]
        fn tightens_a_preexisting_non_private_tier3_user_parent() {
            use std::os::unix::fs::PermissionsExt;

            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("tmp");
            let user_parent = temp_dir.join("rocm-alice");
            fs::create_dir_all(&user_parent).expect("user parent");
            fs::set_permissions(&user_parent, fs::Permissions::from_mode(0o755))
                .expect("make parent non-private");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir,
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("the child must be given a runtime directory");

            assert_eq!(PathBuf::from(value), user_parent.join("lemonade"));
            assert_private_dir(&user_parent);
            assert_private_dir(&user_parent.join("lemonade"));
        }

        /// A tier-3 root without sticky-directory protection lets another UID
        /// rename or replace our predictable user component after validation.
        #[test]
        fn refuses_a_group_or_other_writable_non_sticky_temp_root() {
            use std::os::unix::fs::PermissionsExt;

            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("unsafe-tmp");
            fs::create_dir(&temp_dir).expect("temp root");
            fs::set_permissions(&temp_dir, fs::Permissions::from_mode(0o777))
                .expect("make root unsafe");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: temp_dir.clone(),
            };

            let error = format!(
                "{:#}",
                lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), &parent)
                    .expect_err("a non-sticky writable temp root must be refused")
            );

            assert!(error.contains("sticky"), "{error}");
            assert!(!temp_dir.join("rocm-alice").exists());
        }

        /// A private final temp root is still unstable when an ancestor lets
        /// another UID replace the path entry that names it.
        #[test]
        fn refuses_a_private_temp_root_beneath_a_non_sticky_writable_parent() {
            use std::os::unix::fs::PermissionsExt;

            let scratch = runtime_scratch();
            let unsafe_parent = scratch.path().join("unsafe-parent");
            let temp_dir = unsafe_parent.join("private-tmp");
            fs::create_dir(&unsafe_parent).expect("unsafe parent");
            fs::set_permissions(&unsafe_parent, fs::Permissions::from_mode(0o777))
                .expect("make ancestor unsafe");
            fs::create_dir(&temp_dir).expect("private temp root");
            fs::set_permissions(&temp_dir, fs::Permissions::from_mode(0o700))
                .expect("make final root private");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: temp_dir.clone(),
            };

            let error = format!(
                "{:#}",
                lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), &parent)
                    .expect_err("unsafe temp-root ancestry must be refused")
            );

            assert!(error.contains("ancestor"), "{error}");
            assert!(!temp_dir.join("rocm-alice").exists());
        }

        #[test]
        fn accepts_a_private_temp_root_with_private_ancestry() {
            use std::os::unix::fs::PermissionsExt;

            let scratch = runtime_scratch();
            let private_parent = scratch.path().join("private-parent");
            let temp_dir = private_parent.join("private-tmp");
            fs::create_dir(&private_parent).expect("private parent");
            fs::set_permissions(&private_parent, fs::Permissions::from_mode(0o700))
                .expect("private ancestor mode");
            fs::create_dir(&temp_dir).expect("private temp root");
            fs::set_permissions(&temp_dir, fs::Permissions::from_mode(0o700))
                .expect("private root mode");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: temp_dir.clone(),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("private ancestry must be accepted");

            assert_eq!(PathBuf::from(value), temp_dir.join("rocm-alice/lemonade"));
        }

        #[test]
        fn accepts_standard_sticky_temp_root_semantics() {
            use std::os::unix::fs::PermissionsExt;

            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("sticky-tmp");
            fs::create_dir(&temp_dir).expect("sticky temp root");
            fs::set_permissions(&temp_dir, fs::Permissions::from_mode(0o1777))
                .expect("standard sticky temp mode");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: temp_dir.clone(),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("trusted sticky temp semantics must be accepted");

            assert_eq!(PathBuf::from(value), temp_dir.join("rocm-alice/lemonade"));
            assert_private_dir(&temp_dir.join("rocm-alice"));
        }

        #[test]
        fn accepts_the_host_standard_tmp_when_its_invariants_are_safe() {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::MetadataExt;

            let path = Path::new("/tmp");
            let metadata = fs::metadata(path).expect("/tmp metadata");
            let directory = fs::File::open(path).expect("open /tmp");
            if !directory_prevents_cross_uid_entry_replacement(
                metadata.uid(),
                metadata.mode(),
                effective_user_id(),
                filesystem_is_read_only(directory.as_raw_fd()).expect("/tmp filesystem flags"),
            ) {
                return;
            }

            open_safe_tier3_root(path).expect("safe host /tmp must be accepted");
        }

        #[test]
        fn refuses_a_relative_tier3_temp_root() {
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: PathBuf::from("relative-tmp"),
            };

            let error = format!(
                "{:#}",
                lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), &parent)
                    .expect_err("relative tier-3 roots must be refused")
            );

            assert!(error.contains("relative"), "{error}");
        }

        #[test]
        fn refuses_symlinked_temp_root_ancestry() {
            let scratch = runtime_scratch();
            let real_parent = scratch.path().join("real-parent");
            let linked_parent = scratch.path().join("linked-parent");
            fs::create_dir(&real_parent).expect("real parent");
            std::os::unix::fs::symlink(&real_parent, &linked_parent).expect("linked ancestry");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: None,
                home: None,
                user: Some("alice".to_owned()),
                temp_dir: linked_parent.join("private-tmp"),
            };

            let error = format!(
                "{:#}",
                lemonade_process_environment_vars(&LemonadeProcessEnvironment::default(), &parent)
                    .expect_err("symlinked tier-3 ancestry must be refused")
            );

            assert!(error.contains("ancestry"), "{error}");
            assert!(!real_parent.join("private-tmp").exists());
        }

        /// Sticky mode does not constrain the directory owner. A sticky root
        /// owned by an unrelated non-root UID therefore cannot protect our
        /// component from that UID replacing it.
        #[test]
        fn refuses_an_untrusted_foreign_owned_sticky_temp_root_when_available() {
            use std::os::unix::fs::MetadataExt;

            let current_euid = effective_user_id();
            let candidate = [Path::new("/tmp"), Path::new("/var/tmp")]
                .into_iter()
                .filter_map(|path| fs::metadata(path).ok().map(|metadata| (path, metadata)))
                .find(|(_, metadata)| {
                    metadata.uid() != 0
                        && metadata.uid() != current_euid
                        && metadata.mode() & libc::S_ISVTX != 0
                });
            let Some((path, _)) = candidate else {
                return;
            };

            let error = format!(
                "{:#}",
                open_safe_tier3_root(path)
                    .expect_err("an untrusted sticky-root owner must be refused")
            );
            assert!(error.contains("sticky"), "{error}");
            assert!(error.contains("owner"), "{error}");
        }

        #[test]
        fn ancestry_safety_rejects_foreign_owned_mode_0555_on_writable_filesystem() {
            assert!(!directory_prevents_cross_uid_entry_replacement(
                1234, 0o555, 1000, false
            ));
        }

        #[test]
        fn ancestry_safety_rejects_foreign_owned_mode_0755() {
            assert!(!directory_prevents_cross_uid_entry_replacement(
                1234, 0o755, 1000, false
            ));
        }

        #[test]
        fn ancestry_safety_accepts_foreign_owned_writable_on_read_only_filesystem() {
            assert!(directory_prevents_cross_uid_entry_replacement(
                1234, 0o777, 1000, true
            ));
        }

        #[test]
        fn ancestry_safety_accepts_trusted_owned_mode_0755() {
            assert!(directory_prevents_cross_uid_entry_replacement(
                0, 0o755, 1000, false
            ));
        }

        #[test]
        fn ancestry_safety_rejects_trusted_owned_mode_0777_without_sticky() {
            assert!(!directory_prevents_cross_uid_entry_replacement(
                1000, 0o777, 1000, false
            ));
        }

        #[test]
        fn ancestry_safety_accepts_trusted_owned_mode_1777() {
            assert!(directory_prevents_cross_uid_entry_replacement(
                1000, 0o1777, 1000, false
            ));
        }

        /// Once the temp root is open, replacing its pathname must not redirect
        /// creation into the replacement tree. This deterministically models
        /// the rename-and-replace window from the original implementation.
        #[test]
        fn descriptor_relative_creation_resists_temp_root_replacement() {
            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("tmp");
            fs::create_dir(&temp_dir).expect("temp root");
            let root = open_safe_tier3_root(&temp_dir).expect("open root");

            let original_root = scratch.path().join("original-tmp");
            fs::rename(&temp_dir, &original_root).expect("rename open root");
            fs::create_dir(&temp_dir).expect("replacement root");
            let elsewhere = scratch.path().join("elsewhere");
            fs::create_dir(&elsewhere).expect("elsewhere");
            std::os::unix::fs::symlink(&elsewhere, temp_dir.join("rocm-alice"))
                .expect("attacker replacement");

            create_private_owned_directory_at(
                &root,
                OsStr::new("rocm-alice"),
                &temp_dir.join("rocm-alice"),
            )
            .expect("descriptor-relative creation");

            assert_private_dir(&original_root.join("rocm-alice"));
            assert!(
                fs::symlink_metadata(temp_dir.join("rocm-alice"))
                    .expect("replacement symlink")
                    .file_type()
                    .is_symlink()
            );
            assert!(!elsewhere.join("rocm-alice").exists());
        }

        /// Leaf creation stays anchored to the validated user-parent object even
        /// if its pathname is replaced with a symlink after it was opened.
        #[test]
        fn descriptor_relative_leaf_creation_resists_user_parent_replacement() {
            let scratch = runtime_scratch();
            let temp_dir = scratch.path().join("tmp");
            fs::create_dir(&temp_dir).expect("temp root");
            let root = open_safe_tier3_root(&temp_dir).expect("open root");
            let user_path = temp_dir.join("rocm-alice");
            let user_parent =
                create_private_owned_directory_at(&root, OsStr::new("rocm-alice"), &user_path)
                    .expect("create user parent");

            let original_user_parent = temp_dir.join("original-rocm-alice");
            fs::rename(&user_path, &original_user_parent).expect("rename user parent");
            let elsewhere = scratch.path().join("elsewhere");
            fs::create_dir(&elsewhere).expect("elsewhere");
            std::os::unix::fs::symlink(&elsewhere, &user_path).expect("replacement symlink");

            create_private_owned_directory_at(
                &user_parent,
                OsStr::new("lemonade"),
                &user_path.join("lemonade"),
            )
            .expect("descriptor-relative leaf creation");

            assert_private_dir(&original_user_parent.join("lemonade"));
            assert!(!elsewhere.join("lemonade").exists());
        }

        /// Ownership can be exercised without root when the host exposes a
        /// system-owned temp/root directory. The preparation routine must
        /// reject it before changing its permissions.
        #[test]
        fn rejects_a_foreign_owned_tier3_user_parent_when_available() {
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

            let current_euid = effective_user_id();
            let foreign = [
                (Path::new("/"), OsStr::new("tmp")),
                (Path::new("/"), OsStr::new("var")),
                (Path::new("/"), OsStr::new("run")),
            ]
            .into_iter()
            .filter_map(|(parent, component)| {
                let path = parent.join(component);
                fs::metadata(&path)
                    .ok()
                    .map(|metadata| (parent, component, path, metadata))
            })
            .find(|(_, _, _, metadata)| metadata.is_dir() && metadata.uid() != current_euid);
            let Some((parent_path, component, path, metadata)) = foreign else {
                return;
            };
            let original_mode = metadata.mode();
            let parent = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
                .open(parent_path)
                .expect("open foreign parent");

            let error = format!(
                "{:#}",
                create_private_owned_directory_at(&parent, component, &path)
                    .expect_err("foreign ownership must be refused")
            );
            assert!(error.contains("owned by uid"), "{error}");
            assert!(error.contains(&current_euid.to_string()), "{error}");
            assert_eq!(
                fs::metadata(&path).expect("metadata after refusal").mode(),
                original_mode,
                "foreign directory permissions changed before ownership was validated"
            );
        }
        /// A value the operator (or systemd) already set is handed to the child
        /// unchanged: we never relocate a working runtime directory, and never
        /// create anything of our own beside it.
        #[test]
        fn passes_an_existing_parent_value_through_unchanged() {
            let scratch = tempfile::tempdir().expect("scratch");
            let existing = scratch.path().join("run").join("user").join("1000");
            let home = scratch.path().join("home");
            let parent = ParentRuntimeEnvironment {
                xdg_runtime_dir: Some(existing.clone().into_os_string()),
                home: Some(home.clone().into_os_string()),
                user: Some("alice".to_owned()),
                temp_dir: scratch.path().join("tmp"),
            };

            let value = value_of(&vars_for(&parent), "XDG_RUNTIME_DIR")
                .expect("the child must be given a runtime directory");

            assert_eq!(PathBuf::from(value), existing);
            assert!(
                !home.exists(),
                "no fallback directory may be created when the parent already has one"
            );
        }
    }
}
