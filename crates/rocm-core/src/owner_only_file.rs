// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Creating a file that only the current user can open, for secrets such as a
//! served endpoint's API key.
//!
//! [`open_new_owner_only`] creates the file with its access already narrowed,
//! so the secret is never readable by anyone else, not even between the
//! create and a later permission change:
//!
//! - on Unix the file is created with mode `0600`, and the mode is set again
//!   on the open handle before the caller writes anything, because the umask
//!   can only remove bits at creation and an unusual one could leave the file
//!   unreadable by its own owner;
//! - on Windows it is created with a protected DACL whose single entry grants
//!   full access to the user the process runs as (the token user, which is not
//!   necessarily the file's owner when the process is elevated). Protected means
//!   the file does not inherit entries from its directory, so a data directory
//!   that other accounts can read does not make the secret readable by them.
//!   No entry is added for administrators or `SYSTEM`.
//!
//! The file is created with `create_new` semantics on both platforms: an
//! existing file or symbolic link at `path`, dangling or not, is an error and is
//! never followed or opened, so a caller cannot be made to write a secret
//! through a file or link somebody else prepared. On Windows that takes
//! `FILE_FLAG_OPEN_REPARSE_POINT`, which `std` also passes for `create_new`,
//! and the path goes through the same long-path (`\\?\`) conversion `std`
//! applies, so a path longer than `MAX_PATH` works here as it does for
//! `std::fs`. To replace an existing secret, use
//! [`crate::atomic_write::write_owner_only_file_atomically`], which stages the
//! bytes in a fresh owner-only file and renames it over the old one.

use std::fs::File;
use std::io;
use std::path::Path;

/// Create `path` as a new file that only the current user can open.
///
/// Returns it open for writing. Fails with [`io::ErrorKind::AlreadyExists`]
/// when anything — a file, a directory or a symbolic link — is already at
/// `path`.
pub fn open_new_owner_only(path: &Path) -> io::Result<File> {
    create_new_owner_only(path)
}

/// The DACL [`open_new_owner_only`] gives a file on Windows, in SDDL.
///
/// A protected DACL (`P`, so no entry is inherited from the directory) holding
/// one entry that allows full file access (`FA`) to `user_sid`.
///
/// Built on every platform so the shape it promises is unit-tested everywhere,
/// not only on the Windows job.
pub fn owner_only_sddl(user_sid: &str) -> String {
    format!("D:P(A;;FA;;;{user_sid})")
}

/// The create itself, before the mode is set again on the handle: mode 0600
/// as the file comes into existence, narrowed further by the umask but never
/// widened. Its own function so a test can observe the file as created.
#[cfg(unix)]
fn create_new_at_owner_only_mode(path: &Path) -> io::Result<File> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt as _;

    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(unix)]
fn create_new_owner_only(path: &Path) -> io::Result<File> {
    use std::fs::Permissions;
    use std::os::unix::fs::PermissionsExt as _;

    let file = create_new_at_owner_only_mode(path)?;
    // The umask can only have removed bits from 0600, never added any, so
    // nobody else can open the file even before this call. Setting the mode on
    // the handle makes it exactly 0600 whatever the umask was, so the owner can
    // always read the secret back.
    if let Err(error) = file.set_permissions(Permissions::from_mode(0o600)) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    Ok(file)
}

#[cfg(windows)]
fn create_new_owner_only(path: &Path) -> io::Result<File> {
    let sid = win32::current_user_sid()?;
    win32::create_new_with_sddl(path, &owner_only_sddl(&sid))
}

/// Test support: `None` when the file at `path` has a protected DACL whose
/// only entry grants the current user full access, as [`open_new_owner_only`]
/// creates it; otherwise the DACL found, in SDDL, for the failure message.
///
/// Compiled only for this crate's tests and under the `test-support` feature,
/// which the apps enable as a dev-dependency so their own Windows tests can
/// check the files they store.
#[cfg(all(windows, any(test, feature = "test-support")))]
#[doc(hidden)]
pub fn owner_only_dacl_mismatch(path: &Path) -> io::Result<Option<String>> {
    use win32::test_support::{canonical_sddl, dacl_sddl, flags_and_entries};

    let actual = dacl_sddl(path)?;
    let expected = canonical_sddl(&owner_only_sddl(&win32::current_user_sid()?))?;
    let (flags, entries) = flags_and_entries(&actual);
    let (_, expected_entries) = flags_and_entries(&expected);
    Ok((!(flags.contains('P') && entries == expected_entries)).then_some(actual))
}

// UTF-16 units of the path syntax below; all ASCII.
#[cfg(any(windows, test))]
const SEP: u16 = b'\\' as u16;
#[cfg(any(windows, test))]
const ALT_SEP: u16 = b'/' as u16;
#[cfg(any(windows, test))]
const COLON: u16 = b':' as u16;
/// `\\?\`
#[cfg(any(windows, test))]
const VERBATIM_PREFIX: [u16; 4] = [SEP, SEP, b'?' as u16, SEP];
/// `\??\`
#[cfg(any(windows, test))]
const NT_PREFIX: [u16; 4] = [SEP, b'?' as u16, b'?' as u16, SEP];
/// `\\.\`
#[cfg(any(windows, test))]
const DEVICE_PREFIX: [u16; 4] = [SEP, SEP, b'.' as u16, SEP];
/// `\\?\UNC\`
#[cfg(any(windows, test))]
const UNC_PREFIX: [u16; 8] = [
    SEP,
    SEP,
    b'?' as u16,
    SEP,
    b'U' as u16,
    b'N' as u16,
    b'C' as u16,
    SEP,
];
/// The length below which `std` passes a drive-absolute or UNC path to Windows
/// unchanged (the `CreateDirectoryW` limit, the tightest of the `MAX_PATH`
/// family).
#[cfg(any(windows, test))]
const LEGACY_MAX_PATH: usize = 248;

/// Whether `std` would pass the UTF-16 path `wide` (no NUL) to Windows as it
/// is: a verbatim path, or a short one that is already drive-absolute or UNC.
#[cfg(any(windows, test))]
fn usable_as_is(wide: &[u16]) -> bool {
    if wide.starts_with(&VERBATIM_PREFIX) || wide.starts_with(&NT_PREFIX) {
        return true;
    }
    if wide.len() + 1 >= LEGACY_MAX_PATH {
        return false;
    }
    match wide {
        [drive, COLON] | [drive, COLON, SEP | ALT_SEP, ..] => *drive != SEP && *drive != ALT_SEP,
        [SEP | ALT_SEP, SEP | ALT_SEP, ..] => true,
        _ => false,
    }
}

/// The verbatim, NUL-terminated form of `absolute`, a path `GetFullPathNameW`
/// already made absolute and normalized: `C:\x` becomes `\\?\C:\x`,
/// `\\.\x` becomes `\\?\x`, `\\server\share` becomes
/// `\\?\UNC\server\share`, and a path already verbatim is kept. The same
/// mapping `std` applies before it opens a file.
#[cfg(any(windows, test))]
fn with_verbatim_prefix(absolute: &[u16]) -> Vec<u16> {
    let (prefix, rest): (&[u16], &[u16]) = match absolute {
        [_, COLON, SEP, ..] => (&VERBATIM_PREFIX, absolute),
        _ if absolute.starts_with(&DEVICE_PREFIX) => (&VERBATIM_PREFIX, &absolute[4..]),
        _ if absolute.starts_with(&VERBATIM_PREFIX) || absolute.starts_with(&NT_PREFIX) => {
            (&[], absolute)
        }
        [SEP, SEP, ..] => (&UNC_PREFIX, &absolute[2..]),
        _ => (&[], absolute),
    };
    prefix.iter().chain(rest).copied().chain(Some(0)).collect()
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub(crate) mod win32 {
    //! The Win32 calls behind the Windows side of
    //! [`super::open_new_owner_only`]. Every buffer the system allocates for us
    //! is released with `LocalFree`, and the token handle is closed, on every
    //! path including the error ones.

    use std::ffi::OsString;
    use std::fs::File;
    use std::io;
    use std::os::windows::ffi::{OsStrExt as _, OsStringExt as _};
    use std::os::windows::io::{FromRawHandle as _, OwnedHandle};
    use std::path::Path;
    use std::ptr;

    use windows_sys::Win32::Foundation::{GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
        TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CREATE_NEW, CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    /// A buffer the system allocated with `LocalAlloc` on our behalf, released
    /// with `LocalFree` when dropped.
    pub(crate) struct LocalBuffer(pub(crate) *mut core::ffi::c_void);

    impl Drop for LocalBuffer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: the pointer came from an API documented to allocate
                // it with `LocalAlloc` and hand ownership to the caller, and
                // this guard is its only owner, so it is freed exactly once.
                unsafe {
                    LocalFree(self.0);
                }
            }
        }
    }

    /// Read a NUL-terminated UTF-16 string the system allocated.
    ///
    /// # Safety
    /// `wide` must point to a valid NUL-terminated UTF-16 string.
    pub(crate) unsafe fn string_from_wide(wide: *const u16) -> io::Result<String> {
        let mut len = 0;
        // SAFETY: the caller guarantees a NUL terminator, so every offset read
        // here up to and including it is inside the string.
        while unsafe { *wide.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: `len` units starting at `wide` were just read above.
        let units = unsafe { std::slice::from_raw_parts(wide, len) };
        OsString::from_wide(units)
            .into_string()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "string is not valid Unicode"))
    }

    /// `path` as the NUL-terminated UTF-16 name `std::fs` would hand
    /// `CreateFileW`: unchanged when it is already verbatim, or short and
    /// absolute; otherwise made absolute and given the verbatim prefix, which
    /// lifts the `MAX_PATH` limit. `std` keeps this conversion private, so it
    /// is repeated here on top of the public [`std::path::absolute`]; the
    /// decisions are in [`super::usable_as_is`] and
    /// [`super::with_verbatim_prefix`], which are tested on every host.
    pub(crate) fn long_path_wide(path: &Path) -> io::Result<Vec<u16>> {
        let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        if wide.contains(&0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "strings passed to WinAPI cannot contain NULs",
            ));
        }
        if super::usable_as_is(&wide) {
            return Ok(wide.into_iter().chain(Some(0)).collect());
        }
        let absolute: Vec<u16> = std::path::absolute(path)?
            .as_os_str()
            .encode_wide()
            .collect();
        Ok(super::with_verbatim_prefix(&absolute))
    }

    /// The string form (`S-1-5-21-…`) of the SID of the user this process runs
    /// as, read from the process token.
    pub(crate) fn current_user_sid() -> io::Result<String> {
        let mut raw_token: HANDLE = ptr::null_mut();
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no
        // closing; `raw_token` is a valid out-pointer for the call.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw mut raw_token) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so `raw_token` is an open handle that we
        // own; `OwnedHandle` closes it with `CloseHandle` when dropped.
        let token = unsafe { OwnedHandle::from_raw_handle(raw_token) };
        let token_handle: HANDLE = std::os::windows::io::AsRawHandle::as_raw_handle(&token);

        let mut needed = 0u32;
        // SAFETY: a null buffer of length 0 asks only for the required size,
        // which is written to `needed`. The call reports failure
        // (`ERROR_INSUFFICIENT_BUFFER`) by design, so its result is not checked;
        // a zero `needed` below is what means it really failed.
        unsafe {
            GetTokenInformation(token_handle, TokenUser, ptr::null_mut(), 0, &raw mut needed);
        }
        if needed == 0 {
            return Err(io::Error::last_os_error());
        }
        // `u64` units so the buffer is aligned for `TOKEN_USER`, which holds a
        // pointer.
        let mut buffer = vec![0u64; (needed as usize).div_ceil(size_of::<u64>())];
        // SAFETY: `buffer` holds at least `needed` writable bytes and outlives
        // the call.
        if unsafe {
            GetTokenInformation(
                token_handle,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                needed,
                &raw mut needed,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so the buffer starts with a `TOKEN_USER`
        // whose SID pointer points into the same buffer, still alive here.
        let sid = unsafe { (*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid };

        let mut wide: *mut u16 = ptr::null_mut();
        // SAFETY: `sid` is a valid SID (see above) and `wide` a valid
        // out-pointer.
        if unsafe { ConvertSidToStringSidW(sid, &raw mut wide) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let wide = LocalBuffer(wide.cast());
        // SAFETY: on success the system returned a NUL-terminated string.
        unsafe { string_from_wide(wide.0.cast()) }
        // `token` and `wide` are released here, on this path and on every
        // early return above.
    }

    /// A self-relative security descriptor parsed from `sddl`, owned by the
    /// returned guard.
    pub(crate) fn security_descriptor_from_sddl(sddl: &str) -> io::Result<LocalBuffer> {
        let sddl_wide: Vec<u16> = sddl.encode_utf16().chain(Some(0)).collect();
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        // SAFETY: `sddl_wide` is a NUL-terminated UTF-16 string alive for the
        // call; `descriptor` is a valid out-pointer and the size is not wanted.
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl_wide.as_ptr(),
                SDDL_REVISION_1,
                &raw mut descriptor,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(LocalBuffer(descriptor))
    }

    /// Create `path` as a new file whose security descriptor is `sddl`, and
    /// return it open for writing. The descriptor is applied by the create
    /// itself, so there is no moment at which the file has any other DACL.
    pub(crate) fn create_new_with_sddl(path: &Path, sddl: &str) -> io::Result<File> {
        let descriptor = security_descriptor_from_sddl(sddl)?;
        let attributes = SECURITY_ATTRIBUTES {
            nLength: u32::try_from(size_of::<SECURITY_ATTRIBUTES>())
                .expect("SECURITY_ATTRIBUTES fits in a u32"),
            lpSecurityDescriptor: descriptor.0,
            bInheritHandle: 0,
        };
        let path_wide = long_path_wide(path)?;
        // SAFETY: `path_wide` is NUL-terminated and `attributes` points at a
        // descriptor that `descriptor` keeps alive past the call. The sharing
        // mode, the long-path form of the name and `FILE_FLAG_OPEN_REPARSE_POINT`
        // are what `std::fs::OpenOptions` uses for `create_new`, so the handle
        // behaves like one `std` would have opened: with that flag `CREATE_NEW`
        // fails on a symbolic link at `path`, dangling or not, instead of
        // creating the file where the link points.
        let handle = unsafe {
            CreateFileW(
                path_wide.as_ptr(),
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                &raw const attributes,
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `handle` is a freshly opened file handle owned by nobody
        // else; `File` takes ownership and closes it.
        Ok(unsafe { File::from_raw_handle(handle) })
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) mod test_support {
        //! Reading a file's DACL back, for the Windows tests here, in
        //! `atomic_write`, and (through [`super::super::owner_only_dacl_mismatch`])
        //! in the crates that store secrets with this module.

        use super::{LocalBuffer, long_path_wide, security_descriptor_from_sddl, string_from_wide};
        use std::io;
        use std::path::Path;
        use std::ptr;
        use windows_sys::Win32::Security::Authorization::{
            ConvertSecurityDescriptorToStringSecurityDescriptorW, GetNamedSecurityInfoW,
            SDDL_REVISION_1, SE_FILE_OBJECT,
        };
        use windows_sys::Win32::Security::{DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR};

        fn dacl_sddl_of_descriptor(descriptor: PSECURITY_DESCRIPTOR) -> io::Result<String> {
            let mut wide: *mut u16 = ptr::null_mut();
            // SAFETY: `descriptor` is a valid security descriptor owned by the
            // caller; `wide` is a valid out-pointer and the length is not wanted.
            if unsafe {
                ConvertSecurityDescriptorToStringSecurityDescriptorW(
                    descriptor,
                    SDDL_REVISION_1,
                    DACL_SECURITY_INFORMATION,
                    &raw mut wide,
                    ptr::null_mut(),
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            let wide = LocalBuffer(wide.cast());
            // SAFETY: on success the system returned a NUL-terminated string.
            unsafe { string_from_wide(wide.0.cast()) }
        }

        /// The DACL of the file at `path`, in SDDL (`D:P(A;;FA;;;S-1-…)`).
        pub(crate) fn dacl_sddl(path: &Path) -> io::Result<String> {
            let path_wide = long_path_wide(path)?;
            let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
            // SAFETY: `path_wide` is NUL-terminated; the owner, group and SACL
            // out-pointers are null because they are not requested, and the
            // DACL pointer is not wanted since the descriptor is converted
            // whole.
            let status = unsafe {
                GetNamedSecurityInfoW(
                    path_wide.as_ptr(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &raw mut descriptor,
                )
            };
            if status != 0 {
                return Err(io::Error::from_raw_os_error(
                    i32::try_from(status).unwrap_or(i32::MAX),
                ));
            }
            let descriptor = LocalBuffer(descriptor);
            dacl_sddl_of_descriptor(descriptor.0)
        }

        /// `sddl` as Windows itself writes it back, so a comparison is not
        /// thrown off by the system abbreviating a well-known SID.
        pub(crate) fn canonical_sddl(sddl: &str) -> io::Result<String> {
            let descriptor = security_descriptor_from_sddl(sddl)?;
            dacl_sddl_of_descriptor(descriptor.0)
        }

        /// Split a DACL's SDDL into its control flags (`P`, `AI`, …) and its
        /// entries.
        pub(crate) fn flags_and_entries(sddl: &str) -> (&str, &str) {
            let body = sddl.strip_prefix("D:").unwrap_or(sddl);
            body.find('(')
                .map_or((body, ""), |start| body.split_at(start))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unix_time_millis;
    use std::io::Write as _;
    use std::path::PathBuf;

    fn scratch_dir(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rocm-core-open-new-owner-only-{label}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn owner_only_sddl_is_a_protected_dacl_with_one_full_access_entry() {
        assert_eq!(
            owner_only_sddl("S-1-5-21-1-2-3-1001"),
            "D:P(A;;FA;;;S-1-5-21-1-2-3-1001)"
        );
    }

    #[test]
    fn an_existing_file_is_never_opened() {
        let root = scratch_dir("exists");
        let path = root.join("secret");
        std::fs::write(&path, b"someone else's file").unwrap();

        let error = open_new_owner_only(&path).expect_err("an existing file must not be opened");
        let contents = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(contents, b"someone else's file");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_never_written_through() {
        let root = scratch_dir("symlink");
        let target = root.join("target");
        std::fs::write(&target, b"untouched").unwrap();
        let link = root.join("secret");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let error = open_new_owner_only(&link).expect_err("a link must not be followed");
        let contents = std::fs::read(&target).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(contents, b"untouched");
    }

    /// The file comes out exactly 0600 whatever the umask. The umask is
    /// process-wide, so the narrow one this needs is set in a re-executed copy
    /// of this test rather than here, where it would reach sibling tests.
    ///
    /// `0o477` strips the owner's read bit from a plain `mode(0o600)` create,
    /// leaving a key file its own owner cannot read back; only the mode set on
    /// the handle afterwards brings it to 0600.
    #[cfg(unix)]
    #[test]
    #[allow(unsafe_code)] // libc::umask, in the re-executed child only
    fn a_new_file_is_exactly_0600_whatever_the_umask() {
        use std::os::unix::fs::PermissionsExt as _;

        if let Some(dir) = std::env::var_os("ROCM_TEST_OWNER_ONLY_UMASK_DIR") {
            // SAFETY: `umask` only swaps this process's file-creation mask;
            // this child process runs only this one test.
            let previous = unsafe { libc::umask(0o477) };
            let path = PathBuf::from(dir).join("secret");
            let opened = open_new_owner_only(&path);
            // Put the mask back before anything else creates a file: under
            // `cargo llvm-cov` this child writes its own coverage profile at
            // exit, and one created under 0477 is unreadable, which crashes
            // `llvm-profdata merge` and fails the coverage job.
            // SAFETY: as above.
            unsafe {
                libc::umask(previous);
            }
            let mut file = opened.unwrap();
            file.write_all(b"key").unwrap();
            drop(file);
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "mode under umask 0477 was {mode:o}");
            return;
        }

        let root = scratch_dir("umask");
        let status =
            std::process::Command::new(std::env::current_exe().expect("current test executable"))
                .args([
                    "owner_only_file::tests::a_new_file_is_exactly_0600_whatever_the_umask",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_OWNER_ONLY_UMASK_DIR", &root)
                .status()
                .expect("re-exec the test in a child process");
        let created = root.join("secret").exists();
        let _ = std::fs::remove_dir_all(&root);
        assert!(created, "the child did not run the test arm");
        assert!(status.success(), "{status:?}");
    }

    /// Nobody else can open the file even before its mode is set again on the
    /// handle: the create itself asks for 0600. Checked in a re-executed child
    /// under umask 0, where nothing narrows the requested mode, so what the
    /// create asked for is exactly what this sees.
    #[cfg(unix)]
    #[test]
    #[allow(unsafe_code)] // libc::umask, in the re-executed child only
    fn the_file_is_owner_only_as_created_before_its_mode_is_set_again() {
        use std::os::unix::fs::PermissionsExt as _;

        if let Some(dir) = std::env::var_os("ROCM_TEST_OWNER_ONLY_CREATE_DIR") {
            // SAFETY: `umask` only swaps this process's file-creation mask;
            // this child process runs only this one test.
            let previous = unsafe { libc::umask(0) };
            let path = PathBuf::from(dir).join("secret");
            let created = create_new_at_owner_only_mode(&path);
            // SAFETY: as above; restored before anything else creates a file.
            unsafe {
                libc::umask(previous);
            }
            drop(created.unwrap());
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "mode as created under umask 0 was {mode:o}");
            return;
        }

        let root = scratch_dir("as-created");
        let status =
            std::process::Command::new(std::env::current_exe().expect("current test executable"))
                .args([
                    "owner_only_file::tests::the_file_is_owner_only_as_created_before_its_mode_is_set_again",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_OWNER_ONLY_CREATE_DIR", &root)
                .status()
                .expect("re-exec the test in a child process");
        let created = root.join("secret").exists();
        let _ = std::fs::remove_dir_all(&root);
        assert!(created, "the child did not run the test arm");
        assert!(status.success(), "{status:?}");
    }

    #[cfg(windows)]
    #[test]
    fn a_new_file_has_a_protected_dacl_for_the_current_user_only() {
        let root = scratch_dir("dacl");
        let path = root.join("secret");
        let mut file = open_new_owner_only(&path).unwrap();
        file.write_all(b"key").unwrap();
        drop(file);

        let mismatch = owner_only_dacl_mismatch(&path).unwrap();
        let contents = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(mismatch, None, "the DACL is not protected owner-only");
        assert_eq!(contents, b"key", "the owner cannot read the file back");
    }

    /// The checker itself: a file created with an inherited DACL, as
    /// `std::fs::write` leaves it, is reported, so a `None` above means
    /// something.
    #[cfg(windows)]
    #[test]
    fn the_dacl_check_reports_an_inherited_dacl() {
        let root = scratch_dir("dacl-inherited");
        let path = root.join("plain");
        std::fs::write(&path, b"plain").unwrap();

        let mismatch = owner_only_dacl_mismatch(&path).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(mismatch.is_some(), "an inherited DACL passed the check");
    }

    /// A symbolic link at the destination — dangling, or pointing at a file —
    /// fails the create instead of being followed, so nothing is created or
    /// written where it points. Creating a link needs Developer Mode or the
    /// symlink privilege; where neither is available the test fails on CI
    /// (`CI` set) and elsewhere says so and passes without checking.
    #[cfg(windows)]
    #[test]
    fn a_symlink_is_never_followed_on_windows() {
        let root = scratch_dir("symlink-win");
        let missing_target = root.join("missing-target");
        let dangling = root.join("dangling");
        if let Err(error) = std::os::windows::fs::symlink_file(&missing_target, &dangling) {
            let _ = std::fs::remove_dir_all(&root);
            // On CI this test is the only check of the reparse-point flag, so
            // a runner that cannot create links fails rather than passing
            // unchecked; a developer machine without the privilege skips.
            assert!(
                std::env::var_os("CI").is_none(),
                "cannot create a symbolic link on CI, so the link-following check did not run: {error}"
            );
            eprintln!("SKIPPED: this account cannot create a symbolic link ({error})");
            return;
        }
        let target = root.join("target");
        std::fs::write(&target, b"untouched").unwrap();
        let live = root.join("live");
        std::os::windows::fs::symlink_file(&target, &live).unwrap();

        let dangling_result = open_new_owner_only(&dangling).map(drop);
        let live_result = open_new_owner_only(&live).map(drop);
        let created_at_target = missing_target.exists();
        let target_contents = std::fs::read(&target).unwrap();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(
            dangling_result.map_err(|error| error.kind()),
            Err(io::ErrorKind::AlreadyExists),
            "a dangling link was followed"
        );
        assert!(
            !created_at_target,
            "the file was created at the link's target"
        );
        assert_eq!(
            live_result.map_err(|error| error.kind()),
            Err(io::ErrorKind::AlreadyExists),
            "a link to a file was followed"
        );
        assert_eq!(target_contents, b"untouched");
    }

    /// A destination past `MAX_PATH` is created, as `std::fs` would create it.
    #[cfg(windows)]
    #[test]
    fn a_path_longer_than_max_path_is_created() {
        let root = scratch_dir("long-path");
        let mut dir = root.clone();
        while dir.as_os_str().len() < 300 {
            dir.push("a-directory-name-that-makes-the-path-long");
        }
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secret");

        let created = open_new_owner_only(&path).map(|mut file| file.write_all(b"key"));
        let contents = std::fs::read(&path);
        let mismatch = owner_only_dacl_mismatch(&path);
        let _ = std::fs::remove_dir_all(&root);

        created.unwrap().unwrap();
        assert_eq!(contents.unwrap(), b"key");
        assert_eq!(mismatch.unwrap(), None);
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    fn wide_nul(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    /// The long-path decisions, with expectations written from `std`'s
    /// `get_long_path` (its conversion is private, so it cannot be compared
    /// at run time): short absolute and UNC paths and verbatim ones pass as
    /// they are; anything relative, or long, is converted.
    #[test]
    fn usable_as_is_follows_std_for_each_path_shape() {
        assert!(usable_as_is(&wide(r"C:\Users\me\data\svc.endpoint-key")));
        assert!(usable_as_is(&wide("C:/Users/me/key")));
        assert!(usable_as_is(&wide("C:")));
        assert!(usable_as_is(&wide(r"\\server\share\key")));
        assert!(usable_as_is(&wide(r"\\?\C:\key")));
        assert!(usable_as_is(&wide(r"\??\C:\key")));
        assert!(!usable_as_is(&wide("relative\\key")));
        assert!(!usable_as_is(&wide(r"\rooted-on-the-current-drive")));
        assert!(!usable_as_is(&wide("C:relative-to-the-drive's-cwd")));
        let long = format!(r"C:\{}", "a".repeat(LEGACY_MAX_PATH));
        assert!(!usable_as_is(&wide(&long)));
        // The exact edge: `std` counts the terminating NUL, so a path of 246
        // units (247 with the NUL) still passes as it is and one of 247 does
        // not.
        let of_len = |len: usize| format!(r"C:\{}", "a".repeat(len - 3));
        assert_eq!(of_len(246).encode_utf16().count(), 246);
        assert!(usable_as_is(&wide(&of_len(246))));
        assert!(!usable_as_is(&wide(&of_len(247))));
        assert!(!usable_as_is(&wide(&of_len(248))));
        // A long path that is already verbatim needs nothing.
        assert!(usable_as_is(&wide(&format!(r"\\?\{long}"))));
    }

    /// Expectations written from `std`'s `get_long_path`, as above.
    #[test]
    fn with_verbatim_prefix_follows_std_for_each_absolute_shape() {
        assert_eq!(
            with_verbatim_prefix(&wide(r"C:\Users\me\key")),
            wide_nul(r"\\?\C:\Users\me\key")
        );
        assert_eq!(
            with_verbatim_prefix(&wide(r"\\server\share\key")),
            wide_nul(r"\\?\UNC\server\share\key")
        );
        assert_eq!(
            with_verbatim_prefix(&wide(r"\\.\C:\key")),
            wide_nul(r"\\?\C:\key")
        );
        assert_eq!(
            with_verbatim_prefix(&wide(r"\\?\C:\key")),
            wide_nul(r"\\?\C:\key")
        );
        assert_eq!(
            with_verbatim_prefix(&wide(r"\??\C:\key")),
            wide_nul(r"\??\C:\key")
        );
    }
}
