// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The per-user `Path` that `install.ps1` adds the install directory to.
//!
//! The installer writes it through `[Environment]::SetEnvironmentVariable(..,
//! "User")`, which stores it under `HKCU\Environment`. `rocm uninstall` takes the
//! entry back out here, so nothing it added is left pointing at a removed folder.

use std::path::Path;

/// `list` with every entry naming `dir` removed, or `None` when there is none.
///
/// Entries compare the way Windows resolves them: case-insensitively and
/// ignoring a trailing separator. Every other entry, including empty ones and
/// ones that use `%VARIABLES%`, is kept byte for byte, so the rest of the
/// user's `Path` reads back exactly as it was.
#[must_use]
pub fn remove_path_list_entry(list: &str, dir: &Path) -> Option<String> {
    let target = normalize_entry(&dir.to_string_lossy());
    if target.is_empty() {
        return None;
    }
    let entries: Vec<&str> = list.split(';').collect();
    let kept: Vec<&str> = entries
        .iter()
        .copied()
        .filter(|entry| !normalize_entry(entry).eq_ignore_ascii_case(&target))
        .collect();
    (kept.len() != entries.len()).then(|| kept.join(";"))
}

fn normalize_entry(entry: &str) -> String {
    entry.trim().trim_end_matches(['\\', '/']).to_owned()
}

/// Remove `dir` from the user `Path` in `HKCU\Environment`.
///
/// Keeps the value's registry type (`REG_SZ` or `REG_EXPAND_SZ`), then tells
/// running programs the environment changed. Returns whether an entry was
/// removed.
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn remove_user_path_entry(dir: &Path) -> anyhow::Result<bool> {
    use anyhow::{Context, bail};
    use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_EXPAND_SZ, REG_SZ,
        RegCloseKey, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        HWND_BROADCAST, SMTO_ABORTIFHUNG, SendMessageTimeoutW, WM_SETTINGCHANGE,
    };

    let wide = |text: &str| text.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let subkey = wide("Environment");
    let value_name = wide("Path");

    let mut key: HKEY = std::ptr::null_mut();
    let status = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            subkey.as_ptr(),
            0,
            KEY_QUERY_VALUE | KEY_SET_VALUE,
            &raw mut key,
        )
    };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(false);
    }
    if status != ERROR_SUCCESS {
        bail!(
            "could not open HKCU\\Environment: {}",
            std::io::Error::from_raw_os_error(status.cast_signed())
        );
    }
    let result = (|| -> anyhow::Result<bool> {
        let mut value_type = 0u32;
        let mut size = 0u32;
        let status = unsafe {
            RegQueryValueExW(
                key,
                value_name.as_ptr(),
                std::ptr::null(),
                &raw mut value_type,
                std::ptr::null_mut(),
                &raw mut size,
            )
        };
        if status == ERROR_FILE_NOT_FOUND {
            return Ok(false);
        }
        if status != ERROR_SUCCESS {
            bail!(
                "could not read the user Path: {}",
                std::io::Error::from_raw_os_error(status.cast_signed())
            );
        }
        if value_type != REG_SZ && value_type != REG_EXPAND_SZ {
            bail!("the user Path has an unexpected registry type ({value_type})");
        }
        let mut buffer = vec![0u16; (size as usize).div_ceil(2)];
        let status = unsafe {
            RegQueryValueExW(
                key,
                value_name.as_ptr(),
                std::ptr::null(),
                &raw mut value_type,
                buffer.as_mut_ptr().cast(),
                &raw mut size,
            )
        };
        if status != ERROR_SUCCESS {
            bail!(
                "could not read the user Path: {}",
                std::io::Error::from_raw_os_error(status.cast_signed())
            );
        }
        buffer.truncate((size as usize) / 2);
        while buffer.last() == Some(&0) {
            buffer.pop();
        }
        let current = String::from_utf16(&buffer).context("the user Path is not valid UTF-16")?;
        let Some(updated) = remove_path_list_entry(&current, dir) else {
            return Ok(false);
        };
        let data = wide(&updated);
        let byte_len = u32::try_from(data.len() * 2).context("the user Path is too long")?;
        let status = unsafe {
            RegSetValueExW(
                key,
                value_name.as_ptr(),
                0,
                value_type,
                data.as_ptr().cast(),
                byte_len,
            )
        };
        if status != ERROR_SUCCESS {
            bail!(
                "could not write the user Path: {}",
                std::io::Error::from_raw_os_error(status.cast_signed())
            );
        }
        Ok(true)
    })();
    unsafe {
        RegCloseKey(key);
    }
    let removed = result?;
    if removed {
        // Explorer rereads the user environment on this broadcast, so terminals
        // opened from it afterwards no longer get the folder. Best effort: the
        // registry write above is the change itself.
        let environment = wide("Environment");
        unsafe {
            SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                0,
                environment.as_ptr() as isize,
                SMTO_ABORTIFHUNG,
                5000,
                std::ptr::null_mut(),
            );
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::remove_path_list_entry;
    use std::path::Path;

    const DIR: &str = r"C:\Users\me\.local\bin";

    #[test]
    fn removes_only_the_matching_entry_and_keeps_the_rest_verbatim() {
        let list = r"C:\Users\me\.local\bin;%USERPROFILE%\go\bin;;C:\tools";
        assert_eq!(
            remove_path_list_entry(list, Path::new(DIR)).as_deref(),
            Some(r"%USERPROFILE%\go\bin;;C:\tools")
        );
    }

    #[test]
    fn matches_case_insensitively_and_ignores_a_trailing_separator() {
        let list = r"C:\tools;c:\users\ME\.LOCAL\bin\";
        assert_eq!(
            remove_path_list_entry(list, Path::new(DIR)).as_deref(),
            Some(r"C:\tools")
        );
    }

    #[test]
    fn removes_every_duplicate() {
        let list = format!("{DIR};C:\\tools;{DIR}");
        assert_eq!(
            remove_path_list_entry(&list, Path::new(DIR)).as_deref(),
            Some(r"C:\tools")
        );
    }

    #[test]
    fn returns_none_when_the_dir_is_absent() {
        assert_eq!(
            remove_path_list_entry(r"C:\tools;C:\Users\me\.local\bin2", Path::new(DIR)),
            None
        );
        assert_eq!(remove_path_list_entry("", Path::new(DIR)), None);
    }

    #[test]
    fn an_empty_dir_never_matches_empty_entries() {
        assert_eq!(remove_path_list_entry(r"C:\tools;;", Path::new("")), None);
    }
}
