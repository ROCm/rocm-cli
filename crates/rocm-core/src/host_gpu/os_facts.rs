// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Operating-system facts the host summary reports: kernel version, distro name,
//! CPU model and installed RAM.

use super::capture::{
    OPTIONAL_COMMAND_TIMEOUT, capture_optional_command, capture_optional_command_with_timeout,
};
use crate::{hardware_root, host_path, runtime_is_linux, runtime_is_windows};
use std::fs;

pub(super) fn detect_kernel_version() -> Option<String> {
    if runtime_is_windows() {
        capture_optional_command("cmd", &["/C", "ver"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    } else {
        capture_optional_command("uname", &["-r"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    }
}

pub(crate) fn detect_distro_name() -> Option<String> {
    if runtime_is_windows() {
        return Some("Windows".to_owned());
    }

    if runtime_is_linux() {
        return match hardware_root::read_os_release(hardware_root::host_path_for_str) {
            // Absent matches the pre-fallback behaviour this function has
            // always had: no file at all leaves the distro name unknown
            // entirely, rather than reporting "Linux" for a machine that
            // might not even be running a standard distro.
            hardware_root::OsRelease::Absent => None,
            read => Some(distro_name_from_os_release(&read)),
        };
    }

    None
}

/// The distribution name `rocm examine` prints, from `read`: `PRETTY_NAME` as
/// `sh` reads it, else `Linux`.
///
/// An unreadable line says which file and line made it so, on the line a
/// person reads — `rocm examine`'s `probe_failures`, which carries the same
/// reason, is only in the `--json` form. A blank `PRETTY_NAME` also falls
/// back to `Linux`. A file that exists but could not be read at all is
/// reported as itself, distinctly from a bad line it never reached.
fn distro_name_from_os_release(read: &hardware_root::OsRelease) -> String {
    if let hardware_root::OsRelease::Unreadable { path, error } = read {
        return format!("Linux ({})", hardware_root::unreadable_prefix(path, error));
    }
    match crate::os_release::parse(read.text()) {
        Ok(mut fields) => fields
            .remove("PRETTY_NAME")
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Linux".to_owned()),
        Err(unreadable) => format!("Linux ({} {unreadable})", read.path()),
    }
}

pub(super) fn detect_cpu_model() -> Option<String> {
    if runtime_is_windows() {
        let script =
            "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .map(|value| normalize_cpu_model(&value))
        .filter(|value| !value.is_empty());
    }

    if runtime_is_linux()
        && let Some(model) = fs::read_to_string(host_path("/proc/cpuinfo"))
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    let value = line
                        .strip_prefix("model name")
                        .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
                        .or_else(|| {
                            line.strip_prefix("Hardware")
                                .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
                        })?;
                    let value = normalize_cpu_model(value);
                    (!value.is_empty()).then_some(value)
                })
            })
    {
        return Some(model);
    }

    None
}

pub fn detect_system_ram_gib() -> Option<f64> {
    if runtime_is_windows() {
        let script = "(Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory).TotalPhysicalMemory";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .and_then(|value| bytes_text_to_gib(&value));
    }

    if runtime_is_linux()
        && let Some(kib) = fs::read_to_string(host_path("/proc/meminfo"))
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    let value = line.strip_prefix("MemTotal:")?.trim();
                    let number = value.split_whitespace().next()?.parse::<f64>().ok()?;
                    Some(number)
                })
            })
    {
        return Some(kib / 1024.0 / 1024.0);
    }

    if cfg!(target_os = "macos") {
        return capture_optional_command("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| bytes_text_to_gib(&value));
    }

    None
}

pub(super) fn bytes_text_to_gib(value: &str) -> Option<f64> {
    let bytes = value.trim().parse::<f64>().ok()?;
    (bytes > 0.0).then_some(bytes / 1024.0 / 1024.0 / 1024.0)
}

pub(super) fn format_gib_value(value: f64) -> String {
    if value >= 10.0 {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
    }
}

pub(super) fn normalize_cpu_model(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_os_release_pretty_name() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found(
                "NAME=Ubuntu\nPRETTY_NAME=\"Ubuntu 24.04.2 LTS\"\n"
            )),
            "Ubuntu 24.04.2 LTS"
        );
    }

    /// The distro name is read through the shared `os_release` parser, so it
    /// names the distro `sh` would: the last assignment wins and a trailing
    /// comment is not part of the value. A file with no usable `PRETTY_NAME`
    /// is plain `Linux`; an unreadable one says which line, rather than
    /// letting `Linux` read as a description of the machine.
    #[test]
    fn pretty_name_reads_as_sh_reads_it() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found(
                "PRETTY_NAME=\"Debian GNU/Linux 12\"\nPRETTY_NAME='Ubuntu 24.04 LTS' # LTS\n"
            )),
            "Ubuntu 24.04 LTS"
        );
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found(
                "PRETTY_NAME=\"Ubuntu 24.04 LTS\"\nunset ID\n"
            )),
            "Linux (/etc/os-release line 2 is not a plain assignment: \"unset ID\")"
        );
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found("PRETTY_NAME=\"  \"\n")),
            "Linux"
        );
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found("ID=ubuntu\n")),
            "Linux"
        );
    }

    /// A file that exists but could not be read names itself and the error —
    /// new behavior; it used to be folded into "unknown" like an absent file.
    /// It is distinct from a bad line it never reached, and from "no file at
    /// all", which `detect_distro_name` still reports as unknown (see its own
    /// doc comment) — that part is unchanged.
    #[test]
    fn distro_name_names_an_unreadable_file_instead_of_guessing_linux() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::Unreadable {
                path: "/usr/lib/os-release",
                error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
            }),
            "Linux (/usr/lib/os-release could not be read: denied)"
        );
    }

    /// The same malformed line as the second case in
    /// [`pretty_name_reads_as_sh_reads_it`], but read through the fallback
    /// path: the message must name `/usr/lib/os-release`, not
    /// `/etc/os-release`. That test only exercises
    /// [`hardware_root::OsRelease::found`], which hardcodes
    /// `/etc/os-release`, so a regression that hardcoded the path back into
    /// `distro_name_from_os_release`'s message would still pass it.
    #[test]
    fn a_fallback_reads_malformed_line_names_the_fallback_path() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::Found {
                path: "/usr/lib/os-release",
                text: "PRETTY_NAME=\"Ubuntu 24.04 LTS\"\nunset ID\n".to_owned(),
            }),
            "Linux (/usr/lib/os-release line 2 is not a plain assignment: \"unset ID\")"
        );
    }
}
