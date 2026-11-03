// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Linux driver access: the kernel modules, the device nodes and who may open
//! them, and Secure Boot.

use super::{Device, Examination};
use crate::probe_exec::{SHORT, read_host_text, read_text, run, which};
use std::time::Duration;

pub(super) fn probe_modules(e: &mut Examination) {
    let (rc, out, _) = run("lsmod", &[], SHORT);
    let module_text = if rc == 0 {
        Some(out.lines().skip(1).collect::<Vec<_>>().join("\n"))
    } else {
        let txt = read_host_text("/proc/modules");
        if txt.is_empty() { None } else { Some(txt) }
    };
    if let Some(text) = module_text {
        let modules: Vec<&str> = text
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        e.amdgpu_loaded = Some(modules.contains(&"amdgpu"));
        e.amdkfd_loaded = Some(modules.contains(&"amdkfd"));
    }

    for dir in ["/etc/modprobe.d", "/usr/lib/modprobe.d", "/run/modprobe.d"] {
        let Ok(entries) = std::fs::read_dir(crate::host_path(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("conf") {
                continue;
            }
            let body = read_text(&path.to_string_lossy());
            if body.lines().any(line_blacklists_amdgpu) {
                // The logical path, not where the probe read it.
                e.amdgpu_blacklisted_in
                    .push(format!("{dir}/{}", entry.file_name().to_string_lossy()));
            }
        }
    }
}

/// Matches `^\s*blacklist\s+amdgpu\b`.
pub(super) fn line_blacklists_amdgpu(line: &str) -> bool {
    let rest = line.trim_start();
    let Some(rest) = rest.strip_prefix("blacklist") else {
        return false;
    };
    let rest = rest.trim_start();
    rest == "amdgpu"
        || rest.strip_prefix("amdgpu").is_some_and(|tail| {
            tail.is_empty() || !tail.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        })
}

pub(super) fn probe_devices(e: &mut Examination) {
    e.kfd = Some(stat_device("/dev/kfd", &e.user_name, &e.user_groups));
    if let Ok(entries) = std::fs::read_dir(crate::host_path("/dev/dri")) {
        let mut render: Vec<String> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("renderD")
                    .then(|| format!("/dev/dri/{name}"))
            })
            .collect();
        render.sort();
        for path in render {
            e.render_devices
                .push(stat_device(&path, &e.user_name, &e.user_groups));
        }
    }
}

/// Stat the device node at the logical host path `path` (`/dev/kfd`), reading
/// it where [`crate::host_path`] puts it but reporting the logical path.
pub(super) fn stat_device(path: &str, user_name: &str, user_groups: &[String]) -> Device {
    let host = crate::host_path(path);
    let mut device = Device {
        path: path.to_owned(),
        exists: host.exists(),
        ..Device::default()
    };
    if !device.exists {
        return device;
    }
    let (rc, out, _) = run("stat", &["-c", "%A|%U|%G", &host.to_string_lossy()], SHORT);
    if rc == 0 {
        let fields: Vec<&str> = out.trim().split('|').collect();
        if fields.len() == 3 {
            device.mode = fields[0].to_owned();
            device.owner_user = fields[1].to_owned();
            device.owner_group = fields[2].to_owned();
            let (can_read, can_write) = mode_access(
                &device.mode,
                &device.owner_user,
                &device.owner_group,
                user_name,
                user_groups,
            );
            device.user_can_read = can_read;
            device.user_can_write = can_write;
        }
    }
    device
}

/// Derive read/write access from a `stat`-style mode string and group
/// membership, following POSIX precedence (owner, then group, then other).
pub(super) fn mode_access(
    mode: &str,
    owner_user: &str,
    owner_group: &str,
    user_name: &str,
    user_groups: &[String],
) -> (Option<bool>, Option<bool>) {
    let bytes = mode.as_bytes();
    if bytes.len() < 10 {
        return (None, None);
    }
    let class = if !user_name.is_empty() && user_name == owner_user {
        1
    } else if user_groups.iter().any(|g| g == owner_group) {
        4
    } else {
        7
    };
    let read = bytes[class] == b'r';
    let write = bytes[class + 1] == b'w';
    (Some(read), Some(write))
}

pub(super) fn probe_user(e: &mut Examination) {
    e.user_name = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_default();
    let (rc, out, _) = run("id", &["-Gn"], Duration::from_secs(3));
    if rc == 0 {
        e.user_groups = out.split_whitespace().map(str::to_owned).collect();
    }
    e.in_render_group = Some(e.user_groups.iter().any(|g| g == "render"));
    e.in_video_group = Some(e.user_groups.iter().any(|g| g == "video"));
}

pub(super) fn probe_secure_boot(e: &mut Examination) {
    if !which("mokutil") {
        return;
    }
    let (rc, out, _) = run("mokutil", &["--sb-state"], Duration::from_secs(3));
    if rc == 0 {
        let o = out.to_lowercase();
        if o.contains("enabled") {
            e.secure_boot = "enabled".to_owned();
        } else if o.contains("disabled") {
            e.secure_boot = "disabled".to_owned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blacklist_amdgpu_detection() {
        assert!(line_blacklists_amdgpu("blacklist amdgpu"));
        assert!(line_blacklists_amdgpu("  blacklist   amdgpu"));
        assert!(!line_blacklists_amdgpu("blacklist amdgpufoo"));
        assert!(!line_blacklists_amdgpu("# blacklist amdgpu"));
    }

    #[test]
    fn mode_access_owner_group_other_precedence() {
        // crw-rw---- root render: a render member can write, others cannot.
        let groups = vec!["render".to_owned()];
        let (r, w) = mode_access("crw-rw----", "root", "render", "alice", &groups);
        assert_eq!((r, w), (Some(true), Some(true)));
        let (r, w) = mode_access("crw-rw----", "root", "render", "alice", &[]);
        assert_eq!((r, w), (Some(false), Some(false)));
    }
}
