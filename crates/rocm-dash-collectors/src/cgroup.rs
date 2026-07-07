// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! PID → docker container id resolution via `/proc/<pid>/cgroup`.
//!
//! amd-smi reports the **host** PIDs of GPU compute processes. For a
//! tensor-parallel vLLM server the workers are descendants of the container
//! init PID, so a naive `pid == container.pid` equality fails. The robust join
//! is: GPU-process host PID → read `/proc/<pid>/cgroup` → extract the docker
//! 64-hex container id → match `Instance.container_id`.
//!
//! The parser is pure (string in, `Option` out) and fully testable on a
//! no-container box; only [`container_id_for_pid`] touches `/proc` and it is
//! `cfg`-gated for linux with a non-linux stub. std::fs only — no bollard,
//! no async.

/// Whether `c` is a lowercase hex digit (`0-9a-f`). Docker/containerd ids are
/// lowercase hex; uppercase and other characters act as token delimiters.
const fn is_lower_hex(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, 'a'..='f')
}

/// Extract a docker/containerd 64-hex container id from cgroup file contents.
///
/// Handles the common layouts by scanning for the first run of exactly 64
/// lowercase-hex characters — the affixes (`docker-`, `cri-containerd-`,
/// `.scope`, path separators) are non-hex and delimit the token:
///  - cgroup v2: `0::/system.slice/docker-<64hex>.scope`
///  - cgroup v1: `.../docker/<64hex>` (any controller line)
///  - k8s/containerd: `cri-containerd-<64hex>.scope` or `kubepods/.../<64hex>`
///
/// Returns `None` when no 64-hex token is present (e.g. a non-container
/// `0::/user.slice/...` line). The returned id is the full 64-char lowercase
/// hex, directly comparable to docker discovery's `Instance.container_id`.
pub fn parse_container_id_from_cgroup(contents: &str) -> Option<String> {
    contents
        .split(|c: char| !is_lower_hex(c))
        .find(|tok| tok.len() == 64)
        .map(str::to_string)
}

/// Resolve the docker container id owning host process `pid` by reading `/proc/<pid>/cgroup`.
///
/// Returns `None` on any read error (process gone,
/// permission denied) or when the cgroup names no container — never panics.
#[cfg(target_os = "linux")]
pub fn container_id_for_pid(pid: u32) -> Option<String> {
    let contents = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
    parse_container_id_from_cgroup(&contents)
}

/// Non-linux stub: there is no `/proc/<pid>/cgroup`, so attribution is never
/// available off linux. Always `None`.
#[cfg(not(target_os = "linux"))]
pub const fn container_id_for_pid(_pid: u32) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // A clean 64-char lowercase-hex id (16-char block × 4).
    const ID: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn assert_is_64_hex(s: &str) {
        assert_eq!(s.len(), 64, "container id must be 64 chars");
        assert!(
            s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "container id must be lowercase hex"
        );
    }

    #[test]
    fn extracts_from_cgroup_v2_docker_scope() {
        let line = format!("0::/system.slice/docker-{ID}.scope\n");
        let got = parse_container_id_from_cgroup(&line).unwrap();
        assert_eq!(got, ID);
        assert_is_64_hex(&got);
    }

    #[test]
    fn extracts_from_cgroup_v1_docker_path() {
        let contents = format!("12:devices:/docker/{ID}\n11:cpuset:/docker/{ID}\n");
        let got = parse_container_id_from_cgroup(&contents).unwrap();
        assert_eq!(got, ID);
        assert_is_64_hex(&got);
    }

    #[test]
    fn extracts_from_k8s_cri_containerd_scope() {
        let line = format!(
            "0::/kubepods.slice/kubepods-burstable.slice/\
             kubepods-burstable-podabc.slice/cri-containerd-{ID}.scope\n"
        );
        let got = parse_container_id_from_cgroup(&line).unwrap();
        assert_eq!(got, ID);
        assert_is_64_hex(&got);
    }

    #[test]
    fn extracts_from_k8s_kubepods_path() {
        let line = format!("11:memory:/kubepods/besteffort/pod1234-5678/{ID}\n");
        let got = parse_container_id_from_cgroup(&line).unwrap();
        assert_eq!(got, ID);
    }

    #[test]
    fn non_container_cgroup_returns_none() {
        let contents = "0::/user.slice/user-1000.slice/session-2.scope\n\
                        12:devices:/init.scope\n";
        assert_eq!(parse_container_id_from_cgroup(contents), None);
        // A 32-hex pod uuid fragment must not false-match a 64-hex id.
        assert_eq!(
            parse_container_id_from_cgroup("0::/short/0123456789abcdef0123456789abcdef\n"),
            None
        );
    }

    #[test]
    fn container_id_for_pid_unreadable_is_none_no_panic() {
        // A PID that cannot exist → unreadable /proc path → None (no panic).
        // On non-linux this exercises the stub; on linux, the read error path.
        assert_eq!(container_id_for_pid(u32::MAX), None);
    }
}
