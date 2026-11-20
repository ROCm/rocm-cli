// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Per-instance AXIS policy for `rocm agents run` (PROTOTYPE).
//!
//! Names no `~/...` path: AXIS relocates (moves and symlinks) a home path a
//! policy grants read-write, which mutates the user's real home and makes two
//! concurrent sandboxes of one policy share state. The agent gets its own
//! `HOME` and temp dir under the instance directory instead, and the user's
//! own tool directories are granted read-only by absolute path.

use std::path::{Path, PathBuf};

pub(crate) struct PolicyInputs<'a> {
    pub name: &'a str,
    pub agent_bin: &'a Path,
    pub home: &'a Path,
    pub tmp: &'a Path,
    /// The user's own tool directories that exist (see [`user_tool_dirs`]),
    /// granted read-only.
    pub tool_dirs: &'a [PathBuf],
    /// The shared toolchain store and the mise binary dir, granted read-only
    /// when the workspace was provisioned: the agent runs the tools but can
    /// never change what another agent runs.
    pub provisioned: &'a [PathBuf],
}

/// The provisioning sandbox: runs `mise install --shared` for one agent.
pub(crate) struct ProvisionPolicyInputs<'a> {
    pub name: &'a str,
    /// Directory holding the pinned mise binary (read-only).
    pub mise_dir: &'a Path,
    /// The generated config and lockfile (read-only).
    pub snapshot_dir: &'a Path,
    /// The shared store: the only writable path (installs plus mise's own
    /// data, cache and temp dirs; the sandbox's working directory is in it).
    pub store_dir: &'a Path,
    /// The host paths the provisioning sandbox may read: the ones in
    /// [`PROVISION_HOST_READ_ONLY`] that exist, plus the resolved
    /// [`PROVISION_HOST_PROGRAMS`]. Never all of `/usr`, `/bin` or `/etc`.
    pub host_read_only: &'a [PathBuf],
    /// Grant [`AXIS_123_RESOLVER_DIR`] read-write; see there.
    pub resolver_workaround: bool,
}

/// What the pinned mise needs from the host for a network install, and no
/// more: shared libraries, TLS roots, resolver config and a few devices. No
/// program directory (`/usr/bin`, `/bin`, `/usr/local/bin`) is granted, so
/// mise cannot pick up a host tool: given `/usr/bin`, it used the host's
/// `rustup` for `rust` instead of installing its own, which made the shared
/// store depend on whichever host provisioned it first. A path that does not
/// exist on this host is left out.
pub(crate) const PROVISION_HOST_READ_ONLY: &[&str] = &[
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/etc/ld.so.cache",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
    // `/etc/ssl/certs/*` are symlinks into it on Debian and Ubuntu.
    "/usr/share/ca-certificates",
    // CAs an administrator added (`update-ca-certificates`), linked from
    // `/etc/ssl/certs` too; a network that inspects TLS needs them.
    "/usr/local/share/ca-certificates",
    "/etc/resolv.conf",
    "/etc/nsswitch.conf",
    "/etc/hosts",
    "/etc/host.conf",
    "/etc/gai.conf",
    "/dev/null",
    "/dev/zero",
    "/dev/urandom",
    "/dev/random",
    "/proc",
];

/// The two host programs the provisioning sandbox may run, granted as single
/// files at their resolved paths: `env`, which starts mise with a cleared
/// environment, and `sh`, which redirects mise's output to a file (see
/// `provision::Sandbox::capture`). No other host program is reachable; a
/// backend that needs more (mise's bash `npm` launcher also wants coreutils)
/// is configured not to, as with `MISE_NODE_NPM_SHIM=0` in the provisioning
/// environment.
pub(crate) const PROVISION_HOST_PROGRAMS: [&str; 2] = ["/usr/bin/env", "/bin/sh"];

/// WORKAROUND for ROCm/axis#123 (DNS does not resolve in native AXIS
/// sandboxes): glibc resolves through systemd-resolved's socket in this
/// directory, and AXIS lets a sandbox connect to a host socket only beneath a
/// read-write grant, so a read-only grant is not enough (checked: mise fails
/// with "dns error" then). This is wrong in principle: it grants write access
/// to a host system directory. DAC still blocks writes in practice (the files
/// are owned by systemd-resolve), but the grant itself should not exist. It is
/// used ONLY for the provisioning sandbox, never for an agent, and goes away
/// once an AXIS release with the fix is required.
pub(crate) const AXIS_123_RESOLVER_DIR: &str = "/run/systemd/resolve";
/// Host env var that turns the workaround off (`0`) for an AXIS that already
/// carries the fix; any other value, or unset, leaves it on.
pub(crate) const AXIS_123_WORKAROUND_ENV: &str = "ROCM_AGENTS_AXIS123_RESOLVER_GRANT";

pub(crate) fn axis_123_workaround_enabled() -> bool {
    std::env::var(AXIS_123_WORKAROUND_ENV).map_or(true, |v| v != "0")
}

/// System paths the agent may read. `/proc` is safe to grant: Landlock's
/// ptrace scoping keeps other processes' `environ`/`mem` unreadable.
const SYSTEM_READ_ONLY: &[&str] = &[
    "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/dev", "/proc",
];

/// Per-agent resource limits. Without a `process` section AXIS applies its own
/// defaults, which include an 80% CPU quota nobody chose; these are stated so
/// the policy file shows what the agent is held to. `cpu_rate_percent: 0`
/// means no CPU quota.
pub(crate) const MAX_PROCESSES: u32 = 32;
pub(crate) const MAX_MEMORY_MB: u64 = 8192;
pub(crate) const CPU_RATE_PERCENT: u32 = 0;

/// `~/.local/bin` and `~/bin` under `host_home`, the ones that exist. Read-only:
/// the agent can run the user's tools but not change them. A symlink in them
/// that points elsewhere (most do) only runs if its target is readable too.
pub(crate) fn user_tool_dirs(host_home: &Path) -> Vec<PathBuf> {
    [".local/bin", "bin"]
        .iter()
        .map(|dir| host_home.join(dir))
        .filter(|dir| dir.is_dir())
        .collect()
}

pub(crate) fn render(inputs: &PolicyInputs<'_>) -> String {
    let q = |s: &str| serde_json::to_string(s).expect("string serializes");
    let qp = |p: &Path| q(&p.to_string_lossy());
    let mut ro: Vec<String> = SYSTEM_READ_ONLY.iter().map(|s| q(s)).collect();
    ro.push(qp(inputs.agent_bin));
    ro.extend(inputs.tool_dirs.iter().map(|dir| qp(dir)));
    ro.extend(inputs.provisioned.iter().map(|dir| qp(dir)));
    let rw = [q("{workspace}"), qp(inputs.home), qp(inputs.tmp)];
    format!(
        "# Generated by `rocm agents run`; one file per agent instance.\n\
         version: 1\n\
         name: {name}\n\
         runtime:\n  provider: axis_native\n\
         filesystem:\n  read_only: [{ro}]\n  read_write: [{rw}]\n\
         process:\n  max_processes: {MAX_PROCESSES}\n  max_memory_mb: {MAX_MEMORY_MB}\n  \
         cpu_rate_percent: {CPU_RATE_PERCENT}\n\
         # Egress is NOT restricted yet: strict proxy mode needs the setuid\n\
         # axis-netns-helper or CAP_NET_ADMIN. The provider key never enters\n\
         # the sandbox either way (see the rocm credential proxy).\n\
         network:\n  mode: allow\n",
        name = q(inputs.name),
        ro = ro.join(", "),
        rw = rw.join(", "),
    )
}

pub(crate) fn render_provision(inputs: &ProvisionPolicyInputs<'_>) -> String {
    let q = |s: &str| serde_json::to_string(s).expect("string serializes");
    let qp = |p: &Path| q(&p.to_string_lossy());
    let mut ro: Vec<String> = inputs.host_read_only.iter().map(|p| qp(p)).collect();
    ro.push(qp(inputs.mise_dir));
    ro.push(qp(inputs.snapshot_dir));
    let mut rw = vec![qp(inputs.store_dir)];
    let mut note = String::new();
    if inputs.resolver_workaround {
        rw.push(q(AXIS_123_RESOLVER_DIR));
        note = format!(
            "# {AXIS_123_RESOLVER_DIR} is a WORKAROUND for ROCm/axis#123 (no DNS in\n\
             # native sandboxes); provisioning only. {AXIS_123_WORKAROUND_ENV}=0 drops it.\n"
        );
    }
    format!(
        "# Generated by `rocm agents run`: provisions the agent's toolchain into\n\
         # the shared store, the only writable path. Host access is limited to\n\
         # libraries, TLS and resolver config: no host program directory.\n\
         version: 1\n\
         name: {name}\n\
         runtime:\n  provider: axis_native\n\
         {note}\
         filesystem:\n  read_only: [{ro}]\n  read_write: [{rw}]\n\
         process:\n  max_processes: {MAX_PROCESSES}\n  max_memory_mb: {MAX_MEMORY_MB}\n  \
         cpu_rate_percent: {CPU_RATE_PERCENT}\n\
         network:\n  mode: allow\n",
        name = q(inputs.name),
        ro = ro.join(", "),
        rw = rw.join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(tool_dirs: &[PathBuf]) -> String {
        rendered_with(tool_dirs, &[])
    }

    fn rendered_with(tool_dirs: &[PathBuf], provisioned: &[PathBuf]) -> String {
        render(&PolicyInputs {
            name: "claude-code-ab12",
            agent_bin: Path::new("/opt/claude"),
            home: Path::new("/data/agents/x/home"),
            tmp: Path::new("/data/agents/x/tmp"),
            tool_dirs,
            provisioned,
        })
    }

    fn line(text: &str, key: &str) -> String {
        text.lines()
            .find(|l| l.trim_start().starts_with(key))
            .unwrap()
            .to_owned()
    }

    fn provision(resolver_workaround: bool) -> String {
        render_provision(&ProvisionPolicyInputs {
            name: "provision-ab12",
            mise_dir: Path::new("/data/agents/tools/mise-1"),
            snapshot_dir: Path::new("/data/agents/x/provision/snapshot"),
            store_dir: Path::new("/data/agents/store/mise"),
            host_read_only: &[
                PathBuf::from("/usr/lib"),
                PathBuf::from("/etc/ssl"),
                PathBuf::from("/usr/bin/dash"),
            ],
            resolver_workaround,
        })
    }

    #[test]
    fn the_agent_gets_the_store_read_only_and_never_the_resolver_dir() {
        let store = PathBuf::from("/data/agents/store/mise/installs");
        let text = rendered_with(&[], &[store, PathBuf::from("/data/agents/tools/mise-1")]);
        assert!(
            line(&text, "read_only:").contains("\"/data/agents/store/mise/installs\""),
            "{text}"
        );
        assert!(
            line(&text, "read_only:").contains("\"/data/agents/tools/mise-1\""),
            "{text}"
        );
        assert!(!line(&text, "read_write:").contains("store"), "{text}");
        assert!(!text.contains(AXIS_123_RESOLVER_DIR), "{text}");
    }

    #[test]
    fn provisioning_writes_only_the_store_and_reads_the_snapshot() {
        let text = provision(false);
        assert_eq!(
            line(&text, "read_write:").trim(),
            "read_write: [\"/data/agents/store/mise\"]",
            "{text}"
        );
        let ro = line(&text, "read_only:");
        assert!(
            ro.contains("\"/data/agents/x/provision/snapshot\""),
            "{text}"
        );
        assert!(ro.contains("\"/data/agents/tools/mise-1\""), "{text}");
        assert!(!text.contains("{workspace}"), "{text}");
        assert!(!text.contains(AXIS_123_RESOLVER_DIR), "{text}");
        assert!(
            ro.contains("\"/usr/lib\", \"/etc/ssl\", \"/usr/bin/dash\""),
            "{text}"
        );
        for broad in [
            "\"/usr\"",
            "\"/usr/bin\"",
            "\"/bin\"",
            "\"/etc\"",
            "\"/dev\"",
        ] {
            assert!(!ro.contains(broad), "{broad} granted: {text}");
        }
    }

    #[test]
    fn the_provisioning_host_allowlist_holds_no_program_directory() {
        for path in PROVISION_HOST_READ_ONLY {
            for dir in [
                "/usr",
                "/usr/bin",
                "/bin",
                "/sbin",
                "/usr/sbin",
                "/usr/local",
                "/etc",
                "/dev",
                "/home",
                "/opt",
                "/snap",
            ] {
                assert_ne!(*path, dir);
            }
            assert!(!path.contains("bin"), "{path}");
        }
    }

    #[test]
    fn the_axis_123_resolver_grant_is_read_write_and_only_when_switched_on() {
        let text = provision(true);
        assert_eq!(
            line(&text, "read_write:").trim(),
            "read_write: [\"/data/agents/store/mise\", \"/run/systemd/resolve\"]",
            "{text}"
        );
        assert!(text.contains("ROCm/axis#123"), "{text}");
    }

    #[test]
    fn policy_grants_no_home_relative_paths() {
        let text = rendered(&[]);
        assert!(!text.contains("\"~"), "{text}");
        assert!(
            text.contains(
                "read_write: [\"{workspace}\", \"/data/agents/x/home\", \"/data/agents/x/tmp\"]"
            ),
            "{text}"
        );
        assert!(text.contains("\"/opt/claude\""), "{text}");
        assert!(text.contains("provider: axis_native"), "{text}");
    }

    #[test]
    fn tool_dirs_are_read_only_and_never_writable() {
        let text = rendered(&[PathBuf::from("/home/u/.local/bin")]);
        assert!(
            line(&text, "read_only:").contains("\"/home/u/.local/bin\""),
            "{text}"
        );
        assert!(!line(&text, "read_write:").contains(".local/bin"), "{text}");
    }

    #[test]
    fn the_process_limits_are_stated_with_no_cpu_quota() {
        let text = rendered(&[]);
        assert!(
            text.contains(
                "process:\n  max_processes: 32\n  max_memory_mb: 8192\n  cpu_rate_percent: 0\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn only_tool_dirs_that_exist_are_granted() {
        let home =
            std::env::temp_dir().join(format!("rocm-agents-tooldirs-{}", std::process::id()));
        std::fs::create_dir_all(home.join(".local/bin")).unwrap();
        assert_eq!(user_tool_dirs(&home), [home.join(".local/bin")]);
        std::fs::create_dir_all(home.join("bin")).unwrap();
        assert_eq!(
            user_tool_dirs(&home),
            [home.join(".local/bin"), home.join("bin")]
        );
        std::fs::remove_dir_all(&home).unwrap();
    }
}
