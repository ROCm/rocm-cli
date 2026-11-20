// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Toolchain provisioning before `rocm agents run` starts an agent (PROTOTYPE).
//!
//! The workspace's mise config is snapshotted (see [`super::mise_snapshot`])
//! and installed by a pinned, verified mise ([`super::mise_bin`]) in a
//! separate AXIS sandbox whose only writable path is the shared store under
//! the data root. Installs land in `store/mise/installs/<tool>/<version>`, so
//! two agents asking for the same tool version share one copy. The agent then
//! gets that directory read-only at the same absolute path, the tools' bin
//! dirs on `PATH`, and its own writable mise data dir for anything it installs
//! itself.
//!
//! Layout under `<data>/agents`:
//! - `tools/mise-<version>/mise` — the pinned binary; no sandbox can write it
//! - `store/mise/installs` — shared installs; read-only to agents
//! - `store/mise/work` — mise's data, cache, state and temp dirs while
//!   provisioning; never granted to an agent
//! - `store/mise-store.lock` — host-side lock that serializes provisioning
//! - `<id>/provision/{snapshot,original,policy.yaml}` — per agent

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::mise_snapshot::{self, Snapshot};
use super::{mise_bin, path_str, policy};

/// Whether `rocm agents run` provisions before starting the agent: only when
/// the workspace has a mise config, and never with `--no-provision`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    Provision,
    /// `--no-provision` was given although a config exists.
    SkippedByFlag,
    /// No mise config in the workspace.
    NothingToDo,
}

pub(crate) const fn decide(config_present: bool, no_provision: bool) -> Decision {
    match (config_present, no_provision) {
        (false, _) => Decision::NothingToDo,
        (true, true) => Decision::SkippedByFlag,
        (true, false) => Decision::Provision,
    }
}

/// Decides, and takes the snapshot only when provisioning: with
/// `--no-provision` the config is never parsed, so a config provisioning
/// would refuse cannot keep the agent from starting.
pub(crate) fn snapshot_for(
    workspace: &Path,
    no_provision: bool,
) -> Result<(Decision, Option<Snapshot>)> {
    let present = !mise_snapshot::read_workspace(workspace)?.is_empty();
    let decision = decide(present, no_provision);
    if decision != Decision::Provision {
        return Ok((decision, None));
    }
    let snapshot = mise_snapshot::take(workspace)
        .context(REMEDY)?
        .context("the workspace's mise config disappeared while it was read")?;
    Ok((decision, Some(snapshot)))
}

/// Appended to every provisioning refusal; `snapshot_for` honours the flag.
const REMEDY: &str = "the workspace's mise toolchain cannot be provisioned; \
                      fix its mise config, or pass --no-provision to start the agent without it";

/// What the agent sandbox needs from a finished provisioning.
pub(crate) struct Provisioned {
    /// Tool bin dirs, all inside [`Self::installs`], for the front of `PATH`.
    pub bin_dirs: Vec<PathBuf>,
    /// The shared store, granted read-only.
    pub installs: PathBuf,
    /// The pinned mise binary's directory, granted read-only and on `PATH`.
    pub mise_dir: PathBuf,
}

pub(crate) struct Store {
    pub root: PathBuf,
    pub installs: PathBuf,
    work: PathBuf,
    lock: PathBuf,
    tools: PathBuf,
}

impl Store {
    pub(crate) fn new(agents_root: &Path) -> Self {
        let root = agents_root.join("store").join("mise");
        Self {
            installs: root.join("installs"),
            work: root.join("work"),
            lock: agents_root.join("store").join("mise-store.lock"),
            tools: agents_root.join("tools"),
            root,
        }
    }

    fn work(&self, name: &str) -> PathBuf {
        self.work.join(name)
    }
}

#[derive(Debug, Deserialize)]
struct LsEntry {
    version: String,
    install_path: PathBuf,
    #[serde(default)]
    installed: bool,
}

/// Provisions `snapshot` for agent `id` and reports what was installed.
pub(crate) fn run(
    agents_root: &Path,
    axis: &Path,
    id: &str,
    instance_dir: &Path,
    snapshot: &Snapshot,
) -> Result<Provisioned> {
    for dir in ["data", "cache", "state", "config", "tmp"] {
        std::fs::create_dir_all(Store::new(agents_root).work(dir))?;
    }
    std::fs::create_dir_all(Store::new(agents_root).installs)?;
    // Canonical paths throughout: the agent sees the store at the same
    // absolute path mise installed into, and bin dirs are confined to it.
    let store = Store::new(&agents_root.canonicalize()?);
    let installs = store.installs.clone();
    let mise = mise_bin::ensure(&store.tools)?;
    let mise_dir = mise.parent().expect("binary has a dir").to_path_buf();

    let provision_dir = instance_dir.join("provision");
    let snapshot_dir = provision_dir.join("snapshot");
    snapshot.write(&snapshot_dir, &provision_dir.join("original"))?;
    let resolver_workaround = policy::axis_123_workaround_enabled();
    let host = HostAccess::discover(resolver_workaround, &store.tools.join("host-bin"))?;
    let policy_path = provision_dir.join("policy.yaml");
    std::fs::write(
        &policy_path,
        policy::render_provision(&policy::ProvisionPolicyInputs {
            name: &format!("{id}-provision"),
            mise_dir: &mise_dir,
            snapshot_dir: &snapshot_dir,
            store_dir: &store.root,
            host_read_only: &host.read_only,
            resolver_workaround,
        }),
    )?;
    let sandbox = Sandbox {
        axis,
        host: &host,
        policy_path: &policy_path,
        store: &store,
        env: provision_env(
            &store,
            &[&mise_dir, &host.bin_dir],
            &snapshot_dir,
            snapshot.lock.is_some(),
        )?,
        mise: &mise,
        snapshot_dir: &snapshot_dir,
    };

    // One provisioning at a time per store: `mise install --shared` into the
    // same directory from two sandboxes would race on the same version dir.
    let store_lock = rocm_core::FileLock::acquire(&store.lock)?;
    let before = installed_versions(&installs);
    eprintln!(
        "rocm: provisioning the toolchain in a sandbox (policy {}{})",
        policy_path.display(),
        if resolver_workaround {
            "; DNS via the ROCm/axis#123 resolver workaround"
        } else {
            ""
        }
    );
    let install_args = ["install", "--shared", &path_str(&installs)?];
    let status = sandbox
        .command(&install_args, None)
        .status()
        .with_context(|| format!("failed to start {}", axis.display()))?;
    if !status.success() {
        bail!(
            "provisioning failed (mise exit {}, its output is above); \
             pass --no-provision to start the agent without the toolchain",
            status
                .code()
                .map_or_else(|| "by signal".into(), |c| c.to_string())
        );
    }
    let ls = sandbox.capture(&["ls", "--current", "--json"])?;
    let bin_paths = sandbox.capture(&["bin-paths"])?;
    drop(store_lock);

    let bin_dirs = confine_bin_dirs(&bin_paths, &installs)?;
    let entries: std::collections::BTreeMap<String, Vec<LsEntry>> =
        serde_json::from_str(&ls).context("cannot parse `mise ls --json`")?;
    let mut report = crate::cli_report::ActionReport::new(format!(
        "rocm: toolchain provisioned from a snapshot of {} (sha256 {}) taken before the agent started",
        snapshot.file_names().join(", "),
        snapshot.digest()
    ));
    for (tool, versions) in &entries {
        for entry in versions {
            if !entry.installed || !entry.install_path.starts_with(&installs) {
                bail!(
                    "{tool} {} is not installed in the shared store after provisioning",
                    entry.version
                );
            }
            let key = (tool.clone(), entry.version.clone());
            let how = if before.contains(&key) {
                "reused from the shared store"
            } else {
                "downloaded now"
            };
            report = report.detail(format!("{tool} {}", entry.version), how);
        }
    }
    report = report
        .detail(
            "store",
            format!("{} (read-only to the agent)", installs.display()),
        )
        .detail(
            "lockfile",
            if snapshot.lock.is_some() {
                "mise.lock honoured (--locked: only its URLs and checksums)"
            } else {
                "none"
            },
        )
        .detail("snapshot", provision_dir.join("original").display());
    if !snapshot.ignored.is_empty() {
        report = report.detail("not used", snapshot.ignored.join(", "));
    }
    eprint!("{}", report.render());
    Ok(Provisioned {
        bin_dirs,
        installs,
        mise_dir,
    })
}

/// The host side of the provisioning sandbox: the paths it may read, the
/// programs on its command line (resolved: `/bin/sh` is a symlink), and
/// `bin_dir`, the sandbox's only `PATH` entry besides mise.
struct HostAccess {
    read_only: Vec<PathBuf>,
    env: PathBuf,
    sh: PathBuf,
    bin_dir: PathBuf,
}

impl HostAccess {
    /// `bin_dir` gets one symlink per allowed host program, by its usual name,
    /// so a `#!/usr/bin/env bash` script finds it. Putting `/usr/bin` on `PATH`
    /// instead would let mise *see* every host tool (Landlock does not hide a
    /// file's existence) and pick one, as it did with the host's `rustup`. The
    /// dir is outside the store, so the sandbox cannot repoint the links.
    fn discover(resolver_workaround: bool, bin_dir: &Path) -> Result<Self> {
        let mut programs = Vec::new();
        for program in policy::PROVISION_HOST_PROGRAMS {
            let resolved = Path::new(program)
                .canonicalize()
                .with_context(|| format!("{program} is needed to provision and was not found"))?;
            let name = Path::new(program).file_name().expect("program has a name");
            link(&resolved, &bin_dir.join(name))?;
            programs.push(resolved);
        }
        let existing = policy::PROVISION_HOST_READ_ONLY
            .iter()
            .filter_map(|p| Some((PathBuf::from(p), Path::new(p).canonicalize().ok()?)));
        let mut read_only = without_resolver_overlap(existing, resolver_workaround);
        read_only.extend(programs.iter().cloned());
        read_only.push(bin_dir.to_path_buf());
        let [env, sh] = <[PathBuf; 2]>::try_from(programs).expect("two programs");
        Ok(Self {
            read_only,
            env,
            sh,
            bin_dir: bin_dir.to_path_buf(),
        })
    }
}

/// Points `at` to `target`, replacing what was there in one rename so a
/// concurrent provisioning never sees it missing.
fn link(target: &Path, at: &Path) -> Result<()> {
    std::fs::create_dir_all(at.parent().expect("link has a dir"))?;
    if std::fs::read_link(at).is_ok_and(|current| current == target) {
        return Ok(());
    }
    let tmp = at.with_extension(format!("tmp-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, &tmp)?;
    #[cfg(not(unix))]
    bail!("toolchain provisioning is only supported on Linux");
    #[cfg(unix)]
    {
        std::fs::rename(&tmp, at)?;
        Ok(())
    }
}

/// `(path, resolved)` pairs to grant read-only. With the ROCm/axis#123
/// workaround on, a path resolving under its read-write resolver dir (here
/// `/etc/resolv.conf`, a symlink into it) is left out: AXIS refuses
/// overlapping read-only and read-write grants, and the read-write grant
/// already lets it be read.
fn without_resolver_overlap(
    paths: impl Iterator<Item = (PathBuf, PathBuf)>,
    resolver_workaround: bool,
) -> Vec<PathBuf> {
    paths
        .filter(|(_, resolved)| {
            !(resolver_workaround && resolved.starts_with(policy::AXIS_123_RESOLVER_DIR))
        })
        .map(|(path, _)| path)
        .collect()
}

/// The provisioning sandbox's whole environment (it starts from `env -i`).
fn provision_env(
    store: &Store,
    path: &[&Path],
    snapshot_dir: &Path,
    locked: bool,
) -> Result<Vec<(String, String)>> {
    let w = |name: &str| path_str(&store.work(name));
    let mut env = vec![
        ("HOME".into(), w("tmp")?),
        ("TMPDIR".into(), w("tmp")?),
        // mise and the allowed host programs only; no host program directory.
        (
            "PATH".into(),
            path.iter()
                .map(|dir| path_str(dir))
                .collect::<Result<Vec<_>>>()?
                .join(":"),
        ),
        ("MISE_DATA_DIR".into(), w("data")?),
        ("MISE_CACHE_DIR".into(), w("cache")?),
        ("MISE_STATE_DIR".into(), w("state")?),
        ("MISE_CONFIG_DIR".into(), w("config")?),
        // Nonexistent files: no user or system config takes part.
        (
            "MISE_GLOBAL_CONFIG_FILE".into(),
            path_str(&store.work("config").join("none.toml"))?,
        ),
        (
            "MISE_SYSTEM_CONFIG_FILE".into(),
            path_str(&store.work("config").join("none.toml"))?,
        ),
        (
            "MISE_CEILING_PATHS".into(),
            path_str(snapshot_dir.parent().expect("snapshot has a parent"))?,
        ),
        ("MISE_TRUSTED_CONFIG_PATHS".into(), path_str(snapshot_dir)?),
        // Belt and braces: the generated config carries none of these, but a
        // regression there must not turn into code execution. NO_ENV stops
        // `[env]` `_.source` scripts, which `mise install` otherwise runs;
        // NO_HOOKS stops `[hooks]`. Neither stops a tool's own `postinstall`
        // option (checked), which only the generated config keeps out.
        ("MISE_NO_ENV".into(), "1".into()),
        ("MISE_NO_HOOKS".into(), "1".into()),
        ("MISE_PARANOID".into(), "1".into()),
        // No usage pings to the mise versions host (`GET /t?tool=..&version=..
        // &os=..&arch=..`, seen in MISE_DEBUG output); version lists from it
        // are still used.
        (TELEMETRY_OFF.0.into(), TELEMETRY_OFF.1.into()),
        // mise's default `npm` launcher for node is a bash script that also
        // needs coreutils, none of which the sandbox has; without it, `npm`
        // is npm's own launcher, run by the installed node.
        ("MISE_NODE_NPM_SHIM".into(), "0".into()),
        (
            "MISE_DISABLE_BACKENDS".into(),
            mise_snapshot::DISABLED_BACKENDS.join(","),
        ),
        (
            "MISE_SHARED_INSTALL_DIRS".into(),
            path_str(&store.installs)?,
        ),
    ];
    if locked {
        // Download only what the lockfile names, checked against its
        // checksums; also keeps mise from rewriting the (read-only) lockfile.
        env.push(("MISE_LOCKED".into(), "1".into()));
    }
    Ok(env)
}

/// Turns off mise's install tracking; set for the agent's own mise too.
pub(crate) const TELEMETRY_OFF: (&str, &str) = ("MISE_USE_VERSIONS_HOST_TRACK", "0");

/// One provisioning sandbox: the same policy and environment for every mise
/// command run in it.
struct Sandbox<'a> {
    axis: &'a Path,
    host: &'a HostAccess,
    policy_path: &'a Path,
    store: &'a Store,
    env: Vec<(String, String)>,
    mise: &'a Path,
    snapshot_dir: &'a Path,
}

impl Sandbox<'_> {
    /// `axis run ... -- env -i ENV [sh -c 'exec "$@" > FILE'] mise -C SNAPSHOT ARGS`.
    fn command(&self, args: &[&str], stdout_to: Option<&Path>) -> Command {
        let mut cmd = Command::new(self.axis);
        cmd.args(["run", "--policy"])
            .arg(self.policy_path)
            .arg("--")
            .arg(&self.host.env)
            .arg("-i")
            .args(self.env.iter().map(|(k, v)| format!("{k}={v}")));
        if let Some(file) = stdout_to {
            cmd.arg(&self.host.sh)
                .args(["-c", "out=$1; shift; exec \"$@\" > \"$out\"", "sh"])
                .arg(file);
        }
        cmd.arg(self.mise)
            .arg("-C")
            .arg(self.snapshot_dir)
            .args(args)
            // AXIS makes the working directory an implicit read-write
            // workspace, so it must lie inside the store, the one writable path.
            .current_dir(self.store.work("tmp"))
            .stdin(Stdio::null());
        cmd
    }

    /// Runs `mise <args>` and returns its stdout. The output goes through a
    /// file in the store's temp dir, not the pipe: AXIS writes its own log
    /// lines to stdout when stdin is not a terminal.
    fn capture(&self, args: &[&str]) -> Result<String> {
        let out_file = self
            .store
            .work("tmp")
            .join(format!("rocm-capture-{}.out", super::random_hex(8)?));
        let status = self
            .command(args, Some(&out_file))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| format!("failed to start {}", self.axis.display()))?;
        let text = std::fs::read_to_string(&out_file);
        let _ = std::fs::remove_file(&out_file);
        if !status.success() {
            bail!("`mise {}` failed after provisioning", args.join(" "));
        }
        text.context("cannot read mise's output")
    }
}

/// `<tool>/<version>` pairs present in the store.
fn installed_versions(installs: &Path) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for tool in std::fs::read_dir(installs).into_iter().flatten().flatten() {
        for version in std::fs::read_dir(tool.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            out.insert((
                tool.file_name().to_string_lossy().into_owned(),
                version.file_name().to_string_lossy().into_owned(),
            ));
        }
    }
    out
}

/// The bin dirs mise reported, each required to resolve inside the store: the
/// list came out of a sandbox, so the host trusts none of it.
pub(crate) fn confine_bin_dirs(listing: &str, installs: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for line in listing.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let dir = Path::new(line)
            .canonicalize()
            .with_context(|| format!("mise reported a bin dir that does not exist: {line}"))?;
        if !dir.starts_with(installs) {
            bail!("mise reported a bin dir outside the shared store: {line}");
        }
        dirs.push(dir);
    }
    Ok(dirs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provisioning_runs_only_for_a_config_and_without_the_flag() {
        assert_eq!(decide(true, false), Decision::Provision);
        assert_eq!(decide(true, true), Decision::SkippedByFlag);
        assert_eq!(decide(false, false), Decision::NothingToDo);
        assert_eq!(decide(false, true), Decision::NothingToDo);
    }

    #[test]
    fn no_provision_starts_even_when_the_config_would_be_refused() {
        let ws = std::env::temp_dir().join(format!("rocm-prov-flag-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(
            ws.join("mise.toml"),
            "[tools]\njq = { version = \"1\", postinstall = \"x\" }\n",
        )
        .unwrap();
        let e = format!("{:#}", snapshot_for(&ws, false).unwrap_err());
        assert!(e.contains("pass --no-provision"), "{e}");
        let (decision, snapshot) = snapshot_for(&ws, true).unwrap();
        assert_eq!(decision, Decision::SkippedByFlag);
        assert!(snapshot.is_none());
        std::fs::remove_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        assert_eq!(snapshot_for(&ws, false).unwrap().0, Decision::NothingToDo);
        std::fs::remove_dir_all(&ws).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn bin_dirs_outside_the_store_are_refused_even_through_a_symlink() {
        let root = std::env::temp_dir().join(format!("rocm-prov-bins-{}", std::process::id()));
        let installs = root.join("installs");
        std::fs::create_dir_all(installs.join("jq/1.7.1")).unwrap();
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), installs.join("jq/escape")).unwrap();
        let installs = installs.canonicalize().unwrap();
        let ok = confine_bin_dirs(
            &format!("{}\n", installs.join("jq/1.7.1").display()),
            &installs,
        )
        .unwrap();
        assert_eq!(ok, [installs.join("jq/1.7.1")]);
        let e = confine_bin_dirs("/usr/bin\n", &installs)
            .unwrap_err()
            .to_string();
        assert!(e.contains("outside the shared store"), "{e}");
        let e = confine_bin_dirs(&installs.join("jq/escape").display().to_string(), &installs)
            .unwrap_err()
            .to_string();
        assert!(e.contains("outside the shared store"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_host_path_under_the_rw_resolver_grant_is_dropped_only_with_the_workaround() {
        let paths = || {
            [
                ("/etc/resolv.conf", "/run/systemd/resolve/stub-resolv.conf"),
                ("/etc/ssl", "/etc/ssl"),
            ]
            .into_iter()
            .map(|(p, r)| (PathBuf::from(p), PathBuf::from(r)))
        };
        assert_eq!(
            without_resolver_overlap(paths(), true),
            [PathBuf::from("/etc/ssl")]
        );
        assert_eq!(
            without_resolver_overlap(paths(), false),
            [PathBuf::from("/etc/resolv.conf"), PathBuf::from("/etc/ssl")]
        );
    }

    #[test]
    fn the_provisioning_env_disables_env_scripts_hooks_and_plugin_backends() {
        let store = Store::new(Path::new("/data/agents"));
        let env = provision_env(
            &store,
            &[
                Path::new("/data/agents/tools/mise-1"),
                Path::new("/data/agents/tools/host-bin"),
            ],
            Path::new("/data/agents/x/provision/snapshot"),
            true,
        )
        .unwrap();
        let get = |k: &str| {
            env.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("MISE_NO_ENV"), Some("1"));
        assert_eq!(get("MISE_NO_HOOKS"), Some("1"));
        assert_eq!(get("MISE_LOCKED"), Some("1"));
        assert_eq!(get("MISE_USE_VERSIONS_HOST_TRACK"), Some("0"));
        assert_eq!(
            get("PATH"),
            Some("/data/agents/tools/mise-1:/data/agents/tools/host-bin")
        );
        let disabled = get("MISE_DISABLE_BACKENDS").unwrap();
        for backend in ["asdf", "vfox", "cargo", "npm"] {
            assert!(disabled.split(',').any(|b| b == backend), "{disabled}");
        }
        for allowed in mise_snapshot::ALLOWED_BACKENDS {
            assert!(!disabled.split(',').any(|b| b == *allowed), "{disabled}");
        }
        let unlocked =
            provision_env(&store, &[Path::new("/m")], Path::new("/s/snapshot"), false).unwrap();
        assert!(!unlocked.iter().any(|(k, _)| k == "MISE_LOCKED"));
    }
}
