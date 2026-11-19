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
    let policy_path = provision_dir.join("policy.yaml");
    std::fs::write(
        &policy_path,
        policy::render_provision(&policy::ProvisionPolicyInputs {
            name: &format!("{id}-provision"),
            mise_dir: &mise_dir,
            snapshot_dir: &snapshot_dir,
            store_dir: &store.root,
            resolver_workaround,
        }),
    )?;
    let sandbox = Sandbox {
        axis,
        policy_path: &policy_path,
        store: &store,
        env: provision_env(&store, &snapshot_dir, snapshot.lock.is_some())?,
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

/// The provisioning sandbox's whole environment (it starts from `env -i`).
fn provision_env(
    store: &Store,
    snapshot_dir: &Path,
    locked: bool,
) -> Result<Vec<(String, String)>> {
    let w = |name: &str| path_str(&store.work(name));
    let mut env = vec![
        ("HOME".into(), w("tmp")?),
        ("TMPDIR".into(), w("tmp")?),
        ("PATH".into(), "/usr/bin:/bin".into()),
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

/// One provisioning sandbox: the same policy and environment for every mise
/// command run in it.
struct Sandbox<'a> {
    axis: &'a Path,
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
            .arg("env")
            .arg("-i")
            .args(self.env.iter().map(|(k, v)| format!("{k}={v}")));
        if let Some(file) = stdout_to {
            cmd.args([
                "/bin/sh",
                "-c",
                "out=$1; shift; exec \"$@\" > \"$out\"",
                "sh",
            ])
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
    fn the_provisioning_env_disables_env_scripts_hooks_and_plugin_backends() {
        let store = Store::new(Path::new("/data/agents"));
        let env =
            provision_env(&store, Path::new("/data/agents/x/provision/snapshot"), true).unwrap();
        let get = |k: &str| {
            env.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("MISE_NO_ENV"), Some("1"));
        assert_eq!(get("MISE_NO_HOOKS"), Some("1"));
        assert_eq!(get("MISE_LOCKED"), Some("1"));
        let disabled = get("MISE_DISABLE_BACKENDS").unwrap();
        for backend in ["asdf", "vfox", "cargo", "npm"] {
            assert!(disabled.split(',').any(|b| b == backend), "{disabled}");
        }
        for allowed in mise_snapshot::ALLOWED_BACKENDS {
            assert!(!disabled.split(',').any(|b| b == *allowed), "{disabled}");
        }
        let unlocked = provision_env(&store, Path::new("/s/snapshot"), false).unwrap();
        assert!(!unlocked.iter().any(|(k, _)| k == "MISE_LOCKED"));
    }
}
