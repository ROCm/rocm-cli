// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Prepare the shared managed runtime the GPU E2E lanes serve against.
//!
//! Nearly every GPU serve scenario points its `data/runtimes` at ONE shared,
//! pre-warmed runtime tree (see `E2E_SHARED_RUNTIMES_DIR` and
//! `E2eWorld::use_shared_runtimes`) so a multi-GiB `install sdk` happens once per
//! runner instead of once per scenario. The lanes used to guard that install on
//! directory existence alone:
//!
//! ```text
//! if [ ! -d "$E2E_SHARED_RUNTIMES_DIR/registry" ]; then ... rocm install sdk ...; fi
//! ```
//!
//! which never reinstalls. The tree persists on the runner's PVC, so after the
//! first run ever the pre-warm was a permanent no-op and every lane kept serving
//! against whatever runtime happened to be installed that day — 16 days stale on
//! both MI300X runners when this was measured (EAI-8057). Drift between the
//! shared tree and what a fresh install produces was therefore untested, and
//! widened silently.
//!
//! This keeps the cache and invalidates it only when the channel index actually
//! publishes something newer, reusing the primitives the CLI already ships
//! rather than reimplementing version resolution in workflow shell:
//!
//! * `rocm update` reports, per installed runtime, `status=up_to_date |
//!   update_available | ahead_of_index` by comparing against the channel index.
//! * `rocm update --apply --runtime <key> --activate` installs the newer runtime
//!   SIDE BY SIDE and makes it the default. Side-by-side matters: `install sdk`
//!   bakes ABSOLUTE paths into the runtime manifest, so a runtime must be created
//!   in its final location and never moved afterwards.
//! * `rocm storage remove-old-installs --keep N` bounds the resulting multi-version
//!   cache with a per-channel/format/family retention policy.
//!
//! Living in xtask rather than the workflows is deliberate: the pre-warm block is
//! duplicated across multiple jobs in two shells (bash on the Linux lanes,
//! PowerShell on Strix Windows), so the decision logic would otherwise drift across
//! copies and be untestable. Same reasoning as `e2e.rs` — the recipe belongs in
//! one cross-platform place instead of a shell wrapper.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths::{binary_name, release_binary_dir, workspace_root};

/// What the pre-warm should do with the shared tree, given the current
/// `rocm update` report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// No managed runtime for this channel yet — do the cold `install sdk`.
    Install,
    /// The channel index has a newer version than the installed one; install it
    /// alongside and activate it.
    Update { runtime_key: String },
    /// The tree is current, or its freshness could not be established. Serve
    /// against what is already there.
    Reuse { reason: String },
}

/// Decide from a `rocm update` report what the pre-warm should do for `channel`.
///
/// Pure so every branch is unit-testable without a GPU, a network, or an install.
///
/// The bias is deliberately conservative: anything this cannot read as "a newer
/// version exists for our channel" resolves to [`Decision::Reuse`]. A GPU lane
/// must not be turned red, nor a multi-GiB download triggered, because the
/// package index was briefly unreachable — `rocm update` reports that per runtime
/// as `status=error`, and an offline runner would otherwise reinstall on every run.
#[must_use]
pub fn decide(update_report: &str, channel: &str) -> Decision {
    // The empty-registry wording from `render_update_report`. Checked before the
    // per-runtime scan because there are no `runtime` lines at all in that case.
    if update_report.contains("managed runtimes: none") {
        return Decision::Install;
    }

    let runtimes: Vec<RuntimeLine> = update_report
        .lines()
        .filter_map(RuntimeLine::parse)
        .collect();

    // A degraded `status=error` line omits `channel=` because resolution failed
    // before the renderer had a plan. It proves a runtime exists but cannot be
    // attributed safely, so the conservative choice is reuse, not a fresh
    // multi-GiB install. A later healthy probe will identify the channel.
    // In a mixed-channel tree the error may belong to another channel, but the
    // report has discarded that identity. Reuse remains the safe floor until a
    // healthy probe can distinguish "missing channel" from "unknown freshness".
    if !runtimes
        .iter()
        .any(|line| line.channel.as_deref() == Some(channel))
    {
        if runtimes
            .iter()
            .any(|line| line.channel.is_none() && line.status.as_deref() == Some("error"))
        {
            return Decision::Reuse {
                reason: "could not establish runtime freshness; leaving the shared tree untouched"
                    .to_owned(),
            };
        }

        // Nothing installed for THIS channel. The tree may still hold another
        // channel's runtime, so install this one.
        return Decision::Install;
    }

    if let Some(stale) = runtimes
        .iter()
        .filter(|line| line.channel.as_deref() == Some(channel))
        .find(|line| line.status.as_deref() == Some("update_available"))
    {
        return Decision::Update {
            runtime_key: stale.runtime_key.clone(),
        };
    }

    // `ahead_of_index` means the installed runtime is NEWER than anything the
    // index offers (a hand-placed or pinned build). Reuse it rather than
    // "updating" backwards.
    let reason = if runtimes
        .iter()
        .filter(|line| line.channel.as_deref() == Some(channel))
        .any(|line| line.status.as_deref() == Some("up_to_date"))
    {
        "runtime is up to date with the channel index"
    } else if runtimes
        .iter()
        .filter(|line| line.channel.as_deref() == Some(channel))
        .any(|line| line.status.as_deref() == Some("ahead_of_index"))
    {
        "installed runtime is ahead of the channel index"
    } else {
        // Only `status=error` (or a shape this does not recognise) remains.
        "could not establish runtime freshness; leaving the shared tree untouched"
    };
    Decision::Reuse {
        reason: reason.to_owned(),
    }
}

/// A managed runtime in the shared tree that records an install root somewhere
/// else, so serving against it cannot work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoisonedRuntime {
    pub runtime_key: String,
    pub format: String,
    pub install_root: String,
}

/// Managed runtimes in the pre-warm tree whose recorded install root is not in
/// that tree.
///
/// Pure so every branch is unit-testable without a GPU, a network, or an install
/// — same contract as [`decide`].
///
/// A scenario reaches the shared tree through a symlink at its own
/// `data/runtimes` (`E2eWorld::use_shared_runtimes`). `install sdk` run that way
/// writes the *link's* path — a per-scenario temp dir — into the manifest that
/// lands in the SHARED registry, and into the venv's console-script shebangs. Once
/// the scenario's temp dir is gone the shared runtime records a path that no
/// longer exists, and every later run that resolves it fails.
///
/// The signal is deliberately "install root is outside the tree" and NOT
/// `status=unusable`. Unusable has many causes — a missing `rocm_sdk` probe block
/// alone reports it — and this function's caller DELETES what it returns. Keying
/// a multi-GiB delete off a status that a future tightening of
/// `validate_runtime_manifest_for_activation` could start emitting for healthy
/// runtimes is not a trade worth making. An out-of-tree install root, for a
/// runtime this tree is supposed to own, has exactly one cause.
///
/// `mode=read-only` is exempt: `runtimes import` / `runtimes adopt` record an
/// external folder on purpose, and that is what read-only means.
#[must_use]
pub fn assess(runtimes_list_report: &str, runtimes_dir: &Path) -> Vec<PoisonedRuntime> {
    // The recorded root of a poisoned runtime does not exist, so it cannot be
    // canonicalized. Compare textually instead, against both the path as given and
    // its resolved form, so a caller that passes a path through a symlinked parent
    // still matches the roots the CLI wrote.
    let mut roots = vec![runtimes_dir.to_path_buf()];
    if let Ok(resolved) = runtimes_dir.canonicalize()
        && resolved != *runtimes_dir
    {
        roots.push(resolved);
    }

    let lines: Vec<&str> = runtimes_list_report.lines().collect();
    let mut poisoned = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(entry) = RuntimeEntry::parse(line) else {
            continue;
        };
        if entry.read_only {
            continue;
        }
        // `install_root:` is written immediately under its entry by
        // `render_runtimes_text`. Without it there is nothing to judge, and
        // guessing a path we are about to delete is not acceptable.
        let Some(install_root) = lines
            .get(index + 1)
            .and_then(|next| next.trim().strip_prefix("install_root: "))
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        if roots
            .iter()
            .any(|root| Path::new(install_root).starts_with(root))
        {
            continue;
        }
        poisoned.push(PoisonedRuntime {
            runtime_key: entry.runtime_key,
            format: entry.format,
            install_root: install_root.to_owned(),
        });
    }
    poisoned
}

/// One `  {marker} <key> runtime_id=… format=… mode=… status=…` line from
/// `rocm runtimes list`.
///
/// `status=` is last and its value carries spaces and parentheses
/// (`unusable (install root is missing: /x)`), so it is not read here — every
/// field this needs appears before it and is space-free.
struct RuntimeEntry {
    runtime_key: String,
    format: String,
    read_only: bool,
}

impl RuntimeEntry {
    fn parse(line: &str) -> Option<Self> {
        let mut fields = line.split_whitespace();
        let first = fields.next()?;
        // The active/rollback markers are separate tokens; anything else is the key.
        let runtime_key = if matches!(first, "*" | "-") {
            fields.next()?
        } else {
            first
        };
        let mut format = None;
        let mut mode = None;
        let mut is_entry = false;
        for field in fields {
            match field.split_once('=') {
                Some(("runtime_id", _)) => is_entry = true,
                Some(("format", value)) => format = Some(value.to_owned()),
                Some(("mode", value)) => mode = Some(value.to_owned()),
                _ => {}
            }
        }
        // `runtime_id=` and `mode=` together separate a real entry from the
        // header lines, which use `active_runtime_id:` / `registry:` shapes.
        if !is_entry {
            return None;
        }
        Some(Self {
            runtime_key: runtime_key.to_owned(),
            format: format?,
            read_only: mode.as_deref() == Some("read-only"),
        })
    }
}

/// One `  runtime <key> format=… channel=… … status=…` line from `rocm update`.
///
/// Both shapes that renderer emits are handled: the full report line, and the
/// degraded `runtime <key> format=… status=error message=…` line. `message=` is
/// free text that may contain spaces, but it is last and only `channel`/`status`
/// are read, so a whitespace split is sufficient — trailing words of the message
/// simply carry no `=` and are ignored.
struct RuntimeLine {
    runtime_key: String,
    channel: Option<String>,
    status: Option<String>,
}

impl RuntimeLine {
    fn parse(line: &str) -> Option<Self> {
        let rest = line.trim().strip_prefix("runtime ")?;
        let mut fields = rest.split_whitespace();
        let runtime_key = fields.next()?.to_owned();
        let mut channel = None;
        let mut status = None;
        for field in fields {
            match field.split_once('=') {
                Some(("channel", value)) => channel = Some(value.to_owned()),
                Some(("status", value)) => status = Some(value.to_owned()),
                _ => {}
            }
        }
        Some(Self {
            runtime_key,
            channel,
            status,
        })
    }
}

/// Bring the shared pre-warm tree at `prewarm_dir` to the newest runtime the
/// `channel` index offers, keeping `keep` recent installs per channel/format/family.
pub fn run(channel: &str, keep: usize, prewarm_dir: &Path) -> Result<()> {
    let rocm = resolve_rocm_binary()?;
    for sub in ["config", "data", "cache"] {
        let dir = prewarm_dir.join(sub);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create pre-warm directory {}", dir.display()))?;
    }

    // Before asking whether the tree is FRESH, make sure what is in it is
    // actually usable from this tree. `rocm update` compares versions against the
    // index; it cannot see that a runtime records a folder somewhere else.
    // Repairing first means `decide` reads a registry with nothing dead in it.
    repair_poisoned_runtimes(&rocm, prewarm_dir)?;

    let decision = match probe(&rocm, prewarm_dir) {
        Ok(report) => decide(&report, channel),
        Err(error) => {
            // `rocm update` itself failed (not a per-runtime index error). Fall
            // back to the guard the lanes used before this existed: install only
            // when the registry is genuinely absent, otherwise serve against what
            // is there. Preserves the old floor rather than failing the lane.
            let registry = prewarm_dir.join("data").join("runtimes").join("registry");
            if registry.is_dir() {
                println!("pre-warm: `rocm update` failed ({error:#}); reusing the existing tree");
                Decision::Reuse {
                    reason: "update probe failed".to_owned(),
                }
            } else {
                println!("pre-warm: `rocm update` failed ({error:#}); no registry yet, installing");
                Decision::Install
            }
        }
    };

    match &decision {
        Decision::Install => {
            println!(
                "pre-warm: installing the {channel} SDK into {}",
                prewarm_dir.display()
            );
            rocm_command(&rocm, prewarm_dir)
                .args(["install", "sdk", "--channel", channel])
                .status_ok("rocm install sdk")?;
        }
        Decision::Update { runtime_key } => {
            println!(
                "pre-warm: {runtime_key} is behind the {channel} index; installing the newer runtime alongside it"
            );
            rocm_command(&rocm, prewarm_dir)
                .args(["update", "--apply", "--runtime", runtime_key, "--activate"])
                .status_ok("rocm update --apply")?;
        }
        Decision::Reuse { reason } => {
            println!("pre-warm: reusing the shared {channel} runtime ({reason})");
            return Ok(());
        }
    }

    // An install/update that exits 0 without leaving a registry behind is the
    // confusing case the lanes used to call out by hand: every scenario then falls
    // back to installing its own runtime and the job quietly blows its time cap.
    // Say so loudly, but do not fail — the suite can still run.
    let registry = prewarm_dir.join("data").join("runtimes").join("registry");
    if !registry.is_dir() {
        println!(
            "::warning::pre-warm produced no runtimes registry at {}; scenarios will install their own",
            registry.display()
        );
    }

    // Only reached after an install or update actually added a tree. Housekeeping:
    // a failure here wastes disk but leaves a correct runtime in place, so it must
    // not fail the lane.
    let pruned = rocm_command(&rocm, prewarm_dir)
        .args([
            "storage",
            "remove-old-installs",
            "--keep",
            &keep.to_string(),
            "--yes",
        ])
        .status_ok("rocm storage remove-old-installs");
    if let Err(error) = pruned {
        println!("pre-warm: pruning old installs failed ({error:#}); continuing");
    }
    Ok(())
}

/// Drop any managed runtime in the shared tree that records an install root
/// outside it, so the pre-warm reinstalls instead of serving a dead one.
///
/// Deleting the folder is not belt-and-braces, it is the repair. A poisoned venv
/// keeps a working `bin/python` — that is a symlink to the base interpreter, which
/// is still there — so `ensure_uv_venv` REUSES it, and an already-satisfied
/// package is audited rather than reinstalled, leaving every console-script
/// shebang still pointing at the folder that went away. Measured with uv 0.9.30:
/// re-running the install over a poisoned venv reports success and repairs
/// nothing; only removing the folder first does.
///
/// Failure to repair is fatal, unlike the rest of this module. Everywhere else a
/// conservative fallback keeps the lane green; here the alternative is serving
/// against a runtime already known to be broken, which fails later, elsewhere, and
/// for reasons that name none of this.
fn repair_poisoned_runtimes(rocm: &Path, prewarm_dir: &Path) -> Result<()> {
    let runtimes_dir = prewarm_dir.join("data").join("runtimes");
    if !runtimes_dir.is_dir() {
        return Ok(());
    }

    let listing = match list_runtimes(rocm, prewarm_dir) {
        Ok(listing) => listing,
        Err(error) => {
            // Same floor as everywhere else in this module: an unreadable report
            // must not delete anything, and must not redden the lane.
            println!(
                "pre-warm: could not list runtimes ({error:#}); leaving the shared tree untouched"
            );
            return Ok(());
        }
    };

    for runtime in assess(&listing, &runtimes_dir) {
        println!(
            "::warning::pre-warm: removing the shared runtime {} — it records an install root \
             outside this tree ({}), which a scenario that installed through its own \
             `data/runtimes` symlink would have written. A reinstall follows. See rocm-cli#315.",
            runtime.runtime_key, runtime.install_root
        );

        // Drops the registry entry, the active marker, and the config pointers.
        // Tolerates the recorded folder being absent, which it is.
        rocm_command(rocm, prewarm_dir)
            .args(["runtimes", "uninstall", &runtime.runtime_key])
            .status_ok("rocm runtimes uninstall")?;

        // The physical tree the CLI could not reach: it removed what the manifest
        // POINTED at, and the files are where the manifest should have said.
        let planted = runtimes_dir
            .join(&runtime.format)
            .join(&runtime.runtime_key);
        if !planted.starts_with(&runtimes_dir) {
            bail!(
                "refusing to remove {}: outside the pre-warm tree at {}",
                planted.display(),
                runtimes_dir.display()
            );
        }
        if planted.is_dir() {
            std::fs::remove_dir_all(&planted).with_context(|| {
                format!(
                    "failed to remove the poisoned runtime folder {}",
                    planted.display()
                )
            })?;
            println!("pre-warm: removed {}", planted.display());
        }
    }
    Ok(())
}

/// Ask the CLI what it has registered. Read-only.
fn list_runtimes(rocm: &Path, prewarm_dir: &Path) -> Result<String> {
    let output = rocm_command(rocm, prewarm_dir)
        .args(["runtimes", "list"])
        .output()
        .context("failed to run `rocm runtimes list`")?;
    if !output.status.success() {
        bail!(
            "`rocm runtimes list` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Ask the CLI whether the installed runtimes are current. Check-only: plain
/// `rocm update` without `--apply` never mutates state.
fn probe(rocm: &Path, prewarm_dir: &Path) -> Result<String> {
    let output = rocm_command(rocm, prewarm_dir)
        .arg("update")
        .output()
        .context("failed to run `rocm update`")?;
    if !output.status.success() {
        bail!(
            "`rocm update` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A `rocm` invocation scoped to the pre-warm tree.
///
/// The three `ROCM_CLI_*` directories are what make this a SHARED tree rather
/// than the caller's own; `HF_HOME` / `UV_CACHE_DIR` are inherited from the job
/// environment, which the lanes already export.
fn rocm_command(rocm: &Path, prewarm_dir: &Path) -> Command {
    let mut cmd = Command::new(rocm);
    cmd.env("ROCM_CLI_CONFIG_DIR", prewarm_dir.join("config"))
        .env("ROCM_CLI_DATA_DIR", prewarm_dir.join("data"))
        .env("ROCM_CLI_CACHE_DIR", prewarm_dir.join("cache"));
    cmd
}

/// Run a command for its exit status, turning a non-zero exit into an error that
/// names the command.
trait StatusOk {
    fn status_ok(&mut self, what: &str) -> Result<()>;
}

impl StatusOk for Command {
    fn status_ok(&mut self, what: &str) -> Result<()> {
        let status = self
            .status()
            .with_context(|| format!("failed to run `{what}`"))?;
        if !status.success() {
            bail!("`{what}` exited with {status}");
        }
        Ok(())
    }
}

/// The `rocm` binary to drive: `ROCM_CLI_BINARY` when the caller already built
/// one (as every CI lane does, so the pre-warm and the suite share a build),
/// otherwise the release binary in the active target directory.
fn resolve_rocm_binary() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("ROCM_CLI_BINARY") {
        // Absolutize as `e2e.rs` does: the Strix Windows lane sets a RELATIVE
        // `target\release\rocm.exe` when `CARGO_TARGET_DIR` is unset, which would
        // only resolve while the cwd happens to be the workspace root.
        let path = PathBuf::from(path);
        return if path.is_absolute() {
            Ok(path)
        } else {
            Ok(std::env::current_dir()
                .context("failed to read the current directory")?
                .join(path))
        };
    }
    let root = workspace_root()?;
    let candidate = release_binary_dir(&root, None).join(binary_name("rocm"));
    if !candidate.is_file() {
        bail!(
            "no rocm binary at {}; run `cargo build --release -p rocm` or set ROCM_CLI_BINARY",
            candidate.display()
        );
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `rocm update` report on a machine with no managed runtime, captured
    /// verbatim from the built binary rather than written by hand — the parser has
    /// to survive the whole document, not an idealized excerpt of it.
    ///
    /// Note the `update_surfaces` block: `runtimes: status=none_configured` is one
    /// character away from a real `runtime <key> … status=…` entry, and carries a
    /// `status=` field of its own. See `update_surfaces_block_is_not_a_runtime_entry`.
    const EMPTY: &str = "\
update
  policy: bounded startup check, cached metadata, prompt before mutating state.
  managed runtimes: none
  next step: run `rocm install sdk --channel release --dry-run` to resolve a TheRock runtime
  update_surfaces:
    cli: installed=0.1.0 status=not_configured reason=repository-owned CLI update feed is not published yet
    engines: status=package_managed packaged=[lemonade,vllm] reason=first-party engine binaries update with the rocm-cli package; data-dir plugins are user-managed
    model_recipes: status=built_in count=13 reason=external signed recipe index is not configured
    runtimes: status=none_configured reason=TheRock runtime update checks above are the only live update checks in this build
  note: `rocm update --apply` applies runtime updates only; CLI, engine, and recipe update feeds require published metadata before they can mutate state
";

    fn report(status: &str, channel: &str) -> String {
        format!(
            "update\n  policy: bounded startup check, cached metadata, prompt before mutating state.\n  \
runtime {channel}-wheel-gfx94x-dcgpu-7-13-0 format=wheel channel={channel} \
family=gfx94X-dcgpu installed=7.13.0 latest=7.15.0 status={status}\n    \
install_root: /w/e2e-prewarm/data/runtimes/wheel/{channel}-wheel-gfx94x-dcgpu-7-13-0\n    \
source: index\n"
        )
    }

    /// A real `rocm runtimes list` on a tree holding all three shapes that matter,
    /// captured verbatim from the built binary rather than written by hand: one
    /// poisoned managed runtime (an install root under a per-scenario temp dir that
    /// is gone), one healthy managed runtime inside the tree, and one read-only
    /// runtime adopted from outside it.
    ///
    /// Note that ALL THREE report `status=unusable` here. That is the whole reason
    /// [`assess`] keys off the install root instead: the healthy one is unusable
    /// only for want of a `rocm_sdk` probe block, and treating that as poison would
    /// delete a multi-GiB runtime that a reinstall would have kept.
    const MIXED: &str = "\
registered ROCm runtimes
  active_runtime_id: <unset>
  active_runtime_key: <unset>
  previous_runtime_key: <unset>
  registry: /w/e2e-prewarm/data/runtimes/registry
  marker: /w/e2e-prewarm/data/runtimes/active.json
  installed:
    adopted-external-env runtime_id=external-adopted version=7.14.0 format=wheel family=gfx94X-dcgpu mode=read-only status=unusable (pip runtime manifest is missing rocm_sdk probe data)
      install_root: /opt/external-rocm
    release-wheel-gfx94x-dcgpu-7-15-0 runtime_id=therock-release-gfx94x-dcgpu version=7.15.0 format=wheel family=gfx94X-dcgpu mode=managed status=unusable (pip runtime manifest is missing rocm_sdk probe data)
      install_root: /w/e2e-prewarm/data/runtimes/wheel/release-wheel-gfx94x-dcgpu-7-15-0
    release-wheel-gfx94x-dcgpu-7-13-0 runtime_id=therock-release-gfx94x-dcgpu version=7.13.0 format=wheel family=gfx94X-dcgpu mode=managed status=unusable (install root is missing: /tmp/rocm-e2e-7MidR2/data/runtimes/wheel/release-wheel-gfx94x-dcgpu-7-13-0)
      install_root: /tmp/rocm-e2e-7MidR2/data/runtimes/wheel/release-wheel-gfx94x-dcgpu-7-13-0
";

    /// A real `rocm runtimes list` on an empty tree, captured from the binary.
    const NO_RUNTIMES: &str = "\
registered ROCm runtimes
  active_runtime_id: <unset>
  active_runtime_key: <unset>
  previous_runtime_key: <unset>
  registry: /w/e2e-prewarm/data/runtimes/registry
  marker: /w/e2e-prewarm/data/runtimes/active.json
  installed: none
  next step: rocm install sdk --channel release --format wheel
";

    fn prewarm_runtimes_dir() -> &'static Path {
        Path::new("/w/e2e-prewarm/data/runtimes")
    }

    #[test]
    fn a_runtime_recording_a_scenario_temp_dir_is_poisoned() {
        assert_eq!(
            assess(MIXED, prewarm_runtimes_dir()),
            vec![PoisonedRuntime {
                runtime_key: "release-wheel-gfx94x-dcgpu-7-13-0".to_owned(),
                format: "wheel".to_owned(),
                install_root:
                    "/tmp/rocm-e2e-7MidR2/data/runtimes/wheel/release-wheel-gfx94x-dcgpu-7-13-0"
                        .to_owned(),
            }]
        );
    }

    #[test]
    fn an_unusable_runtime_inside_the_tree_is_left_alone() {
        // The healthy-but-unusable entry in MIXED. Deleting it would throw away a
        // multi-GiB install over a validation detail a reinstall would have fixed.
        let poisoned = assess(MIXED, prewarm_runtimes_dir());
        assert!(
            !poisoned
                .iter()
                .any(|runtime| runtime.runtime_key.ends_with("7-15-0")),
            "an in-tree runtime must never be removed for being unusable: {poisoned:?}"
        );
    }

    #[test]
    fn a_read_only_runtime_adopted_from_outside_is_left_alone() {
        // `runtimes adopt` records an external folder on purpose. Removing it would
        // delete a runtime the tree never owned.
        let poisoned = assess(MIXED, prewarm_runtimes_dir());
        assert!(
            !poisoned
                .iter()
                .any(|runtime| runtime.runtime_key == "adopted-external-env"),
            "a read-only adopted runtime must be exempt: {poisoned:?}"
        );
    }

    #[test]
    fn an_empty_tree_has_nothing_to_repair() {
        assert!(assess(NO_RUNTIMES, prewarm_runtimes_dir()).is_empty());
    }

    #[test]
    fn a_report_that_cannot_be_read_removes_nothing() {
        // The conservative floor: never delete on a shape this does not recognise.
        assert!(assess("", prewarm_runtimes_dir()).is_empty());
        assert!(assess("totally unexpected output\n", prewarm_runtimes_dir()).is_empty());
    }

    #[test]
    fn an_entry_without_its_install_root_line_removes_nothing() {
        // Guessing the folder to delete from the key alone is not acceptable.
        let text = "  installed:\n    k runtime_id=r version=1 format=wheel family=f \
mode=managed status=ready\n";
        assert!(assess(text, prewarm_runtimes_dir()).is_empty());
    }

    #[test]
    fn the_active_and_rollback_markers_do_not_hide_an_entry() {
        // `render_runtimes_text` prefixes the active runtime with `*` and the
        // rollback target with `-`; a poisoned runtime is usually the active one.
        for marker in ["*", "-"] {
            let text = format!(
                "  installed:\n  {marker} k runtime_id=r version=1 format=wheel family=f \
mode=managed status=ready\n      install_root: /tmp/rocm-e2e-XXXX/data/runtimes/wheel/k\n"
            );
            let poisoned = assess(&text, prewarm_runtimes_dir());
            assert_eq!(poisoned.len(), 1, "marker {marker} hid the entry");
            assert_eq!(poisoned[0].runtime_key, "k");
        }
    }

    #[test]
    fn header_lines_are_not_read_as_entries() {
        // `active_runtime_id:` is one character away from the `runtime_id=` field
        // that identifies a real entry.
        assert!(RuntimeEntry::parse("  active_runtime_id: <unset>").is_none());
        assert!(RuntimeEntry::parse("  registry: /w/e2e-prewarm/data/runtimes/registry").is_none());
        assert!(RuntimeEntry::parse("  installed: none").is_none());
        assert!(RuntimeEntry::parse("      install_root: /tmp/x").is_none());
    }

    #[test]
    fn a_tarball_runtime_reports_its_own_format() {
        // The folder to remove is `<runtimes>/<format>/<key>`, so the format has to
        // survive parsing — removing the wheel path for a tarball runtime would
        // silently repair nothing.
        let text = "  installed:\n    k runtime_id=r version=1 format=tarball family=f \
mode=managed status=ready\n      install_root: /tmp/rocm-e2e-XXXX/data/runtimes/tarball/k\n";
        let poisoned = assess(text, prewarm_runtimes_dir());
        assert_eq!(poisoned.len(), 1);
        assert_eq!(poisoned[0].format, "tarball");
    }

    #[test]
    fn no_managed_runtime_installs() {
        assert_eq!(decide(EMPTY, "release"), Decision::Install);
    }

    #[test]
    fn newer_version_in_the_index_updates_that_runtime() {
        assert_eq!(
            decide(&report("update_available", "release"), "release"),
            Decision::Update {
                runtime_key: "release-wheel-gfx94x-dcgpu-7-13-0".to_owned()
            }
        );
    }

    #[test]
    fn current_runtime_is_reused() {
        let Decision::Reuse { reason } = decide(&report("up_to_date", "release"), "release") else {
            panic!("an up-to-date runtime must be reused, not reinstalled");
        };
        assert!(reason.contains("up to date"), "{reason}");
    }

    #[test]
    fn runtime_ahead_of_the_index_is_not_downgraded() {
        // A pinned or hand-placed build newer than the index must be left alone —
        // "updating" it would move the lane backwards.
        let Decision::Reuse { reason } = decide(&report("ahead_of_index", "release"), "release")
        else {
            panic!("a runtime ahead of the index must be reused");
        };
        assert!(reason.contains("ahead of"), "{reason}");
    }

    #[test]
    fn index_error_reuses_rather_than_reinstalling() {
        // The renderer omits `channel=` when resolving the index fails. That is
        // unknown freshness, not proof that this channel has no runtime, so the
        // conservative pre-warm decision must reuse rather than download again.
        let text = "update\n  runtime release-wheel-gfx94x-dcgpu-7-13-0 format=wheel \
status=error message=failed to reach https://repo.amd.com/rocm/whl after 3 tries\n";
        let Decision::Reuse { reason } = decide(text, "release") else {
            panic!("an unattributable index error must reuse the existing tree");
        };
        assert!(reason.contains("could not establish"), "{reason}");
    }

    #[test]
    fn unattributed_error_wins_over_a_known_other_channel() {
        let text = format!(
            "{}  runtime release-wheel-gfx94x-dcgpu-7-13-0 format=wheel \
status=error message=failed to reach the index\n",
            report("up_to_date", "nightly")
        );
        assert!(matches!(decide(&text, "release"), Decision::Reuse { .. }));
    }

    #[test]
    fn index_error_on_an_attributable_line_is_reused() {
        let text = "update\n  runtime release-wheel-gfx94x-dcgpu-7-13-0 format=wheel \
channel=release status=error message=failed to reach the index\n";
        let Decision::Reuse { reason } = decide(text, "release") else {
            panic!("an unreadable freshness status must reuse the existing tree");
        };
        assert!(reason.contains("could not establish"), "{reason}");
    }

    #[test]
    fn another_channels_runtime_does_not_satisfy_this_channel() {
        // A per-channel pre-warm still shares one tree layout, and EAI-8056 adds a
        // nightly lane: a release runtime must never be mistaken for a nightly one.
        assert_eq!(
            decide(&report("up_to_date", "release"), "nightly"),
            Decision::Install
        );
    }

    #[test]
    fn the_stale_runtime_for_this_channel_is_the_one_updated() {
        // Mixed tree: only the line matching our channel may be selected.
        let text = format!(
            "{}{}",
            report("up_to_date", "release"),
            report("update_available", "nightly")
        );
        assert_eq!(
            decide(&text, "nightly"),
            Decision::Update {
                runtime_key: "nightly-wheel-gfx94x-dcgpu-7-13-0".to_owned()
            }
        );
        // …and the release lane reading the same tree still sees a cache hit.
        assert!(matches!(decide(&text, "release"), Decision::Reuse { .. }));
    }

    #[test]
    fn unparseable_report_reuses() {
        let Decision::Reuse { reason } = decide(
            "update\n  runtime weird-key format=wheel channel=release\n",
            "release",
        ) else {
            panic!("a report with no status must reuse");
        };
        assert!(reason.contains("could not establish"), "{reason}");
    }

    #[test]
    fn message_text_containing_spaces_does_not_break_field_parsing() {
        let line = "  runtime k format=wheel channel=release status=error \
message=connect timed out after 30 s";
        let parsed = RuntimeLine::parse(line).expect("line parses");
        assert_eq!(parsed.runtime_key, "k");
        assert_eq!(parsed.channel.as_deref(), Some("release"));
        assert_eq!(parsed.status.as_deref(), Some("error"));
    }

    #[test]
    fn non_runtime_lines_are_ignored() {
        assert!(RuntimeLine::parse("  policy: bounded startup check").is_none());
        assert!(RuntimeLine::parse("    install_root: /tmp/x").is_none());
    }

    #[test]
    fn update_surfaces_block_is_not_a_runtime_entry() {
        // `runtimes: status=none_configured` differs from a real entry only by the
        // `s` where the prefix expects a space, and it does carry a `status=`
        // field. Reading it as a runtime would make an empty tree look like an
        // unrecognised-status one and resolve to Reuse — the pre-warm would then
        // never do the very first install.
        assert!(
            RuntimeLine::parse(
                "    runtimes: status=none_configured reason=TheRock runtime update checks \
                 above are the only live update checks in this build"
            )
            .is_none(),
            "the update_surfaces summary must not parse as an installed runtime"
        );
        assert!(RuntimeLine::parse("    cli: installed=0.1.0 status=not_configured").is_none());
        // …and end to end, the real empty report still resolves to a cold install.
        assert_eq!(decide(EMPTY, "release"), Decision::Install);
    }
}
