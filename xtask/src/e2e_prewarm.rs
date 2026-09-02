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

/// The active-runtime selector the config still records after the runtime it
/// names has gone, if the report says so.
///
/// This is the OTHER way the shared tree goes stale, and it is the one that kills
/// the lane outright. [`assess`] finds a runtime the registry has but the tree
/// does not; this finds a pointer the config has but the registry does not. The
/// two are independent: removing a runtime through
/// `rocm runtimes uninstall` clears the config pointers with it, so a dangling
/// pointer means something removed the tree WITHOUT the CLI — a hand cleanup on
/// the runner, or a scenario that deleted a folder it had symlinked in.
///
/// `rocm engines install` resolves the runtime to build against as
/// `--runtime` → `config.active_runtime_key` → `config.default_runtime_id`, so a
/// dangling pointer in either field fails that resolution before anything else
/// runs:
///
/// ```text
/// pre-warm: ensuring the vllm engine is installed
/// Error: runtime selector `release-wheel-gfx94x-dcgpu-7-14-0` from engine install
///        runtime selection is not an exact usable runtime: installed runtime not found
/// ```
///
/// Nine seconds in, before a single scenario, and the message names neither the
/// pre-warm nor the tree it is about. Detecting it is free because
/// `render_runtimes_text` already reports it, and reporting it is all the CLI
/// does — there is no verb that clears a pointer, so the repair has to be
/// activating something that exists.
///
/// Only `config.json` is read this way. `engines.<name>.last_installed_runtime_id`
/// can dangle too, but nothing resolves through it — it is displayed, not
/// followed — so healing it would be motion without a failure behind it.
#[must_use]
pub fn dangling_active_runtime(runtimes_list_report: &str) -> Option<&str> {
    runtimes_list_report.lines().find_map(|line| {
        let rest = line
            .trim()
            .strip_prefix("active_status: missing manifest for ")?;
        // Whichever field held the dead name — the key when one was activated, the
        // id when only a default was ever set. Both are selectors the engine
        // install would resolve, and both fail it the same way.
        let (_, selector) = rest.split_once('=')?;
        let selector = selector.trim();
        (!selector.is_empty()).then_some(selector)
    })
}

/// Managed runtimes the tree could be pointed at instead, in report order.
///
/// Read-only entries are excluded on purpose. `runtimes adopt` records a folder
/// the pre-warm does not own, and quietly making somebody's external ROCm install
/// the default for every GPU scenario on the runner is a larger decision than
/// repairing a pointer. If the tree holds nothing else, the caller leaves the
/// pointer dangling and the install path replaces it.
#[must_use]
pub fn activation_candidates(runtimes_list_report: &str) -> Vec<String> {
    runtimes_list_report
        .lines()
        .filter_map(RuntimeEntry::parse)
        .filter(|entry| !entry.read_only)
        .map(|entry| entry.runtime_key)
        .collect()
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

    // And point the tree at something that exists before anything resolves the
    // active runtime. Must run after the repair above, which can itself remove
    // the runtime the pointer names.
    repair_dangling_active_runtime(&rocm, prewarm_dir)?;

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
        }
    }

    // Unconditional, and deliberately BEFORE the reuse early return below. The
    // runtime and the serving engine are installed separately: `install sdk`
    // lays down the ROCm runtime, `engines install` builds the engine venv
    // against it. `decide` only ever reasons about the runtime, so a tree whose
    // runtime is current but whose engine was never installed — or was left
    // behind with an older runtime — resolves to `Reuse`, which used to return
    // here having done nothing. The shared tree then served every GPU scenario a
    // runtime with no engine, which is the one thing those lanes exist to
    // exercise. Re-checking a warm tree is cheap: without `--reinstall`,
    // `engines install` on a ready engine installs nothing.
    ensure_default_engine(&rocm, prewarm_dir)?;

    if !runtime_changed(&decision) {
        return Ok(());
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

/// Whether `decision` put a new runtime in the tree, and so whether the registry
/// check and the retention prune at the end of [`run`] have anything to do.
///
/// Read AFTER the engine check, never inside the decision's own match arm: the
/// engine is installed separately from the runtime, so every decision — reuse
/// most of all, since it is the one a warm runner takes every time — has to
/// reach that check before this can end the pre-warm early.
const fn runtime_changed(decision: &Decision) -> bool {
    !matches!(decision, Decision::Reuse { .. })
}

/// Install the engine the active runtime would serve on, so the shared tree has
/// one before a scenario asks it to serve.
///
/// Which engine that is comes from the CLI rather than from a constant here:
/// `rocm engines list` marks the engine `serve` picks for the detected GPU with
/// `* ` (vLLM on Instinct, Lemonade on Strix), and the pre-warm must agree with
/// `serve` on every runner without this file learning the hardware map.
///
/// Fatal on failure, like [`repair_poisoned_runtimes`] and unlike the freshness
/// path: reusing a stale-but-working runtime keeps a lane meaningful, whereas
/// serving with no engine fails every GPU scenario later and for reasons that
/// name none of this.
fn ensure_default_engine(rocm: &Path, prewarm_dir: &Path) -> Result<()> {
    let output = rocm_command(rocm, prewarm_dir)
        .args(["engines", "list"])
        .output()
        .context("failed to run `rocm engines list`")?;
    if !output.status.success() {
        bail!("`rocm engines list` exited with {}", output.status);
    }
    let inventory = String::from_utf8_lossy(&output.stdout);
    let engine = default_engine_from_inventory(&inventory)
        .context("`rocm engines list` did not identify a default engine")?;
    println!("pre-warm: ensuring the {engine} engine is installed");
    rocm_command(rocm, prewarm_dir)
        .args(["engines", "install", engine, "--yes"])
        .status_ok("rocm engines install")
}

/// The engine `rocm engines list` marks as the default for this host, if any.
///
/// The inventory renders one line per engine as `{marker} {name:10} {note}` with
/// the marker in column 0, then indents that engine's detail lines (`    adapter:
/// …`, `    runtime: …`) beneath it. Matching `* ` at the start of the line
/// unindented is therefore what separates the default engine's own line from
/// everything else the report prints.
fn default_engine_from_inventory(inventory: &str) -> Option<&str> {
    inventory
        .lines()
        .find_map(|line| line.strip_prefix("* ")?.split_whitespace().next())
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

/// Re-point the shared tree's active runtime when it names one that is gone, so
/// the engine install resolves instead of dying nine seconds into the lane.
///
/// See [`dangling_active_runtime`] for how the pointer goes stale and why this is
/// worth healing here. The repair is `rocm runtimes activate`: the CLI has no verb
/// that clears a pointer, so the only way out is to name something that exists.
/// Candidates are tried in report order and the first that takes it wins — which
/// may be an older runtime than the tree would ideally serve, and that is fine.
/// `decide` runs next, and the `Update` arm re-activates the newest as part of
/// updating to it.
///
/// Doing nothing is correct when the tree holds no managed runtime at all: the
/// install path activates whatever it installs, and it runs before
/// `ensure_default_engine` — which is the only thing that would have tripped over
/// the pointer. The case that has to be handled here is the one that self-healing
/// misses: a pointer dangling while OTHER runtimes are sitting right there, where
/// `decide` says `Reuse`, nothing installs, and nothing re-activates.
fn repair_dangling_active_runtime(rocm: &Path, prewarm_dir: &Path) -> Result<()> {
    if !prewarm_dir.join("data").join("runtimes").is_dir() {
        return Ok(());
    }

    // Deliberately a second listing rather than one shared with the repair above:
    // that repair uninstalls, which rewrites exactly the pointers read here.
    let listing = match list_runtimes(rocm, prewarm_dir) {
        Ok(listing) => listing,
        Err(error) => {
            println!(
                "pre-warm: could not list runtimes ({error:#}); leaving the active runtime as it is"
            );
            return Ok(());
        }
    };

    let Some(dangling) = dangling_active_runtime(&listing) else {
        return Ok(());
    };

    let candidates = activation_candidates(&listing);
    if candidates.is_empty() {
        println!(
            "pre-warm: the active runtime {dangling} is gone and nothing is installed to take \
             its place; the install below will set it"
        );
        return Ok(());
    }

    println!(
        "::warning::pre-warm: the shared tree still points at the runtime {dangling}, which is \
         no longer installed — something removed it without going through \
         `rocm runtimes uninstall`. Engine installs resolve through that pointer, so re-pointing \
         it at an installed runtime. See rocm-cli#314."
    );

    for candidate in &candidates {
        let activated = rocm_command(rocm, prewarm_dir)
            .args(["runtimes", "activate", candidate])
            .status()
            .with_context(|| format!("failed to run `rocm runtimes activate {candidate}`"))?;
        if activated.success() {
            println!("pre-warm: active runtime is now {candidate}");
            return Ok(());
        }
        // An installed runtime can still be refused — `activate` validates the
        // manifest. Try the next one rather than taking the whole lane down for
        // one bad entry.
        println!("pre-warm: could not activate {candidate}; trying the next runtime");
    }

    bail!(
        "the shared pre-warm tree points at the runtime `{dangling}`, which is not installed, and \
         none of its {} installed runtime(s) could be activated in its place",
        candidates.len()
    )
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

    /// A real `rocm runtimes list` on the tree that killed the lane: the config
    /// still names the 7.14.0 runtime that was removed out of band, while two
    /// runtimes it could be pointed at sit right there.
    ///
    /// Note there is no `*` marker on any entry — the renderer only marks the
    /// active one by matching `active_runtime_key` against a manifest, and that is
    /// exactly the match that fails here. The `active_status:` line is the only
    /// thing in the document that says so.
    const DANGLING_ACTIVE: &str = "\
registered ROCm runtimes
  active_runtime_id: therock-release:gfx94X-dcgpu
  active_runtime_key: release-wheel-gfx94x-dcgpu-7-14-0
  previous_runtime_key: release-wheel-gfx94x-dcgpu-7-13-0
  registry: /w/e2e-prewarm/data/runtimes/registry
  marker: /w/e2e-prewarm/data/runtimes/active.json
  active_status: missing manifest for active_runtime_key=release-wheel-gfx94x-dcgpu-7-14-0
  installed:
    adopted-external-env runtime_id=external-adopted version=7.14.0 format=wheel family=gfx94X-dcgpu mode=read-only status=usable
      install_root: /opt/external-rocm
    release-wheel-gfx94x-dcgpu-7-13-0 runtime_id=therock-release-gfx94x-dcgpu version=7.13.0 format=wheel family=gfx94X-dcgpu mode=managed status=usable
      install_root: /w/e2e-prewarm/data/runtimes/wheel/release-wheel-gfx94x-dcgpu-7-13-0
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
    fn an_active_runtime_that_is_gone_is_reported_by_name() {
        assert_eq!(
            dangling_active_runtime(DANGLING_ACTIVE),
            Some("release-wheel-gfx94x-dcgpu-7-14-0")
        );
    }

    #[test]
    fn a_default_runtime_id_that_is_gone_dangles_the_same_way() {
        // The other field the engine install resolves through, reported by the
        // renderer under the same key when no runtime_key was ever activated.
        let text = "registered ROCm runtimes\n  active_runtime_id: therock-release:gfx94X-dcgpu\n  \
active_runtime_key: <unset>\n  \
active_status: missing manifest for active_runtime_id=therock-release:gfx94X-dcgpu\n";
        assert_eq!(
            dangling_active_runtime(text),
            Some("therock-release:gfx94X-dcgpu")
        );
    }

    #[test]
    fn a_tree_whose_active_runtime_is_installed_is_left_alone() {
        // No active_status line at all is the healthy shape — MIXED has a dead
        // runtime in it, but nothing points AT the dead one.
        assert_eq!(dangling_active_runtime(MIXED), None);
        assert_eq!(dangling_active_runtime(NO_RUNTIMES), None);
        assert_eq!(dangling_active_runtime(""), None);
    }

    #[test]
    fn an_ambiguous_runtime_id_is_not_a_dangling_pointer() {
        // Same `active_status:` prefix, entirely different condition: the runtimes
        // are all there, one runtime_id just names several of them. Activating
        // something would be a guess, and the engine install resolves it fine.
        let text = "registered ROCm runtimes\n  \
active_status: ambiguous runtime_id=therock-release:gfx94X-dcgpu; activate one runtime_key: a, b\n";
        assert_eq!(dangling_active_runtime(text), None);
    }

    #[test]
    fn only_managed_runtimes_are_offered_as_replacements() {
        // The read-only adopted entry is installed and usable, and still excluded:
        // making somebody's external ROCm the default for every GPU scenario is a
        // bigger decision than repairing a pointer.
        assert_eq!(
            activation_candidates(DANGLING_ACTIVE),
            vec!["release-wheel-gfx94x-dcgpu-7-13-0".to_owned()]
        );
    }

    #[test]
    fn a_tree_with_nothing_installed_offers_no_replacement() {
        // The caller leaves the pointer dangling here rather than failing: the
        // install that follows activates whatever it installs.
        assert!(activation_candidates(NO_RUNTIMES).is_empty());
        assert!(activation_candidates("").is_empty());
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

    /// A real `rocm engines list` on an Instinct host, captured verbatim: the
    /// default engine's line carries the `* ` marker in column 0, and its own
    /// detail lines are indented beneath it.
    const ENGINES_READY: &str = "\
Local model engines
  Built-in engines are included with rocm-cli. External plugins are optional.
  ROCm GPU execution is required.
  Plugin folders:
    1. /w/e2e-prewarm/data/engines/plugins (primary)
  lemonade   default embedded Lemonade server with ROCm llama.cpp backend
    adapter: built-in
    runtime: not found
* vllm       Linux/WSL ROCm GPU serving engine through external vLLM
    adapter: built-in
    runtime: /w/e2e-prewarm/data/runtimes/wheel/release-wheel-gfx94x-dcgpu-7-15-0
  protocol: 0.1.0
";

    #[test]
    fn the_marked_engine_is_the_one_pre_warmed() {
        // Which engine to install comes from the CLI's own host detection, not
        // from a hardware map duplicated here.
        assert_eq!(default_engine_from_inventory(ENGINES_READY), Some("vllm"));
    }

    #[test]
    fn an_indented_detail_line_is_not_read_as_the_default() {
        // Every engine's detail lines are indented under it, and a note may well
        // start with a bullet. Only the marker in column 0 names the default.
        let inventory = "\
Local model engines
* lemonade   default embedded Lemonade server with ROCm llama.cpp backend
    adapter: built-in
    * not a marker
";
        assert_eq!(default_engine_from_inventory(inventory), Some("lemonade"));
    }

    #[test]
    fn an_inventory_without_a_default_engine_names_none() {
        // `ensure_default_engine` turns this into an error rather than guessing an
        // engine: installing the wrong one costs a multi-GiB build and still
        // leaves the lane with nothing to serve on.
        assert_eq!(default_engine_from_inventory(""), None);
        assert_eq!(
            default_engine_from_inventory("Local model engines\n  lemonade   embedded\n"),
            None
        );
    }

    #[test]
    fn reusing_the_shared_runtime_still_reaches_the_engine_check() {
        // The regression this guards: `Reuse` — the decision EVERY warm runner
        // takes, run after run — used to end `run` with a `return` inside its own
        // match arm, before anything looked at the engine. The early return is now
        // this predicate, read only AFTER `ensure_default_engine`, so reuse cannot
        // skip the engine. Reuse must still install and update NOTHING, which is
        // what keeps it cheap enough to re-check the engine on every run.
        assert!(!runtime_changed(&Decision::Reuse {
            reason: "up to date".to_owned()
        }));
        assert!(runtime_changed(&Decision::Install));
        assert!(runtime_changed(&Decision::Update {
            runtime_key: "release-wheel-gfx94x-dcgpu-7-13-0".to_owned()
        }));
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
