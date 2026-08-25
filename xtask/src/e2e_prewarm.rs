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
