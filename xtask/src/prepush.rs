// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask pre-push-check`: the three cargo steps of the pre-push hook
//! (`clippy`, `clippy-e2e`, `test`), narrowed to the crates the push affects.
//!
//! Each step used to lint or test the whole workspace whatever the push
//! contained, so a one-file change in a leaf crate paid for every crate. The
//! selection is the one CI uses ([`affected::select_with_readers`]): the crates
//! whose files changed plus every crate that transitively depends on them, with
//! a fall back to the whole workspace for anything that cannot be attributed to
//! one crate (the lockfile, the root manifest, the toolchain file, workflow
//! files, a doc an xtask test reads, a file xtask compiles in by path, an
//! unrecognized path), widened with the
//! crates whose tests read a changed file from outside their own crate
//! directory ([`crate::cross_crate_reads`]). CI stays the complete gate; this
//! only makes the local early warning cheaper.
//!
//! **Fail safe.** Whenever the affected set cannot be determined (`git diff`
//! fails because the base is not fetched, `cargo metadata` fails, the range lists
//! no changed files at all, ...) the step runs over the whole workspace, as it
//! did before, and says why. A step is skipped only when the selection is known
//! and contains nothing for that step to check, and it prints the reason then
//! too.
//!
//! The range comes from the refs prek exports to a pre-push hook:
//! `PRE_COMMIT_FROM_REF` is the commit the remote already has (the remote
//! branch tip for an update, the parent of the first unpushed commit for a new
//! branch) and `PRE_COMMIT_TO_REF` is what is being pushed. With neither, for a
//! manual run, it diffs `origin/main...HEAD` like `cargo xtask affected` does.
//! `--base` overrides the from-ref.
//!
//! A step that already passed on this exact tree is skipped, and a green run is
//! recorded: see [`crate::prepush_stamp`] for what a pass is keyed on (the
//! step's cargo arguments, so a narrow pass never covers a wider run; the tree
//! hash of `HEAD`; the toolchain; the environment) and why a dirty working tree
//! neither uses nor writes one.
//!
//! As in [`crate::affected`], the decisions ([`push_from_env`], [`plan`],
//! [`resolve_selection`]) are pure and unit-tested; running
//! git and cargo is confined to [`run`], and [`execute`] takes both, the stamps
//! and the output as parameters so its own behavior (the range it diffs,
//! `--dry-run`, a failing cargo, a skipped or recorded pass, and what it prints
//! about each) is tested too.

use std::io::Write;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::affected::{self, DEFAULT_BASE, Graph, Selection};
use crate::prepush_stamp::{self, Lookup, Stamps};

/// The e2e harness crate. `cargo test` excludes it (its scenarios run through
/// `cargo xtask e2e`), and its harness target is linted by its own step.
const E2E_CRATE: &str = "e2e-cucumber";

/// One cargo step of the pre-push hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Check {
    /// `cargo clippy --all-targets -- -D warnings`
    Clippy,
    /// `cargo clippy -p e2e-cucumber --test e2e -- -D warnings`: the harness
    /// target that `--all-targets` leaves out of the step above.
    ClippyE2e,
    /// `cargo test --all-targets`, without the e2e harness.
    Test,
}

impl Check {
    const fn label(self) -> &'static str {
        match self {
            Self::Clippy => "clippy",
            Self::ClippyE2e => "clippy-e2e",
            Self::Test => "test",
        }
    }
}

/// The range being pushed: what the remote already has, and what is pushed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Push {
    /// The remote's side of the range; `None` when prek gave none (a manual
    /// run, or a brand-new remote), so the default base applies.
    from: Option<String>,
    /// The commit being pushed.
    to: String,
}

impl Push {
    fn base(&self) -> &str {
        self.from.as_deref().unwrap_or(DEFAULT_BASE)
    }
}

/// Build the [`Push`] from the values of `PRE_COMMIT_FROM_REF` and
/// `PRE_COMMIT_TO_REF`. An unset, empty or all-zero from-ref (git's spelling of
/// "the remote has nothing") means no remote base; an empty to-ref means `HEAD`.
pub fn push_from_env(from: Option<String>, to: Option<String>) -> Push {
    let from = from
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && !value.chars().all(|c| c == '0'));
    let to = to
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "HEAD".to_string());
    Push { from, to }
}

/// What a step does for a given selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// Nothing for this step to check; the reason is printed.
    Skip(String),
    /// Run `cargo` with these arguments.
    Cargo(Vec<String>),
}

/// Decide what `check` runs for `selection`.
///
/// An empty package set is treated as the whole workspace: a bare
/// `cargo clippy --all-targets` in a virtual workspace lints every default
/// member anyway, and an empty set means the selection went wrong, not that
/// nothing changed.
pub fn plan(check: Check, selection: &Selection) -> Plan {
    let packages = match selection {
        Selection::Empty => {
            return Plan::Skip("no Rust crate is affected by the pushed changes".to_string());
        }
        Selection::Packages(packages) if !packages.is_empty() => Some(packages),
        Selection::Workspace | Selection::Packages(_) => None,
    };

    match check {
        Check::Clippy => {
            let mut args = vec!["clippy".to_string()];
            args.extend(package_args(packages.map(|p| p.iter().map(String::as_str))));
            args.extend(["--all-targets", "--", "-D", "warnings"].map(String::from));
            Plan::Cargo(args)
        }
        Check::ClippyE2e => {
            if packages.is_some_and(|p| !p.contains(E2E_CRATE)) {
                return Plan::Skip(format!(
                    "{E2E_CRATE} is not among the affected crates, so its harness is unchanged"
                ));
            }
            Plan::Cargo(
                [
                    "clippy", "-p", E2E_CRATE, "--test", "e2e", "--", "-D", "warnings",
                ]
                .map(String::from)
                .to_vec(),
            )
        }
        Check::Test => {
            let mut args = vec!["test".to_string()];
            match packages {
                None => args.extend(
                    ["--workspace", "--all-targets", "--exclude", E2E_CRATE].map(String::from),
                ),
                Some(packages) => {
                    let testable = packages.iter().filter(|name| *name != E2E_CRATE);
                    let before = args.len();
                    args.extend(package_args(Some(testable.map(String::as_str))));
                    if args.len() == before {
                        return Plan::Skip(format!(
                            "the only affected crate is {E2E_CRATE}, which this step does not test"
                        ));
                    }
                    args.push("--all-targets".to_string());
                }
            }
            Plan::Cargo(args)
        }
    }
}

/// `--workspace` for `None`, otherwise `-p <name>` for each package.
fn package_args<'a>(packages: Option<impl Iterator<Item = &'a str>>) -> Vec<String> {
    match packages {
        None => vec!["--workspace".to_string()],
        Some(names) => names
            .flat_map(|name| ["-p".to_string(), name.to_string()])
            .collect(),
    }
}

/// Resolve the selection for the pushed changes, falling back to the whole
/// workspace (with the reason) when either input cannot be obtained.
pub fn resolve_selection(
    load_graph: impl FnOnce() -> Result<Graph>,
    changed_files: impl FnOnce() -> Result<Vec<String>>,
) -> (Selection, Option<String>) {
    let resolved = changed_files()
        .context("could not list the changed files")
        .and_then(|files| {
            // The hook only runs when the push contains a Rust file (or, for the
            // two clippy hooks, `clippy.toml`), so a range that lists no changed
            // files means the range is wrong, not that nothing changed.
            if files.is_empty() {
                bail!("the pushed range lists no changed files");
            }
            let graph = load_graph().context("could not read the workspace dependency graph")?;
            // CI's `cargo xtask affected` selects through the same function.
            Ok(affected::select_with_readers(&files, &graph))
        });
    match resolved {
        Ok(selection) => (selection, None),
        Err(error) => {
            // git's usage text after the first line adds nothing to the reason.
            let reason = format!("{error:#}");
            let first_line = reason.lines().next().unwrap_or_default().to_string();
            (Selection::Workspace, Some(first_line))
        }
    }
}

fn describe(selection: &Selection) -> String {
    match selection {
        Selection::Workspace => "the whole workspace".to_string(),
        Selection::Empty => "no Rust crate".to_string(),
        Selection::Packages(packages) => format!(
            "{} affected crate(s): {}",
            packages.len(),
            packages
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The line printed before a step, naming its scope and why.
fn banner(check: Check, push: &Push, selection: &Selection, fallback: Option<&str>) -> String {
    let range = format!("{}...{}", push.base(), push.to);
    match fallback {
        Some(reason) => format!(
            "pre-push-check {}: could not determine the affected crates ({reason}); \
             checking the whole workspace",
            check.label()
        ),
        // The claim about CI is backed by `.github/workflows/ci.yml`: the
        // `windows-debug` job runs `cargo test --workspace --all-targets` when
        // the `heavy` category matches, which every Rust change does (and the
        // hook only runs for a push that changes Rust), and the `test` job
        // selects `--workspace` for every event but `pull_request` (the merge
        // queue and pushes to main). The test
        // `the_banner_claim_about_ci_matches_the_workflow` fails if any of these
        // stops being true (the Windows test command, its `heavy` gate, `heavy`
        // matching Rust files, the `--workspace` forcing), so change the text
        // and that test together.
        None => format!(
            "pre-push-check {}: {} for the changes in {range}; CI's Windows job, merge queue \
             and main still check the whole workspace",
            check.label(),
            describe(selection)
        ),
    }
}

/// The refs prek exports to a pre-push hook.
const FROM_REF_VAR: &str = "PRE_COMMIT_FROM_REF";
const TO_REF_VAR: &str = "PRE_COMMIT_TO_REF";

/// The pushed range as prek exported it; either side may be absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Env {
    from_ref: Option<String>,
    to_ref: Option<String>,
}

impl Env {
    /// Read the two refs through `lookup`, so tests need no process environment.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            from_ref: lookup(FROM_REF_VAR),
            to_ref: lookup(TO_REF_VAR),
        }
    }
}

/// What one invocation asks for.
#[derive(Debug, Clone, Default)]
pub struct Request {
    /// Replaces the range's base (`--base`).
    pub base: Option<String>,
    /// Print the plan without running or recording anything (`--dry-run`).
    pub dry_run: bool,
    /// The refs prek exported.
    pub env: Env,
}

/// Run one step: resolve the range (an explicit `base` wins over prek's
/// from-ref), pick the affected crates, print the scope, and hand the planned
/// cargo arguments to `run_cargo` unless the step is skipped, `dry_run`, or a
/// pass of exactly this command on this tree is on record (`stamps`).
///
/// A pass is recorded only after `run_cargo` succeeded. `load_graph`,
/// `changed_files` (given the range's base and tip), `run_cargo` and `out` are
/// parameters so the behavior around them is unit-tested.
pub fn execute(
    check: Check,
    request: Request,
    stamps: &Stamps<'_>,
    load_graph: impl FnOnce() -> Result<Graph>,
    changed_files: impl FnOnce(&str, &str) -> Result<Vec<String>>,
    mut run_cargo: impl FnMut(&[String]) -> Result<()>,
    out: &mut impl Write,
) -> Result<()> {
    let label = check.label();
    let Request { base, dry_run, env } = request;
    let push = push_from_env(base.or(env.from_ref), env.to_ref);
    let (selection, fallback) =
        resolve_selection(load_graph, || changed_files(push.base(), &push.to));
    writeln!(
        out,
        "{}",
        banner(check, &push, &selection, fallback.as_deref())
    )?;

    let args = match plan(check, &selection) {
        Plan::Skip(reason) => {
            writeln!(out, "pre-push-check {label}: skipped, {reason}")?;
            return Ok(());
        }
        Plan::Cargo(args) => args,
    };

    writeln!(out, "pre-push-check {label}: cargo {}", args.join(" "))?;
    let key = match stamps.lookup(label, &args) {
        Lookup::Hit { name, age_secs } => {
            if dry_run {
                writeln!(
                    out,
                    "pre-push-check {label}: a real run would skip this step, it passed {} ago",
                    prepush_stamp::age_text(age_secs)
                )?;
            } else {
                let location = stamps.location();
                for line in prepush_stamp::skip_lines(label, &args, age_secs, &name, &location) {
                    writeln!(out, "{line}")?;
                }
            }
            return Ok(());
        }
        Lookup::Miss { key, note } => {
            if let Some(note) = note {
                writeln!(out, "pre-push-check {label}: {note}")?;
            }
            key
        }
    };
    if dry_run {
        return Ok(());
    }

    // `--rerun` replaces the recorded pass: drop it first, so a run that fails
    // leaves nothing for the next ordinary run to skip on, and write a fresh one
    // only once cargo succeeded.
    if stamps.rerun
        && let Some(note) = key.as_ref().and_then(|key| stamps.invalidate(key))
    {
        writeln!(out, "pre-push-check {label}: {note}")?;
    }
    run_cargo(&args)?;
    if let Some(note) = key.and_then(|key| stamps.record(&key)) {
        writeln!(out, "pre-push-check {label}: {note}")?;
    }
    Ok(())
}

/// Run `cargo` with `args`, failing when it does.
fn spawn_cargo(args: &[String]) -> Result<()> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .args(args)
        .status()
        .context("failed to run cargo")?;
    if !status.success() {
        bail!("`cargo {}` failed ({status})", args.join(" "));
    }
    Ok(())
}

/// A process environment variable as text. A value that is not valid Unicode is
/// still a set value (`std::env::var` reports it as an error, which `.ok()` would
/// read as unset, so a non-Unicode `ROCM_PRE_PUSH_RERUN` would be ignored).
fn process_env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
}

/// What the `pre-push-check` command line asked for.
pub struct Flags {
    pub base: Option<String>,
    pub dry_run: bool,
    pub rerun: bool,
}

/// The world one invocation runs against: where passes are kept, how the tree
/// and toolchain are probed, the clock, and how a process variable is read.
pub struct World<'a> {
    pub probe: &'a dyn prepush_stamp::Probe,
    pub store: &'a dyn prepush_stamp::Store,
    /// Seconds since the epoch.
    pub now: u64,
    pub env: &'a dyn Fn(&str) -> Option<String>,
}

/// Entry point for the `pre-push-check` subcommand.
pub fn run(check: Check, base: Option<String>, dry_run: bool, rerun: bool) -> Result<()> {
    let store: Box<dyn prepush_stamp::Store> = match prepush_stamp::FsStore::in_git_common_dir(None)
    {
        Ok(store) => Box::new(store),
        Err(error) => Box::new(prepush_stamp::UnavailableStore(format!("{error:#}"))),
    };
    let probe = prepush_stamp::SystemProbe::default();
    run_in(
        check,
        Flags {
            base,
            dry_run,
            rerun,
        },
        &World {
            probe: &probe,
            store: store.as_ref(),
            now: prepush_stamp::now_secs(),
            env: &process_env,
        },
        affected::load_graph,
        affected::changed_files_between,
        spawn_cargo,
        &mut std::io::stdout().lock(),
    )
}

/// [`run`] with the world injected: this is where `--rerun` and
/// `ROCM_PRE_PUSH_RERUN` reach the stamp logic, so a test drives it with a fake
/// store and sees whether a recorded pass is honored.
pub fn run_in(
    check: Check,
    flags: Flags,
    world: &World<'_>,
    load_graph: impl FnOnce() -> Result<Graph>,
    changed_files: impl FnOnce(&str, &str) -> Result<Vec<String>>,
    run_cargo: impl FnMut(&[String]) -> Result<()>,
    out: &mut impl Write,
) -> Result<()> {
    let Flags {
        base,
        dry_run,
        rerun,
    } = flags;
    let stamps = Stamps::new(world.probe, world.store, world.now, rerun, |name| {
        (world.env)(name)
    });
    let request = Request {
        base,
        dry_run,
        env: Env::from_lookup(|name| (world.env)(name)),
    };
    execute(
        check,
        request,
        &stamps,
        load_graph,
        changed_files,
        run_cargo,
        out,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::prepush_stamp::fakes::{FakeProbe, MemStore};
    use crate::workflow_contract::tests as ci_filters;

    /// A fixed clock for the stamp tests.
    const NOW: u64 = 1_800_000_000;

    fn packages(names: &[&str]) -> Selection {
        Selection::Packages(names.iter().map(|name| (*name).to_string()).collect())
    }

    fn args(plan: &Plan) -> String {
        match plan {
            Plan::Cargo(args) => args.join(" "),
            Plan::Skip(reason) => panic!("expected a cargo run, got a skip: {reason}"),
        }
    }

    fn skip_reason(plan: &Plan) -> &str {
        match plan {
            Plan::Skip(reason) => reason,
            Plan::Cargo(args) => panic!("expected a skip, got cargo {}", args.join(" ")),
        }
    }

    #[test]
    fn a_real_from_ref_is_the_base() {
        let push = push_from_env(Some("abc123".into()), Some("def456".into()));
        assert_eq!(push.base(), "abc123");
        assert_eq!(push.to, "def456");
    }

    #[test]
    fn no_remote_base_means_the_default_base() {
        for from in [
            None,
            Some(String::new()),
            Some("  ".into()),
            Some("0".repeat(40)),
        ] {
            let push = push_from_env(from.clone(), None);
            assert_eq!(push.base(), DEFAULT_BASE, "from-ref {from:?}");
            assert_eq!(push.to, "HEAD");
        }
    }

    #[test]
    fn clippy_checks_the_whole_workspace_or_the_selected_crates() {
        assert_eq!(
            args(&plan(Check::Clippy, &Selection::Workspace)),
            "clippy --workspace --all-targets -- -D warnings"
        );
        assert_eq!(
            args(&plan(Check::Clippy, &packages(&["rocm", "rocm-core"]))),
            "clippy -p rocm -p rocm-core --all-targets -- -D warnings"
        );
    }

    #[test]
    fn test_excludes_the_e2e_harness_either_way() {
        assert_eq!(
            args(&plan(Check::Test, &Selection::Workspace)),
            "test --workspace --all-targets --exclude e2e-cucumber"
        );
        assert_eq!(
            args(&plan(
                Check::Test,
                &packages(&["e2e-cucumber", "rocm-core"])
            )),
            "test -p rocm-core --all-targets"
        );
    }

    #[test]
    fn test_is_skipped_with_a_reason_when_only_the_e2e_harness_is_affected() {
        let plan = plan(Check::Test, &packages(&["e2e-cucumber"]));
        assert!(skip_reason(&plan).contains("e2e-cucumber"));
    }

    #[test]
    fn the_e2e_harness_is_linted_only_when_it_is_affected() {
        let want = "clippy -p e2e-cucumber --test e2e -- -D warnings";
        assert_eq!(args(&plan(Check::ClippyE2e, &Selection::Workspace)), want);
        assert_eq!(
            args(&plan(
                Check::ClippyE2e,
                &packages(&["e2e-cucumber", "xtask"])
            )),
            want
        );
        let plan = plan(Check::ClippyE2e, &packages(&["xtask"]));
        assert!(skip_reason(&plan).contains("not among the affected crates"));
    }

    #[test]
    fn every_step_is_skipped_with_a_reason_when_no_crate_is_affected() {
        for check in [Check::Clippy, Check::ClippyE2e, Check::Test] {
            let plan = plan(check, &Selection::Empty);
            assert!(skip_reason(&plan).contains("no Rust crate"), "{check:?}");
        }
    }

    #[test]
    fn an_empty_package_set_is_the_whole_workspace_not_a_bare_cargo_call() {
        let empty = Selection::Packages(BTreeSet::new());
        assert_eq!(
            args(&plan(Check::Clippy, &empty)),
            "clippy --workspace --all-targets -- -D warnings"
        );
        assert_eq!(
            args(&plan(Check::Test, &empty)),
            "test --workspace --all-targets --exclude e2e-cucumber"
        );
    }

    fn graph() -> Graph {
        let mut rev_deps = BTreeMap::new();
        rev_deps.insert("core".to_string(), vec!["app".to_string()]);
        rev_deps.insert("app".to_string(), vec![]);
        rev_deps.insert("xtask".to_string(), vec![]);
        Graph {
            crate_dirs: vec![
                ("core".into(), "crates/core".into()),
                ("app".into(), "apps/app".into()),
                ("xtask".into(), "xtask".into()),
            ],
            rev_deps,
        }
    }

    fn files(paths: &[&str]) -> Result<Vec<String>> {
        Ok(paths.iter().map(|path| (*path).to_string()).collect())
    }

    #[test]
    fn a_known_selection_comes_from_the_changed_files_and_has_no_fallback_reason() {
        let (selection, fallback) =
            resolve_selection(|| Ok(graph()), || files(&["xtask/src/main.rs"]));
        assert_eq!(selection, packages(&["xtask"]));
        assert_eq!(fallback, None);

        // Any Rust file also selects xtask, whose contract tests scan them all.
        let (selection, _) =
            resolve_selection(|| Ok(graph()), || files(&["crates/core/src/lib.rs"]));
        assert_eq!(selection, packages(&["core", "app", "xtask"]));

        let (selection, _) = resolve_selection(|| Ok(graph()), || files(&["CHANGELOG.md"]));
        assert_eq!(selection, Selection::Empty);
    }

    /// The hook fires only for a push that contains a Rust file (or, for the
    /// two clippy hooks, `clippy.toml`), so an empty list of changed files is a
    /// wrong range (a base that is not an ancestor, a from-ref equal to the
    /// pushed commit), not a clean push. Skipping all three
    /// steps on it would turn a bad range into a silent pass.
    #[test]
    fn an_empty_list_of_changed_files_falls_back_to_the_whole_workspace_and_says_why() {
        let (selection, fallback) = resolve_selection(|| Ok(graph()), || files(&[]));
        assert_eq!(selection, Selection::Workspace);
        let reason = fallback.expect("an empty range gives a reason");
        assert!(reason.contains("lists no changed files"), "{reason}");
        // Every step then runs over the workspace instead of skipping.
        for check in [Check::Clippy, Check::ClippyE2e, Check::Test] {
            assert!(
                matches!(plan(check, &selection), Plan::Cargo(_)),
                "{check:?}"
            );
        }
        // The banner names the fallback and the reason, not a narrowed scope.
        let line = banner(
            Check::Test,
            &push_from_env(None, None),
            &selection,
            Some(&reason),
        );
        assert!(
            line.contains("could not determine the affected crates"),
            "{line}"
        );
        assert!(line.contains("lists no changed files"), "{line}");
    }

    #[test]
    fn a_failed_diff_falls_back_to_the_whole_workspace_and_says_why() {
        let (selection, fallback) = resolve_selection(
            || Ok(graph()),
            || bail!("`git diff origin/main...HEAD` failed: unknown revision"),
        );
        assert_eq!(selection, Selection::Workspace);
        let reason = fallback.expect("a fallback reason");
        assert!(
            reason.contains("could not list the changed files"),
            "{reason}"
        );
        assert!(reason.contains("unknown revision"), "{reason}");
    }

    #[test]
    fn the_fallback_reason_is_one_line() {
        let (_, fallback) = resolve_selection(
            || Ok(graph()),
            || bail!("`git diff x...HEAD` failed: unknown revision\nUse '--' to separate paths"),
        );
        let reason = fallback.expect("a fallback reason");
        assert!(!reason.contains('\n'), "{reason}");
        assert!(reason.contains("unknown revision"), "{reason}");
    }

    #[test]
    fn a_failed_graph_load_falls_back_to_the_whole_workspace_and_says_why() {
        let (selection, fallback) = resolve_selection(
            || bail!("`cargo metadata` failed"),
            || files(&["xtask/src/main.rs"]),
        );
        assert_eq!(selection, Selection::Workspace);
        let reason = fallback.expect("a fallback reason");
        assert!(reason.contains("dependency graph"), "{reason}");
    }

    #[test]
    fn a_shared_input_still_selects_the_whole_workspace() {
        for path in [
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
            "clippy.toml",
            ".github/workflows/ci.yml",
        ] {
            let (selection, fallback) =
                resolve_selection(|| Ok(graph()), || files(&["xtask/src/main.rs", path]));
            assert_eq!(selection, Selection::Workspace, "{path}");
            assert_eq!(fallback, None, "{path}");
        }
    }

    #[test]
    fn the_banner_names_the_scope_and_where_ci_still_checks_everything() {
        let push = push_from_env(Some("abc123".into()), Some("def456".into()));
        let line = banner(Check::Test, &push, &packages(&["rocm-core", "xtask"]), None);
        assert!(
            line.contains("2 affected crate(s): rocm-core, xtask"),
            "{line}"
        );
        assert!(line.contains("abc123...def456"), "{line}");
        assert!(
            line.contains("Windows job, merge queue and main still check the whole workspace"),
            "{line}"
        );
    }

    /// The banner tells the developer that the narrowing only affects the local
    /// hook. That is true while CI's Windows job tests the whole workspace and the
    /// `test` job is forced to `--workspace` outside pull requests; this reads the
    /// workflow so the claim cannot go stale without a test failing.
    #[test]
    fn the_banner_claim_about_ci_matches_the_workflow() {
        let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/ci.yml"))
            .expect("read ci.yml");
        let mut windows_debug = String::new();
        let mut inside = false;
        for line in workflow.lines() {
            if line == "  windows-debug:" {
                inside = true;
                continue;
            }
            // The next job key at two-space indent ends the job.
            let next_key = line.starts_with("  ")
                && !line.starts_with("   ")
                && !line.trim_start().starts_with('#');
            if inside && next_key {
                break;
            }
            if inside {
                windows_debug.push_str(line);
                windows_debug.push('\n');
            }
        }
        assert!(
            windows_debug.contains("cargo test --workspace --all-targets"),
            "the `windows-debug` job no longer tests the whole workspace; update the banner text"
        );
        // That test step runs when `heavy` matches, and the hook only fires for
        // a push that changes a Rust file, which `heavy` must therefore match.
        let lines: Vec<&str> = windows_debug.lines().collect();
        let run = lines
            .iter()
            .position(|line| line.trim() == "run: cargo test --workspace --all-targets")
            .expect("the `windows-debug` test step");
        let step = lines[..run]
            .iter()
            .rposition(|line| line.trim_start().starts_with("- name:"))
            .expect("the test step's name");
        let gates: Vec<&str> = lines[step..run]
            .iter()
            .filter_map(|line| line.trim().strip_prefix("if:"))
            .map(str::trim)
            .collect();
        assert_eq!(
            gates,
            ["needs.changes.outputs.heavy == 'true'"],
            "the `windows-debug` test step's gate changed; update the banner text"
        );
        let filters = ci_filters::paths_filters(&ci_filters::read_classify_action());
        let heavy = ci_filters::paths_filter(&filters, "heavy");
        for rust_file in [
            "build.rs",
            "crates/rocm-core/src/lib.rs",
            "tests/e2e-cucumber/tests/x.rs",
        ] {
            assert!(
                ci_filters::filter_matches(heavy, rust_file),
                "`heavy` does not match {rust_file}, so a Rust push can skip the Windows tests; \
                 update the banner text"
            );
        }
        assert!(
            workflow.contains("\"${GITHUB_EVENT_NAME}\" = \"pull_request\"")
                && workflow.contains("sel=\"--workspace\""),
            "the `test` job no longer forces `--workspace` outside pull requests; update the banner text"
        );
    }

    #[test]
    fn the_banner_of_a_fallback_names_the_reason_not_a_narrowed_scope() {
        let push = push_from_env(None, None);
        let line = banner(
            Check::Clippy,
            &push,
            &Selection::Workspace,
            Some("unknown revision"),
        );
        assert!(
            line.contains("could not determine the affected crates (unknown revision)"),
            "{line}"
        );
        assert!(line.contains("checking the whole workspace"), "{line}");
    }

    // ---- files read across crate boundaries ----

    /// The directory-based selection attributes the fixture and the step file to
    /// the e2e crate and nothing else, so the `rocm-core` tests that
    /// `include_str!` the fixture never ran and the `test` step skipped. The
    /// widening itself is tested in [`crate::cross_crate_reads`].
    #[test]
    fn a_fixture_edit_with_an_e2e_step_still_tests_the_crate_that_includes_it() {
        let mut rev_deps = BTreeMap::new();
        for name in ["rocm-core", "xtask", "e2e-cucumber"] {
            rev_deps.insert(name.to_string(), vec![]);
        }
        let graph = Graph {
            crate_dirs: vec![
                ("rocm-core".into(), "crates/rocm-core".into()),
                ("xtask".into(), "xtask".into()),
                ("e2e-cucumber".into(), "tests/e2e-cucumber".into()),
            ],
            rev_deps,
        };
        let changed = [
            "tests/e2e-cucumber/fixtures/powershell_text_rule_cases.txt".to_string(),
            "tests/e2e-cucumber/tests/e2e/diagnose_steps.rs".to_string(),
        ];
        let selection = affected::select_with_readers(&changed, &graph);
        let Plan::Cargo(args) = plan(Check::Test, &selection) else {
            panic!("the test step must run, not skip");
        };
        assert!(args.join(" ").contains("-p rocm-core"), "{args:?}");
    }

    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    // ---- execute: the range, --dry-run and cargo's result ----

    type Ranges = Vec<(String, String)>;

    fn env(from: Option<&str>, to: Option<&str>) -> Env {
        Env {
            from_ref: from.map(String::from),
            to_ref: to.map(String::from),
        }
    }

    /// Everything one `execute` call did.
    struct StepRun {
        result: Result<()>,
        ranges: Ranges,
        /// The argument lines handed to cargo.
        cargo: Vec<String>,
        /// What the step printed.
        out: String,
    }

    /// Run a step against canned inputs, with the given stamps; `files` of
    /// `None` makes the diff fail.
    fn run_step_with(
        check: Check,
        request: Request,
        stamps: &Stamps<'_>,
        files: Option<&[&str]>,
        cargo_fails: bool,
    ) -> StepRun {
        let ranges = std::cell::RefCell::new(Vec::new());
        let cargo = std::cell::RefCell::new(Vec::new());
        let mut out = Vec::new();
        let result = execute(
            check,
            request,
            stamps,
            || Ok(graph()),
            |base, tip| {
                ranges
                    .borrow_mut()
                    .push((base.to_string(), tip.to_string()));
                match files {
                    Some(paths) => Ok(paths.iter().map(|path| (*path).to_string()).collect()),
                    None => bail!("`git diff` failed: unknown revision"),
                }
            },
            |args| {
                cargo.borrow_mut().push(args.join(" "));
                if cargo_fails {
                    bail!("`cargo` failed (exit status: 101)")
                }
                Ok(())
            },
            &mut out,
        );
        StepRun {
            result,
            ranges: ranges.into_inner(),
            cargo: cargo.into_inner(),
            out: String::from_utf8(out).expect("the step prints UTF-8"),
        }
    }

    /// Run a step with stamps that never apply (the working tree reads as dirty),
    /// so these tests see only the range, `--dry-run` and cargo's result.
    /// Returns the result, every `(base, tip)` the diff was asked for, and the
    /// argument lines handed to cargo.
    fn run_step(
        check: Check,
        base: Option<&str>,
        dry_run: bool,
        env: Env,
        files: Option<&[&str]>,
        cargo_fails: bool,
    ) -> (Result<()>, Ranges, Vec<String>) {
        let probe = FakeProbe::clean("tree");
        *probe.tree.borrow_mut() = Err("the working tree is dirty".to_string());
        let store = MemStore::default();
        let stamps = Stamps {
            probe: &probe,
            store: &store,
            now: NOW,
            rerun: false,
        };
        let request = Request {
            base: base.map(String::from),
            dry_run,
            env,
        };
        let run = run_step_with(check, request, &stamps, files, cargo_fails);
        (run.result, run.ranges, run.cargo)
    }

    const XTASK_ONLY: Option<&[&str]> = Some(&["xtask/src/main.rs"]);

    #[test]
    fn a_passing_cargo_run_is_ok_and_ran_the_planned_arguments() {
        let (result, _, cargo) =
            run_step(Check::Test, None, false, env(None, None), XTASK_ONLY, false);
        result.expect("a passing step");
        assert_eq!(cargo, ["test -p xtask --all-targets"]);
    }

    #[test]
    fn a_failing_cargo_run_fails_the_step() {
        let (result, _, cargo) =
            run_step(Check::Test, None, false, env(None, None), XTASK_ONLY, true);
        let error = result.expect_err("a failing cargo must fail the hook");
        assert!(format!("{error:#}").contains("101"), "{error:#}");
        assert_eq!(cargo.len(), 1);
    }

    #[test]
    fn the_range_comes_from_the_exported_refs() {
        let (_, ranges, _) = run_step(
            Check::Test,
            None,
            true,
            env(Some("envfrom"), Some("envto")),
            XTASK_ONLY,
            false,
        );
        assert_eq!(ranges, [("envfrom".to_string(), "envto".to_string())]);
    }

    #[test]
    fn an_explicit_base_replaces_the_exported_from_ref_but_not_the_tip() {
        let (_, ranges, _) = run_step(
            Check::Test,
            Some("override"),
            true,
            env(Some("envfrom"), Some("envto")),
            XTASK_ONLY,
            false,
        );
        assert_eq!(ranges, [("override".to_string(), "envto".to_string())]);
    }

    #[test]
    fn without_refs_the_range_is_the_default_base_to_head() {
        let (_, ranges, _) = run_step(Check::Test, None, true, env(None, None), XTASK_ONLY, false);
        assert_eq!(ranges, [(DEFAULT_BASE.to_string(), "HEAD".to_string())]);
        let (_, ranges, _) = run_step(
            Check::Test,
            None,
            true,
            env(Some("envfrom"), None),
            XTASK_ONLY,
            false,
        );
        assert_eq!(ranges, [("envfrom".to_string(), "HEAD".to_string())]);
    }

    #[test]
    fn a_dry_run_selects_but_never_runs_cargo() {
        let (result, ranges, cargo) =
            run_step(Check::Test, None, true, env(None, None), XTASK_ONLY, false);
        result.expect("a dry run");
        assert_eq!(ranges.len(), 1, "the selection is still computed");
        assert!(cargo.is_empty(), "{cargo:?}");
    }

    #[test]
    fn a_skipped_step_never_runs_cargo() {
        let (result, _, cargo) = run_step(
            Check::Test,
            None,
            false,
            env(None, None),
            Some(&["CHANGELOG.md"]),
            true,
        );
        result.expect("a skipped step succeeds even if cargo would fail");
        assert!(cargo.is_empty(), "{cargo:?}");
    }

    #[test]
    fn an_undeterminable_range_runs_the_whole_workspace() {
        let (result, _, cargo) = run_step(Check::Test, None, false, env(None, None), None, false);
        result.expect("the fallback run");
        assert_eq!(
            cargo,
            ["test --workspace --all-targets --exclude e2e-cucumber"]
        );
    }

    #[test]
    fn the_exported_refs_are_read_from_prek_s_variable_names() {
        let lookup = |name: &str| match name {
            "PRE_COMMIT_FROM_REF" => Some("remote-tip".to_string()),
            "PRE_COMMIT_TO_REF" => Some("pushed-tip".to_string()),
            _ => None,
        };
        assert_eq!(
            Env::from_lookup(lookup),
            env(Some("remote-tip"), Some("pushed-tip"))
        );
        assert_eq!(Env::from_lookup(|_| None), Env::default());
    }

    // ---- pass stamps: skip a step that already passed on this exact tree ----

    const XTASK_TEST: &str = "test -p xtask --all-targets";
    const WIDE_TEST: &str = "test --workspace --all-targets --exclude e2e-cucumber";

    fn stamps<'a>(probe: &'a FakeProbe, store: &'a MemStore, now: u64, rerun: bool) -> Stamps<'a> {
        Stamps {
            probe,
            store,
            now,
            rerun,
        }
    }

    fn run_at(
        stamps: &Stamps<'_>,
        check: Check,
        files: &[&str],
        dry_run: bool,
        cargo_fails: bool,
    ) -> StepRun {
        let request = Request {
            dry_run,
            ..Request::default()
        };
        run_step_with(check, request, stamps, Some(files), cargo_fails)
    }

    #[test]
    fn a_green_run_is_recorded_and_the_next_identical_run_is_skipped_with_the_reason() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let first = run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        first.result.expect("the first run");
        assert_eq!(first.cargo, [XTASK_TEST]);
        assert_eq!(store.files.borrow().len(), 1, "a green run is recorded");

        let second = run_at(
            &stamps(&probe, &store, NOW + 90, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        second.result.expect("the skipped run");
        assert!(
            second.cargo.is_empty(),
            "the step must not run again: {:?}",
            second.cargo
        );
        assert!(
            second
                .out
                .contains(&format!("pre-push-check test: skipped, this tree already passed `cargo {XTASK_TEST}` 1 min ago")),
            "{}",
            second.out
        );
        // The skip names what it does not cover and how to force a run.
        assert!(
            second.out.contains("user-level cargo configuration"),
            "{}",
            second.out
        );
        assert!(second.out.contains("--rerun"), "{}", second.out);
        assert!(second.out.contains("files git ignores"), "{}", second.out);
        assert!(second.out.contains("installed tools"), "{}", second.out);
        assert!(second.out.contains("flaky tests"), "{}", second.out);
        assert!(
            second.out.contains(prepush_stamp::RERUN_VAR),
            "{}",
            second.out
        );
        assert!(second.out.contains("<memory>"), "{}", second.out);
        assert_eq!(store.files.borrow().len(), 1, "a skip records nothing");
    }

    /// A pushed range that lists no changed files means the range is wrong, not
    /// that nothing changed, so on the stamped path too it runs the whole
    /// workspace: it is never skipped as "no Rust crate", an earlier narrow pass
    /// does not cover it, and it is skipped only by its own earlier pass.
    #[test]
    fn an_empty_changed_file_list_runs_the_whole_workspace_on_the_stamped_path() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let narrow = run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        assert_eq!(narrow.cargo, [XTASK_TEST]);

        let empty = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &[],
            false,
            false,
        );
        empty.result.expect("the empty-range run");
        assert_eq!(empty.cargo, [WIDE_TEST], "{}", empty.out);
        assert!(
            empty
                .out
                .contains("the pushed range lists no changed files"),
            "{}",
            empty.out
        );
        assert!(!empty.out.contains("skipped"), "{}", empty.out);
        assert_eq!(store.files.borrow().len(), 2, "the wide pass is recorded");

        let again = run_at(
            &stamps(&probe, &store, NOW + 10, false),
            Check::Test,
            &[],
            false,
            false,
        );
        assert!(again.cargo.is_empty(), "{:?}", again.cargo);
        assert!(
            again
                .out
                .contains(&format!("this tree already passed `cargo {WIDE_TEST}`")),
            "{}",
            again.out
        );

        // The lint step falls back the same way and is not skipped as "no crate".
        let clippy = run_at(
            &stamps(&probe, &store, NOW + 15, false),
            Check::Clippy,
            &[],
            false,
            false,
        );
        assert_eq!(
            clippy.cargo,
            ["clippy --workspace --all-targets -- -D warnings"],
            "{}",
            clippy.out
        );
    }

    #[test]
    fn a_narrow_pass_never_lets_a_wider_selection_skip() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let narrow = run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        assert_eq!(narrow.cargo, [XTASK_TEST]);

        // The same tree, but the push now also touches the lockfile: the whole
        // workspace. The earlier `-p xtask` pass says nothing about that.
        let wide = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &["Cargo.lock"],
            false,
            false,
        );
        assert_eq!(wide.cargo, [WIDE_TEST], "{}", wide.out);
        assert!(!wide.out.contains("skipped"), "{}", wide.out);
        assert_eq!(store.files.borrow().len(), 2, "both passes are on record");

        // And each is skipped on its own next run.
        let again = run_at(
            &stamps(&probe, &store, NOW + 10, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        assert!(again.cargo.is_empty(), "{}", again.out);
    }

    #[test]
    fn a_different_tree_runs_again() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        )
        .result
        .expect("first");
        *probe.tree.borrow_mut() = Ok("tree-2".to_string());
        let second = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        assert_eq!(second.cargo, [XTASK_TEST], "{}", second.out);
        assert_eq!(store.files.borrow().len(), 2);
    }

    #[test]
    fn a_different_toolchain_or_environment_runs_again() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        )
        .result
        .expect("first");

        *probe.toolchain.borrow_mut() = Ok("rustc 1.97.0\ncargo 1.97.0\n".to_string());
        let toolchain = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        assert_eq!(
            toolchain.cargo,
            [XTASK_TEST],
            "a new toolchain must not reuse the pass"
        );

        *probe.env.borrow_mut() = "env-b".to_string();
        let env = run_at(
            &stamps(&probe, &store, NOW + 10, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        assert_eq!(
            env.cargo,
            [XTASK_TEST],
            "a new RUSTFLAGS and the like must not reuse the pass"
        );
    }

    #[test]
    fn a_failed_step_records_nothing_and_a_passed_sibling_still_skips_on_retry() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Clippy,
            &files,
            false,
            false,
        )
        .result
        .expect("clippy passes");
        let failed = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &files,
            false,
            true,
        );
        failed.result.expect_err("test fails");
        assert_eq!(
            store.files.borrow().len(),
            1,
            "only the clippy pass is on record"
        );
        assert!(
            store
                .files
                .borrow()
                .keys()
                .all(|name| name.starts_with("clippy-")),
            "{:?}",
            store.files.borrow().keys()
        );

        // The retry skips the step that passed and re-runs the one that failed.
        let clippy = run_at(
            &stamps(&probe, &store, NOW + 10, false),
            Check::Clippy,
            &files,
            false,
            false,
        );
        assert!(clippy.cargo.is_empty(), "{}", clippy.out);
        let test = run_at(
            &stamps(&probe, &store, NOW + 10, false),
            Check::Test,
            &files,
            false,
            false,
        );
        assert_eq!(test.cargo, [XTASK_TEST]);
        assert_eq!(store.files.borrow().len(), 2);
    }

    #[test]
    fn a_dirty_working_tree_neither_skips_nor_records_and_says_so() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("clean run");
        assert_eq!(store.files.borrow().len(), 1);

        *probe.tree.borrow_mut() =
            Err("the working tree has 2 uncommitted or untracked path(s)".to_string());
        let dirty = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &files,
            false,
            false,
        );
        assert_eq!(
            dirty.cargo,
            [XTASK_TEST],
            "a recorded pass is not honored on a dirty tree"
        );
        assert!(
            dirty
                .out
                .contains("pass stamps are not used: the working tree has 2 uncommitted"),
            "{}",
            dirty.out
        );
        assert_eq!(*store.writes.borrow(), 1, "and no pass is recorded from it");
    }

    #[test]
    fn a_tree_the_run_changed_is_not_recorded() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let stamps = stamps(&probe, &store, NOW, false);
        let mut out = Vec::new();
        execute(
            Check::Test,
            Request::default(),
            &stamps,
            || Ok(graph()),
            |_, _| Ok(vec!["xtask/src/main.rs".to_string()]),
            |_| {
                // cargo rewrote a tracked file (`Cargo.lock`, say) while it ran.
                *probe.tree.borrow_mut() =
                    Err("the working tree has 1 uncommitted path".to_string());
                Ok(())
            },
            &mut out,
        )
        .expect("the step itself passed");
        let out = String::from_utf8(out).expect("utf-8");
        assert!(
            out.contains("pass not recorded: the working tree has 1 uncommitted"),
            "{out}"
        );
        assert!(
            store.files.borrow().is_empty(),
            "nothing is recorded for a tree cargo changed"
        );
    }

    /// `--rerun` is used when something the key omits may have changed (user-level
    /// cargo configuration, an installed tool). If that rerun fails, the earlier
    /// pass must not survive to skip the next ordinary run.
    #[test]
    fn a_failed_rerun_does_not_leave_the_earlier_pass_to_skip_the_next_ordinary_run() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("the first, passing run");
        assert_eq!(store.files.borrow().len(), 1);

        let failed = run_at(
            &stamps(&probe, &store, NOW + 100, true),
            Check::Test,
            &files,
            false,
            true,
        );
        assert!(failed.result.is_err(), "the rerun fails");
        assert_eq!(failed.cargo, [XTASK_TEST], "--rerun must run the step");
        assert!(
            store.files.borrow().is_empty(),
            "the earlier pass is dropped before the rerun and nothing replaces it"
        );

        let retry = run_at(
            &stamps(&probe, &store, NOW + 200, false),
            Check::Test,
            &files,
            false,
            false,
        );
        retry.result.expect("the ordinary retry");
        assert_eq!(
            retry.cargo,
            [XTASK_TEST],
            "the ordinary retry must run the step: {}",
            retry.out
        );
        assert!(!retry.out.contains("skipped"), "{}", retry.out);
        assert_eq!(
            store.files.borrow().len(),
            1,
            "the passing retry is recorded"
        );
    }

    #[test]
    fn a_failed_rerun_keeps_the_passes_of_other_steps_and_other_selections() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        for (check, files) in [
            (Check::Test, &["xtask/src/main.rs"][..]),
            (Check::Test, &["Cargo.lock"][..]),
            (Check::Clippy, &["xtask/src/main.rs"][..]),
        ] {
            run_at(
                &stamps(&probe, &store, NOW, false),
                check,
                files,
                false,
                false,
            )
            .result
            .expect("a recorded pass");
        }
        assert_eq!(store.files.borrow().len(), 3);

        let failed = run_at(
            &stamps(&probe, &store, NOW + 100, true),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            true,
        );
        assert!(failed.result.is_err());
        assert_eq!(
            store.files.borrow().len(),
            2,
            "only the stamp the rerun replaces is dropped"
        );
        // The sibling passes still skip.
        for (check, files) in [
            (Check::Test, &["Cargo.lock"][..]),
            (Check::Clippy, &["xtask/src/main.rs"][..]),
        ] {
            let again = run_at(
                &stamps(&probe, &store, NOW + 200, false),
                check,
                files,
                false,
                false,
            );
            assert!(again.cargo.is_empty(), "{:?}: {}", again.cargo, again.out);
        }
    }

    #[test]
    fn a_dry_run_rerun_leaves_every_stamp_in_place() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("the first run");
        let before = store.files.borrow().clone();

        let dry = run_at(
            &stamps(&probe, &store, NOW + 100, true),
            Check::Test,
            &files,
            true,
            false,
        );
        dry.result.expect("the dry run");
        assert!(
            dry.cargo.is_empty(),
            "a dry run runs nothing: {:?}",
            dry.cargo
        );
        assert_eq!(
            *store.files.borrow(),
            before,
            "a dry run neither removes nor rewrites a stamp"
        );
    }

    #[test]
    fn a_passing_rerun_replaces_the_pass_with_a_fresh_one() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("the first run");
        run_at(
            &stamps(&probe, &store, NOW + 100, true),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("the passing rerun");
        assert_eq!(
            store.files.borrow().len(),
            1,
            "one pass, replaced not added"
        );
        let after = run_at(
            &stamps(&probe, &store, NOW + 130, false),
            Check::Test,
            &files,
            false,
            false,
        );
        assert!(after.cargo.is_empty(), "{}", after.out);
        assert!(
            after.out.contains(
                "this tree already passed `cargo test -p xtask --all-targets` under a minute ago"
            ),
            "the fresh pass is the one that skips: {}",
            after.out
        );
    }

    #[test]
    fn rerun_ignores_a_recorded_pass_and_records_a_fresh_one() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("first");
        let forced = run_at(
            &stamps(&probe, &store, NOW + 100, true),
            Check::Test,
            &files,
            false,
            false,
        );
        assert_eq!(forced.cargo, [XTASK_TEST], "--rerun must run the step");
        assert!(
            forced.out.contains("recorded passes are ignored"),
            "{}",
            forced.out
        );
        assert!(
            forced.out.contains(prepush_stamp::RERUN_VAR),
            "{}",
            forced.out
        );
        let recorded: Vec<String> = store
            .files
            .borrow()
            .values()
            .map(|(text, _)| text.clone())
            .collect();
        assert_eq!(recorded.len(), 1, "the new pass replaces the old one");
        assert!(
            recorded[0].contains(&format!("passed-at: {}", NOW + 100)),
            "{recorded:?}"
        );
    }

    #[test]
    fn a_dry_run_reports_a_hit_and_never_runs_or_writes() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        let miss = run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            true,
            false,
        );
        assert!(miss.cargo.is_empty());
        assert!(store.files.borrow().is_empty(), "a dry run records nothing");
        assert!(!miss.out.contains("would skip"), "{}", miss.out);

        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("real run");
        let writes = *store.writes.borrow();
        let hit = run_at(
            &stamps(&probe, &store, NOW + 3600, false),
            Check::Test,
            &files,
            true,
            false,
        );
        assert!(hit.cargo.is_empty());
        assert!(
            hit.out
                .contains("a real run would skip this step, it passed 1 h ago"),
            "{}",
            hit.out
        );
        assert_eq!(*store.writes.borrow(), writes);
    }

    #[test]
    fn a_failed_write_is_reported_and_never_fails_the_step() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        *store.fail_writes.borrow_mut() = true;
        let run = run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        );
        run.result
            .expect("a stamp that cannot be written must not fail a green hook");
        assert_eq!(run.cargo, [XTASK_TEST]);
        assert!(
            run.out.contains(
                "pass not recorded: could not write it under <memory> (read-only file system)"
            ),
            "{}",
            run.out
        );
    }

    #[test]
    fn a_store_that_cannot_be_set_up_still_runs_the_step_and_says_why() {
        let probe = FakeProbe::clean("tree-1");
        let store = prepush_stamp::UnavailableStore("not a git repository".to_string());
        let stamps = Stamps {
            probe: &probe,
            store: &store,
            now: NOW,
            rerun: false,
        };
        let run = run_at(&stamps, Check::Test, &["xtask/src/main.rs"], false, false);
        run.result.expect("the step runs");
        assert_eq!(run.cargo, [XTASK_TEST]);
        assert!(
            run.out
                .contains("could not read the pass stamp (not a git repository)"),
            "{}",
            run.out
        );
        assert!(
            run.out
                .contains("pass not recorded: could not write it under the stamp directory"),
            "{}",
            run.out
        );
    }

    #[test]
    fn a_corrupted_stamp_is_ignored_and_replaced_by_a_real_one() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let files = ["xtask/src/main.rs"];
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &files,
            false,
            false,
        )
        .result
        .expect("first");
        let name = store.files.borrow().keys().next().expect("a stamp").clone();
        store
            .files
            .borrow_mut()
            .insert(name.clone(), ("not a stamp".to_string(), NOW));

        let second = run_at(
            &stamps(&probe, &store, NOW + 5, false),
            Check::Test,
            &files,
            false,
            false,
        );
        assert_eq!(
            second.cargo,
            [XTASK_TEST],
            "garbage under the right name must not skip"
        );
        assert!(
            second
                .out
                .contains(&format!("ignoring the pass stamp {name}")),
            "{}",
            second.out
        );
        let third = run_at(
            &stamps(&probe, &store, NOW + 10, false),
            Check::Test,
            &files,
            false,
            false,
        );
        assert!(
            third.cargo.is_empty(),
            "the replacement is a real stamp: {}",
            third.out
        );
    }

    #[test]
    fn a_skipped_selection_neither_looks_for_nor_writes_a_stamp() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let run = run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["CHANGELOG.md"],
            false,
            false,
        );
        assert!(run.cargo.is_empty());
        assert!(
            run.out.contains("skipped, no Rust crate is affected"),
            "{}",
            run.out
        );
        assert_eq!(*store.writes.borrow(), 0);
    }

    #[test]
    fn a_head_that_moved_during_the_step_is_not_recorded_and_the_note_names_head() {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let stamps = stamps(&probe, &store, NOW, false);
        let mut out = Vec::new();
        execute(
            Check::Test,
            Request::default(),
            &stamps,
            || Ok(graph()),
            |_, _| Ok(vec!["xtask/src/main.rs".to_string()]),
            |_| {
                // HEAD is a different commit by the time the step has finished.
                *probe.tree.borrow_mut() = Ok("tree-2".to_string());
                Ok(())
            },
            &mut out,
        )
        .expect("the step itself passed");
        let out = String::from_utf8(out).expect("utf-8");
        assert!(
            out.contains("pass not recorded: HEAD moved while the step ran"),
            "{out}"
        );
        assert!(
            store.files.borrow().is_empty(),
            "nothing is written for a tree that was not checked"
        );
        assert_eq!(*store.writes.borrow(), 0);
    }

    /// A stamp for `xtask`'s test step on `tree-1`, written by a normal run.
    fn with_a_recorded_pass() -> (FakeProbe, MemStore) {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        run_at(
            &stamps(&probe, &store, NOW, false),
            Check::Test,
            &["xtask/src/main.rs"],
            false,
            false,
        )
        .result
        .expect("the first run");
        assert_eq!(store.files.borrow().len(), 1);
        (probe, store)
    }

    // ---- run_in: the flag and the variable reach the stamp logic ----

    /// Drive the real wiring below `main`: `run_in` builds the stamps from the
    /// flag and the environment, here over a fake store. With `recorded_pass`
    /// a first plain run records a pass for the tree; the run that follows is
    /// the one measured, and the result says whether it ran cargo.
    fn measured_run_ran_cargo(rerun: bool, rerun_var: Option<&str>, recorded_pass: bool) -> bool {
        let (probe, store) = (FakeProbe::clean("tree-1"), MemStore::default());
        let env = |name: &str| {
            if name == prepush_stamp::RERUN_VAR {
                rerun_var.map(String::from)
            } else {
                None
            }
        };
        let ran = std::cell::Cell::new(false);
        let run_once = |now: u64, rerun: bool| {
            ran.set(false);
            run_in(
                Check::Test,
                Flags {
                    base: None,
                    dry_run: false,
                    rerun,
                },
                &World {
                    probe: &probe,
                    store: &store,
                    now,
                    env: &env,
                },
                || Ok(graph()),
                |_, _| Ok(vec!["xtask/src/main.rs".to_string()]),
                |_| {
                    ran.set(true);
                    Ok(())
                },
                &mut Vec::new(),
            )
            .expect("the step");
            ran.get()
        };
        if recorded_pass {
            // Nothing is on record yet, so this run runs cargo and records a pass.
            assert!(run_once(NOW, false), "the first run runs the step");
            assert_eq!(store.files.borrow().len(), 1, "and records a pass");
        }
        run_once(NOW + 60, rerun)
    }

    #[test]
    fn without_the_flag_or_the_variable_a_recorded_pass_skips_the_step() {
        assert!(!measured_run_ran_cargo(false, None, true));
        assert!(
            !measured_run_ran_cargo(false, Some("0"), true),
            "0 means off"
        );
        assert!(
            !measured_run_ran_cargo(false, Some(""), true),
            "empty means off"
        );
    }

    #[test]
    fn the_flag_reaches_the_stamp_logic_and_reruns_despite_a_recorded_pass() {
        assert!(measured_run_ran_cargo(true, None, true));
    }

    #[test]
    fn the_variable_reaches_the_stamp_logic_and_reruns_despite_a_recorded_pass() {
        assert!(measured_run_ran_cargo(false, Some("1"), true));
    }

    #[test]
    fn with_no_recorded_pass_the_step_runs_either_way() {
        assert!(measured_run_ran_cargo(false, None, false));
        assert!(measured_run_ran_cargo(true, None, false));
    }

    #[test]
    fn the_rerun_flag_forces_a_run_despite_a_matching_stamp() {
        let (probe, store) = with_a_recorded_pass();
        let forced = Stamps::new(&probe, &store, NOW + 5, true, |_| None);
        let run = run_at(&forced, Check::Test, &["xtask/src/main.rs"], false, false);
        assert_eq!(run.cargo, [XTASK_TEST], "{}", run.out);
        assert!(
            run.out.contains("recorded passes are ignored"),
            "{}",
            run.out
        );
        assert!(!run.out.contains("skipped"), "{}", run.out);
    }

    #[test]
    fn the_rerun_variable_forces_a_run_despite_a_matching_stamp() {
        let (probe, store) = with_a_recorded_pass();
        let lookup = |name: &str| (name == prepush_stamp::RERUN_VAR).then(|| "1".to_string());
        let forced = Stamps::new(&probe, &store, NOW + 5, false, lookup);
        let run = run_at(&forced, Check::Test, &["xtask/src/main.rs"], false, false);
        assert_eq!(run.cargo, [XTASK_TEST], "{}", run.out);
        assert!(
            run.out.contains("recorded passes are ignored"),
            "{}",
            run.out
        );
    }

    #[test]
    fn without_the_flag_or_with_the_variable_off_the_same_stamp_still_skips() {
        let (probe, store) = with_a_recorded_pass();
        let off = |name: &str| (name == prepush_stamp::RERUN_VAR).then(|| "0".to_string());
        for stamps in [
            Stamps::new(&probe, &store, NOW + 5, false, |_| None),
            Stamps::new(&probe, &store, NOW + 5, false, off),
        ] {
            let run = run_at(&stamps, Check::Test, &["xtask/src/main.rs"], false, false);
            assert!(run.cargo.is_empty(), "{}", run.out);
            assert!(
                run.out.contains("skipped, this tree already passed"),
                "{}",
                run.out
            );
        }
    }
}
