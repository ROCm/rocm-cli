// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask pre-push-check`: the three cargo steps of the pre-push hook
//! (`clippy`, `clippy-e2e`, `test`), narrowed to the crates the push affects.
//!
//! Each step used to lint or test the whole workspace whatever the push
//! contained, so a one-file change in a leaf crate paid for every crate. The
//! selection is the one CI already uses ([`crate::affected`]): the crates whose
//! files changed plus every crate that transitively depends on them, with a
//! fall back to the whole workspace for anything that cannot be attributed to
//! one crate (the lockfile, the root manifest, the toolchain file, workflow
//! files, a doc an xtask test reads, an unrecognized path). On top of that the
//! hook widens for files that another crate's tests read from outside their own
//! crate directory ([`CROSS_CRATE_READS`]), which the directory-based selection
//! cannot see. CI stays the complete gate; this only makes the local early
//! warning cheaper.
//!
//! **Fail safe.** Whenever the affected set cannot be determined (`git diff`
//! fails because the base is not fetched, `cargo metadata` fails, ...) the step
//! runs over the whole workspace, as it did before, and says why. A step is
//! skipped only when the selection is known and contains nothing for that step
//! to check, and it prints the reason then too.
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
//! [`hook_selection`], [`resolve_selection`]) are pure and unit-tested; running
//! git and cargo is confined to [`run`], and [`execute`] takes both, the stamps
//! and the output as parameters so its own behavior (the range it diffs,
//! `--dry-run`, a failing cargo, a skipped or recorded pass, and what it prints
//! about each) is tested too.

use std::collections::BTreeSet;
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

/// Files a crate's tests (or `include_str!`) read from outside that crate's own
/// directory, with the crates that read them. [`affected::select`] attributes a
/// file to the crate whose directory holds it and treats Markdown and `docs/`
/// as inert, so without these entries a push that edits such a file would never
/// run the reader's tests: the old whole-workspace hook caught that, a narrowed
/// one would not.
///
/// A pattern ending in `/` is a directory prefix, one starting with `*` is a
/// file-name suffix, anything else is an exact path. Files that no crate owns
/// and that are not Markdown (`install.sh`, `scripts/`, `.github/`, ...) need no
/// entry: `select` already sends them to the whole workspace. The guard
/// `every_cross_crate_read_is_covered_by_the_widening_rule` scans the workspace
/// for `include_str!`/`include_bytes!` and manifest-relative reads that leave
/// their crate and fails when one is not covered, so a new cross-crate read
/// cannot be missed silently. The e2e harness is not listed as a reader because
/// the `test` step never runs it.
///
/// - `*.rs`: xtask's contract tests scan every Rust source file in the repo
///   (`ascii_output_contract`, `env_mutation_contract`) and the pinned-key test
///   reads `apps/rocm/src/therock/fetch.rs`.
/// - `README.md`, `docs/`, `skills/`: `rocm`'s `advised_commands` test checks
///   every `rocm ...` command shown there, `rocm` compiles in the assistant
///   skill, and xtask's tests read the pinned release key under `docs/keys` and
///   other docs.
/// - the e2e PowerShell fixture: `rocm-core`'s `fix` tests `include_str!` it
///   although it lives in the e2e crate.
pub(crate) const CROSS_CRATE_READS: &[(&str, &[&str])] = &[
    ("*.rs", &["xtask"]),
    ("README.md", &["rocm", "xtask"]),
    ("docs/", &["rocm", "xtask"]),
    ("skills/", &["rocm", "xtask"]),
    ("tests/e2e-cucumber/fixtures/", &["rocm-core"]),
];

fn read_pattern_matches(pattern: &str, file: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        file.ends_with(suffix)
    } else if pattern.ends_with('/') {
        file.starts_with(pattern)
    } else {
        file == pattern
    }
}

/// The hook's selection: [`affected::select`], widened with the reader crates of
/// every changed file in [`CROSS_CRATE_READS`] and their dependents.
pub fn hook_selection(changed_files: &[String], graph: &Graph) -> Selection {
    hook_selection_with(changed_files, graph, CROSS_CRATE_READS)
}

fn hook_selection_with(
    changed_files: &[String],
    graph: &Graph,
    reads: &[(&str, &[&str])],
) -> Selection {
    let base = affected::select(changed_files, graph);
    if base == Selection::Workspace {
        return base;
    }
    let readers: BTreeSet<&str> = reads
        .iter()
        .filter(|(pattern, _)| {
            changed_files
                .iter()
                .any(|file| read_pattern_matches(pattern, file))
        })
        .flat_map(|(_, readers)| readers.iter().copied())
        .collect();
    if readers.is_empty() {
        return base;
    }
    // A reader that is not a workspace member means the table is stale; run
    // everything rather than guess.
    let members: BTreeSet<&str> = graph
        .crate_dirs
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    if readers.iter().any(|reader| !members.contains(reader)) {
        return Selection::Workspace;
    }
    let mut crates = match base {
        Selection::Packages(packages) => packages,
        Selection::Workspace | Selection::Empty => BTreeSet::new(),
    };
    crates.extend(readers.into_iter().map(String::from));
    Selection::Packages(affected::reverse_closure(&crates, &graph.rev_deps))
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
            let graph = load_graph().context("could not read the workspace dependency graph")?;
            Ok(hook_selection(&files, &graph))
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

/// Entry point for the `pre-push-check` subcommand.
pub fn run(check: Check, base: Option<String>, dry_run: bool, rerun: bool) -> Result<()> {
    let store: Box<dyn prepush_stamp::Store> = match prepush_stamp::FsStore::in_git_common_dir(None)
    {
        Ok(store) => Box::new(store),
        Err(error) => Box::new(prepush_stamp::UnavailableStore(format!("{error:#}"))),
    };
    let probe = prepush_stamp::SystemProbe::default();
    let stamps = Stamps::new(
        &probe,
        store.as_ref(),
        prepush_stamp::now_secs(),
        rerun,
        |name| std::env::var(name).ok(),
    );
    let request = Request {
        base,
        dry_run,
        env: Env::from_lookup(|name| std::env::var(name).ok()),
    };
    execute(
        check,
        request,
        &stamps,
        affected::load_graph,
        affected::changed_files_between,
        spawn_cargo,
        &mut std::io::stdout().lock(),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::prepush_stamp::fakes::{FakeProbe, MemStore};

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

    fn reader_graph() -> Graph {
        let mut rev_deps = BTreeMap::new();
        rev_deps.insert(
            "rocm-core".to_string(),
            vec!["rocm".to_string(), "xtask".to_string()],
        );
        for name in ["rocm", "xtask", "e2e-cucumber"] {
            rev_deps.insert(name.to_string(), vec![]);
        }
        Graph {
            crate_dirs: vec![
                ("rocm-core".into(), "crates/rocm-core".into()),
                ("rocm".into(), "apps/rocm".into()),
                ("xtask".into(), "xtask".into()),
                ("e2e-cucumber".into(), "tests/e2e-cucumber".into()),
            ],
            rev_deps,
        }
    }

    fn selected(paths: &[&str]) -> Selection {
        let changed: Vec<String> = paths.iter().map(|path| (*path).to_string()).collect();
        hook_selection(&changed, &reader_graph())
    }

    fn selected_names(paths: &[&str]) -> BTreeSet<String> {
        match selected(paths) {
            Selection::Packages(names) => names,
            other => panic!("expected packages for {paths:?}, got {other:?}"),
        }
    }

    const FIXTURE: &str = "tests/e2e-cucumber/fixtures/powershell_text_rule_cases.txt";

    /// The directory-based selection attributes the fixture and the step file to
    /// the e2e crate and nothing else, so the `rocm-core` tests that
    /// `include_str!` the fixture never ran and the `test` step skipped.
    #[test]
    fn a_fixture_edit_with_an_e2e_step_still_tests_the_crate_that_includes_it() {
        let changed = [FIXTURE, "tests/e2e-cucumber/tests/e2e/diagnose_steps.rs"];
        assert!(selected_names(&changed).contains("rocm-core"));
        let Plan::Cargo(args) = plan(Check::Test, &selected(&changed)) else {
            panic!("the test step must run, not skip");
        };
        assert!(args.join(" ").contains("-p rocm-core"), "{args:?}");
    }

    #[test]
    fn the_fixture_alone_selects_its_reader() {
        assert!(selected_names(&[FIXTURE]).contains("rocm-core"));
    }

    #[test]
    fn docs_and_skills_select_the_crates_whose_tests_read_them() {
        for path in [
            "README.md",
            "docs/testing.md",
            "skills/rocm-cli-assistant/SKILL.md",
        ] {
            let names = selected_names(&[path]);
            assert!(names.contains("rocm"), "{path}: {names:?}");
            assert!(names.contains("xtask"), "{path}: {names:?}");
            assert!(!names.contains("rocm-core"), "{path}: {names:?}");
        }
    }

    #[test]
    fn any_rust_file_selects_xtask_for_its_repo_wide_source_scans() {
        assert!(selected_names(&["apps/rocm/src/main.rs"]).contains("xtask"));
    }

    #[test]
    fn markdown_nothing_reads_still_selects_nothing() {
        assert_eq!(selected(&["CHANGELOG.md"]), Selection::Empty);
    }

    #[test]
    fn the_whole_workspace_stays_the_whole_workspace() {
        assert_eq!(
            selected(&["Cargo.lock", "apps/rocm/src/main.rs"]),
            Selection::Workspace
        );
    }

    #[test]
    fn a_reader_that_is_not_a_workspace_member_runs_everything() {
        let changed = ["docs/testing.md".to_string()];
        let table: &[(&str, &[&str])] = &[("docs/", &["a-crate-that-was-renamed"])];
        assert_eq!(
            hook_selection_with(&changed, &reader_graph(), table),
            Selection::Workspace
        );
    }

    #[test]
    fn read_patterns_match_by_directory_suffix_or_exact_path() {
        assert!(read_pattern_matches("docs/", "docs/keys/a.pem"));
        assert!(!read_pattern_matches("docs/", "documents/a.md"));
        assert!(read_pattern_matches("*.rs", "xtask/src/main.rs"));
        assert!(!read_pattern_matches("*.rs", "xtask/Cargo.toml"));
        assert!(read_pattern_matches("README.md", "README.md"));
        assert!(!read_pattern_matches("README.md", "docs/README.md"));
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

    // ---- drift guard: every cross-crate read must be covered ----

    use std::path::{Path, PathBuf};

    /// A read that leaves the crate: a repo-relative path, or the repo root
    /// itself when the code derives it and joins the rest elsewhere.
    #[derive(Debug, PartialEq, Eq)]
    enum Read {
        Root,
        Path(String),
    }

    /// The text of the sole string literal in `args`, if that is all it holds.
    /// Plain (`"..."`) and raw (`r"..."`, `r#"..."#`) literals are recognised.
    fn sole_literal(args: &str) -> Option<&str> {
        let args = args.trim();
        if let Some(inner) = args.strip_prefix('"') {
            let end = inner.find('"')?;
            return inner[end + 1..].trim().is_empty().then_some(&inner[..end]);
        }
        let raw = args.strip_prefix('r')?;
        let hashes = raw.len() - raw.trim_start_matches('#').len();
        let body = raw[hashes..].strip_prefix('"')?;
        let closing = format!("\"{}", "#".repeat(hashes));
        let end = body.find(&closing)?;
        body[end + closing.len()..]
            .trim()
            .is_empty()
            .then_some(&body[..end])
    }

    /// Index of the `)` that closes the `(` at `open`, skipping string literals.
    fn matching_paren(text: &str, open: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut depth = 0usize;
        let mut i = open;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        i += if bytes[i] == b'\\' { 2 } else { 1 };
                    }
                }
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Lexically resolve `..` and `.`; `None` when the path leaves the root.
    fn normalize(path: &str) -> Option<String> {
        let mut parts: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop()?;
                }
                other => parts.push(other),
            }
        }
        Some(parts.join("/"))
    }

    /// Repo-relative targets of the `include_str!`/`include_bytes!` literals in
    /// the file `file_rel`.
    fn include_targets(file_rel: &str, text: &str) -> Vec<String> {
        let dir = file_rel.rsplit_once('/').map_or("", |(dir, _)| dir);
        let mut targets = Vec::new();
        for macro_name in ["include_str", "include_bytes"] {
            let needle = format!("{macro_name}!(");
            for (at, _) in text.match_indices(&needle) {
                let open = at + needle.len() - 1;
                let Some(close) = matching_paren(text, open) else {
                    continue;
                };
                if let Some(literal) = sole_literal(&text[open + 1..close]) {
                    targets.extend(normalize(&format!("{dir}/{literal}")));
                }
            }
        }
        targets
    }

    /// The reads that start at `env!("CARGO_MANIFEST_DIR")` and leave the crate
    /// directory (through `.parent()`, `.ancestors()` or a `..` segment). Scratch
    /// directories under `.rocm-work` and `target` are not reads of repo files.
    fn manifest_reads(text: &str) -> Vec<Read> {
        let needle = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let bytes = text.as_bytes();
        let mut reads = Vec::new();
        for (at, _) in text.match_indices(&needle) {
            let mut i = at + needle.len();
            let (mut escapes, mut segments) = (false, Vec::new());
            loop {
                while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b')') {
                    i += 1;
                }
                if i >= bytes.len() || bytes[i] != b'.' {
                    break;
                }
                i += 1;
                let name_start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let name = &text[name_start..i];
                if i >= bytes.len() || bytes[i] != b'(' {
                    break;
                }
                let Some(close) = matching_paren(text, i) else {
                    break;
                };
                let literal = sole_literal(&text[i + 1..close]);
                match (name, literal) {
                    ("parent" | "ancestors", _) => escapes = true,
                    ("join", Some(path)) => {
                        for part in path.split('/') {
                            match part {
                                ".." => escapes = true,
                                "" | "." => {}
                                other => segments.push(other.to_string()),
                            }
                        }
                    }
                    _ => {}
                }
                i = close + 1;
            }
            if !escapes
                || matches!(
                    segments.first().map(String::as_str),
                    Some(".rocm-work" | "target")
                )
            {
                continue;
            }
            reads.push(if segments.is_empty() {
                Read::Root
            } else {
                Read::Path(segments.join("/"))
            });
        }
        reads
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Collects the files under `dir`, skipping build output, vendored trees and
    /// every dot-directory (`.git`, `.rocm-work`, and nested checkouts such as
    /// `.claude/worktrees/...`, whose sources belong to another branch).
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        const SKIPPED: &[&str] = &["target", "node_modules", "third_party"];
        for entry in std::fs::read_dir(dir).expect("read dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name();
                let name = name.to_str().unwrap_or_default();
                if !name.starts_with('.') && !SKIPPED.contains(&name) {
                    walk(&path, files);
                }
            } else {
                files.push(path);
            }
        }
    }

    fn relative(root: &Path, path: &Path) -> String {
        path.strip_prefix(root)
            .expect("under the repo root")
            .to_string_lossy()
            .replace('\\', "/")
    }

    /// The workspace crates, found from their manifests on disk.
    fn crates_on_disk(root: &Path, files: &[PathBuf]) -> Vec<(String, String)> {
        files
            .iter()
            .filter(|path| path.file_name().is_some_and(|name| name == "Cargo.toml"))
            .filter_map(|path| {
                let manifest: toml::Value =
                    toml::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
                let name = manifest.get("package")?.get("name")?.as_str()?.to_string();
                let dir = relative(root, path.parent()?);
                (!dir.is_empty()).then_some((name, dir))
            })
            .collect()
    }

    /// A raw-string include is a cross-crate read like any other; the scanner
    /// must not skip it.
    #[test]
    fn include_targets_reads_plain_and_raw_string_literals() {
        // Built with `format!` so this file never contains a literal embed that
        // `ascii_output_contract` would count as a data file compiled into a binary.
        let call = |name: &str, arg: &str| format!("{name}!({arg})");
        let target = "../../../README.md";
        let file = "crates/reader/src/lib.rs";
        for text in [
            call("include_str", &format!("\"{target}\"")),
            call("include_str", &format!("r\"{target}\"")),
            call("include_str", &format!("r#\"{target}\"#")),
            call("include_bytes", &format!("r##\"{target}\"##")),
        ] {
            assert_eq!(
                include_targets(file, &text),
                vec!["README.md".to_string()],
                "{text}"
            );
        }
        assert!(include_targets(file, &call("include_str", "concat!(\"a\", \"b\")")).is_empty());
    }

    /// A nested checkout (`.claude/worktrees/...`) holds another branch's
    /// sources; scanning it reports reads that are not this tree's.
    #[test]
    fn walk_skips_dot_directories_and_build_output() {
        let tmp = tempfile::tempdir().expect("temp dir");
        for file in [
            "crates/a/src/lib.rs",
            ".claude/worktrees/w/crates/b/src/lib.rs",
            ".git/hooks/x.rs",
            "target/debug/y.rs",
        ] {
            let path = tmp.path().join(file);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, "").expect("write");
        }
        let mut files = Vec::new();
        walk(tmp.path(), &mut files);
        let found: Vec<String> = files.iter().map(|p| relative(tmp.path(), p)).collect();
        assert_eq!(found, vec!["crates/a/src/lib.rs".to_string()]);
    }

    /// Fails when a crate reads a file outside its own directory that the hook's
    /// selection would not send to that crate's tests, or when a crate derives
    /// the repo root without being declared as a reader. Add the path to
    /// [`CROSS_CRATE_READS`] (or declare the crate) when this fails.
    #[test]
    fn every_cross_crate_read_is_covered_by_the_widening_rule() {
        let root = repo_root();
        let mut files = Vec::new();
        walk(&root, &mut files);
        let crate_dirs = crates_on_disk(&root, &files);
        assert!(
            crate_dirs.iter().any(|(name, _)| name == "xtask"),
            "found no crates under {}",
            root.display()
        );
        let graph = Graph {
            rev_deps: crate_dirs
                .iter()
                .map(|(name, _)| (name.clone(), Vec::new()))
                .collect(),
            crate_dirs,
        };

        let mut problems = Vec::new();
        let mut root_readers = BTreeSet::new();
        let mut checked = 0usize;
        let mut check = |reader: &str, source: &str, target: &str| {
            checked += 1;
            // `test` never runs the e2e crate, so nothing selects it for its reads.
            if reader == E2E_CRATE
                || affected::owning_crate(target, &graph.crate_dirs) == Some(reader)
            {
                return;
            }
            let covered = match hook_selection(&[target.to_string()], &graph) {
                Selection::Workspace => true,
                Selection::Packages(names) => names.contains(reader),
                Selection::Empty => false,
            };
            if !covered {
                problems.push(format!("{source} ({reader}) reads {target}"));
            }
        };

        for path in files
            .iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        {
            let source = relative(&root, path);
            let Some(reader) = affected::owning_crate(&source, &graph.crate_dirs) else {
                continue;
            };
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            for target in include_targets(&source, &text) {
                check(reader, &source, &target);
            }
            for read in manifest_reads(&text) {
                match read {
                    Read::Root => {
                        root_readers.insert(reader.to_string());
                    }
                    Read::Path(target) => check(reader, &source, &target),
                }
            }
        }

        // A scanner that finds nothing would pass vacuously. xtask's tests derive
        // the repo root, and crates include files from outside themselves.
        assert!(
            root_readers.contains("xtask"),
            "the manifest scanner found nothing"
        );
        assert!(checked > 0, "the include scanner found nothing");

        let declared: BTreeSet<&str> = CROSS_CRATE_READS
            .iter()
            .flat_map(|(_, readers)| readers.iter().copied())
            .collect();
        for reader in root_readers {
            if reader != E2E_CRATE && !declared.contains(reader.as_str()) {
                problems.push(format!(
                    "crate {reader} derives the repo root but is not a reader in CROSS_CRATE_READS"
                ));
            }
        }
        assert!(
            problems.is_empty(),
            "reads the narrowed hook would not run the reader's tests for:\n{}",
            problems.join("\n")
        );
    }

    #[test]
    fn the_scanner_finds_includes_that_leave_the_crate() {
        let include = |name: &str, path: &str| format!("let a = {name}!(\"{path}\");");
        let text = format!(
            "{}\n{}\n{}",
            include("include_str", "../../../tests/e2e-cucumber/fixtures/a.txt"),
            include("include_bytes", "../data.bin"),
            include("include_str", "../../../../../escapes.txt"),
        );
        assert_eq!(
            include_targets("crates/rocm-core/src/fix.rs", &text),
            [
                "tests/e2e-cucumber/fixtures/a.txt",
                "crates/rocm-core/data.bin"
            ]
        );
    }

    #[test]
    fn the_scanner_follows_a_manifest_relative_chain_out_of_the_crate() {
        let manifest = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let chain = |tail: &str| format!("fn p() -> PathBuf {{ Path::new({manifest}){tail} }}");
        assert_eq!(
            manifest_reads(&chain(
                ".join(\"..\").join(\"..\").join(\"skills\").join(\"d\").join(\"f.md\")"
            )),
            [Read::Path("skills/d/f.md".to_string())]
        );
        assert_eq!(
            manifest_reads(&chain(".join(\"../..\").join(\"docs\")")),
            [Read::Path("docs".to_string())]
        );
        assert_eq!(
            manifest_reads(&chain(".parent().expect(\"root\")")),
            [Read::Root]
        );
        assert_eq!(
            manifest_reads(&chain(".ancestors().nth(2).expect(\"root\")")),
            [Read::Root]
        );
    }

    #[test]
    fn the_scanner_ignores_scratch_dirs_and_reads_inside_the_crate() {
        let manifest = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let chain = |tail: &str| format!("fn p() -> PathBuf {{ PathBuf::from({manifest}){tail} }}");
        assert!(
            manifest_reads(&chain(
                ".join(\"..\").join(\"..\").join(\".rocm-work\").join(\"t\")"
            ))
            .is_empty()
        );
        assert!(manifest_reads(&chain(".join(\"target\").join(\"x\")")).is_empty());
        assert!(manifest_reads(&chain(".join(\"features\")")).is_empty());
    }

    #[test]
    fn the_scanner_ignores_a_join_argument_that_is_not_a_plain_literal() {
        let manifest = format!("env!(\"{}\")", "CARGO_MANIFEST_DIR");
        let text = format!(
            "let p = PathBuf::from({manifest}).parent().unwrap().join(format!(\"x-{{}}\", id));"
        );
        assert_eq!(manifest_reads(&text), [Read::Root]);
    }
}
