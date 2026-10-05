// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the recursive-deletion paths that are *not* the runtime
//! install-root guard: `rocm uninstall` and `rocm storage remove-downloads`,
//! both of which end in [`crate::remove_path`].
//!
//! The oracle is the real filesystem, not a restatement of the code's own path
//! logic. Each case builds a sandbox tree (planted symlinks, sibling folders
//! whose names are prefixes of the target, sentinels outside the target),
//! snapshots it with `symlink_metadata` keyed by `(dev, ino)`, runs the real
//! plan + removal code, and compares the snapshot afterwards. "Which entries
//! did a planned path name" is answered by the kernel (`symlink_metadata` on
//! the planned spelling), never by lexical path comparison in the test.
//!
//! Unix-only: the generators plant symlinks with `std::os::unix::fs::symlink`.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::{MetadataExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use rocm_core::AppPaths;

use crate::{UninstallOptions, build_uninstall_plan, remove_path};

// ---------------------------------------------------------------------------
// Sandbox + snapshot helpers (the oracle)
// ---------------------------------------------------------------------------

static SANDBOX_SEQ: AtomicU64 = AtomicU64::new(0);

fn fresh_sandbox(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "rocm-cli-deletion-props-{label}-{}-{}-{}",
        std::process::id(),
        rocm_core::unix_time_millis(),
        SANDBOX_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).expect("create sandbox");
    // Canonical so every later comparison is against the real location.
    root.canonicalize().expect("canonicalize sandbox")
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    Dir,
    File(Vec<u8>),
    Link(PathBuf),
}

#[derive(Debug, Clone)]
struct Entry {
    /// Path as reached by walking real directories from the sandbox root.
    path: PathBuf,
    kind: Kind,
}

/// `(dev, ino)` -> entry, walking with `symlink_metadata` so no link is followed.
fn snapshot(root: &Path) -> BTreeMap<(u64, u64), Entry> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        let kind = if meta.file_type().is_symlink() {
            Kind::Link(std::fs::read_link(&path).unwrap_or_default())
        } else if meta.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    stack.push(entry.path());
                }
            }
            Kind::Dir
        } else {
            Kind::File(std::fs::read(&path).unwrap_or_default())
        };
        out.insert((meta.dev(), meta.ino()), Entry { path, kind });
    }
    out
}

/// Identity of whatever `path` names, resolved by the kernel exactly as
/// `remove_path` would see it (final component not followed).
fn identity_of(path: &Path) -> Option<(u64, u64)> {
    std::fs::symlink_metadata(path)
        .ok()
        .map(|meta| (meta.dev(), meta.ino()))
}

/// Entries (by identity) in the snapshot that sit at or below `top`, using the
/// snapshot's own real-walk paths so the containment test is over real
/// locations, not over the planned spelling.
fn subtree_ids(snap: &BTreeMap<(u64, u64), Entry>, top: (u64, u64)) -> BTreeSet<(u64, u64)> {
    let Some(top_entry) = snap.get(&top) else {
        return BTreeSet::new();
    };
    let mut ids = BTreeSet::from([top]);
    if top_entry.kind == Kind::Dir {
        for (id, entry) in snap {
            if entry.path.starts_with(&top_entry.path) {
                ids.insert(*id);
            }
        }
    }
    ids
}

/// Refuse to hand the code under test anything that would resolve outside the
/// sandbox. A harness bug must never turn into a real deletion on the host.
/// (This is a seatbelt for the test, not part of the oracle.)
fn assert_inside_sandbox(sandbox: &Path, path: &Path) {
    let parent = path
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .unwrap_or_else(|| PathBuf::from("/"));
    assert!(
        parent.starts_with(sandbox) || parent == sandbox,
        "HARNESS SAFETY: planned path {} resolves outside sandbox {}",
        path.display(),
        sandbox.display()
    );
}

fn cleanup(sandbox: &Path) {
    let _ = std::fs::remove_dir_all(sandbox);
}

// ---------------------------------------------------------------------------
// Generator reach bookkeeping
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Reach {
    counts: BTreeMap<&'static str, u64>,
    cases: u64,
}

impl Reach {
    fn hit(&mut self, label: &'static str) {
        *self.counts.entry(label).or_default() += 1;
    }
}

/// 256 by default; `ROCM_DELETION_PROP_CASES` raises it for a deeper local run.
fn case_count() -> u32 {
    std::env::var("ROCM_DELETION_PROP_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256)
}

fn report_reach(name: &str, reach: &Mutex<Reach>) {
    let reach = reach.lock().expect("reach lock");
    eprintln!("[{name}] generator reach over {} case(s):", reach.cases);
    for (label, count) in &reach.counts {
        eprintln!("  {label:<44} {count}");
    }
}

// ---------------------------------------------------------------------------
// remove-downloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Node {
    File,
    Dir(Vec<(String, Self)>),
    /// Link to a directory outside the cache that holds a sentinel.
    LinkOutsideDir,
    /// Link to a sentinel file outside the cache.
    LinkOutsideFile,
    /// Link to a path that does not exist.
    LinkDangling,
    /// Relative link climbing back out to the sandbox's data dir.
    LinkUpToData,
}

#[derive(Debug, Clone)]
enum CacheRoot {
    Real(Vec<(String, Node)>),
    LinkToOutside,
    Missing,
    PlainFile,
}

fn node_name() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("a".to_owned()),
        Just("b".to_owned()),
        Just("therock".to_owned()),
        Just("tools".to_owned()),
        Just("x.tar.gz".to_owned()),
        Just(".hidden".to_owned()),
        Just("data".to_owned()),
    ]
}

fn node(dangling: bool) -> impl Strategy<Value = Node> {
    // Built as a list rather than a zero weight: proptest shrinks a union
    // towards earlier arms regardless of weight, so a weight-0 arm still shows
    // up in shrunk counterexamples.
    let mut arms: Vec<(u32, BoxedStrategy<Node>)> = vec![
        (4, Just(Node::File).boxed()),
        (1, Just(Node::LinkOutsideDir).boxed()),
        (1, Just(Node::LinkOutsideFile).boxed()),
        (1, Just(Node::LinkUpToData).boxed()),
    ];
    if dangling {
        arms.push((1, Just(Node::LinkDangling).boxed()));
    }
    let leaf = proptest::strategy::Union::new_weighted(arms);
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop::collection::vec((node_name(), inner), 0..4).prop_map(Node::Dir)
    })
}

fn cache_root(dangling: bool) -> impl Strategy<Value = CacheRoot> {
    prop_oneof![
        6 => prop::collection::vec((node_name(), node(dangling)), 0..5).prop_map(CacheRoot::Real),
        1 => Just(CacheRoot::LinkToOutside),
        1 => Just(CacheRoot::Missing),
        1 => Just(CacheRoot::PlainFile),
    ]
}

fn build_node(at: &Path, node: &Node, sandbox: &Path, reach: &mut Reach) {
    match node {
        Node::File => {
            reach.hit("node: file");
            let _ = std::fs::write(at, b"archive-bytes");
        }
        Node::Dir(children) => {
            reach.hit("node: dir");
            if std::fs::create_dir_all(at).is_ok() {
                for (name, child) in children {
                    let child_path = at.join(name);
                    if std::fs::symlink_metadata(&child_path).is_ok() {
                        continue; // duplicate name; first one wins
                    }
                    build_node(&child_path, child, sandbox, reach);
                }
            }
        }
        Node::LinkOutsideDir => {
            reach.hit("node: link -> outside dir");
            let _ = symlink(sandbox.join("outside").join("dir"), at);
        }
        Node::LinkOutsideFile => {
            reach.hit("node: link -> outside file");
            let _ = symlink(sandbox.join("outside").join("file.txt"), at);
        }
        Node::LinkDangling => {
            reach.hit("node: dangling link");
            let _ = symlink(sandbox.join("outside").join("does-not-exist"), at);
        }
        Node::LinkUpToData => {
            reach.hit("node: relative link -> ../../data");
            // Climb from the link's own folder back to the sandbox root, then
            // into `data` — a live link, not a dangling one.
            let depth = at
                .parent()
                .and_then(|parent| parent.strip_prefix(sandbox).ok())
                .map_or(0, |rel| rel.components().count());
            let _ = symlink(format!("{}data", "../".repeat(depth)), at);
        }
    }
}

fn build_cache_root(at: &Path, root: &CacheRoot, sandbox: &Path, reach: &mut Reach) {
    match root {
        CacheRoot::Real(children) => {
            reach.hit("root: real dir");
            std::fs::create_dir_all(at).expect("create cache root");
            for (name, child) in children {
                let child_path = at.join(name);
                if std::fs::symlink_metadata(&child_path).is_ok() {
                    continue;
                }
                build_node(&child_path, child, sandbox, reach);
            }
        }
        CacheRoot::LinkToOutside => {
            reach.hit("root: symlink -> outside dir");
            symlink(sandbox.join("outside").join("dir"), at).expect("plant root link");
        }
        CacheRoot::Missing => reach.hit("root: missing"),
        CacheRoot::PlainFile => {
            reach.hit("root: plain file");
            std::fs::write(at, b"not a dir").expect("plant root file");
        }
    }
}

/// Fixed scenery shared by every downloads case: sentinels outside the two
/// cache roots, including siblings whose names are prefixes of the roots.
fn plant_download_scenery(sandbox: &Path) -> AppPaths {
    let cache = sandbox.join("cache");
    let data = sandbox.join("data");
    std::fs::create_dir_all(sandbox.join("outside").join("dir")).expect("outside dir");
    std::fs::write(
        sandbox.join("outside").join("dir").join("sentinel.txt"),
        b"outside dir sentinel",
    )
    .expect("outside dir sentinel");
    std::fs::write(sandbox.join("outside").join("file.txt"), b"outside file")
        .expect("outside file");
    std::fs::create_dir_all(data.join("models")).expect("data models");
    std::fs::write(data.join("models").join("model.bin"), b"model weights")
        .expect("model sentinel");
    std::fs::create_dir_all(cache.join("therock-old")).expect("prefix sibling");
    std::fs::write(cache.join("therock-old").join("keep.tar.gz"), b"keep")
        .expect("prefix sibling sentinel");
    std::fs::create_dir_all(cache.join("toolsx")).expect("prefix sibling 2");
    std::fs::write(cache.join("toolsx").join("keep"), b"keep").expect("sentinel");
    std::fs::write(cache.join("other.bin"), b"keep").expect("cache-level sentinel");
    AppPaths {
        config_dir: sandbox.join("config"),
        data_dir: data,
        cache_dir: cache,
    }
}

/// Paths the dry-run text says it would remove (`  - <kind>: <path>` lines in
/// the would-be-removed block). Parsed from the rendered output the user reads.
fn listed_in_render(rendered: &str) -> BTreeSet<PathBuf> {
    let mut listed = BTreeSet::new();
    let mut in_block = false;
    for line in rendered.lines() {
        if line.contains("would be removed") {
            in_block = true;
            continue;
        }
        if in_block {
            let Some(rest) = line.strip_prefix("  - ") else {
                in_block = false;
                continue;
            };
            if let Some((_, path)) = rest.split_once(": ") {
                listed.insert(PathBuf::from(path));
            }
        }
    }
    listed
}

fn downloads_case(
    therock: &CacheRoot,
    tools: &CacheRoot,
    reach: &Mutex<Reach>,
) -> Result<(), TestCaseError> {
    let sandbox = fresh_sandbox("downloads");
    let result = (|| {
        let paths = plant_download_scenery(&sandbox);
        {
            let mut reach = reach.lock().expect("reach lock");
            reach.cases += 1;
            build_cache_root(
                &paths.cache_dir.join("therock"),
                therock,
                &sandbox,
                &mut reach,
            );
            build_cache_root(&paths.cache_dir.join("tools"), tools, &sandbox, &mut reach);
        }

        let before = snapshot(&sandbox);
        let plan = crate::storage::build_downloads_plan(&paths);
        let rendered = crate::storage::render_downloads_plan(&plan, true);
        let planned: BTreeSet<PathBuf> = plan.actions.iter().map(|e| e.path.clone()).collect();
        {
            let mut reach = reach.lock().expect("reach lock");
            if planned.is_empty() {
                reach.hit("plan: empty");
            } else {
                reach.hit("plan: non-empty");
            }
            if plan
                .actions
                .iter()
                .any(|e| std::fs::metadata(&e.path).is_err())
            {
                reach.hit("plan: contains a dangling link");
            }
        }

        // Dry-run honesty: the review lists exactly the plan.
        prop_assert_eq!(
            &listed_in_render(&rendered),
            &planned,
            "dry-run text and plan disagree\n{}",
            rendered
        );

        // The identities the plan names, resolved by the kernel *before* removal.
        let mut expected_gone = BTreeSet::new();
        for path in &planned {
            assert_inside_sandbox(&sandbox, path);
            if let Some(id) = identity_of(path) {
                expected_gone.extend(subtree_ids(&before, id));
            }
        }

        // Run the real removal loop exactly as `storage()` does.
        for entry in &plan.actions {
            remove_path(&entry.path).map_err(|error| {
                TestCaseError::fail(format!(
                    "remove_path({}) failed: {error:#}",
                    entry.path.display()
                ))
            })?;
        }

        let after = snapshot(&sandbox);
        let actually_gone: BTreeSet<(u64, u64)> = before
            .keys()
            .filter(|id| !after.contains_key(id))
            .copied()
            .collect();

        // 1. Exactly the planned entries are gone — no more (data loss), no
        //    fewer (the command reports space it did not free).
        let describe = |ids: &BTreeSet<(u64, u64)>| -> Vec<String> {
            ids.iter()
                .filter_map(|id| before.get(id))
                .map(|e| format!("{} {:?}", e.path.display(), e.kind))
                .collect()
        };
        let extra: BTreeSet<_> = actually_gone.difference(&expected_gone).copied().collect();
        let missed: BTreeSet<_> = expected_gone.difference(&actually_gone).copied().collect();
        prop_assert!(
            extra.is_empty(),
            "deleted entries the plan never named: {:?}",
            describe(&extra)
        );
        prop_assert!(
            missed.is_empty(),
            "plan named entries that were NOT deleted (reported as removed anyway): {:?}",
            describe(&missed)
        );

        // 2. Nothing that survived changed content (a link-followed overwrite
        //    or truncation would show up here).
        for (id, entry) in &after {
            if let Some(old) = before.get(id) {
                prop_assert_eq!(
                    &old.kind,
                    &entry.kind,
                    "entry changed in place: {}",
                    entry.path.display()
                );
            }
        }

        // 3. Sentinels outside the two cache roots survive.
        for sentinel in [
            sandbox.join("outside/dir/sentinel.txt"),
            sandbox.join("outside/file.txt"),
            sandbox.join("data/models/model.bin"),
            sandbox.join("cache/therock-old/keep.tar.gz"),
            sandbox.join("cache/toolsx/keep"),
            sandbox.join("cache/other.bin"),
        ] {
            prop_assert!(sentinel.is_file(), "sentinel lost: {}", sentinel.display());
        }
        Ok(())
    })();
    cleanup(&sandbox);
    result
}

fn run_downloads_property(cases: u32, dangling: bool) -> Result<(), String> {
    let reach = Mutex::new(Reach::default());
    let mut runner = TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    });
    let outcome = runner.run(
        &(cache_root(dangling), cache_root(dangling)),
        |(therock, tools)| downloads_case(&therock, &tools, &reach),
    );
    report_reach("remove-downloads", &reach);
    outcome.map_err(|error| format!("{error}"))
}

/// Whatever the tree, remove-downloads deletes exactly the entries its review
/// listed, nothing outside its two cache roots, and changes no surviving entry.
///
/// Dangling links are left out of this generator because they hit the known
/// shape pinned by the ignored test below; everything else must hold.
#[test]
fn remove_downloads_deletes_exactly_its_plan_and_nothing_outside() {
    if let Err(error) = run_downloads_property(case_count(), false) {
        panic!("{error}");
    }
}

/// Finding: a dangling symlink in the download cache is listed in the review
/// and counted in "N downloaded file(s) removed", but `remove_path` returns
/// early on `!path.exists()` (which follows the link), so it is never removed.
#[test]
#[ignore = "finding: remove_path skips dangling symlinks the plan lists"]
fn remove_downloads_removes_the_dangling_links_it_lists() {
    if let Err(error) = run_downloads_property(case_count(), true) {
        panic!("{error}");
    }
}

// ---------------------------------------------------------------------------
// rocm uninstall
// ---------------------------------------------------------------------------

/// Where one of the three AppPaths dirs points, and how it is spelled.
#[derive(Debug, Clone, Copy)]
enum Target {
    /// The dir's own real location (`home/.rocm/<name>`).
    Own,
    /// `home/.rocm` — the parent of all three.
    Parent,
    /// The sandbox's `$HOME`.
    Home,
    /// `home/.rocm/<name>base` — a sibling whose name has the real one as prefix.
    PrefixSibling,
    /// A symlink planted in home that points at the dir's real location.
    LinkToOwn,
    /// A symlink planted in home that points at home itself.
    LinkToHome,
}

#[derive(Debug, Clone, Copy)]
enum Spelling {
    Plain,
    TrailingSlash,
    DoubleSlash,
    DotSegment,
    DotDotDetour,
}

fn target() -> impl Strategy<Value = Target> {
    prop_oneof![
        6 => Just(Target::Own),
        1 => Just(Target::Parent),
        1 => Just(Target::Home),
        1 => Just(Target::PrefixSibling),
        1 => Just(Target::LinkToOwn),
        1 => Just(Target::LinkToHome),
    ]
}

fn spelling() -> impl Strategy<Value = Spelling> {
    prop_oneof![
        Just(Spelling::Plain),
        Just(Spelling::TrailingSlash),
        Just(Spelling::DoubleSlash),
        Just(Spelling::DotSegment),
        Just(Spelling::DotDotDetour),
    ]
}

fn respell(path: &Path, spelling: Spelling) -> PathBuf {
    let text = path.display().to_string();
    let (head, leaf) = text.rsplit_once('/').expect("absolute sandbox path");
    PathBuf::from(match spelling {
        Spelling::Plain => text.clone(),
        Spelling::TrailingSlash => format!("{text}/"),
        Spelling::DoubleSlash => format!("{head}//{leaf}"),
        Spelling::DotSegment => format!("{head}/./{leaf}"),
        Spelling::DotDotDetour => format!("{head}/{leaf}/../{leaf}"),
    })
}

fn resolve_target(home: &Path, name: &str, target: Target, spelling: Spelling) -> PathBuf {
    let rocm = home.join(".rocm");
    let raw = match target {
        Target::Own => rocm.join(name),
        Target::Parent => rocm,
        Target::Home => home.to_path_buf(),
        Target::PrefixSibling => rocm.join(format!("{name}base")),
        Target::LinkToOwn => {
            let link = home.join(format!("link-{name}"));
            let _ = symlink(rocm.join(name), &link);
            link
        }
        Target::LinkToHome => {
            let link = home.join(format!("link-home-{name}"));
            let _ = symlink(home, &link);
            link
        }
    };
    respell(&raw, spelling)
}

/// Is `target` one that plainly names somewhere ROCm CLI did not create?
const fn names_foreign_dir(target: Target) -> bool {
    matches!(target, Target::Home | Target::LinkToHome)
}

/// The trailing-slash-on-a-link shape pinned by
/// `remove_path_treats_a_trailing_slash_link_like_the_link`.
const fn is_trailing_slash_link(target: Target, spelling: Spelling) -> bool {
    matches!(target, Target::LinkToOwn | Target::LinkToHome)
        && matches!(spelling, Spelling::TrailingSlash)
}

fn uninstall_case(
    choices: [(Target, Spelling); 3],
    reach: &Mutex<Reach>,
    require_safety: bool,
    strict_links: bool,
) -> Result<(), TestCaseError> {
    if !require_safety
        && choices
            .iter()
            .any(|&(target, spelling)| is_trailing_slash_link(target, spelling))
    {
        return Err(TestCaseError::reject("known trailing-slash link shape"));
    }
    let sandbox = fresh_sandbox("uninstall");
    let result = (|| {
        let home = sandbox.join("home");
        let rocm = home.join(".rocm");
        for name in [
            "config",
            "data",
            "cache",
            "configbase",
            "database",
            "cachebase",
        ] {
            std::fs::create_dir_all(rocm.join(name)).expect("rocm dirs");
            std::fs::write(rocm.join(name).join("payload"), name.as_bytes()).expect("payload");
        }
        std::fs::create_dir_all(home.join("Documents")).expect("documents");
        std::fs::write(home.join("Documents/thesis.txt"), b"irreplaceable").expect("thesis");
        std::fs::create_dir_all(sandbox.join("outside")).expect("outside");
        std::fs::write(sandbox.join("outside/sentinel"), b"outside").expect("outside sentinel");

        let [(ct, cs), (dt, ds), (kt, ks)] = choices;
        let paths = AppPaths {
            config_dir: resolve_target(&home, "config", ct, cs),
            data_dir: resolve_target(&home, "data", dt, ds),
            cache_dir: resolve_target(&home, "cache", kt, ks),
        };
        {
            let mut reach = reach.lock().expect("reach lock");
            reach.cases += 1;
            for (t, s) in [(ct, cs), (dt, ds), (kt, ks)] {
                reach.hit(match t {
                    Target::Own => "target: own dir",
                    Target::Parent => "target: parent (.rocm)",
                    Target::Home => "target: $HOME",
                    Target::PrefixSibling => "target: prefix sibling",
                    Target::LinkToOwn => "target: symlink -> own dir",
                    Target::LinkToHome => "target: symlink -> $HOME",
                });
                reach.hit(match s {
                    Spelling::Plain => "spelling: plain",
                    Spelling::TrailingSlash => "spelling: trailing /",
                    Spelling::DoubleSlash => "spelling: //",
                    Spelling::DotSegment => "spelling: /./",
                    Spelling::DotDotDetour => "spelling: x/../x",
                });
            }
            if [ct, dt, kt].into_iter().any(names_foreign_dir) {
                reach.hit("case: some dir names a foreign folder");
            }
        }

        let options = UninstallOptions {
            yes: true,
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        let plan = build_uninstall_plan(&paths, &options)
            .map_err(|error| TestCaseError::fail(format!("plan failed: {error:#}")))?;
        let rendered = crate::render_uninstall_plan(
            &plan,
            &UninstallOptions {
                dry_run: true,
                ..options
            },
        );
        let planned: BTreeSet<PathBuf> = plan.actions.iter().map(|e| e.path.clone()).collect();
        prop_assert_eq!(&listed_in_render(&rendered), &planned, "{}", rendered);

        let before = snapshot(&sandbox);
        let mut expected_gone = BTreeSet::new();
        for path in &planned {
            assert_inside_sandbox(&sandbox, path);
            if let Some(id) = identity_of(path) {
                expected_gone.extend(subtree_ids(&before, id));
            }
        }
        for entry in &plan.actions {
            remove_path(&entry.path).map_err(|error| {
                TestCaseError::fail(format!(
                    "remove_path({}) failed: {error:#}",
                    entry.path.display()
                ))
            })?;
        }
        let after = snapshot(&sandbox);
        let actually_gone: BTreeSet<(u64, u64)> = before
            .keys()
            .filter(|id| !after.contains_key(id))
            .copied()
            .collect();
        let describe = |ids: &BTreeSet<(u64, u64)>| -> Vec<String> {
            ids.iter()
                .filter_map(|id| before.get(id))
                .map(|e| e.path.display().to_string())
                .collect()
        };
        let extra: BTreeSet<_> = actually_gone.difference(&expected_gone).copied().collect();
        let mut missed: BTreeSet<_> = expected_gone.difference(&actually_gone).copied().collect();
        if !strict_links {
            // Known shape, pinned by `uninstall_removes_a_link_whose_target_an_earlier_step_removed`:
            // a planned link whose target an earlier step already deleted is
            // skipped by `remove_path`'s `exists()` check.
            missed.retain(|id| {
                before.get(id).is_none_or(|entry| {
                    !matches!(entry.kind, Kind::Link(_)) || std::fs::metadata(&entry.path).is_ok()
                })
            });
        }
        prop_assert!(
            extra.is_empty(),
            "deleted beyond the plan: {:?}",
            describe(&extra)
        );
        prop_assert!(
            missed.is_empty(),
            "planned but survived: {:?}",
            describe(&missed)
        );

        if require_safety {
            for sentinel in [
                home.join("Documents/thesis.txt"),
                sandbox.join("outside/sentinel"),
            ] {
                prop_assert!(
                    sentinel.is_file(),
                    "uninstall deleted a file ROCm CLI never created: {} (paths: {:?})",
                    sentinel.display(),
                    paths
                );
            }
        }
        Ok(())
    })();
    cleanup(&sandbox);
    result
}

fn run_uninstall_property(
    cases: u32,
    require_safety: bool,
    strict_links: bool,
) -> Result<(), String> {
    let reach = Mutex::new(Reach::default());
    let mut runner = TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    });
    let strategy = [
        (target(), spelling()),
        (target(), spelling()),
        (target(), spelling()),
    ];
    let outcome = runner.run(&strategy, |choices| {
        uninstall_case(choices, &reach, require_safety, strict_links)
    });
    report_reach(
        if require_safety {
            "uninstall (safety)"
        } else {
            "uninstall (plan == removal)"
        },
        &reach,
    );
    outcome.map_err(|error| format!("{error}"))
}

/// Honesty: whatever the three dirs point at and however they are spelled,
/// `rocm uninstall` removes exactly what its review listed.
#[test]
fn uninstall_removes_exactly_what_the_review_lists() {
    if let Err(error) = run_uninstall_property(case_count(), false, false) {
        panic!("{error}");
    }
}

/// Safety: `rocm uninstall` must never delete a folder ROCm CLI did not create.
/// It applies no guard at all to the three AppPaths roots, so a data/cache/
/// config dir that names `$HOME` (directly, or through a link spelled with a
/// trailing `/`) is deleted wholesale. Shrinks to `cache_dir = $HOME`.
#[test]
#[ignore = "finding: rocm uninstall applies no guard to config/data/cache roots"]
fn uninstall_never_deletes_a_folder_rocm_cli_did_not_create() {
    if let Err(error) = run_uninstall_property(case_count(), true, false) {
        panic!("{error}");
    }
}

/// Finding: a planned link whose target an earlier plan step already deleted
/// (here: `cache_dir` is a link into `data_dir`) dangles by the time its turn
/// comes, `remove_path` returns early on `!path.exists()`, and uninstall still
/// prints "removed cache ...". Same root cause as the dangling-download case.
#[test]
#[ignore = "finding: remove_path skips a planned link once it dangles"]
fn uninstall_removes_a_link_whose_target_an_earlier_step_removed() {
    let sandbox = fresh_sandbox("uninstall-link-order");
    let home = sandbox.join("home");
    let data = home.join(".rocm");
    std::fs::create_dir_all(data.join("cache")).expect("data");
    let link = home.join("rocm-cache");
    symlink(data.join("cache"), &link).expect("link");
    let paths = AppPaths {
        config_dir: home.join("no-config"),
        data_dir: data,
        cache_dir: link.clone(),
    };
    let options = UninstallOptions {
        yes: true,
        keep_binaries: true,
        ..UninstallOptions::default()
    };
    let plan = build_uninstall_plan(&paths, &options).expect("plan");
    let listed = plan.actions.iter().any(|entry| entry.path == link);
    for entry in &plan.actions {
        remove_path(&entry.path).expect("remove");
    }
    let link_left = std::fs::symlink_metadata(&link).is_ok();
    cleanup(&sandbox);
    assert!(
        listed,
        "the link must be in the plan for this case to mean anything"
    );
    assert!(
        !link_left,
        "planned link {} survived uninstall",
        link.display()
    );
}

#[test]
#[ignore = "finding: uninstall honesty with dangling-after-earlier-step links"]
fn uninstall_removes_exactly_what_the_review_lists_including_links() {
    if let Err(error) = run_uninstall_property(case_count(), false, true) {
        panic!("{error}");
    }
}

// ---------------------------------------------------------------------------
// remove_path: trailing-slash spelling of a symlinked directory
// ---------------------------------------------------------------------------

/// `remove_path` decides "link or directory?" with `symlink_metadata`, but a
/// trailing `/` makes the kernel resolve the final link, so `link/` stats as
/// the *target* directory and `remove_dir_all` walks into it. The same folder
/// spelled `link` has only the link removed. Spelling alone must not change
/// what is deleted.
#[test]
#[ignore = "finding: remove_path follows a symlinked dir spelled with a trailing slash"]
fn remove_path_treats_a_trailing_slash_link_like_the_link() {
    let sandbox = fresh_sandbox("trailing-slash");
    let target = sandbox.join("relocated-cache");
    std::fs::create_dir_all(target.join("sub")).expect("target");
    std::fs::write(target.join("precious.txt"), b"keep").expect("precious");
    std::fs::write(target.join("sub/more.txt"), b"keep").expect("more");
    let link = sandbox.join("cache-link");
    symlink(&target, &link).expect("link");

    let spelled = PathBuf::from(format!("{}/", link.display()));
    let outcome = remove_path(&spelled);

    let survived = target.join("precious.txt").is_file() && target.join("sub/more.txt").is_file();
    let link_left = std::fs::symlink_metadata(&link).is_ok();
    cleanup(&sandbox);
    assert!(
        survived,
        "remove_path({}) reached through the link and deleted the target's contents \
         (result: {outcome:?}, link still present: {link_left})",
        spelled.display()
    );
}
