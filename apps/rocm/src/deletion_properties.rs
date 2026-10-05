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
use rocm_core::{AppPathSources, AppPaths};

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

fn node() -> impl Strategy<Value = Node> {
    let leaf = prop_oneof![
        4 => Just(Node::File),
        1 => Just(Node::LinkOutsideDir),
        1 => Just(Node::LinkOutsideFile),
        1 => Just(Node::LinkUpToData),
        1 => Just(Node::LinkDangling),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop::collection::vec((node_name(), inner), 0..4).prop_map(Node::Dir)
    })
}

fn cache_root() -> impl Strategy<Value = CacheRoot> {
    prop_oneof![
        6 => prop::collection::vec((node_name(), node()), 0..5).prop_map(CacheRoot::Real),
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

fn run_downloads_property(cases: u32) -> Result<(), String> {
    let reach = Mutex::new(Reach::default());
    let mut runner = TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    });
    let outcome = runner.run(&(cache_root(), cache_root()), |(therock, tools)| {
        downloads_case(&therock, &tools, &reach)
    });
    report_reach("remove-downloads", &reach);
    outcome.map_err(|error| format!("{error}"))
}

/// Whatever the tree, remove-downloads deletes exactly the entries its review
/// listed, nothing outside its two cache roots, and changes no surviving entry.
///
/// "Exactly" includes dangling links: the review lists them and the command
/// counts them as removed, so they must really be gone afterwards.
#[test]
fn remove_downloads_deletes_exactly_its_plan_and_nothing_outside() {
    if let Err(error) = run_downloads_property(case_count()) {
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
    /// A regular file (`home/.rocm/<name>.txt`), which only a spelling with no
    /// trailing `/` or `/.` can name.
    RegularFile,
    /// A symlink planted in home whose target does not exist.
    DanglingLink,
    /// `$HOME` spelled relative to the process's working directory — what
    /// `ROCM_CLI_DATA_DIR=.` run from home amounts to.
    RelativeHome,
    /// `<home>/Documents/..` spelled relative to the working directory — `..`
    /// run from a project folder in home.
    RelativeHomeViaChild,
    /// `$HOME` reached through a symlink to its parent (`/home -> var/home`).
    HomeViaLinkedParent,
}

#[derive(Debug, Clone, Copy)]
enum Spelling {
    Plain,
    TrailingSlash,
    TrailingDot,
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
        1 => Just(Target::RegularFile),
        1 => Just(Target::DanglingLink),
        1 => Just(Target::RelativeHome),
        1 => Just(Target::RelativeHomeViaChild),
        1 => Just(Target::HomeViaLinkedParent),
    ]
}

fn spelling() -> impl Strategy<Value = Spelling> {
    prop_oneof![
        Just(Spelling::Plain),
        Just(Spelling::TrailingSlash),
        Just(Spelling::TrailingDot),
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
        Spelling::TrailingDot => format!("{text}/."),
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
        Target::RegularFile => {
            let file = rocm.join(format!("{name}.txt"));
            let _ = std::fs::write(&file, name.as_bytes());
            file
        }
        Target::DanglingLink => {
            let link = home.join(format!("dangling-{name}"));
            let _ = symlink(home.join("does-not-exist"), &link);
            link
        }
        Target::RelativeHome => relative_from_cwd(home),
        Target::RelativeHomeViaChild => relative_from_cwd(&home.join("Documents")).join(".."),
        Target::HomeViaLinkedParent => {
            let sandbox = home.parent().expect("home has a parent");
            let link = sandbox.join(format!("linked-parent-{name}"));
            let _ = symlink(sandbox, &link);
            link.join(home.file_name().expect("home has a name"))
        }
    };
    respell(&raw, spelling)
}

/// Is `target` one that plainly names somewhere ROCm CLI did not create?
const fn names_foreign_dir(target: Target) -> bool {
    matches!(
        target,
        Target::Home
            | Target::LinkToHome
            | Target::RelativeHome
            | Target::RelativeHomeViaChild
            | Target::HomeViaLinkedParent
    )
}

/// `target` spelled relative to this process's working directory with `..`
/// segments, so a case can use a relative root without changing the
/// (process-global) working directory.
fn relative_from_cwd(target: &Path) -> PathBuf {
    let cwd = std::env::current_dir()
        .and_then(|cwd| cwd.canonicalize())
        .expect("working directory");
    let mut relative = PathBuf::new();
    for _ in cwd.components().skip(1) {
        relative.push("..");
    }
    relative.join(target.strip_prefix("/").expect("absolute target"))
}

fn uninstall_case(
    choices: [(Target, Spelling); 3],
    reach: &Mutex<Reach>,
    require_safety: bool,
) -> Result<(), TestCaseError> {
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
                    Target::RegularFile => "target: regular file",
                    Target::DanglingLink => "target: dangling symlink",
                    Target::RelativeHome => "target: $HOME, relative to cwd",
                    Target::RelativeHomeViaChild => "target: $HOME/Documents/.., relative",
                    Target::HomeViaLinkedParent => "target: $HOME via a symlinked parent",
                });
                reach.hit(match s {
                    Spelling::Plain => "spelling: plain",
                    Spelling::TrailingSlash => "spelling: trailing /",
                    Spelling::TrailingDot => "spelling: trailing /.",
                    Spelling::DoubleSlash => "spelling: //",
                    Spelling::DotSegment => "spelling: /./",
                    Spelling::DotDotDetour => "spelling: x/../x",
                });
            }
            if [ct, dt, kt].into_iter().any(names_foreign_dir) {
                reach.hit("case: some dir names a foreign folder");
            }
        }

        // Shared caches the review must talk about truthfully: one inside the
        // own cache dir, one inside the own data dir, one outside everything.
        let shared_caches = [
            rocm.join("cache").join("uv-shared"),
            rocm.join("data").join("hf-hub"),
            home.join(".cache").join("uv"),
        ];
        for cache in &shared_caches {
            std::fs::create_dir_all(cache).expect("shared cache");
            std::fs::write(cache.join("blob"), b"cached").expect("shared cache blob");
        }
        let candidates: Vec<crate::SharedCache> = shared_caches
            .iter()
            .map(|path| crate::SharedCache {
                path: path.clone(),
                kept: "a shared cache is",
                name: "a shared cache",
                loss: "It is shared.",
                pronoun: "it",
            })
            .collect();

        let options = UninstallOptions {
            yes: true,
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        let plan = crate::build_uninstall_plan_for_home(
            &paths,
            &AppPathSources::default(),
            &options,
            Some(&home),
        )
        .map_err(|error| TestCaseError::fail(format!("plan failed: {error:#}")))?;
        let notes = crate::shared_cache_notes_for(&plan.actions, &candidates);
        let rendered = crate::render_uninstall_plan(
            &plan,
            &UninstallOptions {
                dry_run: true,
                ..options
            },
        );
        let planned: BTreeSet<PathBuf> = plan.actions.iter().map(|e| e.path.clone()).collect();
        prop_assert_eq!(&listed_in_render(&rendered), &planned, "{}", rendered);
        let refused = !plan.refused.is_empty();
        if refused {
            reach.lock().expect("reach lock").hit("plan: refused");
            // The review names every refused root and the flag that clears it.
            for root in &plan.refused {
                prop_assert!(
                    rendered.contains(&root.path.display().to_string())
                        && rendered.contains(&format!("--keep-{}", root.kind)),
                    "refused {} root not named with its remedy:\n{}",
                    root.kind,
                    rendered
                );
            }
        }

        let before = snapshot(&sandbox);
        let mut expected_gone = BTreeSet::new();
        if !refused {
            for path in &planned {
                assert_inside_sandbox(&sandbox, path);
                if let Some(id) = identity_of(path) {
                    expected_gone.extend(subtree_ids(&before, id));
                }
            }
        }
        // The real removal step, which refuses on its own when the plan does.
        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        prop_assert_eq!(
            outcome.is_err(),
            refused,
            "apply_uninstall_plan result {:?} disagrees with the plan's refusal",
            outcome
        );
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
        // A regular file written as `file/` or `file/.` names nothing to the
        // kernel, so it is neither planned nor removed — unless another root
        // that does name it (or a folder above it) is.
        for ((target, spelling), root) in
            choices
                .iter()
                .zip([&paths.config_dir, &paths.data_dir, &paths.cache_dir])
        {
            if matches!(target, Target::RegularFile)
                && matches!(spelling, Spelling::TrailingSlash | Spelling::TrailingDot)
            {
                reach
                    .lock()
                    .expect("reach lock")
                    .hit("case: regular file spelled as a dir");
                let file = PathBuf::from(
                    root.display()
                        .to_string()
                        .trim_end_matches('.')
                        .trim_end_matches('/'),
                );
                // Each kind's file has its own name, so only this root could
                // have planned it.
                prop_assert!(
                    !planned.contains(&file)
                        && rendered.contains(&format!("path not present: {}\n", root.display())),
                    "{} names no directory but was planned:\n{}",
                    root.display(),
                    rendered
                );
                let named_elsewhere = before
                    .iter()
                    .find_map(|(id, entry)| (entry.path == file).then_some(*id))
                    .is_some_and(|id| expected_gone.contains(&id));
                if !named_elsewhere {
                    prop_assert!(
                        file.is_file(),
                        "{} was deleted through the spelling {}",
                        file.display(),
                        root.display()
                    );
                }
            }
        }
        let extra: BTreeSet<_> = actually_gone.difference(&expected_gone).copied().collect();
        let missed: BTreeSet<_> = expected_gone.difference(&actually_gone).copied().collect();
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

        // Every shared cache is mentioned exactly once, and the review warns
        // that it will be deleted exactly when the removal deleted it.
        for cache in &shared_caches {
            let text = cache.display().to_string();
            let mentions: Vec<&String> = notes.iter().filter(|n| n.contains(&text)).collect();
            prop_assert_eq!(mentions.len(), 1, "{} mentioned {:?}", text, mentions);
            let warned = mentions[0].contains("WILL BE DELETED");
            if !refused {
                let deleted = !cache.join("blob").exists();
                prop_assert_eq!(
                    warned,
                    deleted,
                    "{} deleted={} but the review said: {}",
                    text,
                    deleted,
                    mentions[0]
                );
                reach.lock().expect("reach lock").hit(if deleted {
                    "shared cache: deleted and warned"
                } else {
                    "shared cache: kept and noted"
                });
            }
        }

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

fn run_uninstall_property(cases: u32, require_safety: bool) -> Result<(), String> {
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
        uninstall_case(choices, &reach, require_safety)
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

/// Honesty: whatever the three dirs point at and however they are spelled
/// (a trailing `/` or `/.` on a symlinked root included), `rocm uninstall`
/// removes exactly what its review listed — no more, and no planned link left
/// behind because an earlier step made it dangle.
#[test]
fn uninstall_removes_exactly_what_the_review_lists() {
    if let Err(error) = run_uninstall_property(case_count(), false) {
        panic!("{error}");
    }
}

/// Safety: `rocm uninstall` must never delete a folder ROCm CLI did not create.
/// A data/cache/config dir that names `$HOME` (directly, or through a link,
/// however spelled) must be refused or unlinked, never emptied — the shape
/// this used to shrink to was `cache_dir = $HOME`.
#[test]
fn uninstall_never_deletes_a_folder_rocm_cli_did_not_create() {
    if let Err(error) = run_uninstall_property(case_count(), true) {
        panic!("{error}");
    }
}

/// A planned link whose target an earlier plan step already deleted (here:
/// `cache_dir` is a link into `data_dir`) dangles by the time its turn comes.
/// It was listed and reported as "removed cache ...", so it must be gone.
#[test]
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
    let plan = build_uninstall_plan(&paths, &AppPathSources::default(), &options).expect("plan");
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

/// A cache root that is already a dangling link is still an entry on disk: the
/// review lists it (rather than calling it "not present") and removal unlinks it.
#[test]
fn uninstall_lists_and_removes_a_dangling_link_root() {
    let sandbox = fresh_sandbox("uninstall-dangling-root");
    let link = sandbox.join("rocm-cache");
    symlink(sandbox.join("does-not-exist"), &link).expect("link");
    let paths = AppPaths {
        config_dir: sandbox.join("no-config"),
        data_dir: sandbox.join("no-data"),
        cache_dir: link.clone(),
    };
    let options = UninstallOptions {
        yes: true,
        keep_binaries: true,
        ..UninstallOptions::default()
    };
    let plan = build_uninstall_plan(&paths, &AppPathSources::default(), &options).expect("plan");
    let listed = plan.actions.iter().any(|entry| entry.path == link);
    for entry in &plan.actions {
        remove_path(&entry.path).expect("remove");
    }
    let link_left = std::fs::symlink_metadata(&link).is_ok();
    cleanup(&sandbox);
    assert!(listed, "a dangling link root must be listed: {plan:?}");
    assert!(!link_left, "dangling link {} survived", link.display());
}

// ---------------------------------------------------------------------------
// remove_path: trailing-separator spellings of a symlinked directory
// ---------------------------------------------------------------------------

/// `remove_path` decides "link or directory?" with `symlink_metadata`, but a
/// trailing `/` or `/.` makes the kernel resolve the final link, so `link/`
/// would stat as the *target* directory and `remove_dir_all` would walk into
/// it. Spelling alone must not change what is deleted: only the link goes.
#[test]
fn remove_path_treats_a_trailing_slash_link_like_the_link() {
    for suffix in ["/", "/.", "//"] {
        let sandbox = fresh_sandbox("trailing-slash");
        let target = sandbox.join("relocated-cache");
        std::fs::create_dir_all(target.join("sub")).expect("target");
        std::fs::write(target.join("precious.txt"), b"keep").expect("precious");
        std::fs::write(target.join("sub/more.txt"), b"keep").expect("more");
        let link = sandbox.join("cache-link");
        symlink(&target, &link).expect("link");

        let spelled = PathBuf::from(format!("{}{suffix}", link.display()));
        let outcome = remove_path(&spelled);

        let survived =
            target.join("precious.txt").is_file() && target.join("sub/more.txt").is_file();
        let link_left = std::fs::symlink_metadata(&link).is_ok();
        cleanup(&sandbox);
        assert!(
            survived,
            "remove_path({}) reached through the link and deleted the target's contents \
             (result: {outcome:?}, link still present: {link_left})",
            spelled.display()
        );
        assert!(
            outcome.is_ok(),
            "remove_path({}) failed: {outcome:?}",
            spelled.display()
        );
        assert!(
            !link_left,
            "remove_path({}) left the link in place",
            spelled.display()
        );
    }
}

/// A dangling link is an entry on disk, not an absence: `remove_path` unlinks
/// it instead of returning early, so callers that report it removed are right.
#[test]
fn remove_path_removes_a_dangling_link() {
    let sandbox = fresh_sandbox("dangling");
    let link = sandbox.join("archive.tar.gz");
    symlink(sandbox.join("does-not-exist"), &link).expect("link");
    let outcome = remove_path(&link);
    let link_left = std::fs::symlink_metadata(&link).is_ok();
    cleanup(&sandbox);
    assert!(outcome.is_ok(), "remove_path failed: {outcome:?}");
    assert!(!link_left, "dangling link {} survived", link.display());
}

/// A regular file written as `notes.txt/` or `notes.txt/.` names nothing to the
/// kernel. Respelling it to `notes.txt` must not turn it into something to
/// delete: `remove_path` leaves it, and the uninstall plan calls it not present.
#[test]
fn a_regular_file_spelled_as_a_directory_is_not_removed() {
    for suffix in ["/", "/."] {
        let sandbox = fresh_sandbox("file-as-dir");
        let file = sandbox.join("notes.txt");
        std::fs::write(&file, b"keep").expect("file");
        let spelled = PathBuf::from(format!("{}{suffix}", file.display()));

        let outcome = remove_path(&spelled);
        let removed_by_remove_path = !file.is_file();

        let paths = AppPaths {
            config_dir: sandbox.join("no-config"),
            data_dir: sandbox.join("no-data"),
            cache_dir: spelled.clone(),
        };
        let options = UninstallOptions {
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        let plan =
            build_uninstall_plan(&paths, &AppPathSources::default(), &options).expect("plan");
        cleanup(&sandbox);

        assert!(
            outcome.is_ok(),
            "remove_path({}) failed: {outcome:?}",
            spelled.display()
        );
        assert!(
            !removed_by_remove_path,
            "remove_path({}) deleted the file",
            spelled.display()
        );
        assert!(
            plan.actions.is_empty(),
            "{} was planned: {plan:?}",
            spelled.display()
        );
        assert!(
            plan.skipped
                .iter()
                .any(|line| line == &format!("cache path not present: {}", spelled.display())),
            "{plan:?}"
        );
    }
}

/// The control for the test above: the same spellings of a real directory
/// still name it, and it is removed.
#[test]
fn a_directory_spelled_with_a_trailing_separator_is_removed() {
    for suffix in ["/", "/."] {
        let sandbox = fresh_sandbox("dir-as-dir");
        let dir = sandbox.join("cache");
        std::fs::create_dir_all(dir.join("sub")).expect("dir");
        let spelled = PathBuf::from(format!("{}{suffix}", dir.display()));
        let outcome = remove_path(&spelled);
        let left = std::fs::symlink_metadata(&dir).is_ok();
        cleanup(&sandbox);
        assert!(outcome.is_ok(), "{outcome:?}");
        assert!(
            !left,
            "remove_path({}) left the directory",
            spelled.display()
        );
    }
}
