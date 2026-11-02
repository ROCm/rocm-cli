// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The loader library-copy search: every copy of the ROCm libraries the loader
//! could pick (comgr and the HIP runtime), where each lives, and which install
//! owns it.

use super::{Examination, LibraryCopy};
use crate::FrameworkInterpreter;
use std::path::PathBuf;

/// Every file in `ld` whose name starts with `prefix`, in search order.
///
/// One walker, two callers. The HIP probe wants only the first hit; the code
/// object manager scan wants all of them, because stopping at the first is
/// exactly what made a second copy invisible. Written once so the two searches
/// cannot come to disagree about what "on the library path" means.
///
/// `LD_LIBRARY_PATH` is a Linux concept and so is its `:` separator — a Windows
/// path contains a colon, so splitting one here would destroy it. That costs
/// nothing in practice: the variable is not what the Windows loader reads, the
/// code object manager scan runs only on Linux, and on Windows the HIP caller
/// sees an unset variable and finds nothing, exactly as it did before. It does
/// mean the tests for this are Unix-only, and they say so.
pub(super) fn libraries_on_ld_path(ld: &str, prefix: &str) -> Vec<String> {
    let mut found = Vec::new();
    for dir in ld.split(':') {
        if dir.is_empty() {
            continue;
        }
        collect_libraries_in_dir(std::path::Path::new(dir), prefix, &mut found);
    }
    found
}

/// Append every file in `dir` whose name starts with `prefix`, sorted so the
/// result does not depend on directory iteration order.
///
/// An unreadable directory contributes nothing and is not an error: the library
/// path routinely names directories that do not exist, and a probe that failed
/// on one would report nothing about the machine it was asked to describe.
fn collect_libraries_in_dir(dir: &std::path::Path, prefix: &str, found: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut matches: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
        .map(|entry| entry.path().to_string_lossy().into_owned())
        .collect();
    // Sorted, which does change the HIP probe's tie-break when one directory
    // holds more than one matching file: it used to take whatever `read_dir`
    // happened to yield first. Deterministic is the better answer for a report
    // two people compare, but it is a change, not a no-op.
    matches.sort();
    found.append(&mut matches);
}

/// The library HIP uses to compile device code at run time.
const COMGR_LIB_PREFIX: &str = "libamd_comgr";

/// Sources the loader itself actually consults, in the order it consults them.
///
/// `rocm-install` and `managed-runtime` hits are real copies sitting on disk,
/// but the loader does not search `/opt` or a managed runtime's directory on
/// its own -- a copy found only there loads solely because something *else*
/// (an active runtime's prepended `LD_LIBRARY_PATH`, or the loader cache after
/// `ldconfig` has been run on it) already put it on a path the loader does
/// consult, and that case is already covered by the tiers listed here. A copy
/// whose *only* hit is `rocm-install`/`managed-runtime` is evidence the file
/// exists, not evidence anything would load it.
const LOADER_PATH_SOURCES: [&str; 3] = ["active-runtime", "ld-library-path", "loader-cache"];

/// Find every copy of `prefix`, decide which one (if any) would load, and note
/// the result -- applied identically to the code object manager and the HIP
/// runtime, since both are searched by [`find_library_copies`] and both answer
/// "which one would load" by the same rule: see [`select_loader_copy`].
///
/// Every entry in `copies` is evidence a copy exists; only the selected one is
/// one the loader would actually pick. A machine can hold a system copy and a
/// wheel copy, and when the code object manager that loads does not belong to
/// the active runtime, device code compilation fails with an error naming
/// neither -- which is why the HIP runtime gets exactly the same treatment:
/// deciding whether the code object manager belongs to the active runtime
/// means knowing which runtime is active, and that is the same question about
/// a different file.
///
/// **This emulates the loader; it is not the loader.** It does not account for
/// `RUNPATH`/`RPATH` on the calling binary, `ld.so.preload`, or a container that
/// remaps paths at run time. The selected copy is the one that *would* load on
/// the evidence available, and the list of copies is that evidence -- not a
/// guarantee. Reporting the search honestly is worth more here than a confident
/// answer that cannot be justified.
pub(super) fn probe_comgr(e: &mut Examination, interpreter: Option<&FrameworkInterpreter>) {
    let active_runtime_dirs: Vec<PathBuf> = interpreter
        .map(|interpreter| interpreter.library_paths.clone())
        .unwrap_or_default();
    let roots = known_install_roots(e);
    // Computed once and shared by both searches below: neither the loader
    // cache nor the directory walk depends on which library is being searched
    // for, so computing them per prefix ran `ldconfig -p` and walked every
    // managed runtime's lib/lib64 -> site-packages -> `_rocm_sdk_*` tree twice
    // on every `rocm examine`/`rocm diagnose` invocation -- a command typically
    // run repeatedly while debugging.
    //
    // `interpreter`'s `library_paths` -- the active managed runtime's own
    // library directories, when there is one -- are checked **first**, ahead
    // of this process's own `LD_LIBRARY_PATH`. That is not this process's
    // search order; it is a served child's. `rocm serve`/`rocm chat` prepend
    // exactly those directories onto `LD_LIBRARY_PATH` before launching the
    // engine (see `lemonade_process_environment_vars` and its vLLM
    // counterpart), so they win over whatever this process's own environment
    // or the system loader cache would otherwise resolve to. Reporting the
    // plain-environment answer instead would name the system's copy as "the
    // one that would load" on a machine where every process the CLI actually
    // launches loads the wheel's.
    let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let loader_cache_text = crate::ldconfig_cache();
    let dirs = install_library_dirs(&roots);
    let dir_owners: std::collections::HashMap<std::path::PathBuf, std::path::PathBuf> = dirs
        .iter()
        .map(|(dir, _source, root)| (dir.clone(), root.clone()))
        .collect();

    let comgr = find_library_copies(
        COMGR_LIB_PREFIX,
        &active_runtime_dirs,
        &ld,
        loader_cache_text.as_deref(),
        &dirs,
        &dir_owners,
    );
    let hip = find_library_copies(
        HIP_RUNTIME_LIB_PREFIX,
        &active_runtime_dirs,
        &ld,
        loader_cache_text.as_deref(),
        &dirs,
        &dir_owners,
    );

    let comgr_selected = select_loader_copy(&comgr);
    if let Some(selected) = &comgr_selected {
        e.comgr_version.clone_from(&selected.version);
    }
    if let Some(note) = copies_note(COMGR_LIB_PREFIX, &comgr, comgr_selected.as_ref()) {
        e.notes.push(note);
    }

    let hip_selected = select_loader_copy(&hip);
    if let Some(note) = copies_note(HIP_RUNTIME_LIB_PREFIX, &hip, hip_selected.as_ref()) {
        e.notes.push(note);
    }

    e.comgr_matches_runtime =
        comgr_matches_runtime(&comgr, comgr_selected.as_ref(), hip_selected.as_ref());
    e.comgr_selected = comgr_selected;
    e.hip_selected = hip_selected;
    e.comgr_paths = comgr;
    e.hip_paths = hip;
}

/// Whether the selected code object manager belongs to the same installation as
/// the selected HIP runtime -- and, when it does not, whether that runtime ships
/// a copy of its own to prefer instead.
///
/// This is the one place the conflict rule is stated. `probe_comgr` reports the
/// result as `comgr_matches_runtime` and `check_18_comgr_conflict` in
/// `diagnose.rs` turns a `Some(false)` into the finding; both call this function
/// rather than restating the rule, so the two surfaces of one question cannot
/// disagree about the same machine.
///
/// `None` covers every case where there is not enough evidence to call it either
/// way: either library missing, an unattributed copy on either side, or -- the
/// case that matters most -- a runtime whose own installation ships no code
/// object manager at all. That last one is not a conflict: there is no copy of
/// its own for it to prefer, so pointing the user at "the runtime's own copy"
/// would be pointing at nothing.
pub(crate) fn comgr_matches_runtime(
    comgr_paths: &[LibraryCopy],
    comgr_selected: Option<&LibraryCopy>,
    hip_selected: Option<&LibraryCopy>,
) -> Option<bool> {
    let (comgr, hip) = (comgr_selected?, hip_selected?);
    if comgr.install_root.is_empty() || hip.install_root.is_empty() {
        return None;
    }
    if comgr.install_root == hip.install_root {
        return Some(true);
    }
    let has_own_copy = comgr_paths
        .iter()
        .any(|copy| copy.install_root == hip.install_root);
    if !has_own_copy {
        return None;
    }
    Some(false)
}

/// The HIP runtime. Which copy of it loads decides which installation is the
/// active one, which is the other half of the code object manager question.
const HIP_RUNTIME_LIB_PREFIX: &str = "libamdhip64";

/// The copy the loader would actually pick: the first entry whose source is
/// one of [`LOADER_PATH_SOURCES`]. `None` when no copy was found on one of
/// those tiers, even when `copies` is not empty -- a `rocm-install` or
/// `managed-runtime` hit is evidence a copy exists, not evidence the loader
/// would pick it.
///
/// Applied identically to the code object manager and the HIP runtime: both
/// are searched by [`find_library_copies`], so both answer "which one would
/// load" by the same rule. See `a_search_dir_only_hit_is_not_selected_for_either_prefix`
/// for the test that pins this for both.
fn select_loader_copy(copies: &[LibraryCopy]) -> Option<LibraryCopy> {
    copies
        .iter()
        .find(|copy| LOADER_PATH_SOURCES.contains(&copy.source.as_str()))
        .cloned()
}

/// The note (if any) `probe_comgr` should push about `copies` found for
/// `prefix`, given which one (if any) was `selected`.
///
/// Three cases: no copies at all; more than one copy with one of them
/// selected (the "would load" case); and one or more copies with none
/// selected, meaning every hit sits only in a `rocm-install` or
/// `managed-runtime` directory the loader does not consult on its own, so it
/// is unknown whether any of them loads.
fn copies_note(
    prefix: &str,
    copies: &[LibraryCopy],
    selected: Option<&LibraryCopy>,
) -> Option<String> {
    if copies.is_empty() {
        return Some(format!(
            "no {prefix} found on the library path, in the loader cache, or in any known ROCm install"
        ));
    }
    if let Some(selected) = selected {
        return (copies.len() > 1).then(|| {
            format!(
                "{} copies of {prefix} found; {} would load",
                copies.len(),
                selected.path
            )
        });
    }
    let count = copies.len();
    let plural = if count == 1 { "copy" } else { "copies" };
    Some(format!(
        "{count} {plural} of {prefix} found, but none on a path the loader consults -- only in \
         a ROCm install or a managed runtime, which the loader does not search on its own, so it \
         is unknown whether any of them loads"
    ))
}

/// Every copy of `prefix` on the machine, in loader search order, each attributed
/// to the installation that owns it.
///
/// `active_runtime_dirs` -- the active managed runtime's own library
/// directories, when there is one -- are checked **first**, ahead of
/// `ld_library_path`. That is not this process's search order; it is a served
/// child's. `rocm serve`/`rocm chat` prepend exactly those directories onto
/// `LD_LIBRARY_PATH` before launching the engine (see
/// `lemonade_process_environment_vars` and its vLLM counterpart), so they win
/// over whatever this process's own environment or the system loader cache
/// would otherwise resolve to. Reporting the plain-environment answer instead
/// would name the system's copy as "the one that would load" on a machine
/// where every process the CLI actually launches loads the wheel's.
///
/// `ld_library_path`, `loader_cache_text`, and `dirs`/`dir_owners` are
/// gathered once by the caller and shared between the comgr and HIP searches,
/// rather than each reading the real environment, spawning `ldconfig`, and
/// walking every managed runtime's directories again -- neither depends on
/// which library is being searched for. That sharing is also what makes this
/// fully testable without a real `ldconfig`, a real `/opt`, or a real managed
/// runtime: every input is a plain value a test can construct.
fn find_library_copies(
    prefix: &str,
    active_runtime_dirs: &[PathBuf],
    ld_library_path: &str,
    loader_cache_text: Option<&str>,
    dirs: &[(std::path::PathBuf, &'static str, std::path::PathBuf)],
    dir_owners: &std::collections::HashMap<std::path::PathBuf, std::path::PathBuf>,
) -> Vec<LibraryCopy> {
    let mut copies: Vec<LibraryCopy> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Order is the whole point: this is the order a served child's loader
    // would consult, so the first copy recorded on a loader-consulted tier is
    // the one that wins.
    for dir in active_runtime_dirs {
        let mut hits = Vec::new();
        collect_libraries_in_dir(dir, prefix, &mut hits);
        for path in hits {
            record_library_copy(&mut copies, &mut seen, &path, "active-runtime", dir_owners);
        }
    }
    for path in libraries_on_ld_path(ld_library_path, prefix) {
        record_library_copy(&mut copies, &mut seen, &path, "ld-library-path", dir_owners);
    }
    if let Some(text) = loader_cache_text {
        for path in parse_ldconfig_cache_paths(text, prefix) {
            record_library_copy(&mut copies, &mut seen, &path, "loader-cache", dir_owners);
        }
    }
    for (dir, source, _root) in dirs {
        let mut hits = Vec::new();
        collect_libraries_in_dir(dir, prefix, &mut hits);
        for path in hits {
            record_library_copy(&mut copies, &mut seen, &path, source, dir_owners);
        }
    }
    copies
}

/// The installation that owns `real_path`: whichever root's directory holds
/// it, per `dir_owners`. Empty when no known installation's directory holds
/// it -- a library somewhere unexpected is reported as belonging nowhere
/// rather than guessed into an install it is not part of.
fn owning_install_root(
    real_path: &str,
    dir_owners: &std::collections::HashMap<std::path::PathBuf, std::path::PathBuf>,
) -> String {
    std::path::Path::new(real_path)
        .parent()
        .and_then(|dir| dir_owners.get(dir))
        .map(|root| root.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Record `path` unless an earlier entry already resolved to the same file.
///
/// Deduplicated on the resolved path, not the given one: ROCm ships a versioned
/// library and an unversioned symlink beside it, and reporting those as two
/// copies would invent a conflict on a perfectly ordinary install.
fn record_library_copy(
    copies: &mut Vec<LibraryCopy>,
    seen: &mut std::collections::BTreeSet<String>,
    path: &str,
    source: &str,
    dir_owners: &std::collections::HashMap<std::path::PathBuf, std::path::PathBuf>,
) {
    // A path that cannot be resolved is kept as given rather than dropped: a
    // dangling symlink is still something the loader would try, and reporting
    // it is more use than pretending the machine does not hold it.
    let real_path = std::fs::canonicalize(path).map_or_else(
        |_| path.to_owned(),
        |resolved| resolved.to_string_lossy().into_owned(),
    );
    if !seen.insert(real_path.clone()) {
        return;
    }
    copies.push(LibraryCopy {
        version: comgr_version_from_file_name(&real_path),
        // Attributed by the resolved path: a symlink from one installation into
        // another's file belongs to the installation holding the file.
        install_root: owning_install_root(&real_path, dir_owners),
        path: path.to_owned(),
        real_path,
        source: source.to_owned(),
    });
}

/// Read the version out of a resolved library file name.
///
/// `libamd_comgr.so.2.8.0` carries its version in the soname, which is where
/// this reads it from. Deliberately not by loading the library and asking it:
/// `dlopen` runs the library's initialisers, and running code out of an
/// unknown library is not something a diagnostic tool should do on a machine it
/// has been called to because something is already wrong. A renamed file yields
/// an empty version, which is the honest answer.
fn comgr_version_from_file_name(real_path: &str) -> String {
    let name = std::path::Path::new(real_path)
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    let Some((_, version)) = name.split_once(".so.") else {
        return String::new();
    };
    // Digits and dots only: `libamd_comgr.so.2.8.0` yields a version, while a
    // file that merely happens to carry `.so.` in a longer name does not get a
    // nonsense one read out of it.
    if !version.is_empty() && version.chars().all(|c| c.is_ascii_digit() || c == '.') {
        version.to_owned()
    } else {
        String::new()
    }
}

/// Parse `ldconfig -p`'s own output (`crate::ldconfig_cache()`) for every path
/// it lists against `prefix`.
///
/// `ldconfig -p` prints `libamd_comgr.so.2 (libc6,x86-64) => /opt/rocm/lib/...`.
/// A missing cache, or a nonzero exit from `ldconfig` itself, contributes
/// nothing rather than failing the probe: plenty of hosts have no `ldconfig`
/// on the path, and a machine with no loader cache is still a machine worth
/// describing -- `probe_comgr` reads `crate::ldconfig_cache()` once and passes
/// `None` straight through for exactly that case.
///
/// Takes the already-read text rather than calling `crate::ldconfig_cache()`
/// itself, so the line format -- a fixed property of `ldconfig`, not of this
/// host -- can be pinned against literal text instead of a real `ldconfig`
/// binary, which not every machine running this test has on `PATH`; it also
/// means `probe_comgr` reads the cache once and parses it twice (comgr, HIP)
/// rather than spawning `ldconfig -p` twice for one invocation.
fn parse_ldconfig_cache_paths(cache_text: &str, prefix: &str) -> Vec<String> {
    cache_text
        .lines()
        .filter(|line| line.contains(prefix))
        .filter_map(|line| line.split_once("=> "))
        .map(|(_, path)| path.trim().to_owned())
        .collect()
}

/// Sibling ROCm installs under `/opt`, sorted for a deterministic search order.
///
/// A versioned install left behind by an upgrade is one of the two copies
/// `probe_comgr` exists to find. Takes the directory to scan as a parameter
/// -- always `/opt` in production -- so a test can point it at a temp folder
/// instead of depending on what happens to live under the real `/opt` on the
/// machine running the suite.
fn rocm_install_siblings(opt_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(opt_dir) else {
        return Vec::new();
    };
    let mut roots: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("rocm"))
        })
        .collect();
    roots.sort();
    roots
}

/// Every ROCm installation on the machine, each paired with the source label its
/// libraries are recorded under.
///
/// A managed runtime is **one** root even though its libraries are spread across
/// several `_rocm_sdk_*` directories. That is what lets the diagnosis tell a
/// mixed environment from a healthy managed one: split those directories into
/// separate installations and every healthy managed install looks like a
/// conflict.
fn known_install_roots(e: &Examination) -> Vec<(std::path::PathBuf, &'static str)> {
    let mut roots: Vec<(std::path::PathBuf, &'static str)> = Vec::new();
    if !e.rocm_path.is_empty() {
        roots.push((std::path::PathBuf::from(&e.rocm_path), "rocm-install"));
    }
    // Sibling ROCm installs the active one does not cover. A versioned install
    // left behind by an upgrade is one of the two copies this entry exists to
    // find.
    roots.extend(
        rocm_install_siblings(std::path::Path::new("/opt"))
            .into_iter()
            .map(|root| (root, "rocm-install")),
    );
    // Only the root is carried here; `install_library_dirs` looks the recorded
    // `library_paths` back up by root when it needs it, since this tuple's
    // shape is shared with `rocm-install` roots that have no such thing.
    roots.extend(
        managed_runtime_roots()
            .into_iter()
            .map(|(root, _library_paths)| (root, "managed-runtime")),
    );
    dedup_roots_keeping_first(roots)
}

/// Drop repeats of a root already seen, keeping the first occurrence.
///
/// Order is load-bearing: it is the order the loader would search, and
/// `install_library_dirs` attributes by exact directory membership, so a root
/// recorded twice would walk (and attribute to two distinct, textually-equal)
/// copies of the same installation's directories.
fn dedup_roots_keeping_first(
    roots: Vec<(std::path::PathBuf, &'static str)>,
) -> Vec<(std::path::PathBuf, &'static str)> {
    let mut seen = std::collections::HashSet::new();
    roots
        .into_iter()
        .filter(|(root, _)| seen.insert(root.clone()))
        .collect()
}

/// The library directories of every known installation, in root order, each
/// paired with the source label and the root that owns it.
///
/// Reuses the layout the SDK probe already knows rather than restating it: a
/// second description of where an installation keeps its libraries is a second
/// thing to keep correct.
///
/// The owning root travels with each directory because a managed runtime's
/// directories are not all nested under its root (see the comment on `root`
/// below) -- a caller attributing a found file by string-prefix match against
/// the bare root would find nothing for exactly the directories this function
/// exists to add. Keying attribution off the same directories this search
/// walks, instead, is what `find_library_copies` does with the return value.
fn install_library_dirs(
    roots: &[(std::path::PathBuf, &'static str)],
) -> Vec<(std::path::PathBuf, &'static str, std::path::PathBuf)> {
    // For a managed runtime, `root` is the `_rocm_sdk_devel` package directory
    // the SDK probe resolved (see `managed_runtime_roots`), not a venv root --
    // it has no `lib/<python>/site-packages` of its own to walk, and guessing
    // one there finds nothing. The probe already recorded where the runtime's
    // libraries actually are: `library_paths` is built by importing each real
    // package ("core", "libraries", "device", "profiler") and asking Python for
    // its file location (`ROCM_SDK_PROBE_SCRIPT`), the same value
    // `probe_runtime_devices` puts on `LD_LIBRARY_PATH` for a served process.
    // Prefer that recorded truth over re-deriving the layout, which is the bug
    // `examine-finds-the-managed-runtimes-own-compilation-library` catches on a
    // real install: the guess never matches, so the runtime's own code object
    // manager library goes unseen.
    let managed_library_paths: std::collections::HashMap<
        std::path::PathBuf,
        Vec<std::path::PathBuf>,
    > = managed_runtime_roots().into_iter().collect();
    let mut dirs = Vec::new();
    for (root, source) in roots {
        let mut paths = Vec::new();
        crate::collect_sdk_library_paths(root, &mut paths);
        if *source == "managed-runtime"
            && let Some(recorded) = managed_library_paths.get(root)
        {
            paths.extend(recorded.iter().cloned());
        }
        dirs.extend(
            paths
                .into_iter()
                .filter(|path| path.is_dir())
                .map(|path| (path, *source, root.clone())),
        );
    }
    dirs
}

/// Every managed runtime, as `(root, library_paths)`.
///
/// Read from the runtime registry rather than by listing `<data>/runtimes` on
/// disk. A directory listing yields a root and nothing else, and the root alone
/// does not locate a wheel runtime's libraries -- only the SDK record knows
/// where its packages actually resolved to. Listing the directory is also a
/// second answer to "which runtimes exist", and the registry is the first one.
///
/// Resolved through [`crate::AppPaths::discover`], not
/// `crate::runtime::default_data_dir` directly: the latter only knows
/// `$HOME/.rocm` and the OS default, so it disagrees with the rest of the CLI
/// -- and finds nothing at all -- on any host where `ROCM_CLI_DATA_DIR`
/// relocates the data directory, which is exactly what every GPU e2e scenario
/// does for isolation. That mismatch, not the library-directory guess this
/// search used to make, is why a managed runtime's own libraries were
/// unreachable end to end.
fn managed_runtime_roots() -> Vec<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let Ok(paths) = crate::AppPaths::discover() else {
        return Vec::new();
    };
    let registry = paths.data_dir.join("runtimes").join("registry");
    let mut runtimes: Vec<(std::path::PathBuf, Vec<std::path::PathBuf>)> =
        crate::managed_therock_sdk_probe_candidates(&registry)
            .into_iter()
            .map(|candidate| (candidate.root_path, candidate.library_paths))
            .collect();
    runtimes.sort();
    runtimes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examine::tests::plant;

    /// A scratch directory unique to the calling test, cleaned up by the caller.
    fn scratch_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rocm-comgr-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    // Unix-only: these drive `LD_LIBRARY_PATH` semantics and POSIX symlinks
    // directly. The variable uses `:` as its separator, which a Windows path
    // contains, and `symlink` needs privileges there. The code under test runs
    // only on Linux, so gating the tests loses no coverage of a path that ships.
    #[cfg(unix)]
    #[test]
    fn the_library_path_is_searched_left_to_right_and_every_copy_is_kept() {
        // The defect this whole entry exists for: the old scan stopped at the
        // first hit, so a second copy could never be reported. Order matters as
        // much as completeness -- the first entry is the claim about which copy
        // wins.
        let root = scratch_dir("order");
        let first = root.join("first");
        let second = root.join("second");
        std::fs::create_dir_all(&first).expect("create dir");
        std::fs::create_dir_all(&second).expect("create dir");
        plant(&first, "libamd_comgr.so.2.8.0");
        plant(&second, "libamd_comgr.so.3.0.0");

        let ld = format!("{}:{}", first.display(), second.display());
        let found = libraries_on_ld_path(&ld, COMGR_LIB_PREFIX);

        assert_eq!(found.len(), 2, "both copies must be reported: {found:?}");
        assert!(
            found[0].starts_with(first.to_string_lossy().as_ref()),
            "the earlier library-path entry has to come first: {found:?}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    // Unix-only: these drive `LD_LIBRARY_PATH` semantics and POSIX symlinks
    // directly. The variable uses `:` as its separator, which a Windows path
    // contains, and `symlink` needs privileges there. The code under test runs
    // only on Linux, so gating the tests loses no coverage of a path that ships.
    #[cfg(unix)]
    #[test]
    fn a_missing_library_path_entry_is_skipped_rather_than_failing_the_probe() {
        // `LD_LIBRARY_PATH` routinely names directories that do not exist. A
        // probe that gave up on one would report nothing about the machine it
        // was asked to describe.
        let root = scratch_dir("missing");
        plant(&root, "libamd_comgr.so.2.8.0");
        let ld = format!("/nonexistent-{}:{}", std::process::id(), root.display());

        let found = libraries_on_ld_path(&ld, COMGR_LIB_PREFIX);

        assert_eq!(found.len(), 1, "the readable entry still counts: {found:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    // Unix-only: these drive `LD_LIBRARY_PATH` semantics and POSIX symlinks
    // directly. The variable uses `:` as its separator, which a Windows path
    // contains, and `symlink` needs privileges there. The code under test runs
    // only on Linux, so gating the tests loses no coverage of a path that ships.
    #[cfg(unix)]
    #[test]
    fn a_versioned_library_and_its_symlink_count_as_one_copy() {
        // ROCm ships `libamd_comgr.so.2` beside `libamd_comgr.so.2.8.0`, one a
        // symlink to the other. Counting those as two copies would invent a
        // conflict on an ordinary install -- the false report that matters most
        // to avoid, since it would fire on healthy machines.
        let root = scratch_dir("symlink");
        let real = plant(&root, "libamd_comgr.so.2.8.0");
        let link = root.join("libamd_comgr.so.2");
        std::os::unix::fs::symlink(&real, &link).expect("create symlink");

        let mut copies = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        record_library_copy(
            &mut copies,
            &mut seen,
            &link.to_string_lossy(),
            "test",
            &std::collections::HashMap::new(),
        );
        record_library_copy(
            &mut copies,
            &mut seen,
            &real.to_string_lossy(),
            "test",
            &std::collections::HashMap::new(),
        );

        assert_eq!(
            copies.len(),
            1,
            "one file reached by two names is one copy: {copies:?}"
        );
        assert_eq!(
            copies[0].version, "2.8.0",
            "the version comes from the resolved file, not the name used to reach it"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn two_distinct_files_are_two_copies() {
        // The converse of the symlink case, so that test cannot be satisfied by
        // a probe that simply never reports more than one.
        let root = scratch_dir("distinct");
        let a = plant(&root, "libamd_comgr.so.2.8.0");
        let b = plant(&root, "libamd_comgr.so.3.0.0");

        let mut copies = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        record_library_copy(
            &mut copies,
            &mut seen,
            &a.to_string_lossy(),
            "test",
            &std::collections::HashMap::new(),
        );
        record_library_copy(
            &mut copies,
            &mut seen,
            &b.to_string_lossy(),
            "test",
            &std::collections::HashMap::new(),
        );

        assert_eq!(copies.len(), 2, "two files are two copies: {copies:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_library_deep_inside_a_managed_runtime_belongs_to_the_runtime() {
        // The property the whole diagnosis rests on. A managed runtime spreads
        // its libraries across separate `_rocm_sdk_*` packages that sit
        // *beside* each other under one `site-packages`, not nested under the
        // runtime's own root (`_rocm_sdk_devel`, what the SDK probe records as
        // `root_path`) at all -- a venv-shaped fixture nesting
        // `_rocm_sdk_core` under the runtime's root would pass by construction
        // without proving anything about the real layout, which is exactly the
        // shape that let this gap through review once already. Attribution
        // instead keys on the directories `install_library_dirs` actually
        // recorded for the runtime, which is what `dir_owners` below stands
        // in for.
        let runtime = std::path::PathBuf::from("/data/runtimes/therock/_rocm_sdk_devel");
        let mut dir_owners = std::collections::HashMap::new();
        for package in ["_rocm_sdk_core", "_rocm_sdk_devel"] {
            let lib_dir = std::path::PathBuf::from(format!("/data/runtimes/therock/{package}/lib"));
            dir_owners.insert(lib_dir, runtime.clone());
        }

        for package in ["_rocm_sdk_core", "_rocm_sdk_devel"] {
            let path = format!("/data/runtimes/therock/{package}/lib/libamd_comgr.so.2");
            assert_eq!(
                owning_install_root(&path, &dir_owners),
                runtime.to_string_lossy(),
                "{package} belongs to the runtime that recorded its directory, not to itself"
            );
        }
    }

    #[test]
    fn a_library_no_installation_claims_is_attributed_to_none() {
        // Empty rather than guessed. A library somewhere unexpected is a gap in
        // what the search knows, and inventing an owner for it would turn that
        // gap into a false finding about the user's machine.
        let mut dir_owners = std::collections::HashMap::new();
        dir_owners.insert(
            std::path::PathBuf::from("/opt/rocm/lib"),
            std::path::PathBuf::from("/opt/rocm"),
        );
        assert_eq!(
            owning_install_root("/somewhere/else/libamd_comgr.so.2", &dir_owners),
            ""
        );
        // A sibling whose name merely starts the same way is not a parent --
        // and is not even a near miss here: attribution keys on the exact
        // directory recorded for each root, not a string-prefix match, so
        // `/opt/rocm-other/lib` is simply a different key from `/opt/rocm/lib`.
        assert_eq!(
            owning_install_root("/opt/rocm-other/lib/x.so", &dir_owners),
            ""
        );
    }

    #[test]
    fn a_version_is_read_only_when_the_name_actually_carries_one() {
        for (name, expected) in [
            ("libamd_comgr.so.2.8.0", "2.8.0"),
            ("libamd_comgr.so.2", "2"),
            // No version to read. Empty is the honest answer; the alternative
            // is a confident wrong one, and the version is only ever report
            // text.
            ("libamd_comgr.so", ""),
            ("libamd_comgr-renamed.so.beta", ""),
        ] {
            assert_eq!(
                comgr_version_from_file_name(&format!("/x/{name}")),
                expected,
                "{name} read wrong"
            );
        }
    }

    /// One installation is never counted twice, wherever it appears in the list.
    ///
    /// `dedup_by` only drops *consecutive* duplicates, and these roots arrive
    /// from three independent sources: the active install, a sorted `/opt`
    /// scan, and the managed runtimes. The active install collides with a
    /// sibling whenever it is one, and whether the two land adjacent depends on
    /// where the path happens to sort -- so the bug was invisible for exactly
    /// the inputs where the sort was kind.
    ///
    /// A duplicated root is not cosmetic: attribution is by longest known root,
    /// so the same installation appearing twice can make a machine holding one
    /// stack look like a machine holding two, which is the false conflict this
    /// entry exists to avoid reporting.
    #[test]
    fn one_installation_is_never_counted_twice_however_the_paths_sort() {
        use std::path::PathBuf;
        let collide = PathBuf::from("/opt/rocm-7.1");
        let roots = dedup_roots_keeping_first(vec![
            (collide.clone(), "rocm-install"),
            // A sibling that sorts *before* the collision, so the duplicate is
            // not adjacent to it. This ordering is what the old dedup missed.
            (PathBuf::from("/opt/rocm-6.4"), "rocm-install"),
            (collide.clone(), "rocm-install"),
            (
                PathBuf::from("/home/u/.local/share/rocm-cli/runtimes/wheel/a"),
                "managed-runtime",
            ),
        ]);

        let seen = roots.iter().filter(|(p, _)| *p == collide).count();
        assert_eq!(
            seen, 1,
            "one installation was recorded twice, so a single stack can be read as two: {roots:?}"
        );
        // Non-vacuity: deduplicating must not be achieved by dropping roots.
        assert_eq!(
            roots.len(),
            3,
            "every distinct root has to survive, in first-seen order: {roots:?}"
        );
        assert_eq!(
            roots[0].0, collide,
            "first-seen order is loader search order"
        );
    }

    /// A wheel-format managed runtime, laid out the way the CLI installs one.
    ///
    /// `root` is one of the runtime's own `_rocm_sdk_*` package directories --
    /// `_rocm_sdk_devel`, matching what the probe script's
    /// `_devel.get_devel_root()` records as `root_path` when the `devel` extra
    /// is installed -- never a venv root above `site_packages`. The ROCm
    /// libraries do not sit under it either: they sit in the sibling
    /// `_rocm_sdk_core` package, beside `root`, both directly under
    /// `site_packages`. A fixture built as `root/lib/<python>/site-packages`
    /// (a venv layout the installer never produces) would pass against a shape
    /// production cannot create -- this one matches `apps/rocm/src/therock.rs`'s
    /// `ROCM_SDK_PROBE_SCRIPT` instead.
    #[cfg(target_os = "linux")]
    fn wheel_runtime_on_disk(tag: &str) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
        use std::fs;
        let base = std::env::temp_dir().join(format!(
            "rocm-comgr-wheel-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&base);
        let site_packages = base.join("site-packages");
        let root = site_packages.join("_rocm_sdk_devel");
        let sdk_lib = site_packages.join("_rocm_sdk_core").join("lib");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&sdk_lib).unwrap();
        fs::write(sdk_lib.join("libamd_comgr.so.3"), b"fake").unwrap();
        (root, Some(site_packages))
    }

    /// The managed copy this CLI installs itself is reachable by the search.
    ///
    /// That copy is the whole reason this entry exists: the CLI puts ROCm
    /// wheels into a managed environment, so a user who follows that path on a
    /// host already carrying a system install ends up holding both, having done
    /// nothing unusual. A search that cannot see the copy we put there reports
    /// a conflict-free machine no matter what else is true of it.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_managed_copy_this_cli_installs_is_reachable() {
        let (root, site_packages) = wheel_runtime_on_disk("found");
        let mut dirs = Vec::new();
        crate::managed_runtime::collect_managed_runtime_library_paths(
            &root,
            site_packages.as_deref(),
            &mut dirs,
        );

        let expected = site_packages
            .expect("fixture always records site_packages")
            .join("_rocm_sdk_core")
            .join("lib");
        assert!(
            dirs.contains(&expected),
            "the search missed the copy the CLI installs, which is the case this entry exists \
             for. Looked in: {dirs:?}"
        );
        let _ = std::fs::remove_dir_all(root.parent().and_then(|p| p.parent()).unwrap());
    }

    /// A runtime whose SDK recorded no `site-packages` still contributes what
    /// can be read from its root.
    ///
    /// A direct call with `None`, not a path any real registry record takes
    /// (every real candidate's `site_packages` is recorded unconditionally,
    /// so `collect_managed_runtime_library_paths` never actually receives
    /// `None` from `managed_runtime_roots`'s real caller). What this pins is the
    /// `None` arm's own contract for any caller that does pass it directly: a
    /// plain install still contributes what sits directly under its root,
    /// with no sibling packages to go looking for.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_runtime_with_no_recorded_site_packages_still_contributes_its_root() {
        use std::fs;
        let root = std::env::temp_dir().join(format!(
            "rocm-comgr-rootonly-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("lib")).unwrap();

        let mut dirs = Vec::new();
        crate::managed_runtime::collect_managed_runtime_library_paths(&root, None, &mut dirs);
        assert!(
            dirs.contains(&root.join("lib")),
            "a root-format runtime keeps its libraries under the root, and that has to keep \
             working: {dirs:?}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// A directory holding a `libamd_comgr` file, for [`find_library_copies`]
    /// tests below. Each call gets its own temp directory so the tests can run
    /// concurrently without seeing each other's files.
    ///
    /// `#[cfg(unix)]`, matching its only callers: every one of them feeds the
    /// directory into the `:`-split `LD_LIBRARY_PATH` parser, so leaving this
    /// helper itself ungated would make it unused (and clippy-denied) on
    /// Windows once its callers are gated.
    #[cfg(unix)]
    fn comgr_copy_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rocm-comgr-copy-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("libamd_comgr.so.2"), b"fake").unwrap();
        dir
    }

    /// Like [`comgr_copy_dir`], but for any `prefix` -- used by the tests that
    /// pin a rule shared by comgr and the HIP runtime, where the file name
    /// has to match whichever prefix is under test.
    ///
    /// `#[cfg(unix)]` for the same reason as `comgr_copy_dir`.
    #[cfg(unix)]
    fn library_copy_dir(prefix: &str, tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rocm-library-copy-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{prefix}.so.2")), b"fake").unwrap();
        dir
    }

    /// A synthetic `ldconfig -p` line naming `path`, for the `loader_cache_text`
    /// parameter below -- so these tests pin the ordering without depending on
    /// a real `ldconfig` or a real loader cache on the machine running them.
    /// Unix-only, because every caller is: the loader cache is a POSIX
    /// concept and the tests that build one are gated the same way. Without
    /// this the helper is dead code on Windows, and CI compiles with
    /// `-D warnings`, so a dead helper is a hard error rather than a warning.
    #[cfg(unix)]
    fn loader_cache_line_for(path: &std::path::Path) -> String {
        format!(
            "libamd_comgr.so.2 (libc6,x86-64) => {}",
            path.join("libamd_comgr.so.2").display()
        )
    }

    /// The active managed runtime's own directory wins selection even when a
    /// copy also sits on the plain `LD_LIBRARY_PATH` and in the loader cache.
    ///
    /// This is the fix for the case `find_library_copies`'s own doc comment
    /// describes: `rocm serve`/`rocm chat` prepend the active runtime's
    /// library directories onto `LD_LIBRARY_PATH` before launching the engine,
    /// so that copy is the one a served process actually loads -- regardless
    /// of what this process's own environment or the system loader cache
    /// would otherwise resolve to. Before this fix, `find_library_copies` had
    /// no `active_runtime_dirs` parameter at all, and a system copy reachable
    /// through the loader cache was reported as "would load" even on a host
    /// where every process the CLI actually launches loads the wheel's copy.
    ///
    /// Unix-only: feeds a temp path into the `:`-split `LD_LIBRARY_PATH`
    /// parser, same as the neighbouring LD-path tests above -- a Windows path
    /// contains `:` after its drive letter, which would corrupt the split.
    #[cfg(unix)]
    #[test]
    fn the_active_runtimes_own_copy_outranks_the_ambient_environment() {
        let active = comgr_copy_dir("active");
        let ambient = comgr_copy_dir("ambient");
        let cached = comgr_copy_dir("cached");
        let loader_cache_text = loader_cache_line_for(&cached);

        let copies = find_library_copies(
            COMGR_LIB_PREFIX,
            std::slice::from_ref(&active),
            &ambient.to_string_lossy(),
            Some(&loader_cache_text),
            &[],
            &std::collections::HashMap::new(),
        );
        let selected = select_loader_copy(&copies);

        assert_eq!(
            copies.first().map(|copy| &copy.source),
            Some(&"active-runtime".to_owned()),
            "the active runtime's own copy must be selected ahead of the ambient \
             `LD_LIBRARY_PATH` and the loader cache, found: {copies:?}"
        );
        assert_eq!(
            selected.as_ref().map(|copy| &copy.source),
            Some(&"active-runtime".to_owned()),
            "the selected copy must also be the active runtime's own: {selected:?}"
        );
        assert_eq!(
            copies.len(),
            3,
            "all three copies must still be reported: {copies:?}"
        );

        let _ = std::fs::remove_dir_all(&active);
        let _ = std::fs::remove_dir_all(&ambient);
        let _ = std::fs::remove_dir_all(&cached);
    }

    /// With no active-runtime evidence, the plain `LD_LIBRARY_PATH` still wins
    /// over the loader cache and the trailing search directories -- the
    /// ordering `find_library_copies`'s doc comment says is "the whole point".
    ///
    /// Unix-only: same `:`-split reason as above.
    #[cfg(unix)]
    #[test]
    fn ld_library_path_outranks_loader_cache_and_search_dirs() {
        let ld = comgr_copy_dir("ld");
        let cached = comgr_copy_dir("cached2");
        let known = comgr_copy_dir("known");
        let loader_cache_text = loader_cache_line_for(&cached);

        let copies = find_library_copies(
            COMGR_LIB_PREFIX,
            &[],
            &ld.to_string_lossy(),
            Some(&loader_cache_text),
            &[(known.clone(), "rocm-install", known.clone())],
            &std::collections::HashMap::new(),
        );
        let selected = select_loader_copy(&copies);

        assert_eq!(
            copies.first().map(|copy| &copy.source),
            Some(&"ld-library-path".to_owned()),
            "found: {copies:?}"
        );
        assert_eq!(
            selected.as_ref().map(|copy| &copy.source),
            Some(&"ld-library-path".to_owned()),
            "selected: {selected:?}"
        );
        assert_eq!(copies.len(), 3, "found: {copies:?}");

        let _ = std::fs::remove_dir_all(&ld);
        let _ = std::fs::remove_dir_all(&cached);
        let _ = std::fs::remove_dir_all(&known);
    }

    /// When the active runtime's own directory is *also* one of the generic
    /// managed-runtime search directories -- the normal case, since the
    /// active runtime is itself a managed runtime -- the copy is reported
    /// once, labelled `active-runtime`, not twice under both labels.
    ///
    /// This is the property `examine-finds-the-managed-runtimes-own-compilation-library`'s
    /// step relies on: it accepts either label as proof the CLI's own
    /// installed copy was found, specifically because a healthy host reports
    /// `active-runtime` here and `managed-runtime` is unreachable once an
    /// active runtime is present -- deduplication (keyed on the resolved
    /// path) drops the later, identical hit before it can be recorded a
    /// second time under the other source.
    #[cfg(unix)]
    #[test]
    fn an_active_runtime_directory_that_is_also_a_managed_root_keeps_one_label() {
        let dir = comgr_copy_dir("overlap");

        let copies = find_library_copies(
            COMGR_LIB_PREFIX,
            std::slice::from_ref(&dir),
            "",
            None,
            &[(dir.clone(), "managed-runtime", dir.clone())],
            &std::collections::HashMap::new(),
        );

        assert_eq!(
            copies.len(),
            1,
            "one file found through two sources is one copy, not two: {copies:?}"
        );
        assert_eq!(
            copies[0].source, "active-runtime",
            "the active-runtime hit comes first, so it keeps the label: {copies:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `find_library_copies` with no evidence anywhere reports no copies --
    /// `probe_comgr` is what turns that into the "no ... found" note, since it
    /// alone has the prefix's constant describing where it looked.
    #[test]
    fn no_copies_anywhere_is_an_empty_report() {
        assert!(
            find_library_copies(
                COMGR_LIB_PREFIX,
                &[],
                "",
                None,
                &[],
                &std::collections::HashMap::new(),
            )
            .is_empty()
        );
    }

    /// More than one copy produces the "N copies ... would load" note; exactly
    /// one copy, or none, produces no note at all.
    ///
    /// Unix-only: same `:`-split reason as above.
    #[cfg(unix)]
    #[test]
    fn the_multi_copy_note_only_fires_past_one_copy() {
        let only = comgr_copy_dir("only");
        let one_copy = find_library_copies(
            COMGR_LIB_PREFIX,
            &[],
            &only.to_string_lossy(),
            None,
            &[],
            &std::collections::HashMap::new(),
        );
        let one_selected = select_loader_copy(&one_copy);
        assert_eq!(one_copy.len(), 1);
        assert_eq!(
            copies_note(COMGR_LIB_PREFIX, &one_copy, one_selected.as_ref()),
            None,
            "a single copy must not be reported as a conflict: {one_copy:?}"
        );
        assert_eq!(
            copies_note(COMGR_LIB_PREFIX, &[], None),
            Some(format!(
                "no {COMGR_LIB_PREFIX} found on the library path, in the loader cache, or in any \
                 known ROCm install"
            )),
            "no copies is not a multi-copy conflict either, but it is still worth a note"
        );

        let second = comgr_copy_dir("second");
        let loader_cache_text = loader_cache_line_for(&second);
        let two_copies = find_library_copies(
            COMGR_LIB_PREFIX,
            &[],
            &only.to_string_lossy(),
            Some(&loader_cache_text),
            &[],
            &std::collections::HashMap::new(),
        );
        let two_selected = select_loader_copy(&two_copies);
        let note = copies_note(COMGR_LIB_PREFIX, &two_copies, two_selected.as_ref())
            .expect("more than one copy must be noted");
        assert!(
            note.contains("2 copies"),
            "the note must say how many copies were found: {note:?}"
        );
        assert!(
            note.contains(&two_copies[0].path),
            "the note must name the one that would load: {note:?}"
        );

        let _ = std::fs::remove_dir_all(&only);
        let _ = std::fs::remove_dir_all(&second);
    }

    /// No copies anywhere produces the "no libamd_comgr found" note, naming
    /// every place that was searched.
    #[test]
    fn no_copies_anywhere_names_every_place_searched() {
        let copies = find_library_copies(
            COMGR_LIB_PREFIX,
            &[],
            "",
            None,
            &[],
            &std::collections::HashMap::new(),
        );
        let selected = select_loader_copy(&copies);
        assert!(copies.is_empty());
        assert!(selected.is_none());
        let note = copies_note(COMGR_LIB_PREFIX, &copies, selected.as_ref())
            .expect("an empty search must still explain itself");
        assert!(note.contains("library path"), "{note:?}");
        assert!(note.contains("loader cache"), "{note:?}");
        assert!(note.contains("ROCm install"), "{note:?}");
    }

    /// A copy found only in a search directory -- `rocm-install` or
    /// `managed-runtime` -- is reported among the copies, but is *not*
    /// selected as the one that would load, because nothing puts a
    /// search-dir hit on a path the loader actually consults on its own.
    ///
    /// This pins the fix for exactly the bug this entry exists to catch: a
    /// leftover `/opt/rocm-*` install, or a managed runtime that is not the
    /// active one, used to be reported as "the library that loads" purely
    /// because it was `copies.first()` -- regardless of which tier the hit
    /// came from. A machine whose only copy sits in an inactive install has
    /// no library on the loader's path at all, and the report must say so,
    /// not point at a file nothing would ever load.
    ///
    /// Run once for comgr and once for the HIP runtime: both are searched by
    /// the same [`find_library_copies`] and both are selected by the same
    /// [`select_loader_copy`], so the rule is pinned for each independently
    /// rather than only for whichever one happened to have the bug first.
    ///
    /// Unix-only: same `:`-split reason as the neighbouring LD-path tests.
    #[cfg(unix)]
    #[test]
    fn a_search_dir_only_hit_is_not_selected_for_either_prefix() {
        for prefix in [COMGR_LIB_PREFIX, HIP_RUNTIME_LIB_PREFIX] {
            let leftover = library_copy_dir(prefix, &format!("leftover-install-{prefix}"));

            let copies = find_library_copies(
                prefix,
                &[],
                "",
                None,
                &[(leftover.clone(), "rocm-install", leftover.clone())],
                &std::collections::HashMap::new(),
            );
            let selected = select_loader_copy(&copies);

            assert_eq!(
                copies.len(),
                1,
                "the leftover copy must still be reported for {prefix}: {copies:?}"
            );
            assert_eq!(copies[0].source, "rocm-install");
            assert!(
                selected.is_none(),
                "a search-dir-only hit must not be named as the one that would load for \
                 {prefix}: {selected:?}"
            );
            let note = copies_note(prefix, &copies, selected.as_ref())
                .expect("a copy that cannot load still needs an explanation");
            assert!(
                !note.contains("would load"),
                "the note must not claim a search-dir-only copy would load for {prefix}: {note:?}"
            );

            let _ = std::fs::remove_dir_all(&leftover);
        }
    }

    /// The loader cache outranks a search directory, independent of any
    /// `ld-library-path` hit.
    ///
    /// `ld_library_path_outranks_loader_cache_and_search_dirs` above already
    /// orders all four tiers together, but a `LD_LIBRARY_PATH` hit sorts first
    /// regardless of whether the *loader-cache* loop or the *search-dir* loop
    /// runs first in the implementation -- so that test cannot tell the two
    /// loops apart. This test leaves `LD_LIBRARY_PATH` and the active-runtime
    /// dirs empty, so only the loader-cache-versus-search-dir order is in
    /// play: swapping those two loops in `find_library_copies` leaves every
    /// other existing test green and only this one fails.
    #[cfg(unix)]
    #[test]
    fn loader_cache_outranks_search_dirs() {
        let cached = comgr_copy_dir("cache-wins");
        let searched = comgr_copy_dir("search-dir-loses");
        let loader_cache_text = loader_cache_line_for(&cached);

        let copies = find_library_copies(
            COMGR_LIB_PREFIX,
            &[],
            "",
            Some(&loader_cache_text),
            &[(searched.clone(), "rocm-install", searched.clone())],
            &std::collections::HashMap::new(),
        );

        assert_eq!(
            copies.first().map(|copy| copy.source.as_str()),
            Some("loader-cache"),
            "the loader cache must be consulted ahead of the search directories: {copies:?}"
        );
        assert_eq!(copies.len(), 2, "found: {copies:?}");

        let _ = std::fs::remove_dir_all(&cached);
        let _ = std::fs::remove_dir_all(&searched);
    }

    /// `parse_ldconfig_cache_paths` reads the fixed `ldconfig -p` line format
    /// without needing a real `ldconfig` on the machine running the test.
    #[test]
    fn ldconfig_cache_parsing_reads_only_matching_lines() {
        let text = "2 libs found in cache `/etc/ld.so.cache'\n\
             \tlibamd_comgr.so.2 (libc6,x86-64) => /opt/rocm/lib/libamd_comgr.so.2\n\
             \tlibfoo.so.1 (libc6,x86-64) => /usr/lib/libfoo.so.1\n";
        let paths = parse_ldconfig_cache_paths(text, COMGR_LIB_PREFIX);
        assert_eq!(paths, vec!["/opt/rocm/lib/libamd_comgr.so.2".to_owned()]);
    }

    /// `parse_ldconfig_cache_paths` returns nothing when the cache lists no
    /// match, rather than panicking on a header line with no `=>`.
    #[test]
    fn ldconfig_cache_parsing_handles_no_match() {
        let text = "0 libs found in cache `/etc/ld.so.cache'\n";
        assert!(parse_ldconfig_cache_paths(text, COMGR_LIB_PREFIX).is_empty());
    }

    /// `rocm_install_siblings` finds every `rocm*`-named directory under the
    /// directory it is given, sorted, and ignores everything else -- without
    /// depending on what happens to live under the real `/opt` on the machine
    /// running the test.
    #[test]
    fn rocm_install_siblings_finds_only_rocm_named_dirs_sorted() {
        let opt = std::env::temp_dir().join(format!(
            "rocm-opt-siblings-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&opt);
        for name in ["rocm-6.4", "rocm-5.7", "not-rocm", "other"] {
            std::fs::create_dir_all(opt.join(name)).unwrap();
        }

        let found = rocm_install_siblings(&opt);
        assert_eq!(
            found,
            vec![opt.join("rocm-5.7"), opt.join("rocm-6.4")],
            "must list only `rocm`-prefixed directories, sorted: {found:?}"
        );

        let _ = std::fs::remove_dir_all(&opt);
    }

    /// A directory that does not exist (the common case: most hosts have no
    /// `/opt`) contributes nothing rather than panicking.
    #[test]
    fn rocm_install_siblings_tolerates_a_missing_directory() {
        let missing = std::env::temp_dir().join(format!(
            "rocm-opt-missing-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&missing);
        assert!(rocm_install_siblings(&missing).is_empty());
    }
}
