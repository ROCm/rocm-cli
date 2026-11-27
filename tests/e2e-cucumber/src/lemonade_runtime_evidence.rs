// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What a Lemonade runtime tree says about its llama.cpp backend, for a scenario
//! that found the backend in a state it did not expect (ROCMAI-1318).
//!
//! The runtime-33 install scenario expects the next install to reuse the pinned
//! backend. When it reinstalls instead, the install output alone cannot say
//! whether Lemonade did not list the backend as installed, the pin file moved, or
//! the build on disk is not the one the pin names. Each of those is one read of
//! the tree, so they are all taken at once and attached to the failure.
//!
//! Read-only apart from the `lemonade backends` listing, which is the same
//! command the engine runs to choose a backend. It runs in the harness's own
//! environment, not the engine's managed library environment, so a listing that
//! fails here is evidence about that environment and not about the engine.
//!
//! Nightly run 45950344128 showed the listing calling `llamacpp:rocm`
//! "installable ... not installed" on a tree whose
//! `bin/llamacpp/rocm-stable/version.txt` named the pinned build, which the pin,
//! build-marker and install-log files this module reads do not explain. So the
//! tree under `bin/llamacpp` is listed too, with sizes and, on Unix, modes, to
//! show whether the server binary is where the engine looks for it and can run,
//! and so are the `.rocm-cli-*.json` records the engine keeps in the runtime
//! directory. All of it is read after the install under test has run, so it
//! shows the state that install left, not the one it found.
//!
//! Never panics: it only runs while a scenario failure is being reported, and a
//! second panic would replace that report.

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use crate::serve_log::slugify;

/// Most bytes of one file or one output stream kept in the evidence.
const MAX_BYTES: u64 = 64 * 1024;
/// Time allowed for the `lemonade backends` listing. A bound so a hung listing
/// cannot stall the failure report; whether the listing touches the network is
/// not established here.
const BACKENDS_TIMEOUT: Duration = Duration::from_secs(30);
/// Levels of `bin/llamacpp` the tree lists: the entries of `bin/llamacpp` are
/// level 1, so a directory at level 4 is named and not opened. The engine looks
/// for the server in a backend directory or one build directory below it
/// (`find_binary_in` in `engines/lemonade`'s `llamacpp_backend.rs`), which is
/// levels 2 and 3; level 4 is listed as well so that a server extracted one
/// directory too deep is visible.
const MAX_TREE_DEPTH: usize = 4;
/// Most directories named in the whole tree; the rest are counted.
const MAX_TREE_DIRECTORIES: usize = 100;
/// Most files, links and other entries named in the whole tree, the always-listed
/// names aside; the rest are counted.
const MAX_TREE_FILES: usize = 300;
/// Most files, links and other entries of one directory named, the always-listed
/// names aside; the rest are counted.
const MAX_FILES_PER_DIRECTORY: usize = 40;
/// Most entries read from one directory before the rest are left unread.
const MAX_ENTRIES_READ_PER_DIRECTORY: usize = 2000;
/// Names that start with one of these are listed whatever the file caps, and do
/// not use them up: the server binary the engine looks for
/// (`find_llama_server_binary_for_backend` in `engines/lemonade`'s
/// `llamacpp_backend.rs`, by its `llama-server` name, `.exe` suffix included) and
/// the `version.txt` build marker this module already prints.
const ALWAYS_LISTED_PREFIXES: [&str; 2] = ["llama-server", "version.txt"];
/// Where the evidence is archived, under the suite's results directory.
const ARCHIVE_SUBDIR: &str = "lemonade-runtime-evidence";

/// The evidence for the Lemonade runtime tree at `runtime_dir`: the
/// `lemonade backends` listing, then the files and the directory tree that
/// decide what that listing says.
pub async fn collect(runtime_dir: &Path) -> String {
    let mut report = format!("Lemonade runtime tree: {}\n\n", runtime_dir.display());
    report.push_str(&lemonade_backends(runtime_dir, BACKENDS_TIMEOUT).await);
    report.push('\n');
    report.push_str(&files(runtime_dir));
    report
}

/// Write `evidence` under `results_dir`, which CI uploads, so it outlives the
/// scenario's temporary root. Returns where it went, or why it did not.
#[must_use]
pub fn archive(evidence: &str, results_dir: &Path, scenario: &str) -> String {
    let dir = results_dir.join(ARCHIVE_SUBDIR);
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return format!(
            "<not archived: failed to create {}: {error}>",
            dir.display()
        );
    }
    let name = format!("{}.txt", slugify(scenario));
    match std::fs::write(dir.join(&name), evidence) {
        Ok(()) => format!("archived to {ARCHIVE_SUBDIR}/{name}"),
        Err(error) => format!(
            "<not archived: failed to write {}: {error}>",
            dir.join(&name).display()
        ),
    }
}

fn lemonade_binary(runtime_dir: &Path) -> std::path::PathBuf {
    runtime_dir.join(if cfg!(windows) {
        "lemonade.exe"
    } else {
        "lemonade"
    })
}

async fn lemonade_backends(runtime_dir: &Path, timeout: Duration) -> String {
    let binary = lemonade_binary(runtime_dir);
    let mut report = format!(
        "== `{} backends` (run in the harness environment, not the engine's managed one) ==\n",
        binary.display()
    );
    let mut command = tokio::process::Command::new(&binary);
    command
        .arg("backends")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    match tokio::time::timeout(timeout, command.output()).await {
        Err(_) => {
            let _ = writeln!(report, "<did not finish within {timeout:?}; killed>");
        }
        Ok(Err(error)) => {
            let _ = writeln!(report, "<failed to run: {error}>");
        }
        Ok(Ok(output)) => {
            let _ = writeln!(report, "result: {}", output.status);
            for (stream, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
                let _ = writeln!(report, "-- {stream} --");
                report.push_str(&clamp(bytes));
                if !report.ends_with('\n') {
                    report.push('\n');
                }
            }
        }
    }
    report
}

/// `resources/backend_versions.json`, every `bin/llamacpp/rocm*/version.txt`,
/// the tree under `bin/llamacpp`, the `.rocm-cli-*.json` records and
/// `install-lemond.log`, each named by its path whether or not it exists.
fn files(runtime_dir: &Path) -> String {
    let mut report = String::new();
    section(
        &mut report,
        &runtime_dir.join("resources/backend_versions.json"),
    );
    let llamacpp = runtime_dir.join("bin").join("llamacpp");
    match std::fs::read_dir(&llamacpp) {
        Err(error) => {
            let _ = writeln!(
                report,
                "== {} ==\n<cannot list: {error}>\n",
                llamacpp.display()
            );
        }
        Ok(entries) => {
            let mut dirs: Vec<_> = entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("rocm"))
                .map(|entry| entry.path())
                .collect();
            dirs.sort();
            if dirs.is_empty() {
                let _ = writeln!(
                    report,
                    "== {}/rocm* ==\n<no such directory>\n",
                    llamacpp.display()
                );
            }
            for dir in dirs {
                section(&mut report, &dir.join("version.txt"));
            }
        }
    }
    report.push_str(&llamacpp_tree(runtime_dir));
    report.push_str(&records(runtime_dir));
    section(&mut report, &runtime_dir.join("install-lemond.log"));
    report
}

/// What is on disk under `bin/llamacpp`, one row per entry: its path from there,
/// its kind, and for a file its size and, on Unix, its mode. A link is named with
/// its target and not followed, so the walk does not leave the tree below
/// `bin/llamacpp`. No cap stops the walk, and each says in a row when it left
/// something out: [`MAX_TREE_DIRECTORIES`] directories and [`MAX_TREE_FILES`]
/// other entries in all, and [`MAX_FILES_PER_DIRECTORY`] other entries per
/// directory, each with how many; [`MAX_TREE_DEPTH`] levels and
/// [`MAX_ENTRIES_READ_PER_DIRECTORY`] reads per directory, each with that it did,
/// not how many. The names in [`ALWAYS_LISTED_PREFIXES`] are exempt from the file
/// caps.
fn llamacpp_tree(runtime_dir: &Path) -> String {
    let root = runtime_dir.join("bin").join("llamacpp");
    let mut tree = Tree {
        files_listed: 0,
        directories_listed: 0,
        report: format!(
            "== {} (tree, up to depth {MAX_TREE_DEPTH}) ==\n",
            root.display()
        ),
    };
    if let Err(error) = tree.walk(&root, Path::new(""), 1) {
        let _ = writeln!(tree.report, "<cannot list: {error}>");
    }
    tree.report.push('\n');
    tree.report
}

/// The state [`llamacpp_tree`] carries down the walk.
struct Tree {
    /// Files, links and other entries named so far, the always-listed ones not counted.
    files_listed: usize,
    /// Directories named so far.
    directories_listed: usize,
    report: String,
}

impl Tree {
    /// One level. `Err` is only for the directory it was asked to list; an entry
    /// that cannot be read is counted or written as such and the walk goes on.
    fn walk(&mut self, dir: &Path, shown_as: &Path, depth: usize) -> std::io::Result<()> {
        let mut names = Vec::new();
        let mut unreadable = 0_usize;
        let mut unread_rest = false;
        for entry in std::fs::read_dir(dir)? {
            if names.len() >= MAX_ENTRIES_READ_PER_DIRECTORY {
                unread_rest = true;
                break;
            }
            match entry {
                Ok(entry) => names.push(entry.file_name()),
                Err(_) => unreadable += 1,
            }
        }
        names.sort();
        let mut files_here = 0_usize;
        let mut files_skipped = 0_usize;
        let mut directories_skipped = 0_usize;
        for name in names {
            let path = dir.join(&name);
            let shown = shown_as.join(&name);
            let metadata = match std::fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) => {
                    let _ = writeln!(self.report, "{}  <cannot stat: {error}>", shown.display());
                    continue;
                }
            };
            let file_type = metadata.file_type();
            if file_type.is_dir() {
                if self.directories_listed >= MAX_TREE_DIRECTORIES {
                    directories_skipped += 1;
                    continue;
                }
                self.directories_listed += 1;
                if depth < MAX_TREE_DEPTH {
                    let _ = writeln!(self.report, "{}  dir", shown.display());
                    if let Err(error) = self.walk(&path, &shown, depth + 1) {
                        let _ =
                            writeln!(self.report, "{}  <cannot list: {error}>", shown.display());
                    }
                } else {
                    let _ = writeln!(self.report, "{}  dir  (not descended)", shown.display());
                }
                continue;
            }
            let always = ALWAYS_LISTED_PREFIXES
                .iter()
                .any(|prefix| name.to_string_lossy().starts_with(prefix));
            if !always {
                if files_here >= MAX_FILES_PER_DIRECTORY || self.files_listed >= MAX_TREE_FILES {
                    files_skipped += 1;
                    continue;
                }
                files_here += 1;
                self.files_listed += 1;
            }
            if file_type.is_file() {
                let _ = writeln!(
                    self.report,
                    "{}  file  {} bytes{}",
                    shown.display(),
                    metadata.len(),
                    mode_suffix(&metadata)
                );
            } else if file_type.is_symlink() {
                let target = std::fs::read_link(&path).map_or_else(
                    |error| format!("<unreadable: {error}>"),
                    |target| target.display().to_string(),
                );
                let _ = writeln!(self.report, "{}  link -> {target}", shown.display());
            } else {
                let _ = writeln!(self.report, "{}  other", shown.display());
            }
        }
        let label = if shown_as.as_os_str().is_empty() {
            ".".to_owned()
        } else {
            shown_as.display().to_string()
        };
        for (count, what) in [
            (files_skipped, "more files not listed"),
            (directories_skipped, "more directories not listed"),
            (unreadable, "entries could not be read"),
        ] {
            if count > 0 {
                let _ = writeln!(self.report, "{label}  <{count} {what}>");
            }
        }
        if unread_rest {
            let _ = writeln!(
                self.report,
                "{label}  <only the first {MAX_ENTRIES_READ_PER_DIRECTORY} entries were read>"
            );
        }
        Ok(())
    }
}

#[cfg(unix)]
fn mode_suffix(metadata: &std::fs::Metadata) -> String {
    use std::os::unix::fs::PermissionsExt as _;
    format!("  mode {:o}", metadata.permissions().mode() & 0o777)
}

#[cfg(not(unix))]
const fn mode_suffix(_metadata: &std::fs::Metadata) -> String {
    String::new()
}

/// Every file directly under `runtime_dir` whose name starts `.rocm-cli-` and
/// holds `.json`: the records the engine keeps there between installs (a failed
/// alignment writes `.rocm-cli-alignment-failed.json`, which the steps in
/// `tests/e2e/lemonade_install_steps.rs` hard-code as `ALIGNMENT_FAILURE_RECORD`).
/// The pattern is wider than that one name on purpose: a record renamed or added
/// by the engine is still attached, and so is a temporary file a crashed atomic
/// write left beside one (`rocm-core`'s `write_file_atomically`, in
/// `atomic_write.rs`, names it `<name>.tmp-<pid>-<millis>-<n>`). A link or
/// anything else that is not a regular file is named and not opened (see
/// [`read_clamped`]).
fn records(runtime_dir: &Path) -> String {
    let mut found = Vec::new();
    let mut unreadable = 0_usize;
    match std::fs::read_dir(runtime_dir) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(entry) => {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        if name.starts_with(".rocm-cli-") && name.contains(".json") {
                            found.push(entry.path());
                        }
                    }
                    Err(_) => unreadable += 1,
                }
            }
        }
        Err(error) => {
            return format!(
                "== {} (records) ==\n<cannot list: {error}>\n\n",
                runtime_dir.display()
            );
        }
    }
    found.sort();
    let mut report = String::new();
    if found.is_empty() {
        let _ = writeln!(
            report,
            "== {} ==\n<none>\n",
            runtime_dir.join(".rocm-cli-*.json*").display()
        );
    }
    for path in found {
        section(&mut report, &path);
    }
    if unreadable > 0 {
        let _ = writeln!(
            report,
            "== {} ==\n<{unreadable} entries could not be read>\n",
            runtime_dir.display()
        );
    }
    report
}

fn section(report: &mut String, path: &Path) {
    let _ = writeln!(report, "== {} ==", path.display());
    match read_clamped(path) {
        Ok(text) => report.push_str(&text),
        Err(error) => {
            let _ = write!(report, "<cannot read: {error}>");
        }
    }
    if !report.ends_with('\n') {
        report.push('\n');
    }
    report.push('\n');
}

/// The first [`MAX_BYTES`] of the regular file at `path`. Anything else, a link
/// included, is refused rather than opened, so the report cannot carry a file
/// from outside the tree and a pipe cannot stall it. The kind is read just before
/// the open, so a path swapped for a link in between is not guarded against.
fn read_clamped(path: &Path) -> std::io::Result<String> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err(std::io::Error::other("not a regular file: not opened"));
    }
    let mut body = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut body)?;
    Ok(clamp(&body))
}

fn clamp(bytes: &[u8]) -> String {
    let max = usize::try_from(MAX_BYTES).unwrap_or(usize::MAX);
    if bytes.len() <= max {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    format!(
        "{}\n<truncated at {MAX_BYTES} bytes>",
        String::from_utf8_lossy(&bytes[..max])
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, relative: &str, body: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn every_rocm_build_version_file_is_included_and_other_backends_are_not() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "resources/backend_versions.json",
            "{\"pin\":1}",
        );
        write(root.path(), "bin/llamacpp/rocm-stable/version.txt", "b1111");
        write(root.path(), "bin/llamacpp/rocm/version.txt", "b2222");
        write(root.path(), "bin/llamacpp/vulkan/version.txt", "b3333");
        write(root.path(), "install-lemond.log", "installed ok");
        let report = files(root.path());
        for expected in ["{\"pin\":1}", "b1111", "b2222", "installed ok"] {
            assert!(report.contains(expected), "missing `{expected}`:\n{report}");
        }
        assert!(!report.contains("b3333"), "vulkan is not ROCm:\n{report}");
    }

    /// Whether Lemonade lists a backend as installed presumably depends on what
    /// it finds on disk (not established), so the evidence shows the disk: where
    /// the server binary sits and how big it is.
    #[test]
    fn the_backend_tree_names_the_server_binary_where_it_sits() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "bin/llamacpp/rocm-stable/version.txt", "b1111");
        write(
            root.path(),
            "bin/llamacpp/rocm-stable/build-b1111/llama-server",
            "binary",
        );
        let report = files(root.path());
        let server = Path::new("rocm-stable")
            .join("build-b1111")
            .join("llama-server");
        assert!(
            report.contains(&format!("{}  file  6 bytes", server.display())),
            "{report}"
        );
        assert!(report.contains("(tree, up to depth"), "{report}");
    }

    /// A directory left behind by an alignment that was killed half way, a build
    /// directory with no server in it, and a backend other than ROCm are each a
    /// row of the tree, so none of them has to be inferred from the files read
    /// below it.
    #[test]
    fn the_backend_tree_shows_leftovers_and_empty_build_directories() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "bin/llamacpp/rocm-stable/version.txt", "b1111");
        std::fs::create_dir_all(root.path().join("bin/llamacpp/rocm-stable/build-b2222")).unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/rocm-stable.pre-alignment"))
            .unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/vulkan")).unwrap();
        let report = files(root.path());
        for expected in [
            "rocm-stable.pre-alignment  dir",
            "vulkan  dir",
            "build-b2222  dir",
        ] {
            assert!(report.contains(expected), "missing `{expected}`:\n{report}");
        }
    }

    #[test]
    fn a_missing_backend_tree_is_named_in_its_own_section() {
        let root = tempfile::tempdir().unwrap();
        let report = files(root.path());
        assert!(report.contains("(tree, up to depth"), "{report}");
        assert!(report.matches("<cannot list:").count() >= 2, "{report}");
    }

    /// Past the per-directory cap the files are counted, not listed. The server
    /// and version files sort before the libraries here and are exempt from the
    /// cap without using it up, so exactly the cap's worth of libraries follow
    /// them.
    #[test]
    fn a_crowded_directory_lists_the_cap_counts_the_rest_and_keeps_the_exempt_names() {
        let root = tempfile::tempdir().unwrap();
        for index in 0..(MAX_FILES_PER_DIRECTORY + 25) {
            write(
                root.path(),
                &format!("bin/llamacpp/rocm-stable/zlib{index:04}.so"),
                "x",
            );
        }
        write(root.path(), "bin/llamacpp/rocm-stable/llama-server", "bin");
        write(root.path(), "bin/llamacpp/rocm-stable/version.txt", "b1111");
        let tree = llamacpp_tree(root.path());
        assert!(tree.contains("llama-server  file  3 bytes"), "{tree}");
        assert!(tree.contains("version.txt  file  5 bytes"), "{tree}");
        let last_listed = format!("zlib{:04}.so", MAX_FILES_PER_DIRECTORY - 1);
        let first_unlisted = format!("zlib{MAX_FILES_PER_DIRECTORY:04}.so");
        assert!(tree.contains(&last_listed), "{tree}");
        assert!(!tree.contains(&first_unlisted), "{tree}");
        assert_eq!(
            tree.matches("zlib").count(),
            MAX_FILES_PER_DIRECTORY,
            "{tree}"
        );
        assert!(
            tree.contains("rocm-stable  <25 more files not listed>"),
            "{tree}"
        );
    }

    /// The whole-tree file cap counts what it leaves out and never takes the
    /// server binary or the build marker of a directory that comes later in the
    /// walk, and nothing is said twice.
    #[test]
    fn the_whole_tree_file_cap_spares_the_exempt_names_of_later_directories() {
        let root = tempfile::tempdir().unwrap();
        let backends = ["rocm", "rocm-nightly", "rocm-stable", "vulkan"];
        for backend in backends {
            for index in 0..MAX_FILES_PER_DIRECTORY {
                write(
                    root.path(),
                    &format!("bin/llamacpp/{backend}/lib{index:04}.so"),
                    "x",
                );
            }
            write(
                root.path(),
                &format!("bin/llamacpp/{backend}/llama-server"),
                "bin",
            );
            write(
                root.path(),
                &format!("bin/llamacpp/{backend}/version.txt"),
                "b1",
            );
        }
        // Four directories of 40 libraries is 160 files, under the 300 cap.
        assert!(!llamacpp_tree(root.path()).contains("not listed"));
        // Eight more directories take the total past it.
        for index in 0..8 {
            for lib in 0..MAX_FILES_PER_DIRECTORY {
                write(
                    root.path(),
                    &format!("bin/llamacpp/zz{index}/lib{lib:04}.so"),
                    "x",
                );
            }
            write(
                root.path(),
                &format!("bin/llamacpp/zz{index}/llama-server"),
                "bin",
            );
            write(
                root.path(),
                &format!("bin/llamacpp/zz{index}/version.txt"),
                "b1",
            );
        }
        let tree = llamacpp_tree(root.path());
        assert_eq!(tree.matches("llama-server").count(), 12, "{tree}");
        assert_eq!(tree.matches("version.txt").count(), 12, "{tree}");
        assert_eq!(tree.matches(".so").count(), MAX_TREE_FILES, "{tree}");
        // Twelve directories of 40 libraries against a cap of 300: seven are
        // listed whole, the eighth in part, and the last four not at all.
        assert_eq!(tree.matches("more files not listed").count(), 5, "{tree}");
    }

    #[test]
    fn directories_past_the_cap_are_counted_once() {
        let root = tempfile::tempdir().unwrap();
        for index in 0..(MAX_TREE_DIRECTORIES + 5) {
            std::fs::create_dir_all(root.path().join(format!("bin/llamacpp/d{index:04}"))).unwrap();
        }
        let tree = llamacpp_tree(root.path());
        assert_eq!(
            tree.matches("  dir").count(),
            MAX_TREE_DIRECTORIES,
            "{tree}"
        );
        assert_eq!(
            tree.matches("<5 more directories not listed>").count(),
            1,
            "{tree}"
        );
    }

    /// Entries of the fourth level are named, and a directory there is not opened.
    #[test]
    fn the_depth_cap_names_a_directory_without_opening_it() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "bin/llamacpp/a/b/c/d/e/deep.txt", "x");
        write(root.path(), "bin/llamacpp/a/b/c/shallow.txt", "x");
        let tree = llamacpp_tree(root.path());
        let fourth_level = Path::new("a").join("b").join("c").join("d");
        assert!(
            tree.contains(&format!("{}  dir  (not descended)", fourth_level.display())),
            "{tree}"
        );
        assert!(tree.contains("shallow.txt"), "{tree}");
        assert!(!tree.contains("deep.txt"), "{tree}");
    }

    #[test]
    fn a_directory_with_more_entries_than_are_read_says_so() {
        let root = tempfile::tempdir().unwrap();
        for index in 0..(MAX_ENTRIES_READ_PER_DIRECTORY + 3) {
            write(root.path(), &format!("bin/llamacpp/f{index:05}"), "");
        }
        let tree = llamacpp_tree(root.path());
        assert!(
            tree.contains(&format!(
                ".  <only the first {MAX_ENTRIES_READ_PER_DIRECTORY} entries were read>"
            )),
            "{tree}"
        );
    }

    /// The tree is read-only and bounded: a link is named with its target and not
    /// followed, so a link back up the tree cannot make the walk run away.
    #[cfg(unix)]
    #[test]
    fn a_link_is_named_with_its_target_and_not_followed() {
        let root = tempfile::tempdir().unwrap();
        let llamacpp = root.path().join("bin/llamacpp");
        std::fs::create_dir_all(llamacpp.join("rocm-stable")).unwrap();
        std::os::unix::fs::symlink(&llamacpp, llamacpp.join("rocm-stable/loop")).unwrap();
        let tree = llamacpp_tree(root.path());
        assert!(
            tree.contains(&format!("loop  link -> {}", llamacpp.display())),
            "{tree}"
        );
        assert!(!tree.contains("loop/rocm-stable"), "{tree}");
    }

    /// A server that was extracted without its execute bit is the kind of thing the
    /// tree has to show, so the mode of each file is part of its row.
    #[cfg(unix)]
    #[test]
    fn each_files_mode_is_part_of_its_row() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = tempfile::tempdir().unwrap();
        for (build, mode) in [("runs", 0o755), ("stuck", 0o644)] {
            let server = root
                .path()
                .join(format!("bin/llamacpp/rocm-stable/{build}/llama-server"));
            write(
                root.path(),
                &format!("bin/llamacpp/rocm-stable/{build}/llama-server"),
                "bin",
            );
            std::fs::set_permissions(&server, std::fs::Permissions::from_mode(mode)).unwrap();
        }
        let tree = llamacpp_tree(root.path());
        assert!(
            tree.contains(&format!(
                "{}  file  3 bytes  mode 755",
                Path::new("rocm-stable")
                    .join("runs")
                    .join("llama-server")
                    .display()
            )),
            "{tree}"
        );
        assert!(
            tree.contains(&format!(
                "{}  file  3 bytes  mode 644",
                Path::new("rocm-stable")
                    .join("stuck")
                    .join("llama-server")
                    .display()
            )),
            "{tree}"
        );
    }

    /// A failed alignment writes a record under the runtime directory, and whether
    /// it is there and what it says decide whether the next install skips the
    /// alignment, so every `.rocm-cli-*.json` is attached, and so is the temporary
    /// file a crashed atomic write leaves beside one.
    #[test]
    fn the_records_the_engine_keeps_in_the_runtime_directory_are_attached() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            ".rocm-cli-alignment-failed.json",
            "{\"target_rocm_version\":\"7.99.0\"}",
        );
        write(
            root.path(),
            ".rocm-cli-alignment-failed.json.tmp-12-34-0",
            "half written",
        );
        write(root.path(), ".rocm-cli-notes.txt", "not a record");
        write(root.path(), "unrelated.json", "{\"other\":1}");
        let report = files(root.path());
        assert!(
            report.contains(".rocm-cli-alignment-failed.json =="),
            "{report}"
        );
        assert!(
            report.contains("{\"target_rocm_version\":\"7.99.0\"}"),
            "{report}"
        );
        assert!(report.contains("half written"), "{report}");
        assert!(!report.contains("not a record"), "{report}");
        assert!(!report.contains("{\"other\":1}"), "{report}");
    }

    #[test]
    fn no_record_is_said_rather_than_left_out() {
        let root = tempfile::tempdir().unwrap();
        let report = files(root.path());
        assert!(report.contains(".rocm-cli-*.json*"), "{report}");
        assert!(report.contains("<none>"), "{report}");
    }

    /// A record that is a link is named and not opened: what it points at is
    /// outside the tree, and the report must not carry it.
    #[cfg(unix)]
    #[test]
    fn a_record_that_is_a_link_is_named_and_not_opened() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        write(elsewhere.path(), "secret.txt", "outside the tree");
        std::os::unix::fs::symlink(
            elsewhere.path().join("secret.txt"),
            root.path().join(".rocm-cli-alignment-failed.json"),
        )
        .unwrap();
        let report = records(root.path());
        assert!(
            report.contains("not a regular file: not opened"),
            "{report}"
        );
        assert!(!report.contains("outside the tree"), "{report}");
    }

    #[test]
    fn a_missing_file_is_named_not_skipped() {
        let root = tempfile::tempdir().unwrap();
        let report = files(root.path());
        for name in ["backend_versions.json", "install-lemond.log"] {
            assert!(report.contains(name), "`{name}` not named:\n{report}");
        }
        assert!(report.contains("<cannot read:"), "{report}");
        assert!(report.contains("<cannot list:"), "{report}");
    }

    #[test]
    fn a_llamacpp_directory_with_no_rocm_build_says_so() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/vulkan")).unwrap();
        let report = files(root.path());
        assert!(
            report.contains("rocm*/ ==") || report.contains("/rocm* =="),
            "{report}"
        );
        assert!(report.contains("<no such directory>"), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_listing_that_outlives_its_timeout_is_killed_and_reported() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "lemonade", "#!/bin/sh\nexec sleep 30\n");
        let binary = root.path().join("lemonade");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let started = std::time::Instant::now();
        let report = lemonade_backends(root.path(), Duration::from_millis(200)).await;
        assert!(report.contains("did not finish within"), "{report}");
        assert!(started.elapsed() < Duration::from_secs(10), "not bounded");
    }

    #[test]
    fn a_rocm_build_directory_without_a_version_file_is_still_named() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin/llamacpp/rocm-nightly")).unwrap();
        let report = files(root.path());
        // Separators differ by platform, so the directory and the file are asserted apart.
        assert!(report.contains("rocm-nightly"), "{report}");
        assert!(report.contains("version.txt"), "{report}");
        assert!(report.contains("<cannot read:"), "{report}");
    }

    #[test]
    fn a_large_file_is_truncated_and_says_so() {
        let root = tempfile::tempdir().unwrap();
        let big = "x".repeat(usize::try_from(MAX_BYTES).unwrap() + 10);
        write(root.path(), "install-lemond.log", &big);
        let report = files(root.path());
        assert!(report.contains("<truncated at"), "{report}");
        assert!(report.len() < big.len() + 1024);
    }

    #[tokio::test]
    async fn a_missing_lemonade_binary_is_reported_and_the_files_still_follow() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "install-lemond.log", "log body");
        let report = collect(root.path()).await;
        assert!(report.contains("backends` ("), "{report}");
        assert!(report.contains("<failed to run:"), "{report}");
        assert!(report.contains("log body"), "{report}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_backends_listing_is_captured_with_its_exit_status() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "lemonade",
            "#!/bin/sh\necho \"args: $*\"\necho warn >&2\nexit 3\n",
        );
        let binary = root.path().join("lemonade");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let report = lemonade_backends(root.path(), BACKENDS_TIMEOUT).await;
        assert!(report.contains("args: backends"), "{report}");
        assert!(report.contains("warn"), "{report}");
        assert!(report.contains("result: exit status: 3"), "{report}");
    }

    #[test]
    fn the_archive_lands_under_the_results_directory() {
        let results = tempfile::tempdir().unwrap();
        let outcome = archive("evidence body", results.path(), "runtime-33 - A scenario");
        assert_eq!(
            outcome,
            "archived to lemonade-runtime-evidence/runtime-33-a-scenario.txt"
        );
        let body = std::fs::read_to_string(
            results
                .path()
                .join("lemonade-runtime-evidence/runtime-33-a-scenario.txt"),
        )
        .unwrap();
        assert_eq!(body, "evidence body");
    }
}
