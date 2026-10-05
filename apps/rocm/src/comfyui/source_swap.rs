// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Replacing ComfyUI's code in an existing `source/` folder without touching
//! the user's content there.
//!
//! `rocm comfyui start` runs ComfyUI from `source/` without
//! `--user-directory`, `--output-directory` or `--input-directory`, so ComfyUI
//! keeps the user's models, saved workflows, outputs and uploads inside it.
//! A reinstall therefore swaps the *code* around that content and never moves,
//! copies or deletes anything under a [`PRESERVED_SOURCE_ENTRIES`] entry. The
//! code swap is not atomic, but code can be re-fetched: a marker file records
//! an unfinished swap, and running the swap again finishes it.

use anyhow::{Context, Result, bail};
use std::fs;
use std::io;
use std::path::Path;

/// Entries of the ComfyUI `source/` folder that hold the user's own content
/// and that a reinstall leaves in place.
///
/// `models` (the folder `rocm comfyui models-path` reports), `user` (saved
/// workflows and settings), `output` (generated images; ComfyUI also saves
/// models there), `input` (uploaded images), `custom_nodes` (third-party nodes
/// the user installed, often with their own settings or weights inside) and
/// `extra_model_paths.yaml` (user-written; the release ships only the
/// `.example`). `temp/` is deliberately absent: it holds transient previews
/// that ComfyUI treats as disposable and by default clears on startup.
pub(super) const PRESERVED_SOURCE_ENTRIES: &[&str] = &[
    "models",
    "user",
    "output",
    "input",
    "custom_nodes",
    "extra_model_paths.yaml",
];

/// Written into `source/` before the code swap starts and removed when it
/// finishes. Its presence means a swap was interrupted; its contents are the
/// [`PRESERVED_SOURCE_ENTRIES`] that existed before that swap began, so a
/// re-run reports what was really the user's rather than what the interrupted
/// run may already have added from the release.
pub(super) const SWAP_MARKER: &str = ".rocm-cli-reinstall-in-progress";

/// The [`PRESERVED_SOURCE_ENTRIES`] `source` holds now, in their listed
/// order — what a reinstall started now would leave in place.
pub(super) fn preserved_entries_present(source: &Path) -> Vec<&'static str> {
    PRESERVED_SOURCE_ENTRIES
        .iter()
        .copied()
        .filter(|name| fs::symlink_metadata(source.join(name)).is_ok())
        .collect()
}

/// Whether a code swap into `source` was started and did not finish.
pub(super) fn swap_interrupted(source: &Path) -> bool {
    fs::symlink_metadata(source.join(SWAP_MARKER)).is_ok()
}

/// Installs the unpacked release at `new_tree` into `source`, returning the
/// [`PRESERVED_SOURCE_ENTRIES`] that `source` already held and that were left
/// in place.
///
/// Without an existing `source` the release tree is simply moved there. With
/// one, every top-level entry of `source` that is not preserved is the
/// previous release's code and is removed, the release's own top-level entries
/// are moved in, and inside each preserved folder the release only adds what
/// is missing (see [`add_missing_entries`]). The user's entries are never
/// moved, copied, overwritten or written through when they are symlinks.
pub(super) fn install_release_tree(new_tree: &Path, source: &Path) -> Result<Vec<&'static str>> {
    install_release_tree_with(new_tree, source, &mut |_| Ok(()))
}

/// [`install_release_tree`] with `step` called before every filesystem
/// change, so a test can stop the swap at any point the way a crash or a
/// killed process would: without any cleanup running afterwards.
fn install_release_tree_with(
    new_tree: &Path,
    source: &Path,
    step: &mut dyn FnMut(&Path) -> io::Result<()>,
) -> Result<Vec<&'static str>> {
    if fs::symlink_metadata(source).is_err() {
        let parent = source
            .parent()
            .context("ComfyUI source path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        move_release_entry(new_tree, source, step)?;
        return Ok(Vec::new());
    }
    if !source.is_dir() {
        bail!("{} exists but is not a folder", source.display());
    }

    let marker = source.join(SWAP_MARKER);
    let kept = if swap_interrupted(source) {
        read_marker(&marker)?
    } else {
        let kept = preserved_entries_present(source);
        step(&marker)?;
        fs::write(&marker, kept.join("\n"))
            .with_context(|| format!("failed to write {}", marker.display()))?;
        kept
    };

    // The previous release's code: everything at the top level that is
    // neither preserved nor the marker. After an interrupted run this also
    // clears whatever part of the new code had already been moved in.
    for entry in read_dir(source)? {
        let name = entry.file_name();
        if is_preserved(&name) || name == SWAP_MARKER {
            continue;
        }
        let path = entry.path();
        step(&path)?;
        remove_path(&path)?;
    }

    for entry in read_dir(new_tree)? {
        let name = entry.file_name();
        let target = source.join(&name);
        if is_preserved(&name) {
            if fs::symlink_metadata(&target).is_err() {
                move_release_entry(&entry.path(), &target, step)?;
            } else {
                add_missing_entries(&entry.path(), &target, step)?;
            }
        } else if name != SWAP_MARKER {
            move_release_entry(&entry.path(), &target, step)?;
        }
    }

    step(&marker)?;
    fs::remove_file(&marker).with_context(|| format!("failed to remove {}", marker.display()))?;
    Ok(kept)
}

fn is_preserved(name: &std::ffi::OsStr) -> bool {
    PRESERVED_SOURCE_ENTRIES
        .iter()
        .any(|preserved| name == *preserved)
}

fn read_dir(path: &Path) -> Result<Vec<fs::DirEntry>> {
    fs::read_dir(path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .collect::<io::Result<Vec<_>>>()
        .with_context(|| format!("failed to read {}", path.display()))
}

fn read_marker(marker: &Path) -> Result<Vec<&'static str>> {
    let text = fs::read_to_string(marker)
        .with_context(|| format!("failed to read {}", marker.display()))?;
    Ok(PRESERVED_SOURCE_ENTRIES
        .iter()
        .copied()
        .filter(|name| text.lines().any(|line| line == *name))
        .collect())
}

/// Moves each release entry under `release` that `into` lacks, recursing
/// where both hold a real (not symlinked) directory of the same name. Entries
/// `into` already has are left alone, so the user's copy wins, and a symlinked
/// folder of the user's is never written into: a model folder new in a
/// release appears, but nothing of the user's changes.
fn add_missing_entries(
    release: &Path,
    into: &Path,
    step: &mut dyn FnMut(&Path) -> io::Result<()>,
) -> Result<()> {
    if !is_real_dir(release) || !is_real_dir(into) {
        return Ok(());
    }
    for entry in read_dir(release)? {
        let target = into.join(entry.file_name());
        if fs::symlink_metadata(&target).is_err() {
            move_release_entry(&entry.path(), &target, step)?;
        } else {
            add_missing_entries(&entry.path(), &target, step)?;
        }
    }
    Ok(())
}

fn is_real_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_dir())
}

/// Moves a release entry into `source`. Only release content goes through
/// here — never the user's. A rename is all it takes unless the staging folder
/// and `source` are on different filesystems (`source` itself symlinked
/// elsewhere); only then is the entry copied, without following symlinks.
fn move_release_entry(
    from: &Path,
    to: &Path,
    step: &mut dyn FnMut(&Path) -> io::Result<()>,
) -> Result<()> {
    step(to)?;
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::CrossesDevices => {
            if let Err(copy_error) = copy_tree_no_follow(from, to) {
                remove_path(to).ok();
                return Err(copy_error);
            }
            remove_path(from)
        }
        Err(error) => Err(error)
            .with_context(|| format!("failed to move {} to {}", from.display(), to.display())),
    }
}

/// Copies `from` to `to`, recreating symlinks as symlinks instead of copying
/// what they point at.
fn copy_tree_no_follow(from: &Path, to: &Path) -> Result<()> {
    let file_type = fs::symlink_metadata(from)
        .with_context(|| format!("failed to stat {}", from.display()))?
        .file_type();
    if file_type.is_symlink() {
        let target =
            fs::read_link(from).with_context(|| format!("failed to read {}", from.display()))?;
        return create_symlink(&target, from, to)
            .with_context(|| format!("failed to recreate the symlink {}", to.display()));
    }
    if file_type.is_dir() {
        fs::create_dir(to).with_context(|| format!("failed to create {}", to.display()))?;
        for entry in read_dir(from)? {
            copy_tree_no_follow(&entry.path(), &to.join(entry.file_name()))?;
        }
        return Ok(());
    }
    fs::copy(from, to)
        .map(|_| ())
        .with_context(|| format!("failed to copy {} to {}", from.display(), to.display()))
}

#[cfg(unix)]
fn create_symlink(target: &Path, _original: &Path, link: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn create_symlink(target: &Path, original: &Path, link: &Path) -> io::Result<()> {
    if fs::metadata(original).is_ok_and(|metadata| metadata.is_dir()) {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    }
}

/// Removes a file, symlink or directory tree; a missing path is not an error.
/// A symlink is removed itself, never what it points at.
fn remove_path(path: &Path) -> Result<()> {
    let file_type = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata.file_type(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to stat {}", path.display()));
        }
    };
    let removed = if file_type.is_dir() {
        fs::remove_dir_all(path)
    } else {
        // A directory symlink on Windows is removed with `remove_dir`.
        fs::remove_file(path).or_else(|error| fs::remove_dir(path).map_err(|_| error))
    };
    removed.with_context(|| format!("failed to remove {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "rocm-cli-comfyui-swap-{name}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&root).expect("create test root");
            Self(root)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write(root: &Path, files: &[(&str, &str)]) {
        for (relative, contents) in files {
            let path = root.join(relative);
            fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
            fs::write(&path, contents).expect("write fixture");
        }
    }

    fn read(path: &Path) -> Option<String> {
        fs::read_to_string(path).ok()
    }

    /// The user's content: one file in every preserved entry, plus the
    /// user's own copy of a file the release also ships.
    const USER_FILES: [(&str, &str); 7] = [
        ("models/checkpoints/my-model.safetensors", "user model"),
        ("user/default/workflows/my-workflow.json", "user workflow"),
        ("output/ComfyUI_00001_.png", "user image"),
        ("input/my-upload.png", "user upload"),
        ("custom_nodes/my-node/__init__.py", "user node"),
        ("extra_model_paths.yaml", "user model paths"),
        (
            "custom_nodes/websocket_image_save.py",
            "user's edited sample",
        ),
    ];

    const OLD_RELEASE: [(&str, &str); 4] = [
        ("main.py", "old main"),
        ("comfy/dropped_upstream.py", "old code"),
        ("temp/preview.png", "scratch"),
        ("models/checkpoints/put_checkpoints_here", ""),
    ];

    const NEW_RELEASE: [(&str, &str); 7] = [
        ("main.py", "new main"),
        ("comfy/added_upstream.py", "new code"),
        ("models/checkpoints/put_checkpoints_here", ""),
        ("models/new_kind/put_new_kind_here", ""),
        ("custom_nodes/websocket_image_save.py", "new sample"),
        ("input/example.png", "release example"),
        ("output/_output_images_will_be_put_here", ""),
    ];

    fn used_source(root: &Path) -> PathBuf {
        let source = root.join("source");
        write(&source, &OLD_RELEASE);
        write(&source, &USER_FILES);
        source
    }

    fn new_release(root: &Path) -> PathBuf {
        let tree = root.join("extract").join("ComfyUI-master");
        write(&tree, &NEW_RELEASE);
        tree
    }

    fn assert_user_files_intact(source: &Path) {
        for (relative, contents) in USER_FILES {
            assert_eq!(
                read(&source.join(relative)).as_deref(),
                Some(contents),
                "the user's {relative} must be left as it was"
            );
        }
    }

    fn assert_new_release_installed(source: &Path) {
        assert_eq!(read(&source.join("main.py")).as_deref(), Some("new main"));
        assert!(source.join("comfy/added_upstream.py").is_file());
        assert!(
            !source.join("comfy/dropped_upstream.py").exists(),
            "code the new release no longer ships must not linger"
        );
        assert!(!source.join("temp").exists(), "temp/ is not kept");
        assert!(source.join("models/new_kind/put_new_kind_here").is_file());
        assert!(source.join("input/example.png").is_file());
        assert!(
            !source.join(SWAP_MARKER).exists(),
            "a finished swap leaves no marker"
        );
    }

    #[test]
    fn swap_replaces_code_and_leaves_user_content_in_place() -> Result<()> {
        let dir = TestDir::new("basic");
        let source = used_source(&dir.0);
        let tree = new_release(&dir.0);

        let kept = install_release_tree(&tree, &source)?;

        assert_new_release_installed(&source);
        assert_user_files_intact(&source);
        assert_eq!(kept, PRESERVED_SOURCE_ENTRIES);
        Ok(())
    }

    #[test]
    fn kept_lists_only_entries_the_install_already_had() -> Result<()> {
        let dir = TestDir::new("kept-subset");
        let source = dir.0.join("source");
        write(
            &source,
            &[
                ("main.py", "old main"),
                ("models/mine.safetensors", "user model"),
            ],
        );
        let tree = new_release(&dir.0);

        let kept = install_release_tree(&tree, &source)?;

        // The release ships custom_nodes/, input/ and output/; they are now
        // there, but they were not the user's, so they were not "kept".
        assert_eq!(kept, ["models"]);
        assert!(
            source
                .join("output/_output_images_will_be_put_here")
                .is_file()
        );
        assert_eq!(
            read(&source.join("models/mine.safetensors")).as_deref(),
            Some("user model")
        );
        Ok(())
    }

    #[test]
    fn fresh_install_moves_the_release_into_place() -> Result<()> {
        let dir = TestDir::new("fresh");
        let source = dir.0.join("app").join("source");
        let tree = new_release(&dir.0);

        let kept = install_release_tree(&tree, &source)?;

        assert!(kept.is_empty());
        assert_eq!(read(&source.join("main.py")).as_deref(), Some("new main"));
        assert!(!tree.exists());
        Ok(())
    }

    /// Stops the swap before each filesystem change in turn — the way a crash
    /// or a kill would, with nothing cleaning up after it — then runs the swap
    /// again from a freshly unpacked release, as `--reinstall` would. Every
    /// interruption point must converge on a complete tree, with the user's
    /// files never touched in between.
    #[test]
    fn interrupted_swap_converges_when_run_again() -> Result<()> {
        let mut interrupted_at = 0;
        loop {
            let dir = TestDir::new("interrupt");
            let source = used_source(&dir.0);
            let tree = new_release(&dir.0);
            let mut steps = 0;
            let first = install_release_tree_with(&tree, &source, &mut |_| {
                steps += 1;
                if steps > interrupted_at {
                    Err(io::Error::other("interrupted"))
                } else {
                    Ok(())
                }
            });
            if first.is_ok() {
                assert!(interrupted_at > 0, "the swap took no steps");
                break;
            }
            assert_user_files_intact(&source);
            let left_marker = swap_interrupted(&source);

            let rerun_tree = dir.0.join("extract-2").join("ComfyUI-master");
            write(&rerun_tree, &NEW_RELEASE);
            let kept = install_release_tree(&rerun_tree, &source)?;

            assert_user_files_intact(&source);
            assert_new_release_installed(&source);
            assert_eq!(
                kept, PRESERVED_SOURCE_ENTRIES,
                "after an interruption at step {interrupted_at}, kept must still name what the user had"
            );
            assert!(
                left_marker || interrupted_at == 0,
                "an interrupted swap must leave its marker (step {interrupted_at})"
            );
            interrupted_at += 1;
        }
        Ok(())
    }

    /// The marker records what the user had before the swap began, so a
    /// re-run does not report as "kept" a folder the interrupted run added.
    #[test]
    fn rerun_reports_what_the_user_had_not_what_the_release_added() -> Result<()> {
        let dir = TestDir::new("interrupt-kept");
        let source = dir.0.join("source");
        write(
            &source,
            &[("main.py", "old main"), ("models/mine.ckpt", "m")],
        );
        let tree = new_release(&dir.0);
        // Let the whole swap run except removing the marker.
        let marker = source.join(SWAP_MARKER);
        let interrupted = install_release_tree_with(&tree, &source, &mut |path| {
            if path == marker && marker.exists() {
                Err(io::Error::other("interrupted"))
            } else {
                Ok(())
            }
        });
        assert!(interrupted.is_err());
        assert!(source.join("output").exists(), "the release added output/");

        let rerun_tree = dir.0.join("extract-2").join("ComfyUI-master");
        write(&rerun_tree, &NEW_RELEASE);
        let kept = install_release_tree(&rerun_tree, &source)?;

        assert_eq!(kept, ["models"]);
        Ok(())
    }

    /// The user's files keep their inode and mtime, and neither they, the
    /// preserved folders nor `source/` itself see a ctime change — which a
    /// rename would cause. So nothing of the user's was moved, not merely
    /// moved back.
    #[cfg(unix)]
    #[test]
    fn user_content_is_never_moved_or_rewritten() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        let dir = TestDir::new("inode");
        let source = used_source(&dir.0);
        let tree = new_release(&dir.0);
        let paths: Vec<PathBuf> = std::iter::once(source.clone())
            .chain(USER_FILES.iter().map(|(relative, _)| source.join(relative)))
            .chain(
                PRESERVED_SOURCE_ENTRIES
                    .iter()
                    .map(|name| source.join(name)),
            )
            .collect();
        let identity = |path: &PathBuf| {
            let metadata = fs::symlink_metadata(path).expect("user path");
            (
                metadata.ino(),
                metadata.mtime(),
                metadata.mtime_nsec(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            )
        };
        let before: Vec<_> = paths.iter().map(identity).collect();
        // Let a ctime change, if one happened, land on a different tick.
        std::thread::sleep(std::time::Duration::from_millis(20));

        install_release_tree(&tree, &source)?;

        for (path, before) in paths.iter().zip(before) {
            let after = identity(path);
            if path == &source
                || PRESERVED_SOURCE_ENTRIES
                    .iter()
                    .any(|name| path.ends_with(name))
            {
                // Folders the release adds into get a new mtime/ctime from
                // that; only the inode shows whether they were moved.
                assert_eq!(before.0, after.0, "{} was replaced", path.display());
            } else {
                assert_eq!(before, after, "{} was moved or rewritten", path.display());
            }
        }
        assert_user_files_intact(&source);
        Ok(())
    }

    /// `models` is a symlink to a folder elsewhere (a bigger disk). It stays a
    /// symlink, the folder behind it is not written into, and nothing in it
    /// is copied or removed.
    #[cfg(unix)]
    #[test]
    fn symlinked_models_folder_is_left_alone() -> Result<()> {
        let dir = TestDir::new("symlinked-models");
        let elsewhere = dir.0.join("big-disk").join("models");
        write(&elsewhere, &[("checkpoints/huge.safetensors", "huge")]);
        let source = dir.0.join("source");
        write(&source, &[("main.py", "old main")]);
        std::os::unix::fs::symlink(&elsewhere, source.join("models"))?;
        let tree = new_release(&dir.0);

        let kept = install_release_tree(&tree, &source)?;

        assert!(
            fs::symlink_metadata(source.join("models"))?
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_link(source.join("models"))?, elsewhere);
        let mut entries: Vec<_> = fs::read_dir(&elsewhere)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<io::Result<_>>()?;
        entries.sort();
        assert_eq!(entries, ["checkpoints"], "the release wrote into the link");
        assert_eq!(
            read(&elsewhere.join("checkpoints/huge.safetensors")).as_deref(),
            Some("huge")
        );
        assert!(kept.contains(&"models"));
        Ok(())
    }

    /// A symlink inside `models` (one model folder on another disk, or a link
    /// that has gone dangling) is left exactly as it is.
    #[cfg(unix)]
    #[test]
    fn symlinks_inside_models_are_left_alone() -> Result<()> {
        let dir = TestDir::new("symlink-inside-models");
        let elsewhere = dir.0.join("big-disk").join("checkpoints");
        write(&elsewhere, &[("huge.safetensors", "huge")]);
        let source = dir.0.join("source");
        write(&source, &[("main.py", "old main")]);
        fs::create_dir_all(source.join("models"))?;
        std::os::unix::fs::symlink(&elsewhere, source.join("models/checkpoints"))?;
        std::os::unix::fs::symlink(dir.0.join("gone"), source.join("models/dangling"))?;
        // A loop must not be followed either.
        std::os::unix::fs::symlink(source.join("models"), source.join("models/loop"))?;
        let tree = new_release(&dir.0);

        install_release_tree(&tree, &source)?;

        assert_eq!(fs::read_link(source.join("models/checkpoints"))?, elsewhere);
        assert_eq!(
            fs::read_link(source.join("models/dangling"))?,
            dir.0.join("gone")
        );
        assert_eq!(
            fs::read_link(source.join("models/loop"))?,
            source.join("models")
        );
        assert!(
            !elsewhere.join("put_checkpoints_here").exists(),
            "the release's placeholder must not be written through the link"
        );
        assert!(source.join("models/new_kind/put_new_kind_here").is_file());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn copy_fallback_recreates_symlinks_instead_of_following_them() -> Result<()> {
        let dir = TestDir::new("copy-no-follow");
        let from = dir.0.join("from");
        write(&from, &[("a/file.txt", "a")]);
        std::os::unix::fs::symlink(dir.0.join("outside"), from.join("a/link"))?;
        std::os::unix::fs::symlink(&from, from.join("a/loop"))?;

        copy_tree_no_follow(&from, &dir.0.join("to"))?;

        assert_eq!(read(&dir.0.join("to/a/file.txt")).as_deref(), Some("a"));
        assert_eq!(
            fs::read_link(dir.0.join("to/a/link"))?,
            dir.0.join("outside")
        );
        assert_eq!(fs::read_link(dir.0.join("to/a/loop"))?, from);
        Ok(())
    }

    #[test]
    fn a_failed_release_move_is_an_error_not_a_copy() -> Result<()> {
        let dir = TestDir::new("move-error");
        let to = dir.0.join("to");
        // The source does not exist: a NotFound rename error must surface
        // as an error, not fall back to copying.
        let error = move_release_entry(&dir.0.join("missing"), &to, &mut |_| Ok(()))
            .expect_err("a failed rename that does not cross devices must fail");
        assert!(format!("{error:#}").contains("failed to move"), "{error:#}");
        assert!(!to.exists());
        Ok(())
    }
}
