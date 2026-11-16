// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Recursion backstop for copying the embeddable tree. Containment is enforced
/// per entry, so this exists only to fail loudly instead of overflowing the
/// stack on a pathological tree; it is far above any plausible archive layout.
const MAX_COPY_RECURSION_DEPTH: usize = 64;

/// Resolves an archive entry name to a path inside `destination`.
///
/// Entry names come from the archive, so they are untrusted: this rejects
/// absolute paths, drive-relative and UNC names, `..` traversal, and anything
/// else that is not a plain sequence of normal components. The returned path is
/// built component-by-component, so it cannot leave `destination`.
fn archive_entry_destination(destination: &Path, name: &Path) -> Result<PathBuf> {
    let mut resolved = destination.to_path_buf();
    let mut components = 0usize;
    for component in name.components() {
        match component {
            std::path::Component::Normal(part) => {
                // Reject separators smuggled through a single component (a
                // Windows-style `a\..\b` read on Unix is one component).
                let text = part.to_string_lossy();
                if text.contains('/') || text.contains('\\') || text == ".." {
                    bail!("archive entry has an unsafe name: {}", name.display());
                }
                resolved.push(part);
                components += 1;
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                bail!(
                    "archive entry escapes the extraction root: {}",
                    name.display()
                );
            }
        }
    }
    if components == 0 {
        bail!("archive entry has an empty name");
    }
    Ok(resolved)
}

/// Creates the parent directories for an extracted entry, inside `destination`.
fn create_entry_parent(destination: &Path, path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if !parent.starts_with(destination) {
        bail!(
            "archive entry escapes the extraction root: {}",
            path.display()
        );
    }
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    Ok(())
}

/// Writes one extracted regular file, refusing to follow an existing entry.
///
/// `create_new` matters: without it a directory entry extracted earlier — or a
/// symlink planted by a co-resident process between entries — would be written
/// through rather than rejected.
fn write_entry_file(path: &Path, reader: &mut dyn Read) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    std::io::copy(reader, &mut file)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_entry_mode(path: &Path, mode: Option<u32>) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let Some(mode) = mode else {
        return Ok(());
    };
    // Keep the executable bit the runtime needs; drop setuid/setgid/sticky and
    // any group/other write bit the archive happened to carry.
    let mode = mode & 0o755;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .with_context(|| format!("failed to set permissions on {}", path.display()))
}

#[cfg(not(unix))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "mirrors the Unix signature; Windows has no mode bits to apply"
)]
const fn set_entry_mode(_path: &Path, _mode: Option<u32>) -> Result<()> {
    Ok(())
}

/// Extracts a gzip-compressed tar archive in-process.
fn extract_tar_gz(archive: &Path, destination: &Path) -> Result<()> {
    let file =
        fs::File::open(archive).with_context(|| format!("failed to open {}", archive.display()))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(std::io::BufReader::new(file)));
    for entry in tar
        .entries()
        .with_context(|| format!("failed to read {}", archive.display()))?
    {
        let mut entry = entry.with_context(|| format!("failed to read {}", archive.display()))?;
        let name = entry
            .path()
            .with_context(|| {
                format!(
                    "archive entry has an undecodable name in {}",
                    archive.display()
                )
            })?
            .into_owned();
        let kind = entry.header().entry_type();
        // A pax *global* header carries archive-wide metadata, not a member: its
        // name (`pax_global_header` by convention) is not a path to create. The
        // `tar` crate consumes long-name, long-link and pax *local* headers
        // itself, but hands this one to the caller, so it has to be skipped
        // explicitly — `git archive` and `tar --format=pax` both emit it, and
        // treating it as a member would reject an otherwise valid archive.
        if kind == tar::EntryType::XGlobalHeader {
            continue;
        }
        let path = archive_entry_destination(destination, &name)?;
        match kind {
            tar::EntryType::Directory => {
                fs::create_dir_all(&path)
                    .with_context(|| format!("failed to create {}", path.display()))?;
            }
            tar::EntryType::Regular | tar::EntryType::Continuous => {
                create_entry_parent(destination, &path)?;
                let mode = entry.header().mode().ok();
                write_entry_file(&path, &mut entry)?;
                set_entry_mode(&path, mode)?;
            }
            // A sparse entry would expand correctly — the crate resolves the
            // sparse map — but a few-byte header can declare a multi-gigabyte
            // logical size, so accepting one turns a digest-pinned archive into
            // an unbounded write. No Lemonade artifact stores files sparsely, so
            // this stays rejected until one needs it.
            tar::EntryType::GNUSparse => bail!(
                "archive contains a sparse entry at {}, which is not supported",
                name.display()
            ),
            // Links, devices, fifos and sockets are the entry kinds that let an
            // archive reach outside its own tree; the official artifacts carry
            // none of them. See the link policy on `extract_archive` for what
            // to do if a future release ships one.
            other => bail!(
                "archive contains an unsupported entry ({other:?}) at {}",
                name.display()
            ),
        }
    }
    Ok(())
}

/// Extracts a zip archive in-process.
fn extract_zip(archive: &Path, destination: &Path) -> Result<()> {
    const S_IFMT: u32 = 0o170_000;
    const S_IFLNK: u32 = 0o120_000;
    let file =
        fs::File::open(archive).with_context(|| format!("failed to open {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("failed to read {}", archive.display()))?;
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .with_context(|| format!("failed to read entry {index} of {}", archive.display()))?;
        // `enclosed_name` returns None for absolute or `..`-bearing names; the
        // raw name is still used for diagnostics and for the trailing-slash
        // directory convention.
        let raw = entry
            .name()
            .with_context(|| format!("failed to decode the name of entry {index}"))?
            .into_owned();
        // See the link policy on `extract_archive`. `zip` 9 ignores the high
        // attribute bits of an entry written by a DOS host, so `is_symlink`
        // misses a link recorded there; check the bits directly.
        if entry.is_symlink() || (entry.external_attributes() >> 16) & S_IFMT == S_IFLNK {
            bail!("archive contains symlink at {raw}");
        }
        let name = entry
            .enclosed_name()
            .with_context(|| format!("archive entry escapes the extraction root: {raw}"))?;
        let path = archive_entry_destination(destination, &name)?;
        if entry.is_dir() {
            fs::create_dir_all(&path)
                .with_context(|| format!("failed to create {}", path.display()))?;
            continue;
        }
        // No third entry kind to reject here: `zip` defines `is_file` as
        // "neither a directory nor a symlink", and both are handled above.
        create_entry_parent(destination, &path)?;
        let mode = entry.unix_mode();
        write_entry_file(&path, &mut entry)?;
        set_entry_mode(&path, mode)?;
    }
    Ok(())
}

/// Extracts the embeddable archive into `destination`.
///
/// Extraction is done in-process rather than by shelling out to `tar`: the
/// external binary was resolved from `PATH` (so the extractor unpacking
/// untrusted bytes was itself unpinned), and its traversal behaviour varied
/// between GNU tar, the bsdtar shipped as Windows `tar.exe`, and busybox tar.
///
/// # Link policy
///
/// Extraction writes directories and regular files and nothing else: a link
/// entry is rejected rather than resolved, here and again when the extracted
/// tree is searched and copied out. That is deliberately conservative. The
/// v10.10.0 artifacts contain no links, and refusing them removes the whole
/// class of "entry points outside the tree" bugs instead of trying to reason
/// about each one.
///
/// The cost is that it fails closed on an authentic archive: if a future
/// Lemonade release ships, say, `libfoo.so -> libfoo.so.1`, installs stop with
/// "archive contains symlink at ...". That is a signal to widen the policy
/// deliberately — resolve the target and accept it only if it stays inside the
/// extraction root — not to drop the check.
pub(crate) fn extract_archive(archive: &Path, destination: &Path) -> Result<()> {
    let is_zip = archive
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"));
    if is_zip {
        extract_zip(archive, destination)
    } else {
        extract_tar_gz(archive, destination)
    }
}

fn destination_child(destination: &Path, name: &std::ffi::OsStr) -> Result<PathBuf> {
    let component = Path::new(name);
    if component.components().count() != 1
        || !matches!(
            component.components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        bail!(
            "archive entry is not one destination component: {}",
            Path::new(name).display()
        );
    }
    Ok(destination.join(component))
}

pub(crate) fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    let source = fs::canonicalize(source)
        .with_context(|| format!("failed to canonicalize {}", source.display()))?;
    if !fs::symlink_metadata(&source)?.file_type().is_dir() {
        bail!("copy source is not a directory: {}", source.display());
    }
    fs::create_dir(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let destination = fs::canonicalize(destination)
        .with_context(|| format!("failed to canonicalize {}", destination.display()))?;
    copy_tree_entries(&source, &source, &destination, &destination, 0)
}

fn copy_tree_entries(
    source_root: &Path,
    source: &Path,
    destination_root: &Path,
    destination: &Path,
    depth: usize,
) -> Result<()> {
    if depth > MAX_COPY_RECURSION_DEPTH {
        bail!(
            "archive tree is deeper than {MAX_COPY_RECURSION_DEPTH} levels at {}",
            source.display()
        );
    }
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let file_type = entry.file_type()?;
        // See the link policy on `extract_archive`.
        if file_type.is_symlink() {
            bail!("refusing to copy symlink {}", source_path.display());
        }
        if !file_type.is_dir() && !file_type.is_file() {
            bail!(
                "refusing to copy non-regular entry {}",
                source_path.display()
            );
        }
        let canonical_source = fs::canonicalize(&source_path)?;
        if !canonical_source.starts_with(source_root) {
            bail!(
                "copy source escapes extraction root: {}",
                source_path.display()
            );
        }
        let destination_path = destination_child(destination, &entry.file_name())?;
        if !destination_path.starts_with(destination_root) {
            bail!(
                "copy destination escapes runtime root: {}",
                destination_path.display()
            );
        }
        if file_type.is_dir() {
            fs::create_dir(&destination_path)
                .with_context(|| format!("failed to create {}", destination_path.display()))?;
            let canonical_destination = fs::canonicalize(&destination_path)?;
            if !canonical_destination.starts_with(destination_root) {
                bail!(
                    "copy destination escapes runtime root: {}",
                    destination_path.display()
                );
            }
            copy_tree_entries(
                source_root,
                &canonical_source,
                destination_root,
                &canonical_destination,
                depth + 1,
            )?;
        } else {
            fs::copy(&canonical_source, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    canonical_source.display(),
                    destination_path.display()
                )
            })?;
            if !cfg!(windows) {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let metadata = fs::symlink_metadata(&canonical_source)?;
                    fs::set_permissions(
                        &destination_path,
                        fs::Permissions::from_mode(metadata.permissions().mode()),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use std::io::Write;
    #[cfg(unix)]
    use std::process::{Command as ProcessCommand, Stdio};

    #[test]
    fn destination_child_requires_one_normal_component() {
        let destination = Path::new("runtime");
        assert_eq!(
            destination_child(destination, std::ffi::OsStr::new("bin")).unwrap(),
            destination.join("bin")
        );
        assert!(destination_child(destination, std::ffi::OsStr::new("..")).is_err());
        assert!(destination_child(destination, std::ffi::OsStr::new(".")).is_err());
        assert!(destination_child(destination, std::ffi::OsStr::new("a/b")).is_err());
    }

    /// Writes a `.tar.gz` whose single entry carries `entry_name` verbatim.
    ///
    /// The name is stamped into the raw header bytes because `tar::Builder`
    /// refuses to serialize `..` or absolute paths — the archives this guards
    /// against are hostile ones, which no well-behaved writer would produce.
    fn write_raw_named_tar_gz(path: &Path, entry_name: &str, body: &[u8]) {
        let file = fs::File::create(path).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_ustar();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        let name = header.as_old_mut();
        let bytes = entry_name.as_bytes();
        assert!(bytes.len() < name.name.len());
        name.name[..bytes.len()].copy_from_slice(bytes);
        header.set_cksum();
        builder.append(&header, body).unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    const EXTRACT_PATH_TEST_DIR: &str = "ROCM_LEMONADE_EXTRACT_PATH_TEST_DIR";

    /// Child half of [`extract_archive_does_not_execute_a_path_supplied_tar`].
    ///
    /// Runs in its own process so the hostile `PATH` can be set with
    /// `Command::env` — mutating `PATH` in-process would need `unsafe`, which
    /// the workspace denies.
    #[test]
    fn extract_path_subprocess_helper() {
        let Some(dir) = std::env::var_os(EXTRACT_PATH_TEST_DIR) else {
            return;
        };
        let dir = PathBuf::from(dir);
        let _ = extract_archive(&dir.join("archive.tar.gz"), &dir.join("extract"));
    }

    #[cfg(unix)]
    #[test]
    fn extract_archive_does_not_execute_a_path_supplied_tar() {
        use rocm_core::fake_executable::write_fake_executable;

        let dir = scratch_dir("path-supplied-tar");
        let archive = dir.join("archive.tar.gz");
        fs::create_dir_all(dir.join("extract")).unwrap();
        write_raw_named_tar_gz(&archive, "payload.txt", b"real");

        let fake_bin = dir.join("bin");
        fs::create_dir_all(&fake_bin).unwrap();
        let marker = dir.join("impostor-ran");
        let fake_tar = fake_bin.join("tar");
        write_fake_executable(
            &fake_tar,
            format!("#!/bin/sh\ntouch '{}'\nexit 0\n", marker.display()),
        )
        .unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let status = ProcessCommand::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "archive_extract::tests::extract_path_subprocess_helper",
                "--nocapture",
            ])
            .env(EXTRACT_PATH_TEST_DIR, &dir)
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "helper subprocess failed");

        assert!(
            !marker.exists(),
            "extraction executed a PATH-supplied tar instead of a pinned in-process extractor"
        );
        fs::remove_dir_all(&dir).ok();
    }

    fn extract_hostile_tar_entry(tag: &str, entry_name: &str) -> String {
        let dir = scratch_dir(tag);
        let archive = dir.join("hostile.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();
        write_raw_named_tar_gz(&archive, entry_name, b"pwned");

        let error = extract_archive(&archive, &destination).unwrap_err();
        assert_eq!(
            fs::read_dir(&destination).unwrap().count(),
            0,
            "hostile entry was written into the extraction root"
        );
        fs::remove_dir_all(&dir).ok();
        format!("{error:#}")
    }

    #[test]
    fn extract_archive_rejects_parent_traversal_entry() {
        let error = extract_hostile_tar_entry("traversal-parent", "../escaped.txt");
        assert!(error.contains("escapes the extraction root"), "{error}");
    }

    #[test]
    fn extract_archive_rejects_nested_parent_traversal_entry() {
        let error = extract_hostile_tar_entry("traversal-nested", "inner/../../escaped.txt");
        assert!(error.contains("escapes the extraction root"), "{error}");
    }

    #[test]
    fn extract_archive_rejects_absolute_entry() {
        let error = extract_hostile_tar_entry("traversal-absolute", "/etc/escaped.txt");
        assert!(error.contains("escapes the extraction root"), "{error}");
    }

    #[test]
    fn extract_archive_rejects_symlink_entry() {
        let dir = scratch_dir("traversal-symlink");
        let archive = dir.join("hostile.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_ustar();
        header.set_size(0);
        header.set_mode(0o777);
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_link_name("/etc/passwd").unwrap();
        header.set_path("escape-link").unwrap();
        header.set_cksum();
        builder.append(&header, std::io::empty()).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
        assert!(error.contains("unsupported entry"), "{error}");
        assert!(!destination.join("escape-link").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_archive_writes_a_well_formed_tree() {
        let dir = scratch_dir("extract-well-formed");
        let archive = dir.join("good.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_ustar();
        header.set_size(5);
        header.set_mode(0o755);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_path("embeddable/bin/lemond").unwrap();
        header.set_cksum();
        builder.append(&header, &b"hello"[..]).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        extract_archive(&archive, &destination).unwrap();
        let extracted = destination.join("embeddable").join("bin").join("lemond");
        assert_eq!(fs::read(&extracted).unwrap(), b"hello");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&extracted).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111, "executable bit was not preserved");
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_archive_rejects_hostile_zip_entry() {
        let dir = scratch_dir("traversal-zip");
        let archive = dir.join("hostile.zip");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        // `add_directory`/`start_file` normalize names, so the traversal name is
        // written through the raw entry API a hostile producer would use.
        writer
            .start_file("../escaped.txt", options)
            .or_else(|_| writer.start_file("placeholder.txt", options))
            .unwrap();
        writer.write_all(b"pwned").unwrap();
        writer.finish().unwrap();

        // If the zip writer refused to emit the traversal name there is nothing
        // hostile to assert on; the tar cases cover the shared validator.
        let mut zip = zip::ZipArchive::new(fs::File::open(&archive).unwrap()).unwrap();
        let hostile = zip.by_index(0).unwrap().name().unwrap().into_owned();
        if hostile == "../escaped.txt" {
            let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
            assert!(error.contains("escapes the extraction root"), "{error}");
            assert!(!dir.join("escaped.txt").exists());
        } else {
            extract_archive(&archive, &destination).unwrap();
            assert!(destination.join("placeholder.txt").is_file());
        }
        fs::remove_dir_all(&dir).ok();
    }

    /// A pax global header is archive-wide metadata, not a member, and the
    /// `tar` crate hands it to the caller rather than consuming it the way it
    /// consumes long-name and pax local headers. Treating it as a member
    /// rejects any archive written by `git archive` or `tar --format=pax`.
    #[test]
    fn extract_archive_skips_pax_global_header() {
        let dir = scratch_dir("pax-global-header");
        let archive = dir.join("pax.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);

        let record = b"52 comment=0000000000000000000000000000000000000000\n";
        let mut global = tar::Header::new_ustar();
        global.set_size(record.len() as u64);
        global.set_mode(0o644);
        global.set_entry_type(tar::EntryType::XGlobalHeader);
        global.set_path("pax_global_header").unwrap();
        global.set_cksum();
        builder.append(&global, &record[..]).unwrap();

        let mut header = tar::Header::new_ustar();
        header.set_size(5);
        header.set_mode(0o755);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_path("embeddable/bin/lemond").unwrap();
        header.set_cksum();
        builder.append(&header, &b"hello"[..]).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        extract_archive(&archive, &destination).unwrap();
        assert_eq!(
            fs::read(destination.join("embeddable").join("bin").join("lemond")).unwrap(),
            b"hello"
        );
        assert!(
            !destination.join("pax_global_header").exists(),
            "the global header was written out as a member"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// Zip counterpart to `extract_archive_rejects_symlink_entry`: the zip path
    /// has its own link check, keyed on the entry's mode and attribute bits.
    #[test]
    fn extract_archive_rejects_zip_symlink_entry() {
        let dir = scratch_dir("zip-symlink");
        let archive = dir.join("hostile.zip");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer
            .add_symlink(
                "escape-link",
                "/etc/passwd",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
        writer.finish().unwrap();

        let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
        assert!(error.contains("symlink"), "{error}");
        assert!(!destination.join("escape-link").exists());
        fs::remove_dir_all(&dir).ok();
    }

    /// `zip` 9 ignores the high attribute bits of an entry written by a DOS
    /// host, so an archive can mark a link there while `is_symlink` reports
    /// false. The extractor must still refuse it.
    #[test]
    fn extract_archive_rejects_zip_symlink_entry_from_dos_host() {
        let dir = scratch_dir("zip-symlink-dos");
        let archive = dir.join("hostile.zip");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .system(zip::System::Dos)
            .external_attributes(0o120_777 << 16);
        writer.start_file("escape-link", options).unwrap();
        std::io::Write::write_all(&mut writer, b"/etc/passwd").unwrap();
        writer.finish().unwrap();

        let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
        assert!(
            error.contains("archive contains symlink at escape-link"),
            "{error}"
        );
        assert!(!destination.join("escape-link").exists());
        fs::remove_dir_all(&dir).ok();
    }
}

/// Property tests for the in-process Lemonade extractor, judged on the
/// filesystem by `archive_props` rather than by re-deriving the extractor's
/// own path checks.
#[cfg(all(test, unix))]
mod archive_properties {
    use std::io::Write as _;

    use super::{extract_tar_gz, extract_zip};
    use crate::archive_props::{
        Entry, Expect, Kind, Modes, entries, note_mode_scope, run_case, run_property,
        run_round_trip, tar_gz_bytes,
    };

    /// The in-process extractor rejects every link kind and masks the modes
    /// the archive carries, so it promises the whole contract, not just "no
    /// escape".
    const FULL: Expect = Expect {
        modes: Modes::Safe,
        no_outward_links: true,
    };

    #[test]
    fn tar_extraction_keeps_every_entry_inside_the_root() {
        note_mode_scope("lemonade-tar", FULL.modes);
        run_property(
            "lemonade-tar",
            512,
            entries(),
            Some(5),
            |layout, entries| {
                run_case(
                    layout,
                    "embeddable.tar.gz",
                    &tar_gz_bytes(entries),
                    &[&layout.dest],
                    &layout.dest,
                    |layout, archive| extract_tar_gz(archive, &layout.dest),
                )
                .judge(FULL)
            },
        )
        .unwrap();
    }

    /// Zip names are strings, so non-UTF-8 names are carried lossily; hard
    /// links do not exist in zip and are written as regular files.
    fn zip_bytes(entries: &[Entry]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for entry in entries {
            let name = String::from_utf8_lossy(&entry.name).into_owned();
            let options = zip::write::SimpleFileOptions::default().unix_permissions(entry.mode);
            // The writer refuses a few names (and duplicates); a refused entry
            // is simply left out of the archive.
            match &entry.kind {
                Kind::File(data) | Kind::Hardlink(data) => {
                    if writer.start_file(name, options).is_ok() {
                        writer.write_all(data).unwrap();
                    }
                }
                Kind::Dir => {
                    let _ = writer.add_directory(name, options);
                }
                Kind::Symlink(target) => {
                    let _ = writer.add_symlink(
                        name,
                        String::from_utf8_lossy(target).into_owned(),
                        options,
                    );
                }
            }
        }
        writer.finish().unwrap().into_inner()
    }

    /// The other direction for both formats: a well-formed archive is
    /// extracted in full, into the root it was given.
    #[test]
    fn tar_and_zip_extraction_extract_a_benign_archive_into_the_root() {
        run_round_trip("lemonade-benign-tar", |layout, entries| {
            let archive = layout.base.join("embeddable.tar.gz");
            std::fs::write(&archive, tar_gz_bytes(entries))?;
            extract_tar_gz(&archive, &layout.dest)?;
            Ok(layout.dest.join("top"))
        })
        .unwrap();
        run_round_trip("lemonade-benign-zip", |layout, entries| {
            let archive = layout.base.join("embeddable.zip");
            std::fs::write(&archive, zip_bytes(entries))?;
            extract_zip(&archive, &layout.dest)?;
            Ok(layout.dest.join("top"))
        })
        .unwrap();
    }

    #[test]
    fn zip_extraction_keeps_every_entry_inside_the_root() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        note_mode_scope("lemonade-zip", FULL.modes);
        // Reach is measured on what the zip writer actually stored, since it
        // may drop or normalise some generated names.
        let (cases, traversal, symlink) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        run_property("lemonade-zip", 512, entries(), None, |layout, entries| {
            let bytes = zip_bytes(entries);
            let mut stored = zip::ZipArchive::new(std::io::Cursor::new(&bytes)).unwrap();
            cases.fetch_add(1, Ordering::Relaxed);
            let (mut has_traversal, mut has_symlink) = (false, false);
            for index in 0..stored.len() {
                let entry = stored.by_index_raw(index).unwrap();
                let name = entry.name().unwrap();
                has_traversal |=
                    name.starts_with('/') || name.split(['/', '\\']).any(|part| part == "..");
                has_symlink |= entry.is_symlink();
            }
            if has_traversal {
                traversal.fetch_add(1, Ordering::Relaxed);
            }
            if has_symlink {
                symlink.fetch_add(1, Ordering::Relaxed);
            }
            run_case(
                layout,
                "embeddable.zip",
                &bytes,
                &[&layout.dest],
                &layout.dest,
                |layout, archive| extract_zip(archive, &layout.dest),
            )
            .judge(FULL)
        })
        .unwrap();
        let cases = cases.into_inner().max(1);
        let (traversal, symlink) = (traversal.into_inner(), symlink.into_inner());
        eprintln!(
            "[lemonade-zip] stored traversal/absolute names in {}%, symlinks in {}% of {cases} archives",
            traversal * 100 / cases,
            symlink * 100 / cases
        );
        assert!(traversal * 100 / cases >= 5 && symlink * 100 / cases >= 5);
    }
}
