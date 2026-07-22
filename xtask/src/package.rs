// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Build a release distribution bundle in pure Rust.
//!
//! Produces the same external artifact contract the former
//! `scripts/package-{linux,windows}-release` scripts did, from one
//! cross-platform command:
//!
//! * a top-level `<dist>/` directory holding `bin/rocm[.exe]`, `bin/rocmd[.exe]`,
//!   `README.md`, `LICENSE.TXT`, and the platform installer (`install.sh` on
//!   Unix, `install.ps1` on Windows);
//! * an archive of that directory — `<dist>.tar.gz` on Unix, `<dist>.zip` on
//!   Windows — with the bundle directory as the single top-level entry;
//! * a `<archive>.sha256` sidecar in `sha256sum`/`Get-FileHash` syntax
//!   (`<lowercase-hex>  <archive-file-name>`, two spaces);
//! * a detached `<archive>.sig` when a signing key is configured.
//!
//! Signing inputs come from the environment for parity with the old scripts:
//! `ROCM_CLI_SIGNING_PRIVATE_KEY_PATH` (a PEM file) or
//! `ROCM_CLI_SIGNING_PRIVATE_KEY_PEM` (an inline PEM). Set
//! `ROCM_CLI_REQUIRE_SIGNATURE` to a truthy value to fail unless a key is
//! configured and a signature is produced.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::sign_rsa_pkcs1_sha256_signature;
use sha2::{Digest, Sha256};

use crate::paths::{binary_name, release_binary_dir, workspace_root};

/// Files copied to the top level of the bundle directory, resolved relative to
/// the workspace root. The installer differs per platform; the rest are shared.
const SHARED_TOP_LEVEL_FILES: &[&str] = &["README.md", "LICENSE.TXT"];

/// Platform installer copied into the bundle. The bundle ships the installer for
/// the platform it targets, matching the archive format.
#[cfg(windows)]
const PLATFORM_INSTALLER: &str = "install.ps1";
#[cfg(not(windows))]
const PLATFORM_INSTALLER: &str = "install.sh";

/// Package the release binaries under `dist_name` into `output_dir`.
///
/// `output_dir` is resolved against the workspace root when relative.
/// `target_triple` selects `target/<triple>/release` binaries when set.
pub fn run(dist_name: &str, output_dir: &Path, target_triple: Option<&str>) -> Result<()> {
    let root = workspace_root()?;
    let output_root = if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        root.join(output_dir)
    };
    let bin_dir = release_binary_dir(&root, target_triple);

    let plan = BundlePlan::new(&root, &output_root, &bin_dir, dist_name);
    plan.stage()?;
    let archive = plan.archive()?;
    let checksum = write_checksum(&archive)?;

    let signature = sign_archive(&archive)?;

    report_created(&archive, &checksum, signature.as_deref());
    Ok(())
}

/// A resolved set of paths for one bundle, plus the staging/archiving steps.
struct BundlePlan {
    /// Workspace root — the source of `README.md`, `LICENSE.TXT`, and installer.
    root: PathBuf,
    /// Output directory holding the bundle directory and archive.
    output_root: PathBuf,
    /// Directory holding the built release binaries.
    bin_dir: PathBuf,
    /// Distribution name — the bundle directory name and archive stem.
    dist_name: String,
    /// The bundle directory: `<output_root>/<dist_name>`.
    bundle_dir: PathBuf,
    /// The archive path: `<output_root>/<dist_name>.<ext>`.
    archive_path: PathBuf,
}

impl BundlePlan {
    fn new(root: &Path, output_root: &Path, bin_dir: &Path, dist_name: &str) -> Self {
        let bundle_dir = output_root.join(dist_name);
        let archive_path = output_root.join(format!("{dist_name}{}", archive_suffix()));
        Self {
            root: root.to_path_buf(),
            output_root: output_root.to_path_buf(),
            bin_dir: bin_dir.to_path_buf(),
            dist_name: dist_name.to_string(),
            bundle_dir,
            archive_path,
        }
    }

    /// Create the bundle directory and copy the binaries, docs, and installer.
    ///
    /// The standalone `rocm-engine-*` binaries are intentionally not shipped:
    /// the first-party engines are compiled into `rocm` and run in-process; the
    /// standalone binaries are only an external plugin fallback.
    fn stage(&self) -> Result<()> {
        fs::create_dir_all(&self.output_root)
            .with_context(|| format!("failed to create {}", self.output_root.display()))?;
        if self.bundle_dir.exists() {
            fs::remove_dir_all(&self.bundle_dir)
                .with_context(|| format!("failed to clear {}", self.bundle_dir.display()))?;
        }
        // Remove any prior archive + sidecars so a stale artifact can't survive.
        for suffix in ["", ".sha256", ".sig"] {
            let path = with_suffix(&self.archive_path, suffix);
            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
        }

        let bundle_bin = self.bundle_dir.join("bin");
        fs::create_dir_all(&bundle_bin)
            .with_context(|| format!("failed to create {}", bundle_bin.display()))?;

        for binary in ["rocm", "rocmd"] {
            let file = binary_name(binary);
            copy_required(&self.bin_dir.join(&file), &bundle_bin.join(&file))?;
        }
        for file in SHARED_TOP_LEVEL_FILES {
            copy_required(&self.root.join(file), &self.bundle_dir.join(file))?;
        }
        copy_required(
            &self.root.join(PLATFORM_INSTALLER),
            &self.bundle_dir.join(PLATFORM_INSTALLER),
        )?;
        Ok(())
    }

    /// Archive the staged bundle directory into the platform archive format,
    /// with the bundle directory as the single top-level entry.
    fn archive(&self) -> Result<PathBuf> {
        #[cfg(windows)]
        {
            zip_bundle(&self.bundle_dir, &self.archive_path, &self.dist_name)?;
        }
        #[cfg(not(windows))]
        {
            targz_bundle(&self.output_root, &self.dist_name, &self.archive_path)?;
        }
        Ok(self.archive_path.clone())
    }
}

/// The archive extension for the current platform.
const fn archive_suffix() -> &'static str {
    if cfg!(windows) { ".zip" } else { ".tar.gz" }
}

/// Append `suffix` to a path's file name, keeping it in the same directory.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    name.push(suffix);
    path.with_file_name(name)
}

/// Copy `source` to `destination`, failing with a clear message if the source
/// file is missing (the old scripts hard-failed on a missing required file).
fn copy_required(source: &Path, destination: &Path) -> Result<()> {
    // Refuse a `..` traversal segment on either endpoint before any filesystem
    // access. The source is derived from an environment override (`ROCM_BIN_DIR`
    // / `CARGO_TARGET_DIR`) and the destination from the CLI `output_dir`/
    // `dist_name`; a `..` segment could escape the workspace or output tree. The
    // validated strings are what build the paths used below, so absolute
    // overrides stay valid — only `..` traversal is refused.
    let source = source.to_string_lossy();
    if source.contains("..") {
        bail!("refusing a source path with a `..` traversal segment: {source}");
    }
    let destination = destination.to_string_lossy();
    if destination.contains("..") {
        bail!("refusing a destination path with a `..` traversal segment: {destination}");
    }
    let source = Path::new(source.as_ref());
    let destination = Path::new(destination.as_ref());
    if !source.is_file() {
        bail!("required file not found: {}", source.display());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::copy(source, destination).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

/// Build `<dist>.tar.gz` containing the bundle directory as its top-level entry,
/// mirroring `(cd output; tar -cf dist.tar dist) && gzip dist.tar`.
#[cfg(not(windows))]
fn targz_bundle(output_root: &Path, dist_name: &str, archive_path: &Path) -> Result<()> {
    use flate2::Compression;
    use flate2::write::GzEncoder;

    let file = fs::File::create(archive_path)
        .with_context(|| format!("failed to create {}", archive_path.display()))?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut builder = tar::Builder::new(encoder);
    // Match the old script's relative layout: entries are `<dist>/...` because
    // the tar was created from within the output dir over the `<dist>` dir.
    builder
        .append_dir_all(dist_name, output_root.join(dist_name))
        .with_context(|| format!("failed to add {dist_name} to the tar archive"))?;
    let encoder = builder
        .into_inner()
        .context("failed to finalize the tar archive")?;
    encoder.finish().context("failed to finish gzip stream")?;
    Ok(())
}

/// Build `<dist>.zip` containing the bundle directory as its top-level entry,
/// mirroring PowerShell's `Compress-Archive -LiteralPath <dist-dir>`.
#[cfg(windows)]
fn zip_bundle(bundle_dir: &Path, archive_path: &Path, dist_name: &str) -> Result<()> {
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    let file = fs::File::create(archive_path)
        .with_context(|| format!("failed to create {}", archive_path.display()))?;
    let mut writer = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o755);

    let mut entries = Vec::new();
    collect_files(bundle_dir, &mut entries)?;
    entries.sort();
    for absolute in entries {
        let relative = absolute
            .strip_prefix(bundle_dir)
            .expect("collected path is under the bundle dir");
        // Top-level entry is the `<dist>/...` bundle directory, with `/`
        // separators regardless of host, matching the tar layout.
        let name = format!("{dist_name}/{}", to_zip_path(relative));
        writer
            .start_file(name, options)
            .with_context(|| format!("failed to add {} to the zip archive", absolute.display()))?;
        let bytes = fs::read(&absolute)
            .with_context(|| format!("failed to read {}", absolute.display()))?;
        writer.write_all(&bytes).with_context(|| {
            format!(
                "failed to write {} into the zip archive",
                absolute.display()
            )
        })?;
    }
    writer
        .finish()
        .context("failed to finalize the zip archive")?;
    Ok(())
}

/// Recursively collect regular files under `dir` into `out`.
#[cfg(windows)]
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry =
            entry.with_context(|| format!("failed to read an entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Render a relative path with `/` separators for a zip entry name.
#[cfg(windows)]
fn to_zip_path(relative: &Path) -> String {
    relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Write the SHA-256 sidecar in `sha256sum`/`Get-FileHash` syntax:
/// `<lowercase-hex>  <archive-file-name>` (two spaces), matching both scripts so
/// existing installers and release-readiness checks keep working.
fn write_checksum(archive: &Path) -> Result<PathBuf> {
    let bytes =
        fs::read(archive).with_context(|| format!("failed to read {}", archive.display()))?;
    let digest = Sha256::digest(&bytes);
    let hex = hex_lower(&digest);
    let file_name = archive
        .file_name()
        .context("archive path has no file name")?
        .to_string_lossy()
        .into_owned();
    let checksum_path = with_suffix(archive, ".sha256");
    fs::write(&checksum_path, format!("{hex}  {file_name}\n"))
        .with_context(|| format!("failed to write {}", checksum_path.display()))?;
    Ok(checksum_path)
}

/// Lowercase hex encoding of a byte slice.
fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Whether a string is truthy: `1`, `true`, `yes`, or `on` (case-insensitive),
/// matching the old scripts' parsing.
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Read a truthy environment flag using [`is_truthy`] semantics.
fn truthy_env(key: &str) -> bool {
    std::env::var(key).is_ok_and(|value| is_truthy(&value))
}

/// Resolve the signing private key PEM from the environment, if configured.
///
/// Prefers `ROCM_CLI_SIGNING_PRIVATE_KEY_PATH` (read from disk); falls back to
/// the inline `ROCM_CLI_SIGNING_PRIVATE_KEY_PEM`. Returns `None` when neither is
/// set, so an unsigned bundle is produced (unless a signature is required).
fn signing_private_key_pem() -> Result<Option<String>> {
    if let Some(path) = std::env::var_os("ROCM_CLI_SIGNING_PRIVATE_KEY_PATH") {
        let path = PathBuf::from(path);
        if !path.as_os_str().is_empty() {
            // Refuse a `..` traversal segment in the configured key path before
            // reading it. The validated string is what opens the file, so an
            // absolute key path anywhere stays valid — only `..` is refused.
            let path = path.to_string_lossy();
            if path.contains("..") {
                bail!("refusing a signing key path with a `..` traversal segment: {path}");
            }
            let pem = fs::read_to_string(Path::new(path.as_ref()))
                .with_context(|| format!("failed to read signing key {path}"))?;
            return Ok(Some(pem));
        }
    }
    if let Ok(pem) = std::env::var("ROCM_CLI_SIGNING_PRIVATE_KEY_PEM")
        && !pem.trim().is_empty()
    {
        return Ok(Some(pem));
    }
    Ok(None)
}

/// Sign the archive when a key is configured, honoring
/// `ROCM_CLI_REQUIRE_SIGNATURE`. Returns the signature path when one is written.
fn sign_archive(archive: &Path) -> Result<Option<PathBuf>> {
    let required = truthy_env("ROCM_CLI_REQUIRE_SIGNATURE");
    let private_pem = signing_private_key_pem()?;
    sign_archive_with(archive, private_pem, required)
}

/// Pure core of [`sign_archive`]: sign `archive` given an already-resolved
/// private-key PEM and whether a signature is required. Kept free of environment
/// reads so it is unit-testable without mutating process-global state.
fn sign_archive_with(
    archive: &Path,
    private_pem: Option<String>,
    required: bool,
) -> Result<Option<PathBuf>> {
    let Some(private_pem) = private_pem else {
        if required {
            bail!(
                "signature is required but ROCM_CLI_SIGNING_PRIVATE_KEY_PATH/PEM is not configured"
            );
        }
        return Ok(None);
    };

    let payload =
        fs::read(archive).with_context(|| format!("failed to read {}", archive.display()))?;
    let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload)
        .context("failed to sign the release archive")?;
    let signature_path = with_suffix(archive, ".sig");
    fs::write(&signature_path, signature)
        .with_context(|| format!("failed to write {}", signature_path.display()))?;

    if required && !signature_path.is_file() {
        bail!(
            "signature is required but {} was not produced",
            signature_path.display()
        );
    }
    Ok(Some(signature_path))
}

/// Print the `created:` block the release/nightly workflows read.
fn report_created(archive: &Path, checksum: &Path, signature: Option<&Path>) {
    println!("created:");
    println!("  {}", archive.display());
    println!("  {}", checksum.display());
    if let Some(signature) = signature {
        println!("  {}", signature.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_uses_two_space_lowercase_hex_syntax() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("bundle.tar.gz");
        fs::write(&archive, b"hello world").expect("write archive");

        let checksum_path = write_checksum(&archive).expect("write checksum");
        let contents = fs::read_to_string(&checksum_path).expect("read checksum");

        // Known SHA-256 of "hello world".
        let expected_hex = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert_eq!(contents, format!("{expected_hex}  bundle.tar.gz\n"));
    }

    #[test]
    fn hex_lower_pads_and_lowercases() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
    }

    #[test]
    fn is_truthy_matches_the_script_vocabulary() {
        for value in ["1", "true", "TRUE", " Yes ", "on", "ON"] {
            assert!(is_truthy(value), "{value:?} should be truthy");
        }
        for value in ["0", "false", "no", "off", "", "  "] {
            assert!(!is_truthy(value), "{value:?} should be falsy");
        }
    }

    #[test]
    fn with_suffix_appends_to_file_name() {
        let base = Path::new("/tmp/out/rocm-cli.tar.gz");
        assert_eq!(
            with_suffix(base, ".sha256"),
            PathBuf::from("/tmp/out/rocm-cli.tar.gz.sha256")
        );
        assert_eq!(
            with_suffix(base, ".sig"),
            PathBuf::from("/tmp/out/rocm-cli.tar.gz.sig")
        );
    }

    #[test]
    fn archive_suffix_matches_platform() {
        if cfg!(windows) {
            assert_eq!(archive_suffix(), ".zip");
        } else {
            assert_eq!(archive_suffix(), ".tar.gz");
        }
    }

    #[test]
    fn copy_required_reports_missing_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nope.txt");
        let dest = dir.path().join("dest.txt");
        let error = copy_required(&missing, &dest).unwrap_err();
        assert!(error.to_string().contains("required file not found"));
    }

    #[test]
    fn copy_required_rejects_parent_dir_traversal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("real.txt");
        fs::write(&source, b"data").expect("write source");
        let dest = dir.path().join("bin.txt");

        // A `..` in the source is refused before the file is even read.
        let via_source = Path::new("bundle/../../etc/passwd");
        let error = copy_required(via_source, &dest).unwrap_err();
        assert!(
            error.to_string().contains("traversal"),
            "unexpected error: {error}"
        );

        // A `..` in the destination is refused too, even with a real source.
        let via_dest = dir.path().join("out/../../escape.txt");
        let error = copy_required(&source, &via_dest).unwrap_err();
        assert!(
            error.to_string().contains("traversal"),
            "unexpected error: {error}"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn targz_bundle_produces_expected_layout() {
        use flate2::read::GzDecoder;

        let dir = tempfile::tempdir().expect("tempdir");
        let output_root = dir.path();
        let dist = "rocm-cli-test-amd64";
        let bundle = output_root.join(dist);
        fs::create_dir_all(bundle.join("bin")).expect("bundle bin");
        fs::write(bundle.join("bin/rocm"), b"binary").expect("rocm");
        fs::write(bundle.join("README.md"), b"readme").expect("readme");

        let archive = output_root.join(format!("{dist}.tar.gz"));
        targz_bundle(output_root, dist, &archive).expect("targz");
        assert!(archive.is_file());

        let file = fs::File::open(&archive).expect("open archive");
        let mut tar = tar::Archive::new(GzDecoder::new(file));
        let mut names: Vec<String> = tar
            .entries()
            .expect("entries")
            .map(|entry| {
                entry
                    .expect("entry")
                    .path()
                    .expect("path")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        // Every entry is nested under the `<dist>/` top-level directory.
        assert!(
            names
                .iter()
                .all(|name| name.starts_with(&format!("{dist}/"))),
            "entries must be under {dist}/: {names:?}"
        );
        assert!(names.iter().any(|name| name == &format!("{dist}/bin/rocm")));
        assert!(
            names
                .iter()
                .any(|name| name == &format!("{dist}/README.md"))
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn stage_then_checksum_and_sign_round_trips() {
        use rocm_core::{generate_rsa_signing_keypair, verify_rsa_pkcs1_sha256_signature};

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("ws");
        let bin_dir = root.join("target/release");
        fs::create_dir_all(&bin_dir).expect("bin dir");
        for name in ["rocm", "rocmd"] {
            fs::write(bin_dir.join(name), format!("{name}-bytes")).expect("write binary");
        }
        for name in ["README.md", "LICENSE.TXT", "install.sh"] {
            fs::write(root.join(name), format!("{name}-content")).expect("write top-level");
        }

        let output_root = dir.path().join("out");
        let dist = "rocm-cli-round-trip-amd64";
        let plan = BundlePlan::new(&root, &output_root, &bin_dir, dist);
        plan.stage().expect("stage");
        // Staged layout matches the shipped bundle.
        assert!(output_root.join(dist).join("bin/rocm").is_file());
        assert!(output_root.join(dist).join("bin/rocmd").is_file());
        assert!(output_root.join(dist).join("install.sh").is_file());

        let archive = plan.archive().expect("archive");
        let checksum = write_checksum(&archive).expect("checksum");
        assert!(checksum.is_file());

        // Sign with a generated key and confirm the sidecar verifies.
        let (private_pem, public_pem) = generate_rsa_signing_keypair().expect("keypair");
        let signature = sign_archive_with(&archive, Some(private_pem), true)
            .expect("sign")
            .expect("signature path");
        let payload = fs::read(&archive).expect("read archive");
        let signature_bytes = fs::read(&signature).expect("read signature");
        verify_rsa_pkcs1_sha256_signature(&public_pem, &payload, &signature_bytes, "archive")
            .expect("signature verifies");
    }

    #[test]
    fn required_signature_without_key_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("bundle.tar.gz");
        fs::write(&archive, b"bytes").expect("write archive");
        let error = sign_archive_with(&archive, None, true).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("signature is required but ROCM_CLI_SIGNING_PRIVATE_KEY_PATH/PEM")
        );
    }

    #[test]
    fn no_key_and_not_required_yields_no_signature() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("bundle.tar.gz");
        fs::write(&archive, b"bytes").expect("write archive");
        assert!(sign_archive_with(&archive, None, false).unwrap().is_none());
    }
}
