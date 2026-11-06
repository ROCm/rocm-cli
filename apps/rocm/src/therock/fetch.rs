// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTTP fetching for TheRock installs: the metadata GET with its on-disk
//! etag cache and detached-signature verification, the streaming artifact
//! download, and the size and free-space preflight for the tarball.
//!
//! Owns the pinned metadata trust root,
//! which `cargo xtask verify-pinned-keys` checks against the canonical
//! published key. Depends on the sibling `io` module for atomic writes and on
//! `slugify` in the parent for cache file names.

#[cfg(test)]
use super::io::{publish_temp_file, write_file_atomically_with_publish};
use super::io::{stage_file_for_atomic_publish, write_file_atomically};
use super::slugify;
use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, disk_space, unix_time_millis, verify_rsa_pkcs1_sha256_signature};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Timeout for the best-effort HEAD probe that sizes a download before starting it.
const THEROCK_HEAD_PROBE_TIMEOUT_SECS: u64 = 10;

/// Whole-transfer budget for an artifact download. A single-digit-gigabyte SDK
/// tarball on a slow link needs well past the ten minutes the metadata fetches
/// use; a retry that resumes cannot help if the attempt itself is cut short.
const THEROCK_DOWNLOAD_TIMEOUT: Duration = Duration::from_hours(1);

/// Largest `Content-Length` accepted as a real SDK tarball size.
///
/// SDK tarballs are single-digit gigabytes; anything past this is a
/// misconfigured proxy or a hostile header rather than a real artifact, and
/// must not be allowed to refuse an install on its own authority.
const THEROCK_MAX_PLAUSIBLE_TARBALL_BYTES: u64 = 256 * 1024 * 1024 * 1024;

/// Maximum size accepted for index, catalog, and detached-signature responses.
const THEROCK_MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct CachedHttpText {
    pub(super) text: String,
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
struct CachedHttpMetadata {
    url: String,
    #[serde(default)]
    etag: Option<String>,
    #[serde(default)]
    last_modified: Option<String>,
    #[serde(default)]
    signature: Option<CachedHttpSignatureMetadata>,
    fetched_at_unix_ms: u128,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct CachedHttpSignatureMetadata {
    url: String,
    verified_at_unix_ms: u128,
    public_key_source: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct CachedHttpCacheEntry {
    metadata: CachedHttpMetadata,
    body: String,
    #[serde(default)]
    signature_bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
struct MetadataSignaturePolicy {
    required: bool,
    public_key_path: Option<PathBuf>,
    public_key_pem: Option<String>,
}

#[derive(Debug)]
struct HttpResponseBody {
    status: u16,
    headers: String,
    body: Vec<u8>,
}

/// Pinned production metadata signing public key (trust root). Empty until the
/// repository owner publishes production keys (see docs/release-trust.md,
/// "Remaining Owner Step"). While empty, metadata verification stays opt-in
/// (enabled only via the `ROCM_CLI_METADATA_PUBLIC_KEY_*` env vars). Once
/// populated, metadata signatures are verified by default with this key as the
/// trust root.
const PINNED_METADATA_PUBLIC_KEY_PEM: &str = "";

/// The pinned metadata trust root, or `None` while the sentinel is still empty.
fn pinned_metadata_public_key() -> Option<String> {
    let pem = PINNED_METADATA_PUBLIC_KEY_PEM.trim();
    (!pem.is_empty()).then(|| pem.to_owned())
}

impl MetadataSignaturePolicy {
    fn from_env() -> Self {
        Self::resolve(
            truthy_env("ROCM_CLI_REQUIRE_METADATA_SIGNATURE"),
            env_path("ROCM_CLI_METADATA_PUBLIC_KEY_PATH"),
            env_nonempty("ROCM_CLI_METADATA_PUBLIC_KEY_PEM"),
            pinned_metadata_public_key(),
        )
    }

    /// Combine the env-provided inputs with the pinned trust root. An explicit
    /// env key (path or PEM) wins as an escape hatch; otherwise the pinned key is
    /// used, and its presence makes verification required by default.
    fn resolve(
        env_required: bool,
        env_path: Option<PathBuf>,
        env_pem: Option<String>,
        pinned_pem: Option<String>,
    ) -> Self {
        let pinned = if env_path.is_none() && env_pem.is_none() {
            pinned_pem
        } else {
            None
        };
        Self {
            required: env_required || pinned.is_some(),
            public_key_path: env_path,
            public_key_pem: env_pem.or(pinned),
        }
    }

    const fn active(&self) -> bool {
        self.required || self.public_key_path.is_some() || self.public_key_pem.is_some()
    }

    fn validate_configuration(&self) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        if let Some(public_key_path) = &self.public_key_path {
            if !public_key_path.is_file() {
                bail!(
                    "metadata public key not found: {}",
                    public_key_path.display()
                );
            }
            return Ok(());
        }
        if self.public_key_pem.is_some() {
            return Ok(());
        }
        bail!(
            "metadata signature verification requires ROCM_CLI_METADATA_PUBLIC_KEY_PATH or ROCM_CLI_METADATA_PUBLIC_KEY_PEM"
        )
    }
}

/// Whether an environment variable is set to an affirmative value.
///
/// Deliberately an allowlist rather than "set to anything non-empty": the
/// callers are opt-ins to weaker behaviour, so `=0` and `=false` must read as
/// off rather than as "the variable is present, therefore yes".
pub(crate) fn truthy_env(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn env_path(name: &str) -> Option<PathBuf> {
    env_nonempty(name).map(PathBuf::from)
}

fn with_metadata_public_key<T>(
    policy: &MetadataSignaturePolicy,
    temp_key_path: &Path,
    verify: impl FnOnce(&Path, &'static str) -> Result<T>,
) -> Result<Option<T>> {
    if !policy.active() {
        return Ok(None);
    }
    if let Some(public_key_path) = &policy.public_key_path {
        policy.validate_configuration()?;
        return verify(public_key_path, "path").map(Some);
    }
    if let Some(public_key_pem) = &policy.public_key_pem {
        let staged_key = stage_file_for_atomic_publish(temp_key_path, public_key_pem.as_bytes())?;
        let result = verify(&staged_key, "env-pem");
        let _ = fs::remove_file(staged_key);
        return result.map(Some);
    }
    bail!(
        "metadata signature verification requires ROCM_CLI_METADATA_PUBLIC_KEY_PATH or ROCM_CLI_METADATA_PUBLIC_KEY_PEM"
    )
}

fn metadata_signature_url(url: &str) -> String {
    format!("{url}.sig")
}

fn metadata_signature_path(body_path: &Path) -> PathBuf {
    body_path.with_extension("sig")
}

fn fetch_and_verify_metadata_signature(
    policy: &MetadataSignaturePolicy,
    url: &str,
    payload: &[u8],
    temp_key_path: &Path,
    max_time_secs: Option<u64>,
) -> Result<Option<(CachedHttpSignatureMetadata, Vec<u8>)>> {
    if !policy.active() {
        return Ok(None);
    }
    let signature_url = metadata_signature_url(url);
    let response = http_get(&signature_url, &[], max_time_secs)?;
    if response.status != 200 {
        bail!(
            "HTTP {} while fetching metadata signature {signature_url}",
            response.status
        );
    }
    let signature = response.body;
    let public_key_source =
        with_metadata_public_key(policy, temp_key_path, |public_key, source| {
            verify_metadata_signature_bytes(payload, &signature, public_key)?;
            Ok(source.to_owned())
        })?
        .context("metadata signature policy was active but no public key was resolved")?;
    Ok(Some((
        CachedHttpSignatureMetadata {
            url: signature_url,
            verified_at_unix_ms: unix_time_millis(),
            public_key_source,
        },
        signature,
    )))
}

#[cfg(test)]
fn verify_cached_metadata_signature(
    policy: &MetadataSignaturePolicy,
    payload_path: &Path,
    signature_path: &Path,
    temp_key_path: &Path,
) -> Result<()> {
    if !policy.active() {
        return Ok(());
    }
    if !signature_path.is_file() {
        bail!(
            "metadata signature verification requested but cached signature is missing: {}",
            signature_path.display()
        );
    }
    with_metadata_public_key(policy, temp_key_path, |public_key, _source| {
        verify_metadata_signature(payload_path, signature_path, public_key)
    })?;
    Ok(())
}

#[cfg(test)]
fn verify_metadata_signature(
    payload_path: &Path,
    signature_path: &Path,
    public_key_path: &Path,
) -> Result<()> {
    let public_key_pem = fs::read_to_string(public_key_path).with_context(|| {
        format!(
            "failed to read metadata public key: {}",
            public_key_path.display()
        )
    })?;
    let signature = fs::read(signature_path).with_context(|| {
        format!(
            "failed to read metadata signature: {}",
            signature_path.display()
        )
    })?;
    let payload = fs::read(payload_path).with_context(|| {
        format!(
            "failed to read metadata payload: {}",
            payload_path.display()
        )
    })?;
    verify_rsa_pkcs1_sha256_signature(&public_key_pem, &payload, &signature, "metadata")
}

fn verify_metadata_signature_bytes(
    payload: &[u8],
    signature: &[u8],
    public_key_path: &Path,
) -> Result<()> {
    let public_key_pem = fs::read_to_string(public_key_path).with_context(|| {
        format!(
            "failed to read metadata public key: {}",
            public_key_path.display()
        )
    })?;
    verify_rsa_pkcs1_sha256_signature(&public_key_pem, payload, signature, "metadata")
}

#[cfg(test)]
fn metadata_cache_can_revalidate(
    metadata: &CachedHttpMetadata,
    policy: &MetadataSignaturePolicy,
    signature_path: &Path,
) -> bool {
    !policy.active() || (metadata.signature.is_some() && signature_path.is_file())
}

fn load_cached_http_entry(
    metadata_path: &Path,
    legacy_body_path: &Path,
    legacy_signature_path: &Path,
) -> Option<CachedHttpCacheEntry> {
    let bytes = fs::read(metadata_path).ok()?;
    if let Ok(entry) = serde_json::from_slice(&bytes) {
        return Some(entry);
    }

    let metadata: CachedHttpMetadata = serde_json::from_slice(&bytes).ok()?;
    let body = fs::read_to_string(legacy_body_path).ok()?;
    let signature_bytes = metadata
        .signature
        .as_ref()
        .map(|_| fs::read(legacy_signature_path))
        .transpose()
        .ok()?;
    Some(CachedHttpCacheEntry {
        metadata,
        body,
        signature_bytes,
    })
}

const fn cached_http_entry_can_revalidate(
    entry: &CachedHttpCacheEntry,
    policy: &MetadataSignaturePolicy,
) -> bool {
    !policy.active() || (entry.metadata.signature.is_some() && entry.signature_bytes.is_some())
}

fn verify_cached_http_entry_signature(
    policy: &MetadataSignaturePolicy,
    entry: &CachedHttpCacheEntry,
    temp_key_path: &Path,
) -> Result<()> {
    if !policy.active() {
        return Ok(());
    }
    let signature = entry.signature_bytes.as_deref().context(
        "metadata signature verification requested but cached signature bytes are missing",
    )?;
    with_metadata_public_key(policy, temp_key_path, |public_key, _source| {
        verify_metadata_signature_bytes(entry.body.as_bytes(), signature, public_key)
    })?;
    Ok(())
}

fn write_cached_http_entry(path: &Path, entry: &CachedHttpCacheEntry) -> Result<()> {
    write_file_atomically(
        path,
        &serde_json::to_vec_pretty(entry).context("failed to serialize metadata cache entry")?,
    )
}

#[cfg(test)]
fn write_cached_http_entry_with<S, P, F>(
    path: &Path,
    entry: &CachedHttpCacheEntry,
    suffix_for_attempt: S,
    before_publish: P,
    publish: F,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
    F: FnOnce(&Path, &Path) -> std::io::Result<()>,
{
    write_file_atomically_with_publish(
        path,
        &serde_json::to_vec_pretty(entry).context("failed to serialize metadata cache entry")?,
        suffix_for_attempt,
        before_publish,
        publish,
    )
}

pub(super) fn download_text_cached(
    paths: &AppPaths,
    cache_key: &str,
    url: &str,
    max_time_secs: Option<u64>,
) -> Result<CachedHttpText> {
    let (body_path, metadata_path) = metadata_cache_paths(paths, cache_key);
    let signature_path = metadata_signature_path(&body_path);
    let signature_policy = MetadataSignaturePolicy::from_env();
    signature_policy.validate_configuration()?;
    let previous_entry = load_cached_http_entry(&metadata_path, &body_path, &signature_path)
        .filter(|entry| entry.metadata.url == url);
    let cache_dir = body_path
        .parent()
        .context("metadata cache path has no parent directory")?;
    fs::create_dir_all(cache_dir)?;

    let tmp_public_key = body_path.with_extension("public-key.pem");
    let mut headers = Vec::new();
    if let Some(etag) = previous_entry
        .as_ref()
        .filter(|entry| cached_http_entry_can_revalidate(entry, &signature_policy))
        .and_then(|entry| entry.metadata.etag.as_deref())
    {
        headers.push(("If-None-Match", etag));
    }

    let response = http_get(url, &headers, max_time_secs)?;
    if response.status == 304 {
        let entry = previous_entry.context(
            "metadata cache returned 304 but no complete cached generation is available",
        )?;
        verify_cached_http_entry_signature(&signature_policy, &entry, &tmp_public_key)?;
        return Ok(CachedHttpText { text: entry.body });
    }
    if response.status != 200 {
        bail!("HTTP {} while fetching {url}", response.status);
    }

    let body =
        String::from_utf8(response.body).context("metadata response body was not valid UTF-8")?;
    let fetched_signature = fetch_and_verify_metadata_signature(
        &signature_policy,
        url,
        body.as_bytes(),
        &tmp_public_key,
        max_time_secs,
    )?;
    let signature_metadata = fetched_signature
        .as_ref()
        .map(|(metadata, _)| metadata.clone());
    let metadata = CachedHttpMetadata {
        url: url.to_owned(),
        etag: http_header_value(&response.headers, "etag"),
        last_modified: http_header_value(&response.headers, "last-modified"),
        signature: signature_metadata,
        fetched_at_unix_ms: unix_time_millis(),
    };
    let entry = CachedHttpCacheEntry {
        metadata,
        body,
        signature_bytes: fetched_signature.map(|(_, signature)| signature),
    };
    write_cached_http_entry(&metadata_path, &entry)?;
    let _ = fs::remove_file(&body_path);
    let _ = fs::remove_file(&signature_path);
    Ok(CachedHttpText { text: entry.body })
}

fn metadata_cache_paths(paths: &AppPaths, cache_key: &str) -> (PathBuf, PathBuf) {
    let base = paths
        .cache_dir
        .join("therock")
        .join("metadata")
        .join(slugify(cache_key));
    (base.with_extension("body"), base.with_extension("json"))
}

fn http_header_value(headers: &str, name: &str) -> Option<String> {
    let prefix = format!("{}:", name.to_ascii_lowercase());
    let mut value = None;
    for line in headers.lines() {
        let trimmed = line.trim();
        if trimmed.to_ascii_lowercase().starts_with(&prefix) {
            value = trimmed
                .split_once(':')
                .map(|(_, rest)| rest.trim().to_owned())
                .filter(|rest| !rest.is_empty());
        }
    }
    value
}

/// Fetch an artifact to `destination`, reporting cumulative bytes written and
/// (when known) the total size to `on_progress` as the transfer proceeds.
///
/// Streams rather than buffers: SDK tarballs are single-digit gigabytes, and
/// holding one in memory to write it out again costs that much RAM on top of
/// the same amount of disk. The primitive also handles the free-space
/// preflight, retry with resume, and the length cross-check that catches a
/// transfer the server ended early.
pub(super) fn download_file(
    url: &str,
    destination: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<()> {
    let parent = destination
        .parent()
        .context("download destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    let request = rocm_core::DownloadRequest {
        max_bytes: Some(THEROCK_MAX_PLAUSIBLE_TARBALL_BYTES),
        ..rocm_core::DownloadRequest::new(url, destination, THEROCK_DOWNLOAD_TIMEOUT)
    };
    rocm_core::download_file_streaming_with_progress(&request, on_progress)
        .with_context(|| format!("failed to fetch {url}"))?;
    Ok(())
}

/// Content length of `url` from a HEAD request, when the server reports one.
///
/// Best effort: any failure yields `None`, so a server that rejects HEAD or
/// omits `Content-Length` simply skips the preflight instead of blocking the
/// install.
fn head_content_length(url: &str) -> Option<u64> {
    let timeout = Duration::from_secs(THEROCK_HEAD_PROBE_TIMEOUT_SECS);
    let agent = ureq::AgentBuilder::new()
        // `timeout_connect` takes precedence over `timeout` and defaults to 30s,
        // so without it a host that blackholes rather than refuses would stall
        // the probe well past the intended ceiling.
        .timeout_connect(timeout)
        .timeout(timeout)
        .build();
    let response = agent.head(url).set("User-Agent", "rocm-cli").call().ok()?;
    if response.status() != 200 {
        return None;
    }
    let length: u64 = response.header("Content-Length")?.trim().parse().ok()?;
    // The header is unauthenticated and is never cross-checked against the body
    // the subsequent GET delivers, so an inflated value from a proxy or CDN
    // would refuse an install that would in fact succeed. Treat an implausible
    // size as no answer at all: the preflight is skipped and `download_file`
    // still checks the real, buffered body length before writing.
    (length <= THEROCK_MAX_PLAUSIBLE_TARBALL_BYTES).then_some(length)
}

/// Refuse (or warn) before a multi-GB SDK tarball download and extraction.
///
/// The download requirement comes from `Content-Length` and is exact, so a
/// shortfall is a hard error — it saves the user a long download that cannot
/// possibly succeed. The extraction requirement is only an estimate (see
/// [`disk_space::EXTRACTED_SIZE_MULTIPLIER`]), so a shortfall there is a
/// warning: a false refusal that blocks a valid install would be worse than a
/// late failure.
///
/// Any extraction warning is returned rather than printed, so the caller can
/// place it in the same accumulated output block as the rest of the install
/// report instead of having it appear ahead of that block.
pub(super) fn preflight_tarball_space(
    url: &str,
    cache_path: &Path,
    install_root: &Path,
) -> Result<Option<String>> {
    let Some(download_bytes) = head_content_length(url) else {
        return Ok(None);
    };
    disk_space::ensure_space_for(
        "download the SDK tarball",
        cache_path,
        disk_space::with_margin(download_bytes),
    )?;

    // When the cache and the install root share a mount, the archive and the
    // extracted tree must both fit at the same time. Conservative in the other
    // direction: two mounts backed by one filesystem also share a pool, and this
    // treats them as separate, so the estimate can come in under the true need.
    let mut extract_estimate = disk_space::estimated_extracted_size(download_bytes);
    if disk_space::on_same_mount(cache_path, install_root) == Some(true) {
        extract_estimate = extract_estimate.saturating_add(download_bytes);
    }
    Ok(disk_space::warn_if_low_space(
        "extract the SDK tarball",
        install_root,
        disk_space::with_margin(extract_estimate),
    ))
}

fn http_get(
    url: &str,
    headers: &[(&str, &str)],
    max_time_secs: Option<u64>,
) -> Result<HttpResponseBody> {
    let timeout = max_time_secs
        .filter(|value| *value > 0)
        .map_or_else(|| Duration::from_mins(10), Duration::from_secs);
    // Connecting should always be fast if the host is reachable at all, so cap it
    // independently of the (possibly very generous, e.g. 10-minute default) overall
    // timeout. Without this, a blackholed host stalls the connect phase for the full
    // overall timeout on every request instead of failing fast.
    let connect_timeout = timeout.min(Duration::from_secs(THEROCK_HEAD_PROBE_TIMEOUT_SECS));
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(connect_timeout)
        .timeout(timeout)
        .build();
    let mut request = agent.get(url).set("User-Agent", "rocm-cli");
    for (name, value) in headers {
        request = request.set(name, value);
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(_, response)) => response,
        Err(error) => bail!("HTTP request failed for {url}: {error}"),
    };
    let status = response.status();
    let headers = response
        .headers_names()
        .into_iter()
        .filter_map(|name| {
            response
                .header(&name)
                .map(|value| format!("{name}: {value}"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut reader = response
        .into_reader()
        .take(THEROCK_MAX_METADATA_BYTES.saturating_add(1));
    let mut body = Vec::new();
    reader
        .read_to_end(&mut body)
        .with_context(|| format!("failed to read HTTP response body for {url}"))?;
    if body.len() as u64 > THEROCK_MAX_METADATA_BYTES {
        bail!(
            "HTTP response body for {url} exceeded the approved metadata limit of {THEROCK_MAX_METADATA_BYTES} bytes"
        );
    }
    Ok(HttpResponseBody {
        status,
        headers,
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::{generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature};
    use std::io::Write;

    fn workspace_test_artifact_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("therock")
    }

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-cli-fetch-test-{name}-{}",
            crate::test_support::unique_suffix()
        ));
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    #[test]
    fn tarball_space_preflight_skips_when_the_download_size_is_unknown() {
        // No HEAD response (unroutable host) must not block an install.
        let temp = std::env::temp_dir();
        let warning = preflight_tarball_space("http://127.0.0.1:1/rocm.tar.gz", &temp, &temp)
            .expect("an unknown download size must not fail the preflight");
        assert_eq!(
            warning, None,
            "an unknown download size must not produce an extraction warning either"
        );
    }

    fn cached_http_entry(generation: &str) -> CachedHttpCacheEntry {
        CachedHttpCacheEntry {
            metadata: CachedHttpMetadata {
                url: format!("https://example.invalid/{generation}"),
                etag: Some(format!("etag-{generation}")),
                last_modified: None,
                signature: Some(CachedHttpSignatureMetadata {
                    url: format!("https://example.invalid/{generation}.sig"),
                    verified_at_unix_ms: 1,
                    public_key_source: generation.to_owned(),
                }),
                fetched_at_unix_ms: 2,
            },
            body: format!("body-{generation}"),
            signature_bytes: Some(generation.as_bytes().to_vec()),
        }
    }

    #[test]
    fn concurrent_cached_http_commits_publish_one_complete_generation() {
        let root = std::env::temp_dir().join(format!(
            "rocm-cache-generation-collision-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.json");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = ["first", "second"]
            .into_iter()
            .enumerate()
            .map(|(writer, generation)| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    let entry = cached_http_entry(generation);
                    write_cached_http_entry_with(
                        &destination,
                        &entry,
                        |attempt| {
                            if attempt == 0 {
                                OsString::from("same-millisecond")
                            } else {
                                OsString::from(format!("same-millisecond-{writer}-{attempt}"))
                            }
                        },
                        || {
                            before_publish.wait();
                        },
                        publish_temp_file,
                    )
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published: CachedHttpCacheEntry =
            serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);

        assert!(
            published == cached_http_entry("first") || published == cached_http_entry("second"),
            "published cache mixed generations: {published:?}"
        );
        assert!(leftovers.is_empty(), "cache commit leaked: {leftovers:?}");
    }

    #[test]
    fn failed_cached_http_commit_preserves_previous_complete_generation() {
        let root = std::env::temp_dir().join(format!(
            "rocm-cache-generation-failure-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.json");
        let previous = cached_http_entry("previous");
        write_cached_http_entry(&destination, &previous).unwrap();

        write_cached_http_entry_with(
            &destination,
            &cached_http_entry("replacement"),
            |attempt| OsString::from(format!("commit-failure-{attempt}")),
            || {},
            |tmp, path| {
                let staged: CachedHttpCacheEntry =
                    serde_json::from_slice(&fs::read(tmp).unwrap()).unwrap();
                assert_eq!(staged, cached_http_entry("replacement"));
                assert_eq!(path, destination);
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "simulated cache generation commit failure",
                ))
            },
        )
        .expect_err("simulated commit failure must be returned");

        let preserved: CachedHttpCacheEntry =
            serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(preserved, previous);
        assert_eq!(leftovers, vec![OsString::from("index.json")]);
    }

    #[test]
    fn metadata_cache_paths_stay_under_rocm_cli_cache() {
        let (root, paths) = test_paths("metadata-cache-paths");
        let (body, metadata) = metadata_cache_paths(&paths, "simple-index:https://example.invalid");

        assert!(body.starts_with(paths.cache_dir.join("therock").join("metadata")));
        assert_eq!(
            body.extension().and_then(|value| value.to_str()),
            Some("body")
        );
        assert_eq!(
            metadata.extension().and_then(|value| value.to_str()),
            Some("json")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_signature_paths_are_sidecars() {
        let (root, paths) = test_paths("metadata-signature-paths");
        let (body, _) = metadata_cache_paths(&paths, "simple-index:https://example.invalid");

        assert_eq!(
            metadata_signature_url("https://example.invalid/index").as_str(),
            "https://example.invalid/index.sig"
        );
        assert_eq!(
            metadata_signature_path(&body)
                .extension()
                .and_then(|value| value.to_str()),
            Some("sig")
        );
        assert!(metadata_signature_path(&body).starts_with(&paths.cache_dir));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_signature_policy_requires_public_key_when_enabled() {
        let (root, paths) = test_paths("metadata-signature-requires-key");
        let policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: None,
            public_key_pem: None,
        };
        let temp_key = paths.cache_dir.join("metadata-key.pem");

        let error = with_metadata_public_key(&policy, &temp_key, |_path, _source| Ok(()))
            .unwrap_err()
            .to_string();

        assert!(error.contains("ROCM_CLI_METADATA_PUBLIC_KEY_PATH"));
        assert!(error.contains("ROCM_CLI_METADATA_PUBLIC_KEY_PEM"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_signature_policy_writes_inline_public_key_temporarily() -> Result<()> {
        let (root, paths) = test_paths("metadata-signature-inline-key");
        let policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: None,
            public_key_pem: Some(
                "-----BEGIN PUBLIC KEY-----\nunit-test\n-----END PUBLIC KEY-----\n".to_owned(),
            ),
        };
        let temp_key = paths.cache_dir.join("metadata-key.pem");

        let observed = with_metadata_public_key(&policy, &temp_key, |path, source| {
            assert_eq!(source, "env-pem");
            assert_eq!(
                fs::read_to_string(path)?,
                "-----BEGIN PUBLIC KEY-----\nunit-test\n-----END PUBLIC KEY-----\n"
            );
            Ok(path.to_path_buf())
        })?
        .expect("inline key should be active");

        assert_eq!(observed.parent(), temp_key.parent());
        assert!(
            observed
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("metadata-key.pem.tmp-")),
            "inline key must use a reserved sibling path: {}",
            observed.display()
        );
        assert!(!observed.exists());
        assert!(!temp_key.exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn metadata_policy_uses_pinned_key_and_requires_by_default() {
        // A pinned trust root with no env inputs: verification becomes required
        // by default and the pinned PEM is used.
        let pinned = "-----BEGIN PUBLIC KEY-----\npinned\n-----END PUBLIC KEY-----\n";
        let policy = MetadataSignaturePolicy::resolve(false, None, None, Some(pinned.to_owned()));
        assert!(policy.required);
        assert!(policy.active());
        assert_eq!(policy.public_key_pem.as_deref(), Some(pinned));
        assert!(policy.public_key_path.is_none());
    }

    #[test]
    fn metadata_policy_env_key_overrides_pinned_key() {
        // An explicit env PEM wins over the pinned root (escape hatch), and does
        // not force `required` on its own.
        let pinned = "-----BEGIN PUBLIC KEY-----\npinned\n-----END PUBLIC KEY-----\n";
        let env_pem = "-----BEGIN PUBLIC KEY-----\nenv\n-----END PUBLIC KEY-----\n";
        let policy = MetadataSignaturePolicy::resolve(
            false,
            None,
            Some(env_pem.to_owned()),
            Some(pinned.to_owned()),
        );
        assert_eq!(policy.public_key_pem.as_deref(), Some(env_pem));

        let env_path = PathBuf::from("/keys/metadata.pem");
        let policy = MetadataSignaturePolicy::resolve(
            false,
            Some(env_path.clone()),
            None,
            Some(pinned.to_owned()),
        );
        assert_eq!(policy.public_key_path, Some(env_path));
        assert!(policy.public_key_pem.is_none());
    }

    #[test]
    fn metadata_policy_without_pinned_key_preserves_optin_behavior() {
        // Empty pinned sentinel + no env inputs: verification stays inactive,
        // exactly as before pinning was introduced.
        let policy = MetadataSignaturePolicy::resolve(false, None, None, None);
        assert!(!policy.required);
        assert!(!policy.active());

        // `ROCM_CLI_REQUIRE_METADATA_SIGNATURE=1` alone still activates it.
        let policy = MetadataSignaturePolicy::resolve(true, None, None, None);
        assert!(policy.required);
    }

    #[test]
    fn metadata_cache_revalidation_requires_cached_signature_when_policy_is_active() -> Result<()> {
        let (root, paths) = test_paths("metadata-signature-revalidate");
        fs::create_dir_all(&paths.cache_dir)?;
        let signature_path = paths.cache_dir.join("index.sig");
        let unsigned_metadata = CachedHttpMetadata {
            url: "https://example.invalid/index".to_owned(),
            etag: Some("etag".to_owned()),
            last_modified: None,
            signature: None,
            fetched_at_unix_ms: 1,
        };
        let signed_metadata = CachedHttpMetadata {
            signature: Some(CachedHttpSignatureMetadata {
                url: "https://example.invalid/index.sig".to_owned(),
                verified_at_unix_ms: 2,
                public_key_source: "path".to_owned(),
            }),
            ..unsigned_metadata.clone()
        };
        let inactive_policy = MetadataSignaturePolicy::default();
        let active_policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: None,
            public_key_pem: Some("key".to_owned()),
        };

        assert!(metadata_cache_can_revalidate(
            &unsigned_metadata,
            &inactive_policy,
            &signature_path
        ));
        assert!(!metadata_cache_can_revalidate(
            &unsigned_metadata,
            &active_policy,
            &signature_path
        ));
        assert!(!metadata_cache_can_revalidate(
            &signed_metadata,
            &active_policy,
            &signature_path
        ));

        fs::write(&signature_path, "signature")?;
        assert!(metadata_cache_can_revalidate(
            &signed_metadata,
            &active_policy,
            &signature_path
        ));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn metadata_signature_verification_accepts_generated_key_and_rejects_tamper() -> Result<()> {
        let (root, paths) = test_paths("metadata-signature-generated-key");
        fs::create_dir_all(&paths.cache_dir)?;
        let private_key = paths.cache_dir.join("metadata-private.pem");
        let public_key = paths.cache_dir.join("metadata-public.pem");
        let payload_path = paths.cache_dir.join("index.body");
        let signature_path = paths.cache_dir.join("index.sig");
        let temp_key = paths.cache_dir.join("metadata-public.tmp.pem");

        generate_test_signing_key(&private_key, &public_key)?;
        fs::write(&payload_path, "version = 1\n")?;
        sign_test_payload(&private_key, &payload_path, &signature_path)?;

        let policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: Some(public_key),
            public_key_pem: None,
        };
        verify_cached_metadata_signature(&policy, &payload_path, &signature_path, &temp_key)?;

        fs::write(&payload_path, "version = 2\n")?;
        let error =
            verify_cached_metadata_signature(&policy, &payload_path, &signature_path, &temp_key)
                .unwrap_err()
                .to_string();

        assert!(error.contains("metadata signature verification failed"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn http_header_value_uses_last_response_header_block() {
        let headers =
            "HTTP/2 302\r\netag: old\r\n\r\nHTTP/2 200\r\nETag: new\r\nLast-Modified: today\r\n";

        assert_eq!(http_header_value(headers, "etag").as_deref(), Some("new"));
        assert_eq!(
            http_header_value(headers, "last-modified").as_deref(),
            Some("today")
        );
    }

    #[test]
    fn native_http_download_and_get_round_trip_without_powershell() -> Result<()> {
        use std::net::TcpListener;
        use std::thread;

        // Serve a fixed body from a localhost HTTP/1.1 server so the request
        // exercises the native `ureq` transport that `http_get`/`download_file`
        // now use on every platform. This runs under `cargo test` on the
        // windows-latest CI job, where `runtime_is_windows()` is true and the
        // removed PowerShell/ExecutionPolicy-Bypass backend used to run — so it
        // verifies the native replacement on real Windows.
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let body = b"native-http-smoke-body".to_vec();
        let served = body.clone();
        // Two requests: one for download_file, one for http_get.
        let server = thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 1024];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    served.len()
                )?;
                stream.write_all(&served)?;
                stream.flush()?;
            }
            Ok(())
        });

        let url = format!("http://127.0.0.1:{port}/artifact.bin");

        let temp = workspace_test_artifact_dir().join(format!(
            "native-http-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&temp)?;
        let destination = temp.join("artifact.bin");

        download_file(&url, &destination, &mut |_, _| {})?;
        assert_eq!(fs::read(&destination)?, body);

        let response = http_get(&url, &[], Some(5))?;
        assert_eq!(response.status, 200);
        assert_eq!(response.body, body);

        server.join().expect("localhost server thread panicked")?;
        let _ = fs::remove_dir_all(&temp);
        Ok(())
    }

    #[test]
    fn http_get_respects_max_time_secs() {
        use std::net::TcpListener;
        use std::thread;
        use std::time::Instant;

        // Regression test for `http_get`'s `max_time_secs` bound, pre-existing
        // and unchanged by this PR, which `download_timeout_secs` relies on
        // once threaded through `resolve_latest_for_manifest`: `http_get` must
        // actually bound the request to `max_time_secs`, not just accept the
        // argument and fall back to the 10-minute default. A listener that
        // accepts the connection but never writes a response simulates a
        // stalled server past the connect phase, so this exercises the
        // overall `timeout` (what `max_time_secs` controls), not just
        // `connect_timeout` (fixed at `THEROCK_HEAD_PROBE_TIMEOUT_SECS`).
        //
        // This does not exercise `resolve_latest_for_manifest` itself: its
        // wheel/tarball index URLs come from `canonical_source`, a `const
        // fn` over fixed real hostnames with no test-time override, so a
        // hermetic test can't reach that exact call site without either
        // hitting the real network or adding a production-code test seam.
        // `max_time_secs`/`download_timeout_secs` is a single value passed
        // unchanged through plain pass-through parameters down to here (no
        // branching on it in between), so a regression in its plumbing would
        // either fail to compile (type mismatch) or show up as this test
        // hanging instead of returning quickly.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let _server = thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                thread::sleep(Duration::from_secs(5));
                drop(stream);
            }
        });

        let url = format!("http://127.0.0.1:{port}/stalled");
        let started = Instant::now();
        let result = http_get(&url, &[], Some(1));
        let elapsed = started.elapsed();

        assert!(
            result.is_err(),
            "a stalled server must not be treated as a successful response"
        );
        assert!(
            elapsed < Duration::from_secs(4),
            "max_time_secs=Some(1) must bound the request; took {elapsed:?}"
        );
    }

    #[test]
    fn download_file_reports_cumulative_progress_to_its_caller() -> Result<()> {
        use std::net::TcpListener;
        use std::thread;

        // A multi-chunk body (`download_file_streaming` reads in 64 KiB
        // chunks) so a single callback firing wouldn't already satisfy the
        // "monotonically increasing" assertion below.
        let body: Vec<u8> = (0..200_000_u32).map(|i| (i % 256) as u8).collect();

        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let served = body.clone();
        let server = thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            )?;
            stream.write_all(&served)?;
            stream.flush()?;
            Ok(())
        });

        let url = format!("http://127.0.0.1:{port}/artifact.bin");

        let temp = workspace_test_artifact_dir().join(format!(
            "download-progress-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&temp)?;
        let destination = temp.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        download_file(&url, &destination, &mut |bytes, total| {
            calls.push((bytes, total));
        })?;
        assert_eq!(fs::read(&destination)?, body);

        server.join().expect("localhost server thread panicked")?;
        let _ = fs::remove_dir_all(&temp);

        let total = Some(body.len() as u64);
        assert!(
            calls.len() >= 2,
            "expected at least a pre-transfer and a final callback: {calls:?}"
        );
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "byte counts must never regress: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last callback must report the complete transfer: {calls:?}"
        );
        assert!(
            calls
                .iter()
                .all(|&(_, reported_total)| reported_total == total),
            "the reported total must stay consistent across callbacks: {calls:?}"
        );
        Ok(())
    }

    fn generate_test_signing_key(private_key: &Path, public_key: &Path) -> Result<()> {
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        fs::write(private_key, private_pem.as_bytes())?;
        fs::write(public_key, public_pem.as_bytes())?;
        Ok(())
    }

    fn sign_test_payload(private_key: &Path, payload: &Path, signature: &Path) -> Result<()> {
        let private_pem = fs::read_to_string(private_key)?;
        let payload_bytes = fs::read(payload)?;
        let produced = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload_bytes)?;
        fs::write(signature, produced)?;
        Ok(())
    }
}
