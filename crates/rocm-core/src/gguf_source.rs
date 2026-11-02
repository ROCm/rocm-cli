// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Where a GGUF header comes from: a local file, or a Hugging Face repository
//! read by HTTP range requests.
//!
//! A remote file is found by one repository-listing request (plus a `HEAD`
//! request for a file the listing does not size), then read in windows.
//!
//! The point of the range reader is that a model's *shape* is in the first few
//! megabytes of a file that may be tens of gigabytes. [`RangedReader`] presents
//! the remote file as `Read + Seek` and fetches it in 4 MiB windows on demand, so
//! [`crate::gguf::parse_gguf_header`] stops reading where the header ends and
//! the weights are never requested. A server that ignores `Range` is refused
//! rather than streamed, because streaming it is exactly the download this
//! exists to avoid.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How much of the remote file one request asks for.
const WINDOW_BYTES: u64 = 4 * 1024 * 1024;
/// The most a header read may fetch in total. Real headers are a few megabytes
/// (mostly the tokenizer); a read that goes beyond this is not reading a header.
const MAX_FETCH_BYTES: u64 = crate::gguf::MAX_HEADER_BYTES;
/// A repository listing larger than this is not read.
const MAX_LISTING_BYTES: u64 = 16 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_mins(1);
/// How long listing a repository and reading a header may take in all. The
/// per-request timeout alone would let sixteen slow windows hold the command for
/// sixteen minutes.
const TOTAL_BUDGET: Duration = Duration::from_mins(2);
const DEFAULT_HUB: &str = "https://huggingface.co";

/// A model reference that names a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufRef {
    /// A path on this machine.
    Local(PathBuf),
    /// `owner/repo` on Hugging Face, with the quantization label or file name
    /// after a colon. The variant is optional here so that a reference without
    /// one can be refused with a message, not mistaken for something else.
    Hub {
        owner: String,
        repo: String,
        variant: Option<String>,
    },
}

/// Whether a string is a safe Hugging Face path component: the charset of
/// `is_safe_hf_component` in the lemonade engine, which is what `rocm serve`
/// applies to `owner/repo:variant`, so a reference serve would refuse is not sent
/// to the hub either.
fn is_safe_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && component
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Whether a string is a valid `:variant`: a quantization label, or a file name
/// ending in `.gguf` that may carry a repository sub-directory
/// (`dir/model-Q4_K_M.gguf`), which is the only way to name one of two files that
/// share a name in different directories. `rocm serve` takes no `/` in a variant;
/// here it is safe because every segment is checked and the whole is sent to the
/// hub as percent-encoded path segments, never to the filesystem.
fn is_safe_variant(variant: &str) -> bool {
    if is_safe_component(variant) {
        return true;
    }
    variant.to_ascii_lowercase().ends_with(".gguf")
        && variant.split('/').count() <= 8
        && variant.split('/').all(is_safe_component)
}

/// Read a model reference as a GGUF file, if it is shaped like one.
///
/// A file that exists on this machine is that file, whatever its name looks
/// like. Otherwise `owner/repo:variant` is a Hugging Face repository, a name
/// ending in `.gguf` is a local path (that does not exist, so it is reported as
/// not openable rather than sent to the hub), and a bare `owner/repo` is a
/// repository with no variant, recognised so the caller can say a quantization is
/// needed. Returns `None` for names, aliases and anything else, so a catalog model
/// never reaches this module.
#[must_use]
pub fn parse_gguf_ref(model_ref: &str) -> Option<GgufRef> {
    parse_gguf_ref_with(model_ref, &|path| path.is_file())
}

/// [`parse_gguf_ref`] with the "does this file exist" question answered by the
/// caller, so a test can say a path exists without a working directory to put it
/// in.
fn parse_gguf_ref_with(model_ref: &str, is_file: &dyn Fn(&Path) -> bool) -> Option<GgufRef> {
    let trimmed = model_ref.trim();
    if trimmed.is_empty() {
        return None;
    }
    if is_file(Path::new(trimmed)) {
        return Some(GgufRef::Local(PathBuf::from(trimmed)));
    }
    let is_gguf_name = |name: &str| name.to_ascii_lowercase().ends_with(".gguf");
    if let Some((repo_path, variant)) = trimmed.split_once(':')
        && let Some((owner, repo)) = repo_path.split_once('/')
        && is_safe_component(owner)
        && is_safe_component(repo)
        && is_safe_variant(variant)
    {
        return Some(GgufRef::Hub {
            owner: owner.to_owned(),
            repo: repo.to_owned(),
            variant: Some(variant.to_owned()),
        });
    }
    if is_gguf_name(trimmed) {
        return Some(GgufRef::Local(PathBuf::from(trimmed)));
    }
    let (owner, repo) = trimmed.split_once('/')?;
    (is_safe_component(owner) && is_safe_component(repo)).then(|| GgufRef::Hub {
        owner: owner.to_owned(),
        repo: repo.to_owned(),
        variant: None,
    })
}

/// A file of a Hugging Face repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubFile {
    pub name: String,
    /// The size the listing reported; `None` when it did not.
    pub size: Option<u64>,
}

fn is_boundary(c: Option<char>, extra: &[char]) -> bool {
    c.is_none_or(|c| matches!(c, '-' | '.' | '/') || extra.contains(&c))
}

/// Whether `label` appears in `name` as a whole quantization label: bounded by
/// separators, so `Q4_0` does not match `Q4_0_4_4`.
fn has_label(name: &str, label: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let label = label.to_ascii_lowercase();
    name.match_indices(&label).any(|(start, _)| {
        let before = name[..start].chars().next_back();
        let after = name[start + label.len()..].chars().next();
        is_boundary(before, &['_']) && is_boundary(after, &[])
    })
}

/// Pick the GGUF file a `variant` names out of a repository's files.
///
/// An exact file name, or a quantization label as `rocm serve` takes it. Never
/// picks between two quantizations: a label that matches several files is an
/// error listing them, except that the shards of one split model resolve to
/// their first shard.
///
/// # Errors
///
/// When nothing matches, or the match is ambiguous.
pub fn select_gguf_file<'a>(files: &'a [HubFile], variant: &str) -> Result<&'a HubFile> {
    let ggufs: Vec<&HubFile> = files
        .iter()
        .filter(|file| file.name.to_ascii_lowercase().ends_with(".gguf"))
        .filter(|file| !file.name.to_ascii_lowercase().contains("mmproj"))
        .collect();
    let matches: Vec<&HubFile> = if variant.to_ascii_lowercase().ends_with(".gguf") {
        // A name is unique in a repository, so one that matches in full is the
        // file asked for, even where another file in a directory has the same
        // last component (`m.gguf` at the root, `dir/m.gguf` below it).
        let exact: Vec<&HubFile> = ggufs
            .iter()
            .copied()
            .filter(|file| file.name == variant)
            .collect();
        if exact.is_empty() {
            ggufs
                .iter()
                .copied()
                .filter(|file| file.name.rsplit('/').next() == Some(variant))
                .collect()
        } else {
            exact
        }
    } else {
        ggufs
            .iter()
            .copied()
            .filter(|file| has_label(&file.name, variant))
            .collect()
    };
    match matches.as_slice() {
        [only] => Ok(only),
        [] => bail!(
            "no GGUF file matching `{variant}` in the repository (it has: {})",
            names_for_error(&ggufs)
        ),
        several => first_shard_of_one_set(several).ok_or_else(|| {
            anyhow!(
                "`{variant}` matches several files ({}); name one exactly, as \
                 `owner/repo:file.gguf` (or `owner/repo:dir/file.gguf` when the name is not \
                 unique)",
                names_for_error(several)
            )
        }),
    }
}

/// Split a shard's name `<stem>-00002-of-00004.gguf` into what the shards of one
/// model share (the stem and the total) and the shard's own index.
fn shard_parts(name: &str) -> Option<(String, u32)> {
    let of = name.find("-of-")?;
    let digits = |text: &str| -> Option<u32> {
        (text.len() == 5 && text.bytes().all(|b| b.is_ascii_digit()))
            .then(|| text.parse().ok())
            .flatten()
    };
    let index_start = of.checked_sub(5)?;
    let index = digits(name.get(index_start..of)?)?;
    let stem = name.get(..index_start)?.strip_suffix('-')?;
    let rest = name.get(of + 4..)?;
    let total_end = rest.find('.').unwrap_or(rest.len());
    let total = digits(rest.get(..total_end)?)?;
    Some((format!("{stem}|{total}|{}", &rest[total_end..]), index))
}

/// The first shard, when every candidate is a shard of the same split model.
/// Shards that belong to different models (two sets under one label) are
/// ambiguous, not one model: taking "the" first shard would size the wrong one.
fn first_shard_of_one_set<'a>(candidates: &[&'a HubFile]) -> Option<&'a HubFile> {
    let parts: Vec<(String, u32)> = candidates
        .iter()
        .map(|file| shard_parts(&file.name))
        .collect::<Option<_>>()?;
    let set = &parts.first()?.0;
    if parts.iter().any(|(other, _)| other != set) {
        return None;
    }
    candidates
        .iter()
        .zip(&parts)
        .find(|(_, (_, index))| *index == 1)
        .map(|(file, _)| *file)
}

fn names_for_error(files: &[&HubFile]) -> String {
    const SHOWN: usize = 8;
    let mut names: Vec<&str> = files.iter().take(SHOWN).map(|f| f.name.as_str()).collect();
    if files.len() > SHOWN {
        names.push("...");
    }
    if names.is_empty() {
        "no GGUF files".to_owned()
    } else {
        names.join(", ")
    }
}

/// `Read + Seek`, as one object-safe trait.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

/// A GGUF that has been opened for header reading.
pub struct OpenedGguf {
    pub reader: Box<dyn ReadSeek>,
    /// The size of the whole file, in bytes.
    pub size: u64,
    /// What to call the file in a report.
    pub description: String,
}

/// Opens a GGUF named by a [`GgufRef`]. A trait so the estimate can be tested
/// without a network, and so the one real implementation is the only place that
/// reaches for one.
pub trait GgufSource {
    /// # Errors
    ///
    /// When the file cannot be found or reached.
    fn open(&self, reference: &GgufRef) -> Result<OpenedGguf>;
}

/// The real source: the local filesystem and Hugging Face.
#[derive(Debug, Clone)]
pub struct LiveGgufSource {
    hub: String,
}

impl Default for LiveGgufSource {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveGgufSource {
    /// Uses `HF_ENDPOINT` as the Hugging Face base URL when it is set (the
    /// variable the Hugging Face tooling reads), else `https://huggingface.co`.
    #[must_use]
    pub fn new() -> Self {
        let hub = std::env::var("HF_ENDPOINT")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_HUB.to_owned());
        Self { hub }
    }

    /// A source against an explicit hub base URL.
    #[must_use]
    pub fn with_hub(hub: &str) -> Self {
        Self {
            hub: hub.trim_end_matches('/').to_owned(),
        }
    }

    fn agent() -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
    }

    fn list_files(
        &self,
        agent: &ureq::Agent,
        owner: &str,
        repo: &str,
        deadline: Instant,
    ) -> Result<Vec<HubFile>> {
        let url = format!("{}/api/models/{owner}/{repo}?blobs=true", self.hub);
        let response = agent
            .get(&url)
            .timeout(time_left(deadline).map_err(|error| anyhow!("{error}"))?)
            .call()
            .map_err(|error| describe_http_error(&error, &format!("`{owner}/{repo}`")))?;
        let mut text = String::new();
        // One byte past the cap, so a listing that is exactly too long is told
        // apart from one that fits instead of being cut short and misread as bad
        // JSON.
        response
            .into_reader()
            .take(MAX_LISTING_BYTES + 1)
            .read_to_string(&mut text)
            .with_context(|| format!("could not read the file list of `{owner}/{repo}`"))?;
        if text.len() as u64 > MAX_LISTING_BYTES {
            bail!(
                "the file list of `{owner}/{repo}` is larger than {} MiB, which is not read",
                MAX_LISTING_BYTES / (1024 * 1024)
            );
        }
        let body: Value = serde_json::from_str(&text)
            .with_context(|| format!("the file list of `{owner}/{repo}` was not JSON"))?;
        let siblings = body
            .get("siblings")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("the file list of `{owner}/{repo}` has no `siblings`"))?;
        // A name the hub lists is put into a request path, so a name that could
        // climb out of the repository (`..`), or hide in a terminal (a control,
        // bidirectional or zero-width character), is dropped here rather than
        // offered, requested or printed.
        // A real repository has none; a hostile mirror is what could.
        Ok(siblings
            .iter()
            .filter_map(|sibling| {
                let name = sibling.get("rfilename")?.as_str()?;
                is_safe_listed_name(name).then(|| HubFile {
                    name: name.to_owned(),
                    size: sibling.get("size").and_then(Value::as_u64),
                })
            })
            .collect())
    }
}

/// Whether a file name from a repository listing may be requested and shown: no
/// empty, `.` or `..` path segment, and no character that changes how a terminal
/// shows the text around it (a control character, a bidirectional override or
/// isolate, or a zero-width character).
fn is_safe_listed_name(name: &str) -> bool {
    !name.is_empty()
        && !name.chars().any(|c| {
            // Code points, not char literals: the ASCII output contract
            // (`xtask/src/ascii_output_contract.rs`) rejects a `\u{..}` escape
            // for a non-ASCII character in a literal.
            c.is_control()
                || matches!(
                    u32::from(c),
                    0x200B..=0x200F | 0x202A..=0x202E | 0x2066..=0x2069 | 0xFEFF
                )
        })
        && name
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn describe_http_error(error: &ureq::Error, what: &str) -> anyhow::Error {
    match error {
        // The hub answers 401 for a repository that does not exist as well as
        // for one that is gated or private, so the two cannot be told apart. No
        // token is ever sent, so authentication is not something to offer.
        ureq::Error::Status(401 | 403 | 404, _) => anyhow!(
            "{what} was not found, or its repository is gated or private; no token is sent, so \
             only public repositories can be read"
        ),
        ureq::Error::Status(code, _) => anyhow!("the hub answered {code} for {what}"),
        ureq::Error::Transport(transport) => {
            anyhow!("could not reach the hub for {what}: {transport}")
        }
    }
}

/// Percent-encode one path segment of a repository file name.
fn encode_segment(segment: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

/// What to say when a path that does not exist is shaped like a file in a Hugging
/// Face repository (`owner/repo/dir/file.gguf`): the hub form of the same name.
/// Empty for anything else.
fn hub_hint(path: &Path) -> String {
    let Some(text) = path.to_str() else {
        return String::new();
    };
    let parts: Vec<&str> = text.split('/').collect();
    if path.is_absolute()
        || parts.len() < 3
        || !text.to_ascii_lowercase().ends_with(".gguf")
        || !parts.iter().all(|part| is_safe_component(part))
    {
        return String::new();
    }
    format!(
        " (if you meant a file in a Hugging Face repository, write it `{}/{}:{}`, with a colon \
         after the repository)",
        parts[0],
        parts[1],
        parts[2..].join("/")
    )
}

impl GgufSource for LiveGgufSource {
    fn open(&self, reference: &GgufRef) -> Result<OpenedGguf> {
        match reference {
            GgufRef::Local(path) => {
                // Checked before opening: opening a named pipe blocks until
                // something writes to it, and a directory opens and then fails
                // on its first read.
                let metadata = std::fs::metadata(path).with_context(|| {
                    format!("could not open {}{}", path.display(), hub_hint(path))
                })?;
                if !metadata.is_file() {
                    bail!("{} is not a regular file", path.display());
                }
                let file = File::open(path)
                    .with_context(|| format!("could not open {}", path.display()))?;
                let size = metadata.len();
                Ok(OpenedGguf {
                    reader: Box::new(BufReader::new(file)),
                    size,
                    description: path.display().to_string(),
                })
            }
            GgufRef::Hub {
                owner,
                repo,
                variant,
            } => {
                let Some(variant) = variant else {
                    bail!(
                        "`{owner}/{repo}` needs a quantization to read: use \
                         `{owner}/{repo}:QUANT` (for example `{owner}/{repo}:Q4_K_M`) or a file \
                         name ending in .gguf"
                    );
                };
                let agent = Self::agent();
                let deadline = Instant::now() + TOTAL_BUDGET;
                let files = self.list_files(&agent, owner, repo, deadline)?;
                let file = select_gguf_file(&files, variant)?;
                let path: Vec<String> = file.name.split('/').map(encode_segment).collect();
                let url = format!(
                    "{}/{owner}/{repo}/resolve/main/{}",
                    self.hub,
                    path.join("/")
                );
                // The listing carries a size for a file the hub stores whole; if
                // it left one out, the file's own headers say.
                let size = match file.size {
                    Some(size) => size,
                    None => head_size(&agent, &url, deadline)?,
                };
                Ok(OpenedGguf {
                    reader: Box::new(RangedReader::new(agent, url, size).with_deadline(deadline)),
                    size,
                    description: format!("{owner}/{repo}: {}", file.name),
                })
            }
        }
    }
}

/// How long is left before `deadline`, as the timeout for the next request.
fn time_left(deadline: Instant) -> io::Result<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(io::Error::other(format!(
            "reading the header took longer than {} seconds",
            TOTAL_BUDGET.as_secs()
        )));
    }
    Ok(left.min(REQUEST_TIMEOUT))
}

/// The size of a remote file from its `Content-Length`, without fetching it.
fn head_size(agent: &ureq::Agent, url: &str, deadline: Instant) -> Result<u64> {
    let response = agent
        .head(url)
        .timeout(time_left(deadline).map_err(|error| anyhow!("{error}"))?)
        .call()
        .map_err(|error| describe_http_error(&error, "the model file"))?;
    response
        .header("Content-Length")
        .and_then(|value| value.trim().parse().ok())
        .ok_or_else(|| anyhow!("the hub did not report how large the model file is"))
}

/// A remote file as `Read + Seek`, fetched in windows by `Range` request.
pub struct RangedReader {
    agent: ureq::Agent,
    url: String,
    len: u64,
    position: u64,
    window: Vec<u8>,
    window_start: u64,
    fetched: u64,
    deadline: Instant,
}

impl RangedReader {
    #[must_use]
    pub fn new(agent: ureq::Agent, url: String, len: u64) -> Self {
        Self {
            agent,
            url,
            len,
            position: 0,
            window: Vec::new(),
            window_start: 0,
            fetched: 0,
            deadline: Instant::now() + TOTAL_BUDGET,
        }
    }

    /// Stop reading at `deadline`, however many windows are left.
    #[must_use]
    pub const fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = deadline;
        self
    }

    /// Bytes requested from the server so far.
    #[must_use]
    pub const fn fetched(&self) -> u64 {
        self.fetched
    }

    fn fill(&mut self) -> io::Result<()> {
        let end = self.position.saturating_add(WINDOW_BYTES).min(self.len);
        let wanted = end - self.position;
        if self.fetched.saturating_add(wanted) > MAX_FETCH_BYTES {
            return Err(io::Error::other(format!(
                "the header is larger than {} MiB; this is not a GGUF header read",
                MAX_FETCH_BYTES / (1024 * 1024)
            )));
        }
        let response = self
            .agent
            .get(&self.url)
            .timeout(time_left(self.deadline)?)
            .set("Range", &format!("bytes={}-{}", self.position, end - 1))
            .call()
            .map_err(|error| io::Error::other(describe_http_error(&error, "the model file")))?;
        // 200 means the server ignored the range and would send the whole file.
        if response.status() != 206 {
            return Err(io::Error::other(format!(
                "the server answered {} to a range request; it does not support reading part of \
                 a file, and the whole file is not downloaded to read its header",
                response.status()
            )));
        }
        // A 206 says which bytes it carries. One that starts somewhere else would
        // be parsed as if it began at `position` -- a header read from the wrong
        // offset -- so it is refused, as the download path refuses it.
        let start = crate::download::content_range_start(&response);
        if start != Some(self.position) {
            return Err(io::Error::other(format!(
                "the server answered a range that starts at {} when {} was asked for",
                start.map_or_else(|| "an unstated offset".to_owned(), |at| at.to_string()),
                self.position
            )));
        }
        // `wanted` is at most WINDOW_BYTES already; the clamp states that bound
        // where the allocation is made. A comparison rather than `min`, so a
        // static analyser sees the bound as well as a reader does.
        let capacity = if wanted <= WINDOW_BYTES {
            wanted
        } else {
            WINDOW_BYTES
        };
        let mut window = Vec::with_capacity(usize::try_from(capacity).unwrap_or(0));
        // A body that ends early surfaces as `UnexpectedEof`, which would read as
        // "the header ends here". It is the connection that ended.
        if let Err(error) = response.into_reader().take(wanted).read_to_end(&mut window) {
            return Err(io::Error::other(format!(
                "the connection ended after {} of the {wanted} bytes asked for: {error}",
                window.len()
            )));
        }
        if window.len() as u64 != wanted {
            // Not `UnexpectedEof`: that reads as "the header ends here", which
            // would blame the file for a connection that dropped.
            return Err(io::Error::other(format!(
                "the connection ended after {} of the {wanted} bytes asked for",
                window.len()
            )));
        }
        self.fetched += wanted;
        self.window = window;
        self.window_start = self.position;
        Ok(())
    }
}

impl Read for RangedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let window_end = self.window_start + self.window.len() as u64;
        if self.position < self.window_start || self.position >= window_end {
            self.fill()?;
        }
        let offset = usize::try_from(self.position - self.window_start)
            .map_err(|_| io::Error::other("offset out of range"))?;
        let count = buf.len().min(self.window.len() - offset);
        buf[..count].copy_from_slice(&self.window[offset..offset + count]);
        self.position += count as u64;
        Ok(count)
    }
}

impl Seek for RangedReader {
    fn seek(&mut self, target: SeekFrom) -> io::Result<u64> {
        let next = match target {
            SeekFrom::Start(offset) => Some(offset),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
            SeekFrom::End(delta) => self.len.checked_add_signed(delta),
        };
        let next = next.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before the start of the file",
            )
        })?;
        self.position = next;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_builder::llama_like;
    use crate::gguf::{GgufError, parse_gguf_header};
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    fn file(name: &str, size: u64) -> HubFile {
        HubFile {
            name: name.to_owned(),
            size: Some(size),
        }
    }

    #[test]
    fn a_reference_is_read_as_a_hub_repository_a_path_or_neither() {
        assert_eq!(
            parse_gguf_ref("owner/repo:Q4_K_M"),
            Some(GgufRef::Hub {
                owner: "owner".into(),
                repo: "repo".into(),
                variant: Some("Q4_K_M".into())
            })
        );
        assert_eq!(
            parse_gguf_ref("owner/repo:model-Q4_K_M.gguf"),
            Some(GgufRef::Hub {
                owner: "owner".into(),
                repo: "repo".into(),
                variant: Some("model-Q4_K_M.gguf".into())
            })
        );
        // A bare repository is recognised so it can be refused with a message.
        assert_eq!(
            parse_gguf_ref("owner/repo"),
            Some(GgufRef::Hub {
                owner: "owner".into(),
                repo: "repo".into(),
                variant: None
            })
        );
        // A path ending in .gguf is a file even though it looks like owner/repo.
        assert_eq!(
            parse_gguf_ref("models/thing.gguf"),
            Some(GgufRef::Local(PathBuf::from("models/thing.gguf")))
        );
        assert_eq!(
            parse_gguf_ref("/abs/path/Thing.GGUF"),
            Some(GgufRef::Local(PathBuf::from("/abs/path/Thing.GGUF")))
        );
        for not_a_gguf in [
            "qwen",
            "",
            "  ",
            "a/b/c",
            "../x/y:Q4",
            "owner/..:Q4",
            "o/r:a/b",
        ] {
            assert_eq!(parse_gguf_ref(not_a_gguf), None, "{not_a_gguf:?}");
        }
    }

    /// `m.gguf` at the root and `dir/m.gguf` share a last component. Naming the
    /// full path of either picks that one; the bare name picks the root file, which
    /// matches in full, instead of reporting the two as ambiguous.
    #[test]
    fn a_full_path_match_beats_a_match_on_the_last_component() {
        let files = [file("dir/m.gguf", 4), file("m.gguf", 4)];
        assert_eq!(select_gguf_file(&files, "m.gguf").unwrap().name, "m.gguf");
        assert_eq!(
            select_gguf_file(&files, "dir/m.gguf").unwrap().name,
            "dir/m.gguf"
        );
        // With no root file the bare name finds the one nested file it names...
        let nested = [file("dir/m.gguf", 4), file("dir/other.gguf", 4)];
        assert_eq!(
            select_gguf_file(&nested, "m.gguf").unwrap().name,
            "dir/m.gguf"
        );
        // ...and is ambiguous when it is the last component of two.
        let two = [file("a/m.gguf", 4), file("b/m.gguf", 4)];
        assert!(select_gguf_file(&two, "m.gguf").is_err());
    }

    #[test]
    fn listed_names_that_could_climb_out_of_the_repository_or_hide_in_a_terminal_are_dropped() {
        for name in [
            "../m-Q4_K_M.gguf",
            "a/../m-Q4_K_M.gguf",
            "./m-Q4_K_M.gguf",
            "a//m-Q4_K_M.gguf",
            "/m-Q4_K_M.gguf",
            "m-Q4_K_M\n.gguf",
            "m-Q4_K_M\u{1b}[2J.gguf",
            "m-\u{202E}fuggQ4_K_M.gguf",
            "m-Q4_K_M\u{200B}.gguf",
            "m-Q4_K_M\u{2066}.gguf",
            // Both ends of each range, and the byte order mark.
            "m-Q4_K_M\u{200F}.gguf",
            "m-Q4_K_M\u{202A}.gguf",
            "m-Q4_K_M\u{2069}.gguf",
            "m-Q4_K_M\u{FEFF}.gguf",
        ] {
            assert!(!is_safe_listed_name(name), "{name:?}");
        }
        for name in ["m-Q4_K_M.gguf", "dir/m-Q4_K_M.gguf", "a.b/c_d-e.gguf"] {
            assert!(is_safe_listed_name(name), "{name:?}");
        }
    }

    /// The same through a listing a hub sent: the hostile names are not offered,
    /// so the one safe file is the only match and nothing but it is requested.
    #[test]
    fn a_hostile_listing_is_not_followed_out_of_the_repository() {
        let listing = r#"{"siblings":[
            {"rfilename":"../m-Q4_K_M.gguf","size":1048576},
            {"rfilename":"a/../m-Q4_K_M.gguf","size":1048576},
            {"rfilename":"m-Q4_K_M\u001b[2J.gguf","size":1048576},
            {"rfilename":"ok-Q4_K_M.gguf","size":1048576}]}"#;
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::Honour,
            listing.to_owned(),
        );
        let mut opened = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .unwrap();
        assert_eq!(opened.description, "o/r: ok-Q4_K_M.gguf");
        parse_gguf_header(&mut opened.reader).unwrap();
        let requests = hub.requests.lock().unwrap().clone();
        assert!(
            !requests.is_empty()
                && requests
                    .iter()
                    .all(|request| request.ends_with("/o/r/resolve/main/ok-Q4_K_M.gguf")),
            "{requests:?}"
        );
    }

    #[test]
    fn a_quantization_label_picks_one_file_and_never_a_lookalike() {
        let files = [
            file("m-Q4_0.gguf", 4),
            file("m-Q4_0_4_4.gguf", 4),
            file("m-Q4_K_M.gguf", 5),
            file("m-Q8_0.gguf", 8),
            file("README.md", 1),
            file("mmproj-Q8_0.gguf", 1),
        ];
        assert_eq!(
            select_gguf_file(&files, "Q4_K_M").unwrap().name,
            "m-Q4_K_M.gguf"
        );
        assert_eq!(
            select_gguf_file(&files, "q4_k_m").unwrap().name,
            "m-Q4_K_M.gguf"
        );
        assert_eq!(
            select_gguf_file(&files, "Q4_0").unwrap().name,
            "m-Q4_0.gguf"
        );
        // The vision projector shares the label and is not the model.
        assert_eq!(
            select_gguf_file(&files, "Q8_0").unwrap().name,
            "m-Q8_0.gguf"
        );
        assert_eq!(
            select_gguf_file(&files, "m-Q8_0.gguf").unwrap().name,
            "m-Q8_0.gguf"
        );
        let missing = select_gguf_file(&files, "IQ2_XS").unwrap_err().to_string();
        assert!(
            missing.contains("no GGUF file matching `IQ2_XS`") && missing.contains("m-Q4_K_M.gguf"),
            "{missing}"
        );
    }

    #[test]
    fn two_files_for_one_label_are_refused_but_the_shards_of_one_model_are_not() {
        let twins = [file("a/m-Q4_K_M.gguf", 1), file("b/m-Q4_K_M.gguf", 1)];
        let error = select_gguf_file(&twins, "Q4_K_M").unwrap_err().to_string();
        assert!(error.contains("matches several files"), "{error}");
        // The exact name still resolves one of them.
        assert_eq!(
            select_gguf_file(&twins, "a/m-Q4_K_M.gguf").unwrap().name,
            "a/m-Q4_K_M.gguf"
        );
        let shards = [
            file("Q4_K_M/m-Q4_K_M-00002-of-00002.gguf", 1),
            file("Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", 1),
        ];
        assert_eq!(
            select_gguf_file(&shards, "Q4_K_M").unwrap().name,
            "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf"
        );
    }

    #[test]
    fn a_bare_repository_is_refused_without_touching_the_network() {
        // No server is listening on this port: a refusal that came from a
        // connection error rather than from the missing variant would show it.
        let source = LiveGgufSource::with_hub("http://127.0.0.1:1");
        let error = source
            .open(&GgufRef::Hub {
                owner: "o".into(),
                repo: "r".into(),
                variant: None,
            })
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("needs a quantization"), "{error}");
    }

    #[test]
    fn a_local_file_is_opened_with_its_size() {
        let dir = std::env::temp_dir().join(format!("rocm-gguf-local-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        let bytes = llama_like(5).file(100);
        std::fs::write(&path, &bytes).unwrap();
        let mut opened = LiveGgufSource::with_hub("http://127.0.0.1:1")
            .open(&GgufRef::Local(path))
            .unwrap();
        assert_eq!(opened.size, bytes.len() as u64);
        assert_eq!(
            parse_gguf_header(&mut opened.reader).unwrap().block_count,
            Some(32)
        );
        let missing = LiveGgufSource::new()
            .open(&GgufRef::Local(dir.join("absent.gguf")))
            .err()
            .unwrap()
            .to_string();
        assert!(missing.contains("could not open"), "{missing}");
        // An ordinary missing file is not told it might be a hub reference...
        assert!(!missing.contains("Hugging Face"), "{missing}");
        // ...but one shaped like a file in a repository is shown its hub form.
        let hubby = LiveGgufSource::new()
            .open(&GgufRef::Local(PathBuf::from("owner/repo/dir/file.gguf")))
            .err()
            .unwrap()
            .to_string();
        assert!(hubby.contains("`owner/repo:dir/file.gguf`"), "{hubby}");
        let two = LiveGgufSource::new()
            .open(&GgufRef::Local(PathBuf::from("owner/file.gguf")))
            .err()
            .unwrap()
            .to_string();
        assert!(!two.contains("Hugging Face"), "{two}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// How the fake hub answers a request for the model file.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Serve {
        /// A correct `206` for the range asked.
        Honour,
        /// A `200` with the whole (here tiny) body, ignoring `Range`.
        IgnoreRange,
        /// A `206` whose `Content-Range` starts one byte after the asked offset.
        WrongRange,
        /// A `206` that promises the whole range and drops the connection partway.
        ShortBody,
        /// A `206` whose `Content-Range` names the whole range but whose body, with
        /// a `Content-Length` that matches it, is only half of it: nothing breaks
        /// at the HTTP level, the body is simply short.
        ShortConsistent,
        /// A `401`, as for a gated repository.
        Unauthorized,
        /// A `206` that arrives after a pause.
        Slow,
    }

    /// A hub on loopback. The "model" is a real header followed by zeros out to
    /// `virtual_size`, served only as far as a range asks for it, so a client that
    /// tried to read the weights would show in `ranges`.
    struct FakeHub {
        base: String,
        ranges: Arc<Mutex<Vec<String>>>,
        /// Every request line that reached the model file, with its method.
        requests: Arc<Mutex<Vec<String>>>,
    }

    fn fake_hub(header: Vec<u8>, virtual_size: u64, honour_range: bool) -> FakeHub {
        let serve = if honour_range {
            Serve::Honour
        } else {
            Serve::IgnoreRange
        };
        let listing = format!(
            r#"{{"siblings":[{{"rfilename":"m-Q4_K_M.gguf","size":{virtual_size}}},{{"rfilename":"README.md","size":10}}]}}"#
        );
        fake_hub_with(header, virtual_size, serve, listing)
    }

    fn fake_hub_with(header: Vec<u8>, virtual_size: u64, serve: Serve, listing: String) -> FakeHub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&ranges);
        let logged = Arc::clone(&requests);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut request = Vec::new();
                let mut byte = [0_u8; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    if stream.read(&mut byte).unwrap_or(0) == 0 {
                        break;
                    }
                    request.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&request).into_owned();
                let method = text.split_whitespace().next().unwrap_or("").to_owned();
                let path = text.split_whitespace().nth(1).unwrap_or("").to_owned();
                let range = text
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("Range: bytes=")
                            .or_else(|| line.strip_prefix("range: bytes="))
                    })
                    .map(str::to_owned);
                if path.starts_with("/api/models/o/r") {
                    let body = listing.clone();
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                } else if path.starts_with("/o/r/resolve/main/") {
                    logged.lock().unwrap().push(format!("{method} {path}"));
                    if method == "HEAD" {
                        let _ = write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Length: {virtual_size}\r\nConnection: close\r\n\r\n"
                        );
                        continue;
                    }
                    seen.lock()
                        .unwrap()
                        .push(range.clone().unwrap_or_else(|| "none".into()));
                    match (range, serve) {
                        (_, Serve::Unauthorized) => {
                            let _ = write!(
                                stream,
                                "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            );
                        }
                        (
                            Some(range),
                            Serve::Honour
                            | Serve::WrongRange
                            | Serve::ShortBody
                            | Serve::ShortConsistent
                            | Serve::Slow,
                        ) => {
                            let (start, end) = range.split_once('-').unwrap();
                            let (start, end): (u64, u64) =
                                (start.parse().unwrap(), end.parse().unwrap());
                            let mut body: Vec<u8> = (start..=end)
                                .map(|at| header.get(at as usize).copied().unwrap_or(0))
                                .collect();
                            if serve == Serve::ShortConsistent {
                                body.truncate(body.len() / 2);
                            }
                            let claimed_start = if serve == Serve::WrongRange {
                                start + 1
                            } else {
                                start
                            };
                            if serve == Serve::Slow {
                                thread::sleep(Duration::from_millis(1500));
                            }
                            let _ = write!(
                                stream,
                                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {claimed_start}-{end}/{virtual_size}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            if serve == Serve::ShortBody {
                                let _ = stream.write_all(&body[..body.len() / 2]);
                            } else {
                                let _ = stream.write_all(&body);
                            }
                        }
                        _ => {
                            let _ = write!(
                                stream,
                                "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nGGUF"
                            );
                        }
                    }
                } else {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                }
            }
        });
        FakeHub {
            base,
            ranges,
            requests,
        }
    }

    #[test]
    fn only_the_header_is_requested_from_a_file_far_larger_than_it() {
        // 40 000 tokens makes the header a few hundred KB: bigger than nothing,
        // far smaller than the 8 GiB the server claims the file to be.
        let header = llama_like(40_000).header_bytes();
        let virtual_size: u64 = 8 * 1024 * 1024 * 1024;
        let hub = fake_hub(header.clone(), virtual_size, true);
        let source = LiveGgufSource::with_hub(&hub.base);
        let mut opened = source
            .open(&GgufRef::Hub {
                owner: "o".into(),
                repo: "r".into(),
                variant: Some("Q4_K_M".into()),
            })
            .unwrap();
        assert_eq!(opened.size, virtual_size);
        assert_eq!(opened.description, "o/r: m-Q4_K_M.gguf");
        let parsed = parse_gguf_header(&mut opened.reader).unwrap();
        assert_eq!(parsed.block_count, Some(32));
        let ranges = hub.ranges.lock().unwrap().clone();
        assert!(
            !ranges.is_empty() && ranges.iter().all(|r| r != "none"),
            "{ranges:?}"
        );
        let fetched: u64 = ranges
            .iter()
            .map(|r| {
                let (start, end) = r.split_once('-').unwrap();
                end.parse::<u64>().unwrap() - start.parse::<u64>().unwrap() + 1
            })
            .sum();
        assert!(
            fetched <= 2 * WINDOW_BYTES && fetched < virtual_size / 1000,
            "{fetched} bytes were requested for a header of {} bytes: {ranges:?}",
            header.len()
        );
    }

    #[test]
    fn a_server_that_ignores_range_is_refused_rather_than_streamed() {
        let hub = fake_hub(llama_like(10).header_bytes(), 1 << 30, false);
        let source = LiveGgufSource::with_hub(&hub.base);
        let mut opened = source
            .open(&GgufRef::Hub {
                owner: "o".into(),
                repo: "r".into(),
                variant: Some("Q4_K_M".into()),
            })
            .unwrap();
        let error = parse_gguf_header(&mut opened.reader)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not support reading part of a file"),
            "{error}"
        );
    }

    #[test]
    fn a_missing_repository_is_reported_as_not_found() {
        let hub = fake_hub(Vec::new(), 1, true);
        let source = LiveGgufSource::with_hub(&hub.base);
        let error = source
            .open(&GgufRef::Hub {
                owner: "nobody".into(),
                repo: "nothing".into(),
                variant: Some("Q4_K_M".into()),
            })
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("was not found, or its repository is gated or private"),
            "{error}"
        );
    }

    #[test]
    fn the_reader_never_fetches_more_than_its_budget() {
        let mut reader = RangedReader::new(
            ureq::AgentBuilder::new().build(),
            "http://127.0.0.1:1/unused".into(),
            1 << 40,
        );
        reader.fetched = MAX_FETCH_BYTES;
        let mut buf = [0_u8; 1];
        let error = reader.read(&mut buf).unwrap_err().to_string();
        assert!(error.contains("not a GGUF header read"), "{error}");
    }

    #[test]
    fn seeking_within_the_window_costs_no_request_and_past_the_end_reads_nothing() {
        let header = llama_like(10).header_bytes();
        let hub = fake_hub(header, 1 << 20, true);
        let mut reader = RangedReader::new(
            ureq::AgentBuilder::new().build(),
            format!("{}/o/r/resolve/main/m-Q4_K_M.gguf", hub.base),
            1 << 20,
        );
        let mut four = [0_u8; 4];
        reader.read_exact(&mut four).unwrap();
        assert_eq!(&four, b"GGUF");
        reader.seek(SeekFrom::Current(100)).unwrap();
        reader.read_exact(&mut four).unwrap();
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_exact(&mut four).unwrap();
        assert_eq!(hub.ranges.lock().unwrap().len(), 1);
        assert_eq!(reader.fetched(), 1 << 20);
        reader.seek(SeekFrom::Start(1 << 21)).unwrap();
        assert_eq!(reader.read(&mut four).unwrap(), 0);
        assert!(reader.seek(SeekFrom::Current(-(1 << 30))).is_err());
    }

    #[test]
    fn a_file_that_exists_wins_over_the_hub_reading_of_its_name() {
        // `m/x:y.gguf` has the shape of `owner/repo:variant`. When such a file is
        // there it is the file; when it is not, the same string is the hub's.
        let exists = |path: &Path| path == Path::new("m/x:y.gguf");
        assert_eq!(
            parse_gguf_ref_with("m/x:y.gguf", &exists),
            Some(GgufRef::Local(PathBuf::from("m/x:y.gguf")))
        );
        assert_eq!(
            parse_gguf_ref_with("m/x:y.gguf", &|_| false),
            Some(GgufRef::Hub {
                owner: "m".into(),
                repo: "x".into(),
                variant: Some("y.gguf".into())
            })
        );
        // A `.gguf` name that does not exist is a path that cannot be opened, not
        // a repository to ask the hub about.
        assert_eq!(
            parse_gguf_ref_with("nosuch.gguf", &|_| false),
            Some(GgufRef::Local(PathBuf::from("nosuch.gguf")))
        );
    }

    #[test]
    fn a_variant_may_carry_a_directory_only_when_it_names_a_gguf_file() {
        assert_eq!(
            parse_gguf_ref("o/r:b/m-Q4_K_M.gguf"),
            Some(GgufRef::Hub {
                owner: "o".into(),
                repo: "r".into(),
                variant: Some("b/m-Q4_K_M.gguf".into())
            })
        );
        // None of these is a hub name: a directory without a file name, a
        // traversal, an empty segment, a backslash, an escape. The ones ending in
        // `.gguf` are read as local paths (which then cannot be opened), never
        // sent to the hub.
        for rejected in [
            "o/r:b/Q4_K_M",
            "o/r:../m.gguf",
            "o/r:b/../m.gguf",
            "o/r:/m.gguf",
            "o/r:b//m.gguf",
            "o/r:b\\m.gguf",
            "o/r:b/m%2F.gguf",
        ] {
            assert!(
                !matches!(parse_gguf_ref(rejected), Some(GgufRef::Hub { .. })),
                "{rejected:?}"
            );
        }
    }

    #[test]
    fn shards_of_two_different_models_are_not_one_model() {
        let two_sets = [
            file("a/m-Q4_K_M-00001-of-00002.gguf", 1),
            file("a/m-Q4_K_M-00002-of-00002.gguf", 1),
            file("b/m-Q4_K_M-00001-of-00003.gguf", 1),
            file("b/m-Q4_K_M-00002-of-00003.gguf", 1),
            file("b/m-Q4_K_M-00003-of-00003.gguf", 1),
        ];
        let error = select_gguf_file(&two_sets, "Q4_K_M")
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches several files"), "{error}");
        // Naming one shard exactly still selects it.
        assert_eq!(
            select_gguf_file(&two_sets, "b/m-Q4_K_M-00001-of-00003.gguf")
                .unwrap()
                .name,
            "b/m-Q4_K_M-00001-of-00003.gguf"
        );
        // The same stem with a different shard count is a different split.
        let recut = [
            file("m-Q4_K_M-00001-of-00002.gguf", 1),
            file("m-Q4_K_M-00001-of-00003.gguf", 1),
        ];
        assert!(select_gguf_file(&recut, "Q4_K_M").is_err());
        // A match that is not shards at all is never "the first shard".
        let mixed = [
            file("m-Q4_K_M-00001-of-00002.gguf", 1),
            file("other-Q4_K_M.gguf", 1),
        ];
        assert!(select_gguf_file(&mixed, "Q4_K_M").is_err());
    }

    fn hub_ref(variant: &str) -> GgufRef {
        parse_gguf_ref(&format!("o/r:{variant}")).unwrap()
    }

    #[test]
    fn an_exact_file_in_a_directory_is_reachable_from_the_reference_string() {
        // Two files share a name in different directories. The label is
        // ambiguous, and the error says how to be exact; doing what it says,
        // as the user would type it, reads that file and no other.
        let listing = r#"{"siblings":[{"rfilename":"a/m-Q4_K_M.gguf","size":1048576},{"rfilename":"b/m-Q4_K_M.gguf","size":1048576}]}"#;
        let header = llama_like(10).header_bytes();
        let hub = fake_hub_with(header, 1 << 20, Serve::Honour, listing.to_owned());
        let source = LiveGgufSource::with_hub(&hub.base);
        let error = source.open(&hub_ref("Q4_K_M")).err().unwrap().to_string();
        assert!(
            error.contains("matches several files") && error.contains("owner/repo:dir/file.gguf"),
            "{error}"
        );
        let mut opened = source.open(&hub_ref("b/m-Q4_K_M.gguf")).unwrap();
        assert_eq!(opened.description, "o/r: b/m-Q4_K_M.gguf");
        parse_gguf_header(&mut opened.reader).unwrap();
        let requests = hub.requests.lock().unwrap().clone();
        assert!(
            !requests.is_empty()
                && requests
                    .iter()
                    .all(|request| request.ends_with("/o/r/resolve/main/b/m-Q4_K_M.gguf")),
            "{requests:?}"
        );
    }

    #[test]
    fn a_range_that_starts_elsewhere_is_refused_rather_than_read_from_the_wrong_offset() {
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::WrongRange,
            r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf","size":1048576}]}"#.to_owned(),
        );
        let mut opened = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .unwrap();
        let error = parse_gguf_header(&mut opened.reader).unwrap_err();
        assert!(
            matches!(&error, GgufError::Io(_)) && error.to_string().contains("starts at"),
            "{error}"
        );
    }

    #[test]
    fn a_connection_that_drops_is_not_reported_as_a_header_that_ends() {
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::ShortBody,
            r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf","size":1048576}]}"#.to_owned(),
        );
        let mut opened = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .unwrap();
        let error = parse_gguf_header(&mut opened.reader).unwrap_err();
        // `Truncated` is "the GGUF header ends before it is complete": a claim
        // about the file that a dropped connection does not support.
        assert!(
            matches!(&error, GgufError::Io(_)) && error.to_string().contains("connection ended"),
            "{error}"
        );
    }

    #[test]
    fn a_short_but_well_formed_body_is_reported_as_the_connection_ending() {
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::ShortConsistent,
            r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf","size":1048576}]}"#.to_owned(),
        );
        let mut opened = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .unwrap();
        let error = parse_gguf_header(&mut opened.reader).unwrap_err();
        // The body ended cleanly, so the HTTP layer reported nothing: it is the
        // length check that says fewer bytes came than were asked for, and it must
        // not say the header ends (`Truncated`), which blames the file.
        assert!(
            matches!(&error, GgufError::Io(_))
                && error.to_string().contains("connection ended after")
                && !error.to_string().contains("bytes asked for:"),
            "{error}"
        );
    }

    #[test]
    fn a_gated_file_is_said_to_be_unreadable_without_offering_a_token() {
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::Unauthorized,
            r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf","size":1048576}]}"#.to_owned(),
        );
        let mut opened = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .unwrap();
        let error = parse_gguf_header(&mut opened.reader)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("gated or private") && error.contains("no token is sent"),
            "{error}"
        );
        assert!(
            !error.contains("authentication"),
            "no token is sent, so authentication is not on offer: {error}"
        );
    }

    #[test]
    fn a_listing_without_a_size_falls_back_to_the_files_own_length() {
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::Honour,
            r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf"}]}"#.to_owned(),
        );
        let opened = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .unwrap();
        assert_eq!(opened.size, 1 << 20);
        let requests = hub.requests.lock().unwrap().clone();
        assert!(
            requests.iter().any(|request| request.starts_with("HEAD ")),
            "{requests:?}"
        );
    }

    #[test]
    fn a_file_the_listing_gave_no_size_for_is_still_named_in_a_refusal() {
        // A sibling the listing gave no size for is still a file the repository
        // has, so the "it has: ..." list names it.
        let files = [
            HubFile {
                name: "sized-Q8_0.gguf".into(),
                size: Some(1),
            },
            HubFile {
                name: "unsized-Q6_K.gguf".into(),
                size: None,
            },
        ];
        let error = select_gguf_file(&files, "Q2_K").unwrap_err().to_string();
        assert!(
            error.contains("sized-Q8_0.gguf") && error.contains("unsized-Q6_K.gguf"),
            "{error}"
        );
    }

    #[test]
    fn a_listing_beyond_the_cap_is_refused_not_cut_short() {
        let padding = "x".repeat(usize::try_from(MAX_LISTING_BYTES).unwrap());
        let listing = format!(r#"{{"siblings":[],"pad":"{padding}"}}"#);
        let hub = fake_hub_with(Vec::new(), 1, Serve::Honour, listing);
        let error = LiveGgufSource::with_hub(&hub.base)
            .open(&hub_ref("Q4_K_M"))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("is larger than 16 MiB"), "{error}");
    }

    #[test]
    fn the_whole_read_has_a_deadline_not_only_each_request() {
        let hub = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::Honour,
            r#"{"siblings":[{"rfilename":"m-Q4_K_M.gguf","size":1048576}]}"#.to_owned(),
        );
        let url = format!("{}/o/r/resolve/main/m-Q4_K_M.gguf", hub.base);
        let mut spent = RangedReader::new(ureq::AgentBuilder::new().build(), url, 1 << 20)
            .with_deadline(Instant::now());
        let mut buf = [0_u8; 4];
        let error = spent.read(&mut buf).unwrap_err().to_string();
        assert!(error.contains("took longer than"), "{error}");
        assert!(
            hub.requests.lock().unwrap().is_empty(),
            "a request was made after the deadline"
        );
        // A reply slower than what is left is cut off, not waited for.
        let slow = fake_hub_with(
            llama_like(10).header_bytes(),
            1 << 20,
            Serve::Slow,
            String::new(),
        );
        let url = format!("{}/o/r/resolve/main/m-Q4_K_M.gguf", slow.base);
        let mut reader = RangedReader::new(ureq::AgentBuilder::new().build(), url, 1 << 20)
            .with_deadline(Instant::now() + Duration::from_millis(300));
        let began = Instant::now();
        assert!(reader.read(&mut buf).is_err());
        assert!(
            began.elapsed() < Duration::from_millis(1400),
            "waited for the slow reply"
        );
    }

    #[test]
    fn a_path_that_is_not_a_regular_file_is_refused_without_being_opened() {
        let dir = std::env::temp_dir().join(format!("rocm-gguf-kind-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = LiveGgufSource::with_hub("http://127.0.0.1:1");
        let error = source
            .open(&GgufRef::Local(dir.clone()))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("is not a regular file"), "{error}");
        // Opening a named pipe would block until something wrote to it.
        #[cfg(unix)]
        {
            let fifo = dir.join("pipe.gguf");
            let made = std::process::Command::new("mkfifo").arg(&fifo).status();
            if made.is_ok_and(|status| status.success()) {
                let error = source
                    .open(&GgufRef::Local(fifo))
                    .err()
                    .unwrap()
                    .to_string();
                assert!(error.contains("is not a regular file"), "{error}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn path_segments_are_percent_encoded() {
        assert_eq!(encode_segment("m Q4+K.gguf"), "m%20Q4%2BK.gguf");
        assert_eq!(encode_segment("a-b_c.d~e"), "a-b_c.d~e");
    }
}
