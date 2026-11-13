// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Checks on `runtimes/mise.lock`, the lock for the third-party binaries
//! pinned in `runtimes/mise.toml`.
//!
//! `mise lock` succeeds in two situations where the lock is not usable, and
//! both were seen while adopting it, so they are checked here rather than
//! trusted:
//!
//! - when it cannot resolve a tool for a platform it warns, exits 0 and writes
//!   a version-only entry with no URL or checksum;
//! - an `asset_pattern` that matches both platforms' assets has locked the
//!   Windows archive for `linux-x64`.
//!
//! It also refuses an asset that is not a release download of the tool's own
//! GitHub repository: the lock is what mise downloads binaries from, and the
//! weekly update commits a lock produced by a job that ran upstream code.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

/// Workspace-relative paths of the mise config and lock.
pub(crate) const MISE_CONFIG: &str = "runtimes/mise.toml";
pub(crate) const MISE_LOCK: &str = "runtimes/mise.lock";

/// File-name tokens that identify an x86-64 build; every platform rocm-cli
/// locks is `*-x64`.
const X64_TOKENS: [&str; 4] = ["x64", "x86_64", "amd64", "x86-64"];

/// File-name tokens that identify which OS a release asset is built for.
const OS_TOKENS: [(&str, &[&str]); 2] = [
    ("linux", &["linux", "ubuntu", "debian"]),
    ("windows", &["windows", "win64", "win_amd64", "win-x64"]),
];

/// The `provenance` mise records when it verified a release's build
/// attestation; the only value a lock may carry, and only on `ATTESTED_TOOLS`.
pub(crate) const ATTESTED: &str = "github-attestations";

/// Tools whose releases carry build attestations, so their locks must record
/// that mise verified one. Any other tool may carry no `provenance` at all.
/// The string is not proof by itself: `deps verify-downloads` re-hashes every
/// locked download, these tools' included.
pub(crate) const ATTESTED_TOOLS: [&str; 1] = [super::pins_gen::UV_TOOL];

/// One locked platform of one tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LockedAsset {
    pub(crate) url: String,
    pub(crate) sha256: String,
    /// The GitHub API URL mise downloads through when it is set, instead of
    /// `url`; so it is bound to the same release as `url`.
    pub(crate) url_api: Option<String>,
    /// How mise verified the asset when it locked it.
    pub(crate) provenance: Option<String>,
}

/// Keys a locked platform table may carry. Anything else is refused: mise
/// acts on fields this module does not check, and the weekly update commits
/// a lock written by a job that ran upstream code.
const PLATFORM_KEYS: [&str; 6] = [
    "checksum",
    "checksum_source",
    "url",
    "url_api",
    "provenance",
    "size",
];

/// Keys a tool entry may carry, besides its `platforms.<platform>` tables.
const ENTRY_KEYS: [&str; 4] = ["version", "backend", "specifiers", "options"];

/// Keys an entry's `options` table may carry.
const OPTION_KEYS: [&str; 1] = ["asset_pattern"];

/// A tool as recorded in `mise.lock`: one version, one asset per platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LockedTool {
    pub(crate) version: String,
    /// The mise backend that resolved it (`github:owner/repo`, `aqua:owner/repo`).
    pub(crate) backend: Option<String>,
    pub(crate) platforms: BTreeMap<String, LockedAsset>,
}

/// The tools a `mise.toml` declares and the platforms it locks for.
#[derive(Debug, Clone)]
pub(crate) struct MiseConfig {
    pub(crate) tools: Vec<String>,
    /// The version each tool pins, where it pins one (`uv = "0.13.6"` or an
    /// inline table's `version`).
    pub(crate) versions: BTreeMap<String, String>,
    pub(crate) platforms: Vec<String>,
}

impl MiseConfig {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let doc: toml::Table = toml::from_str(text).context("not valid TOML")?;
        let settings = doc.get("settings").and_then(toml::Value::as_table);
        ensure!(
            settings
                .and_then(|s| s.get("lockfile"))
                .and_then(toml::Value::as_bool)
                == Some(true),
            "[settings] lockfile = true is required"
        );
        let platforms: Vec<String> = settings
            .and_then(|s| s.get("lockfile_platforms"))
            .and_then(toml::Value::as_array)
            .context("[settings] lockfile_platforms must list the platforms to lock")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("platforms must be strings")
            })
            .collect::<Result<_>>()?;
        // Exactly the platforms the generated constants cover: one fewer and
        // `deps check` would never look at that platform's asset; one more and
        // the build would refuse the lock.
        let mut sorted = platforms.clone();
        sorted.sort();
        let mut required: Vec<String> = super::pins_gen::REQUIRED_PLATFORMS
            .iter()
            .map(|p| (*p).to_owned())
            .collect();
        required.sort();
        ensure!(
            sorted == required,
            "lockfile_platforms must be exactly {required:?}, got {platforms:?}"
        );
        let declared = doc
            .get("tools")
            .and_then(toml::Value::as_table)
            .context("no [tools] table")?;
        let tools: Vec<String> = declared.keys().cloned().collect();
        let versions = declared
            .iter()
            .filter_map(|(name, spec)| {
                let version = spec
                    .as_str()
                    .or_else(|| spec.get("version").and_then(toml::Value::as_str))?;
                Some((name.clone(), version.to_owned()))
            })
            .collect();
        Ok(Self {
            tools,
            versions,
            platforms,
        })
    }
}

/// Parse `mise.lock` into one [`LockedTool`] per tool.
///
/// A tool with per-platform options is written as several entries (one per
/// option set); they are merged here, and must agree on the version.
pub(crate) fn parse_lock(text: &str) -> Result<BTreeMap<String, LockedTool>> {
    let doc: toml::Table = toml::from_str(text).context("not valid TOML")?;
    // As at every other level: a key this check does not know is refused.
    if let Some(key) = doc
        .keys()
        .find(|key| !matches!(key.as_str(), "lockfile_version" | "tools"))
    {
        bail!("unknown top-level field {key:?}");
    }
    let tools = doc
        .get("tools")
        .and_then(toml::Value::as_table)
        .context("no [tools] table")?;
    let mut out = BTreeMap::new();
    for (name, entries) in tools {
        let entries = entries
            .as_array()
            .with_context(|| format!("tools.{name:?} is not an array of entries"))?;
        let mut locked: Option<LockedTool> = None;
        for entry in entries {
            let entry = entry
                .as_table()
                .with_context(|| format!("tools.{name:?} entry is not a table"))?;
            let version = entry
                .get("version")
                .and_then(toml::Value::as_str)
                .with_context(|| format!("tools.{name:?} entry has no version"))?;
            let backend = entry.get("backend").and_then(toml::Value::as_str);
            let tool = locked.get_or_insert_with(|| LockedTool {
                version: version.to_owned(),
                backend: backend.map(str::to_owned),
                platforms: BTreeMap::new(),
            });
            ensure!(
                tool.version == version,
                "{name}: entries lock different versions ({} and {version})",
                tool.version
            );
            ensure!(
                tool.backend.as_deref() == backend,
                "{name}: entries name different backends"
            );
            for (key, value) in entry {
                let Some(platform) = key.strip_prefix("platforms.") else {
                    ensure!(
                        ENTRY_KEYS.contains(&key.as_str()),
                        "{name}: entry key {key:?} is not one this check understands"
                    );
                    if key == "options" {
                        for option in value.as_table().into_iter().flatten().map(|(k, _)| k) {
                            ensure!(
                                OPTION_KEYS.contains(&option.as_str()),
                                "{name}: option {option:?} is not one this check understands"
                            );
                        }
                    }
                    continue;
                };
                let table = value
                    .as_table()
                    .with_context(|| format!("{name}: {key} is not a table"))?;
                for field in table.keys() {
                    ensure!(
                        PLATFORM_KEYS.contains(&field.as_str()),
                        "{name}: {platform} field {field:?} is not one this check understands"
                    );
                }
                let text_field = |field: &str| {
                    table
                        .get(field)
                        .and_then(toml::Value::as_str)
                        .map(str::to_owned)
                };
                let url = table.get("url").and_then(toml::Value::as_str);
                let checksum = table.get("checksum").and_then(toml::Value::as_str);
                let (Some(url), Some(checksum)) = (url, checksum) else {
                    bail!("{name}: {platform} has no url and checksum");
                };
                let sha256 = checksum
                    .strip_prefix("sha256:")
                    .with_context(|| format!("{name}: {platform} checksum is not sha256"))?;
                let asset = LockedAsset {
                    url: url.to_owned(),
                    sha256: sha256.to_owned(),
                    url_api: text_field("url_api"),
                    provenance: text_field("provenance"),
                };
                if let Some(previous) = tool.platforms.insert(platform.to_owned(), asset) {
                    bail!("{name}: {platform} is locked twice ({})", previous.url);
                }
            }
        }
        if let Some(tool) = locked {
            out.insert(name.clone(), tool);
        }
    }
    Ok(out)
}

/// Every problem with `lock` against `config`, as readable messages.
pub(crate) fn problems(config: &MiseConfig, lock: &BTreeMap<String, LockedTool>) -> Vec<String> {
    let mut out = Vec::new();
    for name in &config.tools {
        let Some(tool) = lock.get(name) else {
            out.push(format!(
                "{name}: declared in {MISE_CONFIG} but missing from {MISE_LOCK}"
            ));
            continue;
        };
        // The offline check is the only gate that runs on every change, so it
        // is what catches a pin edited without re-locking.
        if let Some(pinned) = config.versions.get(name)
            && *pinned != tool.version
        {
            out.push(format!(
                "{name}: {MISE_CONFIG} pins {pinned} but {MISE_LOCK} locks {}",
                tool.version
            ));
        }
        // The repository a tool may download from is the one its `mise.toml`
        // key names, which the weekly update cannot change; the lock's own
        // `backend` field is output of the job that ran upstream code, so it
        // must agree with the key rather than be trusted.
        if release_prefix(name).is_none() {
            out.push(format!(
                "{name}: name tools in {MISE_CONFIG} by `github:` or `aqua:` owner/repo, \
                 so their downloads can be checked against that repository"
            ));
        }
        if tool.backend.as_deref() != Some(name.as_str()) {
            out.push(format!(
                "{name}: {MISE_LOCK} records backend {:?}, not the {name:?} {MISE_CONFIG} declares",
                tool.backend
            ));
        }
        for platform in &config.platforms {
            let Some(asset) = tool.platforms.get(platform) else {
                out.push(format!(
                    "{name}@{}: no {platform} asset locked (mise falls back to a \
                     version-only entry when it cannot resolve one)",
                    tool.version
                ));
                continue;
            };
            out.extend(asset_problems(name, platform, asset));
            out.extend(source_problems(name, &tool.version, platform, asset));
            out.extend(api_problems(name, platform, asset));
            // An attested tool must record that mise verified its attestation.
            // Any other tool may record no provenance at all: a value there
            // would claim a check that never ran.
            if ATTESTED_TOOLS.contains(&name.as_str()) {
                if asset.provenance.as_deref() != Some(ATTESTED) {
                    out.push(format!(
                        "{name}: {platform} was not locked with provenance = \"{ATTESTED}\""
                    ));
                }
            } else if let Some(value) = asset.provenance.as_deref() {
                out.push(format!(
                    "{name}: {platform} records provenance {value:?}, but only {ATTESTED_TOOLS:?} are attested"
                ));
            }
        }
        // A platform the config does not ask for is checked by nothing above,
        // so it must not be in the lock at all.
        for platform in tool.platforms.keys() {
            if !config.platforms.contains(platform) {
                out.push(format!(
                    "{name}: {MISE_LOCK} locks {platform:?}, which {MISE_CONFIG} does not list in lockfile_platforms"
                ));
            }
        }
    }
    for name in lock.keys() {
        if !config.tools.contains(name) {
            out.push(format!(
                "{name}: locked but no longer declared in {MISE_CONFIG}"
            ));
        }
    }
    out
}

/// The release-download prefix every asset of a tool resolved by `backend`
/// must sit under, or `None` for a backend this repository does not use.
fn release_prefix(backend: &str) -> Option<String> {
    let repo = backend
        .strip_prefix("github:")
        .or_else(|| backend.strip_prefix("aqua:"))?;
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return None;
    };
    (is_plain_segment(owner) && is_plain_segment(name))
        .then(|| format!("https://github.com/{owner}/{name}/releases/download/"))
}

/// The values `deps diff` renders for one locked asset, checked with the
/// same building blocks `deps check` uses: the tool is a `github:`/`aqua:`
/// owner/repo of plain segments ([`release_prefix`]), the platform and the
/// version are plain segments, and the digest is 64 lowercase hex digits.
/// `deps check` ties the version to its release tag (the tag, or the tag less
/// a leading `v`), so a version it accepts is a plain segment too, except a
/// degenerate one such as `.`, which this refuses. Each rendered value is
/// made only of plain-segment characters or hex, so none can break the
/// Markdown table; the update's validator may still refuse a row as not a
/// version change (it wants a version starting with a digit).
pub(crate) fn row_problems(
    name: &str,
    version: &str,
    platform: &str,
    asset: &LockedAsset,
) -> Vec<String> {
    let mut out = Vec::new();
    if release_prefix(name).is_none() {
        out.push(format!("{name:?} is not a `github:` or `aqua:` owner/repo"));
    }
    if !is_plain_segment(platform) {
        out.push(format!("{name}: {platform:?} is not a platform name"));
    }
    if !is_plain_segment(version) {
        out.push(format!("{name}: {version:?} is not a release tag"));
    }
    if !is_sha256_hex(&asset.sha256) {
        out.push(format!(
            "{name}: {platform} checksum is not a sha256 hex digest"
        ));
    }
    out
}

/// 64 lowercase hex digits, as mise records a SHA-256.
pub(crate) fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A URL path segment that means only itself: no `.`/`..`, no `%` escape, no
/// separator or other character a URL parser could reinterpret.
fn is_plain_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'))
}

/// An asset that is not a release download of the tool's own repository.
fn source_problems(
    name: &str,
    version: &str,
    platform: &str,
    asset: &LockedAsset,
) -> Option<String> {
    // A name that is not owner/repo is reported once, by `problems`.
    let prefix = release_prefix(name)?;
    // Exactly `<prefix><tag>/<file>`: a prefix match alone would accept
    // `<prefix>../../../evil/x/...`, which URL normalisation resolves to
    // another repository.
    let rest = asset.url.strip_prefix(&prefix);
    let plain = rest.is_some_and(|rest| {
        let mut segments = rest.split('/');
        matches!(
            (segments.next(), segments.next(), segments.next()),
            (Some(tag), Some(file), None) if is_plain_segment(tag) && is_plain_segment(file)
        )
    });
    if !plain {
        return Some(format!(
            "{name}: {platform} asset {} is not a release download under {prefix}<tag>/<file>",
            asset.url
        ));
    }
    // The release must be the one the version names: otherwise an older real
    // release, with its own valid digest, could be locked under a newer label.
    let tag = rest
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_default();
    (tag != version && tag.strip_prefix('v') != Some(version)).then(|| {
        format!(
            "{name}: {platform} asset comes from release {tag:?}, not the locked version {version}"
        )
    })
}

/// A `url_api` that is not an asset of the tool's own repository. mise
/// downloads through it in preference to `url`.
fn api_problems(name: &str, platform: &str, asset: &LockedAsset) -> Option<String> {
    let api = asset.url_api.as_deref()?;
    let repo = name
        .strip_prefix("github:")
        .or_else(|| name.strip_prefix("aqua:"))?;
    let prefix = format!("https://api.github.com/repos/{repo}/releases/assets/");
    let ok = api
        .strip_prefix(&prefix)
        .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()));
    (!ok).then(|| format!("{name}: {platform} url_api {api} is not {prefix}<asset id>"))
}

fn asset_problems(name: &str, platform: &str, asset: &LockedAsset) -> Vec<String> {
    let mut out = Vec::new();
    if crate::deps::pylock::https_host(&asset.url).is_none() {
        out.push(format!(
            "{name}: {platform} url is not https: {}",
            asset.url
        ));
    }
    if !is_sha256_hex(&asset.sha256) {
        out.push(format!(
            "{name}: {platform} checksum is not a sha256 hex digest"
        ));
    }
    let file = asset
        .url
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let os = platform.split('-').next().unwrap_or_default();
    if platform.ends_with("-x64") && !X64_TOKENS.iter().any(|t| file.contains(t)) {
        out.push(format!(
            "{name}: {platform} asset {file} is not an x86-64 build"
        ));
    }
    for (token_os, tokens) in OS_TOKENS {
        let mentions = tokens.iter().any(|t| file.contains(t));
        if token_os == os && !mentions {
            out.push(format!(
                "{name}: {platform} asset {file} does not name {os}"
            ));
        } else if token_os != os && mentions {
            out.push(format!(
                "{name}: {platform} asset {file} is a {token_os} build"
            ));
        }
    }
    out
}

/// What the `rocm-deps` build would say about `lock_text`: it reads the same
/// file at build time, so a lock this module accepts but the build rejects
/// would only surface as a compile error on someone else's change.
pub(crate) fn generator_problems(lock_text: &str) -> Option<String> {
    super::pins_gen::render(lock_text)
        .err()
        .map(|err| format!("{MISE_LOCK}: crates/rocm-deps would reject it: {err}"))
}

/// Load and check the committed config and lock under `root`.
/// A file that cannot be read or parsed is reported as a problem rather than
/// returned as an error, so it does not hide every other problem `deps check`
/// would report.
pub(crate) fn check(root: &Path) -> Result<Vec<String>> {
    let read = |file: &str| {
        std::fs::read_to_string(root.join(file))
            .map_err(|err| format!("{file}: cannot read it ({err})"))
    };
    let config = match read(MISE_CONFIG)
        .and_then(|text| MiseConfig::parse(&text).map_err(|err| format!("{MISE_CONFIG}: {err:#}")))
    {
        Ok(config) => config,
        Err(problem) => return Ok(vec![problem]),
    };
    let lock_text = match read(MISE_LOCK) {
        Ok(text) => text,
        Err(problem) => return Ok(vec![problem]),
    };
    let lock = match parse_lock(&lock_text) {
        Ok(lock) => lock,
        Err(err) => return Ok(vec![format!("{MISE_LOCK}: {err:#}")]),
    };
    let mut out = problems(&config, &lock);
    out.extend(generator_problems(&lock_text));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    const CONFIG: &str = r#"
[settings]
lockfile = true
lockfile_platforms = ["linux-x64", "windows-x64"]

[tools]
"github:example/tool" = "1.2.3"
"#;

    const LINUX: &str =
        "https://github.com/example/tool/releases/download/v1.2.3/tool-1.2.3-ubuntu-x64.tar.gz";
    const WINDOWS: &str =
        "https://github.com/example/tool/releases/download/v1.2.3/tool-1.2.3-windows-x64.zip";

    fn lock(linux_url: &str, windows: Option<&str>) -> String {
        let sum = "a".repeat(64);
        let mut text = format!(
            "lockfile_version = 3\n\n[[tools.\"github:example/tool\"]]\nversion = \"1.2.3\"\nbackend = \"github:example/tool\"\n\n\
             [tools.\"github:example/tool\".\"platforms.linux-x64\"]\n\
             checksum = \"sha256:{sum}\"\nurl = \"{linux_url}\"\n"
        );
        if let Some(windows_url) = windows {
            write!(
                text,
                "\n[[tools.\"github:example/tool\"]]\nversion = \"1.2.3\"\nbackend = \"github:example/tool\"\n\n\
                 [tools.\"github:example/tool\".\"platforms.windows-x64\"]\n\
                 checksum = \"sha256:{sum}\"\nurl = \"{windows_url}\"\n"
            )
            .unwrap();
        }
        text
    }

    fn problems_of(lock_text: &str) -> Vec<String> {
        problems(
            &MiseConfig::parse(CONFIG).unwrap(),
            &parse_lock(lock_text).unwrap(),
        )
    }

    #[test]
    fn the_committed_lock_is_complete() {
        let root = crate::paths::workspace_root().unwrap();
        assert_eq!(check(&root).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn per_option_entries_merge_into_one_tool() {
        let tools = parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap();
        let tool = &tools["github:example/tool"];
        assert_eq!(tool.version, "1.2.3");
        assert_eq!(tool.platforms.len(), 2);
        assert!(problems_of(&lock(LINUX, Some(WINDOWS))).is_empty());
    }

    #[test]
    fn a_version_only_fallback_is_reported() {
        let found = problems_of(&lock(LINUX, None));
        assert!(
            found
                .iter()
                .any(|p| p.contains("no windows-x64 asset locked")),
            "{found:?}"
        );
    }

    #[test]
    fn the_windows_archive_locked_for_linux_is_reported() {
        let found = problems_of(&lock(WINDOWS, Some(WINDOWS)));
        assert!(
            found
                .iter()
                .any(|p| p.contains("linux-x64 asset") && p.contains("windows build")),
            "{found:?}"
        );
        assert!(
            found.iter().any(|p| p.contains("does not name linux")),
            "{found:?}"
        );
    }

    #[test]
    fn entries_that_disagree_on_the_version_are_refused() {
        let text =
            lock(LINUX, Some(WINDOWS)).replacen("version = \"1.2.3\"", "version = \"1.2.4\"", 1);
        assert!(parse_lock(&text).is_err());
    }

    #[test]
    fn an_entry_without_url_or_checksum_is_refused() {
        let text = lock(LINUX, Some(WINDOWS)).replace(&format!("url = \"{LINUX}\"\n"), "");
        assert!(parse_lock(&text).is_err());
    }

    #[test]
    fn a_tool_missing_from_the_lock_or_left_behind_is_reported() {
        let config =
            MiseConfig::parse(&CONFIG.replace("[tools]", "[tools]\nuv = \"0.13.6\"")).unwrap();
        let found = problems(&config, &parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap());
        assert!(
            found.iter().any(|p| p.starts_with("uv: declared")),
            "{found:?}"
        );

        let config =
            MiseConfig::parse(&CONFIG.replace("\"github:example/tool\" = \"1.2.3\"", "")).unwrap();
        let found = problems(&config, &parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap());
        assert!(
            found.iter().any(|p| p.contains("no longer declared")),
            "{found:?}"
        );
    }

    #[test]
    fn a_config_without_locking_is_refused() {
        assert!(MiseConfig::parse(&CONFIG.replace("lockfile = true", "lockfile = false")).is_err());
        assert!(MiseConfig::parse(&CONFIG.replace("lockfile = true\n", "")).is_err());
        assert!(
            MiseConfig::parse(
                &CONFIG.replace("lockfile_platforms = [\"linux-x64\", \"windows-x64\"]", "")
            )
            .is_err()
        );
    }

    #[test]
    fn an_asset_for_another_architecture_is_reported() {
        let arm = LINUX.replace("ubuntu-x64", "ubuntu-arm64");
        let found = problems_of(&lock(&arm, Some(WINDOWS)));
        assert!(
            found.iter().any(|p| p.contains("is not an x86-64 build")),
            "{found:?}"
        );
    }

    #[test]
    fn plain_http_and_malformed_digests_are_reported() {
        let http = LINUX.replace("https://", "http://");
        let found = problems_of(&lock(&http, Some(WINDOWS)));
        assert!(
            found.iter().any(|p| p.contains("url is not https")),
            "{found:?}"
        );
        let short = lock(LINUX, Some(WINDOWS)).replacen(&"a".repeat(64), "abc", 1);
        let found = problems_of(&short);
        assert!(
            found.iter().any(|p| p.contains("not a sha256 hex digest")),
            "{found:?}"
        );
    }

    #[test]
    fn a_platform_locked_twice_or_an_md5_checksum_is_refused() {
        let twice =
            lock(LINUX, Some(WINDOWS)).replace("platforms.windows-x64", "platforms.linux-x64");
        assert!(format!("{:#}", parse_lock(&twice).unwrap_err()).contains("locked twice"));
        let md5 = lock(LINUX, Some(WINDOWS)).replacen("sha256:", "md5:", 1);
        assert!(format!("{:#}", parse_lock(&md5).unwrap_err()).contains("not sha256"));
    }

    #[test]
    fn a_pin_edited_without_relocking_is_reported() {
        let config =
            MiseConfig::parse(&CONFIG.replace("= \"1.2.3\"", "= { version = \"1.2.4\" }")).unwrap();
        let found = problems(&config, &parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap());
        assert!(
            found
                .iter()
                .any(|p| p.contains("pins 1.2.4 but runtimes/mise.lock locks 1.2.3")),
            "{found:?}"
        );
        let same = MiseConfig::parse(CONFIG).unwrap();
        assert!(problems(&same, &parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap()).is_empty());
    }

    #[test]
    fn an_asset_from_outside_the_tools_releases_is_reported() {
        let foreign = "https://evil.example/tool-1.2.3-ubuntu-x64.tar.gz";
        let found = problems_of(&lock(foreign, Some(WINDOWS)));
        assert!(
            found.iter().any(|p| p.contains(
                "is not a release download under https://github.com/example/tool/releases/download/"
            )),
            "{found:?}"
        );
        let other_repo = LINUX.replace("example/tool", "example/other");
        let found = problems_of(&lock(&other_repo, Some(WINDOWS)));
        assert!(
            found
                .iter()
                .any(|p| p.contains("is not a release download")),
            "{found:?}"
        );
    }

    #[test]
    fn a_backend_other_than_the_declared_one_is_reported() {
        // Backend and URLs swapped together: every URL is a plain release
        // download of the repository the lock claims, so only the binding to
        // the mise.toml key catches it.
        let swapped = lock(LINUX, Some(WINDOWS))
            .replace(
                "backend = \"github:example/tool\"",
                "backend = \"github:evil/x\"",
            )
            .replace("github.com/example/tool/", "github.com/evil/x/");
        let found = problems_of(&swapped);
        assert!(
            found
                .iter()
                .any(|p| p.contains("records backend Some(\"github:evil/x\")")),
            "{found:?}"
        );
        assert!(
            found
                .iter()
                .any(|p| p.contains("is not a release download")),
            "{found:?}"
        );
        assert_eq!(
            release_prefix("aqua:astral-sh/uv").as_deref(),
            Some("https://github.com/astral-sh/uv/releases/download/")
        );
        assert_eq!(release_prefix("github:a/b/c"), None);
    }

    #[test]
    fn a_string_form_pin_edited_without_relocking_is_reported() {
        let config = MiseConfig::parse(&CONFIG.replace("= \"1.2.3\"", "= \"1.2.4\"")).unwrap();
        let found = problems(&config, &parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap());
        assert!(found.iter().any(|p| p.contains("pins 1.2.4")), "{found:?}");
    }

    #[test]
    fn a_traversal_or_escape_after_the_release_prefix_is_reported() {
        for url in [
            "https://github.com/example/tool/releases/download/../../../../evil/x/releases/download/v1/t-ubuntu-x64.tar.gz",
            "https://github.com/example/tool/releases/download/%2e%2e/t-ubuntu-x64.tar.gz",
            "https://github.com/example/tool/releases/download/v1/../t-ubuntu-x64.tar.gz",
            "https://github.com/example/tool/releases/download/v1/sub/t-ubuntu-x64.tar.gz",
            "https://github.com/example/tool/releases/download/v1/",
        ] {
            let found = problems_of(&lock(url, Some(WINDOWS)));
            assert!(
                found
                    .iter()
                    .any(|p| p.contains("is not a release download")),
                "{url}: {found:?}"
            );
        }
        assert!(problems_of(&lock(LINUX, Some(WINDOWS))).is_empty());
    }

    #[test]
    fn a_url_api_outside_the_tools_repository_is_reported() {
        let with_api = |api: &str| {
            lock(LINUX, Some(WINDOWS)).replacen(
                &format!("url = \"{LINUX}\"\n"),
                &format!("url = \"{LINUX}\"\nurl_api = \"{api}\"\n"),
                1,
            )
        };
        assert!(
            problems_of(&with_api(
                "https://api.github.com/repos/example/tool/releases/assets/123"
            ))
            .is_empty()
        );
        for api in [
            "https://evil.example/repos/example/tool/releases/assets/1",
            "https://api.github.com/repos/evil/x/releases/assets/1",
            "https://api.github.com/repos/example/tool/releases/assets/1/../../x",
            "https://api.github.com/repos/example/tool/releases/assets/",
        ] {
            let found = problems_of(&with_api(api));
            assert_eq!(
                found,
                [format!(
                    "github:example/tool: linux-x64 url_api {api} is not \
                     https://api.github.com/repos/example/tool/releases/assets/<asset id>"
                )],
                "{api}"
            );
        }
    }

    #[test]
    fn fields_this_check_does_not_understand_are_refused() {
        let base = lock(LINUX, Some(WINDOWS));
        let root = format!("mirror = \"https://evil.example/\"\n{base}");
        assert_eq!(
            format!("{:#}", parse_lock(&root).unwrap_err()),
            "unknown top-level field \"mirror\""
        );
        for (what, text) in [
            (
                "platform field",
                base.replacen("url = ", "mirror = \"https://evil.example/\"\nurl = ", 1),
            ),
            (
                "entry key",
                base.replacen("backend = ", "install_env = \"x\"\nbackend = ", 1),
            ),
            (
                "option",
                base.replacen(
                    "backend = \"github:example/tool\"\n",
                    "backend = \"github:example/tool\"\noptions = { url = \"https://evil.example/\" }\n",
                    1,
                ),
            ),
        ] {
            let err = parse_lock(&text).unwrap_err();
            assert!(
                format!("{err:#}").contains("is not one this check understands"),
                "{what}: {err:#}"
            );
        }
    }

    /// uv is an `aqua:` tool; its `url_api` is bound to its repository the
    /// same way a `github:` tool's is.
    #[test]
    fn an_aqua_tools_url_api_outside_its_repository_is_reported() {
        let config = MiseConfig::parse(
            "[settings]\nlockfile = true\nlockfile_platforms = [\"linux-x64\", \"windows-x64\"]\n\n[tools]\n\"aqua:astral-sh/uv\" = \"0.13.6\"\n",
        )
        .unwrap();
        let committed =
            std::fs::read_to_string(crate::paths::workspace_root().unwrap().join(MISE_LOCK))
                .unwrap();
        let mut uv_only: BTreeMap<_, _> = parse_lock(&committed)
            .unwrap()
            .into_iter()
            .filter(|(name, _)| name == "aqua:astral-sh/uv")
            .collect();
        let asset = uv_only
            .get_mut("aqua:astral-sh/uv")
            .unwrap()
            .platforms
            .get_mut("linux-x64")
            .unwrap();
        asset.url_api = Some("https://api.github.com/repos/evil/uv/releases/assets/1".to_owned());
        assert_eq!(
            problems(&config, &uv_only),
            [
                "aqua:astral-sh/uv: linux-x64 url_api https://api.github.com/repos/evil/uv/releases/assets/1 \
              is not https://api.github.com/repos/astral-sh/uv/releases/assets/<asset id>"
            ]
        );
    }

    #[test]
    fn a_provenance_other_than_a_verified_attestation_is_refused() {
        let text = lock(LINUX, Some(WINDOWS)).replacen(
            &format!("url = \"{LINUX}\"\n"),
            &format!("url = \"{LINUX}\"\nprovenance = \"trust-me\"\n"),
            1,
        );
        assert_eq!(
            problems_of(&text),
            [
                "github:example/tool: linux-x64 records provenance \"trust-me\", but only [\"aqua:astral-sh/uv\"] are attested"
            ]
        );
    }

    #[test]
    fn an_attestation_on_a_tool_that_is_not_attested_is_refused() {
        let text = lock(LINUX, Some(WINDOWS)).replacen(
            &format!("url = \"{LINUX}\"\n"),
            &format!("url = \"{LINUX}\"\nprovenance = \"github-attestations\"\n"),
            1,
        );
        assert_eq!(
            problems_of(&text),
            [
                "github:example/tool: linux-x64 records provenance \"github-attestations\", but only [\"aqua:astral-sh/uv\"] are attested"
            ]
        );
    }

    #[test]
    fn uv_must_be_locked_with_its_build_attestation() {
        let config = MiseConfig::parse(
            "[settings]\nlockfile = true\nlockfile_platforms = [\"linux-x64\", \"windows-x64\"]\n\n[tools]\n\"aqua:astral-sh/uv\" = \"0.13.6\"\n",
        )
        .unwrap();
        let committed =
            std::fs::read_to_string(crate::paths::workspace_root().unwrap().join(MISE_LOCK))
                .unwrap();
        let uv_only: BTreeMap<_, _> = parse_lock(&committed)
            .unwrap()
            .into_iter()
            .filter(|(name, _)| name == "aqua:astral-sh/uv")
            .collect();
        assert_eq!(problems(&config, &uv_only), Vec::<String>::new());
        let mut dropped = uv_only;
        for asset in dropped
            .get_mut("aqua:astral-sh/uv")
            .unwrap()
            .platforms
            .values_mut()
        {
            asset.provenance = None;
        }
        let found = problems(&config, &dropped);
        assert!(
            found.contains(&"aqua:astral-sh/uv: linux-x64 was not locked with provenance = \"github-attestations\"".to_owned()),
            "{found:?}"
        );
        // A present but wrong value is reported once per platform, not twice.
        let mut wrong = dropped;
        for asset in wrong
            .get_mut("aqua:astral-sh/uv")
            .unwrap()
            .platforms
            .values_mut()
        {
            asset.provenance = Some("trust-me".to_owned());
        }
        assert_eq!(
            problems(&config, &wrong),
            [
                "aqua:astral-sh/uv: linux-x64 was not locked with provenance = \"github-attestations\"",
                "aqua:astral-sh/uv: windows-x64 was not locked with provenance = \"github-attestations\"",
            ]
        );
    }

    #[test]
    fn a_file_name_that_is_not_plain_is_reported() {
        for file in ["..", ".", "a%2fb", "a?b", "a#b", "a\\b"] {
            let url = format!("https://github.com/example/tool/releases/download/v1.2.3/{file}");
            let found = problems_of(&lock(&url, Some(WINDOWS)));
            assert!(
                found
                    .iter()
                    .any(|p| p.contains("is not a release download")),
                "{file}: {found:?}"
            );
        }
    }

    #[test]
    fn an_asset_from_a_release_other_than_the_locked_version_is_reported() {
        let older = LINUX.replace("/v1.2.3/", "/v0.0.1/");
        let found = problems_of(&lock(&older, Some(WINDOWS)));
        assert_eq!(
            found,
            [
                "github:example/tool: linux-x64 asset comes from release \"v0.0.1\", not the locked version 1.2.3"
            ],
        );
        // An unprefixed tag naming the version is the same release.
        let bare = LINUX.replace("/v1.2.3/", "/1.2.3/");
        assert!(problems_of(&lock(&bare, Some(WINDOWS))).is_empty());
    }

    #[test]
    fn a_platform_the_config_does_not_list_is_reported() {
        let mut text = lock(LINUX, Some(WINDOWS));
        write!(
            text,
            "\n[tools.\"github:example/tool\".\"platforms.macos-x64\"]\nchecksum = \"sha256:{}\"\nurl = \"https://evil.example/x\"\n",
            "a".repeat(64)
        )
        .unwrap();
        let found = problems_of(&text);
        assert!(
            found
                .iter()
                .any(|p| p.contains("locks \"macos-x64\", which runtimes/mise.toml does not list")),
            "{found:?}"
        );
    }

    #[test]
    fn lockfile_platforms_must_be_exactly_the_supported_ones() {
        for platforms in [
            "[]",
            "[\"linux-x64\"]",
            "[\"linux-x64\", \"windows-x64\", \"macos-arm64\"]",
        ] {
            let text = CONFIG.replace("[\"linux-x64\", \"windows-x64\"]", platforms);
            let err = MiseConfig::parse(&text).unwrap_err();
            assert!(
                format!("{err:#}").contains("lockfile_platforms must be exactly"),
                "{platforms}: {err:#}"
            );
        }
        let reordered = CONFIG.replace(
            "[\"linux-x64\", \"windows-x64\"]",
            "[\"windows-x64\", \"linux-x64\"]",
        );
        assert!(MiseConfig::parse(&reordered).is_ok());
    }

    #[test]
    fn a_repository_name_that_is_not_plain_is_refused() {
        assert_eq!(release_prefix("github:../x"), None);
        assert_eq!(release_prefix("github:o/x y"), None);
        assert_eq!(release_prefix("github:o/%2e"), None);
        assert!(release_prefix("github:o-1/x_y.z").is_some());
    }

    #[test]
    fn entries_naming_different_backends_are_refused() {
        let text = lock(LINUX, Some(WINDOWS)).replacen(
            "backend = \"github:example/tool\"",
            "backend = \"github:example/other\"",
            1,
        );
        assert!(format!("{:#}", parse_lock(&text).unwrap_err()).contains("different backends"));
    }

    #[test]
    fn a_sixty_four_character_digest_that_is_not_hex_is_reported() {
        let text = lock(LINUX, Some(WINDOWS)).replacen(&"a".repeat(64), &"z".repeat(64), 1);
        let found = problems_of(&text);
        assert!(
            found.iter().any(|p| p.contains("not a sha256 hex digest")),
            "{found:?}"
        );
    }

    #[test]
    fn a_lock_the_build_would_reject_is_reported() {
        let text =
            lock(LINUX, Some(WINDOWS)).replace("lockfile_version = 3", "lockfile_version = 4");
        assert!(generator_problems(&text).is_some_and(|p| p.contains("lockfile_version")));
        assert_eq!(generator_problems(&lock(LINUX, Some(WINDOWS))), None);
    }

    #[test]
    fn a_tool_named_without_its_repository_is_reported() {
        let config =
            MiseConfig::parse(&CONFIG.replace("[tools]", "[tools]\nuv = \"0.13.6\"")).unwrap();
        let mut lock = parse_lock(&lock(LINUX, Some(WINDOWS))).unwrap();
        lock.insert(
            "uv".to_owned(),
            LockedTool {
                version: "0.13.6".to_owned(),
                backend: Some("aqua:astral-sh/uv".to_owned()),
                platforms: BTreeMap::new(),
            },
        );
        let found = problems(&config, &lock);
        assert!(
            found.iter().any(|p| p.starts_with("uv: name tools in")),
            "{found:?}"
        );
    }

    fn root_with(lock_text: Option<&str>) -> tempfile::TempDir {
        let source = crate::paths::workspace_root().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("runtimes")).unwrap();
        std::fs::copy(source.join(MISE_CONFIG), root.path().join(MISE_CONFIG)).unwrap();
        if let Some(text) = lock_text {
            std::fs::write(root.path().join(MISE_LOCK), text).unwrap();
        }
        root
    }

    #[test]
    fn check_reports_a_lock_the_build_would_reject() {
        let committed =
            std::fs::read_to_string(crate::paths::workspace_root().unwrap().join(MISE_LOCK))
                .unwrap();
        let root = root_with(Some(
            &committed.replace("lockfile_version = 3", "lockfile_version = 4"),
        ));
        let found = check(root.path()).unwrap();
        assert!(
            found
                .iter()
                .any(|p| p.contains("crates/rocm-deps would reject it")),
            "{found:?}"
        );
        let root = root_with(Some(&committed));
        assert_eq!(check(root.path()).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn check_reports_a_missing_or_unparsable_lock_as_a_problem() {
        let missing = check(root_with(None).path()).unwrap();
        assert!(
            missing.len() == 1 && missing[0].starts_with("runtimes/mise.lock: cannot read it"),
            "{missing:?}"
        );
        let garbled = check(root_with(Some("not = [toml")).path()).unwrap();
        assert!(
            garbled.len() == 1
                && garbled[0].starts_with("runtimes/mise.lock: ")
                && !garbled[0].contains("cannot read"),
            "{garbled:?}"
        );
    }

    #[test]
    fn check_reports_a_missing_or_unparsable_config_as_a_problem() {
        let committed =
            std::fs::read_to_string(crate::paths::workspace_root().unwrap().join(MISE_LOCK))
                .unwrap();
        let root = root_with(Some(&committed));
        std::fs::remove_file(root.path().join(MISE_CONFIG)).unwrap();
        let missing = check(root.path()).unwrap();
        assert!(
            missing.len() == 1 && missing[0].starts_with("runtimes/mise.toml: cannot read it"),
            "{missing:?}"
        );
        std::fs::write(root.path().join(MISE_CONFIG), "not = [toml").unwrap();
        let garbled = check(root.path()).unwrap();
        assert!(
            garbled.len() == 1
                && garbled[0].starts_with("runtimes/mise.toml: ")
                && !garbled[0].contains("cannot read"),
            "{garbled:?}"
        );
    }
}
