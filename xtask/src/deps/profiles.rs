// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `runtimes/profiles.toml`: the Python runtime stacks and the policy their
//! locks must satisfy.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// Workspace-relative path of the profile file.
pub(crate) const PROFILES_FILE: &str = "runtimes/profiles.toml";

/// Workspace-relative directory every lock is written to. Fixed rather than
/// configurable: `deps lock` deletes whatever in it no profile produces, and
/// the CI gate, the update workflow's commit step and `.gitattributes` all
/// name it.
pub(crate) const LOCK_DIR: &str = "runtimes/locks";

/// A workspace-relative path as it is spelled in messages, docs and git:
/// `/`-separated on every OS. `Path::join` uses `\\` on Windows, which would
/// print `runtimes/locks\\pylock…`.
pub(crate) fn slash_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Operating systems a profile can be locked for.
pub(crate) const KNOWN_OSES: [&str; 2] = ["linux", "windows"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Profiles {
    pub(crate) policy: Policy,
    #[serde(default)]
    pub(crate) profile: BTreeMap<String, Profile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub(crate) struct Policy {
    pub(crate) uv_config: PathBuf,
    pub(crate) vendor_hosts: Vec<String>,
    pub(crate) vendor_packages: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Profile {
    /// ROCm line every `+rocm…` build in this profile's locks must belong to.
    pub(crate) rocm: Option<String>,
    /// Python version the profile is resolved for (`3.14`).
    pub(crate) python: String,
    pub(crate) requirements: Vec<String>,
    /// OS -> uv `--python-platform`.
    pub(crate) platforms: BTreeMap<String, String>,
}

impl Profiles {
    pub(crate) fn load(root: &Path) -> Result<Self> {
        let path = root.join(PROFILES_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("invalid {}", path.display()))
    }

    pub(crate) fn parse(text: &str) -> Result<Self> {
        let profiles: Self = toml::from_str(text)?;
        profiles.validate()?;
        Ok(profiles)
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.profile.is_empty(), "no [profile.<name>] tables");
        for (name, profile) in &self.profile {
            ensure!(
                !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "profile name {name:?} must be lowercase letters, digits and `-` \
                 (it becomes part of a pylock.<name>.toml file name)"
            );
            ensure!(
                !profile.platforms.is_empty(),
                "profile.{name} has no platforms"
            );
            for (os, platform) in &profile.platforms {
                ensure!(
                    KNOWN_OSES.contains(&os.as_str()),
                    "profile.{name} names unknown os {os:?}"
                );
                // `deps check` accepts x86-64 wheels only, as rocm-cli supports
                // x86-64 hosts only; any other platform would fail every wheel.
                ensure!(
                    platform.starts_with("x86_64-"),
                    "profile.{name} resolves {os} for {platform:?}; only x86_64 platforms are supported"
                );
            }
            ensure!(
                !profile.requirements.is_empty(),
                "profile.{name} has no requirements"
            );
            for requirement in &profile.requirements {
                Requirement::parse(requirement)
                    .with_context(|| format!("profile.{name} requirement {requirement:?}"))?;
            }
        }
        Ok(())
    }

    /// Lock file path for a profile and OS, relative to the workspace root.
    pub(crate) fn lock_path(profile: &str, os: &str) -> PathBuf {
        Path::new(LOCK_DIR).join(lock_file_name(profile, os))
    }

    /// Every entry in [`LOCK_DIR`] that no profile produces — files, dotfiles
    /// and directories alike — as workspace-relative paths in a stable order;
    /// none when the directory does not exist. `deps check` reports exactly
    /// these and `deps lock` removes exactly these, so the two cannot disagree.
    pub(crate) fn stray_lock_entries(&self, root: &Path) -> Result<Vec<PathBuf>> {
        let expected: BTreeSet<PathBuf> = self
            .targets()
            .map(|(name, _, os, _)| Self::lock_path(name, os))
            .collect();
        let dir = root.join(LOCK_DIR);
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err).with_context(|| format!("failed to read {}", dir.display()));
            }
        };
        let mut strays = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| format!("failed to read {}", dir.display()))?;
            let relative = Path::new(LOCK_DIR).join(entry.file_name());
            if !expected.contains(&relative) {
                strays.push(relative);
            }
        }
        strays.sort();
        Ok(strays)
    }

    /// Hex SHA-256 of the per-profile inputs the `(name, os)` lock is resolved
    /// from: the profile's Python version, the platform it is resolved for,
    /// its requirements (in a fixed order) and the text of `uv.toml`. Not
    /// covered: `policy.vendor-packages`, which sets the cutoff exemptions, so
    /// an edit to it is caught by the `runtime-locks` re-lock, not offline. `deps lock`
    /// records it in the lock's header and `deps check` recomputes it, so an
    /// edited input is caught offline even when the old lock still happens to
    /// satisfy the requirements.
    pub(crate) fn inputs_digest(&self, name: &str, os: &str, uv_config: &str) -> String {
        use sha2::Digest as _;
        let profile = &self.profile[name];
        let mut requirements = profile.requirements.clone();
        requirements.sort();
        let mut hasher = sha2::Sha256::new();
        hasher.update(format!(
            "python={}\nplatform={}\n",
            profile.python,
            profile.platforms.get(os).map_or("", String::as_str)
        ));
        for requirement in requirements {
            hasher.update(format!("requirement={requirement}\n"));
        }
        // Line endings normalised, so a CRLF checkout digests the same.
        hasher.update(format!("uv.toml=\n{}", uv_config.replace("\r\n", "\n")));
        hasher
            .finalize()
            .iter()
            .fold(String::new(), |mut hex, byte| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
                hex
            })
    }

    /// Every `(profile, os)` pair that must have a lock, in a stable order.
    pub(crate) fn targets(&self) -> impl Iterator<Item = (&str, &Profile, &str, &str)> {
        self.profile.iter().flat_map(|(name, profile)| {
            profile
                .platforms
                .iter()
                .map(move |(os, platform)| (name.as_str(), profile, os.as_str(), platform.as_str()))
        })
    }

    pub(crate) fn is_vendor_package(&self, name: &str) -> bool {
        let name = normalize_name(name);
        self.policy.vendor_packages.iter().any(|pattern| {
            let pattern = normalize_name(pattern);
            match pattern.strip_suffix('*') {
                Some(prefix) => name.starts_with(prefix),
                None => name == pattern,
            }
        })
    }
}

/// `pylock.<profile>-<os>.toml` — the PEP 751 named-lock form uv accepts
/// (`pylock.<name>.toml`, `<name>` without dots).
pub(crate) fn lock_file_name(profile: &str, os: &str) -> String {
    format!("pylock.{profile}-{os}.toml")
}

/// PEP 503 normalised project name.
pub(crate) fn normalize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_sep = false;
    for c in name.chars() {
        if matches!(c, '-' | '_' | '.') {
            if !last_sep {
                out.push('-');
            }
            last_sep = true;
        } else {
            out.push(c.to_ascii_lowercase());
            last_sep = false;
        }
    }
    out
}

/// The subset of PEP 508 a profile may use: `name[extras]==version`, where the
/// version may end in `.*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Requirement {
    /// Normalised name.
    pub(crate) name: String,
    pub(crate) version: String,
    /// `true` for `==X.Y.*`.
    pub(crate) prefix: bool,
}

impl Requirement {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let (head, version) = text
            .split_once("==")
            .context("only `name==version` and `name==version.*` are supported")?;
        // `name` or `name[extra,...]`, with nothing after the closing `]`.
        let name = match head.trim().split_once('[') {
            None => head.trim(),
            Some((name, extras)) => {
                let extras = extras
                    .strip_suffix(']')
                    .with_context(|| format!("malformed extras in {text:?}"))?;
                ensure!(
                    !extras.is_empty()
                        && extras.split(',').all(|extra| {
                            !extra.is_empty()
                                && extra.chars().all(|c| {
                                    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')
                                })
                        }),
                    "malformed extras in {text:?}"
                );
                name.trim()
            }
        };
        let version = version.trim();
        // The line is written into uv's requirements file, so no whitespace
        // may split it and the name must start and end alphanumeric (PEP 508);
        // otherwise `-r` or `--pre` could pass as a package name.
        ensure!(
            !text.contains(char::is_whitespace),
            "requirement {text:?} contains whitespace"
        );
        ensure!(
            name.starts_with(|c: char| c.is_ascii_alphanumeric())
                && name.ends_with(|c: char| c.is_ascii_alphanumeric())
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
            "malformed package name {name:?}"
        );
        ensure!(
            !version.is_empty() && !version.contains([';', ',', ' ', '<', '>', '!', '~', '=', '+']),
            "malformed version {version:?}: no markers, ranges or local versions"
        );
        let (version, prefix) = match version.strip_suffix(".*") {
            Some(stem) => (stem, true),
            None => (version, false),
        };
        ensure!(
            !version.contains('*'),
            "`*` is only allowed as a trailing `.*`"
        );
        // An allow-list: the version is written into uv's requirements file,
        // so a newline or tab could otherwise smuggle in a uv option.
        ensure!(
            version
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.'),
            "malformed version {version:?}: only letters, digits and `.` are allowed"
        );
        ensure!(
            version.starts_with(|c: char| c.is_ascii_digit()),
            "malformed version {version:?}: it must start with a digit"
        );
        Ok(Self {
            name: normalize_name(name),
            version: version.to_owned(),
            prefix,
        })
    }

    /// Whether a locked version satisfies this requirement. The local segment
    /// (`+rocm10.1.0`) is ignored, as PEP 440 does for a specifier without one.
    pub(crate) fn matches(&self, locked: &str) -> bool {
        let public = locked.split_once('+').map_or(locked, |(public, _)| public);
        if !self.prefix {
            return public == self.version;
        }
        public == self.version
            || public.strip_prefix(&self.version).is_some_and(|rest| {
                // `0.29.*` admits `0.29.1.dev0` and `0.29rc1`, never `0.290`.
                rest.starts_with('.')
                    || ["a", "b", "rc", "dev", "post"]
                        .iter()
                        .any(|suffix| rest.starts_with(suffix))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
[policy]
uv-config = "runtimes/uv.toml"
vendor-hosts = ["vendor.example"]
vendor-packages = ["rocm", "rocm-sdk-*"]

[profile.p]
rocm = "10.1.0"
python = "3.14"
requirements = ["rocm[libraries]==10.1.0", "torch==2.13.*"]

[profile.p.platforms]
linux = "x86_64-manylinux_2_28"
windows = "x86_64-pc-windows-msvc"
"#;

    /// PEP 503: runs of `-`, `_` and `.` become one `-`, case folded.
    #[test]
    fn names_are_normalized_like_pep_503() {
        assert_eq!(normalize_name("Torch_Audio"), "torch-audio");
        assert_eq!(normalize_name("a-_.b"), "a-b");
        assert_eq!(normalize_name("a__b..c"), "a-b-c");
    }

    #[test]
    fn the_committed_profiles_parse() {
        let root = crate::paths::workspace_root().unwrap();
        Profiles::load(&root).expect("runtimes/profiles.toml must stay valid");
    }

    #[test]
    fn targets_cover_every_profile_and_os() {
        let profiles = Profiles::parse(MINIMAL).unwrap();
        let targets: Vec<_> = profiles.targets().map(|(n, _, os, _)| (n, os)).collect();
        assert_eq!(targets, [("p", "linux"), ("p", "windows")]);
        assert_eq!(
            Profiles::lock_path("p", "windows"),
            PathBuf::from("runtimes/locks/pylock.p-windows.toml")
        );
    }

    #[test]
    fn an_unknown_os_is_refused() {
        let text = MINIMAL.replace("windows =", "macos =");
        let err = Profiles::parse(&text).unwrap_err();
        assert!(format!("{err:#}").contains("unknown os"), "{err:#}");
    }

    #[test]
    fn a_profile_name_that_cannot_be_a_lock_name_is_refused() {
        let text = MINIMAL
            .replace("[profile.p]", "[profile.\"p.1\"]")
            .replace("[profile.p.platforms]", "[profile.\"p.1\".platforms]");
        assert!(Profiles::parse(&text).is_err());
    }

    #[test]
    fn requirements_outside_the_supported_subset_are_refused() {
        for bad in [
            "torch>=2.13",
            "torch==2.13; sys_platform == 'linux'",
            "torch==2.13+rocm10.1.0",
            "torch",
            "torch==2.*.1",
            "torch==2.13,!=2.13.1",
        ] {
            assert!(Requirement::parse(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn a_platform_other_than_x86_64_is_refused() {
        let text = MINIMAL.replace("x86_64-manylinux_2_28", "aarch64-manylinux_2_28");
        assert!(
            format!("{:#}", Profiles::parse(&text).unwrap_err())
                .contains("only x86_64 platforms are supported")
        );
    }

    #[test]
    fn malformed_requirements_are_refused() {
        for bad in [
            "torch==.*",
            "torch==",
            "rocm[x]junk==1",
            "rocm[x==1",
            "rocm[]==1",
            "rocm[a,,b]==1",
            "==1",
            "torch==2.*.1",
            "torch>=2",
            "torch==2.13\n--pre",
            "torch==2.13\t--index-url\thttp://x",
            "--pre\n==1",
            "-r\n==1",
            "-e\t==1",
            "-torch==1",
            "torch-==1",
        ] {
            assert!(Requirement::parse(bad).is_err(), "{bad} should be refused");
        }
        for good in ["torch==2.13.*", "rocm[libraries,device-gfx1100]==10.1.0"] {
            assert!(Requirement::parse(good).is_ok(), "{good}");
        }
    }

    #[test]
    fn exact_and_prefix_requirements_match_like_pep_440() {
        let exact = Requirement::parse("torchaudio==2.11.0.2").unwrap();
        assert!(exact.matches("2.11.0.2+rocm10.1.0"));
        assert!(!exact.matches("2.11.0.3+rocm10.1.0"));
        assert!(!exact.matches("2.11.0"));

        let series = Requirement::parse("vllm==0.29.*").unwrap();
        assert!(series.matches("0.29.1.dev0+rocm10.1.0.g98dff2a81.d20261002"));
        assert!(series.matches("0.29"));
        for pre in ["0.29rc1", "0.29a1", "0.29b2", "0.29dev1", "0.29post1"] {
            assert!(series.matches(pre), "{pre}");
        }
        assert!(!series.matches("0.290.0"));
        assert!(!series.matches("0.30.0"));

        let extras = Requirement::parse("Rocm_SDK[libraries,device-gfx1151]==10.1.0").unwrap();
        assert_eq!(extras.name, "rocm-sdk");
    }

    #[test]
    fn vendor_patterns_match_normalised_names() {
        let profiles = Profiles::parse(MINIMAL).unwrap();
        assert!(profiles.is_vendor_package("rocm"));
        assert!(profiles.is_vendor_package("rocm_sdk_core"));
        assert!(!profiles.is_vendor_package("rocm-bootstrap"));
        assert!(!profiles.is_vendor_package("numpy"));
    }

    #[test]
    fn a_profile_without_platforms_or_requirements_is_refused() {
        let no_platforms = MINIMAL
            .replace("linux = \"x86_64-manylinux_2_28\"\n", "")
            .replace("windows = \"x86_64-pc-windows-msvc\"\n", "");
        assert!(
            format!("{:#}", Profiles::parse(&no_platforms).unwrap_err())
                .contains("has no platforms")
        );
        let no_requirements = MINIMAL.replace(
            "requirements = [\"rocm[libraries]==10.1.0\", \"torch==2.13.*\"]",
            "requirements = []",
        );
        assert!(
            format!("{:#}", Profiles::parse(&no_requirements).unwrap_err())
                .contains("has no requirements")
        );
    }

    #[test]
    fn the_lock_directory_is_not_configurable() {
        // `deps lock` deletes from it, so a profile file must not be able to
        // point it at `runtimes/bin` or anywhere else.
        let text = MINIMAL.replace("[policy]\n", "[policy]\ndirectory = \"runtimes/bin\"\n");
        assert!(
            format!("{:#}", Profiles::parse(&text).unwrap_err())
                .contains("unknown field `directory`")
        );
    }

    #[test]
    fn a_file_without_profiles_is_refused() {
        let policy_only = &MINIMAL[..MINIMAL.find("[profile.p]").unwrap()];
        assert!(
            format!("{:#}", Profiles::parse(policy_only).unwrap_err())
                .contains("no [profile.<name>] tables")
        );
    }
}
