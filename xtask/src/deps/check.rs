// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask deps check`: offline verification that every runtime lock is
//! complete and consistent with what it was generated from.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Result, bail};

use super::mise_lock;
use super::profiles::{Profile, Profiles, Requirement, normalize_name, slash_path};
use super::pylock::{self, PyLock};

pub(crate) fn run() -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let problems = collect(&root)?;
    if problems.is_empty() {
        eprintln!("runtime locks are complete and consistent");
        return Ok(());
    }
    for problem in &problems {
        eprintln!("  - {problem}");
    }
    bail!("{}", remediation(problems.len()))
}

/// What to do about `count` problems. Re-locking clears a lock that is missing
/// or stale; it cannot clear one the resolver keeps producing from the same
/// inputs, so the advice says which is which instead of promising a fix.
fn remediation(count: usize) -> String {
    format!(
        "{count} problem(s) in the runtime locks. A missing or stale lock (a pin, \
         profile or uv.toml edited without re-locking) is fixed by running `{}` \
         and committing the result. A problem that is still reported after that \
         (a package from the wrong host, a build for another ROCm line, a \
         requirement nothing satisfies) needs runtimes/profiles.toml, uv.toml or \
         mise.toml changed first.",
        super::LOCK_COMMAND
    )
}

fn collect(root: &Path) -> Result<Vec<String>> {
    let mut problems = mise_lock::check(root)?;
    let profiles = Profiles::load(root)?;
    let uv_path = &profiles.policy.uv_config;
    // Without uv.toml the inputs digest cannot be recomputed; that is
    // reported once here rather than as every lock being stale.
    let uv_config = match fs::read_to_string(root.join(uv_path)) {
        Ok(text) => Some(text),
        Err(err) => {
            problems.push(format!(
                "{}: missing or unreadable ({err})",
                slash_path(uv_path)
            ));
            None
        }
    };
    for (name, profile, os, _) in profiles.targets() {
        let path = Profiles::lock_path(name, os);
        match fs::read_to_string(root.join(&path)) {
            Ok(text) => problems.extend(
                lock_problems(&profiles, name, profile, os, uv_config.as_deref(), &text)
                    .into_iter()
                    .map(|p| format!("{}: {p}", slash_path(&path))),
            ),
            Err(err) => problems.push(format!(
                "{}: missing or unreadable ({err})",
                slash_path(&path)
            )),
        }
    }
    problems.extend(lock_dir_problems(root, &profiles)?);
    Ok(problems)
}

/// Every entry in the lock directory that no profile produces.
fn lock_dir_problems(root: &Path, profiles: &Profiles) -> Result<Vec<String>> {
    Ok(profiles
        .stray_lock_entries(root)?
        .into_iter()
        .map(|path| format!("{}: not a lock any profile produces", slash_path(&path)))
        .collect())
}

/// Every problem with one lock file, as readable messages.
pub(crate) fn lock_problems(
    profiles: &Profiles,
    name: &str,
    profile: &Profile,
    os: &str,
    uv_config: Option<&str>,
    text: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    let header = pylock::header(name, os);
    if !text.starts_with(&header) {
        out.push("does not start with the `cargo xtask deps lock` header".to_owned());
    } else if let Some(uv_config) = uv_config
        && !text[header.len()..].starts_with(&pylock::inputs_line(
            &profiles.inputs_digest(name, os, uv_config),
        ))
    {
        out.push(
            "was locked from other inputs: the profile's python, platform or \
             requirements, or uv.toml, changed since"
                .to_owned(),
        );
    }
    // Without its cutoff a lock cannot be reproduced: a re-lock would
    // resolve against a new one and move packages.
    if text.starts_with(&header) && pylock::recorded_cutoff(text).is_none() {
        out.push("records no `# exclude-newer:` cutoff to re-lock at".to_owned());
    }
    let lock = match PyLock::parse(text) {
        Ok(lock) => lock,
        Err(err) => {
            out.push(format!("{err:#}"));
            return out;
        }
    };

    let mut versions: BTreeMap<String, &str> = BTreeMap::new();
    for package in &lock.packages {
        let package_name = &package.name;
        if !pylock::is_valid_name(package_name) {
            out.push(format!("{package_name:?}: is not a valid package name"));
            continue;
        }
        let key = normalize_name(&package.name);
        let Some(version) = package.version.as_deref() else {
            out.push(format!("{}: no version", package.name));
            continue;
        };
        if !pylock::is_valid_version(version) {
            out.push(format!(
                "{package_name}: {version:?} is not a valid version"
            ));
            continue;
        }
        if versions.insert(key, version).is_some() {
            out.push(format!("{}: locked more than once", package.name));
        }
        out.extend(package_problems(profiles, profile, os, package, version));
    }

    for requirement in &profile.requirements {
        // `Profiles::parse` refused any requirement that does not parse.
        let requirement =
            Requirement::parse(requirement).expect("profile requirements are validated on load");
        match versions.get(&requirement.name) {
            None => out.push(format!("{}: required but not locked", requirement.name)),
            Some(version) if !requirement.matches(version) => out.push(format!(
                "{}: locked {version}, which does not satisfy =={}{}",
                requirement.name,
                requirement.version,
                if requirement.prefix { ".*" } else { "" }
            )),
            Some(_) => {}
        }
    }
    out
}

/// The host PyPI serves its files from.
const PYPI_FILES_HOST: &str = "files.pythonhosted.org";

fn package_problems(
    profiles: &Profiles,
    profile: &Profile,
    os: &str,
    package: &pylock::Package,
    version: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    let name = &package.name;
    if package.artifacts().next().is_none() {
        out.push(format!("{name}: no wheel or sdist to install from"));
    }
    for key in package.unchecked_sources() {
        out.push(format!(
            "{name}: has a `{key}` source, which this check does not verify; only wheels and sdists are allowed"
        ));
    }
    let vendor = profiles.is_vendor_package(name);
    // Every lock is resolved on a Linux runner, the Windows ones included, so
    // nothing but this check stops a wheel for the wrong OS reaching a lock.
    for wheel in &package.wheels {
        let url = wheel.url.as_deref().unwrap_or_default();
        let file = url.rsplit('/').next().unwrap_or_default();
        if !wheel_fits_os(file, os) {
            out.push(format!("{name}: {file} is not a wheel for {os}"));
        }
        if !wheel_fits_python(file, &profile.python) {
            out.push(format!(
                "{name}: {file} is not a wheel for Python {}",
                profile.python
            ));
        }
    }
    for artifact in package.artifacts() {
        let url = artifact.url.as_deref().unwrap_or_default();
        let Some(host) = pylock::https_host(url) else {
            out.push(format!("{name}: artifact is not an https URL: {url:?}"));
            continue;
        };
        if !artifact.hashes.get("sha256").is_some_and(|h| {
            h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        }) {
            out.push(format!("{name}: {url} has no sha256"));
        }
        let from_vendor = profiles
            .policy
            .vendor_hosts
            .iter()
            .any(|allowed| allowed == host);
        // Every artifact comes from an index `uv.toml` names: AMD's or PyPI's.
        // A lock is written by a job that ran upstream code, and the update's
        // summary shows versions and a short digest but no URLs or hosts, so a
        // repointed download would otherwise go unseen.
        if !from_vendor && host != PYPI_FILES_HOST {
            out.push(format!(
                "{name}: {url} is served by {host}, which is neither {PYPI_FILES_HOST} nor one of {}",
                profiles.policy.vendor_hosts.join(", ")
            ));
        }
        if vendor && !from_vendor {
            out.push(format!(
                "{name}: AMD-only package taken from {host}, not from {}",
                profiles.policy.vendor_hosts.join(" or ")
            ));
        }
    }
    if let (Some(line), Some(built_for)) = (profile.rocm.as_deref(), rocm_line(version))
        && built_for != line
    {
        out.push(format!(
            "{name} {version}: built for ROCm {built_for}, but this profile is ROCm {line}"
        ));
    }
    out
}

/// Whether a wheel file name's platform tag can install on `os`: `any`, or
/// for Linux an x86-64 `linux_*`/`manylinux*` tag, for Windows `win_amd64`. A
/// compressed tag set (`manylinux_2_17_x86_64.manylinux2014_x86_64`) fits
/// when every tag in it does.
fn wheel_fits_os(file: &str, os: &str) -> bool {
    let Some(stem) = file.strip_suffix(".whl") else {
        return false;
    };
    let Some(platforms) = stem.rsplit('-').next() else {
        return false;
    };
    platforms.split('.').all(|platform| match (os, platform) {
        (_, "any") => true,
        ("linux", tag) => {
            (tag.starts_with("linux_") || tag.starts_with("manylinux")) && tag.ends_with("_x86_64")
        }
        ("windows", tag) => tag == "win_amd64",
        _ => false,
    })
}

/// Whether a wheel file name's Python and ABI tags can install on CPython
/// `python` (`3.14`): a pure-Python tag (`py3`, or `py3X` up to that minor),
/// that CPython's own tag and ABI (`cp314-cp314`), or a stable-ABI wheel
/// built for that minor or an older one (`cp312-abi3`). Not a wheel for
/// another CPython (`cp312-cp312`) or a free-threaded one (`cp314t`).
fn wheel_fits_python(file: &str, python: &str) -> bool {
    let Some(stem) = file.strip_suffix(".whl") else {
        return false;
    };
    let mut tags = stem.rsplit('-');
    let (Some(_platform), Some(abi), Some(pythons)) = (tags.next(), tags.next(), tags.next())
    else {
        return false;
    };
    let Some(minor) = python
        .strip_prefix("3.")
        .and_then(|minor| minor.parse::<u32>().ok())
    else {
        return false;
    };
    let cp_minor = |tag: &str| tag.strip_prefix("cp3").and_then(|m| m.parse::<u32>().ok());
    pythons.split('.').any(|tag| {
        if tag == "py3" {
            return true;
        }
        if let Some(m) = tag.strip_prefix("py3").and_then(|m| m.parse::<u32>().ok()) {
            return m <= minor && abi == "none";
        }
        match cp_minor(tag) {
            Some(m) if abi == "abi3" => m <= minor,
            Some(m) => m == minor && (abi == tag || abi == "none"),
            None => false,
        }
    })
}

/// The ROCm line in a version's local segment: `2.13.0+rocm10.1.0` and
/// `3.8.0+git669b31ac.rocm10.1.0` both give `10.1.0`.
fn rocm_line(version: &str) -> Option<&str> {
    let local = version.split_once('+')?.1;
    // `rocm` must open a dot-separated segment and be followed by a digit.
    let at = std::iter::once(0)
        .chain(local.match_indices('.').map(|(i, _)| i + 1))
        .find(|&i| {
            local[i..]
                .strip_prefix("rocm")
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
        })?;
    let rest = &local[at + "rocm".len()..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(rest.len());
    Some(rest[..end].trim_end_matches('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILES: &str = r#"
[policy]
uv-config = "runtimes/uv.toml"
vendor-hosts = ["stable.repo.amd.com"]
vendor-packages = ["rocm", "torch"]

[profile.p]
rocm = "10.1.0"
python = "3.14"
requirements = ["rocm[libraries]==10.1.0", "torch==2.13.*"]

[profile.p.platforms]
linux = "x86_64-manylinux_2_28"
"#;

    const SUM: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    const UV: &str = "[[index]]\nurl = \"https://stable.repo.amd.com/simple/\"\n";

    /// The header a fresh `deps lock` of profile `p` writes for `os`.
    fn head(os: &str) -> String {
        let profiles = Profiles::parse(PROFILES).unwrap();
        format!(
            "{}{}{}",
            pylock::header("p", os),
            pylock::inputs_line(&profiles.inputs_digest("p", os, UV)),
            pylock::cutoff_line("2026-10-30T08:00:00Z")
        )
    }

    fn lock_text(torch_version: &str, torch_host: &str) -> String {
        format!(
            "{}lock-version = \"1.0\"\n\n\
             [[packages]]\nname = \"rocm\"\nversion = \"10.1.0\"\n\
             sdist = {{ url = \"https://stable.repo.amd.com/rocm/rocm-10.1.0.tar.gz\", hashes = {{ sha256 = \"{SUM}\" }} }}\n\n\
             [[packages]]\nname = \"torch\"\nversion = \"{torch_version}\"\n\
             wheels = [{{ url = \"https://{torch_host}/torch-2.13.0-cp314-cp314-linux_x86_64.whl\", hashes = {{ sha256 = \"{SUM}\" }} }}]\n\n\
             [[packages]]\nname = \"numpy\"\nversion = \"2.3.0\"\n\
             wheels = [{{ url = \"https://files.pythonhosted.org/numpy-2.3.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl\", hashes = {{ sha256 = \"{SUM}\" }} }}]\n",
            head("linux")
        )
    }

    fn problems_of(text: &str) -> Vec<String> {
        let profiles = Profiles::parse(PROFILES).unwrap();
        lock_problems(
            &profiles,
            "p",
            &profiles.profile["p"],
            "linux",
            Some(UV),
            text,
        )
    }

    #[test]
    fn the_committed_locks_pass() {
        let root = crate::paths::workspace_root().unwrap();
        let problems = collect(&root).unwrap();
        assert!(
            problems.is_empty(),
            "{problems:#?}\n{}\nA profile's first lock comes from the `runtime-locks` CI \
             artifact; see docs/runtime-dependencies.md.",
            remediation(problems.len())
        );
    }

    /// A field `deps check` does not know is refused at every level, as in
    /// `mise.lock`; the fields uv writes today are accepted.
    #[test]
    fn a_python_lock_field_this_check_does_not_understand_is_refused() {
        let known = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
            .replacen(
                "lock-version = \"1.0\"\n",
                "lock-version = \"1.0\"\ncreated-by = \"uv\"\nrequires-python = \"==3.14.*\"\n",
                1,
            )
            .replacen(
                "name = \"numpy\"\n",
                "name = \"numpy\"\nmarker = \"sys_platform == 'linux'\"\n",
                1,
            )
            .replacen(
                &format!("hashes = {{ sha256 = \"{SUM}\" }} }}]\n"),
                &format!(
                    "hashes = {{ sha256 = \"{SUM}\" }}, upload-time = 2026-01-01T00:00:00Z, size = 1 }}]\n"
                ),
                1,
            );
        assert_eq!(problems_of(&known), Vec::<String>::new());
        for (level, text) in [
            (
                "top",
                known.replacen(
                    "lock-version = \"1.0\"\n",
                    "lock-version = \"1.0\"\nextra = 1\n",
                    1,
                ),
            ),
            (
                "package",
                known.replacen("name = \"numpy\"\n", "name = \"numpy\"\nextra = 1\n", 1),
            ),
            (
                "artifact",
                known.replacen(", size = 1 }]", ", size = 1, extra = 1 }]", 1),
            ),
        ] {
            assert_ne!(text, known, "{level}: the field was added");
            let found = problems_of(&text);
            assert_eq!(found.len(), 1, "{level}: {found:?}");
            assert!(
                found[0].contains("unknown field `extra`"),
                "{level}: {found:?}"
            );
        }
    }

    #[test]
    fn a_consistent_lock_has_no_problems() {
        assert_eq!(
            problems_of(&lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")),
            Vec::<String>::new()
        );
    }

    #[test]
    fn an_amd_package_from_another_host_is_reported() {
        let found = problems_of(&lock_text("2.13.0", "files.pythonhosted.org"));
        assert!(
            found
                .iter()
                .any(|p| p.contains("AMD-only package taken from files.pythonhosted.org")),
            "{found:?}"
        );
    }

    /// Each input `deps lock` resolves from makes an existing lock stale when
    /// it changes, even where the old lock still satisfies the requirements.
    #[test]
    fn a_lock_resolved_from_other_inputs_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com");
        assert!(problems_of(&text).is_empty());
        let stale = "was locked from other inputs: the profile's python, platform or \
                     requirements, or uv.toml, changed since";
        for (what, profiles_text, uv) in [
            (
                "an added extra",
                PROFILES.replace("\"torch==2.13.*\"", "\"torch[device-gfx1100]==2.13.*\""),
                UV.to_owned(),
            ),
            (
                "another python",
                PROFILES.replace("3.14", "3.13"),
                UV.to_owned(),
            ),
            (
                "another platform",
                PROFILES.replace("manylinux_2_28", "manylinux_2_34"),
                UV.to_owned(),
            ),
            (
                "a dropped requirement",
                PROFILES.replace(", \"torch==2.13.*\"", ""),
                UV.to_owned(),
            ),
            (
                "a loosened pin the lock still satisfies",
                PROFILES.replace("torch==2.13.*", "torch==2.*"),
                UV.to_owned(),
            ),
            (
                "an edited uv.toml",
                PROFILES.to_owned(),
                format!("{UV}index-strategy = \"first-index\"\n"),
            ),
        ] {
            let profiles = Profiles::parse(&profiles_text).unwrap();
            let found = lock_problems(
                &profiles,
                "p",
                &profiles.profile["p"],
                "linux",
                Some(&uv),
                &text,
            );
            assert!(found.iter().any(|p| p == stale), "{what}: {found:?}");
        }
        // Requirement order and CRLF line endings are not input changes.
        let reordered = PROFILES.replace(
            "[\"rocm[libraries]==10.1.0\", \"torch==2.13.*\"]",
            "[\"torch==2.13.*\", \"rocm[libraries]==10.1.0\"]",
        );
        let profiles = Profiles::parse(&reordered).unwrap();
        let crlf = UV.replace('\n', "\r\n");
        assert!(
            lock_problems(
                &profiles,
                "p",
                &profiles.profile["p"],
                "linux",
                Some(&crlf),
                &text
            )
            .is_empty()
        );
    }

    /// An unreadable uv.toml is one problem, reported by `collect`; it does
    /// not make every lock look stale, which no re-lock could clear.
    #[test]
    fn without_uv_toml_locks_are_not_called_stale() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com");
        let profiles = Profiles::parse(PROFILES).unwrap();
        assert!(
            lock_problems(&profiles, "p", &profiles.profile["p"], "linux", None, &text).is_empty()
        );
    }

    /// A missing uv.toml must fail `deps check` itself: without it the
    /// inputs digest is not compared, so nothing else would.
    #[test]
    fn collect_reports_a_missing_uv_toml() {
        let source = crate::paths::workspace_root().unwrap();
        let root = tempfile::tempdir().unwrap();
        let runtimes = root.path().join("runtimes");
        std::fs::create_dir_all(&runtimes).unwrap();
        for file in ["mise.toml", "mise.lock", "profiles.toml"] {
            std::fs::copy(source.join("runtimes").join(file), runtimes.join(file)).unwrap();
        }
        let found = collect(root.path()).unwrap();
        assert!(
            found
                .iter()
                .any(|p| p.starts_with("runtimes/uv.toml: missing or unreadable (")),
            "{found:?}"
        );
    }

    #[test]
    fn a_lock_without_its_cutoff_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
            .replace(&pylock::cutoff_line("2026-10-30T08:00:00Z"), "");
        assert_eq!(
            problems_of(&text),
            ["records no `# exclude-newer:` cutoff to re-lock at"]
        );
    }

    /// A host is AMD's only when it is exactly one of `vendor-hosts`: a
    /// lookalike that merely ends with it is someone else's.
    #[test]
    fn a_lookalike_of_an_amd_host_is_not_amd() {
        let found = problems_of(&lock_text("2.13.0+rocm10.1.0", "evilstable.repo.amd.com"));
        assert!(
            found
                .iter()
                .any(|p| p.contains("AMD-only package taken from evilstable.repo.amd.com")),
            "{found:?}"
        );
    }

    /// A source PEP 751 allows besides wheels and an sdist is refused, since
    /// nothing here checks its host or hash.
    #[test]
    fn a_vcs_directory_or_archive_source_is_refused() {
        for (key, entry) in [
            (
                "vcs",
                "vcs = { type = \"git\", url = \"https://evil.example/x.git\", commit-id = \"abc\" }",
            ),
            ("directory", "directory = { path = \"/tmp/x\" }"),
            (
                "archive",
                "archive = { url = \"https://evil.example/x.zip\", hashes = { sha256 = \"aa\" } }",
            ),
        ] {
            let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com").replacen(
                "[[packages]]\nname = \"numpy\"\nversion = \"2.3.0\"\n",
                &format!("[[packages]]\nname = \"numpy\"\nversion = \"2.3.0\"\n{entry}\n"),
                1,
            );
            assert_eq!(
                problems_of(&text),
                [format!(
                    "numpy: has a `{key}` source, which this check does not verify; only wheels and sdists are allowed"
                )],
                "{key}"
            );
        }
    }

    /// A name that is not a PEP 508 name is refused: it would otherwise be
    /// rendered as-is into the update's Markdown table.
    #[test]
    fn a_package_name_outside_pep_508_is_refused() {
        for bad in ["num|py", "num`py", "-numpy", "numpy-", "num py", "numpé"] {
            // numpy's wheel also comes from a host nothing allows, so the
            // entry's other checks would report it if the refused entry were
            // not skipped.
            let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
                .replacen("name = \"numpy\"", &format!("name = \"{bad}\""), 1)
                .replace(
                    "https://files.pythonhosted.org/numpy",
                    "https://evil.example/numpy",
                );
            let found = problems_of(&text);
            // Exactly one problem: the entry is refused and skipped.
            assert_eq!(
                found,
                [format!("{bad:?}: is not a valid package name")],
                "{bad}"
            );
        }
    }

    #[test]
    fn a_version_outside_pep_440_characters_is_refused() {
        for bad in ["2.3.0|x", "2.3.0 x", "v2.3.0", "2.3é", "2.3.0\n| `x` | y |"] {
            let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com").replacen(
                "version = \"2.3.0\"",
                &format!("version = {bad:?}"),
                1,
            );
            let found = problems_of(&text);
            assert_eq!(
                found,
                [format!("numpy: {bad:?} is not a valid version")],
                "{bad:?}"
            );
        }
        // A refused version is skipped, not checked against the requirement
        // too: torch is then reported as required but not locked.
        let torch = lock_text("2.13.0|x", "stable.repo.amd.com");
        assert_eq!(
            problems_of(&torch),
            [
                "torch: \"2.13.0|x\" is not a valid version",
                "torch: required but not locked"
            ]
        );
        // Every allowed character, in names and versions, is accepted.
        for (name, version) in [
            ("zope.interface", "1.0-1"),
            ("a_b", "1.0+a_b"),
            ("a-b", "1!2.3.0"),
        ] {
            let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
                .replacen("name = \"numpy\"", &format!("name = \"{name}\""), 1)
                .replacen(
                    "version = \"2.3.0\"",
                    &format!("version = \"{version}\""),
                    1,
                );
            assert_eq!(problems_of(&text), Vec::<String>::new(), "{name} {version}");
        }
        // An epoch (`1!2.3.0`) is a valid version.
        let epoch = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com").replacen(
            "version = \"2.3.0\"",
            "version = \"1!2.3.0\"",
            1,
        );
        assert_eq!(problems_of(&epoch), Vec::<String>::new());
    }

    #[test]
    fn a_package_served_by_neither_pypi_nor_amd_is_reported() {
        let text = lock_text("2.13.0", "stable.repo.amd.com").replace(
            "https://files.pythonhosted.org/numpy",
            "https://evil.example/numpy",
        );
        let found = problems_of(&text);
        assert_eq!(
            found,
            [
                "numpy: https://evil.example/numpy-2.3.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
                 is served by evil.example, which is neither files.pythonhosted.org nor one of stable.repo.amd.com"
            ],
        );
        assert!(problems_of(&lock_text("2.13.0", "stable.repo.amd.com")).is_empty());
    }

    #[test]
    fn a_build_for_another_rocm_line_is_reported() {
        let found = problems_of(&lock_text("2.13.0+rocm10.0.0", "stable.repo.amd.com"));
        assert!(
            found.iter().any(|p| p.contains("built for ROCm 10.0.0")),
            "{found:?}"
        );
    }

    #[test]
    fn a_version_outside_the_requirement_is_reported() {
        let found = problems_of(&lock_text("2.14.0+rocm10.1.0", "stable.repo.amd.com"));
        assert!(
            found
                .iter()
                .any(|p| p.contains("does not satisfy ==2.13.*")),
            "{found:?}"
        );
    }

    #[test]
    fn a_missing_requirement_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
            .replace("name = \"rocm\"", "name = \"rocm-other\"");
        let found = problems_of(&text);
        assert!(
            found.iter().any(|p| p == "rocm: required but not locked"),
            "{found:?}"
        );
    }

    #[test]
    fn a_hand_edited_header_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com").replacen(
            "# Generated",
            "# Edited",
            1,
        );
        let found = problems_of(&text);
        assert!(found.iter().any(|p| p.contains("header")), "{found:?}");
    }

    #[test]
    fn an_artifact_without_a_hash_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com").replacen(
            &format!("sha256 = \"{SUM}\""),
            "md5 = \"x\"",
            1,
        );
        let found = problems_of(&text);
        assert!(
            found.iter().any(|p| p.contains("has no sha256")),
            "{found:?}"
        );
    }

    #[test]
    fn rocm_lines_are_read_from_local_versions() {
        assert_eq!(rocm_line("2.13.0+rocm10.1.0"), Some("10.1.0"));
        assert_eq!(rocm_line("3.8.0+git669b31ac.rocm10.1.0"), Some("10.1.0"));
        assert_eq!(
            rocm_line("0.29.1.dev0+rocm10.1.0.g98dff2a81.d20261002"),
            Some("10.1.0")
        );
        assert_eq!(rocm_line("2.13.0"), None);
        assert_eq!(rocm_line("1.0+cu128"), None);
        assert_eq!(rocm_line("1.0+rocmx"), None);
    }

    #[test]
    fn a_wheel_for_the_other_os_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
            .replace("cp314-linux_x86_64.whl", "cp314-win_amd64.whl");
        let found = problems_of(&text);
        assert!(
            found.iter().any(|p| p.contains("is not a wheel for linux")),
            "{found:?}"
        );
    }

    #[test]
    fn wheel_platform_tags_are_matched_per_os() {
        assert!(wheel_fits_os("x-1-py3-none-any.whl", "windows"));
        assert!(wheel_fits_os(
            "x-1-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "linux"
        ));
        assert!(wheel_fits_os("x-1-cp314-cp314-win_amd64.whl", "windows"));
        assert!(!wheel_fits_os("x-1-cp314-cp314-win_amd64.whl", "linux"));
        assert!(!wheel_fits_os(
            "x-1-cp314-cp314-linux_x86_64.whl",
            "windows"
        ));
        assert!(!wheel_fits_os(
            "x-1-cp314-cp314-macosx_14_0_arm64.whl",
            "linux"
        ));
        assert!(!wheel_fits_os("x-1.tar.gz", "linux"));
        // A compressed tag set fits only when every tag in it does.
        assert!(!wheel_fits_os(
            "x-1-cp314-cp314-manylinux_2_17_x86_64.macosx_11_0_arm64.whl",
            "linux"
        ));
        assert!(!wheel_fits_os(
            "x-1-cp314-cp314-manylinux_2_28_aarch64.whl",
            "linux"
        ));
        assert!(!wheel_fits_os("x-1-cp314-cp314-linux_aarch64.whl", "linux"));
    }

    #[test]
    fn a_package_locked_twice_or_without_a_version_is_reported() {
        let base = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com");
        let twice = base.replace("name = \"numpy\"", "name = \"torch\"");
        assert!(
            problems_of(&twice)
                .iter()
                .any(|p| p == "torch: locked more than once")
        );
        let unversioned = base.replace("version = \"2.3.0\"\n", "");
        assert!(
            problems_of(&unversioned)
                .iter()
                .any(|p| p == "numpy: no version")
        );
    }

    #[test]
    fn a_package_with_nothing_to_install_is_reported() {
        let text = format!(
            "{}[[packages]]\nname = \"empty\"\nversion = \"1.0\"\n",
            lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
        );
        assert!(
            problems_of(&text)
                .iter()
                .any(|p| p == "empty: no wheel or sdist to install from")
        );
    }

    #[test]
    fn a_plain_http_artifact_or_a_short_hash_is_reported() {
        let base = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com");
        let http = base.replace(
            "https://files.pythonhosted.org",
            "http://files.pythonhosted.org",
        );
        assert!(
            problems_of(&http)
                .iter()
                .any(|p| p.contains("numpy: artifact is not an https URL"))
        );
        let short = base.replacen(&format!("sha256 = \"{SUM}\""), "sha256 = \"abc\"", 1);
        assert!(
            problems_of(&short)
                .iter()
                .any(|p| p.contains("has no sha256"))
        );
    }

    #[test]
    fn a_linux_wheel_in_a_windows_lock_is_reported() {
        let profiles = Profiles::parse(PROFILES).unwrap();
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
            .replace(&head("linux"), &head("windows"));
        let found = lock_problems(
            &profiles,
            "p",
            &profiles.profile["p"],
            "windows",
            Some(UV),
            &text,
        );
        assert!(
            found
                .iter()
                .any(|p| p.contains("is not a wheel for windows")),
            "{found:?}"
        );
    }

    #[test]
    fn wheels_must_fit_the_profiles_python() {
        for fits in [
            "x-1-py3-none-any.whl",
            "x-1-py2.py3-none-any.whl",
            "x-1-py39-none-any.whl",
            "x-1-py314-none-any.whl",
            "x-1-cp314-cp314-linux_x86_64.whl",
            "x-1-cp312-abi3-linux_x86_64.whl",
            "x-1-cp37-abi3-linux_x86_64.whl",
        ] {
            assert!(wheel_fits_python(fits, "3.14"), "{fits}");
        }
        for misfit in [
            "x-1-cp312-cp312-linux_x86_64.whl",
            "x-1-cp314-cp314t-linux_x86_64.whl",
            "x-1-cp315-cp315-linux_x86_64.whl",
            "x-1-cp315-abi3-linux_x86_64.whl",
            "x-1-py315-none-any.whl",
            "x-1-py2-none-any.whl",
        ] {
            assert!(!wheel_fits_python(misfit, "3.14"), "{misfit}");
        }
    }

    #[test]
    fn a_wheel_for_another_python_in_a_lock_is_reported() {
        let text = lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com")
            .replace("numpy-2.3.0-cp314-cp314-", "numpy-2.3.0-cp312-cp312-");
        assert_eq!(
            problems_of(&text),
            [
                "numpy: numpy-2.3.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl is not a wheel for Python 3.14"
            ]
        );
    }

    #[test]
    fn tags_outside_each_os_family_do_not_fit() {
        assert!(!wheel_fits_os("x-1-cp314-cp314-foo_x86_64.whl", "linux"));
        assert!(!wheel_fits_os("x-1-cp314-cp314-win32.whl", "windows"));
        assert!(!wheel_fits_os(
            "x-1-cp314-cp314-macosx_14_0_x86_64.whl",
            "windows"
        ));
        assert!(!wheel_fits_os("x-1-cp314-cp314-win_arm64.whl", "windows"));
        // The family is a prefix, not a substring.
        assert!(!wheel_fits_os(
            "x-1-cp314-cp314-notlinux_x86_64.whl",
            "linux"
        ));
    }

    /// `collect` is what `deps check` reports: it must include the binary-lock
    /// checks and the scan for lock-directory entries no profile produces.
    #[test]
    fn collect_reports_binary_lock_problems_and_stray_lock_files() {
        let source = crate::paths::workspace_root().unwrap();
        let root = tempfile::tempdir().unwrap();
        let runtimes = root.path().join("runtimes");
        std::fs::create_dir_all(runtimes.join("locks")).unwrap();
        for file in ["mise.toml", "profiles.toml"] {
            std::fs::copy(source.join("runtimes").join(file), runtimes.join(file)).unwrap();
        }
        let lock = std::fs::read_to_string(source.join("runtimes/mise.lock")).unwrap();
        std::fs::write(
            runtimes.join("mise.lock"),
            lock.replacen(
                "https://github.com/lemonade-sdk/lemonade/",
                "https://evil.example/",
                1,
            ),
        )
        .unwrap();
        std::fs::write(runtimes.join("locks/pylock.gone-linux.toml"), "").unwrap();
        std::fs::copy(source.join("runtimes/uv.toml"), runtimes.join("uv.toml")).unwrap();
        std::fs::write(runtimes.join("locks/pylock.sdk-rocm10-1-linux.toml"), "").unwrap();
        let found = collect(root.path()).unwrap();
        // A declared lock that is present is checked; one that is absent is
        // reported as such.
        assert!(
            found.contains(
                &"runtimes/locks/pylock.sdk-rocm10-1-linux.toml: does not start with the \
                  `cargo xtask deps lock` header"
                    .to_owned()
            ),
            "{found:?}"
        );
        assert!(
            found.iter().any(|p| p.starts_with(
                "runtimes/locks/pylock.vllm-rocm10-1-linux.toml: missing or unreadable ("
            )),
            "{found:?}"
        );
        assert!(
            found
                .iter()
                .any(|p| p.contains("is not a release download")),
            "{found:?}"
        );
        assert!(
            found
                .iter()
                .any(|p| p.ends_with("pylock.gone-linux.toml: not a lock any profile produces")),
            "{found:?}"
        );
    }

    #[test]
    fn a_sixty_four_character_hash_that_is_not_hex_is_reported() {
        let text =
            lock_text("2.13.0+rocm10.1.0", "stable.repo.amd.com").replacen(SUM, &"z".repeat(64), 1);
        assert!(
            problems_of(&text)
                .iter()
                .any(|p| p.contains("has no sha256"))
        );
    }

    #[test]
    fn every_entry_no_profile_produces_is_reported_and_only_those() {
        let root = tempfile::tempdir().unwrap();
        let profiles = Profiles::parse(PROFILES).unwrap();
        let dir = root.path().join(super::super::profiles::LOCK_DIR);
        std::fs::create_dir_all(dir.join("old")).unwrap();
        let declared: Vec<_> = profiles
            .targets()
            .map(|(name, _, os, _)| Profiles::lock_path(name, os))
            .collect();
        assert!(!declared.is_empty());
        for lock in &declared {
            std::fs::write(root.path().join(lock), "").unwrap();
        }
        let strays = [
            "pylock.gone-linux.toml",
            "README.md",
            ".keep",
            "pylock.x.toml.partial",
            "old",
        ];
        for file in &strays[..4] {
            std::fs::write(dir.join(file), "").unwrap();
        }
        let found = lock_dir_problems(root.path(), &profiles).unwrap();
        let mut expected: Vec<String> = strays
            .iter()
            .map(|name| {
                format!(
                    "{}: not a lock any profile produces",
                    format_args!("runtimes/locks/{name}")
                )
            })
            .collect();
        expected.sort();
        assert_eq!(
            found, expected,
            "workspace-relative, declared locks left out"
        );
    }

    /// Pins the wording only. That re-locking does clear a stale lock, and
    /// does not clear a policy problem, is proved by
    /// `deps::lock::tests::relocking_clears_a_stale_lock_but_not_a_policy_problem`.
    #[test]
    fn the_remediation_separates_what_relocking_fixes_from_what_it_cannot() {
        let text = remediation(2);
        assert!(text.starts_with("2 problem(s)"), "{text}");
        assert!(text.contains(super::super::LOCK_COMMAND), "{text}");
        assert!(text.contains("still reported after that"), "{text}");
        assert!(text.contains("runtimes/profiles.toml"), "{text}");
    }
}
