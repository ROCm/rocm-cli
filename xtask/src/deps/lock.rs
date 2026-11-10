// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask deps lock`: compile each profile in `runtimes/profiles.toml`,
//! once per OS, into `runtimes/locks/pylock.<profile>-<os>.toml` with uv.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};

use super::LOCK_COMMAND;
use super::mise_lock::{self, MISE_LOCK};
use super::profiles::{Profile, Profiles, lock_file_name};
use super::pylock;

/// `UV_*` variables a lock run keeps. They change how uv fetches — caching,
/// timeouts, TLS — never what it resolves; every other `UV_*` variable is
/// cleared, so a developer's own uv setup cannot leak into a committed lock.
const UV_ENV_KEPT: [&str; 7] = [
    "UV_CACHE_DIR",
    "UV_SYSTEM_CERTS",
    "UV_NO_CACHE",
    "UV_HTTP_TIMEOUT",
    "UV_CONCURRENT_DOWNLOADS",
    "UV_NATIVE_TLS",
    "UV_PYTHON_INSTALL_DIR",
];

pub(crate) struct LockOptions {
    /// Lock only this profile.
    pub(crate) profile: Option<String>,
    /// Resolve against a new cutoff (now, less the cooldown) instead of the
    /// one each lock records, so packages move to newer releases.
    pub(crate) upgrade: bool,
    /// Ignore PyPI releases younger than this many days when a new cutoff is
    /// set: with `--upgrade`, for a lock that does not exist yet, or for one
    /// whose recorded cutoff is missing or invalid. Without it, 0.
    pub(crate) cooldown_days: Option<u32>,
}

/// What a lock run asks of uv, behind a seam so the run's wiring can
/// be tested without uv, the network or the wheels.
pub(crate) trait Uv {
    /// `uv --version` output.
    fn version(&self) -> Result<String>;
    /// Run a prepared `uv pip compile` to completion.
    fn compile(&self, command: Command) -> Result<()>;
    /// The HTML of a simple-index root page, to list the projects it serves.
    fn index_page(&self, url: &str) -> Result<String>;
}

/// The `uv` on PATH.
struct SystemUv {
    /// Where fetched index pages are written.
    work: PathBuf,
}

impl Uv for SystemUv {
    fn version(&self) -> Result<String> {
        let output = Command::new("uv")
            .arg("--version")
            .output()
            .with_context(|| {
                format!("`uv` not found; run `{LOCK_COMMAND}` to use the pinned uv")
            })?;
        version_output(output.status.success(), &output.stdout, &output.stderr)
    }

    fn index_page(&self, url: &str) -> Result<String> {
        fs::create_dir_all(&self.work)?;
        let page = self.work.join("index-page.html");
        rocm_core::download_file_to_path(url, &page, Duration::from_mins(2))
            .with_context(|| format!("failed to list {url}"))?;
        Ok(fs::read_to_string(&page)?)
    }

    fn compile(&self, mut command: Command) -> Result<()> {
        let status = command.status().with_context(|| {
            format!("failed to run `uv`; run `{LOCK_COMMAND}` to use the pinned uv")
        })?;
        ensure!(status.success(), "uv exited with {status}");
        Ok(())
    }
}

pub(crate) fn run(options: &LockOptions) -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let work = crate::paths::target_dir(&root).join("deps-lock");
    let env_keys: Vec<OsString> = std::env::vars_os().map(|(key, _)| key).collect();
    let uv = SystemUv { work: work.clone() };
    run_with(
        &root,
        &work,
        options,
        &uv,
        env_keys,
        std::time::SystemTime::now().into(),
    )
}

fn run_with(
    root: &Path,
    work: &Path,
    options: &LockOptions,
    uv: &dyn Uv,
    env_keys: Vec<OsString>,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    let profiles = Profiles::load(root)?;
    if let Some(name) = &options.profile {
        ensure!(
            profiles.profile.contains_key(name),
            "no profile named {name:?}"
        );
    }
    ensure_pinned_uv(root, uv)?;

    fs::create_dir_all(work)?;
    let uv_config = root.join(&profiles.policy.uv_config);
    // Every resolution has a cutoff, so the exemptions always apply.
    let exemptions = cooldown_exemptions(&profiles, undated_index_projects(&uv_config, uv)?);
    let new_cutoff = cutoff_before(now, options.cooldown_days.unwrap_or(0));
    let cleared: Vec<String> = uv_env_to_clear(env_keys.into_iter());

    let uv_config_text = fs::read_to_string(&uv_config)
        .with_context(|| format!("failed to read {}", uv_config.display()))?;
    let mut resolved = Vec::new();
    for (name, profile, os, platform) in profiles.targets() {
        if options.profile.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        let committed = root.join(Profiles::lock_path(name, os));
        let requirements = work.join(format!("{name}-{os}.in"));
        fs::write(&requirements, profile.requirements.join("\n") + "\n")?;
        // What keeps an ordinary re-lock from moving anything is the cutoff
        // the lock records: a fresh resolution against the same cutoff and
        // the same indexes reproduces it. AMD-only packages are the exception:
        // they are exempt from the cutoff, so a new AMD build inside a range
        // moves a re-lock. The old lock is never handed to uv
        // as version preferences: with these indexes that does not converge
        // (successive runs alternate between AMD's mirrored versions of a
        // package and PyPI's), so a re-lock would never match.
        let output = work.join(lock_file_name(name, os));
        let _ = fs::remove_file(&output);
        let cutoff = match fs::read_to_string(&committed) {
            Ok(text) if !options.upgrade => pylock::recorded_cutoff(&text).unwrap_or_else(|| {
                eprintln!(
                    "{} records no valid cutoff; resolving at a new one, {new_cutoff}",
                    super::profiles::slash_path(&Profiles::lock_path(name, os))
                );
                new_cutoff.clone()
            }),
            _ => new_cutoff.clone(),
        };

        let command = uv_command(
            compile_args(
                &CompileInputs {
                    uv_config: &uv_config,
                    requirements: &requirements,
                    output: &output,
                    profile,
                    platform,
                    cutoff: &cutoff,
                },
                &exemptions,
            ),
            &cleared,
        );
        eprintln!("locking {name} for {os} ({platform})");
        uv.compile(command)
            .with_context(|| format!("uv could not lock profile {name} for {os}"))?;

        let locked = pylock::finalize(
            &fs::read_to_string(&output)?,
            name,
            os,
            &profiles.inputs_digest(name, os, &uv_config_text),
            &cutoff,
        )?;
        // A resolution that changes nothing but the cutoff keeps the lock as
        // it was, so a weekly `--upgrade` with nothing new rewrites no file
        // and proposes nothing.
        let locked = match fs::read_to_string(&committed) {
            Ok(previous) if pylock::same_but_cutoff(&previous, &locked) => previous,
            _ => locked,
        };
        resolved.push((committed, locked));
    }

    // Nothing is written until every target has resolved, and every new lock
    // is written beside its committed one before any is replaced: a failed
    // resolution or write (a full disk, say) changes no lock. Each lock is
    // then replaced by a rename within `runtimes/locks/`, so a reader never
    // sees a half-written file. A rename can still fail part-way (the
    // directory changed under the run, or on Windows a lock is held open),
    // which leaves the earlier locks replaced; the error names the lock it
    // stopped at, and no partial file is left behind.
    let mut written = Vec::with_capacity(resolved.len());
    for (committed, locked) in &resolved {
        let partial = committed.with_extension("toml.partial");
        let write = committed
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::write(&partial, locked));
        if let Err(err) = write {
            let _ = fs::remove_file(&partial);
            for (_, done) in &written {
                let _ = fs::remove_file(done);
            }
            return Err(err).with_context(|| format!("failed to write {}", committed.display()));
        }
        written.push((committed, partial));
    }
    for (index, (committed, partial)) in written.iter().enumerate() {
        if let Err(err) = fs::rename(partial, committed) {
            for (_, left) in &written[index..] {
                let _ = fs::remove_file(left);
            }
            return Err(err).with_context(|| format!("failed to write {}", committed.display()));
        }
    }

    // Stale means "no profile produces it", judged against every profile even
    // under `--profile`, so a partial run removes only true leftovers.
    remove_stale_locks(root, &profiles)?;
    Ok(())
}

struct CompileInputs<'a> {
    uv_config: &'a Path,
    requirements: &'a Path,
    output: &'a Path,
    profile: &'a Profile,
    platform: &'a str,
    /// RFC 3339 `--exclude-newer` cutoff.
    cutoff: &'a str,
}

/// `days` days before `now`, to the second, as the RFC 3339 cutoff uv takes.
fn cutoff_before(now: chrono::DateTime<chrono::Utc>, days: u32) -> String {
    (now - chrono::Duration::days(i64::from(days)))
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string()
}

/// The `uv` arguments for one profile and OS.
fn compile_args(inputs: &CompileInputs<'_>, exemptions: &BTreeSet<String>) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["pip", "compile", "--quiet", "--no-header", "--config-file"]
        .into_iter()
        .map(OsString::from)
        .collect();
    args.push(inputs.uv_config.into());
    args.extend(
        [
            "--python-version",
            &inputs.profile.python,
            "--python-platform",
            inputs.platform,
        ]
        .map(OsString::from),
    );
    args.push(inputs.requirements.into());
    args.push("-o".into());
    args.push(inputs.output.into());
    args.push("--exclude-newer".into());
    args.push(inputs.cutoff.into());
    for project in exemptions {
        args.push("--exclude-newer-package".into());
        args.push(format!("{project}=false").into());
    }
    args
}

/// The `uv` invocation for `args`, with every variable in `cleared` removed
/// from the environment it inherits.
fn uv_command(args: Vec<OsString>, cleared: &[String]) -> Command {
    let mut command = Command::new("uv");
    command.args(args);
    for var in cleared {
        command.env_remove(var);
    }
    // uv builds AMD's `rocm` sdist, i.e. runs upstream code; it needs no
    // GitHub credential, so none reaches it.
    for var in GITHUB_TOKENS {
        command.env_remove(var);
    }
    command
}

/// GitHub credentials a CI job may export, which uv never needs.
const GITHUB_TOKENS: [&str; 2] = ["GITHUB_TOKEN", "GH_TOKEN"];

/// Every `UV_*` variable among `keys` that a lock run must not inherit.
fn uv_env_to_clear(keys: impl Iterator<Item = OsString>) -> Vec<String> {
    keys.filter_map(|key| key.into_string().ok())
        .filter(|key| key.starts_with("UV_") && !UV_ENV_KEPT.contains(&key.as_str()))
        .collect()
}

/// Refuse to lock with a uv other than the one pinned in `runtimes/mise.lock`:
/// two uv versions can resolve the same inputs differently.
fn ensure_pinned_uv(root: &Path, uv: &dyn Uv) -> Result<()> {
    let tools = mise_lock::parse_lock(&fs::read_to_string(root.join(MISE_LOCK))?)?;
    let pinned = &tools
        .get(super::pins_gen::UV_TOOL)
        .with_context(|| format!("runtimes/mise.toml must pin {}", super::pins_gen::UV_TOOL))?
        .version;
    let stdout = uv.version()?;
    let found = uv_version(&stdout).map_or_else(
        || format!("a uv whose version cannot be read ({:?})", stdout.trim()),
        |version| format!("uv {version}"),
    );
    if found != format!("uv {pinned}") {
        bail!(
            "{found} is on PATH but runtimes/mise.toml pins uv {pinned}; \
             run `{LOCK_COMMAND}` to use the pinned version"
        );
    }
    Ok(())
}

/// `uv --version`'s stdout, or why it cannot be trusted: a failed run's
/// stdout is not a version.
fn version_output(success: bool, stdout: &[u8], stderr: &[u8]) -> Result<String> {
    ensure!(
        success,
        "`uv --version` failed: {}",
        String::from_utf8_lossy(stderr).trim()
    );
    Ok(String::from_utf8_lossy(stdout).into_owned())
}

/// The version in `uv --version` output (`uv 0.13.6 (x86_64-unknown-linux-gnu)`).
fn uv_version(stdout: &str) -> Option<&str> {
    let mut words = stdout.split_whitespace();
    (words.next() == Some("uv")).then(|| words.next()).flatten()
}

/// Projects the cooldown must not hold back: those AMD's indexes serve that
/// the policy names as AMD-only.
///
/// AMD's indexes publish no upload times, and uv treats an undated file as
/// newer than any `--exclude-newer` cutoff, so without an exemption a
/// cooldown would drop every AMD build. Only the AMD-only projects are
/// exempted: the common packages those indexes also mirror (setuptools,
/// numpy) fall through to PyPI under `unsafe-first-match`, and keep the
/// cooldown there. An AMD-only dependency missing from `vendor-packages`
/// fails resolution loudly rather than slipping past the cooldown — add it to
/// the policy, which `deps check` then also enforces.
fn cooldown_exemptions(profiles: &Profiles, served: BTreeSet<String>) -> BTreeSet<String> {
    served
        .into_iter()
        .filter(|project| profiles.is_vendor_package(project))
        .collect()
}

/// Project names served by the non-PyPI indexes in `uv_config`.
fn undated_index_projects(uv_config: &Path, uv: &dyn Uv) -> Result<BTreeSet<String>> {
    let config: toml::Table = toml::from_str(&fs::read_to_string(uv_config)?)
        .with_context(|| format!("invalid {}", uv_config.display()))?;
    let mut projects = BTreeSet::new();
    let indexes = config.get("index").and_then(toml::Value::as_array);
    for index in indexes.into_iter().flatten() {
        let Some(url) = index.get("url").and_then(toml::Value::as_str) else {
            continue;
        };
        if is_pypi(url) {
            continue;
        }
        projects.extend(simple_index_projects(&uv.index_page(url)?));
    }
    Ok(projects)
}

fn is_pypi(url: &str) -> bool {
    pylock::https_host(url).is_some_and(|host| host == "pypi.org" || host.ends_with(".pypi.org"))
}

/// Project names linked from a PEP 503 simple-index root page.
fn simple_index_projects(html: &str) -> BTreeSet<String> {
    html.split("href=\"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .map(|href| {
            href.trim_end_matches('/')
                .rsplit('/')
                .next()
                .unwrap_or_default()
        })
        .filter(|name| {
            // `../` and the like are navigation, not projects.
            name.chars().any(|c| c.is_ascii_alphanumeric())
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
        .map(super::profiles::normalize_name)
        .collect()
}

fn remove_stale_locks(root: &Path, profiles: &Profiles) -> Result<()> {
    for stray in profiles.stray_lock_entries(root)? {
        let path = root.join(&stray);
        eprintln!(
            "removing {}, which no profile locks",
            super::profiles::slash_path(&stray)
        );
        // `symlink_metadata`, so a symlink is removed rather than followed.
        let is_dir = fs::symlink_metadata(&path)
            .with_context(|| format!("failed to inspect {}", path.display()))?
            .is_dir();
        if is_dir {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        }
        .with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILES: &str = r#"
[policy]
uv-config = "runtimes/uv.toml"
vendor-hosts = ["vendor.example"]
vendor-packages = ["rocm", "rocm-sdk-*", "torch"]

[profile.p]
python = "3.14"
requirements = ["torch==2.13.*"]

[profile.p.platforms]
linux = "x86_64-manylinux_2_28"
"#;

    /// The fixed "now" of every test run: 2026-11-06T12:00:00Z.
    fn now() -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-11-06T12:00:00Z")
            .unwrap()
            .to_utc()
    }

    fn strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    fn compile(cutoff: &str, exemptions: &[&str]) -> Vec<String> {
        let profiles = Profiles::parse(PROFILES).unwrap();
        let inputs = CompileInputs {
            uv_config: Path::new("uv.toml"),
            requirements: Path::new("p.in"),
            output: Path::new("pylock.p-linux.toml"),
            profile: &profiles.profile["p"],
            platform: "x86_64-manylinux_2_28",
            cutoff,
        };
        let exemptions = exemptions.iter().map(|e| (*e).to_owned()).collect();
        strings(&compile_args(&inputs, &exemptions))
    }

    #[test]
    fn every_resolution_pins_python_platform_config_and_its_cutoff() {
        let args = compile("2026-10-30T12:00:00Z", &["rocm", "torch"]);
        assert_eq!(
            args,
            [
                "pip",
                "compile",
                "--quiet",
                "--no-header",
                "--config-file",
                "uv.toml",
                "--python-version",
                "3.14",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "p.in",
                "-o",
                "pylock.p-linux.toml",
                "--exclude-newer",
                "2026-10-30T12:00:00Z",
                "--exclude-newer-package",
                "rocm=false",
                "--exclude-newer-package",
                "torch=false",
            ]
        );
    }

    #[test]
    fn a_cutoff_is_the_cooldown_before_now_to_the_second() {
        assert_eq!(cutoff_before(now(), 7), "2026-10-30T12:00:00Z");
        assert_eq!(cutoff_before(now(), 0), "2026-11-06T12:00:00Z");
    }

    #[test]
    fn only_amd_only_projects_escape_the_cooldown() {
        let profiles = Profiles::parse(PROFILES).unwrap();
        let served: BTreeSet<String> = ["rocm", "rocm-sdk-core", "torch", "numpy", "setuptools"]
            .map(str::to_owned)
            .into();
        let exempt: Vec<String> = cooldown_exemptions(&profiles, served).into_iter().collect();
        assert_eq!(exempt, ["rocm", "rocm-sdk-core", "torch"]);
    }

    #[test]
    fn every_uv_variable_but_the_fetch_settings_is_cleared() {
        let keys = [
            "UV_INDEX",
            "UV_TORCH_BACKEND",
            "UV_CONSTRAINT",
            "UV_NO_CACHE",
            "UV_HTTP_TIMEOUT",
            "PATH",
            "HOME",
        ]
        .map(OsString::from);
        let mut cleared = uv_env_to_clear(keys.into_iter());
        cleared.sort();
        assert_eq!(cleared, ["UV_CONSTRAINT", "UV_INDEX", "UV_TORCH_BACKEND"]);
    }

    #[test]
    fn the_uv_version_is_read_from_its_version_line() {
        assert_eq!(
            uv_version("uv 0.13.6 (x86_64-unknown-linux-gnu)\n"),
            Some("0.13.6")
        );
        assert_eq!(uv_version("uv 0.13.6\n"), Some("0.13.6"));
        assert_eq!(uv_version("ruff 0.13.6\n"), None);
        assert_eq!(uv_version(""), None);
    }

    #[test]
    fn only_pypi_hosts_are_skipped_when_listing_indexes() {
        assert!(is_pypi("https://pypi.org/simple/"));
        assert!(is_pypi("https://test.pypi.org/simple/"));
        assert!(!is_pypi("https://stable.repo.amd.com/rocm/whl-next/"));
        assert!(!is_pypi("https://mirror.example/pypi.org/simple/"));
    }

    #[test]
    fn simple_index_roots_yield_normalised_project_names() {
        let html = r#"<html><body>
            <a href="amd-torch-device-gfx1151/">amd-torch-device-gfx1151</a><br/>
            <a href="Rocm_SDK_Core/">Rocm_SDK_Core</a>
            <a href="https://stable.repo.amd.com/rocm/whl-next/torch/">torch</a>
            <a href="../">..</a>
        </body></html>"#;
        let names: Vec<String> = simple_index_projects(html).into_iter().collect();
        assert_eq!(
            names,
            ["amd-torch-device-gfx1151", "rocm-sdk-core", "torch"]
        );
    }

    #[test]
    fn the_uv_command_runs_without_the_cleared_variables() {
        let command = uv_command(
            vec!["pip".into()],
            &["UV_INDEX".to_owned(), "UV_CONSTRAINT".to_owned()],
        );
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key.to_string_lossy().into_owned())
            .collect();
        // The GitHub credentials always, whatever the environment holds.
        assert_eq!(
            removed,
            ["GH_TOKEN", "GITHUB_TOKEN", "UV_CONSTRAINT", "UV_INDEX"]
        );
        assert_eq!(command.get_program(), "uv");
    }

    /// A uv that exits non-zero fails the run, so a lock it did not write is
    /// never finalised from whatever the output path held before.
    #[test]
    fn a_failed_uv_run_is_an_error() {
        let uv = SystemUv {
            work: PathBuf::from("unused"),
        };
        // Any program that exits non-zero on every OS the tests run on.
        let mut failing = Command::new("cargo");
        failing.arg("--no-such-flag-for-this-test");
        failing.stderr(std::process::Stdio::null());
        let err = uv.compile(failing).unwrap_err();
        assert!(format!("{err:#}").starts_with("uv exited with"), "{err:#}");
    }

    /// Records what the run asked uv to do, and writes a minimal lock where
    /// `uv pip compile -o <path>` would.
    struct FakeUv {
        version: &'static str,
        /// Per compile: (what the output held beforehand, env vars removed).
        calls: std::cell::RefCell<Vec<(Option<String>, Vec<String>)>>,
        /// Every compile's arguments.
        args: std::cell::RefCell<Vec<Vec<String>>>,
        /// What each compile writes.
        body: String,
    }

    impl FakeUv {
        fn new(version: &'static str) -> Self {
            Self {
                version,
                calls: std::cell::RefCell::default(),
                args: std::cell::RefCell::default(),
                body: "lock-version = \"1.0\"\ncreated-by = \"test\"\n".to_owned(),
            }
        }

        fn writing(body: String) -> Self {
            Self {
                body,
                ..Self::new("0.13.6")
            }
        }
    }

    impl Uv for FakeUv {
        fn version(&self) -> Result<String> {
            Ok(format!("uv {} (test)\n", self.version))
        }

        fn compile(&self, command: Command) -> Result<()> {
            let args: Vec<_> = command
                .get_args()
                .map(std::borrow::ToOwned::to_owned)
                .collect();
            let output = PathBuf::from(&args[args.iter().position(|a| a == "-o").unwrap() + 1]);
            let removed = command
                .get_envs()
                .filter(|(_, value)| value.is_none())
                .map(|(key, _)| key.to_string_lossy().into_owned())
                .collect();
            self.calls
                .borrow_mut()
                .push((fs::read_to_string(&output).ok(), removed));
            self.args.borrow_mut().push(
                args.iter()
                    .map(|a| a.to_string_lossy().into_owned())
                    .collect(),
            );
            fs::write(&output, &self.body)?;
            Ok(())
        }

        fn index_page(&self, url: &str) -> Result<String> {
            assert_eq!(
                url, "https://vendor.example/simple/",
                "only the vendor index is listed"
            );
            Ok(r#"<a href="rocm/">rocm</a><a href="numpy/">numpy</a><a href="rocm-sdk-core/">x</a>"#.to_owned())
        }
    }

    /// A workspace with one profile, locked for Linux, and uv pinned at 0.13.6.
    fn workspace() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let runtimes = root.path().join("runtimes");
        fs::create_dir_all(runtimes.join("locks")).unwrap();
        fs::write(
            runtimes.join("profiles.toml"),
            format!("{PROFILES}\n[profile.q]\npython = \"3.14\"\nrequirements = [\"rocm==10.1.0\"]\n\n[profile.q.platforms]\nlinux = \"x86_64-manylinux_2_28\"\n"),
        )
        .unwrap();
        fs::write(
            runtimes.join("uv.toml"),
            "[[index]]\nurl = \"https://vendor.example/simple/\"\n\n[[index]]\nurl = \"https://pypi.org/simple/\"\n",
        )
        .unwrap();
        fs::write(
            runtimes.join("mise.lock"),
            "lockfile_version = 3\n\n[[tools.\"aqua:astral-sh/uv\"]]\nversion = \"0.13.6\"\n",
        )
        .unwrap();
        root
    }

    fn plain() -> LockOptions {
        LockOptions {
            profile: None,
            upgrade: false,
            cooldown_days: None,
        }
    }

    #[test]
    fn a_uv_other_than_the_pinned_one_is_refused() {
        let root = workspace();
        let err = run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &FakeUv::new("0.13.5"),
            vec![],
            now(),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("pins uv 0.13.6"), "{err:#}");
    }

    #[test]
    fn a_run_writes_each_lock_with_its_header_and_drops_stale_ones() {
        let root = workspace();
        let locks = root.path().join("runtimes/locks");
        fs::write(locks.join("pylock.gone-linux.toml"), "").unwrap();
        run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &FakeUv::new("0.13.6"),
            vec![],
            now(),
        )
        .unwrap();
        let written = fs::read_to_string(locks.join("pylock.p-linux.toml")).unwrap();
        assert!(
            written.starts_with(&pylock::header("p", "linux")),
            "{written}"
        );
        assert!(!locks.join("pylock.gone-linux.toml").exists());
    }

    /// The old lock is never handed to uv. A re-lock resolves at the cutoff
    /// the lock records; `--upgrade`, or a lock that does not exist yet, takes
    /// a new cutoff the cooldown before now.
    #[test]
    fn a_relock_reuses_the_recorded_cutoff_and_upgrade_takes_a_new_one() {
        let root = workspace();
        let committed = root.path().join("runtimes/locks/pylock.p-linux.toml");
        fs::write(
            &committed,
            "# header\n# exclude-newer: 2026-01-01T00:00:00Z\nlock-version = \"1.0\"\n",
        )
        .unwrap();
        let uv = FakeUv::new("0.13.6");
        let cooled = LockOptions {
            cooldown_days: Some(7),
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &cooled,
            &uv,
            vec![],
            now(),
        )
        .unwrap();
        // The re-lock records the cutoff it actually resolved at.
        assert_eq!(
            pylock::recorded_cutoff(&fs::read_to_string(&committed).unwrap()).as_deref(),
            Some("2026-01-01T00:00:00Z")
        );
        let upgrade = LockOptions {
            upgrade: true,
            ..cooled
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &upgrade,
            &uv,
            vec![],
            now(),
        )
        .unwrap();

        let cutoffs: Vec<String> = uv
            .args
            .borrow()
            .iter()
            .map(|args| args[args.iter().position(|a| a == "--exclude-newer").unwrap() + 1].clone())
            .collect();
        // Profiles run in order p, q; only p has a committed lock.
        assert_eq!(
            cutoffs,
            [
                "2026-01-01T00:00:00Z",
                "2026-10-30T12:00:00Z",
                "2026-10-30T12:00:00Z",
                "2026-10-30T12:00:00Z",
            ]
        );
        assert!(
            uv.calls.borrow().iter().all(|(before, _)| before.is_none()),
            "uv must never be given a previous lock as preferences"
        );
        // The upgrade resolved at the new cutoff, but resolved the same lock
        // (the fake uv writes the same body every time), so the lock keeps
        // its old cutoff: `an_upgrade_that_resolves_nothing_new_rewrites_no_lock`
        // covers the case where the resolution does change.
        assert_eq!(
            pylock::recorded_cutoff(&fs::read_to_string(&committed).unwrap()).as_deref(),
            Some("2026-01-01T00:00:00Z")
        );
    }

    #[test]
    fn each_target_is_resolved_for_its_own_platform() {
        let root = workspace();
        let profiles = root.path().join("runtimes/profiles.toml");
        let text = fs::read_to_string(&profiles).unwrap().replace(
            "[profile.q]\npython = \"3.14\"",
            "[profile.q]\npython = \"3.13\"",
        ) + "windows = \"x86_64-pc-windows-msvc\"\n";
        fs::write(&profiles, text).unwrap();
        let uv = FakeUv::new("0.13.6");
        let options = LockOptions {
            profile: Some("q".to_owned()),
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &uv,
            vec![],
            now(),
        )
        .unwrap();
        let value_after = |args: &[String], flag: &str| {
            args[args.iter().position(|a| a == flag).unwrap() + 1].clone()
        };
        let seen: Vec<(String, String)> = uv
            .args
            .borrow()
            .iter()
            .map(|args| {
                (
                    value_after(args, "--python-platform"),
                    value_after(args, "--python-version"),
                )
            })
            .collect();
        assert_eq!(
            seen,
            [
                ("x86_64-manylinux_2_28".to_owned(), "3.13".to_owned()),
                ("x86_64-pc-windows-msvc".to_owned(), "3.13".to_owned()),
            ]
        );
        assert!(
            root.path()
                .join("runtimes/locks/pylock.q-windows.toml")
                .is_file()
        );
    }

    /// A committed lock whose cutoff is missing or unreadable is resolved at
    /// a new cutoff, the cooldown before now, never at an empty one.
    #[test]
    fn a_lock_without_a_valid_cutoff_is_resolved_at_a_new_one() {
        let root = workspace();
        fs::write(
            root.path().join("runtimes/locks/pylock.p-linux.toml"),
            "# exclude-newer: last week\nlock-version = \"1.0\"\n",
        )
        .unwrap();
        let uv = FakeUv::new("0.13.6");
        let options = LockOptions {
            profile: Some("p".to_owned()),
            cooldown_days: Some(7),
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &uv,
            vec![],
            now(),
        )
        .unwrap();
        let args = &uv.args.borrow()[0];
        let cutoff = &args[args.iter().position(|a| a == "--exclude-newer").unwrap() + 1];
        assert_eq!(cutoff, "2026-10-30T12:00:00Z");
    }

    /// A stray symlink in the lock directory is removed, never followed: the
    /// directory it points at, and what is in it, stay.
    #[cfg(unix)]
    #[test]
    fn a_stray_symlink_is_removed_and_not_followed() {
        let root = workspace();
        let outside = root.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("keep"), "x").unwrap();
        let link = root.path().join("runtimes/locks/linked");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &FakeUv::new("0.13.6"),
            vec![],
            now(),
        )
        .unwrap();
        assert!(
            fs::symlink_metadata(&link).is_err(),
            "the symlink is removed"
        );
        assert!(outside.join("keep").is_file(), "its target is untouched");
    }

    /// CI's first run: the lock directory does not exist yet.
    #[test]
    fn the_first_run_creates_the_lock_directory() {
        let root = workspace();
        let locks = root.path().join("runtimes/locks");
        fs::remove_dir(&locks).unwrap();
        run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &FakeUv::new("0.13.6"),
            vec![],
            now(),
        )
        .unwrap();
        assert!(locks.join("pylock.p-linux.toml").is_file());
        assert!(locks.join("pylock.q-linux.toml").is_file());
    }

    #[test]
    fn the_run_clears_inherited_uv_variables_from_the_uv_it_starts() {
        let root = workspace();
        let uv = FakeUv::new("0.13.6");
        let env = ["UV_INDEX", "UV_NO_CACHE", "HOME"]
            .map(OsString::from)
            .to_vec();
        run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &uv,
            env,
            now(),
        )
        .unwrap();
        assert_eq!(
            uv.calls.borrow()[0].1,
            ["GH_TOKEN", "GITHUB_TOKEN", "UV_INDEX"]
        );
    }

    #[test]
    fn a_cooldown_exempts_only_the_amd_only_projects_the_indexes_serve() {
        let root = workspace();
        let uv = FakeUv::new("0.13.6");
        let options = LockOptions {
            cooldown_days: Some(7),
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &uv,
            vec![],
            now(),
        )
        .unwrap();
        // Both profiles were compiled, so the loop below checks something.
        assert_eq!(uv.args.borrow().len(), 2);
        for args in uv.args.borrow().iter() {
            assert!(args.contains(&"rocm=false".to_owned()), "{args:?}");
            assert!(args.contains(&"rocm-sdk-core=false".to_owned()), "{args:?}");
            assert!(!args.contains(&"numpy=false".to_owned()), "{args:?}");
        }
    }

    #[test]
    fn profile_locks_only_that_profile_and_keeps_the_other_locks() {
        let root = workspace();
        let locks = root.path().join("runtimes/locks");
        fs::write(locks.join("pylock.q-linux.toml"), "kept").unwrap();
        let uv = FakeUv::new("0.13.6");
        let options = LockOptions {
            profile: Some("p".to_owned()),
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &uv,
            vec![],
            now(),
        )
        .unwrap();
        assert_eq!(uv.args.borrow().len(), 1, "only profile p is compiled");
        assert!(locks.join("pylock.p-linux.toml").exists());
        assert_eq!(
            fs::read_to_string(locks.join("pylock.q-linux.toml")).unwrap(),
            "kept"
        );
    }

    #[test]
    fn an_unknown_profile_is_refused() {
        let root = workspace();
        let options = LockOptions {
            profile: Some("nope".to_owned()),
            ..plain()
        };
        let err = run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &FakeUv::new("0.13.6"),
            vec![],
            now(),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("no profile named"), "{err:#}");
    }

    #[test]
    fn a_failure_on_a_later_target_leaves_every_lock_untouched() {
        struct FailsOnQ(FakeUv);
        impl Uv for FailsOnQ {
            fn version(&self) -> Result<String> {
                self.0.version()
            }
            fn compile(&self, command: Command) -> Result<()> {
                let is_q = command
                    .get_args()
                    .any(|a| a.to_string_lossy().ends_with("q-linux.toml"));
                ensure!(!is_q, "resolution failed");
                self.0.compile(command)
            }
            fn index_page(&self, url: &str) -> Result<String> {
                self.0.index_page(url)
            }
        }
        let root = workspace();
        let p_lock = root.path().join("runtimes/locks/pylock.p-linux.toml");
        fs::write(&p_lock, "old").unwrap();
        let err = run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &FailsOnQ(FakeUv::new("0.13.6")),
            vec![],
            now(),
        );
        assert!(err.is_err());
        assert_eq!(
            fs::read_to_string(&p_lock).unwrap(),
            "old",
            "p must not be rewritten when q fails"
        );
    }

    #[test]
    fn an_unreadable_uv_version_is_named_as_such() {
        struct Garbled;
        impl Uv for Garbled {
            fn version(&self) -> Result<String> {
                Ok("something else\n".to_owned())
            }
            fn compile(&self, _: Command) -> Result<()> {
                unreachable!()
            }
            fn index_page(&self, _: &str) -> Result<String> {
                unreachable!()
            }
        }
        let root = workspace();
        let err = run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &Garbled,
            vec![],
            now(),
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").starts_with("a uv whose version cannot be read"),
            "{err:#}"
        );
    }

    /// The lock directory is the tool's: `deps lock` removes every entry no
    /// profile produces — files, dotfiles and directories — also under
    /// `--profile`, and leaves nothing for `deps check`'s stray-entry rule.
    #[test]
    fn a_partial_run_removes_every_file_no_profile_produces() {
        let root = workspace();
        let locks = root.path().join("runtimes/locks");
        for file in [
            "pylock.gone-linux.toml",
            "README.md",
            "notes.toml",
            "pylock.q-linux.toml.partial",
            ".keep",
            "pylock.q-linux.toml",
        ] {
            fs::write(locks.join(file), "x").unwrap();
        }
        fs::create_dir_all(locks.join("old/nested")).unwrap();
        fs::write(locks.join("old/nested/pylock.x.toml"), "x").unwrap();
        let options = LockOptions {
            profile: Some("p".to_owned()),
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &FakeUv::new("0.13.6"),
            vec![],
            now(),
        )
        .unwrap();
        for removed in [
            "pylock.gone-linux.toml",
            "README.md",
            "notes.toml",
            "pylock.q-linux.toml.partial",
            ".keep",
            "old",
        ] {
            assert!(
                !locks.join(removed).exists(),
                "{removed} is not a lock any profile produces"
            );
        }
        assert_eq!(
            fs::read_to_string(locks.join("pylock.q-linux.toml")).unwrap(),
            "x",
            "q's lock is kept"
        );
        let profiles = Profiles::load(root.path()).unwrap();
        assert_eq!(
            profiles.stray_lock_entries(root.path()).unwrap(),
            Vec::<PathBuf>::new(),
            "deps check finds nothing left to report"
        );
    }

    const SUM: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn torch_lock(host: &str) -> String {
        format!(
            "lock-version = \"1.0\"\ncreated-by = \"uv\"\n\n[[packages]]\nname = \"torch\"\nversion = \"2.13.0\"\n\
             wheels = [{{ url = \"https://{host}/torch-2.13.0-cp314-cp314-linux_x86_64.whl\", hashes = {{ sha256 = \"{SUM}\" }} }}]\n"
        )
    }

    /// Re-lock the one profile with uv producing `body`, and report what
    /// `deps check` says about the result.
    fn relock_then_check(body: String) -> Vec<String> {
        let root = workspace();
        let options = LockOptions {
            profile: Some("p".to_owned()),
            upgrade: true,
            ..plain()
        };
        run_with(
            root.path(),
            &root.path().join("work"),
            &options,
            &FakeUv::writing(body),
            vec![],
            now(),
        )
        .unwrap();
        let profiles = Profiles::load(root.path()).unwrap();
        let text = fs::read_to_string(root.path().join(Profiles::lock_path("p", "linux"))).unwrap();
        let uv_config = fs::read_to_string(root.path().join("runtimes/uv.toml")).unwrap();
        super::super::check::lock_problems(
            &profiles,
            "p",
            &profiles.profile["p"],
            "linux",
            Some(&uv_config),
            &text,
        )
    }

    /// The two halves of `deps check`'s remediation: re-locking clears a
    /// lock that was stale, and does not clear what the resolver itself
    /// produces from the same inputs, which needs the profile changed.
    /// The weekly update runs `--upgrade`, which takes a new cutoff every
    /// time. When nothing new resolves, every lock must stay byte for byte
    /// as it was, or the update would propose a pull request every week.
    #[test]
    fn an_upgrade_that_resolves_nothing_new_rewrites_no_lock() {
        let root = workspace();
        let only_p = LockOptions {
            profile: Some("p".to_owned()),
            cooldown_days: Some(7),
            ..plain()
        };
        let upgrade = LockOptions {
            profile: Some("p".to_owned()),
            upgrade: true,
            cooldown_days: Some(7),
        };
        let first = FakeUv::writing(torch_lock("vendor.example"));
        run_with(
            root.path(),
            &root.path().join("work"),
            &only_p,
            &first,
            vec![],
            now(),
        )
        .unwrap();
        let path = root.path().join(Profiles::lock_path("p", "linux"));
        let before = fs::read_to_string(&path).unwrap();
        assert_eq!(
            pylock::recorded_cutoff(&before).as_deref(),
            Some("2026-10-30T12:00:00Z")
        );

        let later = now() + chrono::Duration::days(7);
        let same = FakeUv::writing(torch_lock("vendor.example"));
        run_with(
            root.path(),
            &root.path().join("work"),
            &upgrade,
            &same,
            vec![],
            later,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            before,
            "unchanged lock rewritten"
        );

        // A resolution that does change records the new cutoff.
        let moved = FakeUv::writing(torch_lock("vendor.example").replace("2.13.0", "2.13.1"));
        run_with(
            root.path(),
            &root.path().join("work"),
            &upgrade,
            &moved,
            vec![],
            later,
        )
        .unwrap();
        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("2.13.1"), "{after}");
        assert_eq!(
            pylock::recorded_cutoff(&after).as_deref(),
            Some("2026-11-06T12:00:00Z")
        );
    }

    #[test]
    fn relocking_clears_a_stale_lock_but_not_a_policy_problem() {
        assert_eq!(
            relock_then_check(torch_lock("vendor.example")),
            Vec::<String>::new()
        );
        let still = relock_then_check(torch_lock("files.pythonhosted.org"));
        assert!(
            still
                .iter()
                .any(|p| p.contains("AMD-only package taken from")),
            "{still:?}"
        );
    }

    #[test]
    fn a_failed_uv_version_run_is_an_error_not_a_version() {
        let err = version_output(false, b"uv 0.13.6\n", b"boom").unwrap_err();
        assert!(format!("{err:#}").contains("failed: boom"), "{err:#}");
        assert_eq!(
            version_output(true, b"uv 0.13.6\n", b"").unwrap(),
            "uv 0.13.6\n"
        );
    }

    #[test]
    fn a_lock_that_cannot_be_written_leaves_no_partial_file() {
        let root = workspace();
        let locks = root.path().join("runtimes/locks");
        // A non-empty directory where the lock goes makes the rename fail.
        fs::create_dir_all(locks.join("pylock.p-linux.toml")).unwrap();
        fs::write(locks.join("pylock.p-linux.toml/keep"), "").unwrap();
        let err = run_with(
            root.path(),
            &root.path().join("work"),
            &plain(),
            &FakeUv::new("0.13.6"),
            vec![],
            now(),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("failed to write"), "{err:#}");
        assert!(
            !locks.join("pylock.p-linux.toml.partial").exists(),
            "the partial file is cleaned up"
        );
    }

    /// A rename that fails part-way names the lock it stopped at, leaves the
    /// locks renamed before it replaced and the ones after it untouched, and
    /// leaves no partial file of any lock behind.
    #[test]
    fn a_failed_rename_names_its_lock_and_leaves_no_partial_file() {
        let names = ["pylock.p-linux.toml", "pylock.q-linux.toml"];
        for (failing, blocked) in names.iter().enumerate() {
            let root = workspace();
            let locks = root.path().join("runtimes/locks");
            for name in names.iter().filter(|name| *name != blocked) {
                fs::write(locks.join(name), "the committed lock\n").unwrap();
            }
            // A non-empty directory where this lock goes makes its rename
            // fail; its partial file is written first, so the write phase
            // passes.
            fs::create_dir_all(locks.join(blocked)).unwrap();
            fs::write(locks.join(blocked).join("keep"), "").unwrap();
            let err = run_with(
                root.path(),
                &root.path().join("work"),
                &plain(),
                &FakeUv::new("0.13.6"),
                vec![],
                now(),
            )
            .unwrap_err();
            let message = format!("{err:#}");
            assert!(
                message.contains("failed to write") && message.contains(blocked),
                "{blocked}: {message}"
            );
            for (index, name) in names.iter().enumerate() {
                assert!(
                    !locks.join(format!("{name}.partial")).exists(),
                    "{name}.partial is cleaned up with {blocked} blocked"
                );
                if index == failing {
                    continue;
                }
                let replaced =
                    fs::read_to_string(locks.join(name)).unwrap() != "the committed lock\n";
                assert_eq!(replaced, index < failing, "{name} with {blocked} blocked");
            }
        }
    }

    /// A lock that cannot be written, whichever target it is, replaces no
    /// lock: every new lock is written before any committed one is replaced.
    #[test]
    fn a_failed_write_of_any_lock_replaces_none() {
        for blocked in ["pylock.p-linux.toml", "pylock.q-linux.toml"] {
            let root = workspace();
            let locks = root.path().join("runtimes/locks");
            for name in ["pylock.p-linux.toml", "pylock.q-linux.toml"] {
                fs::write(locks.join(name), "the committed lock\n").unwrap();
            }
            // A non-empty directory where this target's partial file goes
            // makes its write fail.
            let partial = locks.join(format!("{blocked}.partial"));
            fs::create_dir_all(&partial).unwrap();
            fs::write(partial.join("keep"), "").unwrap();
            let err = run_with(
                root.path(),
                &root.path().join("work"),
                &plain(),
                &FakeUv::new("0.13.6"),
                vec![],
                now(),
            )
            .unwrap_err();
            assert!(format!("{err:#}").contains("failed to write"), "{err:#}");
            for name in ["pylock.p-linux.toml", "pylock.q-linux.toml"] {
                assert_eq!(
                    fs::read_to_string(locks.join(name)).unwrap(),
                    "the committed lock\n",
                    "{name} with {blocked} blocked"
                );
                let other = locks.join(format!("{name}.partial"));
                assert!(
                    name == blocked || !other.exists(),
                    "{name}.partial is cleaned up with {blocked} blocked"
                );
            }
        }
    }
}
