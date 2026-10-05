// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for state `rocm` persists under the config/data/cache dirs:
//! the runtime registry, the startup update-check cache, the active-runtime
//! marker, and how a config writer in this module treats a `config.json` it
//! cannot parse.
//!
//! A child of `therock` so it can drive the real private load/save functions.
//! No test here reads or writes the process environment: every `AppPaths` is
//! built by hand under the system temp dir.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config as RunnerConfig, TestCaseError, TestError, TestRunner};
use rocm_core::{AppPaths, RocmCliConfig};

use super::{
    InstalledRuntimeManifest, StartupUpdateCheckRecord, load_runtime_manifests,
    load_startup_update_check, record_managed_python_config, save_runtime_manifest,
    save_startup_update_check,
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn fresh_paths(tag: &str) -> (PathBuf, AppPaths) {
    let root = std::env::temp_dir().join(format!(
        "rocm-props-{tag}-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    let paths = AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
    };
    (root, paths)
}

fn report<T: std::fmt::Debug>(name: &str, result: Result<(), TestError<T>>) {
    match result {
        Ok(()) => {}
        Err(TestError::Fail(reason, value)) => {
            panic!("{name}: minimal failing input {value:#?}\nreason: {reason}")
        }
        Err(TestError::Abort(reason)) => panic!("{name}: aborted: {reason}"),
    }
}

fn text() -> BoxedStrategy<String> {
    prop_oneof![
        2 => Just(String::new()),
        2 => "[a-z0-9:._-]{1,12}",
        1 => Just("gfx1151-é-日本-\u{1F600}".to_owned()),
        2 => any::<String>(),
    ]
    .boxed()
}

fn opt_text() -> BoxedStrategy<Option<String>> {
    prop_oneof![3 => Just(None), 2 => text().prop_map(Some)].boxed()
}

fn u128_edge() -> BoxedStrategy<u128> {
    prop_oneof![
        Just(0_u128),
        Just(u128::from(u64::MAX)),
        Just(u128::from(u64::MAX) + 1),
        Just(u128::MAX),
        any::<u128>(),
        (1_600_000_000_000_u128..2_000_000_000_000),
    ]
    .boxed()
}

// ---------------------------------------------------------------------------
// config.json: a corrupt file must not be silently replaced by defaults.
// ---------------------------------------------------------------------------

/// The user settings a later command acts on. If any of these vanish, `rocm`
/// forgets which runtime is active, where TheRock lives, and its defaults.
#[derive(Debug, Clone)]
struct UserSettings {
    active_runtime_key: String,
    default_runtime_id: String,
    default_engine: String,
    therock_venv: PathBuf,
}

fn user_settings() -> BoxedStrategy<UserSettings> {
    (
        "[a-z0-9-]{1,16}",
        "[a-z0-9:-]{1,16}",
        prop_oneof![Just("vllm".to_owned()), Just("lemonade".to_owned())],
        "/[a-z/]{1,16}",
    )
        .prop_map(|(key, id, engine, venv)| UserSettings {
            active_runtime_key: key,
            default_runtime_id: id,
            default_engine: engine,
            therock_venv: PathBuf::from(venv),
        })
        .boxed()
}

/// Ways a real `config.json` becomes unreadable to this binary. Each is applied
/// to the bytes `RocmCliConfig::save` actually wrote, not to JSON built from
/// scratch.
#[derive(Debug, Clone)]
enum Corruption {
    /// Interrupted write, as a plain `fs::write` in `RocmCliConfig::save` used to
    /// leave behind (it is atomic now), or a truncating hand edit.
    TruncateAt(usize),
    /// What a non-finite tick serializes to (see the rocm-core property test).
    NullTick,
    /// A hand edit or a newer binary that changed a field's type.
    RetypeOnboardingDismissed,
    /// A value the loader's own validators reject.
    TopPOutOfRange,
}

fn corruption() -> BoxedStrategy<Corruption> {
    prop_oneof![
        (1_usize..400).prop_map(Corruption::TruncateAt),
        Just(Corruption::NullTick),
        Just(Corruption::RetypeOnboardingDismissed),
        Just(Corruption::TopPOutOfRange),
    ]
    .boxed()
}

fn corrupt(bytes: &[u8], how: &Corruption) -> Vec<u8> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    match how {
        Corruption::TruncateAt(at) => return bytes[..(*at).min(bytes.len() - 1)].to_vec(),
        Corruption::NullTick => {
            value["dashboard"]["daemon"]["gpu_tick_secs"] = serde_json::Value::Null;
        }
        Corruption::RetypeOnboardingDismissed => {
            value["onboarding_dismissed"] = serde_json::json!("yes");
        }
        Corruption::TopPOutOfRange => {
            value["dashboard"]["tui"]["chat_top_p"] = serde_json::json!(1.5);
        }
    }
    serde_json::to_vec_pretty(&value).unwrap()
}

/// `RocmCliConfig::load` deliberately fails loudly on a corrupt file. A writer
/// that meets the same file must not "repair" it by saving defaults over it:
/// that turns "unreadable, fix it" into "valid but empty", and every later
/// command acts on the empty state.
#[test]
fn managed_python_recording_never_replaces_an_unreadable_config_with_defaults() {
    let total = AtomicUsize::new(0);
    let unreadable = AtomicUsize::new(0);
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 128,
        ..RunnerConfig::default()
    });
    let result = runner.run(&(user_settings(), corruption()), |(settings, how)| {
        total.fetch_add(1, Ordering::Relaxed);
        let (root, paths) = fresh_paths("config-wipe");
        let mut config = RocmCliConfig {
            active_runtime_key: Some(settings.active_runtime_key.clone()),
            default_runtime_id: Some(settings.default_runtime_id.clone()),
            default_engine: Some(settings.default_engine.clone()),
            ..RocmCliConfig::default()
        };
        config.setup.therock_venv = Some(settings.therock_venv.clone());
        config.save(&paths).unwrap();
        let original = std::fs::read(paths.config_path()).unwrap();
        let damaged = corrupt(&original, &how);
        std::fs::write(paths.config_path(), &damaged).unwrap();
        if RocmCliConfig::load(&paths).is_ok() {
            // A truncation that happens to land on valid JSON; not this property.
            let _ = std::fs::remove_dir_all(root);
            return Ok(());
        }
        unreadable.fetch_add(1, Ordering::Relaxed);

        let _ = record_managed_python_config(&paths, Path::new("/usr/bin/python3"));

        let after = std::fs::read(paths.config_path()).unwrap();
        let _ = std::fs::remove_dir_all(root);
        if after == damaged {
            return Ok(()); // left alone: the user still sees the loud error
        }
        let reloaded: RocmCliConfig = serde_json::from_slice(&after)
            .map_err(|e| TestCaseError::fail(format!("rewritten config unreadable: {e}")))?;
        prop_assert_eq!(
            reloaded.active_runtime_key.as_deref(),
            Some(settings.active_runtime_key.as_str()),
            "config.json was rewritten and the active runtime was forgotten"
        );
        prop_assert_eq!(
            reloaded.setup.therock_venv.as_deref(),
            Some(settings.therock_venv.as_path())
        );
        Ok(())
    });
    eprintln!(
        "reach: {} of {} cases produced a config.json that load() rejects",
        unreadable.load(Ordering::Relaxed),
        total.load(Ordering::Relaxed)
    );
    report("config wipe", result);
}

/// The property above stops at its first (shrunk) counterexample. This walks
/// every corruption class once, and asserts what the user is told together
/// with what is left on disk: the error says the file was left unchanged, and
/// it was, byte for byte.
#[test]
fn no_corruption_class_is_wiped_by_managed_python_recording() {
    let classes = [
        Corruption::TruncateAt(40),
        Corruption::NullTick,
        Corruption::RetypeOnboardingDismissed,
        Corruption::TopPOutOfRange,
    ];
    for how in classes {
        let (root, paths) = fresh_paths("config-wipe-class");
        let mut config = RocmCliConfig {
            active_runtime_key: Some("therock-release-gfx1151-7.13.0".to_owned()),
            ..RocmCliConfig::default()
        };
        config.setup.therock_venv = Some(PathBuf::from("/data/rocm"));
        config.save(&paths).unwrap();
        let damaged = corrupt(&std::fs::read(paths.config_path()).unwrap(), &how);
        std::fs::write(paths.config_path(), &damaged).unwrap();
        assert!(
            RocmCliConfig::load(&paths).is_err(),
            "{how:?} is unreadable"
        );

        let error = record_managed_python_config(&paths, Path::new("/usr/bin/python3"))
            .expect_err("recording over an unreadable config must fail");
        let after = std::fs::read(paths.config_path()).unwrap();
        let _ = std::fs::remove_dir_all(&root);

        let message = format!("{error:#}");
        assert!(
            message.contains("cannot record the managed Python")
                && message.contains("it was left unchanged")
                && message.contains("config.json"),
            "{how:?}: unexpected error: {message}"
        );
        assert_eq!(after, damaged, "{how:?}: config.json was rewritten");
    }
}

/// With no `config.json` at all, recording the managed Python is the first
/// write and must still succeed: "missing" is defaults, as `load` promises.
#[test]
fn managed_python_recording_creates_a_missing_config() {
    let (root, paths) = fresh_paths("config-missing");
    record_managed_python_config(&paths, Path::new("/usr/bin/python3")).unwrap();
    let loaded = RocmCliConfig::load(&paths).unwrap();
    let _ = std::fs::remove_dir_all(&root);
    let tool = loaded.tools.get("python").expect("python was recorded");
    assert!(tool.managed);
    assert_eq!(tool.path.as_deref(), Some(Path::new("/usr/bin/python3")));
}

/// The end-to-end path a user could reach without hand-editing `config.json`:
/// a legacy rocm-dash `config.toml` holding `inf`. It used to migrate into a
/// config the loader rejects, which the next managed-Python step then replaced
/// with defaults. Now the migration refuses the file, so nothing unreadable is
/// written, and the managed-Python step meets a missing config, which it may
/// create.
#[test]
fn legacy_inf_tick_never_reaches_config_json() {
    let (root, paths) = fresh_paths("legacy-chain");
    std::fs::create_dir_all(&root).unwrap();
    let legacy = root.join("config.toml");
    let text = "default_engine = \"vllm\"\n[daemon]\ngpu_tick = inf\n[tui]\ntheme = \"nord\"\n";
    std::fs::write(&legacy, text).unwrap();

    let error = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)
        .expect_err("a legacy file with an infinite tick must not migrate");
    assert!(
        format!("{error:#}").contains("daemon.gpu_tick = inf"),
        "unexpected error: {error:#}"
    );
    assert!(!paths.config_path().exists(), "config.json was written");

    record_managed_python_config(&paths, Path::new("/usr/bin/python3")).unwrap();
    let after = RocmCliConfig::load(&paths);
    let legacy_after = std::fs::read_to_string(&legacy).unwrap();
    let _ = std::fs::remove_dir_all(&root);
    assert!(after.is_ok(), "config.json is readable: {:?}", after.err());
    assert_eq!(legacy_after, text, "the legacy TOML was modified");
}

/// rocm-dash's reader and the migration into `config.json` read the same
/// legacy file. They must agree on whether it is readable: a file the
/// dashboard rejects must not be migrated, and one it accepts must migrate
/// into a `config.json` that loads.
#[test]
fn legacy_tick_readers_agree_on_which_files_are_readable() {
    let ticks = [
        "0",
        "0.5",
        "120",
        "nan",
        "inf",
        "-inf",
        "-1.0",
        "-0.0",
        "1.7976931348623157e308",
    ];
    let mut disagreements = Vec::new();
    for tick in ticks {
        let (root, paths) = fresh_paths("legacy-agree");
        std::fs::create_dir_all(&root).unwrap();
        let legacy = root.join("config.toml");
        // rocm-dash requires the whole `[daemon]` table, so every other key
        // holds a value both readers accept; only `instance_tick` varies.
        std::fs::write(
            &legacy,
            format!(
                "[daemon]\nlisten = \"unix:/tmp/rocm-dash.sock\"\ngpu_tick = 1.0\n\
                 discovery_tick = 5.0\ninstance_tick = {tick}\n"
            ),
        )
        .unwrap();
        let dash_accepts = rocm_dash_core::config::Config::load(&legacy).is_ok();
        let migrated = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy);
        let migration_accepts = matches!(migrated, Ok(Some(_)));
        let loadable = RocmCliConfig::load(&paths).is_ok();
        let _ = std::fs::remove_dir_all(&root);
        if dash_accepts != migration_accepts || !loadable {
            disagreements.push(format!(
                "{tick}: dashboard accepts={dash_accepts}, migration accepts={migration_accepts}, \
                 config.json loadable={loadable}"
            ));
        }
    }
    assert!(disagreements.is_empty(), "{disagreements:#?}");
}

// ---------------------------------------------------------------------------
// Runtime registry: save_runtime_manifest / load_runtime_manifests.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum RootName {
    Ascii(String),
    Unicode,
    Spaced,
    #[cfg(unix)]
    NonUtf8,
}

fn root_name() -> BoxedStrategy<RootName> {
    // Only unix adds the non-UTF-8 option below.
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut options = vec![
        "[a-z0-9_.-]{1,12}".prop_map(RootName::Ascii).boxed(),
        Just(RootName::Unicode).boxed(),
        Just(RootName::Spaced).boxed(),
    ];
    #[cfg(unix)]
    options.push(Just(RootName::NonUtf8).boxed());
    proptest::strategy::Union::new(options).boxed()
}

fn install_root_under(base: &Path, name: &RootName) -> PathBuf {
    match name {
        RootName::Ascii(name) if name != "." && name != ".." => base.join(name),
        RootName::Ascii(_) => base.join("dot"),
        RootName::Unicode => base.join("ROCm-é-日本"),
        // Windows strips a trailing space from a folder name when it creates
        // it, so a root that ends in one cannot round-trip there; keep the
        // inner and leading spaces, which it does store. Tracked separately:
        // the stored install_root and pip_cache_dir then disagree.
        #[cfg(windows)]
        RootName::Spaced => base.join(" ROCm venvs"),
        #[cfg(not(windows))]
        RootName::Spaced => base.join(" ROCm venvs "),
        #[cfg(unix)]
        RootName::NonUtf8 => {
            use std::os::unix::ffi::OsStrExt;
            base.join(std::ffi::OsStr::from_bytes(b"rocm-\xff\xfe-root"))
        }
    }
}

#[derive(Debug, Clone)]
struct ManifestSeed {
    runtime_key: String,
    strings: [String; 6],
    optionals: [Option<String>; 6],
    root: RootName,
    read_only: bool,
    devel: bool,
    installed_at_unix_ms: u128,
}

fn manifest_seed() -> BoxedStrategy<ManifestSeed> {
    (
        "therock-[a-z]{1,8}-[a-z0-9]{1,8}-[0-9.]{1,8}",
        [text(), text(), text(), text(), text(), text()],
        [
            opt_text(),
            opt_text(),
            opt_text(),
            opt_text(),
            opt_text(),
            opt_text(),
        ],
        root_name(),
        any::<bool>(),
        any::<bool>(),
        u128_edge(),
    )
        .prop_map(
            |(runtime_key, strings, optionals, root, read_only, devel, at)| ManifestSeed {
                runtime_key,
                strings,
                optionals,
                root,
                read_only,
                devel,
                installed_at_unix_ms: at,
            },
        )
        .boxed()
}

fn build_manifest(seed: &ManifestSeed, install_root: PathBuf) -> InstalledRuntimeManifest {
    let [runtime_id, channel, format, family, family_source, version] = seed.strings.clone();
    let [layout, index_url, tarball, launcher, executable, sdk_torch] = seed.optionals.clone();
    InstalledRuntimeManifest {
        runtime_key: seed.runtime_key.clone(),
        runtime_id,
        channel,
        format,
        family,
        family_source,
        version,
        pip_cache_dir: Some(install_root.join("pip-cache")),
        imported_from: None,
        install_root,
        selected_artifact_url: "https://example.invalid/a.whl".to_owned(),
        source_layout_generation: layout,
        index_url,
        tarball_file_name: tarball,
        python_launcher: launcher,
        python_executable: executable,
        rocm_sdk: None,
        sdk_torch,
        wheel_composition: None,
        read_only: seed.read_only,
        devel: seed.devel,
        installed_at_unix_ms: seed.installed_at_unix_ms,
    }
}

fn manifest_text(manifest: &InstalledRuntimeManifest) -> String {
    serde_json::to_string(manifest).expect("manifest serializes")
}

/// Every manifest `save_runtime_manifest` accepts is listed back unchanged by
/// `load_runtime_manifests`. And a save that fails must not leave a registry
/// record behind that disagrees with the folder it names.
#[test]
#[cfg_attr(
    unix,
    ignore = "CONFIRMED BUG (Linux): a non-UTF-8 install_root is lossily rewritten before it is persisted"
)]
fn runtime_manifest_round_trips_through_the_registry() {
    let total = AtomicUsize::new(0);
    let non_utf8 = AtomicUsize::new(0);
    let empty = AtomicUsize::new(0);
    let none = AtomicUsize::new(0);
    let big_ts = AtomicUsize::new(0);
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 256,
        ..RunnerConfig::default()
    });
    let result = runner.run(&manifest_seed(), |seed| {
        total.fetch_add(1, Ordering::Relaxed);
        #[cfg(unix)]
        if matches!(seed.root, RootName::NonUtf8) {
            non_utf8.fetch_add(1, Ordering::Relaxed);
        }
        if seed.strings.iter().any(String::is_empty) {
            empty.fetch_add(1, Ordering::Relaxed);
        }
        if seed.optionals.iter().any(Option::is_none) {
            none.fetch_add(1, Ordering::Relaxed);
        }
        if seed.installed_at_unix_ms > u128::from(u64::MAX) {
            big_ts.fetch_add(1, Ordering::Relaxed);
        }
        let (root, paths) = fresh_paths("manifest");
        let install_root = install_root_under(&root.join("installs"), &seed.root);
        std::fs::create_dir_all(&install_root).unwrap();
        let manifest = build_manifest(&seed, install_root.clone());
        let saved = save_runtime_manifest(&paths, &manifest);
        let listed = load_runtime_manifests(&paths);
        let _ = std::fs::remove_dir_all(&root);
        let listed = listed.map_err(|e| TestCaseError::fail(format!("{e:#}")))?;
        match saved {
            Ok(()) => {
                prop_assert_eq!(listed.len(), 1, "saved manifest was not listed back");
                prop_assert_eq!(manifest_text(&listed[0]), manifest_text(&manifest));
            }
            Err(error) => {
                // Refusing is fine, but nothing half-written may be listed.
                prop_assert!(
                    listed.is_empty(),
                    "save failed ({error:#}) yet the registry lists {:?} for a folder at {:?}",
                    listed[0].install_root,
                    install_root
                );
            }
        }
        Ok(())
    });
    eprintln!(
        "reach: of {} cases, non-UTF-8 root {}, an empty string {}, a None optional {}, \
         timestamp above u64::MAX {}",
        total.load(Ordering::Relaxed),
        non_utf8.load(Ordering::Relaxed),
        empty.load(Ordering::Relaxed),
        none.load(Ordering::Relaxed),
        big_ts.load(Ordering::Relaxed)
    );
    report("runtime manifest round-trip", result);
}

/// Same property with the non-UTF-8 class removed, so it can stay enabled.
#[test]
fn runtime_manifest_round_trips_for_utf8_install_roots() {
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 256,
        ..RunnerConfig::default()
    });
    let utf8_only = manifest_seed().prop_filter("UTF-8 install roots only", |seed| {
        !format!("{:?}", seed.root).starts_with("NonUtf8")
    });
    let total = AtomicUsize::new(0);
    let classes = [
        AtomicUsize::new(0), // unicode root
        AtomicUsize::new(0), // spaced root
        AtomicUsize::new(0), // an empty string field
        AtomicUsize::new(0), // a None optional
        AtomicUsize::new(0), // timestamp above u64::MAX
        AtomicUsize::new(0), // non-ASCII in a string field
    ];
    let result = runner.run(&utf8_only, |seed| {
        total.fetch_add(1, Ordering::Relaxed);
        let hits = [
            matches!(seed.root, RootName::Unicode),
            matches!(seed.root, RootName::Spaced),
            seed.strings.iter().any(String::is_empty),
            seed.optionals.iter().any(Option::is_none),
            seed.installed_at_unix_ms > u128::from(u64::MAX),
            seed.strings.iter().any(|s| !s.is_ascii()),
        ];
        for (counter, hit) in classes.iter().zip(hits) {
            if hit {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
        let (root, paths) = fresh_paths("manifest-utf8");
        let install_root = install_root_under(&root.join("installs"), &seed.root);
        std::fs::create_dir_all(&install_root).unwrap();
        let manifest = build_manifest(&seed, install_root);
        let saved = save_runtime_manifest(&paths, &manifest);
        let listed = load_runtime_manifests(&paths);
        let _ = std::fs::remove_dir_all(&root);
        saved.map_err(|e| TestCaseError::fail(format!("{e:#}")))?;
        let listed = listed.map_err(|e| TestCaseError::fail(format!("{e:#}")))?;
        prop_assert_eq!(listed.len(), 1);
        prop_assert_eq!(manifest_text(&listed[0]), manifest_text(&manifest));
        Ok(())
    });
    let [unicode, spaced, empty, none, big_ts, non_ascii] =
        classes.map(|counter| counter.load(Ordering::Relaxed));
    eprintln!(
        "reach: of {} cases, unicode root {unicode}, spaced root {spaced}, an empty string \
         {empty}, a None optional {none}, timestamp above u64::MAX {big_ts}, non-ASCII field \
         {non_ascii}",
        total.load(Ordering::Relaxed)
    );
    report("runtime manifest round-trip (UTF-8)", result);
}

/// Forward compatibility in the upgrade direction, applied to REAL output: each
/// key this binary writes is dropped in turn, as an older binary would have
/// omitted it. Reports which drops make the record vanish from the listing.
#[test]
fn registry_reports_which_missing_keys_drop_a_runtime_from_the_listing() {
    let (root, paths) = fresh_paths("manifest-drop");
    let install_root = root.join("installs").join("r");
    std::fs::create_dir_all(&install_root).unwrap();
    let seed = ManifestSeed {
        runtime_key: "therock-release-gfx1151-7.13.0".to_owned(),
        strings: [
            "therock-release:gfx1151".to_owned(),
            "release".to_owned(),
            "wheel".to_owned(),
            "gfx1151".to_owned(),
            "manifest".to_owned(),
            "7.13.0".to_owned(),
        ],
        optionals: [None, None, None, None, None, None],
        root: RootName::Ascii("r".to_owned()),
        read_only: false,
        devel: false,
        installed_at_unix_ms: 1_700_000_000_000,
    };
    save_runtime_manifest(&paths, &build_manifest(&seed, install_root)).unwrap();
    let registry_file = paths
        .data_dir
        .join("runtimes")
        .join("registry")
        .join(format!("{}.json", seed.runtime_key));
    let real: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&registry_file).unwrap()).unwrap();
    let mut vanishing = Vec::new();
    for key in real.as_object().unwrap().keys() {
        let mut mutated = real.clone();
        mutated.as_object_mut().unwrap().remove(key);
        std::fs::write(&registry_file, serde_json::to_vec(&mutated).unwrap()).unwrap();
        if load_runtime_manifests(&paths).unwrap().is_empty() {
            vanishing.push(key.clone());
        }
    }
    // Unknown keys (a newer binary's) must not hide the record either.
    let mut extended = real;
    extended["field_from_a_newer_binary"] = serde_json::json!({"x": 1});
    std::fs::write(&registry_file, serde_json::to_vec(&extended).unwrap()).unwrap();
    let with_unknown = load_runtime_manifests(&paths).unwrap().len();
    let _ = std::fs::remove_dir_all(&root);
    eprintln!("registry keys whose absence hides the runtime: {vanishing:?}");
    assert_eq!(with_unknown, 1, "an unknown key hid the runtime");
    // Pinned so a change to this set is a reviewed decision, not drift. The
    // module doc of `load_runtime_manifests_reporting_unparsed` names the
    // older-binary case; the install consent gate is what has to see these.
    assert_eq!(
        vanishing,
        [
            "channel",
            "family",
            "family_source",
            "format",
            "install_root",
            "installed_at_unix_ms",
            "runtime_id",
            "runtime_key",
            "selected_artifact_url",
            "version",
        ]
    );
}

// ---------------------------------------------------------------------------
// Startup update-check cache.
// ---------------------------------------------------------------------------

#[test]
fn startup_update_check_record_round_trips() {
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 256,
        ..RunnerConfig::default()
    });
    let strategy = (
        [text(), text(), text(), text(), text(), text(), text()],
        opt_text(),
        opt_text(),
        u128_edge(),
    );
    let result = runner.run(&strategy, |(strings, latest, message, at)| {
        let [
            runtime_key,
            runtime_id,
            channel,
            format,
            family,
            installed,
            status,
        ] = strings;
        let record = StartupUpdateCheckRecord {
            runtime_key,
            runtime_id,
            channel,
            format,
            family,
            installed_version: installed,
            latest_version: latest,
            status,
            message,
            checked_at_unix_ms: at,
        };
        let (root, paths) = fresh_paths("update-check");
        let saved = save_startup_update_check(&paths, &record);
        let loaded = load_startup_update_check(&paths);
        let _ = std::fs::remove_dir_all(&root);
        saved.map_err(|e| TestCaseError::fail(format!("{e:#}")))?;
        let loaded = loaded
            .map_err(|e| TestCaseError::fail(format!("{e:#}")))?
            .ok_or_else(|| TestCaseError::fail("saved record not found"))?;
        prop_assert_eq!(
            serde_json::to_string(&loaded).unwrap(),
            serde_json::to_string(&record).unwrap()
        );
        Ok(())
    });
    report("startup update check round-trip", result);
}

// ---------------------------------------------------------------------------
// Active-runtime marker: which real-output mutations the strict reader rejects.
// ---------------------------------------------------------------------------

/// `active_runtime_marker_matches` (uninstall path) fails loudly on a marker it
/// cannot parse; `storage::read_active_runtime_marker` turns the same file into
/// "no marker". This enumerates which single-key drops from REAL output put a
/// marker into that disagreement, so the reach of the known asymmetry is
/// measured rather than assumed.
#[test]
fn marker_key_drops_that_split_the_two_readers() {
    let (root, paths) = fresh_paths("marker");
    crate::write_active_runtime_marker(
        &paths,
        crate::ActiveRuntimeMarker {
            runtime_id: "therock-release:gfx1151".to_owned(),
            runtime_key: "therock-release-gfx1151-7.13.0".to_owned(),
            manifest_path: PathBuf::from("/x/registry/k.json"),
            install_root: PathBuf::from("/x/r"),
            previous_runtime_id: None,
            previous_runtime_key: Some("old".to_owned()),
            // Realistic, and below u64::MAX on purpose: the mutation below
            // goes through `serde_json::Value`, which holds an integer above
            // u64::MAX as an f64, and the strict `u128` field then rejects it.
            // That would make every mutation look rejected.
            activated_at_unix_ms: 1_700_000_000_000,
        },
    )
    .unwrap();
    let marker_path = crate::active_runtime_marker_path(&paths);
    let real: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker_path).unwrap()).unwrap();
    assert!(
        crate::active_runtime_marker_matches(&paths, "therock-release-gfx1151-7.13.0").unwrap()
    );
    let mut rejected = Vec::new();
    for key in real.as_object().unwrap().keys() {
        let mut mutated = real.clone();
        mutated.as_object_mut().unwrap().remove(key);
        std::fs::write(&marker_path, serde_json::to_vec(&mutated).unwrap()).unwrap();
        if crate::active_runtime_marker_matches(&paths, "therock-release-gfx1151-7.13.0").is_err() {
            rejected.push(key.clone());
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    eprintln!("marker keys whose absence the strict reader rejects: {rejected:?}");
    assert_eq!(
        rejected,
        [
            "activated_at_unix_ms",
            "install_root",
            "manifest_path",
            "runtime_id",
            "runtime_key",
        ]
    );
}
