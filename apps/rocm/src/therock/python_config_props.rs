// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for how the managed-Python step, a `config.json` writer in
//! this module, treats a `config.json` it cannot read, and for the legacy
//! rocm-dash TOML migration that could produce one.
//!
//! A child of `python` so it can drive the private
//! `record_managed_python_config`. No test here reads or writes the process
//! environment: every `AppPaths` is built by hand under the system temp dir.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config as RunnerConfig, TestCaseError, TestError, TestRunner};
use rocm_core::{AppPaths, RocmCliConfig};

use super::record_managed_python_config;

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
    assert!(
        unreadable.load(Ordering::Relaxed) > 0,
        "no case produced a config.json that load() rejects"
    );
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

/// A `config.json` that cannot be read at all — here a directory, which
/// fails the read on every platform without chmod or root — is not "no
/// settings yet" either. Recording must refuse with the same message, and
/// leave what is at that path alone.
#[test]
fn managed_python_recording_refuses_a_config_path_it_cannot_read() {
    let (root, paths) = fresh_paths("config-unreadable-path");
    std::fs::create_dir_all(paths.config_path()).unwrap();
    let kept = paths.config_path().join("kept");
    std::fs::write(&kept, b"user data").unwrap();

    let error = record_managed_python_config(&paths, Path::new("/usr/bin/python3"))
        .expect_err("recording over an unreadable config path must fail");
    let still_a_dir = paths.config_path().is_dir();
    let kept_after = std::fs::read(&kept);
    let _ = std::fs::remove_dir_all(&root);

    let message = format!("{error:#}");
    assert!(
        message.contains("cannot record the managed Python")
            && message.contains("it was left unchanged"),
        "unexpected error: {message}"
    );
    // Pinned here; `config_rename_advice_keeps_a_recorded_data_dir_when_followed`
    // in rocm-core's paths.rs follows the advice and checks the data dir.
    assert!(
        message.contains(&paths.config_aside_advice()),
        "the error does not name the file to rename: {message}"
    );
    assert!(still_a_dir, "the path was replaced");
    assert_eq!(kept_after.unwrap(), b"user data");
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
