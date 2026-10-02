// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the persisted `config.json` (`RocmCliConfig`) and its
//! one-shot migration from the legacy rocm-dash `config.toml`.
//!
//! Each property runs through an explicit `TestRunner` rather than the
//! `proptest!` macro so it can count how often the generator actually reached
//! the interesting classes (absent optionals, empty and non-ASCII strings,
//! non-finite floats) and print that, instead of passing vacuously.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config as RunnerConfig, TestCaseError, TestError, TestRunner};
use rocm_core::{AppPaths, EngineUserConfig, RocmCliConfig};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

/// A fresh, empty `AppPaths` under the system temp dir. No environment is read
/// or written.
fn fresh_paths(tag: &str) -> (PathBuf, AppPaths) {
    let root = std::env::temp_dir().join(format!(
        "rocm-core-props-{tag}-{}-{}",
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

/// Strings biased toward the cases uniform generators miss.
fn text() -> impl Strategy<Value = String> {
    prop_oneof![
        2 => Just(String::new()),
        2 => "[a-z0-9:._-]{1,12}",
        1 => Just("gfx1151-é-日本-\u{1F600}".to_owned()),
        1 => Just(" padded ".to_owned()),
        2 => any::<String>(),
    ]
}

fn opt_text() -> impl Strategy<Value = Option<String>> {
    prop_oneof![3 => Just(None), 2 => text().prop_map(Some)]
}

fn path_text() -> impl Strategy<Value = PathBuf> {
    prop_oneof![
        Just(PathBuf::from("")),
        Just(PathBuf::from("/opt/rocm/../rocm-7/")),
        Just(PathBuf::from(r"D:\ROCm\therock_venvs")),
        Just(PathBuf::from("/mnt/d/ROCm venvs/é")),
        "[a-z/. ]{1,20}".prop_map(PathBuf::from),
    ]
}

fn tick() -> BoxedStrategy<f64> {
    prop_oneof![
        4 => 0.001_f64..120.0,
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(-0.0),
        1 => Just(-1.0),
        1 => Just(f64::MAX),
        1 => Just(f64::from_bits(1)),
    ]
    .boxed()
}

fn engine_cfg() -> impl Strategy<Value = EngineUserConfig> {
    (opt_text(), opt_text(), opt_text(), opt_text()).prop_map(|(a, b, c, d)| EngineUserConfig {
        preferred_runtime_id: a,
        preferred_env_id: b,
        last_installed_runtime_id: c,
        last_installed_env_id: d,
    })
}

#[derive(Debug, Clone)]
struct ConfigSeed {
    default_engine: Option<String>,
    default_runtime_id: Option<String>,
    active_runtime_key: Option<String>,
    previous_runtime_key: Option<String>,
    onboarding_dismissed: bool,
    telemetry_mode: String,
    therock_venv: Option<PathBuf>,
    engines: BTreeMap<String, EngineUserConfig>,
    ticks: (f64, f64, f64),
    chat_temperature: Option<f32>,
    chat_top_p: Option<f32>,
    chat_max_tokens: Option<u32>,
}

fn config_seed(ticks: BoxedStrategy<f64>) -> impl Strategy<Value = ConfigSeed> {
    (
        (opt_text(), opt_text(), opt_text(), opt_text()),
        any::<bool>(),
        text(),
        prop_oneof![2 => Just(None), 1 => path_text().prop_map(Some)],
        prop::collection::btree_map(text(), engine_cfg(), 0..3),
        (ticks.clone(), ticks.clone(), ticks),
        (
            prop_oneof![Just(None), (0.0_f32..2.0).prop_map(Some)],
            prop_oneof![Just(None), (0.0_f32..=1.0).prop_map(Some)],
            prop_oneof![Just(None), (1_u32..=u32::MAX).prop_map(Some)],
        ),
    )
        .prop_map(
            |((de, dr, ak, pk), od, tm, tv, engines, ticks, (temp, top_p, max_tokens))| {
                ConfigSeed {
                    default_engine: de,
                    default_runtime_id: dr,
                    active_runtime_key: ak,
                    previous_runtime_key: pk,
                    onboarding_dismissed: od,
                    telemetry_mode: tm,
                    therock_venv: tv,
                    engines,
                    ticks,
                    chat_temperature: temp,
                    chat_top_p: top_p,
                    chat_max_tokens: max_tokens,
                }
            },
        )
}

fn build_config(seed: &ConfigSeed) -> RocmCliConfig {
    let mut config = RocmCliConfig {
        default_engine: seed.default_engine.clone(),
        default_runtime_id: seed.default_runtime_id.clone(),
        active_runtime_key: seed.active_runtime_key.clone(),
        previous_runtime_key: seed.previous_runtime_key.clone(),
        onboarding_dismissed: seed.onboarding_dismissed,
        engines: seed.engines.clone(),
        ..RocmCliConfig::default()
    };
    config.telemetry.mode.clone_from(&seed.telemetry_mode);
    config.setup.therock_venv.clone_from(&seed.therock_venv);
    config.dashboard.daemon.gpu_tick_secs = seed.ticks.0;
    config.dashboard.daemon.discovery_tick_secs = seed.ticks.1;
    config.dashboard.daemon.instance_tick_secs = seed.ticks.2;
    config.dashboard.tui.chat_temperature = seed.chat_temperature;
    config.dashboard.tui.chat_top_p = seed.chat_top_p;
    config.dashboard.tui.chat_max_tokens = seed.chat_max_tokens;
    config
}

/// Compare configs by their JSON text re-read through the same parser, not by
/// `serde_json::to_value`: `to_value` widens an `f32` to `f64` (`0.17120193`
/// becomes `0.1712019294500351`), and serde_json's default float parser is not
/// correctly rounded (`59.913248351550116` reads back as `59.91324835155011`).
/// Both are serde_json properties, harmless for these fields, and must not mask
/// real loss.
fn as_value(config: &RocmCliConfig) -> serde_json::Value {
    let text = serde_json::to_string(config).expect("config serializes");
    serde_json::from_str(&text).expect("serialized config parses as JSON")
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

/// `load(save(x)) == x` for every config whose ticks are finite. This is the
/// half of the round-trip that holds; it stays enabled as a regression guard.
#[test]
fn config_round_trips_through_save_and_load_for_finite_values() {
    let (root, paths) = fresh_paths("roundtrip-finite");
    let total = AtomicUsize::new(0);
    let none_active = AtomicUsize::new(0);
    let empty_strings = AtomicUsize::new(0);
    let non_ascii = AtomicUsize::new(0);
    let finite_ticks = (0.001_f64..120.0).boxed();
    let finite = prop_oneof![
        3 => finite_ticks,
        1 => Just(-0.0),
        1 => Just(f64::MAX),
        1 => Just(f64::from_bits(1)),
        1 => Just(-1.0),
    ]
    .boxed();
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 512,
        ..RunnerConfig::default()
    });
    let result = runner.run(&config_seed(finite), |seed| {
        total.fetch_add(1, Ordering::Relaxed);
        let config = build_config(&seed);
        let json = serde_json::to_string(&config).unwrap_or_default();
        if seed.active_runtime_key.is_none() {
            none_active.fetch_add(1, Ordering::Relaxed);
        }
        if json.contains("\"\"") {
            empty_strings.fetch_add(1, Ordering::Relaxed);
        }
        if !json.is_ascii() {
            non_ascii.fetch_add(1, Ordering::Relaxed);
        }
        config
            .save(&paths)
            .map_err(|e| TestCaseError::fail(format!("{e:#}")))?;
        let loaded =
            RocmCliConfig::load(&paths).map_err(|e| TestCaseError::fail(format!("{e:#}")))?;
        prop_assert_eq!(as_value(&loaded), as_value(&config));
        Ok(())
    });
    eprintln!(
        "reach: of {} cases, active_runtime_key=None {}, empty strings {}, non-ASCII {}",
        total.load(Ordering::Relaxed),
        none_active.load(Ordering::Relaxed),
        empty_strings.load(Ordering::Relaxed),
        non_ascii.load(Ordering::Relaxed)
    );
    let _ = std::fs::remove_dir_all(root);
    report("finite round-trip", result);
}

/// `save` must never write a file that `load` then refuses. A non-finite tick
/// serializes as JSON `null`, which the `f64` field then rejects, so the saved
/// config cannot be read back at all.
#[test]
fn config_save_never_writes_a_file_load_rejects() {
    let (root, paths) = fresh_paths("roundtrip-any");
    let non_finite = AtomicUsize::new(0);
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 256,
        ..RunnerConfig::default()
    });
    let result = runner.run(&config_seed(tick()), |seed| {
        if ![seed.ticks.0, seed.ticks.1, seed.ticks.2]
            .iter()
            .all(|t| t.is_finite())
        {
            non_finite.fetch_add(1, Ordering::Relaxed);
        }
        let config = build_config(&seed);
        if config.save(&paths).is_err() {
            // Refusing to save is acceptable; writing an unreadable file is not.
            return Ok(());
        }
        RocmCliConfig::load(&paths)
            .map(|_| ())
            .map_err(|e| TestCaseError::fail(format!("{e:#}")))
    });
    eprintln!(
        "reach: non-finite tick in {} of 256",
        non_finite.load(Ordering::Relaxed)
    );
    let _ = std::fs::remove_dir_all(root);
    report("save/load agreement", result);
}

fn toml_float(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value == f64::INFINITY {
        "inf".to_owned()
    } else if value == f64::NEG_INFINITY {
        "-inf".to_owned()
    } else {
        format!("{value:?}")
    }
}

/// Run the legacy-TOML migration property over the given tick strategy.
///
/// The migration writes `config.json` once and never again, so it has to write
/// something the unified loader accepts, carrying the user's values. Generated
/// as TOML *text* so the real `toml` 1.x parser is in the loop.
fn legacy_migration_property(ticks: &BoxedStrategy<f64>, label: &str) {
    use std::fmt::Write as _;
    let total = AtomicUsize::new(0);
    let non_finite = AtomicUsize::new(0);
    let non_ascii_engine = AtomicUsize::new(0);
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 256,
        ..RunnerConfig::default()
    });
    let strategy = (ticks.clone(), ticks.clone(), ticks.clone(), opt_text());
    let result = runner.run(&strategy, |(g, d, i, engine)| {
        total.fetch_add(1, Ordering::Relaxed);
        if ![g, d, i].iter().all(|t| t.is_finite()) {
            non_finite.fetch_add(1, Ordering::Relaxed);
        }
        if engine.as_deref().is_some_and(|e| !e.is_ascii()) {
            non_ascii_engine.fetch_add(1, Ordering::Relaxed);
        }
        let (root, paths) = fresh_paths("legacy");
        std::fs::create_dir_all(&root).unwrap();
        let legacy = root.join("config.toml");
        let mut toml_text = String::new();
        if let Some(engine) = &engine {
            let _ = writeln!(
                toml_text,
                "default_engine = {}",
                toml_string_literal(engine)
            );
        }
        let _ = write!(
            toml_text,
            "[daemon]\ngpu_tick = {}\ndiscovery_tick = {}\ninstance_tick = {}\n",
            toml_float(g),
            toml_float(d),
            toml_float(i)
        );
        std::fs::write(&legacy, &toml_text).unwrap();
        let migrated = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy);
        let loaded = RocmCliConfig::load(&paths);
        let _ = std::fs::remove_dir_all(root);
        match migrated {
            // Refusing to migrate is fine: the legacy file is left untouched.
            Err(_) | Ok(None) => Ok(()),
            Ok(Some(_)) => {
                let loaded = loaded.map_err(|e| {
                    TestCaseError::fail(format!(
                        "migrated config.json is unreadable: {e:#}\nlegacy toml:\n{toml_text}"
                    ))
                })?;
                prop_assert_eq!(loaded.default_engine, engine);
                Ok(())
            }
        }
    });
    eprintln!(
        "reach ({label}): of {} cases, non-finite tick {}, non-ASCII default_engine {}",
        total.load(Ordering::Relaxed),
        non_finite.load(Ordering::Relaxed),
        non_ascii_engine.load(Ordering::Relaxed)
    );
    report(label, result);
}

/// The half that holds: finite ticks and arbitrary strings migrate and load
/// back. Kept enabled; it is also the `toml` 0.8 -> 1.x string guard.
#[test]
fn legacy_toml_migration_round_trips_finite_values() {
    let finite = prop_oneof![
        3 => 0.001_f64..120.0,
        1 => Just(-1.0),
        1 => Just(-0.0),
        1 => Just(f64::MAX),
    ]
    .boxed();
    legacy_migration_property(&finite, "legacy migration (finite)");
}

#[test]
fn legacy_toml_migration_always_produces_a_loadable_config() {
    legacy_migration_property(&tick(), "legacy migration (any)");
}

fn toml_string_literal(value: &str) -> String {
    use std::fmt::Write as _;
    // A basic string with every char escaped as \u/\U keeps any input valid TOML.
    let mut out = String::from("\"");
    for ch in value.chars() {
        let code = u32::from(ch);
        if code <= 0xFFFF {
            let _ = write!(out, "\\u{code:04X}");
        } else {
            let _ = write!(out, "\\U{code:08X}");
        }
    }
    out.push('"');
    out
}

/// Forward compatibility, downgrade direction: a key this binary does not know
/// (written by a newer one sharing the same config dir) should survive a
/// load-then-save by this binary. Mutates REAL serialized output.
#[test]
#[ignore = "CONFIRMED (by-design gap): unknown keys are dropped on the next save"]
fn unknown_keys_survive_a_load_save_cycle() {
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 64,
        ..RunnerConfig::default()
    });
    let finite = (0.001_f64..120.0).boxed();
    let result = runner.run(&(config_seed(finite), "[a-z_]{3,12}"), |(seed, key)| {
        let (root, paths) = fresh_paths("unknown-key");
        build_config(&seed).save(&paths).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(paths.config_path()).unwrap()).unwrap();
        let key = format!("future_{key}");
        value["setup"][&key] = serde_json::json!("kept");
        std::fs::write(
            paths.config_path(),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
        let loaded = RocmCliConfig::load(&paths).unwrap();
        loaded.save(&paths).unwrap();
        let after: serde_json::Value =
            serde_json::from_slice(&std::fs::read(paths.config_path()).unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(root);
        prop_assert_eq!(&after["setup"][&key], &serde_json::json!("kept"));
        Ok(())
    });
    report("unknown key survival", result);
}

/// Windows path normalization is applied on every manifest load and save, so
/// it must be idempotent: normalizing an already-normalized path is a no-op.
/// Exercised through the platform-parameterized entry point so it also runs on
/// a Linux host.
#[test]
fn windows_path_normalization_is_idempotent() {
    use rocm_core::runtime::{RuntimePlatform, normalize_runtime_path_text_for_platform as norm};
    let segment = prop_oneof![
        Just(String::new()),
        Just("..".to_owned()),
        Just(" spaced ".to_owned()),
        Just("é日".to_owned()),
        "[A-Za-z0-9_.-]{1,6}",
    ];
    let separator = prop_oneof![Just("/"), Just("\\"), Just("\\/"), Just("//")];
    let prefix = prop_oneof![
        Just(String::new()),
        Just("C:".to_owned()),
        Just("d:".to_owned()),
        Just("/d".to_owned()),
        Just("\\\\server\\share".to_owned()),
        Just(" ".to_owned()),
    ];
    let strategy = (
        prefix,
        prop::collection::vec((separator, segment), 0..5),
        prop_oneof![Just(""), Just("/"), Just("\\"), Just(" ")],
    )
        .prop_map(|(prefix, parts, trailing)| {
            let mut path = prefix;
            for (sep, seg) in parts {
                path.push_str(sep);
                path.push_str(&seg);
            }
            path.push_str(trailing);
            path
        });
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 1024,
        ..RunnerConfig::default()
    });
    let result = runner.run(&strategy, |path| {
        let once = norm(&path, RuntimePlatform::Windows);
        let twice = norm(&once, RuntimePlatform::Windows);
        prop_assert_eq!(&twice, &once, "input {:?}", path);
        Ok(())
    });
    report("windows normalization idempotence", result);
}
