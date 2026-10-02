// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property: the two readers of `config.json` agree about where the data dir
//! is.
//!
//! `RocmCliConfig::load` fails loudly on a file it cannot parse.
//! `AppPaths::discover` reads the same file a second way, through
//! `configured_managed_root_from_config`, for the one field that decides where
//! the runtime registry, the active-runtime marker and the service records
//! live. It used to turn any read or parse error into "no managed root", so a
//! config one reader called corrupt the other called "absent", and the data
//! dir moved to the default with no message. Discovery must now either fail
//! or still find the recorded root.
//!
//! No environment is read or written: `discover_from_paths` is the env-free
//! core of `discover`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config as RunnerConfig, TestError, TestRunner};

use crate::{AppPaths, RocmCliConfig};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn fresh_paths() -> (PathBuf, AppPaths) {
    let root = std::env::temp_dir().join(format!(
        "rocm-core-paths-props-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    let paths = AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("default-data"),
        cache_dir: root.join("default-cache"),
    };
    (root, paths)
}

#[test]
fn data_dir_never_silently_relocates_when_config_is_unreadable() {
    let total = AtomicUsize::new(0);
    let unreadable = AtomicUsize::new(0);
    let mut runner = TestRunner::new(RunnerConfig {
        cases: 256,
        ..RunnerConfig::default()
    });
    let strategy = ("[a-z]{1,10}", 1_usize..2048);
    let result = runner.run(&strategy, |(leaf, cut)| {
        total.fetch_add(1, Ordering::Relaxed);
        let (root, paths) = fresh_paths();
        let managed_root = root.join("prefix").join(&leaf);
        let mut config = RocmCliConfig::default();
        config.setup.therock_venv = Some(managed_root.clone());
        config.active_runtime_key = Some("therock-release-gfx1151-7.13.0".to_owned());
        config.save(&paths).unwrap();
        let healthy = AppPaths::discover_from_paths(paths.clone(), false, false).unwrap();
        prop_assert_eq!(&healthy.data_dir, &managed_root);

        // An interrupted `RocmCliConfig::save` (a plain `fs::write`).
        let bytes = std::fs::read(paths.config_path()).unwrap();
        std::fs::write(paths.config_path(), &bytes[..cut.min(bytes.len() - 1)]).unwrap();
        let strict = RocmCliConfig::load(&paths);
        let discovered = AppPaths::discover_from_paths(paths, false, false);
        let _ = std::fs::remove_dir_all(&root);
        if strict.is_ok() {
            return Ok(());
        }
        unreadable.fetch_add(1, Ordering::Relaxed);
        // Failing is the agreement with `load`; a data dir is only acceptable
        // if it is still the recorded one.
        if let Ok(discovered) = discovered {
            prop_assert_eq!(
                &discovered.data_dir,
                &managed_root,
                "config.json is unreadable to RocmCliConfig::load, yet AppPaths::discover \
                 silently moved data_dir (registry, marker, services) to the default"
            );
        }
        Ok(())
    });
    eprintln!(
        "reach: {} of {} cases left a config.json that load() rejects",
        unreadable.load(Ordering::Relaxed),
        total.load(Ordering::Relaxed)
    );
    match result {
        Ok(()) => {}
        Err(TestError::Fail(reason, value)) => {
            panic!("minimal failing input {value:?}\nreason: {reason}")
        }
        Err(TestError::Abort(reason)) => panic!("aborted: {reason}"),
    }
}
