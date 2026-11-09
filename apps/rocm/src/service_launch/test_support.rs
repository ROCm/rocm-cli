// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Fixtures shared by the tests of more than one launch submodule.

use anyhow::Result;
use rocm_core::{AppPaths, ManagedServiceRecord};
#[cfg(not(windows))]
use std::thread;
#[cfg(not(windows))]
use std::time::Duration;

/// A record with every reuse-relevant field set to the "reusable" value, so
/// each test that uses it breaks exactly one term and nothing else.
pub(super) fn reusable_record(paths: &AppPaths) -> ManagedServiceRecord {
    let mut record = ManagedServiceRecord::new(
        paths,
        "vllm-qwen-1000",
        "vllm",
        "qwen",
        "qwen-canonical".to_owned(),
        "127.0.0.1",
        8000,
        "managed",
        0,
        None,
        None,
        Some("gpu_required".to_owned()),
    );
    record.engine_recipe_json = None;
    record.requires_api_key = true;
    record
}

pub(super) fn write_live_service_record(
    paths: &AppPaths,
    service_id: &str,
    canonical_model_id: &str,
    port: u16,
    status: &str,
) -> Result<ManagedServiceRecord> {
    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        "lemonade",
        canonical_model_id,
        canonical_model_id.to_owned(),
        rocm_core::DEFAULT_LOCAL_HOST,
        port,
        "managed",
        std::process::id(),
        None,
        None,
        None,
    );
    record.status = status.to_owned();
    record.engine_pid = Some(std::process::id());
    record.write()?;
    Ok(record)
}

/// The helper the cleanup tests re-run this test binary as, to get a live child
/// process that outlives the spawn they are checking. It is `#[ignore]`d so a
/// normal run skips it, and it only sleeps when the tests that spawn it set
/// `ROCM_TEST_SLEEPER_CHILD`. They select it by [`sleeper_test_name`].
#[test]
#[cfg(not(windows))]
#[ignore = "spawned as a child process by abandoned_launch_terminates_the_spawned_child"]
fn managed_launch_cleanup_sleeper_child() {
    if std::env::var_os("ROCM_TEST_SLEEPER_CHILD").is_some() {
        thread::sleep(Duration::from_mins(2));
    }
}

/// The name libtest lists [`managed_launch_cleanup_sleeper_child`] under, which
/// is what `--exact` has to be given to run it alone.
///
/// It is read off the function item itself, so renaming the function or moving
/// this module changes it with them and the three tests that spawn the helper
/// cannot be left naming a test that no longer exists. libtest names a test by
/// its path inside the crate, while the item path starts with the crate name,
/// which is dropped. The assertions check that against the module path as well.
#[cfg(not(windows))]
pub(super) fn sleeper_test_name() -> String {
    let item_path = std::any::type_name_of_val(&managed_launch_cleanup_sleeper_child);
    let name = item_path
        .split_once("::")
        .map_or(item_path, |(_krate, rest)| rest);
    let module = module_path!()
        .split_once("::")
        .map_or("", |(_krate, rest)| rest);
    assert_eq!(
        name,
        format!("{module}::managed_launch_cleanup_sleeper_child"),
        "the sleeper helper is no longer where its name is derived from"
    );
    assert!(
        name.starts_with("service_launch::"),
        "libtest names tests by their path inside the crate: {name}"
    );
    name.to_owned()
}
