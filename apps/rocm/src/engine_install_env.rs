// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which runtime an engine installs into, and where its environment lives.
//!
//! This is the neutral layer between `engines_cmd.rs` (the `rocm engines`
//! command, and the readiness check `serve` runs for self-managed engines) and
//! `runtime_install_finalize.rs` (settling an engine onto a runtime after an
//! install). Both need to answer the same questions, so the answers live here
//! and neither of those modules has to import the other for them:
//! `engine_manages_own_runtime` says whether an engine brings its own runtime,
//! `env_root_for_engine_install`, `env_root_for_runtime` and
//! `env_root_for_self_managed_engine` resolve the engine environment root,
//! `runtime_manifest_for_selector` resolves a selector to one installed
//! runtime, and `runtime_key_for_python` attributes an interpreter to the
//! runtime whose install root contains it.
//!
//! Relocated from `engines_cmd.rs` with no behavior change. It calls into
//! `runtimes_cmd.rs` for registry selection and validation, and nothing in
//! `runtimes_cmd.rs` calls back here.

use std::path::{Path, PathBuf};

use anyhow::Result;
use rocm_core::{AppPaths, RocmCliConfig};

use crate::runtimes_cmd::{
    recover_setup_runtime_registration, select_runtime_manifest,
    validate_runtime_manifest_for_activation,
};
use crate::therock;

pub(crate) fn engine_manages_own_runtime(engine: &str) -> bool {
    engine == "lemonade" || engine == "aim"
}

pub(crate) fn env_root_for_runtime(
    paths: &AppPaths,
    engine: &str,
    runtime_id: &str,
) -> Result<Option<PathBuf>> {
    if engine_manages_own_runtime(engine) {
        return Ok(None);
    }
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, runtime_id)?;
    Ok(Some(manifest.install_root.join("engines")))
}

pub(crate) fn env_root_for_engine_install(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: &str,
) -> Result<Option<PathBuf>> {
    if engine_manages_own_runtime(engine) {
        return env_root_for_self_managed_engine(paths, config, engine);
    }
    env_root_for_runtime(paths, engine, runtime_id)
}

/// Only `lemonade` installs into a therock-managed Python runtime; `aim` runs
/// entirely in Docker and has no env root to match against. Answering `None`
/// for it here (rather than returning whatever unrelated runtime happens to
/// be active) keeps `detect_runtime_matches_env_root` from failing aim's
/// Detect response for a reason that has nothing to do with aim.
pub(crate) fn env_root_for_self_managed_engine(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
) -> Result<Option<PathBuf>> {
    if engine != "lemonade" {
        return Ok(None);
    }
    recover_setup_runtime_registration(paths, config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    for selector in [
        config.active_runtime_key.as_deref(),
        config.default_runtime_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(manifest) = runtime_manifest_for_selector(&manifests, selector) {
            return Ok(Some(manifest.install_root.join("engines")));
        }
    }
    let ready = manifests
        .iter()
        .filter(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok())
        .collect::<Vec<_>>();
    Ok(match ready.as_slice() {
        [manifest] => Some(manifest.install_root.join("engines")),
        _ => None,
    })
}

pub(crate) fn runtime_manifest_for_selector<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    selector: &str,
) -> Option<&'a therock::InstalledRuntimeManifest> {
    manifests
        .iter()
        .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
        .or_else(|| {
            let mut matches = manifests
                .iter()
                .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector));
            let first = matches.next()?;
            if matches.next().is_none() {
                Some(first)
            } else {
                None
            }
        })
}

/// The `runtime_key` of the runtime whose install root contains `python`.
pub(crate) fn runtime_key_for_python(paths: &AppPaths, python: &Path) -> Option<String> {
    let manifests = therock::load_runtime_manifests(paths).ok()?;
    runtime_key_owning_python(&manifests, python).map(str::to_owned)
}

/// Which runtime owns an interpreter, decided by install root.
///
/// Split from the registry read so the decision can be tested without a
/// registry on disk, matching `sdk_torch_build_from_manifest`.
///
/// `runtime_id` cannot answer this: it is shared by every side-by-side install
/// of one channel and family, which is exactly the situation an engine install
/// has to be attributed in. An install root contains one runtime by
/// construction, so the interpreter's path settles it.
///
/// Both sides are compared verbatim *and* canonicalized. The CLI writes
/// `install_root` canonicalized while an engine adapter reports back whatever
/// path it was handed, and comparing a single form makes ownership fail
/// silently on a symlinked runtimes directory. Roots can nest, so the longest
/// containing root wins.
fn runtime_key_owning_python<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    python: &Path,
) -> Option<&'a str> {
    fn both_forms(path: &Path) -> Vec<PathBuf> {
        let verbatim = path.to_path_buf();
        match path.canonicalize() {
            Ok(resolved) if resolved != verbatim => vec![verbatim, resolved],
            _ => vec![verbatim],
        }
    }

    let pythons = both_forms(python);
    manifests
        .iter()
        .filter(|manifest| {
            both_forms(&manifest.install_root)
                .iter()
                .any(|root| pythons.iter().any(|python| python.starts_with(root)))
        })
        .max_by_key(|manifest| manifest.install_root.as_os_str().len())
        .map(|manifest| manifest.runtime_key.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::{test_paths, test_runtime_manifest_for_update};
    use std::fs;

    /// Two runtimes installed side by side, as a pre-warmed CI tree holds them.
    ///
    /// They differ in `runtime_key`, `version` and install root, and share one
    /// `runtime_id` — that is what the field means, so this is not a corrupt
    /// registry.
    fn side_by_side_runtimes() -> Vec<therock::InstalledRuntimeManifest> {
        let mut older = test_runtime_manifest_for_update(
            "release-wheel-gfx94x-dcgpu-7-13-0",
            "therock-release:gfx94X-dcgpu",
            "gfx94X-dcgpu",
            "7.13.0",
        );
        older.install_root = PathBuf::from("/runtimes/release-wheel-gfx94x-dcgpu-7-13-0");
        let mut newer = test_runtime_manifest_for_update(
            "release-wheel-gfx94x-dcgpu-7-14-0",
            "therock-release:gfx94X-dcgpu",
            "gfx94X-dcgpu",
            "7.14.0",
        );
        newer.install_root = PathBuf::from("/runtimes/release-wheel-gfx94x-dcgpu-7-14-0");
        vec![older, newer]
    }

    #[test]
    fn env_root_for_runtime_uses_runtime_install_root() -> Result<()> {
        let (root, paths) = test_paths("engine-env-root-runtime");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;

        let engine_root = env_root_for_runtime(&paths, "vllm", &manifest.runtime_key)?;

        assert_eq!(engine_root, Some(manifest.install_root.join("engines")));
        assert_eq!(
            env_root_for_runtime(&paths, "lemonade", &manifest.runtime_key)?,
            None
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn env_root_for_engine_install_uses_active_runtime_root_for_lemonade() -> Result<()> {
        let (root, paths) = test_paths("lemonade-engine-env-root-runtime");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        let engine_root =
            env_root_for_engine_install(&paths, &config, "lemonade", "lemonade-embeddable")?;

        assert_eq!(engine_root, Some(manifest.install_root.join("engines")));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// The interpreter names its runtime where the shared `runtime_id` cannot.
    ///
    /// This is the cross-wiring that settled the active runtime's torch into an
    /// older runtime's environment: the engine's env id drops the version, so
    /// the environment belongs to 7.13.0 while the caller's selector says only
    /// "release, gfx94X-dcgpu". Resolving by install root has to pick 7.13.0.
    #[test]
    fn the_runtime_is_resolved_by_its_interpreter_not_the_shared_runtime_id() {
        let manifests = side_by_side_runtimes();

        assert_eq!(
            runtime_manifest_for_selector(&manifests, "therock-release:gfx94X-dcgpu")
                .map(|manifest| manifest.runtime_key.as_str()),
            None,
            "the shared runtime_id names two runtimes, so a selector cannot resolve it"
        );
        assert_eq!(
            runtime_key_owning_python(
                &manifests,
                Path::new("/runtimes/release-wheel-gfx94x-dcgpu-7-13-0/bin/python3"),
            ),
            Some("release-wheel-gfx94x-dcgpu-7-13-0"),
            "the interpreter's install root names the runtime being settled"
        );
    }

    /// An interpreter outside every install root leaves the caller's selector alone.
    ///
    /// External and self-managed environments live outside the registry, and
    /// inventing an owner for them would settle a runtime nobody asked about.
    #[test]
    fn an_interpreter_outside_every_install_root_owns_nothing() {
        assert_eq!(
            runtime_key_owning_python(
                &side_by_side_runtimes(),
                Path::new("/opt/somewhere-else/bin/python3"),
            ),
            None
        );
    }

    /// A prefix match alone is ambiguous once roots nest, so the longest wins.
    #[test]
    fn the_longest_containing_install_root_owns_the_interpreter() {
        let mut outer = test_runtime_manifest_for_update(
            "outer",
            "therock-release:gfx94X-dcgpu",
            "gfx94X-dcgpu",
            "7.13.0",
        );
        outer.install_root = PathBuf::from("/runtimes");
        let mut inner = test_runtime_manifest_for_update(
            "inner",
            "therock-release:gfx94X-dcgpu",
            "gfx94X-dcgpu",
            "7.14.0",
        );
        inner.install_root = PathBuf::from("/runtimes/release-wheel-gfx94x-dcgpu-7-14-0");

        assert_eq!(
            runtime_key_owning_python(
                &[outer, inner],
                Path::new("/runtimes/release-wheel-gfx94x-dcgpu-7-14-0/bin/python3"),
            ),
            Some("inner")
        );
    }
}
