// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Filesystem-location primitives.
//!
//! `AppPaths` (the config/data/cache root discovery and its derived
//! per-engine/per-service subpaths), the configured managed-root override,
//! and engine plugin directories.

use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::runtime::env_path_override;
use crate::runtime::{
    default_cache_dir, default_config_dir, default_data_dir, managed_runtime_cache_dir,
    managed_runtime_data_root, normalize_runtime_path_for_host,
};

#[derive(Debug, Clone, Serialize)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    /// Resolve the config, data and cache dirs from the environment and
    /// `config.json`.
    ///
    /// Errors when no data dir is given through `ROCM_CLI_DATA_DIR` and
    /// `config.json` exists but cannot be read: the managed root it records
    /// decides the data dir, so guessing the default would point every command
    /// at the wrong runtimes. Callers that only want paths for best-effort work
    /// (logging, diagnosis) already treat an error here as "no paths".
    pub fn discover() -> Result<Self> {
        let data_dir_override = env_path_override("ROCM_CLI_DATA_DIR");
        let cache_dir_override = env_path_override("ROCM_CLI_CACHE_DIR");
        let paths = Self {
            config_dir: env_path_override("ROCM_CLI_CONFIG_DIR")
                .or_else(default_config_dir)
                .context("unable to determine config directory for rocm-cli")?,
            data_dir: data_dir_override
                .clone()
                .or_else(default_data_dir)
                .context("unable to determine data directory for rocm-cli")?,
            cache_dir: cache_dir_override
                .clone()
                .or_else(default_cache_dir)
                .context("unable to determine cache directory for rocm-cli")?,
        }
        .normalize_for_host();
        Self::discover_from_paths(
            paths,
            data_dir_override.is_some(),
            cache_dir_override.is_some(),
        )
    }

    /// Fails when `ROCM_CLI_DATA_DIR` is unset and `config.json` exists but
    /// cannot be read, because the data dir it records is then unknown; see
    /// `configured_managed_root_from_config`.
    fn discover_from_paths(
        mut paths: Self,
        data_dir_overridden: bool,
        cache_dir_overridden: bool,
    ) -> Result<Self> {
        if !data_dir_overridden
            && let Some(managed_root) = configured_managed_root_from_config(&paths)?
        {
            paths = paths.with_managed_root(managed_root, cache_dir_overridden);
        }
        Ok(paths.normalize_for_host())
    }

    fn normalize_for_host(mut self) -> Self {
        self.config_dir = normalize_runtime_path_for_host(&self.config_dir);
        self.data_dir = normalize_runtime_path_for_host(&self.data_dir);
        self.cache_dir = normalize_runtime_path_for_host(&self.cache_dir);
        self
    }

    #[must_use]
    pub fn with_managed_root(mut self, root: impl Into<PathBuf>, keep_cache_dir: bool) -> Self {
        self.data_dir = managed_runtime_data_root(&root.into());
        if !keep_cache_dir {
            self.cache_dir = managed_runtime_cache_dir(&self.data_dir);
        }
        self.normalize_for_host()
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.cache_dir,
            &self.audit_dir(),
            &self.automations_dir(),
            &self.data_dir.join("engines"),
            &self.data_dir.join("envs"),
            &self.data_dir.join("logs"),
            &self.data_dir.join("services"),
            &self.data_dir.join("models"),
            &self.data_dir.join("runtimes"),
            &self.telemetry_state_dir(),
        ] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
        Ok(())
    }

    pub fn engine_dir(&self, engine: &str) -> PathBuf {
        self.data_dir.join("engines").join(engine)
    }

    pub fn primary_engine_plugin_dir(&self) -> PathBuf {
        self.data_dir.join("engines").join("plugins")
    }

    pub fn engine_logs_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("logs")
    }

    /// Where engine virtualenvs live, honouring `ROCM_CLI_ENGINE_ENVS_ROOT`.
    ///
    /// Read-side of a write-only contract at present: `apps/rocm` exports this
    /// key into the children it spawns, and this is the only code that reads it
    /// back, so no in-tree production path reaches here today. That predates
    /// the seam below and is deliberate — the key exists for an engine process
    /// to honour. Being `pub` says nothing either way: `unreachable_pub` is
    /// switched off workspace-wide (see the root `Cargo.toml`), so nothing warns
    /// about an unused one. Delete this and the key together if the contract is
    /// dropped; `engine_envs_dir_reads_its_root_from_the_environment` is what
    /// keeps the lookup honest meanwhile.
    pub fn engine_envs_root(&self) -> PathBuf {
        self.engine_envs_root_from(env_path_override("ROCM_CLI_ENGINE_ENVS_ROOT").as_deref())
    }

    /// [`Self::engine_envs_root`] against a caller-supplied override.
    ///
    /// Lets a test drive the override without `std::env::set_var`, which is
    /// process-global and races other tests under a threaded runner. Same seam
    /// as [`discover_rocm_installs_in_layout`].
    fn engine_envs_root_from(&self, override_root: Option<&Path>) -> PathBuf {
        override_root.map_or_else(
            || self.data_dir.join("engines"),
            normalize_runtime_path_for_host,
        )
    }

    pub fn engine_envs_dir(&self, engine: &str) -> PathBuf {
        self.engine_envs_root().join(engine).join("envs")
    }

    /// [`Self::engine_envs_dir`] against a caller-supplied override.
    #[cfg(test)]
    fn engine_envs_dir_from(&self, engine: &str, override_root: Option<&Path>) -> PathBuf {
        self.engine_envs_root_from(override_root)
            .join(engine)
            .join("envs")
    }

    pub fn engine_locks_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("locks")
    }

    pub fn engine_manifests_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("manifests")
    }

    pub fn engine_state_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("state")
    }

    pub fn config_path(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }

    pub fn services_dir(&self) -> PathBuf {
        self.data_dir.join("services")
    }

    /// Lock file serializing the managed-serve GPU select-then-claim sequence, so
    /// two concurrent `rocm serve` invocations cannot read the same free GPU and
    /// both launch on it. Held from auto-selection through the claiming service
    /// record write (see [`FileLock`](crate::FileLock)).
    pub fn managed_launch_lock_path(&self) -> PathBuf {
        self.services_dir().join("launch.lock")
    }

    /// Where `rocm remote` records the sessions it started on other machines.
    ///
    /// Kept beside [`Self::services_dir`] and following the same file-per-record
    /// shape, but deliberately separate: these describe work running on a
    /// *different* machine, and anything walking the local service registry
    /// (status rendering, the daemon's recovery supervisor) must not mistake a
    /// remote session for a local server it can supervise.
    pub fn remote_sessions_dir(&self) -> PathBuf {
        self.data_dir.join("remote-sessions")
    }

    pub fn audit_dir(&self) -> PathBuf {
        self.data_dir.join("audit")
    }

    pub fn audit_events_path(&self) -> PathBuf {
        self.audit_dir().join("events.jsonl")
    }

    pub fn automations_dir(&self) -> PathBuf {
        self.data_dir.join("automations")
    }

    pub fn automation_state_path(&self) -> PathBuf {
        self.automations_dir().join("runtime-state.json")
    }

    /// Lock file serializing the daemon autostart check-then-spawn, so two
    /// concurrent callers cannot both observe "not running" and each spawn a
    /// background automation daemon (see [`FileLock`](crate::FileLock)).
    pub fn automation_autostart_lock_path(&self) -> PathBuf {
        self.automations_dir().join("autostart.lock")
    }

    /// Short-lived claim written by the autostart holder right after it spawns
    /// the daemon, recording the child PID and spawn time. It bridges the gap
    /// between `spawn()` and the child publishing its runtime state: a concurrent
    /// caller that acquires the autostart lock in that window sees the claim and
    /// defers instead of spawning a duplicate daemon.
    pub fn automation_autostart_claim_path(&self) -> PathBuf {
        self.automations_dir().join("autostart.claim")
    }

    pub fn automation_events_path(&self) -> PathBuf {
        self.automations_dir().join("events.jsonl")
    }

    pub fn automation_proposals_path(&self) -> PathBuf {
        self.automations_dir().join("proposals.jsonl")
    }

    pub fn service_manifest_path(&self, service_id: &str) -> PathBuf {
        self.services_dir().join(format!("{service_id}.json"))
    }

    pub fn service_log_path(&self, service_id: &str) -> PathBuf {
        self.services_dir().join(format!("{service_id}.log"))
    }

    pub fn service_engine_state_path(&self, engine: &str, service_id: &str) -> PathBuf {
        self.engine_state_dir(engine)
            .join(format!("{service_id}.json"))
    }

    /// Directory holding rocm-dash telemetry daemon state.
    /// (G3 rocm-cli maintainer sign-off pending — engineering implementation only.)
    pub fn telemetry_state_dir(&self) -> PathBuf {
        self.data_dir.join("telemetry")
    }

    /// The directive file that moves `rocm dash`'s telemetry daemon off wall time
    /// and onto a logical observation clock (see `docs/release-trust.md`).
    ///
    /// `rocm dash` reads it in every build, and nothing in rocm-cli creates it —
    /// only the E2E harness plants one, in its isolated data root. Both resolve it
    /// through this one method, so the path the harness plants and the path the
    /// dashboard reads cannot drift apart. If they did, the dashboard would
    /// silently stay on wall time, and the clock-driven scenarios would not fail
    /// outright: they would keep passing, but only by timing.
    pub fn dash_test_clock_file(&self) -> PathBuf {
        self.telemetry_state_dir().join("test-clock-offset")
    }

    /// Log file for the rocm-dash telemetry daemon, under the shared logs dir.
    ///
    /// Deliberately under the canonical `AppPaths` data root
    /// (`~/.rocm/logs/rocmdashd.log`), NOT the legacy standalone rocm-dash XDG
    /// state path (`~/.local/state/rocm-dash/`). D6 unifies the dual-dir split
    /// onto `~/.rocm`; do not "restore" the old XDG location.
    pub fn daemon_log_path(&self) -> PathBuf {
        self.data_dir.join("logs").join("rocmdashd.log")
    }

    /// Directory for client-side (CLI/TUI process) `tracing` log files.
    ///
    /// Siblings the daemon's `rocmdashd.log` under the same canonical
    /// `~/.rocm/logs` root; the client writer rotates files inside this
    /// directory itself (see `apps/rocm/src/logging.rs`), so only the
    /// directory — not a single fixed file name — is exposed here.
    pub fn client_log_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }
}

/// The managed root recorded in `config.json` (`setup.therock_venv`), if any.
///
/// Agrees with [`crate::RocmCliConfig::load`] that a missing file means "nothing
/// configured" and an unreadable one is an error. Treating an unreadable file
/// as "nothing configured" would quietly move the data dir — the runtime
/// registry, the active-runtime marker, service records — back to the default,
/// so a `--prefix` install's runtimes would vanish and new ones would land in
/// the wrong place.
///
/// Only what this one field needs is checked: a file that is valid JSON but
/// that `load` rejects for some other field still yields its recorded root.
/// That keeps the data dir right, and leaves the complaint about the other
/// field to `load`, which names it.
fn configured_managed_root_from_config(paths: &AppPaths) -> Result<Option<PathBuf>> {
    let path = paths.config_path();
    let unreadable = |detail: String| {
        anyhow::anyhow!(
            "cannot tell where rocm-cli keeps its data: {} cannot be read ({detail}). Repair \
             that file, or move it aside to start from default settings, or set \
             ROCM_CLI_DATA_DIR to choose the data directory explicitly",
            path.display()
        )
    };
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(unreadable(error.to_string())),
    };
    let value = serde_json::from_slice::<serde_json::Value>(&bytes)
        .map_err(|error| unreadable(format!("not valid JSON: {error}")))?;
    let Some(top) = value.as_object() else {
        return Err(unreadable(format!("expected a JSON object, found {value}")));
    };
    let setup = match top.get("setup") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::Object(setup)) => setup,
        Some(other) => {
            return Err(unreadable(format!(
                "setup should be an object, found {other}"
            )));
        }
    };
    match setup.get("therock_venv") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text.trim())
            .filter(|text| !text.is_empty())
            .map(PathBuf::from)),
        Some(other) => Err(unreadable(format!(
            "setup.therock_venv should be a path, found {other}"
        ))),
    }
}

pub fn engine_plugin_dirs(paths: &AppPaths) -> Vec<PathBuf> {
    vec![
        paths.primary_engine_plugin_dir(),
        paths.data_dir.join("engines"),
    ]
}

// Declared here rather than at the crate root so the property can reach the
// private, env-free `AppPaths::discover_from_paths`.
#[cfg(test)]
#[path = "persisted_paths_props.rs"]
mod persisted_paths_props;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::RocmCliConfig;
    use crate::test_support::temp_app_paths;

    #[test]
    fn audit_events_path_lives_under_data_audit() {
        let (_root, paths) = temp_app_paths("audit-path");
        assert_eq!(
            paths.audit_events_path(),
            paths.data_dir.join("audit").join("events.jsonl")
        );
        assert_eq!(
            paths.automation_proposals_path(),
            paths.data_dir.join("automations").join("proposals.jsonl")
        );
    }
    #[test]
    fn with_managed_root_keeps_reprovisioning_flat_from_runtime_leaf() {
        let data_root = PathBuf::from("/tmp/rocm-cli-reprovision");
        let paths = AppPaths {
            config_dir: data_root.clone(),
            data_dir: data_root.clone(),
            cache_dir: data_root.join("cache"),
        };
        // A prior install persisted the runtime's own install_root as the managed
        // root; rebasing onto it must recover the canonical data root, not append
        // a second `runtimes/wheel` when the next runtime is provisioned.
        let leaf = data_root
            .join("runtimes")
            .join("wheel")
            .join("release-wheel-gfx942-7-0");
        let rebased = paths.with_managed_root(leaf, false);

        assert_eq!(rebased.data_dir, data_root);
        let next_root = rebased
            .data_dir
            .join("runtimes")
            .join("wheel")
            .join("nightly-wheel-gfx942-7-1");
        // Count path components, not a literal separator, so the assertion holds
        // on Windows too.
        let runtimes_segments = next_root
            .components()
            .filter(|component| component.as_os_str() == std::ffi::OsStr::new("runtimes"))
            .count();
        assert_eq!(runtimes_segments, 1);
    }
    #[test]
    fn app_paths_apply_configured_managed_root_when_unoverridden() -> Result<()> {
        let (root, paths) = temp_app_paths("configured-managed-root");
        let managed_root = root.join("managed");
        let persisted_runtime = managed_root
            .join("runtimes")
            .join("wheel")
            .join("release-wheel-gfx942-7-0");
        fs::create_dir_all(&paths.config_dir)?;
        fs::write(
            paths.config_path(),
            serde_json::to_vec_pretty(&serde_json::json!({
                "setup": { "therock_venv": persisted_runtime }
            }))?,
        )?;

        let discovered = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(discovered.config_dir, paths.config_dir);
        assert_eq!(discovered.data_dir, managed_root);
        assert_eq!(discovered.cache_dir, managed_root.join("cache"));

        let data_overridden = AppPaths::discover_from_paths(paths.clone(), true, false)?;
        assert_eq!(data_overridden.data_dir, paths.data_dir);
        assert_eq!(data_overridden.cache_dir, paths.cache_dir);

        let cache_overridden = AppPaths::discover_from_paths(paths.clone(), false, true)?;
        assert_eq!(cache_overridden.data_dir, managed_root);
        assert_eq!(cache_overridden.cache_dir, paths.cache_dir);

        fs::remove_dir_all(root).ok();
        Ok(())
    }
    /// An unreadable `config.json` stops discovery with a message naming the
    /// file and the ways out, and discovery leaves the file alone. Each way
    /// out the message names is then shown to clear the condition.
    #[test]
    fn app_paths_refuse_to_guess_the_data_dir_from_an_unreadable_config() -> Result<()> {
        let (root, paths) = temp_app_paths("configured-managed-root-unreadable");
        let managed_root = root.join("managed");
        let mut config = RocmCliConfig::default();
        config.setup.therock_venv = Some(managed_root.clone());
        config.save(&paths)?;
        let saved = fs::read(paths.config_path())?;

        for (label, damaged) in [
            ("truncated", saved[..saved.len() / 2].to_vec()),
            ("not an object", b"[]".to_vec()),
            ("setup retyped", br#"{"setup": "x"}"#.to_vec()),
            (
                "therock_venv retyped",
                br#"{"setup": {"therock_venv": 7}}"#.to_vec(),
            ),
        ] {
            fs::write(paths.config_path(), &damaged)?;
            let error = AppPaths::discover_from_paths(paths.clone(), false, false)
                .expect_err("discovery must not guess the data dir");
            let message = format!("{error:#}");
            assert!(
                message.contains("cannot tell where rocm-cli keeps its data")
                    && message.contains("config.json")
                    && message.contains("ROCM_CLI_DATA_DIR"),
                "{label}: unexpected error: {message}"
            );
            assert_eq!(
                fs::read(paths.config_path())?,
                damaged,
                "{label}: discovery changed config.json"
            );

            // Way out: choose the data dir explicitly.
            let overridden = AppPaths::discover_from_paths(paths.clone(), true, false)?;
            assert_eq!(overridden.data_dir, paths.data_dir, "{label}");
        }

        // Way out: repair the file.
        fs::write(paths.config_path(), &saved)?;
        let repaired = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(repaired.data_dir, managed_root);

        // Way out: move it aside. A missing config is "nothing configured".
        fs::rename(paths.config_path(), root.join("config.json.bak"))?;
        let moved_aside = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(moved_aside.data_dir, paths.data_dir);

        // A file that is valid JSON but that `load` rejects for another field
        // still names the right data dir; `load` reports that field itself.
        let mut value = serde_json::from_slice::<serde_json::Value>(&saved)?;
        value["dashboard"]["tui"]["chat_top_p"] = serde_json::json!(1.5);
        fs::write(paths.config_path(), serde_json::to_vec(&value)?)?;
        assert!(RocmCliConfig::load(&paths).is_err());
        let other_field_bad = AppPaths::discover_from_paths(paths, false, false)?;
        assert_eq!(other_field_bad.data_dir, managed_root);

        fs::remove_dir_all(root).ok();
        Ok(())
    }
    #[test]
    fn engine_envs_dir_honors_dedicated_root_override() {
        let (root, paths) = temp_app_paths("engine-envs-root-override");
        let override_root = root.join("runtime").join("engines");

        // Passed in rather than set via `std::env::set_var`: the environment is
        // process-global, so mutating it here would race any concurrent test
        // that reads the same key under a threaded runner.
        //
        // What this assertion covers is the override being consulted and the
        // `<root>/<engine>/envs` shape built on it. It deliberately does NOT
        // claim to cover the host-normalisation step, and routing the expected
        // value through the same call the production code makes is why: on a
        // non-Windows target `normalize_runtime_path_for_host` returns its
        // input unchanged (see `normalize_runtime_path_text_for_platform`) for
        // any path these tests can produce, so dropping normalisation from the
        // override arm cannot fail this on a Linux lane — nor, with an
        // already-normal temp path, on a Windows one. Not the identity in the
        // strict sense: the helper round-trips through `Path::display()`, which
        // is lossy for a non-UTF-8 path. Temp paths here are UTF-8, so the
        // distinction does not reach this assertion, but it is not a no-op.
        // Making it fail would need the platform threaded through the seam, and
        // the normaliser itself is already covered on every host by
        // `runtime_path_normalization_accepts_windows_drive_forms` and its
        // neighbours, which pass the platform in explicitly.
        assert_eq!(
            paths.engine_envs_dir_from("vllm", Some(&override_root)),
            normalize_runtime_path_for_host(&override_root)
                .join("vllm")
                .join("envs")
        );

        fs::remove_dir_all(root).ok();
    }
    /// Serializes tests that replace a process-global env var while they run.
    ///
    /// Named `*_TEST_LOCK` so the env-mutation contract guard recognises the
    /// discipline by suffix rather than by a hardcoded list of lock names.
    static ENGINE_ENVS_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    /// The two tests around this one drive the seam, which deliberately does
    /// not read the environment — so on their own the production lookup, and
    /// the key it names, could both be deleted without failing anything. This
    /// drives `engine_envs_dir` itself against a real variable.
    #[test]
    fn engine_envs_dir_reads_its_root_from_the_environment() {
        let _guard = ENGINE_ENVS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, paths) = temp_app_paths("engine-envs-root-env");
        let override_root = root.join("runtime").join("engines");

        // Restored on drop rather than on the next line, so a panic inside
        // `engine_envs_dir` cannot leave this key pointing at the directory
        // removed below. `RestoredEnvVar` only restores; the lock above is what
        // serializes, and the contract guard requires it here because
        // `RestoredEnvVar::set(` is in its mutation list.
        let restore =
            crate::test_env::RestoredEnvVar::set("ROCM_CLI_ENGINE_ENVS_ROOT", &override_root);
        let resolved = paths.engine_envs_dir("vllm");
        drop(restore);

        // Same scope as the seam test above: this pins that `engine_envs_dir`
        // reaches the variable, not that the value is host-normalised on the
        // way through. See that test for why the normalisation step is not
        // observable here.
        assert_eq!(
            resolved,
            normalize_runtime_path_for_host(&override_root)
                .join("vllm")
                .join("envs"),
            "engine_envs_dir must reach $ROCM_CLI_ENGINE_ENVS_ROOT"
        );

        fs::remove_dir_all(root).ok();
    }
    #[test]
    fn engine_envs_dir_falls_back_to_the_data_dir_without_an_override() {
        // The other half of the override contract: with nothing supplied the
        // root is the data dir, which is what the unset-environment production
        // path resolves to.
        let (root, paths) = temp_app_paths("engine-envs-root-default");

        assert_eq!(
            paths.engine_envs_dir_from("vllm", None),
            paths.data_dir.join("engines").join("vllm").join("envs")
        );

        fs::remove_dir_all(root).ok();
    }
    #[test]
    fn engine_plugin_dirs_are_data_owned_and_ordered() {
        let (_root, paths) = temp_app_paths("engine-plugin-dirs");

        assert_eq!(
            engine_plugin_dirs(&paths),
            vec![
                paths.primary_engine_plugin_dir(),
                paths.data_dir.join("engines")
            ]
        );
    }
    #[test]
    fn app_paths_expose_telemetry_and_daemon_log_paths() -> Result<()> {
        let (root, paths) = temp_app_paths("telemetry-paths");
        assert_eq!(
            paths.telemetry_state_dir(),
            paths.data_dir.join("telemetry")
        );
        assert_eq!(
            paths.daemon_log_path(),
            paths.data_dir.join("logs").join("rocmdashd.log")
        );
        assert_eq!(paths.client_log_dir(), paths.data_dir.join("logs"));
        // ensure() creates the telemetry state dir alongside the others.
        paths.ensure()?;
        assert!(paths.telemetry_state_dir().is_dir());
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }
}
