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
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::runtime::env_path_override;
use crate::runtime::{
    default_cache_dir, default_config_dir, default_data_dir, managed_runtime_cache_dir,
    managed_runtime_data_root, normalize_runtime_path_for_host,
};

/// The `ROCM_CLI_*_DIR` overrides, one field per root, each `None` when unset.
#[derive(Debug, Clone, Default)]
struct PathOverrides {
    config: Option<PathBuf>,
    data: Option<PathBuf>,
    cache: Option<PathBuf>,
}

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
        let paths = Self::from_overrides(PathOverrides {
            config: env_path_override("ROCM_CLI_CONFIG_DIR"),
            data: data_dir_override.clone(),
            cache: cache_dir_override.clone(),
        })?;
        Self::discover_from_paths(
            paths,
            data_dir_override.is_some(),
            cache_dir_override.is_some(),
        )
    }

    /// The three roots from their `ROCM_CLI_*_DIR` overrides, before any
    /// `setup.therock_venv` managed root is applied.
    ///
    /// The cache folder follows a data folder moved by `ROCM_CLI_DATA_DIR` to
    /// `<data>/cache` — the same place it sits by default, `~/.rocm/cache`
    /// under `~/.rocm`, and where [`Self::with_managed_root`] puts it when the
    /// data folder moves to a managed root — unless `ROCM_CLI_CACHE_DIR` names
    /// one of its own. Falling back to the default instead would leave an
    /// install isolated by its config and data folders sharing the default
    /// install's cache, and `rocm uninstall` would delete it.
    ///
    /// Takes the overrides as a named struct so tests can drive it without
    /// mutating the process environment, and so the three optional folders
    /// cannot be transposed.
    fn from_overrides(overrides: PathOverrides) -> Result<Self> {
        let PathOverrides {
            config: config_dir,
            data: data_dir,
            cache,
        } = overrides;
        let cache_dir = cache.or_else(|| data_dir.as_deref().map(managed_runtime_cache_dir));
        Ok(Self {
            config_dir: config_dir
                .or_else(default_config_dir)
                .context("unable to determine config directory for rocm-cli")?,
            data_dir: data_dir
                .or_else(default_data_dir)
                .context("unable to determine data directory for rocm-cli")?,
            cache_dir: cache_dir
                .or_else(default_cache_dir)
                .context("unable to determine cache directory for rocm-cli")?,
        }
        .normalize_for_host())
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

    /// The manifest an engine's installer writes for its installed runtime.
    ///
    /// One definition for the writer (the engine) and for every reader of the
    /// file, so a reader cannot drift to a different name: a reader that
    /// misses it degrades to "no manifest", which is indistinguishable from an
    /// engine that was never installed.
    ///
    /// This is the engine's own install manifest, not an entry in the runtime
    /// registry (that is `apps/rocm`'s `runtime_manifest_path`), and the name keeps
    /// the two apart: the registry has a text guard that treats any file spelling
    /// that helper's name and writing a file as a second registry writer.
    pub fn engine_install_manifest_path(&self, engine: &str) -> PathBuf {
        self.engine_manifests_dir(engine).join("runtime.json")
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
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(unreadable_config_error(&path, &error.to_string(), None)),
    };
    let unreadable = |detail: String| {
        unreadable_config_error(&path, &detail, recorded_root_in_damaged_config(&bytes))
    };
    // One reading for every file: a file `load` accepts holds `setup` as an
    // object (or not at all) and `therock_venv` as a string (or null), and this
    // reads those exactly as `load` does, so a typed parse first would only
    // repeat the work.
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

/// The error for a `config.json` that discovery cannot read, with recovery
/// advice that is true of what each way out actually does:
///
/// - repairing the file restores everything;
/// - `ROCM_CLI_DATA_DIR` only tells discovery where the data is. The file
///   stays broken: every command that changes settings loads it strictly and
///   still stops, and of the read-only ones some stop too while the rest load
///   it leniently and run on default settings. None of the lenient ones saves
///   what it loaded, so none of them overwrites the file;
/// - renaming the file (the message names the exact path to rename and a name
///   to give it; nothing renames it for the user) resets every setting, and
///   puts the data dir back to the default, so runtimes in a custom folder
///   need `ROCM_CLI_DATA_DIR`.
///
/// When the damaged bytes still name the recorded folder, the advice names the
/// exact `ROCM_CLI_DATA_DIR` value that finds it.
fn unreadable_config_error(path: &Path, detail: &str, recorded: Option<PathBuf>) -> anyhow::Error {
    let (data_dir_hint, aside_hint) = match recorded {
        Some(recorded) => {
            let data_root = managed_runtime_data_root(&recorded);
            (
                format!(
                    "It still names {} as the ROCm folder: setting ROCM_CLI_DATA_DIR={} points \
                     commands back at it meanwhile",
                    recorded.display(),
                    data_root.display()
                ),
                runtimes_need_data_dir_override(&data_root),
            )
        }
        None => (
            "Setting ROCM_CLI_DATA_DIR to the folder that holds your runtimes points commands \
             back at it meanwhile"
                .to_owned(),
            "runtimes installed into a custom folder then need ROCM_CLI_DATA_DIR set to that \
             folder"
                .to_owned(),
        ),
    };
    anyhow::anyhow!(
        "cannot tell where rocm-cli keeps its data: {} cannot be read ({detail}). Repair that \
         file to restore your settings. {data_hint}. Until the file is repaired, any command \
         that changes settings still stops on it, and other commands either stop too or run on \
         default settings. Renaming it to {renamed} resets every setting to its default, and \
         {aside_hint}",
        path.display(),
        renamed = config_rename_target(path).display(),
        data_hint = data_dir_hint,
    )
}

/// The data dir discovery settles on when `config.json` records no managed
/// root: `ROCM_CLI_DATA_DIR`, else the default. One function, so the advice
/// about renaming `config.json` cannot disagree with discovery.
fn data_dir_without_config() -> Option<PathBuf> {
    env_path_override("ROCM_CLI_DATA_DIR").or_else(default_data_dir)
}

/// The clause naming the `ROCM_CLI_DATA_DIR` value that keeps `data_root`
/// once `config.json` no longer records it.
fn runtimes_need_data_dir_override(data_root: &Path) -> String {
    format!(
        "the runtimes under {} then also need ROCM_CLI_DATA_DIR={}",
        data_root.display(),
        data_root.display()
    )
}

impl AppPaths {
    /// How to start from default settings by renaming `config.json`, and what
    /// that does to these paths, for a command
    /// that found its data dir through the file but cannot load its settings
    /// (discovery succeeded, `load` rejected another field).
    ///
    /// Every setting goes back to its default. When the data dir came from
    /// the `setup.therock_venv` the file records, it goes back to the default
    /// too, so the advice names the `ROCM_CLI_DATA_DIR` value that keeps it.
    #[must_use]
    pub fn config_aside_advice(&self) -> String {
        let without_config =
            data_dir_without_config().map(|dir| normalize_runtime_path_for_host(&dir));
        config_aside_advice(
            &self.config_path(),
            &self.data_dir,
            without_config.as_deref(),
        )
    }
}

/// The name the recovery advice tells the user to give an unreadable
/// `config.json`: a sibling, so it stays next to the original and is never
/// read as settings again.
fn config_rename_target(config_path: &Path) -> PathBuf {
    let mut name = config_path
        .file_name()
        .map_or_else(|| OsString::from("config.json"), ToOwned::to_owned);
    name.push(".broken");
    config_path.with_file_name(name)
}

/// [`AppPaths::config_aside_advice`], with the data dir discovery falls back
/// to when there is no `config.json` passed in rather than read from the
/// environment.
fn config_aside_advice(
    config_path: &Path,
    data_dir: &Path,
    without_config: Option<&Path>,
) -> String {
    let rename = format!(
        "To start from default settings instead, rename {} to {}",
        config_path.display(),
        config_rename_target(config_path).display()
    );
    if without_config == Some(data_dir) {
        rename
    } else {
        format!("{rename}; {}", runtimes_need_data_dir_override(data_dir))
    }
}

/// Best-effort recovery of `setup.therock_venv` from a `config.json` that does
/// not parse — typically one cut short. Finds the key and reads the JSON string
/// after it only if that string is complete; never guesses past a truncation.
fn recorded_root_in_damaged_config(bytes: &[u8]) -> Option<PathBuf> {
    let text = String::from_utf8_lossy(bytes);
    let after_key = &text[text.find("\"therock_venv\"")? + "\"therock_venv\"".len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let mut escaped = false;
    for (index, ch) in after_colon.char_indices().skip(1) {
        match ch {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '"' => {
                let literal = &after_colon[..=index];
                let value = serde_json::from_str::<String>(literal).ok()?;
                let value = value.trim();
                return (!value.is_empty()).then(|| PathBuf::from(value));
            }
            _ => {}
        }
    }
    None
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

    /// For every `config.json` that `load` accepts, discovery reads the
    /// recorded folder as `load` does: the saved `therock_venv`, trimmed, and
    /// nothing when it is absent or blank.
    #[test]
    fn discovery_reads_the_recorded_folder_of_a_loadable_config_as_load_does() -> Result<()> {
        let (root, paths) = temp_app_paths("discovery-agrees-with-load");
        for recorded in [
            None,
            Some(""),
            Some("   "),
            Some("/data/rocm"),
            Some("  /data/rocm  "),
            Some("/données/rocm"),
        ] {
            let config = RocmCliConfig {
                setup: crate::SetupConfig {
                    therock_venv: recorded.map(PathBuf::from),
                    ..crate::SetupConfig::default()
                },
                ..RocmCliConfig::default()
            };
            config.save(&paths)?;
            let loaded = RocmCliConfig::load(&paths)?;
            let expected = loaded.setup.therock_venv.and_then(|root| {
                let text = root.to_string_lossy().trim().to_owned();
                (!text.is_empty()).then(|| PathBuf::from(text))
            });
            assert_eq!(
                configured_managed_root_from_config(&paths)?,
                expected,
                "recorded {recorded:?}"
            );
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

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
    fn engine_install_manifest_lives_in_the_engine_manifests_dir() {
        let (_root, paths) = temp_app_paths("engine-runtime-manifest-path");
        assert_eq!(
            paths.engine_install_manifest_path("lemonade"),
            paths.engine_manifests_dir("lemonade").join("runtime.json")
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
    /// out the message names is then shown to do what the message says it
    /// does — including what it does *not* fix.
    #[test]
    fn app_paths_refuse_to_guess_the_data_dir_from_an_unreadable_config() -> Result<()> {
        let (root, paths) = temp_app_paths("configured-managed-root-unreadable");
        let managed_root = root.join("managed");
        let mut config = RocmCliConfig::default();
        config.setup.therock_venv = Some(managed_root.clone());
        config.save(&paths)?;
        let saved = fs::read(paths.config_path())?;
        let healthy = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(healthy.data_dir, managed_root);

        let recorded = serde_json::to_string(&managed_root)?;
        let value_at = String::from_utf8_lossy(&saved)
            .find(&recorded)
            .expect("the saved config records the managed root");
        let cut_after_root = saved[..value_at + recorded.len() + 2].to_vec();
        let cut_inside_root = saved[..value_at + recorded.len() / 2].to_vec();
        let suggested = format!("ROCM_CLI_DATA_DIR={}", managed_root.display());
        let rename_advice = format!(
            "Renaming it to {} resets every setting",
            paths.config_dir.join("config.json.broken").display()
        );

        for (label, damaged, names_the_root) in [
            ("cut short after the folder", cut_after_root, true),
            ("cut short inside the folder", cut_inside_root, false),
            ("not an object", br#""x""#.to_vec(), false),
            ("setup retyped", br#"{"setup": "x"}"#.to_vec(), false),
            (
                "therock_venv retyped",
                br#"{"setup": {"therock_venv": 7}}"#.to_vec(),
                false,
            ),
        ] {
            fs::write(paths.config_path(), &damaged)?;
            let error = AppPaths::discover_from_paths(paths.clone(), false, false)
                .expect_err("discovery must not guess the data dir");
            let message = format!("{error:#}");
            for needle in [
                "cannot tell where rocm-cli keeps its data",
                "config.json",
                "Repair that file",
                "any command that changes settings still stops on it",
                "run on default settings",
                rename_advice.as_str(),
            ] {
                assert!(
                    message.contains(needle),
                    "{label}: missing {needle:?} in: {message}"
                );
            }
            assert_eq!(
                message.contains(&suggested),
                names_the_root,
                "{label}: whether the recorded folder is named: {message}"
            );
            assert_eq!(
                fs::read(paths.config_path())?,
                damaged,
                "{label}: discovery changed config.json"
            );

            // ROCM_CLI_DATA_DIR set to the suggested folder finds the same data
            // dir the healthy file did...
            let overridden = AppPaths::discover_from_paths(
                AppPaths {
                    data_dir: managed_root.clone(),
                    ..paths.clone()
                },
                true,
                false,
            )?;
            assert_eq!(overridden.data_dir, healthy.data_dir, "{label}");
            // ...but does not make the settings readable, as the message says.
            assert!(RocmCliConfig::load(&overridden).is_err(), "{label}");
        }

        // A config.json that exists but cannot be read at all (here, a
        // directory: a non-NotFound read error on every platform, no root or
        // chmod needed) is an error too, not "nothing configured".
        fs::remove_file(paths.config_path())?;
        fs::create_dir(paths.config_path())?;
        let error = AppPaths::discover_from_paths(paths.clone(), false, false)
            .expect_err("an unreadable config.json must not read as absent");
        let message = format!("{error:#}");
        assert!(
            message.contains("cannot tell where rocm-cli keeps its data")
                && message.contains("Repair that file"),
            "unexpected error: {message}"
        );
        assert!(paths.config_path().is_dir(), "discovery touched the path");
        // `load` draws the same line: the directory is not "no settings yet".
        assert!(
            RocmCliConfig::load(&paths).is_err(),
            "load read an unreadable config.json as absent"
        );
        fs::remove_dir(paths.config_path())?;

        // Way out: repair the file.
        fs::write(paths.config_path(), &saved)?;
        let repaired = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(repaired.data_dir, managed_root);

        // Way out: rename it to the name the message gives. Settings are back
        // to defaults, and the data dir is back to the default too — which is
        // why the message says a custom folder then needs ROCM_CLI_DATA_DIR.
        fs::rename(
            paths.config_path(),
            paths.config_dir.join("config.json.broken"),
        )?;
        let moved_aside = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(moved_aside.data_dir, paths.data_dir);
        assert_ne!(moved_aside.data_dir, managed_root);
        assert!(
            RocmCliConfig::load(&moved_aside)?
                .setup
                .therock_venv
                .is_none()
        );
        fs::remove_file(paths.config_dir.join("config.json.broken"))?;

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
    /// The text between `start` and the next `end` in `text`.
    fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        let rest = &text[text
            .find(start)
            .unwrap_or_else(|| panic!("no {start:?} in: {text}"))
            + start.len()..];
        rest.find(end).map_or(rest, |at| &rest[..at])
    }

    /// A command that found its data dir through `config.json` but cannot
    /// load the settings in it (another field is bad) is told which file to
    /// rename, to what, and what renaming it does. Following that advice to
    /// the letter — the exact paths it prints — keeps the data dir, while
    /// renaming the file without the override it names would not.
    #[test]
    fn config_rename_advice_keeps_a_recorded_data_dir_when_followed() -> Result<()> {
        let (root, paths) = temp_app_paths("config-aside-advice");
        let managed_root = root.join("managed");
        let mut config = RocmCliConfig::default();
        config.setup.therock_venv = Some(managed_root.clone());
        config.save(&paths)?;
        let mut value =
            serde_json::from_slice::<serde_json::Value>(&fs::read(paths.config_path())?)?;
        value["dashboard"]["tui"]["chat_top_p"] = serde_json::json!(1.5);
        let damaged = serde_json::to_vec(&value)?;
        fs::write(paths.config_path(), &damaged)?;

        let discovered = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(discovered.data_dir, managed_root);
        assert!(RocmCliConfig::load(&discovered).is_err());
        // `paths.data_dir` stands in for the default data dir.
        let advice = config_aside_advice(
            &discovered.config_path(),
            &discovered.data_dir,
            Some(&paths.data_dir),
        );
        assert_eq!(
            advice,
            format!(
                "To start from default settings instead, rename {config} to {config}.broken; \
                 the runtimes under {dir} then also need ROCM_CLI_DATA_DIR={dir}",
                config = paths.config_path().display(),
                dir = managed_root.display()
            )
        );
        // The file it names is the one discovery read — under the custom
        // config folder these paths use, not a default location.
        let from = PathBuf::from(between(&advice, "rename ", " to "));
        let to = PathBuf::from(between(&advice, &format!("{} to ", from.display()), ";"));
        let suggested = PathBuf::from(between(&advice, "ROCM_CLI_DATA_DIR=", "\n"));
        assert_eq!(from, paths.config_path());

        fs::rename(&from, &to)?;
        assert_eq!(
            fs::read(&to)?,
            damaged,
            "the renamed file lost its contents"
        );
        // Without the override the data dir moves back to the default...
        let unadvised = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert_eq!(unadvised.data_dir, paths.data_dir);
        // ...and with the value the advice printed it stays where it was, on
        // default settings: the renamed file is not read as settings.
        let advised = AppPaths::discover_from_paths(
            AppPaths {
                data_dir: suggested,
                ..paths.clone()
            },
            true,
            false,
        )?;
        assert_eq!(advised.data_dir, discovered.data_dir);
        assert_eq!(
            RocmCliConfig::load(&advised)?.setup.therock_venv,
            None,
            "the renamed file was still read as settings"
        );
        fs::remove_file(&to)?;

        // A file that records no folder: renaming it leaves the data dir
        // alone, so the advice names no override.
        fs::write(
            paths.config_path(),
            br#"{"dashboard": {"tui": {"chat_top_p": 1.5}}}"#,
        )?;
        let default_root = AppPaths::discover_from_paths(paths.clone(), false, false)?;
        assert!(RocmCliConfig::load(&default_root).is_err());
        let advice = config_aside_advice(
            &default_root.config_path(),
            &default_root.data_dir,
            Some(&paths.data_dir),
        );
        assert_eq!(
            advice,
            format!(
                "To start from default settings instead, rename {config} to {config}.broken",
                config = paths.config_path().display()
            )
        );
        let from = PathBuf::from(between(&advice, "rename ", " to "));
        fs::rename(
            &from,
            between(&advice, &format!("{} to ", from.display()), "\n"),
        )?;
        let renamed = AppPaths::discover_from_paths(paths, false, false)?;
        assert_eq!(renamed.data_dir, default_root.data_dir);
        assert!(RocmCliConfig::load(&renamed).is_ok());

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// The public method reads the same fallback discovery uses: paths whose
    /// data dir is that fallback get no override advice, and any other data
    /// dir is named as the override to set. Reads the environment as it is
    /// rather than setting it, so it cannot race other tests.
    #[test]
    fn config_rename_advice_names_an_override_only_for_a_data_dir_discovery_would_lose() {
        let (root, paths) = temp_app_paths("config-aside-advice-method");
        let rename = format!(
            "To start from default settings instead, rename {config} to {config}.broken",
            config = paths.config_path().display()
        );
        let fallback = AppPaths {
            data_dir: normalize_runtime_path_for_host(
                &data_dir_without_config().expect("this host has a default data dir"),
            ),
            ..paths.clone()
        };
        assert_eq!(fallback.config_aside_advice(), rename);
        assert_eq!(
            paths.config_aside_advice(),
            format!(
                "{rename}; the runtimes under {dir} then also need ROCM_CLI_DATA_DIR={dir}",
                dir = paths.data_dir.display()
            )
        );
        fs::remove_dir_all(root).ok();
    }

    /// The recorded folder is recovered from a damaged file only when its
    /// string is complete: a value cut off part-way is never guessed at.
    #[test]
    fn recorded_root_is_recovered_only_from_a_complete_value() {
        let escaped = br#"{"setup": {"therock_venv": "C:\\ROCm \"x\"\\venv", "completed"#;
        assert_eq!(
            recorded_root_in_damaged_config(escaped),
            Some(PathBuf::from(r#"C:\ROCm "x"\venv"#))
        );
        for damaged in [
            &br#"{"setup": {"therock_venv": "/opt/ro"#[..],
            br#"{"setup": {"therock_venv": 7"#,
            br#"{"setup": {"therock_venv": "  ", "#,
            br#"{"setup": {"#,
        ] {
            assert_eq!(
                recorded_root_in_damaged_config(damaged),
                None,
                "{}",
                String::from_utf8_lossy(damaged)
            );
        }
    }
    #[test]
    fn cache_dir_follows_an_overridden_data_dir() -> Result<()> {
        let (root, _) = temp_app_paths("cache-follows-data");
        let config = root.join("config");
        let data = root.join("data");

        let paths = AppPaths::from_overrides(PathOverrides {
            config: Some(config.clone()),
            data: Some(data.clone()),
            cache: None,
        })?;
        assert_eq!(paths.config_dir, normalize_runtime_path_for_host(&config));
        assert_eq!(paths.data_dir, normalize_runtime_path_for_host(&data));
        assert_eq!(
            paths.cache_dir,
            normalize_runtime_path_for_host(&data.join("cache")),
            "an install isolated by ROCM_CLI_DATA_DIR must not share the default cache"
        );

        // An explicit cache folder still wins over the data folder's.
        let cache = root.join("elsewhere").join("cache");
        let pinned = AppPaths::from_overrides(PathOverrides {
            config: Some(config),
            data: Some(data),
            cache: Some(cache.clone()),
        })?;
        assert_eq!(pinned.cache_dir, normalize_runtime_path_for_host(&cache));
        Ok(())
    }
    #[test]
    fn a_managed_root_does_not_move_a_cache_that_followed_the_data_dir() -> Result<()> {
        // The path `discover()` takes with ROCM_CLI_DATA_DIR set and no cache
        // override, with a `setup.therock_venv` managed root in the config.
        let (root, _) = temp_app_paths("cache-follows-data-managed-root");
        let config = root.join("config");
        let data = root.join("data");
        let managed_runtime = root
            .join("managed")
            .join("runtimes")
            .join("wheel")
            .join("release-wheel-gfx942-7-0");
        fs::create_dir_all(&config)?;
        fs::write(
            config.join("config.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "setup": { "therock_venv": managed_runtime }
            }))?,
        )?;

        let paths = AppPaths::from_overrides(PathOverrides {
            config: Some(config),
            data: Some(data.clone()),
            cache: None,
        })?;
        assert_eq!(
            configured_managed_root_from_config(&paths)?,
            Some(managed_runtime),
            "the fixture must be a config the managed-root lookup reads"
        );
        let discovered = AppPaths::discover_from_paths(paths, true, false)?;
        assert_eq!(discovered.data_dir, normalize_runtime_path_for_host(&data));
        assert_eq!(
            discovered.cache_dir,
            normalize_runtime_path_for_host(&data.join("cache"))
        );
        Ok(())
    }
    #[test]
    fn cache_dir_keeps_its_default_without_a_data_override() -> Result<()> {
        // Only the config folder moved: data and cache both stay at their
        // defaults, so the cache is not pulled under the config folder.
        let (root, _) = temp_app_paths("cache-default");
        let default_cache =
            default_cache_dir().expect("the test host has a home folder for the default cache");
        let paths = AppPaths::from_overrides(PathOverrides {
            config: Some(root.join("config")),
            ..PathOverrides::default()
        })?;
        assert_eq!(
            paths.cache_dir,
            normalize_runtime_path_for_host(&default_cache)
        );
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
