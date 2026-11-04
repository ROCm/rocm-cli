// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Examine and engine-inventory rendering: the `rocm examine` report (both
//! its JSON-adjacent text path and its plain-language human report) and the
//! `rocm engines`/`examine` engine-inventory listing.
//!
//! Relocated from `main.rs` with no behavior change (see
//! `docs/architecture.md`). The `rocm model` catalog renderer used to live
//! here too; it moved to `model_catalog_text.rs`, which this module does not
//! call into outside its own tests.
//!
//! This module owns: the launch-summary/examine text surfaces
//! (`render_launch_summary`, `render_examine_text`,
//! `render_examine_text_with_paths`, `examine_human_report`,
//! `render_examine_plain_header`, `plain_status_label`) and the
//! engine-inventory listing shared by `rocm engines` and `rocm examine`
//! (`write_default_engine_legend`, the `DEFAULT_ENGINE_MARKER` constant,
//! `render_engine_inventory_text_with_paths`, `append_examine_runtime_state`,
//! `append_examine_engine_inventory`, `append_engine_detect_summary`,
//! `friendly_engine_runtime_kind`, `friendly_engine_detect_notes`,
//! `friendly_engine_detect_note_fallback`, `engine_runtime_status_label` and
//! `runtime_status_label_for`). `DEFAULT_ENGINE_MARKER` is shared by both
//! engine-inventory renderers so the two copies cannot drift apart.
//!
//! Callers: `examine_cmd.rs`'s `examine()`, which the `Command::Examine` arm
//! calls, imports and calls `examine_human_report`; `main.rs`'s own
//! `launch_default` calls `render_launch_summary`, re-imported there.
//! `render_examine_text`, `render_examine_text_with_paths` and
//! `render_engine_inventory_text_with_paths` are `pub(crate)`, read from
//! [`chat.rs`](../apps/rocm/src/chat.rs)'s deterministic tool-output
//! summaries; `render_engine_inventory_text_with_paths` is also read from
//! [`engines_cmd.rs`](../apps/rocm/src/engines_cmd.rs)'s
//! `render_engine_inventory_text`. `append_examine_runtime_state` is also
//! `pub(crate)`, read only from
//! [`runtimes_cmd.rs`](../apps/rocm/src/runtimes_cmd.rs)'s own test module, so
//! `main.rs`'s re-import of it is itself `#[cfg(test)]`, since it would
//! otherwise be an unused import in a non-test build. Every other item in this
//! list has no caller outside this module and stays private.
//!
//! This module reaches three items that live at the crate root because they
//! have other callers: `engine_inventory()` (the built-in engine list, also
//! read from `update_cmd.rs`), `PlatformRuleOut` (which
//! `engine_runtime_status_label` matches on) and
//! `engine_ruled_out_by_platform` (both defined in `model_fit.rs` and
//! re-imported in `main.rs`), and `missing_packaged_engine_reason` (also read
//! from `model_catalog_text.rs`) — all reached via `use crate::{...};`.
//!
//! The regression tests for these surfaces live in this module's own
//! `mod tests`, reaching the shared `test_paths` fixture through
//! `crate::tests::` and `write_test_pip_runtime` through
//! `crate::test_support::`. One of them,
//! `linux_only_engine_runtime_status_is_explicit_on_native_windows`, also
//! calls `model_catalog_text.rs`'s `model_registry_adapter_availability_note`,
//! which is `pub(crate)` for that reason; its primary subject is
//! `engine_runtime_status_label`, so it stays here.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Result;
use rocm_core::{
    AppPaths, ExamineSummary, RocmCliConfig, engine_plugin_dirs, managed_pip_cache_dir,
};
use rocm_engine_protocol::{DetectRequest, DetectResponse, EngineMethod};

use crate::cli_report::write_marked_row;
use crate::engine_ipc::{
    builtin_engine_available, engine_request, resolve_engine_binary_path,
    resolve_engine_binary_path_with_paths,
};
use crate::freeform_planner;
use crate::runtimes_cmd::{
    current_runtime_manifest, default_runtime_id_matches, recover_setup_runtime_registration,
    runtime_keys_text, runtime_usability_status, setup_runtime_paths, toolchain_state_text,
};
use crate::therock;
use crate::{
    PlatformRuleOut, engine_inventory, engine_ruled_out_by_platform, missing_packaged_engine_reason,
};

pub(crate) fn render_launch_summary(paths: &AppPaths, config: &RocmCliConfig) -> String {
    // User-facing text, so it must agree with what `serve` would pick — the same
    // contradiction `examine` used to show on Instinct.
    let selected_default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or_else(|| freeform_planner::host_default_engine(Some(paths)));
    let mut output = String::new();
    let _ = writeln!(output, "rocm interactive shell");
    let _ = writeln!(output, "  terminal: non-interactive");
    let _ = writeln!(output, "  default engine: {selected_default_engine}");
    let _ = writeln!(
        output,
        "  default runtime: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active runtime key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(output, "  config dir: {}", paths.config_dir.display());
    let _ = writeln!(output, "  config file: {}", paths.config_path().display());
    let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    let _ = writeln!(output, "  cache dir: {}", paths.cache_dir.display());
    let _ = writeln!(
        output,
        "  note: launch from an interactive terminal to enter the TUI."
    );
    output
}

pub(crate) fn render_examine_text() -> Result<String> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    render_examine_text_with_paths(&paths, &config)
}

pub(crate) fn render_examine_text_with_paths(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<String> {
    Ok(examine_human_report(paths, config)?.0)
}

/// Build the human examine report and return it alongside the `ExamineSummary`
/// it was built from, so callers can derive scope/exit-code without re-probing.
pub(crate) fn examine_human_report(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<(String, ExamineSummary)> {
    recover_setup_runtime_registration(paths, config)?;
    let summary = ExamineSummary::gather()?;
    let mut output = render_examine_plain_header(&summary);
    output.push_str(&summary.render_text());
    append_examine_runtime_state(&mut output, paths, config)?;
    append_examine_engine_inventory(&mut output, paths, config, &summary.default_engine);
    Ok((output, summary))
}

fn render_examine_plain_header(summary: &ExamineSummary) -> String {
    let gpu = if summary.detected_gfx_target.is_some() {
        "AMD GPU detected"
    } else {
        "AMD GPU not detected yet"
    };
    // This line counted only CLI-managed runtimes, so a machine with ROCm
    // already installed was greeted with "No ROCm installs saved yet" -- read,
    // reasonably, as "nothing is installed". Now that the resolver finds those
    // installs and their versions, say what is actually on the machine, and that
    // a pre-existing install is left alone on purpose rather than missed.
    let managed = match summary.managed_runtime_count {
        0 => None,
        1 => Some("1 ROCm install saved".to_owned()),
        count => Some(format!("{count} ROCm installs saved")),
    };
    let existing = (summary.legacy_rocm.status != "not_detected").then(|| {
        summary.legacy_rocm.version.as_deref().map_or_else(
            || "existing ROCm install found, left unmanaged by design".to_owned(),
            |version| format!("existing ROCm {version} found, left unmanaged by design"),
        )
    });
    let runtime = match (managed, existing) {
        (Some(managed), Some(existing)) => format!("{managed}; {existing}"),
        (Some(managed), None) => managed,
        (None, Some(existing)) => format!("No ROCm installs saved yet; {existing}"),
        (None, None) => "No ROCm installs saved yet".to_owned(),
    };
    format!(
        "ROCm setup check\n  {gpu}\n  {runtime}\n  Driver: {}\n\nDetails\n",
        plain_status_label(&summary.driver.status)
    )
}

fn plain_status_label(status: &str) -> String {
    status.replace('_', " ")
}

/// Marker shown beside the engine `serve`/CLI commands default to. Every
/// place that renders this glyph MUST use this constant so the rendered
/// character and the legend text stay in sync.
const DEFAULT_ENGINE_MARKER: &str = "*";

/// Writes the legend line explaining [`DEFAULT_ENGINE_MARKER`]. Shared by both
/// engine-inventory renderers so the two copies cannot drift apart; their rows
/// share [`write_marked_row`] for the same reason.
fn write_default_engine_legend(output: &mut String) {
    let _ = writeln!(output, "  legend: {DEFAULT_ENGINE_MARKER} = default engine");
}

pub(crate) fn render_engine_inventory_text_with_paths(paths: Option<&AppPaths>) -> String {
    // Mark the engine this GPU actually serves on as primary. Using the platform
    // constant put the `*` on Lemonade even on Instinct, where `serve` picks vLLM.
    let host_gpu = rocm_core::detect_host_gpu_summary(paths);
    let host_default_engine = rocm_core::default_engine_for_host(&host_gpu);
    // A configured `default_engine` still wins over the host preference, mirroring
    // `select_serve_engine` (and `append_examine_engine_inventory`). Without this,
    // the legend could mark an engine `serve` will not actually default to.
    let configured_default_engine = paths.and_then(|paths| {
        RocmCliConfig::load(paths)
            .ok()
            .and_then(|config| config.default_engine)
    });
    // Mirrors `select_serve_engine`'s guard: a blank configured value must not
    // out-rank the host preference, or the marker would land on an engine name
    // that is empty rather than falling back.
    let default_engine = configured_default_engine
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(host_default_engine);
    // A configured default naming an external plugin (or a stale/typo'd name)
    // won't appear in `engine_inventory()`'s built-ins list below — printing
    // the legend in that case would explain a marker that lands on zero rows.
    let any_marked = engine_inventory()
        .iter()
        .any(|(name, _)| *name == default_engine);
    let mut output = String::new();
    let _ = writeln!(output, "Local model engines");
    let _ = writeln!(
        output,
        "  Built-in engines are included with rocm-cli. External plugins are optional."
    );
    let _ = writeln!(output, "  ROCm GPU execution is required.");
    if let Some(paths) = paths {
        let _ = writeln!(output, "  Plugin folders:");
        for (index, path) in engine_plugin_dirs(paths).iter().enumerate() {
            let note = if index == 0 { "primary" } else { "legacy" };
            let _ = writeln!(output, "    {}. {} ({note})", index + 1, path.display());
        }
    } else {
        let _ = writeln!(output, "  Plugin folders: not checked");
    }
    if any_marked {
        write_default_engine_legend(&mut output);
    }
    for (name, note) in engine_inventory() {
        let marker = (*name == default_engine).then_some(DEFAULT_ENGINE_MARKER);
        write_marked_row(&mut output, "", marker, format_args!("{name:10} {note}"));
        append_engine_detect_summary(&mut output, name, paths);
    }
    let _ = writeln!(
        output,
        "  protocol: {}",
        rocm_engine_protocol::ENGINE_PROTOCOL_VERSION
    );
    output
}

pub(crate) fn append_examine_runtime_state(
    output: &mut String,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<()> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let active = current_runtime_manifest(config, &manifests);
    let default_runtime_matches = default_runtime_id_matches(config, &manifests);
    let ambiguous_default_keys =
        if config.active_runtime_key.is_none() && default_runtime_matches.len() > 1 {
            Some(runtime_keys_text(&default_runtime_matches))
        } else {
            None
        };
    let _ = writeln!(output, "runtime_state:");
    let _ = writeln!(
        output,
        "  active_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let active_status = match active {
        Some(manifest) => runtime_usability_status(manifest),
        None if ambiguous_default_keys.is_some() => "ambiguous_runtime_id".to_owned(),
        None if config.active_runtime_key.is_some() || config.default_runtime_id.is_some() => {
            "missing_manifest".to_owned()
        }
        None => "unset".to_owned(),
    };
    let _ = writeln!(output, "  active_runtime_status: {active_status}");
    if let Some(keys) = ambiguous_default_keys {
        let _ = writeln!(output, "  active_runtime_matches: {keys}");
        let _ = writeln!(
            output,
            "  active_runtime_action: rocm runtimes activate <runtime_key>"
        );
    }
    if let Some(manifest) = active {
        let _ = writeln!(
            output,
            "  active_runtime_root: {}",
            manifest.install_root.display()
        );
        let pip_cache_dir = manifest
            .pip_cache_dir
            .clone()
            .unwrap_or_else(|| managed_pip_cache_dir(&manifest.install_root));
        let _ = writeln!(
            output,
            "  active_runtime_pip_cache_dir: {}",
            pip_cache_dir.display()
        );
        let _ = writeln!(
            output,
            "  active_runtime_version: {}",
            therock::runtime_version_display(&manifest.version)
        );
        let _ = writeln!(output, "  active_runtime_family: {}", manifest.family);
        // The same fact `rocm runtimes list` reports as `toolchain=`, for the
        // one runtime that is actually in use. `examine` is where a user looks
        // when a build fails on a missing `hipcc`, and without this line the
        // command that exists to answer "what is my ROCm state" could not say
        // whether the active runtime has a compiler at all.
        let _ = writeln!(
            output,
            "  active_runtime_toolchain: {}",
            toolchain_state_text(manifest.includes_devel())
        );
        let mode = if manifest.read_only {
            "read-only"
        } else {
            "managed"
        };
        let _ = writeln!(output, "  active_runtime_mode: {mode}");
    }
    if let Some((setup_root, pip_cache_dir)) = setup_runtime_paths(config) {
        let _ = writeln!(output, "  setup_runtime_root: {}", setup_root.display());
        let _ = writeln!(
            output,
            "  setup_runtime_pip_cache_dir: {}",
            pip_cache_dir.display()
        );
    }
    let keys = if manifests.is_empty() {
        "<none>".to_owned()
    } else {
        manifests
            .iter()
            .map(|manifest| manifest.runtime_key.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(output, "  registered_runtime_keys: {keys}");
    Ok(())
}

/// `host_default_engine` is the engine this GPU serves on, already resolved by
/// the caller from the `ExamineSummary` it holds — passed in rather than
/// re-probed so the two halves of one `rocm examine` run cannot disagree.
fn append_examine_engine_inventory(
    output: &mut String,
    paths: &AppPaths,
    config: &RocmCliConfig,
    host_default_engine: &str,
) {
    // Mirrors `select_serve_engine`'s guard: a blank configured value must not
    // out-rank the host preference, or `effective_default` below would become
    // an engine name that is empty rather than falling back.
    let configured_default = config
        .default_engine
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    // A configured value still wins, mirroring `select_serve_engine`. Only the
    // fallback becomes GPU-aware: it used to be the platform constant, which
    // reported Lemonade on Instinct where serve picks vLLM.
    let effective_default = configured_default.unwrap_or(host_default_engine);
    // A configured default naming an external plugin (or a stale/typo'd name)
    // won't appear in `engine_inventory()`'s built-ins list below — printing
    // the legend in that case would explain a marker that lands on zero rows.
    let any_marked = engine_inventory()
        .iter()
        .any(|(name, _)| *name == effective_default);
    let _ = writeln!(output, "engine_inventory:");
    // Unchanged on purpose: this line means "what the user configured", so an
    // unset value must keep reading as unset rather than borrowing the host default.
    let _ = writeln!(
        output,
        "  configured_default_engine: {}",
        configured_default.unwrap_or("<platform default>")
    );
    let _ = writeln!(output, "  effective_default_engine: {effective_default}");
    // Say so when a configured engine is holding this host off the one its GPU
    // would pick. Installs provisioned before the installer stopped seeding
    // `default_engine` still carry `lemonade` on disk, and nothing distinguishes
    // that seed from a deliberate choice -- so point at the fix rather than
    // rewriting a file the user owns on a guess. This also catches a genuinely
    // stale choice made by hand, which no migration would have covered.
    if let Some(configured) = configured_default
        && configured != host_default_engine
    {
        let _ = writeln!(
            output,
            "  configured_default_note: overrides this host's {host_default_engine} preference; run `rocm config clear-default-engine` to follow the host"
        );
    }
    let _ = writeln!(
        output,
        "  plugin_policy: first-party engines are built in; external data-dir plugins are optional overrides"
    );
    let _ = writeln!(
        output,
        "  external_plugin_policy: optional overrides only; no fallback engine is selected automatically"
    );
    let _ = writeln!(
        output,
        "  plugin_dirs: {}",
        engine_plugin_dirs(paths)
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if any_marked {
        write_default_engine_legend(output);
    }
    for (engine, note) in engine_inventory() {
        let marker = (*engine == effective_default).then_some(DEFAULT_ENGINE_MARKER);
        let adapter = if builtin_engine_available(engine) {
            "built_in".to_owned()
        } else {
            match resolve_engine_binary_path_with_paths(engine, paths) {
                Ok(path) => format!("external path={}", path.display()),
                Err(error) => format!("missing reason={error}"),
            }
        };
        let runtime_pref = config
            .engine_config(engine)
            .and_then(|entry| {
                entry
                    .preferred_runtime_id
                    .as_deref()
                    .or(entry.preferred_env_id.as_deref())
                    .or(entry.last_installed_runtime_id.as_deref())
                    .or(entry.last_installed_env_id.as_deref())
            })
            .unwrap_or("<unset>");
        write_marked_row(
            output,
            "  ",
            marker,
            format_args!("{engine} adapter={adapter} runtime_pref={runtime_pref} note={note}"),
        );
    }
}

fn append_engine_detect_summary(output: &mut String, engine: &str, paths: Option<&AppPaths>) {
    let engine_binary = if builtin_engine_available(engine) {
        Ok(PathBuf::new())
    } else {
        match paths {
            Some(paths) => resolve_engine_binary_path_with_paths(engine, paths),
            None => resolve_engine_binary_path(engine),
        }
    };
    let binary_status = if builtin_engine_available(engine) {
        "adapter: built-in".to_owned()
    } else {
        match &engine_binary {
            Ok(_) => "adapter: available".to_owned(),
            Err(_) => "adapter: not found".to_owned(),
        }
    };
    let _ = writeln!(output, "    {binary_status}");
    if !builtin_engine_available(engine)
        && engine_binary.is_err()
        && let Some(reason) = missing_packaged_engine_reason(engine)
    {
        let _ = writeln!(output, "    note: {reason}");
        return;
    }

    let Ok(detect) = engine_request::<_, DetectResponse>(
        paths,
        engine,
        EngineMethod::Detect,
        &DetectRequest {
            runtime_id: None,
            device_filter: None,
        },
    ) else {
        return;
    };

    let _ = writeln!(
        output,
        "    runtime: {}",
        engine_runtime_status_label(engine, &detect)
    );
    if let Some(kind) = detect.runtime_kind.as_deref() {
        let _ = writeln!(
            output,
            "    runtime kind: {}",
            friendly_engine_runtime_kind(kind)
        );
    }
    if detect.runtime_executable.is_some() {
        let _ = writeln!(output, "    runtime executable: available");
    }
    if let Some(note) = friendly_engine_detect_notes(engine, &detect.notes) {
        let _ = writeln!(output, "    note: {note}");
    }
}

fn friendly_engine_runtime_kind(kind: &str) -> String {
    kind.replace('_', " ")
}

fn friendly_engine_detect_notes(engine: &str, notes: &[String]) -> Option<String> {
    if notes.is_empty() {
        return None;
    }
    let combined = notes.join("; ");
    let lower = combined.to_ascii_lowercase();
    if engine.eq_ignore_ascii_case("lemonade") {
        if lower.contains("not installed") || lower.contains("not found") {
            return Some("Lemonade is not installed yet.".to_owned());
        }
        if lower.contains("lemonade embeddable")
            || lower.contains("llamacpp:rocm")
            || lower.contains("configured")
        {
            return Some("Lemonade is ready on your AMD GPU.".to_owned());
        }
    }
    if engine.eq_ignore_ascii_case("vllm")
        && (lower.contains("not installed") || lower.contains("command was not found"))
    {
        return Some("vLLM is not installed in a Linux/WSL ROCm Python environment.".to_owned());
    }
    if rocm_core::runtime_is_windows()
        && (lower.contains("unsupported_native_windows")
            || lower.contains("native windows")
            || lower.contains("linux/wsl")
            || lower.contains("wsl/linux"))
    {
        return Some(format!("{engine} is available from WSL/Linux on Windows."));
    }
    Some(friendly_engine_detect_note_fallback(&combined))
}

fn friendly_engine_detect_note_fallback(note: &str) -> String {
    let lower = note.to_ascii_lowercase();
    if rocm_core::runtime_is_windows()
        && (lower.contains("unsupported_native_windows")
            || lower.contains("native windows")
            || lower.contains("linux/wsl")
            || lower.contains("wsl/linux"))
    {
        return "This engine is available from WSL/Linux on Windows.".to_owned();
    }
    note.to_owned()
}

/// The `runtime:` label `rocm engines list` prints for an engine.
///
/// `unsupported_native_windows` comes from `engine_ruled_out_by_platform`, the
/// same decision `rocm serve` and `rocm engines install` refuse on
/// (`ensure_engine_allowed_on_platform`), so the inventory cannot call an
/// engine unsupported while those commands go ahead with it, or the reverse.
fn engine_runtime_status_label(engine: &str, detect: &DetectResponse) -> &'static str {
    runtime_status_label_for(detect.installed, engine_ruled_out_by_platform(engine))
}

/// [`engine_runtime_status_label`] with the platform decision passed in rather
/// than re-derived, so every branch is unit-testable on any host (the same seam
/// as `unsupported_here_for` and `platform_refusal`).
fn runtime_status_label_for(installed: bool, ruled_out: PlatformRuleOut) -> &'static str {
    if installed {
        "ready"
    } else if ruled_out == PlatformRuleOut::NativeWindowsVllm {
        "unsupported_native_windows"
    } else {
        "not found"
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::model_catalog_text::model_registry_adapter_availability_note;
    use crate::runtimes_cmd::runtime_manifest_path;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::test_paths;

    /// An `ExamineSummary` with the install-reporting fields under test and
    /// everything else inert.
    fn summary_with_legacy(
        managed_runtime_count: usize,
        legacy_status: &str,
        legacy_version: Option<&str>,
    ) -> ExamineSummary {
        ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "lemonade".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: rocm_core::DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
                windows: None,
            },
            legacy_rocm: rocm_core::LegacyRocmSummary {
                status: legacy_status.to_owned(),
                paths: Vec::new(),
                detail: None,
                version: legacy_version.map(str::to_owned),
            },
            wsl: None,
            managed_runtime_count,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        }
    }

    #[test]
    fn header_does_not_claim_nothing_is_installed_when_rocm_is_present() {
        // The reported case: ROCm on the machine, none of it CLI-managed. The
        // header counted only managed runtimes, so it said "No ROCm installs
        // saved yet" — read as "nothing is installed".
        let header = render_examine_plain_header(&summary_with_legacy(
            0,
            "detected_unmanaged",
            Some("7.14.0"),
        ));
        assert!(
            header.contains("existing ROCm 7.14.0 found"),
            "the detected install and its version must appear:\n{header}"
        );
        assert!(
            header.contains("left unmanaged by design"),
            "it must read as a deliberate choice, not an oversight:\n{header}"
        );
    }

    #[test]
    fn header_reports_a_detected_install_whose_version_is_unknown() {
        // Degrades to naming the install without inventing a version.
        let header =
            render_examine_plain_header(&summary_with_legacy(0, "detected_unmanaged", None));
        assert!(
            header.contains("existing ROCm install found"),
            "the install must still be reported:\n{header}"
        );
        assert!(
            !header.contains("<unknown>"),
            "an unknown version must not leak a placeholder into the summary:\n{header}"
        );
    }

    #[test]
    fn header_accounts_for_managed_and_existing_installs_together() {
        let header = render_examine_plain_header(&summary_with_legacy(
            2,
            "detected_unmanaged",
            Some("6.4.1"),
        ));
        assert!(
            header.contains("2 ROCm installs saved"),
            "the managed count must survive:\n{header}"
        );
        assert!(
            header.contains("existing ROCm 6.4.1 found"),
            "the unmanaged install must be reported alongside it:\n{header}"
        );
    }

    #[test]
    fn header_still_says_nothing_is_installed_when_nothing_is() {
        // The wording only changes when there is something to report.
        let header = render_examine_plain_header(&summary_with_legacy(0, "not_detected", None));
        assert!(
            header.contains("No ROCm installs saved yet"),
            "an empty machine must keep the original wording:\n{header}"
        );
        assert!(
            !header.contains("existing ROCm"),
            "nothing should be claimed on an empty machine:\n{header}"
        );
    }

    #[test]
    fn render_engine_inventory_text_surfaces_external_plugin_policy() {
        let (root, paths) = test_paths("engine-plugin-policy");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Local model engines"));
        assert!(rendered.contains("Built-in engines are included with rocm-cli"));
        assert!(rendered.contains("ROCm GPU execution is required"));
        assert!(rendered.contains("Plugin folders:"));
        assert!(rendered.contains(&paths.primary_engine_plugin_dir().display().to_string()));
    }

    #[test]
    fn render_engine_inventory_text_includes_marker_legend() {
        let (root, paths) = test_paths("engine-inventory-marker-legend");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        let legend = format!("legend: {DEFAULT_ENGINE_MARKER} = default engine");
        let legend_pos = rendered.find(&legend).expect("legend line present");
        // Anchor on the marked row itself (not just `"{DEFAULT_ENGINE_MARKER} "`,
        // which also matches inside the legend line and would pass even if no
        // row were actually marked).
        let marker_pos = rendered
            .find(&format!("{DEFAULT_ENGINE_MARKER} {host_default}"))
            .expect("the default engine entry present");
        assert!(
            legend_pos < marker_pos,
            "legend must appear before the entries it explains:\n{rendered}"
        );
    }

    #[test]
    fn render_engine_inventory_text_keeps_every_row_in_the_marker_layout() {
        let (root, paths) = test_paths("engine-inventory-row-layout");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        for (name, _) in engine_inventory() {
            let expected = if *name == host_default {
                format!("{DEFAULT_ENGINE_MARKER} {name:10} ")
            } else {
                format!("  {name:10} ")
            };
            assert!(
                rendered.lines().any(|line| line.starts_with(&expected)),
                "expected a row starting {expected:?}:\n{rendered}"
            );
        }
    }

    #[test]
    fn render_engine_inventory_text_omits_legend_when_configured_default_matches_nothing() {
        // A configured default naming an external plugin (or a stale/typo'd
        // name) matches zero rows in `engine_inventory()`'s built-ins list —
        // the legend would then explain a marker that appears nowhere.
        let (root, paths) = test_paths("engine-inventory-unmatched-default");
        let config = RocmCliConfig {
            default_engine: Some("totally-unknown-plugin".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths).expect("save config");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(
            !rendered.contains("legend:"),
            "legend must be absent when the configured default matches no built-in engine; got:\n{rendered}"
        );
    }

    #[test]
    fn render_engine_inventory_text_honors_configured_default_engine() {
        // Regression: this renderer used to mark only `default_engine_for_host`,
        // ignoring a configured `default_engine` — the same host-vs-configured
        // precedence `select_serve_engine` and `append_examine_engine_inventory`
        // already honor. Pick whichever engine the host does NOT prefer so the
        // configured value is guaranteed to actually change the marked engine.
        let (root, paths) = test_paths("engine-inventory-configured-default");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));
        let configured = if host_default == "vllm" {
            "lemonade"
        } else {
            "vllm"
        };
        let config = RocmCliConfig {
            default_engine: Some(configured.to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths).expect("save config");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(
            rendered.contains(&format!("{DEFAULT_ENGINE_MARKER} {configured}")),
            "configured default engine {configured} must be marked; got:\n{rendered}"
        );
        assert!(
            !rendered.contains(&format!("{DEFAULT_ENGINE_MARKER} {host_default}")),
            "host default {host_default} must not be marked once a different engine is configured; got:\n{rendered}"
        );
    }

    #[test]
    fn examine_treats_a_blank_configured_engine_as_unset() {
        // Mirrors `select_serve_engine`'s guard: a config file with
        // `default_engine = ""` must fall back to the host preference rather
        // than reporting an empty engine name as "effective" and marking none
        // of the real ones.
        let (root, paths) = test_paths("examine-engine-inventory-blank-configured");
        let config = RocmCliConfig {
            default_engine: Some(String::new()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            output.contains("configured_default_engine: <platform default>"),
            "a blank configured value must read as unset:\n{output}"
        );
        assert!(
            output.contains("effective_default_engine: vllm"),
            "a blank configured value must fall back to the host default:\n{output}"
        );
        assert!(
            output.contains("  * vllm "),
            "the '*' marker must land on the host's default, not an empty name:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn friendly_engine_detect_notes_hide_probe_and_path_noise() {
        let lemonade = friendly_engine_detect_notes(
            "lemonade",
            &["Lemonade embeddable 10.6.0 is installed at D:/ROCm/temp/runtime; Lemonade is configured for llamacpp:rocm; no CPU fallback is used".to_owned()],
        )
        .expect("lemonade note");
        assert_eq!(lemonade, "Lemonade is ready on your AMD GPU.");

        let vllm = friendly_engine_detect_notes(
            "vllm",
            &["vLLM is not installed in a Linux/WSL ROCm Python environment. Native Windows is skipped; no CPU fallback is used.".to_owned()],
        )
        .expect("vllm note");
        assert_eq!(
            vllm,
            "vLLM is not installed in a Linux/WSL ROCm Python environment."
        );
        // Raw install paths from the probe body must not leak into the friendly note.
        assert!(!lemonade.contains("D:/"));
    }

    #[test]
    fn linux_only_engine_runtime_status_is_explicit_on_native_windows() {
        let detect = DetectResponse {
            installed: false,
            env_id: None,
            runtime_kind: Some("external_vllm".to_owned()),
            runtime_executable: None,
            managed_env: Some(false),
            python_version: None,
            torch_version: None,
            transformers_version: None,
            available_devices: vec![rocm_engine_protocol::EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: false,
                reason: Some(
                    "vLLM ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows vLLM is skipped. No CPU fallback is used."
                        .to_owned(),
                ),
            }],
            capabilities: rocm_engine_protocol::EngineCapabilities {
                cpu: false,
                rocm_gpu: false,
                openai_compatible: true,
                tool_calling: false,
                quantized_models: "vllm-supported".to_owned(),
                reasoning_parser: false,
            },
            notes: Vec::new(),
        };

        if cfg!(windows) {
            assert_eq!(
                engine_runtime_status_label("vllm", &detect),
                "unsupported_native_windows"
            );
            assert!(
                model_registry_adapter_availability_note("vllm", &|| true)
                    .is_some_and(|note| note.contains("unsupported_native_windows"))
            );
        } else {
            assert_eq!(engine_runtime_status_label("vllm", &detect), "not found");
            assert!(model_registry_adapter_availability_note("vllm", &|| true).is_none());
        }
    }

    /// The native-Windows branch of the `runtime:` label, exercised with an
    /// injected platform decision so it runs on every host rather than only on
    /// a Windows runner. An engine that is installed stays `ready`; only an
    /// engine that is absent AND ruled out gets the platform label, and an
    /// engine ruled out for another reason (AIM) does not borrow it.
    #[test]
    fn runtime_status_label_names_the_native_windows_rule_out_on_any_host() {
        assert_eq!(
            runtime_status_label_for(false, PlatformRuleOut::NativeWindowsVllm),
            "unsupported_native_windows"
        );
        assert_eq!(
            runtime_status_label_for(true, PlatformRuleOut::NativeWindowsVllm),
            "ready"
        );
        assert_eq!(
            runtime_status_label_for(false, PlatformRuleOut::Allowed),
            "not found"
        );
        assert_eq!(
            runtime_status_label_for(false, PlatformRuleOut::NativeWindowsAim),
            "not found"
        );
        assert_eq!(
            runtime_status_label_for(false, PlatformRuleOut::AimDockerUnavailable),
            "not found"
        );
    }

    #[test]
    fn examine_runtime_state_reports_active_runtime_key_and_status() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-state");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            1,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key),
            previous_runtime_key: Some("older-release".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("runtime_state:"));
        assert!(output.contains("active_runtime_id: therock-release:gfx120X-all"));
        assert!(output.contains("active_runtime_key: release-pip-gfx120x-all"));
        assert!(output.contains("previous_runtime_key: older-release"));
        assert!(output.contains("active_runtime_status: ready"));
        assert!(output.contains("active_runtime_family: gfx120X-all"));
        assert!(output.contains("registered_runtime_keys: release-pip-gfx120x-all"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// `rocm examine` is the first command a user runs when something ROCm is
    /// wrong, and "my build cannot find `hipcc`" is now a reachable state by
    /// design. Reporting the toolchain only from `rocm runtimes list` leaves
    /// the primary diagnostic unable to answer the question the opt-in
    /// created. Both polarities are pinned, because a field hardcoded to either
    /// word reads as working from a single run.
    #[test]
    fn examine_runtime_state_reports_whether_the_active_runtime_has_the_toolchain() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-toolchain");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        // The fixture records `devel: true` with no composition.
        let mut output = String::new();
        append_examine_runtime_state(&mut output, &paths, &config)?;
        assert!(
            output.contains("active_runtime_toolchain: included"),
            "a toolchain runtime must say so:\n{output}"
        );

        // A runtime-only install reports the other way. Written through the
        // recorded specs, which is what `includes_devel` actually reads, so
        // this exercises the same path a real `rocm install sdk` produces.
        let runtime_only = therock::InstalledRuntimeManifest {
            devel: false,
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "canonical".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some("gfx1201".to_owned()),
            }),
            ..manifest
        };
        fs::write(
            runtime_manifest_path(&paths, &runtime_only.runtime_key),
            serde_json::to_vec_pretty(&runtime_only)?,
        )?;
        output.clear();
        append_examine_runtime_state(&mut output, &paths, &config)?;
        assert!(
            output.contains("active_runtime_toolchain: excluded"),
            "a runtime-only runtime must say so:\n{output}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn examine_runtime_state_reports_ambiguous_default_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-ambiguous");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            1,
        )?;
        write_test_pip_runtime(
            &paths,
            "vllm-source-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            2,
        )?;
        let mut config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("active_runtime_status: ambiguous_runtime_id"));
        assert!(output.contains("active_runtime_matches:"));
        assert!(output.contains("release-pip-gfx120x-all"));
        assert!(output.contains("vllm-source-pip-gfx120x-all"));
        assert!(output.contains("active_runtime_action: rocm runtimes activate <runtime_key>"));
        assert!(!output.contains("active_runtime_status: missing_manifest"));

        config.active_runtime_key = Some("release-pip-gfx120x-all".to_owned());
        output.clear();
        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("active_runtime_status: ready"));
        assert!(!output.contains("ambiguous_runtime_id"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn examine_engine_inventory_reports_config_without_engine_detect() {
        let (root, paths) = test_paths("examine-engine-inventory");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.engine_config_mut("vllm").preferred_runtime_id =
            Some("therock-release:gfx120X-all".to_owned());
        let mut output = String::new();

        // Host default deliberately disagrees with the configured value: an
        // engine the user chose must outrank whatever the GPU would prefer.
        append_examine_engine_inventory(&mut output, &paths, &config, "lemonade");

        assert!(output.contains("engine_inventory:"));
        assert!(output.contains("configured_default_engine: vllm"));
        assert!(output.contains("effective_default_engine: vllm"));
        assert!(output.contains("* vllm"));
        assert!(output.contains("runtime_pref=therock-release:gfx120X-all"));
        assert!(output.contains("plugin_dirs:"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_engine_inventory_includes_marker_legend() {
        let (root, paths) = test_paths("examine-engine-inventory-marker-legend");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        let legend = format!("legend: {DEFAULT_ENGINE_MARKER} = default engine");
        let legend_pos = output.find(&legend).expect("legend line present");
        let marker_pos = output
            .find(&format!("{DEFAULT_ENGINE_MARKER} vllm"))
            .expect("the default engine entry present");
        assert!(
            legend_pos < marker_pos,
            "legend must appear before the entries it explains:\n{output}"
        );
    }

    #[test]
    fn examine_engine_inventory_keeps_every_row_in_the_marker_layout() {
        let (root, paths) = test_paths("examine-engine-inventory-row-layout");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        for (engine, _) in engine_inventory() {
            let expected = if *engine == "vllm" {
                format!("  {DEFAULT_ENGINE_MARKER} {engine} adapter=")
            } else {
                format!("    {engine} adapter=")
            };
            assert!(
                output.lines().any(|line| line.starts_with(&expected)),
                "expected a row starting {expected:?}:\n{output}"
            );
        }
    }

    #[test]
    fn examine_engine_inventory_omits_legend_when_effective_default_matches_nothing() {
        // Same gap as `render_engine_inventory_text_with_paths`: an effective
        // default naming an external plugin (or a stale/typo'd name) matches
        // zero rows in `engine_inventory()`'s built-ins list.
        let (root, paths) = test_paths("examine-engine-inventory-unmatched-default");
        let config = RocmCliConfig {
            default_engine: Some("totally-unknown-plugin".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        assert!(
            !output.contains("legend:"),
            "legend must be absent when the effective default matches no built-in engine; got:\n{output}"
        );
    }

    // ---------- engine shell prompt shim ----------

    #[test]
    fn examine_engine_inventory_falls_back_to_the_host_engine_when_unconfigured() {
        // With nothing configured, the reported default must be the engine this
        // GPU actually serves on. On an Instinct host that is vLLM; reporting the
        // platform constant here is what made `examine` contradict `serve`.
        let (root, paths) = test_paths("examine-engine-inventory-host");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        // The configured line still reads as unset — the host default must not be
        // mistaken for something the user chose.
        assert!(output.contains("configured_default_engine: <platform default>"));
        assert!(output.contains("effective_default_engine: vllm"));
        assert!(
            output.contains("* vllm"),
            "the primary marker should follow the host engine:\n{output}"
        );
        // Nothing is being overridden, so there is nothing to warn about.
        assert!(
            !output.contains("configured_default_note"),
            "an unconfigured host must not be warned about an override:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_flags_a_configured_engine_that_overrides_the_host() {
        // Installs provisioned before the installer stopped seeding
        // `default_engine` still carry `lemonade` on disk. That seed is
        // indistinguishable from a deliberate choice, so it is not rewritten --
        // but the user is told why their Instinct box is not on vLLM, and how to
        // change it.
        let (root, paths) = test_paths("examine-engine-inventory-override");
        let config = RocmCliConfig {
            default_engine: Some("lemonade".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            output.contains("effective_default_engine: lemonade"),
            "the configured engine still wins:\n{output}"
        );
        assert!(
            output.contains("configured_default_note:"),
            "the override must be called out:\n{output}"
        );
        assert!(
            output.contains("rocm config clear-default-engine"),
            "the note must name the remedy, not just the problem:\n{output}"
        );
        assert!(
            output.contains("  * lemonade "),
            "the '*' marker must land on the configured engine, not the host's:\n{output}"
        );
        assert!(
            !output.contains("  * vllm "),
            "the host's preferred engine must not also be marked once overridden:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_is_silent_when_the_configured_engine_agrees_with_the_host() {
        let (root, paths) = test_paths("examine-engine-inventory-agrees");
        let config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            !output.contains("configured_default_note"),
            "there is no override to report when the two agree:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }
}
