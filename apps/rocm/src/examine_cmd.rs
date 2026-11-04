// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm examine`: the command handler (`examine`), which prints the human
//! report through `examine_report_text.rs`'s `examine_human_report`, or with
//! `--json` the `ExamineJson` document built here — the `Examination`
//! verbatim plus an `ExamineJsonSummary` of what the CLI knows but the probe
//! cannot (`ExamineJson`, `ExamineJsonSummary`).
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `examine` is `pub(crate)`: `dispatch()` calls it
//! through the re-import `use crate::examine_cmd::examine;`. The two document
//! types have no user outside this module and stay private. There is no test
//! in this crate that calls `examine` or builds the document directly.

use anyhow::Result;
use rocm_core::{AppPaths, ExamineSummary, RocmCliConfig};

use crate::examine_report_text::examine_human_report;
use crate::runtimes_cmd::{current_runtime_manifest, setup_runtime_paths};
use crate::therock;

/// The `--json` document.
///
/// `Examination` stays at the top level, verbatim: its field names deliberately
/// mirror `examine.py`'s dataclass so the JSON contract is identical, and moving
/// them would break every consumer written against it.
///
/// Everything the CLI knows but the probe cannot is added under `summary`. The
/// probe runs before any config is read, so it has no idea which engine this
/// host will serve on, whether an existing ROCm install was found, or where the
/// CLI keeps its files — all of which the human report prints. A tool reading
/// `--json` used to have to scrape that text to find out.
#[derive(serde::Serialize)]
struct ExamineJson<'a> {
    #[serde(flatten)]
    examination: &'a rocm_core::Examination,
    summary: ExamineJsonSummary<'a>,
}

/// The human report's own view, plus the few facts it derives from config while
/// rendering rather than storing on the summary.
#[derive(serde::Serialize)]
struct ExamineJsonSummary<'a> {
    #[serde(flatten)]
    host: &'a ExamineSummary,
    /// What the user configured — `None` means unset, exactly as the text form's
    /// `<platform default>` does. Distinct from `effective_default_engine`.
    configured_default_engine: Option<&'a str>,
    /// What `serve` will actually pick: the configured engine if there is one,
    /// else the host default. Mirrors `select_serve_engine`.
    effective_default_engine: &'a str,
    active_runtime_id: Option<&'a str>,
    active_runtime_key: Option<&'a str>,
    previous_runtime_key: Option<&'a str>,
    /// Where the active runtime lives, completing the `id`/`key`/`root` triple.
    /// Not derivable from the key: `install_root` is its own field on the
    /// manifest, and `install sdk --prefix`, `runtimes adopt` and `runtimes
    /// import` all set it freely.
    active_runtime_root: Option<String>,
    /// The folder setup was configured for, straight from config. Distinct from
    /// `active_runtime_root`, not a fallback for it: setup's folder can be a
    /// stale or removed install while a different runtime is active. Reachable
    /// without the registry, which is what makes it worth carrying — see
    /// `recover_setup_runtime_registration`, the write `--json` will not do.
    setup_runtime_root: Option<String>,
    /// The text form's sibling fact, derived from `setup_runtime_root` rather
    /// than stored, so it cannot drift from where the pip cache actually goes.
    setup_runtime_pip_cache_dir: Option<String>,
}

pub(crate) fn examine(json: bool, framework: rocm_core::FrameworkProbe) -> Result<()> {
    // `rocm examine` is the general system inspector: the exit code reports
    // whether it RAN, not what it found. Any finding (no GPU, WSL, degraded) is
    // surfaced in the output and the `--json` `status` field, and the command
    // exits 0; a genuine inability to examine propagates as an error via `?`.
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    if json {
        // Prefer the active runtime's interpreter: in the managed configuration
        // torch lives only in its site-packages, so probing PATH would report
        // `unknown` for a host that has one.
        let interpreter = rocm_core::active_managed_framework_interpreter(&paths, &config);
        let examination =
            rocm_core::Examination::probe_with_interpreter(framework, interpreter.as_ref());
        // `gather` rather than `examine_human_report`: the latter first runs
        // `recover_setup_runtime_registration`, which writes. Asking a machine a
        // question should not change it, and `--json` is the form tooling calls
        // in a loop.
        let host = ExamineSummary::gather()?;
        let configured_default_engine = config.default_engine.as_deref();
        // Same resolution the human report uses, minus the recovery write above:
        // an unreadable registry leaves the root `null` rather than failing the
        // inspection, which is the weaker answer but still an answer.
        let manifests = therock::load_runtime_manifests(&paths).unwrap_or_default();
        let active_runtime_root = current_runtime_manifest(&config, &manifests)
            .map(|manifest| manifest.install_root.display().to_string());
        let (setup_runtime_root, setup_runtime_pip_cache_dir) = setup_runtime_paths(&config)
            .map(|(root, cache)| (root.display().to_string(), cache.display().to_string()))
            .unzip();
        let document = ExamineJson {
            examination: &examination,
            summary: ExamineJsonSummary {
                configured_default_engine,
                effective_default_engine: configured_default_engine.unwrap_or(&host.default_engine),
                active_runtime_id: config.default_runtime_id.as_deref(),
                active_runtime_key: config.active_runtime_key.as_deref(),
                previous_runtime_key: config.previous_runtime_key.as_deref(),
                active_runtime_root,
                setup_runtime_root,
                setup_runtime_pip_cache_dir,
                host: &host,
            },
        };
        println!("{}", serde_json::to_string_pretty(&document)?);
        return Ok(());
    }
    let (text, summary) = examine_human_report(&paths, &config)?;
    print!("{text}");
    if summary.wsl.as_ref().is_some_and(|w| w.is_wsl) {
        // Informational platform guidance for humans (the verdict is also in
        // the `status` field for `--json` consumers).
        println!("\n{}", rocm_core::WSL_PLATFORM_NOTE);
    }
    Ok(())
}
