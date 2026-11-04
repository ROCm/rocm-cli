// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Installing ComfyUI: fetching and unpacking its source, installing its Python
//! dependencies into the selected ROCm runtime, and checking the AMD GPU.
//!
//! [`install`] is the `rocm comfyui install` action, including its dry run and
//! the reinstall path that replaces ComfyUI's code (see
//! [`super::source_swap`]) while refusing to do so under a running ComfyUI. The
//! dependency step pins the runtime's own torch stack with a constraints file so
//! a transitive dependency cannot replace it with a CUDA build, and streams uv's
//! output to the install log.
//!
//! The manifest, the log paths and the GPU probe stay in the parent module.

use super::process::{refuse_while_running, running_advice, running_from};
use super::runtime_env::{
    apply_runtime_environment, runtime_environment_from_runtime, select_runtime,
};
use super::{
    APP_ID, APP_NAME, COMFYUI_SOURCE_ARCHIVE_NAME, ComfyUiInstallOptions, ComfyUiManifest,
    ComfyUiProbe, ComfyUiRuntimeEnvironment, capture_configured_command,
    comfyui_source_archive_url, install_log_path_from_app_root, interrupted_reinstall_advice,
    models_folder_for_source, probe_comfyui, quote_log_arg, reused_source_url, runtime_app_root,
    save_manifest, source_path_from_app_root, source_swap,
};
use crate::cli_progress::AnimatedSpinner;
use crate::freeform_planner::format_structured_tool_call;
use crate::therock;
use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use rocm_core::{
    AppPaths, RocmCliConfig, download_file_to_path_with_progress, ensure_uv_binary,
    unix_time_millis, uv_command_env, uv_pip_install_base,
};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, IsTerminal, Read, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

pub(crate) fn install(
    paths: &AppPaths,
    config: &RocmCliConfig,
    options: ComfyUiInstallOptions,
) -> Result<String> {
    paths.ensure()?;
    let runtime = select_runtime(paths, config, options.runtime_id.as_deref())?;
    let app_root = runtime_app_root(&runtime.manifest);
    let source_path = source_path_from_app_root(&app_root);
    let log_path = install_log_path_from_app_root(&app_root);
    let requirements_path = source_path.join("requirements.txt");
    let models_folder = models_folder_for_source(&source_path);
    let runtime_env = runtime_environment_from_runtime(&runtime.manifest, &runtime.python);

    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output, "  action: install")?;
    writeln!(
        output,
        "  ROCm runtime: {}",
        therock::runtime_version_display(&runtime.manifest.version)
    )?;
    writeln!(output, "  folder: {}", source_path.display())?;
    writeln!(output, "  models path: {}", models_folder.display())?;

    // A reinstall that was interrupted part-way through replacing the code is
    // finished by the next install, with or without `--reinstall`, rather than
    // left as a half-replaced tree that would be reused as it is.
    let finishes_interrupted = source_swap::swap_interrupted(&source_path);
    let replaces_existing = source_path.exists() && (options.reinstall || finishes_interrupted);
    if options.dry_run {
        writeln!(output, "  mode: dry-run")?;
        if replaces_existing {
            if finishes_interrupted {
                writeln!(
                    output,
                    "  note: a previous reinstall was interrupted; installing finishes it"
                )?;
            }
            writeln!(
                output,
                "  reinstall: replaces the ComfyUI code in {}",
                source_path.display()
            )?;
            // The real run checks the same thing and refuses.
            let mut warnings = Vec::new();
            if let Some(state) = running_from(paths, &source_path, &mut warnings) {
                writeln!(
                    output,
                    "  note: {}; a reinstall is refused until then",
                    running_advice(&source_path, &state)
                )?;
            }
            for warning in warnings {
                writeln!(output, "  note: {warning}")?;
            }
            // The same list the real run reports as `kept:` — read from the
            // interrupted run's marker when there is one.
            let keeps = source_swap::planned_kept(&source_path);
            if !keeps.is_empty() {
                writeln!(output, "  keeps: {}", keeps.join(", "))?;
            }
            // Which entries those are is only known once the new release is
            // unpacked. `source_swap`'s
            // `swap_replaces_release_code_and_leaves_everything_else_in_place`,
            // `user_entry_named_like_new_release_code_is_set_aside` and
            // `install_from_before_the_record_removes_only_what_the_new_release_ships`
            // prove these lines; comfyui-05 and comfyui-10 the lines the real
            // run prints.
            if source_swap::has_release_record(&source_path) {
                writeln!(
                    output,
                    "  leaves in place: everything else in that folder that the installed release \
                     did not ship"
                )?;
                writeln!(
                    output,
                    "  sets aside: anything of yours named like code the new release adds, \
                     renamed to <name>{}<time>",
                    source_swap::SET_ASIDE_INFIX
                )?;
            } else {
                writeln!(
                    output,
                    "  replaces: anything in that folder named like code the new release ships \
                     (this install has no record of what its release shipped)"
                )?;
                writeln!(output, "  leaves in place: everything else in that folder")?;
            }
        }
        let mut command = vec![
            "comfyui".to_owned(),
            "install".to_owned(),
            "--runtime-id".to_owned(),
            runtime.manifest.runtime_key,
        ];
        if options.reinstall {
            command.push("--reinstall".to_owned());
        }
        writeln!(
            output,
            "  install command: {}",
            format_structured_tool_call("rocm", &command)
        )?;
        return Ok(output);
    }

    // Checked before the download, so a refusal costs nothing, and again
    // just before the code is replaced, so a `rocm comfyui start` during a
    // long download is caught. `start` takes no lock, so one that begins
    // during the swap itself is not; it refuses once the swap's marker exists.
    if replaces_existing {
        let mut warnings = Vec::new();
        let running = running_from(paths, &source_path, &mut warnings);
        for warning in warnings {
            println!("Note: {warning}");
        }
        refuse_while_running(&source_path, running)?;
    }
    let recheck_running = || -> Result<()> {
        if !replaces_existing {
            return Ok(());
        }
        // Any warning about the saved state was printed above.
        refuse_while_running(
            &source_path,
            running_from(paths, &source_path, &mut Vec::new()),
        )
    };

    fs::create_dir_all(
        log_path
            .parent()
            .context("ComfyUI install log path has no parent directory")?,
    )?;
    let mut log = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    writeln!(log, "{APP_NAME} install")?;
    writeln!(log, "runtime_key={}", runtime.manifest.runtime_key)?;
    writeln!(log, "python={}", runtime.python.display())?;

    let mut swap = source_swap::SwapReport::default();
    let mut swapped = false;
    let source_url = if source_path.exists() && !replaces_existing {
        println!("Using existing ComfyUI source folder...");
        let _ = io::stdout().flush();
        writeln!(
            log,
            "Using existing ComfyUI folder at {}.",
            source_path.display()
        )?;
        reused_source_url(paths)
    } else {
        if finishes_interrupted && !options.reinstall {
            println!("Finishing an interrupted ComfyUI reinstall...");
        }
        println!("Downloading ComfyUI source...");
        let _ = io::stdout().flush();
        // The new source is fully downloaded and unpacked before the existing
        // folder is touched, so a failed download or a bad archive leaves the
        // current install as it was. The user's content in it is never moved.
        let (source_url, report) =
            download_and_extract_source(&app_root, &source_path, &mut log, &recheck_running)
                .map_err(|error| {
                    // Whatever stopped the swap part-way, the install for this
                    // runtime is what finishes it.
                    if source_swap::swap_interrupted(&source_path) {
                        error.context(interrupted_reinstall_advice(&runtime.manifest.runtime_key))
                    } else {
                        error
                    }
                })?;
        for warning in &report.warnings {
            println!("Note: {warning}");
        }
        swap = report;
        swapped = true;
        source_url
    };
    // A failure from here on, after new code was put in place, leaves an
    // install that is not finished. The error says how to finish it, and
    // names the entries the swap replaced or set aside, which the success
    // report would have listed.
    let finish = || -> Result<ComfyUiProbe> {
        fs::create_dir_all(&models_folder)
            .with_context(|| format!("failed to create {}", models_folder.display()))?;

        let packages = filtered_requirement_specs(&requirements_path)?;
        writeln!(
            log,
            "Installing {} ComfyUI dependency specs.",
            packages.len()
        )?;
        if !packages.is_empty() {
            println!("Installing ComfyUI dependencies...");
            let _ = io::stdout().flush();
            let uv = ensure_uv_binary(paths)
                .context("failed to acquire uv binary for ComfyUI dependency install")?;
            // Pin the managed runtime's ROCm torch stack to its exact installed
            // versions so a transitive ComfyUI dependency cannot resolve `torch` (and
            // its `nvidia-*` CUDA deps) from PyPI and clobber it (EAI-8051). Filtering
            // torch out of the direct specs is not enough — the resolver reaches it
            // transitively. A constraint fails loudly on a genuine version conflict,
            // which leaves the runtime intact rather than silently corrupting it.
            let constraints =
                torch_stack_constraints(&runtime.python, Some(&runtime_env), &mut log)?;
            let constraints_path = write_torch_constraints(&app_root, &constraints, &mut log)?;
            run_uv_logged_command(
                paths,
                &uv,
                uv_install_args(&runtime.python, &packages, constraints_path.as_deref()),
                Some(&runtime_env),
                &mut log,
                &log_path,
                "install ComfyUI dependencies",
                "Resolving and installing packages with uv…",
            )?;
        }

        println!("Checking AMD GPU access for ComfyUI...");
        let _ = io::stdout().flush();
        let probe = probe_comfyui(&runtime.python, &source_path, Some(&runtime_env))?;
        if !probe.torch_cuda_available {
            bail!("ComfyUI install finished, but the AMD GPU check failed. No CPU mode was used.");
        }
        let manifest = ComfyUiManifest {
            app_id: APP_ID.to_owned(),
            runtime_key: runtime.manifest.runtime_key.clone(),
            runtime_id: runtime.manifest.runtime_id.clone(),
            runtime_version: runtime.manifest.version.clone(),
            runtime_root: runtime.manifest.install_root.clone(),
            python_executable: runtime.python.clone(),
            source_url,
            source_path: source_path.clone(),
            requirements_path,
            pip_cache_dir: None,
            log_path,
            torch_version: probe.torch_version.clone(),
            torch_cuda_available: probe.torch_cuda_available,
            installed_at_unix_ms: unix_time_millis(),
        };
        save_manifest(paths, &manifest)?;
        Ok(probe)
    };
    let probe = finish().map_err(|error| {
        if !swapped {
            return error;
        }
        let changes = swap_changes_summary(&swap)
            .map(|summary| format!(" ({summary})"))
            .unwrap_or_default();
        error.context(format!(
            "the new ComfyUI code is in place{changes}, but the install did not finish; once \
             the cause below is fixed, run `rocm comfyui install --runtime-id {}` to finish it",
            runtime.manifest.runtime_key
        ))
    })?;

    writeln!(output, "  installed: yes")?;
    if !swap.kept.is_empty() {
        writeln!(output, "  kept: {}", swap.kept.join(", "))?;
    }
    if !swap.replaced.is_empty() {
        writeln!(output, "  replaced: {}", swap.replaced.join(", "))?;
    }
    if !swap.set_aside.is_empty() {
        let renamed: Vec<String> = swap
            .set_aside
            .iter()
            .map(|(name, new_name)| format!("{name} (now {new_name})"))
            .collect();
        writeln!(output, "  set aside: {}", renamed.join(", "))?;
    }
    if !swap.left_in_place.is_empty() {
        writeln!(output, "  left in place: {}", swap.left_in_place.join(", "))?;
    }
    writeln!(
        output,
        "  AMD GPU: ready ({} device{})",
        probe.device_count,
        if probe.device_count == 1 { "" } else { "s" }
    )?;
    if !probe.devices.is_empty() {
        writeln!(output, "  GPU: {}", probe.devices.join(", "))?;
    }
    writeln!(output, "  next step: rocm comfyui start")?;
    Ok(output)
}

/// Names the entries a finished swap replaced or set aside, if any.
fn swap_changes_summary(swap: &source_swap::SwapReport) -> Option<String> {
    let mut parts = Vec::new();
    if !swap.replaced.is_empty() {
        parts.push(format!("replaced: {}", swap.replaced.join(", ")));
    }
    if !swap.set_aside.is_empty() {
        let renamed: Vec<String> = swap
            .set_aside
            .iter()
            .map(|(name, new_name)| format!("{name} (now {new_name})"))
            .collect();
        parts.push(format!("set aside: {}", renamed.join(", ")));
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

/// Downloads (if not already cached) and unpacks the ComfyUI source archive
/// into a staging folder, then installs its code at `source_path` with
/// [`source_swap::install_release_tree`]. Returns the source URL it resolved
/// — so the caller can record it on the install manifest without re-resolving
/// [`comfyui_source_archive_url`] a second time — and the swap's report: what
/// it kept, replaced, set aside and left in place, and its warnings.
///
/// Nothing at `source_path` is touched until the new tree is fully unpacked
/// and `before_swap` has allowed it, so a failed download, a bad archive, one
/// that is not a ComfyUI release, or a refusal from `before_swap` leaves an
/// existing install as it was.
fn download_and_extract_source(
    app_root: &Path,
    source_path: &Path,
    log: &mut fs::File,
    before_swap: &dyn Fn() -> Result<()>,
) -> Result<(String, source_swap::SwapReport)> {
    let source_url = comfyui_source_archive_url();
    let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
    fs::create_dir_all(
        archive_path
            .parent()
            .context("ComfyUI archive path has no parent directory")?,
    )?;
    if archive_path.is_file() {
        writeln!(
            log,
            "Using downloaded source archive {}.",
            archive_path.display()
        )?;
    } else {
        writeln!(log, "Downloading {source_url}.")?;
        let download_label = "Fetching ComfyUI source archive…";
        let spinner = AnimatedSpinner::start(download_label);
        let download_result = download_file(&source_url, &archive_path, &mut |bytes, total| {
            spinner.set_progress(download_label, bytes, total);
        });
        drop(spinner);
        download_result?;
    }
    let extract_root = app_root
        .join("extract")
        .join(format!("source-{}", unix_time_millis()));
    // Replacing an install with something that is not ComfyUI would leave
    // nothing startable where it was; a first install has nothing to lose.
    let replaces_an_install = fs::symlink_metadata(source_path).is_ok();
    let result = unpack_source_archive(&archive_path, &extract_root, replaces_an_install).and_then(
        |extracted| {
            before_swap()?;
            let report = source_swap::install_release_tree(&extracted, source_path)?;
            writeln!(log, "Installed source at {}.", source_path.display())?;
            for name in &report.kept {
                writeln!(log, "Kept {name}.")?;
            }
            for name in &report.replaced {
                writeln!(log, "Replaced {name}.")?;
            }
            for (name, new_name) in &report.set_aside {
                writeln!(log, "Set aside {name} as {new_name}.")?;
            }
            for name in &report.left_in_place {
                writeln!(log, "Left in place {name}.")?;
            }
            for warning in &report.warnings {
                writeln!(log, "Warning: {warning}")?;
            }
            Ok(report)
        },
    );
    fs::remove_dir_all(&extract_root).ok();
    Ok((source_url, result?))
}

/// Unpacks `archive_path` under `extract_root` and returns the archive's
/// top-level folder. With `require_main_py` (an install is there to be
/// replaced), an archive that is not a ComfyUI release (no `main.py` in that
/// folder) is refused. A cached archive that cannot be unpacked, or that is
/// refused, is dropped, so the next attempt downloads a fresh copy instead of
/// failing the same way.
fn unpack_source_archive(
    archive_path: &Path,
    extract_root: &Path,
    require_main_py: bool,
) -> Result<PathBuf> {
    fs::create_dir_all(extract_root)
        .with_context(|| format!("failed to create {}", extract_root.display()))?;
    let archive = fs::File::open(archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let mut tar = tar::Archive::new(GzDecoder::new(archive));
    let unpacked = tar
        .unpack(extract_root)
        .with_context(|| format!("failed to extract {}", archive_path.display()))
        .and_then(|()| first_child_dir(extract_root))
        .and_then(|release| {
            if !require_main_py || release.join("main.py").is_file() {
                Ok(release)
            } else {
                bail!(
                    "{} is not a ComfyUI release: it has no main.py in {}",
                    archive_path.display(),
                    release.display()
                )
            }
        });
    if unpacked.is_err() {
        fs::remove_file(archive_path).ok();
    }
    unpacked
}

fn first_child_dir(root: &Path) -> Result<PathBuf> {
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            return Ok(path);
        }
    }
    bail!("ComfyUI source archive did not contain a directory")
}

fn download_file(
    url: &str,
    destination: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<()> {
    download_file_to_path_with_progress(url, destination, Duration::from_mins(2), on_progress)
}

fn filtered_requirement_specs(requirements_path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(requirements_path)
        .with_context(|| format!("failed to read {}", requirements_path.display()))?;
    let mut output = Vec::new();
    for token in requirement_tokens(&text) {
        if requirement_package_name(&token)
            .is_some_and(|name| matches!(name.as_str(), "torch" | "torchvision" | "torchaudio"))
        {
            continue;
        }
        output.push(token);
    }
    Ok(output)
}

fn requirement_tokens(text: &str) -> Vec<String> {
    text.lines()
        .flat_map(|line| {
            line.split('#')
                .next()
                .unwrap_or_default()
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|token| !token.trim().is_empty())
        .collect()
}

fn requirement_package_name(spec: &str) -> Option<String> {
    let trimmed = spec.trim();
    if trimmed.starts_with('-') || trimmed.contains("://") {
        return None;
    }
    let end = trimmed
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .unwrap_or(trimmed.len());
    (end > 0).then(|| trimmed[..end].replace('_', "-").to_ascii_lowercase())
}

fn uv_install_args(
    venv_python: &Path,
    packages: &[String],
    constraints_path: Option<&Path>,
) -> Vec<String> {
    let mut args = uv_pip_install_base(venv_python);
    args.push("--upgrade".to_owned());
    if let Some(path) = constraints_path {
        args.push("--constraint".to_owned());
        args.push(path.to_string_lossy().into_owned());
    }
    args.extend(packages.iter().cloned());
    args
}

/// The torch-stack packages whose managed ROCm versions must be preserved across
/// a ComfyUI dependency install.
const TORCH_STACK_PACKAGES: [&str; 3] = ["torch", "torchvision", "torchaudio"];

/// Probe of an installed distribution's version, read from metadata WITHOUT
/// importing the package (importing torch needs the runtime's native env). Emits
/// JSON: a map from package name to version string for the ones present.
const TORCH_STACK_VERSION_PROBE: &str = "import json,sys\n\
     from importlib import metadata\n\
     out={}\n\
     for name in ('torch','torchvision','torchaudio'):\n\
     \x20 try:\n\
     \x20   out[name]=metadata.version(name)\n\
     \x20 except Exception:\n\
     \x20   pass\n\
     sys.stdout.write(json.dumps(out))\n";

/// Read the exact installed versions of the managed runtime's torch stack via
/// dist metadata (no native import). Returns constraint lines (`torch==X`) for
/// each package currently present, so the ComfyUI dependency resolution is pinned
/// to them and cannot pull a CUDA build. An absent package yields no line — we
/// only constrain what the runtime actually ships.
///
/// Constraining only the installed packages is safe here because the full
/// torch/torchvision/torchaudio trio is an invariant of a managed runtime:
/// `therock` installs all three atomically and refuses to install unless it finds
/// a mutually-compatible set (see `select_matching_pip_package_versions` — it
/// bails with "no mutually compatible ... torch, torchvision, and torchaudio
/// versions"), and
/// ComfyUI install only ever targets such a runtime (`select_runtime` requires a
/// managed wheel install). So there is no "torch present, torchvision/torchaudio
/// absent" runtime in which an unpinned CUDA `torchvision` could resolve against
/// the release part of the pinned ROCm torch and install a mixed CUDA/ROCm stack.
fn torch_stack_constraints(
    python: &Path,
    runtime_env: Option<&ComfyUiRuntimeEnvironment>,
    log: &mut fs::File,
) -> Result<Vec<String>> {
    let versions = probe_torch_stack_versions(python, runtime_env)?;
    let constraints = torch_constraint_lines(&versions);
    writeln!(
        log,
        "Pinning managed torch stack: {}",
        if constraints.is_empty() {
            "none found".to_owned()
        } else {
            constraints.join(", ")
        }
    )?;
    Ok(constraints)
}

fn probe_torch_stack_versions(
    python: &Path,
    runtime_env: Option<&ComfyUiRuntimeEnvironment>,
) -> Result<Vec<(String, String)>> {
    let mut command = Command::new(python);
    command.arg("-c").arg(TORCH_STACK_VERSION_PROBE);
    if let Some(runtime_env) = runtime_env {
        apply_runtime_environment(&mut command, runtime_env)?;
    }
    let output = capture_configured_command(
        command,
        &format!("probe managed torch stack with {}", python.display()),
    )?;
    if !output.status.success() {
        bail!(
            "failed to read the managed runtime's torch versions (status {}).\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: std::collections::BTreeMap<String, String> = serde_json::from_str(stdout.trim())
        .with_context(|| format!("failed to parse torch stack version probe output: {stdout}"))?;
    Ok(parsed.into_iter().collect())
}

/// Build `name==version` constraint lines for the torch-stack packages that were
/// found, in a stable order. Pure and side-effect free for unit testing.
fn torch_constraint_lines(versions: &[(String, String)]) -> Vec<String> {
    TORCH_STACK_PACKAGES
        .iter()
        .filter_map(|package| {
            versions
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(package))
                .map(|(_, version)| format!("{package}=={version}"))
        })
        .collect()
}

/// Write the torch-stack constraints to a file uv can read via `--constraint`.
/// Returns `None` when there is nothing to pin (no constraint file is needed).
fn write_torch_constraints(
    app_root: &Path,
    constraints: &[String],
    log: &mut fs::File,
) -> Result<Option<PathBuf>> {
    if constraints.is_empty() {
        return Ok(None);
    }
    let path = app_root.join("torch-constraints.txt");
    fs::create_dir_all(
        path.parent()
            .context("ComfyUI constraints path has no parent directory")?,
    )?;
    let mut contents = constraints.join("\n");
    contents.push('\n');
    fs::write(&path, contents).with_context(|| format!("failed to write {}", path.display()))?;
    writeln!(log, "Wrote torch constraints to {}.", path.display())?;
    Ok(Some(path))
}

#[allow(clippy::too_many_arguments)]
fn run_uv_logged_command(
    paths: &AppPaths,
    uv: &Path,
    args: Vec<String>,
    runtime_env: Option<&ComfyUiRuntimeEnvironment>,
    log: &mut fs::File,
    log_path: &Path,
    context_text: &str,
    spinner_label: &str,
) -> Result<()> {
    writeln!(
        log,
        "command: {} {}",
        uv.display(),
        args.iter()
            .map(|arg| quote_log_arg(arg))
            .collect::<Vec<_>>()
            .join(" ")
    )?;
    let mut command = Command::new(uv);
    command
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in uv_command_env(paths) {
        command.env(key, value);
    }
    if let Some(runtime_env) = runtime_env {
        apply_runtime_environment(&mut command, runtime_env)?;
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("{context_text}: failed to run {}", uv.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("child stdout was not captured")?;
    let stderr = child
        .stderr
        .take()
        .context("child stderr was not captured")?;
    let stdout_log = log
        .try_clone()
        .context("failed to clone ComfyUI install log for stdout")?;
    let stderr_log = log
        .try_clone()
        .context("failed to clone ComfyUI install log for stderr")?;
    // `AnimatedSpinner` (see cli_progress.rs) is TTY-gated and a hard no-op off
    // a terminal, so on its own a piped/CI install would print nothing for the
    // entire uv resolve, indistinguishable from a hang. `stream_logged_output`
    // below is the off-TTY fallback: it tees the child's stdout/stderr through
    // to our real stdout/stderr whenever stderr isn't a terminal, so a
    // non-interactive install still shows live progress.
    let spinner = AnimatedSpinner::start(spinner_label);
    let stdout_thread =
        thread::spawn(move || stream_logged_output(stdout, stdout_log, OutputTarget::Stdout));
    let stderr_thread =
        thread::spawn(move || stream_logged_output(stderr, stderr_log, OutputTarget::Stderr));
    let status = child
        .wait()
        .with_context(|| format!("{context_text}: failed waiting for {}", uv.display()))?;
    stdout_thread
        .join()
        .map_err(|_| anyhow::anyhow!("{context_text}: stdout reader failed"))?
        .context("failed to stream command stdout")?;
    stderr_thread
        .join()
        .map_err(|_| anyhow::anyhow!("{context_text}: stderr reader failed"))?
        .context("failed to stream command stderr")?;
    drop(spinner);
    if status.success() {
        return Ok(());
    }
    bail!(
        "{context_text}: uv exited with {status}; see {} for details",
        log_path.display()
    );
}

/// Which real stream a `stream_logged_output` reader mirrors to when not
/// attached to a terminal.
enum OutputTarget {
    Stdout,
    Stderr,
}

fn stream_logged_output<R: Read>(
    mut reader: R,
    mut log: fs::File,
    target: OutputTarget,
) -> io::Result<()> {
    // When stderr is a terminal, the `AnimatedSpinner` is the progress signal
    // and raw uv output would visually clash with it, so only tee through
    // when we're not attached to one (piped output, CI, etc).
    let tee = !io::stderr().is_terminal();
    let mut buffer = [0_u8; 8192];
    loop {
        let len = reader.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        log.write_all(&buffer[..len])?;
        if tee {
            match target {
                OutputTarget::Stdout => {
                    let mut stdout = io::stdout().lock();
                    stdout.write_all(&buffer[..len])?;
                    stdout.flush()?;
                }
                OutputTarget::Stderr => {
                    let mut stderr = io::stderr().lock();
                    stderr.write_all(&buffer[..len])?;
                    stderr.flush()?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::comfyui::load_manifest;
    #[cfg(unix)]
    use crate::comfyui::process::ensure_source_is_startable;
    use crate::comfyui::tests::{
        assert_user_files_intact, ready_runtime_manifest, reinstall_fixture, reinstall_options,
        remove_test_root, spawn_stand_in_process, test_paths, unused_local_port,
        write_runtime_manifest,
    };
    #[cfg(unix)]
    use crate::comfyui::tests::{plant_cached_source_archive, plant_installable_release};
    use crate::comfyui::{
        ComfyUiRunState, ComfyUiState, app_root, evaluate_running_state, save_state, state_path,
        stop,
    };
    use rocm_core::format_http_base_url;
    use std::net::TcpListener;

    #[test]
    fn requirements_filter_preserves_therock_torch_stack() {
        let text = "torch torchvision>=1 torchaudio\nnumpy>=1.25\n# comment\naiohttp\n";
        assert_eq!(
            requirement_tokens(text),
            vec![
                "torch".to_owned(),
                "torchvision>=1".to_owned(),
                "torchaudio".to_owned(),
                "numpy>=1.25".to_owned(),
                "aiohttp".to_owned()
            ]
        );
        let kept = requirement_tokens(text)
            .into_iter()
            .filter(|token| {
                !requirement_package_name(token).is_some_and(|name| {
                    matches!(name.as_str(), "torch" | "torchvision" | "torchaudio")
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(kept, vec!["numpy>=1.25".to_owned(), "aiohttp".to_owned()]);
    }

    #[test]
    fn torch_constraint_lines_pin_present_stack_in_stable_order() {
        // Probe output arrives unordered and case-mixed; constraints come out in
        // the canonical torch/torchvision/torchaudio order, one per present pkg.
        let versions = vec![
            ("torchvision".to_owned(), "0.22.0+gitabc".to_owned()),
            ("Torch".to_owned(), "2.11.0+gitd0c8b1f".to_owned()),
        ];
        assert_eq!(
            torch_constraint_lines(&versions),
            vec![
                "torch==2.11.0+gitd0c8b1f".to_owned(),
                "torchvision==0.22.0+gitabc".to_owned(),
            ]
        );
    }

    #[test]
    fn torch_constraint_lines_empty_when_stack_absent() {
        assert!(torch_constraint_lines(&[]).is_empty());
        assert!(torch_constraint_lines(&[("numpy".to_owned(), "1.25.0".to_owned())]).is_empty());
    }

    #[test]
    fn uv_install_args_include_constraint_flag_when_pinned() {
        let python = Path::new("/runtime/bin/python");
        let packages = vec!["numpy".to_owned()];
        let constraints = PathBuf::from("/tmp/torch-constraints.txt");

        let with_pin = uv_install_args(python, &packages, Some(&constraints));
        assert!(
            with_pin
                .windows(2)
                .any(|pair| pair == ["--constraint".to_owned(), constraints.display().to_string()]),
            "expected --constraint <path> in {with_pin:?}"
        );

        let without_pin = uv_install_args(python, &packages, None);
        assert!(
            !without_pin.iter().any(|arg| arg == "--constraint"),
            "no --constraint expected when nothing to pin: {without_pin:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn run_uv_logged_command_reports_concrete_log_path_on_failure() -> Result<()> {
        // `rocm comfyui logs` can still find this run's log via a directory
        // scan even without a saved manifest, but naming the path directly
        // in the error is more precise and needs no second command.
        use rocm_core::fake_executable::write_fake_executable;

        let paths = test_paths("comfyui-uv-failure");
        fs::create_dir_all(&paths.cache_dir)?;
        let uv_path = paths.cache_dir.join("fake-uv");
        write_fake_executable(&uv_path, "#!/bin/sh\nexit 1\n")?;

        let log_path = paths.cache_dir.join("install.log");
        let mut log = fs::File::create(&log_path)?;

        let error = run_uv_logged_command(
            &paths,
            &uv_path,
            vec!["pip".to_owned(), "install".to_owned()],
            None,
            &mut log,
            &log_path,
            "install ComfyUI dependencies",
            "Resolving and installing packages with uv…",
        )
        .expect_err("uv exiting non-zero should fail");

        let message = error.to_string();
        assert!(
            message.contains(&log_path.display().to_string()),
            "error should name the concrete log path so a failed install's log stays discoverable: {message}"
        );

        fs::remove_dir_all(&paths.cache_dir).ok();
        Ok(())
    }

    #[test]
    fn install_dry_run_uses_selected_runtime_folder() -> Result<()> {
        let paths = test_paths("comfyui-selected-runtime-folder");
        let runtime = ready_runtime_manifest(&paths, "selected-runtime")?;
        write_runtime_manifest(&paths, &runtime)?;
        let config = RocmCliConfig {
            active_runtime_key: Some(runtime.runtime_key.clone()),
            ..Default::default()
        };

        let rendered = install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: None,
                reinstall: false,
                dry_run: true,
            },
        )?;

        let runtime_app = runtime_app_root(&runtime);
        assert!(rendered.contains(&runtime_app.display().to_string()));
        assert!(!rendered.contains(&app_root(&paths).display().to_string()));
        // The runtime is labelled "ROCm runtime" to match the noun used by the
        // `select_runtime` error messages this command can also emit; the stale
        // "ROCm install:" label must not come back.
        assert!(
            rendered.contains("ROCm runtime:"),
            "install output should label the runtime as `ROCm runtime:`, got: {rendered}"
        );
        assert!(
            !rendered.contains("ROCm install:"),
            "install output must not reintroduce the `ROCm install:` label, got: {rendered}"
        );
        let expected_folder_line = format!(
            "  folder: {}\n",
            source_path_from_app_root(&runtime_app).display()
        );
        assert!(
            rendered.contains(&expected_folder_line),
            "install `folder:` should report the ComfyUI source checkout, not the app root, got: {rendered}"
        );
        assert!(
            !rendered.contains(&format!("  folder: {}\n", runtime_app.display())),
            "install `folder:` must not report the bare app root, got: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn download_file_reports_cumulative_progress_to_its_caller() -> Result<()> {
        // A multi-chunk body (the streaming downloader reads in 64 KiB
        // chunks) so a single callback firing wouldn't already satisfy the
        // "monotonically increasing" assertion below.
        let body: Vec<u8> = (0..200_000_u32).map(|i| (i % 256) as u8).collect();

        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let served = body.clone();
        let server = thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            )?;
            stream.write_all(&served)?;
            stream.flush()?;
            Ok(())
        });

        let url = format!("http://127.0.0.1:{port}/archive.tar.gz");
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-comfyui-download-progress-{}",
            unix_time_millis()
        ));
        fs::create_dir_all(&root)?;
        let destination = root.join("archive.tar.gz");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        download_file(&url, &destination, &mut |bytes, total| {
            calls.push((bytes, total));
        })?;
        assert_eq!(fs::read(&destination)?, body);

        server.join().expect("localhost server thread panicked")?;
        let _ = fs::remove_dir_all(&root);

        let total = Some(body.len() as u64);
        assert!(
            calls.len() >= 2,
            "expected at least a pre-transfer and a final callback: {calls:?}"
        );
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "byte counts must never regress: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last callback must report the complete transfer: {calls:?}"
        );
        Ok(())
    }

    /// Issue #530: a reinstall whose new source cannot be obtained must leave
    /// the existing install and every user file exactly as they were. Before
    /// the fix the source folder was deleted before anything was fetched.
    #[test]
    fn reinstall_that_cannot_obtain_new_source_changes_nothing() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-failed-source")?;
        let source = source_path_from_app_root(&app_root);
        // A truncated download: not a gzip stream, so unpacking fails.
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        fs::write(&archive_path, "not a gzip stream")?;

        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("a reinstall with an unreadable source archive must fail");

        assert!(
            format!("{error:#}").contains("failed to extract"),
            "unexpected error: {error:#}"
        );
        // No swap marker was written, so no interrupted reinstall is
        // reported and no command is named to finish one.
        assert!(
            !format!("{error:#}").contains("rocm comfyui install"),
            "nothing was changed, so there is nothing to finish: {error:#}"
        );
        assert_user_files_intact(&source);
        assert_eq!(
            fs::read_to_string(source.join("main.py"))?,
            "old main",
            "the existing ComfyUI code must survive a failed reinstall"
        );
        assert!(source.join("comfy/removed_upstream.py").is_file());
        assert!(
            !archive_path.exists(),
            "an archive that cannot be unpacked must not be reused by the next attempt"
        );
        assert_staging_folder_empty(&app_root)?;
        remove_test_root(&paths);
        Ok(())
    }

    /// A downloaded archive that unpacks but is not a ComfyUI release (no
    /// `main.py` in its top folder) leaves the existing install as it was, and
    /// is dropped so the next attempt downloads again instead of failing the
    /// same way.
    #[cfg(unix)]
    #[test]
    fn reinstall_from_an_archive_without_main_py_changes_nothing() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-not-comfyui")?;
        let source = source_path_from_app_root(&app_root);
        plant_cached_source_archive(
            &app_root,
            &[("README.md", "not comfyui"), ("comfy/x.py", "code")],
        )?;
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);

        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("an archive without main.py must not be installed");

        let message = format!("{error:#}");
        assert!(message.contains("is not a ComfyUI release"), "{message}");
        assert!(
            !message.contains("rocm comfyui install"),
            "nothing was changed, so there is nothing to finish: {message}"
        );
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "old main");
        assert!(source.join("comfy/removed_upstream.py").is_file());
        assert!(!source_swap::swap_interrupted(&source));
        assert_user_files_intact(&source);
        assert!(
            !archive_path.exists(),
            "an archive that is not a ComfyUI release must not be reused"
        );
        assert_staging_folder_empty(&app_root)?;
        remove_test_root(&paths);
        Ok(())
    }

    /// Nothing is left behind in the staging folder a release is unpacked to.
    fn assert_staging_folder_empty(app_root: &Path) -> Result<()> {
        let extract = app_root.join("extract");
        let left: Vec<_> = match fs::read_dir(&extract) {
            Ok(entries) => entries
                .map(|entry| entry.map(|entry| entry.file_name()))
                .collect::<io::Result<_>>()?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        assert!(
            left.is_empty(),
            "the staging folder must be removed: {left:?}"
        );
        Ok(())
    }

    /// Issue #530: a successful reinstall replaces ComfyUI's code but leaves
    /// the user's models, workflows, outputs, inputs, custom nodes and extra
    /// model paths in place, and the report names exactly those.
    #[cfg(unix)]
    #[test]
    fn reinstall_replaces_code_and_keeps_user_data() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-keeps-data")?;
        let source = source_path_from_app_root(&app_root);
        plant_cached_source_archive(
            &app_root,
            &[
                ("main.py", "new main"),
                // Torch-stack only, so the dependency install is skipped.
                ("requirements.txt", "torch\n"),
                ("comfy/added_upstream.py", "new code"),
                ("models/checkpoints/put_checkpoints_here", ""),
                ("models/new_kind/put_new_kind_here", ""),
                ("custom_nodes/websocket_image_save.py", "new sample node"),
                ("input/example.png", "release example"),
                ("output/_output_images_will_be_put_here", ""),
            ],
        )?;
        let runtime = ready_runtime_manifest(&paths, "reinstall-runtime")?;
        let python = PathBuf::from(runtime.python_executable.context("runtime has no Python")?);
        rocm_core::fake_executable::write_fake_executable(
            &python,
            "#!/bin/sh\nprintf '%s' '{\"torch_version\": \"2.4.0\", \"torch_cuda_available\": true, \"device_count\": 1, \"devices\": [\"Fake GPU\"]}' > \"$2\"\n",
        )?;

        let rendered = install(&paths, &config, reinstall_options(false))?;

        assert_eq!(fs::read_to_string(source.join("main.py"))?, "new main");
        assert!(source.join("comfy/added_upstream.py").is_file());
        assert!(
            !source.join("comfy/removed_upstream.py").exists(),
            "code the new release no longer ships must not linger"
        );
        assert_user_files_intact(&source);
        // What the release adds inside a kept folder is merged in alongside.
        assert!(source.join("models/new_kind/put_new_kind_here").is_file());
        assert!(
            source
                .join("models/checkpoints/put_checkpoints_here")
                .is_file()
        );
        assert!(source.join("input/example.png").is_file());
        assert!(
            rendered.contains(
                "  kept: models, user, output, input, custom_nodes, datasets, extra_model_paths.yaml\n"
            ),
            "the report must name what was kept: {rendered}"
        );
        assert!(
            rendered.contains("  left in place: styles.csv, temp\n"),
            "the report must name what it left alone: {rendered}"
        );
        // The release was unpacked into a staging folder; nothing of it is
        // left there once it is installed.
        assert_staging_folder_empty(&app_root)?;
        remove_test_root(&paths);
        Ok(())
    }

    /// Issue #530: the dry run states what a reinstall replaces and keeps, and
    /// the command it prints is the one that was asked for. Nothing on disk
    /// changes.
    #[test]
    fn reinstall_dry_run_names_what_is_replaced_and_kept() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-dry-run")?;
        let source = source_path_from_app_root(&app_root);

        let rendered = install(&paths, &config, reinstall_options(true))?;

        assert!(
            rendered.contains(&format!(
                "  reinstall: replaces the ComfyUI code in {}\n",
                source.display()
            )),
            "{rendered}"
        );
        assert!(
            rendered.contains(
                "  keeps: models, user, output, input, custom_nodes, datasets, extra_model_paths.yaml\n"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains(" --reinstall"),
            "the printed command must carry --reinstall: {rendered}"
        );
        // Nothing was interrupted here, so nothing claims it was.
        assert!(!rendered.contains("interrupted"), "{rendered}");
        assert_user_files_intact(&source);
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "old main");
        remove_test_root(&paths);
        Ok(())
    }

    /// A plain `install --dry-run` over an existing install previews reusing
    /// it: no reinstall lines, and the command it prints is the plain install
    /// that was asked for, not the destructive `--reinstall`.
    #[test]
    fn plain_install_dry_run_previews_no_reinstall() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-plain-dry-run")?;
        let source = source_path_from_app_root(&app_root);

        let rendered = install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: None,
                reinstall: false,
                dry_run: true,
            },
        )?;

        let command = rendered
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("install command: "))
            .unwrap_or_else(|| panic!("no install command in:\n{rendered}"));
        assert!(command.contains("comfyui install"), "{command}");
        assert!(!command.contains("--reinstall"), "{command}");
        assert!(!rendered.contains("  reinstall:"), "{rendered}");
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }

    /// `--reinstall --dry-run` where nothing is installed yet: there is no
    /// code to replace, so the preview says nothing about replacing or keeping
    /// anything in a folder that does not exist.
    #[test]
    fn reinstall_dry_run_without_an_install_previews_no_replacement() -> Result<()> {
        let paths = test_paths("comfyui-reinstall-dry-run-no-install");
        let runtime = ready_runtime_manifest(&paths, "reinstall-runtime")?;
        write_runtime_manifest(&paths, &runtime)?;
        let config = RocmCliConfig {
            active_runtime_key: Some(runtime.runtime_key.clone()),
            ..Default::default()
        };
        let source = source_path_from_app_root(&runtime_app_root(&runtime));
        assert!(!source.exists());

        let rendered = install(&paths, &config, reinstall_options(true))?;

        assert!(!rendered.contains("  reinstall:"), "{rendered}");
        assert!(!rendered.contains("  keeps:"), "{rendered}");
        assert!(!source.exists(), "a dry run creates nothing");
        remove_test_root(&paths);
        Ok(())
    }

    /// The running guard is checked again once the new release is unpacked,
    /// just before the code is replaced. When it refuses there, the existing
    /// install is exactly as it was, and the staging folder is gone.
    #[cfg(unix)]
    #[test]
    fn a_refusal_just_before_the_swap_changes_nothing() -> Result<()> {
        let (paths, _config, app_root) = reinstall_fixture("comfyui-refused-before-swap")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        fs::create_dir_all(app_root.join("logs"))?;
        let mut log = fs::File::create(app_root.join("logs").join("install.log"))?;

        let error = download_and_extract_source(&app_root, &source, &mut log, &|| {
            bail!("ComfyUI started meanwhile")
        })
        .expect_err("the refusal must stop the install");

        assert!(
            format!("{error:#}").contains("ComfyUI started meanwhile"),
            "{error:#}"
        );
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "old main");
        assert!(source.join("comfy/removed_upstream.py").is_file());
        assert!(!source_swap::swap_interrupted(&source));
        assert_user_files_intact(&source);
        assert_staging_folder_empty(&app_root)?;
        remove_test_root(&paths);
        Ok(())
    }

    /// After an interrupted reinstall, a plain `install` finishes the swap
    /// rather than reusing the half-replaced folder as it is.
    #[test]
    fn install_after_an_interrupted_reinstall_finishes_it() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-finish-interrupted")?;
        let source = source_path_from_app_root(&app_root);
        fs::write(source.join(source_swap::SWAP_MARKER), "models\n")?;
        let plain = ComfyUiInstallOptions {
            runtime_id: None,
            reinstall: false,
            dry_run: true,
        };

        let rendered = install(&paths, &config, plain.clone())?;
        assert!(
            rendered
                .contains("  note: a previous reinstall was interrupted; installing finishes it\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("  reinstall: replaces the ComfyUI code in"),
            "{rendered}"
        );
        // Every preserved folder is present now, but the interrupted run
        // recorded only `models` as the user's; the dry run must say what the
        // real run's `kept:` will, which reads the same record.
        assert!(rendered.contains("  keeps: models\n"), "{rendered}");

        // And the real run does go to fetch new code (stopped here by an
        // unreadable archive) instead of "Using existing ComfyUI source".
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        fs::write(&archive_path, "not a gzip stream")?;
        let error = install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                dry_run: false,
                ..plain
            },
        )
        .expect_err("the planted archive is unreadable");
        assert!(
            format!("{error:#}").contains("failed to extract"),
            "{error:#}"
        );
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }

    /// The running guard only stops a reinstall for a ComfyUI that is really
    /// running from this folder. A saved state whose process is gone and whose
    /// port is closed (ComfyUI exited without `stop`) does not block it.
    #[test]
    fn reinstall_proceeds_over_a_stale_running_state() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-stale-state")?;
        let source = source_path_from_app_root(&app_root);
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        fs::write(&archive_path, "not a gzip stream")?;
        let mut exited = spawn_stand_in_process()?;
        let dead_pid = exited.id();
        exited.kill()?;
        exited.wait()?;
        save_running_state(&paths, &app_root, &source, dead_pid)?;

        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("the planted archive is unreadable");

        assert!(
            format!("{error:#}").contains("failed to extract"),
            "a stale state must not block the reinstall: {error:#}"
        );
        remove_test_root(&paths);
        Ok(())
    }

    /// A ComfyUI running from a different folder (another runtime's install)
    /// is not affected by replacing this one's code, so it does not block it.
    #[test]
    fn reinstall_proceeds_while_a_comfyui_runs_from_another_folder() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-other-folder")?;
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        fs::write(&archive_path, "not a gzip stream")?;
        let mut child = spawn_stand_in_process()?;
        let elsewhere = paths.data_dir.join("other-runtime").join("source");
        save_running_state(&paths, &app_root, &elsewhere, child.id())?;

        let result = install(&paths, &config, reinstall_options(false));
        let _ = child.kill();
        let _ = child.wait();

        let error = result.expect_err("the planted archive is unreadable");
        assert!(
            format!("{error:#}").contains("failed to extract"),
            "a ComfyUI running from another folder must not block the reinstall: {error:#}"
        );
        remove_test_root(&paths);
        Ok(())
    }

    fn save_running_state(
        paths: &AppPaths,
        app_root: &Path,
        source: &Path,
        pid: u32,
    ) -> Result<()> {
        let port = unused_local_port()?;
        save_state(
            paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: format_http_base_url("127.0.0.1", port),
                host: "127.0.0.1".to_owned(),
                port,
                pid,
                source_path: source.to_path_buf(),
                python_executable: PathBuf::from("python"),
                log_path: app_root.join("logs").join("start.log"),
                started_at_unix_ms: unix_time_millis(),
            },
        )
    }

    /// A reinstall is refused while the ComfyUI this CLI started runs from the
    /// folder it would change, here answering on its port, and nothing
    /// changes. The remedy the refusal names, `rocm comfyui stop`, is then run
    /// and does clear it: the same reinstall goes ahead (here as far as the
    /// planted unreadable archive).
    #[test]
    fn reinstall_is_refused_while_comfyui_runs_until_it_is_stopped() -> Result<()> {
        reinstall_is_refused_until_stopped("comfyui-reinstall-while-running", true)
    }

    /// The same for a ComfyUI that was started and is still loading: its
    /// process is alive but nothing answers on its port yet.
    #[test]
    fn reinstall_is_refused_while_comfyui_starts_until_it_is_stopped() -> Result<()> {
        reinstall_is_refused_until_stopped("comfyui-reinstall-while-starting", false)
    }

    fn reinstall_is_refused_until_stopped(name: &str, answering: bool) -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture(name)?;
        let source = source_path_from_app_root(&app_root);
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        fs::write(&archive_path, "not a gzip stream")?;
        let mut child = spawn_stand_in_process()?;
        // Held for the test, so the port answers as a running ComfyUI's does.
        let listener = answering
            .then(|| std::net::TcpListener::bind("127.0.0.1:0"))
            .transpose()?;
        let port = match &listener {
            Some(listener) => listener.local_addr()?.port(),
            None => unused_local_port()?,
        };
        let state = ComfyUiState {
            app_id: APP_ID.to_owned(),
            url: format_http_base_url("127.0.0.1", port),
            host: "127.0.0.1".to_owned(),
            port,
            pid: child.id(),
            source_path: source.clone(),
            python_executable: PathBuf::from("python"),
            log_path: app_root.join("logs").join("start.log"),
            started_at_unix_ms: unix_time_millis(),
        };
        save_state(&paths, &state)?;
        assert_eq!(
            evaluate_running_state(&state).state,
            if answering {
                ComfyUiRunState::Running
            } else {
                ComfyUiRunState::Starting
            },
            "the planted ComfyUI must read as the state this test is about"
        );

        // The dry run says what the real run then does.
        let preview = install(&paths, &config, reinstall_options(true))?;
        assert!(
            preview.contains(&format!(
                "  note: ComfyUI is running from {} (process {}, {}). Stop it with `rocm \
                 comfyui stop`; a reinstall is refused until then\n",
                source.display(),
                child.id(),
                format_http_base_url("127.0.0.1", port)
            )),
            "{preview}"
        );
        let refused = install(&paths, &config, reinstall_options(false))
            .expect_err("a reinstall must be refused while ComfyUI runs from the folder");
        let message = format!("{refused:#}");
        assert!(message.contains("`rocm comfyui stop`"), "{message}");
        assert!(message.contains("Nothing was changed"), "{message}");
        assert_user_files_intact(&source);
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "old main");
        assert!(
            archive_path.is_file(),
            "the refusal must come before the archive is even opened"
        );

        let stopped = stop(&paths);
        let _ = child.kill();
        let _ = child.wait();
        stopped?;

        let preview = install(&paths, &config, reinstall_options(true))?;
        assert!(!preview.contains("is running"), "{preview}");
        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("the planted archive is unreadable");
        assert!(
            format!("{error:#}").contains("failed to extract"),
            "after `stop` the reinstall must no longer be refused: {error:#}"
        );
        drop(listener);
        remove_test_root(&paths);
        Ok(())
    }

    /// A reinstall stopped by a release entry it cannot remove names the
    /// install for this runtime (`--runtime-id`), not a bare `rocm comfyui
    /// install` that would finish whichever runtime is the default. That
    /// command, run as printed once the entry can be removed, finishes it.
    #[cfg(unix)]
    #[test]
    fn a_reinstall_that_cannot_remove_an_entry_names_the_install_that_finishes_it() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-cannot-remove")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        let locked = source.join("comfy");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o555))?;
        if fs::write(locked.join("probe"), "").is_ok() {
            // Running as root: permissions do not stop removal here. The
            // advice itself is also covered, for any user, by
            // `a_reinstall_that_fails_after_the_swap_started_names_the_install_that_finishes_it`.
            eprintln!("skipped: running as root, so a read-only folder cannot make a removal fail");
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
            remove_test_root(&paths);
            return Ok(());
        }

        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("a release entry that cannot be removed must stop the reinstall");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        let message = format!("{error:#}");

        assert!(
            message.starts_with(
                "a reinstall was interrupted; run `rocm comfyui install --runtime-id \
                 reinstall-runtime` to finish it: failed to remove "
            ),
            "{message}"
        );
        assert!(message.contains("close any program using it"), "{message}");
        assert!(source_swap::swap_interrupted(&source));
        let command = message
            .split('`')
            .nth(1)
            .context("the message names a command")?;
        assert_eq!(
            command,
            "rocm comfyui install --runtime-id reinstall-runtime"
        );

        // The named command, as printed.
        install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: Some("reinstall-runtime".to_owned()),
                reinstall: false,
                dry_run: false,
            },
        )?;

        assert!(!source_swap::swap_interrupted(&source));
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "new main");
        let manifest = load_manifest(&paths)?.context("manifest")?;
        ensure_source_is_startable(&manifest)?;
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }

    /// Whatever stops a reinstall after its swap has started — here, writing
    /// the release record, which fails for any user because a folder holds
    /// its name — the error names the install that finishes it, and that
    /// command, run as printed once the cause is gone, does.
    #[cfg(unix)]
    #[test]
    fn a_reinstall_that_fails_after_the_swap_started_names_the_install_that_finishes_it()
    -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-fails-mid-swap")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        let blocker = source.join(source_swap::RELEASE_RECORD);
        fs::create_dir_all(blocker.join("in-the-way"))?;

        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("the release record cannot be written over a folder");
        let message = format!("{error:#}");

        assert!(
            message.starts_with(
                "a reinstall was interrupted; run `rocm comfyui install --runtime-id \
                 reinstall-runtime` to finish it: "
            ),
            "{message}"
        );
        assert!(source_swap::swap_interrupted(&source));
        assert_user_files_intact(&source);
        assert_staging_folder_empty(&app_root)?;

        fs::remove_dir_all(&blocker)?;
        let command = message
            .split('`')
            .nth(1)
            .context("the message names a command")?;
        assert_eq!(
            command,
            "rocm comfyui install --runtime-id reinstall-runtime"
        );
        install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: Some("reinstall-runtime".to_owned()),
                reinstall: false,
                dry_run: false,
            },
        )?;

        assert!(!source_swap::swap_interrupted(&source));
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "new main");
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }

    /// When the install fails after the swap (here the GPU check), the error
    /// names the entries the swap replaced, since the report that would have
    /// listed them is never printed, and the command that finishes the
    /// install, which does once the cause is gone.
    #[cfg(unix)]
    #[test]
    fn a_failure_after_the_swap_names_what_it_replaced() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-fails-after-swap")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        let runtime = ready_runtime_manifest(&paths, "reinstall-runtime")?;
        let python = PathBuf::from(runtime.python_executable.context("runtime has no Python")?);
        rocm_core::fake_executable::write_fake_executable(
            &python,
            "#!/bin/sh\nprintf '%s' '{\"torch_version\": \"2.4.0\", \"torch_cuda_available\": false, \"device_count\": 0, \"devices\": []}' > \"$2\"\n",
        )?;

        let error =
            install(&paths, &config, reinstall_options(false)).expect_err("the GPU check fails");
        let message = format!("{error:#}");

        // The install predates the release record, so these were replaced.
        assert!(
            message.starts_with(
                "the new ComfyUI code is in place (replaced: comfy, main.py, requirements.txt), \
                 but the install did not finish; once the cause below is fixed, run `rocm \
                 comfyui install --runtime-id reinstall-runtime` to finish it: "
            ),
            "{message}"
        );
        assert!(message.contains("AMD GPU check failed"), "{message}");
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "new main");
        assert!(!source.join("comfy/removed_upstream.py").exists());
        assert_user_files_intact(&source);

        // With the GPU check passing again, the named command, as printed.
        plant_installable_release(&paths, &app_root)?;
        let rendered = install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: Some("reinstall-runtime".to_owned()),
                reinstall: false,
                dry_run: false,
            },
        )?;
        assert!(rendered.contains("  installed: yes\n"), "{rendered}");
        let manifest = load_manifest(&paths)?.context("manifest")?;
        assert!(manifest.torch_cuda_available);
        ensure_source_is_startable(&manifest)?;
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "new main");
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }

    /// An install that fails without having put new code in place (here a
    /// plain install reusing the existing folder) names nothing to finish.
    /// A first install that fails after its download put the code in place
    /// says so, with nothing replaced to list, and names the command.
    #[cfg(unix)]
    #[test]
    fn only_an_install_that_put_new_code_in_place_names_the_command_that_finishes_it() -> Result<()>
    {
        let (paths, config, app_root) = reinstall_fixture("comfyui-install-fails-no-swap")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        let runtime = ready_runtime_manifest(&paths, "reinstall-runtime")?;
        let python = PathBuf::from(runtime.python_executable.context("runtime has no Python")?);
        rocm_core::fake_executable::write_fake_executable(
            &python,
            "#!/bin/sh\nprintf '%s' '{\"torch_version\": \"2.4.0\", \"torch_cuda_available\": false, \"device_count\": 0, \"devices\": []}' > \"$2\"\n",
        )?;
        let plain = ComfyUiInstallOptions {
            runtime_id: None,
            reinstall: false,
            dry_run: false,
        };

        let reused = install(&paths, &config, plain.clone()).expect_err("the GPU check fails");
        let message = format!("{reused:#}");
        assert!(message.contains("AMD GPU check failed"), "{message}");
        assert!(!message.contains("in place"), "{message}");
        assert!(!message.contains("rocm comfyui install"), "{message}");
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "old main");

        fs::remove_dir_all(&source)?;
        let fresh = install(&paths, &config, plain).expect_err("the GPU check fails");
        let message = format!("{fresh:#}");
        assert!(
            message.starts_with(
                "the new ComfyUI code is in place, but the install did not finish; once the \
                 cause below is fixed, run `rocm comfyui install --runtime-id reinstall-runtime` \
                 to finish it: "
            ),
            "{message}"
        );
        assert_eq!(fs::read_to_string(source.join("main.py"))?, "new main");
        remove_test_root(&paths);
        Ok(())
    }

    /// What a failure after the swap says it changed: both kinds of change,
    /// either alone, or nothing to name.
    #[test]
    fn swap_changes_summary_names_what_was_replaced_and_set_aside() {
        let report = |replaced: &[&str], set_aside: &[(&str, &str)]| source_swap::SwapReport {
            replaced: replaced.iter().map(|name| (*name).to_owned()).collect(),
            set_aside: set_aside
                .iter()
                .map(|(name, new_name)| ((*name).to_owned(), (*new_name).to_owned()))
                .collect(),
            ..Default::default()
        };
        assert_eq!(swap_changes_summary(&report(&[], &[])), None);
        assert_eq!(
            swap_changes_summary(&report(&["comfy", "main.py"], &[])).as_deref(),
            Some("replaced: comfy, main.py")
        );
        assert_eq!(
            swap_changes_summary(&report(
                &["main.py"],
                &[
                    ("app", "app.rocm-cli-kept-1"),
                    ("api", "api.rocm-cli-kept-1")
                ]
            ))
            .as_deref(),
            Some(
                "replaced: main.py; set aside: app (now app.rocm-cli-kept-1), api (now \
                 api.rocm-cli-kept-1)"
            )
        );
    }

    /// The running guard only applies to a run that replaces the code: a
    /// plain install that reuses the existing folder goes ahead while
    /// ComfyUI runs from it.
    #[cfg(unix)]
    #[test]
    fn a_plain_install_is_not_refused_while_comfyui_runs() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-plain-install-while-running")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        let mut child = spawn_stand_in_process()?;
        save_running_state(&paths, &app_root, &source, child.id())?;
        let plain = ComfyUiInstallOptions {
            runtime_id: None,
            reinstall: false,
            dry_run: true,
        };

        let preview = install(&paths, &config, plain.clone());
        let result = install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                dry_run: false,
                ..plain
            },
        );
        let _ = child.kill();
        let _ = child.wait();

        let preview = preview?;
        assert!(!preview.contains("is running"), "{preview}");
        let rendered = result?;
        assert!(rendered.contains("  installed: yes\n"), "{rendered}");
        assert_eq!(
            fs::read_to_string(source.join("main.py"))?,
            "old main",
            "a plain install reuses the existing code"
        );
        remove_test_root(&paths);
        Ok(())
    }

    /// A saved state that cannot be read does not block a reinstall, which
    /// says so: `stop` cannot read it either, so nothing would clear a
    /// refusal.
    #[test]
    fn a_reinstall_proceeds_over_an_unreadable_saved_state() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-reinstall-corrupt-state")?;
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        fs::write(&archive_path, "not a gzip stream")?;
        let state = state_path(&paths);
        fs::create_dir_all(state.parent().context("state path has no parent")?)?;
        fs::write(&state, "{\"app_id\": \"comfy")?;
        assert!(
            stop(&paths).is_err(),
            "the premise: `stop` cannot clear this state"
        );

        let preview = install(&paths, &config, reinstall_options(true))?;
        assert!(
            preview.contains(&format!("  note: failed to parse {}", state.display())),
            "{preview}"
        );
        let error = install(&paths, &config, reinstall_options(false))
            .expect_err("the planted archive is unreadable");
        assert!(
            format!("{error:#}").contains("failed to extract"),
            "an unreadable state must not block the reinstall: {error:#}"
        );
        remove_test_root(&paths);
        Ok(())
    }

    /// Without a release record the dry run says that entries named like the
    /// new release's code are replaced; with one, that they are set aside.
    /// The lines are pinned here; `source_swap`'s
    /// `install_from_before_the_record_lists_what_it_replaced` and
    /// `user_entry_named_like_new_release_code_is_set_aside` prove them.
    #[test]
    fn reinstall_dry_run_says_whether_a_colliding_entry_is_replaced_or_set_aside() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-dry-run-collisions")?;
        let source = source_path_from_app_root(&app_root);

        let without_record = install(&paths, &config, reinstall_options(true))?;
        assert!(
            without_record.contains(
                "  replaces: anything in that folder named like code the new release ships \
                 (this install has no record of what its release shipped)\n"
            ),
            "{without_record}"
        );
        assert!(
            without_record.contains("  leaves in place: everything else in that folder\n"),
            "{without_record}"
        );
        assert!(!without_record.contains("sets aside"), "{without_record}");

        fs::write(
            source.join(source_swap::RELEASE_RECORD),
            "comfy\nmain.py\nrequirements.txt",
        )?;
        let with_record = install(&paths, &config, reinstall_options(true))?;
        assert!(
            with_record.contains(&format!(
                "  sets aside: anything of yours named like code the new release adds, renamed \
                 to <name>{}<time>\n",
                source_swap::SET_ASIDE_INFIX
            )),
            "{with_record}"
        );
        assert!(!with_record.contains("  replaces:"), "{with_record}");
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }
}

/// Property tests for the ComfyUI source unpack, judged on the filesystem by
/// `crate::archive_props`. The archive is pre-placed where the installer
/// caches it, so `download_and_extract_source` reads it instead of fetching.
#[cfg(all(test, unix))]
mod archive_properties {
    use std::fs;
    use std::path::Path;

    use super::{
        COMFYUI_SOURCE_ARCHIVE_NAME, download_and_extract_source, models_folder_for_source,
        source_path_from_app_root,
    };
    use proptest::prelude::*;

    use crate::archive_props::{
        ESCAPES_ONLY, Entry, Expect, Kind, Layout, Modes, entries, note_mode_scope, run_case,
        run_property, run_round_trip, tar_gz_bytes,
    };

    /// The unpack plus the first write `install` makes into the result: the
    /// models folder under the source tree.
    fn unpack_then_first_install_write(layout: &Layout, archive: &Path) -> anyhow::Result<()> {
        let app_root = &layout.dest;
        let cached = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(cached.parent().unwrap())?;
        fs::copy(archive, &cached)?;
        let mut log = fs::File::create(layout.base.join("install.log"))?;
        let source_path = source_path_from_app_root(app_root);
        download_and_extract_source(app_root, &source_path, &mut log, &|| Ok(()))?;
        fs::create_dir_all(models_folder_for_source(&source_path))?;
        Ok(())
    }

    /// Whether an entry name or link target is absolute in the generator's
    /// terms: a leading `/`, which [`concretize`](crate::archive_props::concretize)
    /// re-roots at the case's fake root, or a leading `@`, the `@OUT`
    /// placeholder it replaces with the absolute `outside/` path. Moving such
    /// a path under `a/` would change what it names, so both
    /// [`moved_one_level_down`] and [`entries_without_top_level_symlink`] leave
    /// it alone.
    fn is_absolute(path: &[u8]) -> bool {
        path.starts_with(b"/") || path.starts_with(b"@")
    }

    /// `entry` moved under `a/`, unless its name is absolute (re-rooted, it
    /// already lands deep inside the extraction root). A relative symlink
    /// target resolves from the link's own directory, now one level deeper,
    /// so it gains exactly one `../`, whatever the link's depth. A target that
    /// climbed out of the extraction root then lands where it did unmoved:
    /// `../outside` still escapes rather than stopping at `dest/outside`. One
    /// that stayed inside still stays inside, though for a nested link not
    /// always at the same path.
    fn moved_one_level_down(mut entry: Entry) -> Entry {
        if is_absolute(&entry.name) {
            return entry;
        }
        entry.name.splice(0..0, *b"a/");
        if let Kind::Symlink(target) = &mut entry.kind
            && !is_absolute(target)
        {
            target.splice(0..0, *b"../");
        }
        entry
    }

    /// [`entries`] with every relative name moved one level down, under
    /// `a/`, and every symlink dropped that could still land directly in the
    /// extraction root: the shape that hits the known #522 defect
    /// (`first_child_dir` follows it). A symlink is dropped when its name,
    /// read as text, has at most one component, or when it walks through an
    /// earlier symlink's name, which could point anywhere. Absolute names are
    /// left alone: re-rooted, they land deep inside the extraction root.
    /// Moving names down rather than only dropping top-level links keeps
    /// "link, then write through it" in play, one level down where the
    /// source tree is; a moved symlink's relative target gains a `../`, so an
    /// escaping target still escapes, to the same place.
    fn entries_without_top_level_symlink() -> impl Strategy<Value = Vec<Entry>> {
        /// Lexical components, or `None` when the name climbs above the root.
        fn components(name: &[u8]) -> Option<Vec<&[u8]>> {
            let mut out = Vec::new();
            for part in name.split(|byte| *byte == b'/') {
                match part {
                    b"" | b"." => {}
                    b".." => {
                        out.pop()?;
                    }
                    _ => out.push(part),
                }
            }
            Some(out)
        }
        entries().prop_map(|entries| {
            let mut links: Vec<Vec<&[u8]>> = Vec::new();
            let mut kept = Vec::new();
            let moved: Vec<Entry> = entries.into_iter().map(moved_one_level_down).collect();
            for entry in &moved {
                if matches!(entry.kind, Kind::Symlink(_)) && !is_absolute(&entry.name) {
                    let Some(parts) = components(&entry.name) else {
                        continue;
                    };
                    if parts.len() <= 1 || links.iter().any(|link| parts.starts_with(link)) {
                        continue;
                    }
                    links.push(parts);
                }
                kept.push(entry.clone());
            }
            kept
        })
    }

    /// Moving an entry down must not change whether its symlink escapes the
    /// extraction root, or where to: the target, resolved lexically from the
    /// link's directory, escapes before exactly when it escapes after, and
    /// then names the same place. A top-level link names the same place
    /// whether or not it escapes. A nested link is checked too, so the single
    /// `../` is shown to be enough at any depth. Absolute names and targets,
    /// in either spelling, are not touched.
    #[test]
    fn moving_an_entry_down_keeps_its_symlink_target_in_place() {
        /// `dir` joined with `target`, `..` applied lexically; a leading `..`
        /// that climbs above the root is kept.
        fn resolve(dir: &[u8], target: &[u8]) -> Vec<Vec<u8>> {
            let mut out: Vec<Vec<u8>> = Vec::new();
            for part in dir
                .split(|b| *b == b'/')
                .chain(target.split(|b| *b == b'/'))
            {
                match part {
                    b"" | b"." => {}
                    b".." if out.last().is_some_and(|last| last != b"..") => {
                        out.pop();
                    }
                    _ => out.push(part.to_vec()),
                }
            }
            out
        }
        for (name, dir) in [
            (&b"l"[..], &b""[..]),
            (b"sub/l", b"sub"),
            (b"sub/deeper/l", b"sub/deeper"),
        ] {
            for target in [
                &b"../outside"[..],
                b"../../outside",
                b"../../../outside",
                b"../victim",
                b".",
                b"l/../..",
                b"a",
            ] {
                let moved = moved_one_level_down(Entry {
                    name: name.to_vec(),
                    kind: Kind::Symlink(target.to_vec()),
                    mode: 0o777,
                });
                assert_eq!(moved.name, [&b"a/"[..], name].concat());
                let Kind::Symlink(moved_target) = moved.kind else {
                    panic!("still a symlink");
                };
                let label = format!(
                    "{} -> {} moved to {}",
                    String::from_utf8_lossy(name),
                    String::from_utf8_lossy(target),
                    String::from_utf8_lossy(&moved_target)
                );
                // Exactly one `../`, whatever the link's depth: the move adds
                // one directory level, not one per level already there.
                assert_eq!(moved_target, [&b"../"[..], target].concat(), "{label}");
                let before = resolve(dir, target);
                let after = resolve(&[&b"a/"[..], dir].concat(), &moved_target);
                let escapes = |path: &[Vec<u8>]| path.first().is_some_and(|part| part == b"..");
                assert_eq!(escapes(&after), escapes(&before), "{label}");
                if escapes(&before) || dir.is_empty() {
                    assert_eq!(after, before, "{label}");
                }
            }
        }
        // Absolute targets and absolute names are left alone, for both
        // spellings of absolute: a leading `/` and the `@OUT` placeholder.
        for target in [&b"@OUT/sentinel"[..], b"/etc/passwd"] {
            let moved = moved_one_level_down(Entry {
                name: b"l".to_vec(),
                kind: Kind::Symlink(target.to_vec()),
                mode: 0o777,
            });
            assert_eq!(moved.name, b"a/l");
            assert!(
                matches!(&moved.kind, Kind::Symlink(t) if t == target),
                "target {} was rewritten",
                String::from_utf8_lossy(target)
            );
        }
        for name in [&b"/x"[..], b"@OUT/x"] {
            assert_eq!(
                moved_one_level_down(Entry {
                    name: name.to_vec(),
                    kind: Kind::Dir,
                    mode: 0o755
                })
                .name,
                name,
                "name {} was moved",
                String::from_utf8_lossy(name)
            );
            // An absolute-named symlink is left whole: its relative target
            // does not gain the `../` a moved link gets.
            let link = moved_one_level_down(Entry {
                name: name.to_vec(),
                kind: Kind::Symlink(b"t".to_vec()),
                mode: 0o777,
            });
            assert_eq!(link.name, name, "name {}", String::from_utf8_lossy(name));
            assert!(
                matches!(&link.kind, Kind::Symlink(t) if t == b"t"),
                "target of {} was rewritten",
                String::from_utf8_lossy(name)
            );
        }
    }

    fn property(
        label: &str,
        strategy: impl Strategy<Value = Vec<Entry>>,
        expect: Expect,
        source_must_be_real_dir: bool,
    ) -> Result<(), String> {
        note_mode_scope(label, expect.modes);
        run_property(label, 512, strategy, Some(5), |layout, entries| {
            let log = layout.base.join("install.log");
            let outcome = run_case(
                layout,
                "source.tar.gz",
                &tar_gz_bytes(entries),
                &[&layout.dest, &log],
                &layout.dest,
                unpack_then_first_install_write,
            );
            outcome.judge(expect)?;
            let source = source_path_from_app_root(&layout.dest);
            if source_must_be_real_dir
                && outcome.error.is_none()
                && fs::symlink_metadata(&source).is_ok_and(|meta| meta.file_type().is_symlink())
            {
                return Err(format!(
                    "the installed source path is a symlink to {:?}: {outcome:#?}",
                    fs::read_link(&source)
                ));
            }
            Ok(())
        })
    }

    /// What holds today: with the #522 shape (a top-level symlink) left out,
    /// the unpack and the first install write stay inside the app root.
    #[test]
    fn source_unpack_without_a_top_level_symlink_never_writes_outside_the_app_root() {
        property(
            "comfyui-no-top-link",
            entries_without_top_level_symlink(),
            ESCAPES_ONLY,
            true,
        )
        .unwrap();
    }

    /// The other direction: a well-formed source archive is installed in
    /// full, its single top-level directory becoming `source/`.
    #[test]
    fn source_unpack_installs_a_benign_archive_as_the_source_dir() {
        run_round_trip("comfyui-benign", |layout, entries| {
            let archive = layout.base.join("source.tar.gz");
            fs::write(&archive, tar_gz_bytes(entries))?;
            unpack_then_first_install_write(layout, &archive)?;
            Ok(source_path_from_app_root(&layout.dest))
        })
        .unwrap();
    }

    /// Known defect: `first_child_dir` tests `is_dir`, which follows symlinks,
    /// and the result is renamed to `source/` as-is. An archive whose first
    /// top-level entry is a symlink therefore installs `source` as a link out
    /// of the app root, and the very next install step creates `models/`
    /// through it. Minimal shrunk counterexample: one entry, `l -> ..`.
    #[test]
    #[ignore = "known gap: a top-level symlink in the source archive becomes the install's source dir (#522)"]
    fn source_unpack_never_writes_outside_the_app_root() {
        property("comfyui", entries(), ESCAPES_ONLY, false).unwrap();
    }

    #[test]
    #[ignore = "known gap: a top-level symlink in the source archive becomes the install's source dir (#522)"]
    fn installed_source_path_is_a_real_directory() {
        property("comfyui-source-dir", entries(), ESCAPES_ONLY, true).unwrap();
    }

    /// Known gap: `tar::Archive::unpack` applies `mode & 0o777` (setuid and
    /// setgid are dropped) without the process umask, so a `0777`/`0666`
    /// entry stays world-writable. Shrunk counterexample: one file, mode 0777.
    /// The top-level symlink shape is left out, so this fails for the modes
    /// alone and can be un-ignored on its own when they are fixed.
    #[test]
    #[ignore = "known gap: the ComfyUI unpack keeps world-writable modes from the archive (#522)"]
    fn source_unpack_leaves_no_setuid_or_world_writable_entry() {
        property(
            "comfyui-modes",
            entries_without_top_level_symlink(),
            Expect {
                modes: Modes::Safe,
                no_outward_links: false,
            },
            false,
        )
        .unwrap();
    }

    /// Deterministic form of the shrunk counterexample: a single top-level
    /// symlink to the sibling `outside/` directory. After the unpack and the
    /// first install write, nothing may exist under `outside/` and `source`
    /// must be a real directory.
    #[test]
    #[ignore = "known gap: a top-level symlink in the source archive becomes the install's source dir (#522)"]
    fn top_level_symlink_in_source_archive_does_not_redirect_the_install() {
        use crate::archive_props::{OUTSIDE_TOKEN, concretize};
        let layout = Layout::new();
        let entries = concretize(
            &[Entry {
                name: b"ComfyUI-master".to_vec(),
                kind: Kind::Symlink(OUTSIDE_TOKEN.as_bytes().to_vec()),
                mode: 0o777,
            }],
            &layout,
        );
        let archive = layout.base.join("source.tar.gz");
        fs::write(&archive, tar_gz_bytes(&entries)).unwrap();
        let result = unpack_then_first_install_write(&layout, &archive);
        let source = source_path_from_app_root(&layout.dest);
        assert!(
            !layout.outside.join("models").exists(),
            "the install created models/ outside the app root (result: {result:?}, source -> {:?})",
            fs::read_link(&source)
        );
        assert!(
            !fs::symlink_metadata(&source).is_ok_and(|meta| meta.file_type().is_symlink()),
            "source is a symlink to {:?}",
            fs::read_link(&source)
        );
    }
}
