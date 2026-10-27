// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

use crate::install::{LemonadeInstallManifest, platform_binary_name};
use crate::process::{
    LemonadeProcessEnvironment, apply_lemonade_process_environment_with_parent,
    hide_child_console_window, lemonade_process_environment, run_lemonade_backend_install,
    spawn_lemond, wait_for_lemonade_cli_status,
};
use crate::runtime_dir::ParentRuntimeEnvironment;
use crate::state::{free_local_port, terminate_pid};
use crate::{DEFAULT_HOST, LLAMACPP_RECIPE, ROCM_BACKEND_NAME};

/// Preferred llama.cpp backends, best first. Lemonade reports per-GPU support;
/// we pick the highest-priority backend it considers supported on this host.
/// GPU backends only — `cpu` is intentionally excluded so the router path never
/// selects CPU under the GPU-required policy (AGENTS.md §6, matching
/// `DIRECT_LLAMA_SERVER_BACKENDS`). When only CPU is usable, selection returns
/// `None` and the caller fails with actionable guidance.
const LLAMACPP_BACKEND_PRIORITY: [&str; 2] = ["rocm", "vulkan"];

pub(crate) fn install_best_llamacpp_backend(
    manifest: &mut LemonadeInstallManifest,
    force_reinstall: bool,
) -> Result<()> {
    let port = free_local_port()?;
    let log_path_buf = manifest.runtime_dir.join("install-lemond.log");
    let log_path = Some(log_path_buf.as_path());
    let process_env = lemonade_process_environment()?;
    let mut child = spawn_lemond(manifest, DEFAULT_HOST, port, log_path, &process_env)?;
    let result = (|| -> Result<String> {
        wait_for_lemonade_cli_status(
            manifest,
            DEFAULT_HOST,
            port,
            Duration::from_secs(30),
            log_path,
            &process_env,
        )?;
        ensure_best_llamacpp_backend(manifest, DEFAULT_HOST, port, &process_env, force_reinstall)
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    manifest.backend_name = result?;
    Ok(())
}

/// Which llama.cpp backend Lemonade would select on this host
/// (`LLAMACPP_BACKEND_PRIORITY`), without installing anything -- mirrors
/// [`install_best_llamacpp_backend`]'s spawn/query steps but stops short of the
/// install, so the ROCm backend alignment dance can check whether it is even
/// reachable before patching `backend_versions.json` or forcing a single reinstall.
pub(crate) fn best_llamacpp_backend_for_host(
    manifest: &LemonadeInstallManifest,
) -> Result<Option<String>> {
    let port = free_local_port()?;
    let log_path_buf = manifest.runtime_dir.join("install-lemond.log");
    let log_path = Some(log_path_buf.as_path());
    let process_env = lemonade_process_environment()?;
    let mut child = spawn_lemond(manifest, DEFAULT_HOST, port, log_path, &process_env)?;
    let result = (|| -> Result<Option<String>> {
        wait_for_lemonade_cli_status(
            manifest,
            DEFAULT_HOST,
            port,
            Duration::from_secs(30),
            log_path,
            &process_env,
        )?;
        let listing = run_lemonade_backends_list(
            manifest,
            &process_env,
            &ParentRuntimeEnvironment::current(),
        )?;
        let backends = parse_llamacpp_backend_statuses(&listing);
        Ok(select_best_llamacpp_backend(&backends).map(|(name, _)| name))
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    result
}

/// Ask Lemonade which llama.cpp backends it supports on this GPU, choose the best
/// one (`LLAMACPP_BACKEND_PRIORITY`), install it if necessary, and return its name.
/// Retry the backend install itself once, rather than the whole Lemonade runtime
/// preparation. The embeddable download already has bounded transport retries;
/// repeating that outer operation would redo deterministic failures and may
/// re-extract a healthy runtime. A backend subprocess can instead fail after a
/// completed download when its connection to lemond is interrupted, and a second
/// call can reuse the backend cache immediately without a delay.
pub(crate) fn install_llamacpp_backend_with_retry(
    mut install: impl FnMut() -> Result<()>,
) -> Result<()> {
    match install() {
        Ok(()) => Ok(()),
        Err(first_error) => {
            eprintln!("Lemonade backend installation failed; retrying once: {first_error:#}");
            install().with_context(|| {
                "Lemonade backend installation failed again; run `rocm engines install \
                 lemonade --reinstall` and then retry `rocm serve`"
            })
        }
    }
}

pub(crate) fn ensure_best_llamacpp_backend(
    manifest: &mut LemonadeInstallManifest,
    host: &str,
    port: u16,
    process_env: &LemonadeProcessEnvironment,
    force_reinstall: bool,
) -> Result<String> {
    ensure_best_llamacpp_backend_with_parent(
        manifest,
        host,
        port,
        process_env,
        force_reinstall,
        &ParentRuntimeEnvironment::current(),
    )
}

/// [`ensure_best_llamacpp_backend`] with the parent runtime environment supplied
/// by the caller, so a serve that was given a confined parent never reads or
/// creates anything in the host's own.
pub(crate) fn ensure_best_llamacpp_backend_with_parent(
    manifest: &mut LemonadeInstallManifest,
    host: &str,
    port: u16,
    process_env: &LemonadeProcessEnvironment,
    force_reinstall: bool,
    parent: &ParentRuntimeEnvironment,
) -> Result<String> {
    let listing = run_lemonade_backends_list(manifest, process_env, parent)?;
    let backends = parse_llamacpp_backend_statuses(&listing);
    let Some((backend, already_installed)) = select_best_llamacpp_backend(&backends) else {
        bail!(
            "Lemonade reports no supported GPU llama.cpp backend for this host (status: {}). \
             The GPU-required policy does not fall back to CPU; install a ROCm or Vulkan \
             backend (e.g. `rocm engines install lemonade`) or verify the GPU driver with \
             `rocm examine`.",
            describe_llamacpp_backends(&backends)
        );
    };
    // Record the attempted backend before the fallible install step below, so a caller
    // verifying the result can scope its check to this backend even when the install
    // itself errors out.
    manifest.backend_name = backend.clone();
    if already_installed && !force_reinstall {
        eprintln!("Using installed Lemonade {LLAMACPP_RECIPE}:{backend} backend.");
    } else {
        eprintln!("Installing Lemonade {LLAMACPP_RECIPE}:{backend} backend...");
        install_llamacpp_backend_with_retry(|| {
            run_lemonade_backend_install(manifest, host, port, &backend, process_env, parent)
        })?;
    }
    Ok(backend)
}

/// Parse `lemonade backends` table output into `(backend, status)` pairs for the
/// `llamacpp` recipe only. Status is one of `installed`/`installable`/`unsupported`.
fn parse_llamacpp_backend_statuses(output: &str) -> Vec<(String, String)> {
    const RECIPES: [&str; 8] = [
        "flm",
        "kokoro",
        "llamacpp",
        "ryzenai-llm",
        "sd-cpp",
        "vllm",
        "whispercpp",
        "embeddings",
    ];
    const STATUSES: [&str; 3] = ["installed", "installable", "unsupported"];
    let mut current_recipe = "";
    let mut result = Vec::new();
    for line in output.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&first) = tokens.first() else {
            continue;
        };
        if first == "Recipe" || first.starts_with("---") {
            continue;
        }
        // A row either starts with a recipe name (recipe + its first backend) or,
        // for grouped recipes, with a backend name continuing the previous recipe.
        let (backend, rest) = if RECIPES.contains(&first) {
            current_recipe = match RECIPES.iter().find(|r| **r == first) {
                Some(r) => r,
                None => current_recipe,
            };
            match tokens.get(1) {
                Some(backend) => (*backend, &tokens[2..]),
                None => continue,
            }
        } else {
            (first, &tokens[1..])
        };
        if current_recipe != LLAMACPP_RECIPE {
            continue;
        }
        if let Some(status) = rest.iter().find(|t| STATUSES.contains(t)) {
            result.push((backend.to_owned(), (*status).to_owned()));
        }
    }
    result
}

/// Choose the highest-priority backend Lemonade considers usable (installed or
/// installable). Returns `(backend, already_installed)`.
fn select_best_llamacpp_backend(backends: &[(String, String)]) -> Option<(String, bool)> {
    for candidate in LLAMACPP_BACKEND_PRIORITY {
        if let Some((name, status)) = backends
            .iter()
            .find(|(b, s)| b == candidate && (s == "installed" || s == "installable"))
        {
            return Some((name.clone(), status == "installed"));
        }
    }
    None
}

fn describe_llamacpp_backends(backends: &[(String, String)]) -> String {
    if backends.is_empty() {
        return "none reported".to_owned();
    }
    backends
        .iter()
        .map(|(b, s)| format!("{b}={s}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_lemonade_backends_list(
    manifest: &LemonadeInstallManifest,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
) -> Result<String> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .arg("backends")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_lemonade_process_environment_with_parent(&mut command, process_env, parent)?;
    hide_child_console_window(&mut command);
    let output = command
        .output()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !output.status.success() {
        bail!(
            "Lemonade backends query failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The ROCm-family llama.cpp backend directory names under `bin/llamacpp/`. Lemonade's
/// installer extracts each build into its own build-numbered subdirectory rather than
/// replacing one in place, so all three names can accumulate across attempts.
pub(crate) const ROCM_LLAMACPP_BACKEND_DIRS: [&str; 3] = ["rocm-stable", "rocm-nightly", "rocm"];

/// The `bin/llamacpp/` directories that may hold the `llama-server` for a manifest's
/// selected backend, or `None` when that backend cannot be served directly. Only
/// `rocm` and `vulkan` are valid manifest backend names: installs record whatever
/// Lemonade selects from its `rocm`/`vulkan` priority list, so the `rocm-stable` and
/// `rocm-nightly` directory names are members of the `rocm` family, never backend
/// names of their own. `cpu` and unknown names fail closed (no CPU fallback).
pub(crate) fn backend_server_dirs(backend_name: &str) -> Option<&'static [&'static str]> {
    match backend_name {
        ROCM_BACKEND_NAME => Some(&ROCM_LLAMACPP_BACKEND_DIRS),
        "vulkan" => Some(&["vulkan"]),
        _ => None,
    }
}

/// Find a GPU llama-server scoped to the backend Lemonade selected
/// (`manifest.backend_name`, one of `LLAMACPP_BACKEND_PRIORITY`'s `"rocm"` /
/// `"vulkan"`), rather than scanning every known backend directory. `rocm` accepts
/// its directory family (`rocm-stable`, `rocm-nightly`, `rocm`); Vulkan never crosses
/// over to a stale ROCm install or the reverse. This is what lets alignment
/// verification reject a stale ROCm directory left by an earlier attempt when this
/// round actually picked `vulkan`. CPU and unknown names fail closed. Both flat and
/// one-level build-numbered extraction layouts are supported.
pub(crate) fn find_llama_server_binary_for_backend(
    manifest: &LemonadeInstallManifest,
    backend_name: &str,
) -> Option<PathBuf> {
    let llamacpp_dir = manifest.runtime_dir.join("bin").join("llamacpp");
    let binary = platform_binary_name("llama-server");
    backend_server_dirs(backend_name)?
        .iter()
        .find_map(|backend| find_binary_in(&llamacpp_dir.join(backend), &binary))
}

/// Look for `binary` directly in `dir`, then one level down in each subdirectory.
pub(crate) fn find_binary_in(dir: &Path, binary: &str) -> Option<PathBuf> {
    let direct = dir.join(binary);
    if direct.is_file() {
        return Some(direct);
    }
    fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .map(|subdir| subdir.join(binary))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use std::fs;

    #[test]
    fn backend_install_succeeds_without_retry() {
        let mut attempts = 0;

        install_llamacpp_backend_with_retry(|| {
            attempts += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(attempts, 1);
    }

    #[test]
    fn backend_install_recovers_on_the_second_attempt() {
        let mut attempts = 0;

        install_llamacpp_backend_with_retry(|| {
            attempts += 1;
            if attempts == 1 {
                bail!("first backend connection was interrupted");
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(attempts, 2);
    }

    #[test]
    fn backend_install_stops_after_one_retry_with_reinstall_guidance() {
        let mut attempts = 0;

        let error = install_llamacpp_backend_with_retry(|| {
            attempts += 1;
            if attempts == 1 {
                bail!("first backend connection was interrupted");
            }
            bail!("second backend connection was interrupted");
        })
        .unwrap_err();
        let rendered = format!("{error:#}");

        assert_eq!(attempts, 2);
        assert!(
            rendered.contains("second backend connection was interrupted"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("first backend connection was interrupted"),
            "the terminal error must be the retry's failure: {rendered}"
        );
        assert!(
            rendered.contains("rocm engines install lemonade --reinstall"),
            "{rendered}"
        );
        assert!(rendered.contains("retry `rocm serve`"), "{rendered}");
    }

    fn test_manifest(runtime_dir: PathBuf) -> LemonadeInstallManifest {
        LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: rocm_deps::LEMONADE_VERSION.to_owned(),
            runtime_dir,
            lemond: PathBuf::from("lemond"),
            lemonade: PathBuf::from("lemonade"),
            backend_recipe: LLAMACPP_RECIPE.to_owned(),
            backend_name: ROCM_BACKEND_NAME.to_owned(),
            installed_at_unix_ms: 0,
        }
    }

    #[test]
    fn parses_llamacpp_backends_from_table() {
        let output = "\
Recipe              Backend     Status          Message/Version                               Action
----------------------------------------------------------------------------------------------------
kokoro              cpu         installable     Backend is supported but not installed.       lemonade backends install kokoro:cpu
                    metal       unsupported     Requires macOS                                -
llamacpp            cpu         installable     Backend is supported but not installed.       lemonade backends install llamacpp:cpu
                    metal       unsupported     Requires macOS                                -
                    rocm        unsupported     Unsupported GPU                               -
                    system      unsupported     Requires Linux                               -
                    vulkan      installable     Backend is supported but not installed.       lemonade backends install llamacpp:vulkan
vllm                rocm        unsupported     Requires Linux                               -
";
        let backends = parse_llamacpp_backend_statuses(output);
        assert_eq!(
            backends,
            vec![
                ("cpu".to_owned(), "installable".to_owned()),
                ("metal".to_owned(), "unsupported".to_owned()),
                ("rocm".to_owned(), "unsupported".to_owned()),
                ("system".to_owned(), "unsupported".to_owned()),
                ("vulkan".to_owned(), "installable".to_owned()),
            ]
        );
    }

    #[test]
    fn selects_vulkan_when_rocm_unsupported() {
        let backends = vec![
            ("cpu".to_owned(), "installable".to_owned()),
            ("rocm".to_owned(), "unsupported".to_owned()),
            ("vulkan".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("vulkan".to_owned(), false))
        );
    }

    #[test]
    fn prefers_installed_rocm_when_supported() {
        let backends = vec![
            ("rocm".to_owned(), "installed".to_owned()),
            ("vulkan".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("rocm".to_owned(), true))
        );
    }

    #[test]
    fn never_selects_cpu_when_no_gpu_backend() {
        // Under the GPU-required policy there is no silent CPU fallback: when only
        // a CPU backend is usable, selection returns nothing and the caller fails.
        let backends = vec![
            ("rocm".to_owned(), "unsupported".to_owned()),
            ("cpu".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(select_best_llamacpp_backend(&backends), None);
    }

    #[test]
    fn selects_nothing_when_all_unsupported() {
        let backends = vec![("rocm".to_owned(), "unsupported".to_owned())];
        assert_eq!(select_best_llamacpp_backend(&backends), None);
    }

    #[test]
    fn find_llama_server_binary_for_backend_scopes_to_rocm_family() {
        let dir = scratch_dir("find-server-scoped-rocm");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        // A stale vulkan install from an earlier attempt must never satisfy a "rocm"
        // scoped lookup.
        let vulkan_dir = llamacpp.join("vulkan");
        fs::create_dir_all(&vulkan_dir).unwrap();
        fs::write(vulkan_dir.join(&server), b"x").unwrap();
        let manifest = test_manifest(runtime_dir);
        assert_eq!(
            find_llama_server_binary_for_backend(&manifest, ROCM_BACKEND_NAME),
            None
        );

        // Any directory in the ROCm family satisfies it, matching the unscoped scan's
        // priority order.
        let nightly_dir = llamacpp.join("rocm-nightly");
        fs::create_dir_all(&nightly_dir).unwrap();
        fs::write(nightly_dir.join(&server), b"x").unwrap();
        assert_eq!(
            find_llama_server_binary_for_backend(&manifest, ROCM_BACKEND_NAME),
            Some(nightly_dir.join(&server))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_llama_server_binary_for_backend_scopes_to_vulkan_only() {
        let dir = scratch_dir("find-server-scoped-vulkan");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        // A stale rocm-stable install from an earlier attempt must never satisfy a
        // "vulkan" scoped lookup, even though it would win the unscoped priority scan.
        let rocm_dir = llamacpp.join("rocm-stable");
        fs::create_dir_all(&rocm_dir).unwrap();
        fs::write(rocm_dir.join(&server), b"x").unwrap();
        let manifest = test_manifest(runtime_dir);
        assert_eq!(
            find_llama_server_binary_for_backend(&manifest, "vulkan"),
            None
        );

        let vulkan_dir = llamacpp.join("vulkan");
        fs::create_dir_all(&vulkan_dir).unwrap();
        fs::write(vulkan_dir.join(&server), b"x").unwrap();
        assert_eq!(
            find_llama_server_binary_for_backend(&manifest, "vulkan"),
            Some(vulkan_dir.join(&server))
        );
        fs::remove_dir_all(&dir).ok();
    }
}
