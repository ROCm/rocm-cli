// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

use crate::direct_llama::find_binary_in;
use crate::install::{LemonadeInstallManifest, platform_binary_name};
use crate::process::{
    LemonadeProcessEnvironment, apply_lemonade_process_environment, hide_child_console_window,
    lemonade_process_environment, run_lemonade_backend_install, spawn_lemond,
    wait_for_lemonade_cli_status,
};
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
        let listing = run_lemonade_backends_list(manifest, &process_env)?;
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
    let listing = run_lemonade_backends_list(manifest, process_env)?;
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
            run_lemonade_backend_install(manifest, host, port, &backend, process_env)
        })?;
    }
    Ok(backend)
}

/// Parse `lemonade backends` table output into `(backend, status)` pairs for the
/// `llamacpp` recipe only. Status is one of `installed`/`installable`/`unsupported`.
///
/// Lemonade also reports two update states, read here as the state they act
/// like: `update_available` (an installed build that still runs, with a newer
/// release out) as `installed`, and `update_required` (an installed build
/// Lemonade will not run until it is reinstalled, e.g. after the pin in
/// `backend_versions.json` moved) as `installable`, so it is reinstalled rather
/// than reused or dropped. The other state Lemonade 2026.44.1 defines,
/// `action_required` (used by other recipes), leaves the row out.
///
/// A row that starts in the first column names its recipe; an indented row
/// continues the recipe above it with another backend. The recipe is read from
/// the layout rather than matched against a list of known recipe names: each
/// Lemonade release adds recipes (`llamacpp-hrx`, `ds4`, `thenoise` in
/// 2026.44), and a row for a recipe missing from such a list used to be read as
/// one more `llamacpp` backend, along with every indented row below it.
fn parse_llamacpp_backend_statuses(output: &str) -> Vec<(String, String)> {
    fn is_state(token: &str) -> bool {
        token.starts_with("update_available")
            || matches!(
                token,
                "installed" | "installable" | "update_required" | "unsupported" | "action_required"
            )
    }
    fn usable_status(token: &str) -> Option<&'static str> {
        // Lemonade pads the status column to 16 characters with no separator
        // after it, so the one 16-character state runs straight into the
        // message (`update_availableNewer upstream release ...`).
        if token.starts_with("update_available") {
            return Some("installed");
        }
        match token {
            "installed" => Some("installed"),
            "installable" | "update_required" => Some("installable"),
            "unsupported" => Some("unsupported"),
            _ => None,
        }
    }
    let mut current_recipe = String::new();
    let mut result = Vec::new();
    for line in output.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&first) = tokens.first() else {
            continue;
        };
        if first == "Recipe" || first.starts_with("---") {
            continue;
        }
        let continues_recipe = line.starts_with(char::is_whitespace);
        let (backend, rest) = if continues_recipe {
            (first, &tokens[1..])
        } else {
            first.clone_into(&mut current_recipe);
            match tokens.get(1) {
                Some(backend) => (*backend, &tokens[2..]),
                None => continue,
            }
        };
        if current_recipe != LLAMACPP_RECIPE {
            continue;
        }
        // The first state token is the Status column; a state word later in
        // the row is message text.
        let Some(state) = rest.iter().find(|token| is_state(token)) else {
            continue;
        };
        if let Some(status) = usable_status(state) {
            result.push((backend.to_owned(), status.to_owned()));
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
) -> Result<String> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .arg("backends")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_lemonade_process_environment(&mut command, process_env)?;
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

/// Like [`crate::direct_llama::find_llama_server_binary`], but scoped to the single backend Lemonade
/// actually attempted (`manifest.backend_name`, one of `LLAMACPP_BACKEND_PRIORITY`'s
/// `"rocm"` / `"vulkan"`), rather than scanning every known backend directory in
/// priority order. Lemonade reports the generic `"rocm"` — never the more specific
/// `rocm-stable`/`rocm-nightly` — but may extract the actual build into either
/// versioned directory, so `"rocm"` maps to the whole ROCm family; `"vulkan"` never
/// does. This is what lets alignment verification reject a stale ROCm directory left
/// by an earlier attempt when this round actually picked `vulkan` (or the reverse).
pub(crate) fn find_llama_server_binary_for_backend(
    manifest: &LemonadeInstallManifest,
    backend_name: &str,
) -> Option<PathBuf> {
    let llamacpp_dir = manifest.runtime_dir.join("bin").join("llamacpp");
    let binary = platform_binary_name("llama-server");
    let family: &[&str] = if backend_name == ROCM_BACKEND_NAME {
        &ROCM_LLAMACPP_BACKEND_DIRS
    } else {
        &["vulkan"]
    };
    family
        .iter()
        .find_map(|backend| find_binary_in(&llamacpp_dir.join(backend), &binary))
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

    /// `lemonade backends` as Lemonade 2026.44.1 prints it on a Linux host with
    /// an AMD GPU: a `Tier` and a `Format` column, and recipes (`ds4`,
    /// `llamacpp-hrx`, `thenoise`) no earlier release listed.
    const LEMONADE_2026_44_BACKENDS_TABLE: &str = "\
Recipe              Backend     Tier          Format     Status          Message/Version                               Action
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
acestep             rocm        experimental  native     installable     Backend is supported but not installed.        lemonade backends install acestep:rocm
                    vulkan      experimental  native     installable     Backend is supported but not installed.        lemonade backends install acestep:vulkan
ds4                 rocm        guest         native     installable     Backend is supported but not installed.        lemonade backends install ds4:rocm
flm                 npu         core          native     installable     Backend is supported but not installed.        lemonade backends install flm:npu
kokoro              cpu         core          native     installable     Backend is supported but not installed.        lemonade backends install kokoro:cpu
llamacpp            cpu         core          native     installable     Backend is supported but not installed.        lemonade backends install llamacpp:cpu
                    rocm        core          native     installable     Backend is supported but not installed.        lemonade backends install llamacpp:rocm
                    vulkan      core          native     installable     Backend is supported but not installed.        lemonade backends install llamacpp:vulkan
llamacpp-hrx        hrx         experimental  native     installable     Backend is supported but not installed.        lemonade backends install llamacpp-hrx:hrx
moonshine           cpu         core          python     installable     Backend is supported but not installed.        lemonade backends install moonshine:cpu
onnxruntime         cpu         experimental  native     installable     Backend is supported but not installed.        lemonade backends install onnxruntime:cpu
openmoss            vulkan      experimental  native     installable     Backend is supported but not installed.        lemonade backends install openmoss:vulkan
sd-cpp              cpu         core          native     installable     Backend is supported but not installed.        lemonade backends install sd-cpp:cpu
                    rocm        core          native     installable     Backend is supported but not installed.        lemonade backends install sd-cpp:rocm
                    vulkan      core          native     installable     Backend is supported but not installed.        lemonade backends install sd-cpp:vulkan
thenoise            rocm        core          python     installable     Backend is supported but not installed.        lemonade backends install thenoise:rocm
thinksound          rocm        experimental  native     installable     Backend is supported but not installed.        lemonade backends install thinksound:rocm
                    vulkan      experimental  native     installable     Backend is supported but not installed.        lemonade backends install thinksound:vulkan
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
";

    #[test]
    fn parses_only_llamacpp_rows_from_the_2026_44_table() {
        let backends = parse_llamacpp_backend_statuses(LEMONADE_2026_44_BACKENDS_TABLE);
        assert_eq!(
            backends,
            vec![
                ("cpu".to_owned(), "installable".to_owned()),
                ("rocm".to_owned(), "installable".to_owned()),
                ("vulkan".to_owned(), "installable".to_owned()),
            ]
        );
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("rocm".to_owned(), false))
        );
    }

    #[test]
    fn another_recipes_backend_is_never_read_as_a_llamacpp_backend() {
        // Plain `lemonade backends` (no `--all`) leaves unsupported backends
        // out, so on a host with no GPU backend for llama.cpp its only row is
        // `cpu`. The indented `vulkan` row belongs to `openmoss`; read as a
        // `llamacpp` backend, it would select a GPU backend Lemonade does not
        // offer for llama.cpp here.
        let output = "\
Recipe              Backend     Tier          Format     Status          Message/Version                               Action
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
llamacpp            cpu         core          native     installable     Backend is supported but not installed.        lemonade backends install llamacpp:cpu
openmoss            rocm        experimental  native     installable     Backend is supported but not installed.        lemonade backends install openmoss:rocm
                    vulkan      experimental  native     installable     Backend is supported but not installed.        lemonade backends install openmoss:vulkan
";
        let backends = parse_llamacpp_backend_statuses(output);
        assert_eq!(backends, vec![("cpu".to_owned(), "installable".to_owned())]);
        assert_eq!(select_best_llamacpp_backend(&backends), None);
    }

    #[test]
    fn update_states_read_as_the_state_they_act_like() {
        // `update_required`: the installed build will not run until it is
        // reinstalled, so it must be installed again, not reused. An
        // `update_available` build still runs and is reused; Lemonade prints
        // that 16-character state with no space before the message. An
        // `action_required` row is left out, whatever words its message holds.
        let output = "\
Recipe              Backend     Tier          Format     Status          Message/Version                               Action
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
llamacpp            rocm        core          native     update_required Backend update is required before use.         lemonade backends install llamacpp:rocm
                    vulkan      core          native     update_availableNewer upstream release available: b11400       lemonade backends install llamacpp:vulkan
                    system      core          native     action_required Use the installed driver instead.              -
";
        let backends = parse_llamacpp_backend_statuses(output);
        assert_eq!(
            backends,
            vec![
                ("rocm".to_owned(), "installable".to_owned()),
                ("vulkan".to_owned(), "installed".to_owned()),
            ]
        );
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("rocm".to_owned(), false)),
            "an update_required rocm backend is reinstalled, not dropped for vulkan"
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
