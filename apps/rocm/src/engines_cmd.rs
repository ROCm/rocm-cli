// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm engines` command handlers and env-root/runtime resolution.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `dispatch()` call site stays byte-identical (`engines(command)`,
//! re-imported via `use crate::engines_cmd::{engine_manages_own_runtime,
//! engines, env_root_for_engine_install, env_root_for_service,
//! runtime_key_for_python, runtime_manifest_for_selector};`). `EnginesCommand`
//! remains in the crate root and is reached via `use crate::EnginesCommand`;
//! `Cli` is not referenced from this file. `engine_manages_own_runtime` is
//! called from `main.rs` and also from `serve_cmd.rs`; `ensure_self_managed_engine_ready`'s
//! only caller is `serve_cmd.rs`; `env_root_for_engine_install`,
//! `env_root_for_service`, `runtime_key_for_python`, and
//! `runtime_manifest_for_selector` are used only from other root-level
//! commands in `main.rs` — all stay `pub(crate)` rather than private either
//! way. `render_engine_inventory_text` is the one exception: its only caller
//! is `engines()` in this same file, not `main.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, default_interactive_shell_program, interactive_terminal,
    prepend_runtime_path, runtime_path_is_same_or_inside, runtime_python_activation_hint,
    runtime_python_env_bin_dir,
};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, EngineMethod, InstallRequest, InstallResponse,
};
use serde::Deserialize;

use crate::EnginesCommand;
use crate::therock;
use crate::{
    apply_app_path_env, engine_request, engine_request_with_env_root, ensure_libatomic_for_torch,
    ensure_libnuma_for_torch, ensure_openmpi_for_vllm, normalize_path_for_compare,
    record_cli_audit_event, recover_setup_runtime_registration,
    render_engine_inventory_text_with_paths, resolve_engine_selection,
    resolve_runtime_selector_to_exact_key, select_runtime_manifest, settle_engine_install,
    validate_engine_selection_runtime, validate_runtime_manifest_for_activation,
};

pub(crate) fn render_engine_inventory_text() -> String {
    let paths = AppPaths::discover().ok();
    render_engine_inventory_text_with_paths(paths.as_ref())
}

fn path_is_same_or_inside(path: &Path, base: &Path) -> bool {
    let path = normalize_path_for_compare(path);
    let base = normalize_path_for_compare(base);
    runtime_path_is_same_or_inside(&path, &base)
}

pub(crate) fn engines(command: EnginesCommand) -> Result<()> {
    match command {
        EnginesCommand::List => {
            print!("{}", render_engine_inventory_text());
            Ok(())
        }
        EnginesCommand::Install {
            engine,
            runtime_id,
            python_version,
            reinstall,
            yes,
        } => {
            let paths = AppPaths::discover()?;
            let mut config = RocmCliConfig::load(&paths)?;
            let runtime_id =
                resolve_engine_install_runtime_id(&paths, &config, &engine, runtime_id)?;
            let env_root = env_root_for_engine_install(&paths, &config, &engine, &runtime_id)?;
            if engine == "vllm" {
                ensure_openmpi_for_vllm(yes)?;
                ensure_libatomic_for_torch(yes);
                ensure_libnuma_for_torch(yes);
            }
            let response = engine_request_with_env_root::<_, InstallResponse>(
                Some(&paths),
                &engine,
                EngineMethod::Install,
                &InstallRequest {
                    runtime_id: runtime_id.clone(),
                    python_version,
                    reinstall,
                    env_root: env_root.clone(),
                },
                env_root.as_deref(),
            )?;
            println!("engine install");
            println!("  engine: {engine}");
            println!("  runtime_id: {runtime_id}");
            println!("  reinstall: {reinstall}");
            println!("  env_id: {}", response.env_id);
            println!("  env_path: {}", response.env_path);
            for warning in &response.warnings {
                println!("  warning: {warning}");
            }
            if response.managed_env == Some(false) {
                println!("  note: external runtime");
            } else {
                let engine_config = config.engine_config_mut(&engine);
                engine_config.last_installed_runtime_id = Some(runtime_id.clone());
                engine_config.last_installed_env_id = Some(response.env_id.clone());
                let mut seeded_preference = false;
                if engine_config.preferred_runtime_id.is_none()
                    && engine_config.preferred_env_id.is_none()
                {
                    engine_config.preferred_env_id = Some(response.env_id.clone());
                    seeded_preference = true;
                }
                config.save(&paths)?;
                let _ = seeded_preference;
            }
            // Settle last, matching `maybe_auto_install_sdk_preferred_engine`. The
            // check blocks then print under the `engine:`/`runtime_id:`/`env_id:`
            // lines they describe instead of above them, and the config bookkeeping
            // above still lands when settling fails — the engine did install; it is
            // the runtime it left behind that is being reported on.
            settle_engine_install(&paths, &engine, &runtime_id, &response)?;
            record_cli_audit_event(
                &paths,
                "engine",
                "engine_install",
                "info",
                format!(
                    "installed engine={} runtime_id={} env_id={} reinstall={}",
                    engine, runtime_id, response.env_id, reinstall
                ),
                None,
            );
            Ok(())
        }
        EnginesCommand::Shell {
            engine,
            runtime_id,
            env_id,
            shell,
        } => engine_shell(
            &engine,
            runtime_id.as_deref(),
            env_id.as_deref(),
            shell.as_deref(),
        ),
    }
}

fn resolve_engine_install_runtime_id(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<String>,
) -> Result<String> {
    if engine_manages_own_runtime(engine) {
        return Ok(runtime_id.unwrap_or_else(|| managed_engine_runtime_id(engine)));
    }
    let Some(selector) = runtime_id
        .or_else(|| config.active_runtime_key.clone())
        .or_else(|| config.default_runtime_id.clone())
    else {
        bail!(
            "no active ROCm runtime is configured; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id"
        );
    };
    resolve_runtime_selector_to_exact_key(paths, &selector, "engine install runtime selection")
}

pub(crate) fn engine_manages_own_runtime(engine: &str) -> bool {
    engine == "lemonade"
}

fn env_root_for_runtime(
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
        return env_root_for_self_managed_engine(paths, config);
    }
    env_root_for_runtime(paths, engine, runtime_id)
}

fn env_root_for_self_managed_engine(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<PathBuf>> {
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

pub(crate) fn env_root_for_service(
    paths: &AppPaths,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<Option<PathBuf>> {
    if env_id.is_some() {
        return Ok(None);
    }
    match runtime_id {
        Some(runtime_id) => env_root_for_runtime(paths, engine, runtime_id),
        None => Ok(None),
    }
}

/// Label recorded for the runtime a self-managing engine installs for itself.
///
/// For `lemonade` this must be the `env_id` its adapter reports, which is
/// derived from the single Lemonade pin — it was previously a hand-written
/// literal and had drifted several minor versions behind what is installed.
fn managed_engine_runtime_id(engine: &str) -> String {
    match engine {
        "lemonade" => format!("lemonade-embeddable-{}", rocm_deps::LEMONADE_VERSION),
        _ => "managed-engine-runtime".to_owned(),
    }
}

pub(crate) fn ensure_self_managed_engine_ready(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    engine: &str,
) -> Result<()> {
    if !engine_manages_own_runtime(engine) {
        return Ok(());
    }
    let runtime_id = managed_engine_runtime_id(engine);
    let env_root = env_root_for_self_managed_engine(paths, config)?;
    let detect = engine_request::<_, DetectResponse>(
        Some(paths),
        engine,
        EngineMethod::Detect,
        &DetectRequest {
            runtime_id: Some(runtime_id.clone()),
            device_filter: None,
        },
    )
    .ok();
    // For a self-managing engine the runtime id *is* the env id its adapter
    // reports for the pinned version, so a version bump leaves an older
    // install detected-but-not-current. Requiring the ids to match makes the
    // bump trigger an install instead of silently keeping the old runtime.
    let installed = detect.as_ref().is_some_and(|detect| {
        detect.installed
            && detect.env_id.as_deref() == Some(runtime_id.as_str())
            && detect_runtime_matches_env_root(detect, env_root.as_deref())
    });
    let response = if installed {
        None
    } else {
        eprintln!("Preparing {engine} for GPU serving...");
        let response = engine_request_with_env_root::<_, InstallResponse>(
            Some(paths),
            engine,
            EngineMethod::Install,
            &InstallRequest {
                runtime_id: runtime_id.clone(),
                python_version: None,
                reinstall: false,
                env_root: env_root.clone(),
            },
            env_root.as_deref(),
        )?;
        // No `settle_engine_install` here. This function returns at the top unless
        // `engine_manages_own_runtime(engine)`, and that is exactly the case
        // `settles_runtime_torch` declines: the runtime holds the engine's own
        // binary, not an interpreter with a torch in it. Calling it would be inert
        // at best, and a call that provably cannot act invites someone to "fix" the
        // gate later.
        Some(response)
    };

    let engine_config = config.engine_config_mut(engine);
    engine_config.last_installed_runtime_id = Some(runtime_id);
    if let Some(response) = response {
        engine_config.last_installed_env_id = Some(response.env_id.clone());
        if engine_config.preferred_runtime_id.is_none() && engine_config.preferred_env_id.is_none()
        {
            engine_config.preferred_env_id = Some(response.env_id);
        }
    }
    config.save(paths)?;
    Ok(())
}

fn detect_runtime_matches_env_root(detect: &DetectResponse, env_root: Option<&Path>) -> bool {
    let Some(env_root) = env_root else {
        return true;
    };
    detect
        .runtime_executable
        .as_deref()
        .map(PathBuf::from)
        .is_some_and(|runtime_executable| path_is_same_or_inside(&runtime_executable, env_root))
}

#[derive(Debug, Clone, Deserialize)]
struct ManagedEngineEnvManifest {
    env_id: String,
    runtime_id: String,
    python_executable: String,
    env_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ResolvedEngineEnv {
    env_id: String,
    runtime_id: String,
    python_executable: String,
    env_path: PathBuf,
    source: String,
}

/// Extra argv, environment, and files needed to make a spawned shell *look*
/// like a managed engine shell.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellPromptShim {
    /// Appended to the shell's argv.
    args: Vec<String>,
    /// Added to the child environment.
    envs: Vec<(String, String)>,
    /// Written before the shell starts, as (path, contents).
    files: Vec<(PathBuf, String)>,
}

/// Work out how to mark `shell_program`'s prompt with `prompt`.
///
/// Passing the marker through the `PS1` *environment variable* does not work:
/// bash assigns `PS1` from `/etc/bash.bashrc` and `~/.bashrc` on every
/// interactive start, so the inherited value is overwritten and the engine shell
/// ends up looking exactly like the shell it was launched from. The marker has to
/// be applied from inside the shell's own startup, after the user's files have
/// run — which is what these shims do.
///
/// Pure: decides *what* to write and *how* to invoke, and leaves the I/O to the
/// caller so the decision can be unit-tested. Returns `None` for shells that
/// cannot be marked safely; the caller's handover banner covers those instead of
/// this failing.
///
/// `original_zdotdir` is the caller's `ZDOTDIR`, if it had one, so the zsh shim
/// can still find the user's real startup files after we redirect `ZDOTDIR` at
/// our own directory.
fn engine_shell_prompt_shim(
    shell_program: &str,
    prompt: &str,
    shim_dir: &Path,
    original_zdotdir: Option<&str>,
) -> Option<ShellPromptShim> {
    // Match on the file stem so `--shell /usr/bin/zsh` and a bare `bash` behave
    // the same. `bash5`-style names are deliberately not matched: guessing wrong
    // is worse than falling back to the banner.
    let stem = Path::new(shell_program)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)?
        .to_ascii_lowercase();

    match stem.as_str() {
        "bash" => {
            let rcfile = shim_dir.join("engine-shell.bash");
            // `--rcfile` replaces ~/.bashrc ONLY -- bash still sources
            // /etc/bash.bashrc itself, so sourcing that here would apply it twice.
            let contents = format!(
                "# Generated by `rocm engines shell`. Sources your own startup file\n\
                 # first, then marks the prompt so this shell is distinguishable.\n\
                 if [ -r \"$HOME/.bashrc\" ]; then . \"$HOME/.bashrc\"; fi\n\
                 PS1='{prompt}'\"$PS1\"\n"
            );
            Some(ShellPromptShim {
                args: vec![
                    "--rcfile".to_owned(),
                    rcfile.display().to_string(),
                    "-i".to_owned(),
                ],
                envs: Vec::new(),
                files: vec![(rcfile, contents)],
            })
        }
        "zsh" => {
            // Redirecting ZDOTDIR makes zsh skip the user's `.zshenv` AND their
            // `.zshrc`. Losing `.zshenv` would silently strip their PATH and
            // exports -- a worse bug than the unmarked prompt -- so both are
            // restored, and the original location is passed through for the shim
            // to read at startup.
            let user_zdotdir = "${ROCM_CLI_ORIG_ZDOTDIR:-$HOME}";
            let zshenv = format!(
                "# Generated by `rocm engines shell`; restores your own .zshenv.\n\
                 __rocm_zdotdir=\"{user_zdotdir}\"\n\
                 [ -r \"$__rocm_zdotdir/.zshenv\" ] && . \"$__rocm_zdotdir/.zshenv\"\n"
            );
            let zshrc = format!(
                "# Generated by `rocm engines shell`. Sources your own .zshrc first,\n\
                 # then marks the prompt so this shell is distinguishable.\n\
                 __rocm_zdotdir=\"{user_zdotdir}\"\n\
                 [ -r \"$__rocm_zdotdir/.zshrc\" ] && . \"$__rocm_zdotdir/.zshrc\"\n\
                 PROMPT='{prompt}'$PROMPT\n"
            );
            let mut envs = vec![("ZDOTDIR".to_owned(), shim_dir.display().to_string())];
            if let Some(original) = original_zdotdir.filter(|value| !value.trim().is_empty()) {
                envs.push(("ROCM_CLI_ORIG_ZDOTDIR".to_owned(), original.to_owned()));
            }
            Some(ShellPromptShim {
                args: Vec::new(),
                envs,
                files: vec![
                    (shim_dir.join(".zshenv"), zshenv),
                    (shim_dir.join(".zshrc"), zshrc),
                ],
            })
        }
        // fish, sh, dash, cmd, PowerShell, anything else: no safe way to inject a
        // marker without taking over startup, so the banner carries the message.
        _ => None,
    }
}

/// Write a [`ShellPromptShim`]'s files, creating the directory if needed.
///
/// The files live under the app's own engine state directory rather than a temp
/// dir: they must outlive this process's setup and stay readable for the whole
/// life of the spawned shell, and a fixed path is regenerated on every run
/// instead of accumulating.
fn write_engine_shell_shim(shim_dir: &Path, shim: &ShellPromptShim) -> Result<()> {
    fs::create_dir_all(shim_dir)
        .with_context(|| format!("failed to create {}", shim_dir.display()))?;
    for (path, contents) in &shim.files {
        fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

fn engine_shell(
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    shell_override: Option<&str>,
) -> Result<()> {
    if !interactive_terminal() {
        bail!("`rocm engines shell` requires an interactive terminal");
    }

    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    let resolved = resolve_engine_env(&paths, &config, engine, runtime_id, env_id)?;
    let shell_program = shell_override
        .map(str::to_owned)
        .or_else(default_interactive_shell_program)
        .context("unable to determine an interactive shell; set --shell or SHELL")?;
    let venv_bin = runtime_python_env_bin_dir(&resolved.env_path);
    let shell_hint = runtime_python_activation_hint(&resolved.env_path);

    println!("engine shell");
    println!("  engine: {engine}");
    println!("  source: {}", resolved.source);
    println!("  env_id: {}", resolved.env_id);
    println!("  runtime_id: {}", resolved.runtime_id);
    println!("  env_path: {}", resolved.env_path.display());
    println!("  python: {}", resolved.python_executable);
    println!("  shell: {shell_program}");
    println!("  activate_hint: {shell_hint}");
    println!("  exit_hint: use `exit` or Ctrl-D to leave the managed env shell");

    let path_with_env = prepend_runtime_path(&venv_bin, std::env::var_os("PATH").as_deref())
        .context("failed to compose PATH for managed engine env shell")?;
    let mut command = ProcessCommand::new(&shell_program);
    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("VIRTUAL_ENV", &resolved.env_path)
        .env("PATH", path_with_env)
        .env("ROCM_CLI_ENGINE", engine)
        .env("ROCM_CLI_ENV_ID", &resolved.env_id)
        .env("ROCM_CLI_RUNTIME_ID", &resolved.runtime_id)
        .env("ROCM_CLI_PYTHON", &resolved.python_executable);
    apply_app_path_env(&mut command, &paths);

    let prompt = format!("(rocm:{engine}) ");
    let mut prompt_marked = false;
    if !rocm_core::runtime_is_windows() {
        // Kept for prompt frameworks (starship, powerlevel10k, oh-my-posh) that
        // read this directly -- that is why the missing marker went unnoticed by
        // anyone using one. Plain bash/zsh need the shim below.
        command.env("VIRTUAL_ENV_PROMPT", &prompt);

        let shim_dir = paths.engine_state_dir(engine).join("shell");
        // `engine` is constrained by clap to the supported-engine list, so the
        // prompt cannot carry shell metacharacters into the generated files.
        if let Some(shim) = engine_shell_prompt_shim(
            &shell_program,
            &prompt,
            &shim_dir,
            std::env::var("ZDOTDIR").ok().as_deref(),
        ) {
            // A shim that cannot be written is not worth failing the command over
            // -- the shell still works, it just looks unmarked, and the banner
            // below adapts to say so.
            match write_engine_shell_shim(&shim_dir, &shim) {
                Ok(()) => {
                    command.args(&shim.args);
                    for (key, value) in &shim.envs {
                        command.env(key, value);
                    }
                    prompt_marked = true;
                }
                Err(error) => {
                    eprintln!("warning: could not prepare the engine shell prompt: {error}");
                }
            }
        }

        if !prompt_marked {
            // Shells we have no shim for (sh, dash) do honour an inherited PS1, so
            // this is still worth setting -- but as a self-contained value. The
            // previous `{prompt}${PS1:-}` referred to the variable being assigned,
            // which dash expanded into itself and rendered as
            // `(rocm:vllm) (rocm:vllm) ${PS1:-}`. Shells that ignore PS1 entirely
            // (fish) are unaffected either way.
            command.env("PS1", format!("{prompt}$ "));
        }
    }

    // The block above describes the environment; this is the handover. Without
    // it, a shell we could not mark is indistinguishable from the parent and
    // reads as "the command only printed information" -- which is how this was
    // reported.
    println!();
    if prompt_marked {
        println!("Entering the {engine} engine shell — your prompt is now prefixed {prompt}");
    } else {
        println!(
            "Entering the {engine} engine shell — your prompt may look unchanged; \
             run `echo $ROCM_CLI_ENGINE` to confirm you are inside it."
        );
    }
    println!("Run `exit` (or Ctrl-D) to return to your previous shell.");

    let status = command
        .status()
        .with_context(|| format!("failed to launch shell `{shell_program}`"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("managed engine shell exited with status {status}");
    }
}

fn resolve_engine_env(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<ResolvedEngineEnv> {
    let selection = validate_engine_selection_runtime(
        paths,
        resolve_engine_selection(config, engine, runtime_id, env_id),
    )?;
    if let Some(env_id) = selection.env_id.as_deref() {
        let manifest = load_engine_env_manifest(paths, engine, env_id)?;
        return Ok(ResolvedEngineEnv {
            env_id: manifest.env_id,
            runtime_id: manifest.runtime_id,
            python_executable: manifest.python_executable,
            env_path: manifest.env_path,
            source: selection
                .source
                .unwrap_or_else(|| "manifest_env_id".to_owned()),
        });
    }

    let runtime_id = selection.runtime_id.with_context(|| {
        "no active ROCm runtime is configured; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id"
    })?;
    let env_root = env_root_for_engine_install(paths, config, engine, &runtime_id)?;
    let response = engine_request_with_env_root::<_, InstallResponse>(
        Some(paths),
        engine,
        EngineMethod::Install,
        &InstallRequest {
            runtime_id: runtime_id.clone(),
            python_version: None,
            reinstall: false,
            env_root: env_root.clone(),
        },
        env_root.as_deref(),
    )?;
    settle_engine_install(paths, engine, &runtime_id, &response)?;
    Ok(ResolvedEngineEnv {
        env_id: response.env_id,
        runtime_id,
        python_executable: response.python_executable,
        env_path: PathBuf::from(response.env_path),
        source: selection
            .source
            .unwrap_or_else(|| "auto_install".to_owned()),
    })
}

fn load_engine_env_manifest(
    paths: &AppPaths,
    engine: &str,
    env_id: &str,
) -> Result<ManagedEngineEnvManifest> {
    let path = paths
        .engine_manifests_dir(engine)
        .join(format!("{env_id}.json"));
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_paths, test_runtime_manifest_for_update, write_test_pip_runtime};

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
    fn engine_install_runtime_selection_requires_configured_runtime() -> Result<()> {
        let (root, paths) = test_paths("engine-install-runtime-selection");
        let error =
            resolve_engine_install_runtime_id(&paths, &RocmCliConfig::default(), "vllm", None)
                .unwrap_err()
                .to_string();
        assert!(error.contains("no active ROCm runtime is configured"));
        assert_eq!(
            resolve_engine_install_runtime_id(&paths, &RocmCliConfig::default(), "lemonade", None)?,
            format!("lemonade-embeddable-{}", rocm_deps::LEMONADE_VERSION),
        );
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;

        let config = RocmCliConfig {
            active_runtime_key: Some("release-pip-gfx120x-all".to_owned()),
            ..RocmCliConfig::default()
        };
        assert_eq!(
            resolve_engine_install_runtime_id(&paths, &config, "vllm", None)?,
            "release-pip-gfx120x-all"
        );
        assert_eq!(
            resolve_engine_install_runtime_id(
                &paths,
                &config,
                "vllm",
                Some("therock-release:gfx120X-all".to_owned())
            )?,
            "release-pip-gfx120x-all"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
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

    #[test]
    fn engine_runtime_selection_rejects_ambiguous_default_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("engine-runtime-ambiguous-default");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        write_test_pip_runtime(
            &paths,
            "vllm-source-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            2,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let error = resolve_engine_install_runtime_id(&paths, &config, "vllm", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple installed runtimes"));
        assert!(error.contains("rocm runtimes activate <runtime_key>"));

        let selection = resolve_engine_selection(&config, "vllm", None, None);
        let error = validate_engine_selection_runtime(&paths, selection)
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple installed runtimes"));

        let selection =
            resolve_engine_selection(&config, "vllm", Some("release-pip-gfx120x-all"), None);
        let selection = validate_engine_selection_runtime(&paths, selection)?;
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all")
        );
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

    #[test]
    fn bash_shim_sources_the_user_rc_and_prefixes_the_prompt() {
        let dir = PathBuf::from("/tmp/shim");
        let shim = engine_shell_prompt_shim("/bin/bash", "(rocm:vllm) ", &dir, None)
            .expect("bash must be shimmable");

        // `--rcfile` is what makes bash run our file at all; `-i` keeps it
        // interactive even if stdin is not a terminal in some caller.
        assert_eq!(
            shim.args,
            vec![
                "--rcfile".to_owned(),
                dir.join("engine-shell.bash").display().to_string(),
                "-i".to_owned(),
            ]
        );
        assert!(shim.envs.is_empty(), "bash needs no extra env");

        let (path, contents) = shim.files.first().expect("one rc file");
        assert_eq!(path, &dir.join("engine-shell.bash"));
        assert!(
            contents.contains("$HOME/.bashrc"),
            "must restore the user's own rc:\n{contents}"
        );
        // bash sources the system file itself even with --rcfile, so sourcing it
        // here too would apply it twice.
        assert!(
            !contents.contains("/etc/bash.bashrc"),
            "must not re-source the system rc:\n{contents}"
        );
        assert!(
            contents.contains("PS1='(rocm:vllm) '\"$PS1\""),
            "must prefix rather than replace the prompt:\n{contents}"
        );
    }

    #[test]
    fn zsh_shim_restores_both_startup_files() {
        let dir = PathBuf::from("/tmp/shim");
        let shim = engine_shell_prompt_shim("/usr/bin/zsh", "(rocm:vllm) ", &dir, None)
            .expect("zsh must be shimmable");

        assert!(shim.args.is_empty(), "zsh is redirected via env, not argv");
        assert!(
            shim.envs
                .contains(&("ZDOTDIR".to_owned(), dir.display().to_string())),
            "zsh needs ZDOTDIR pointed at the shim dir: {:?}",
            shim.envs
        );

        let names: Vec<_> = shim
            .files
            .iter()
            .map(|(path, _)| path.file_name().unwrap().to_str().unwrap())
            .collect();
        // Redirecting ZDOTDIR hides BOTH of the user's files. Missing `.zshenv`
        // would strip their exports — worse than the unmarked prompt this fixes.
        assert!(
            names.contains(&".zshenv") && names.contains(&".zshrc"),
            "both startup files must be restored, got {names:?}"
        );
        for (path, contents) in &shim.files {
            assert!(
                contents.contains("ROCM_CLI_ORIG_ZDOTDIR:-$HOME"),
                "{} must fall back to $HOME:\n{contents}",
                path.display()
            );
        }
        let zshrc = shim
            .files
            .iter()
            .find(|(path, _)| path.ends_with(".zshrc"))
            .map(|(_, contents)| contents)
            .expect(".zshrc present");
        assert!(
            zshrc.contains("PROMPT='(rocm:vllm) '$PROMPT"),
            "must prefix rather than replace the prompt:\n{zshrc}"
        );
    }

    #[test]
    fn zsh_shim_passes_through_an_existing_zdotdir() {
        let dir = PathBuf::from("/tmp/shim");
        let shim = engine_shell_prompt_shim("zsh", "(rocm:vllm) ", &dir, Some("/home/u/.zsh"))
            .expect("zsh must be shimmable");
        assert!(
            shim.envs.contains(&(
                "ROCM_CLI_ORIG_ZDOTDIR".to_owned(),
                "/home/u/.zsh".to_owned()
            )),
            "a caller's ZDOTDIR must survive so the shim can find their files: {:?}",
            shim.envs
        );

        // A blank value is not a location; the shim's $HOME fallback must win.
        let blank = engine_shell_prompt_shim("zsh", "(rocm:vllm) ", &dir, Some("   "))
            .expect("zsh must be shimmable");
        assert!(
            !blank
                .envs
                .iter()
                .any(|(key, _)| key == "ROCM_CLI_ORIG_ZDOTDIR"),
            "a blank ZDOTDIR must not be passed through: {:?}",
            blank.envs
        );
    }

    #[test]
    fn shells_without_a_safe_shim_are_left_alone() {
        // Guessing at an unknown shell's startup is worse than the banner: these
        // must opt out rather than have a marker forced on them.
        let dir = PathBuf::from("/tmp/shim");
        for shell in [
            "/bin/sh",
            "/bin/dash",
            "/usr/bin/fish",
            "cmd",
            "powershell",
            "pwsh",
            "",
        ] {
            assert!(
                engine_shell_prompt_shim(shell, "(rocm:vllm) ", &dir, None).is_none(),
                "{shell} should not be shimmed"
            );
        }
    }

    #[test]
    fn shim_matches_on_the_shell_name_not_the_full_path() {
        let dir = PathBuf::from("/tmp/shim");
        for shell in ["bash", "/bin/bash", "/usr/local/bin/bash"] {
            assert!(
                engine_shell_prompt_shim(shell, "(rocm:x) ", &dir, None).is_some(),
                "{shell} should resolve to bash"
            );
        }
    }
}
