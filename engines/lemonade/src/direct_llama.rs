// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Result, anyhow, bail};
use rocm_core::AppPaths;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::backend_alignment::{LemonadeRuntime, ROCM_LLAMACPP_BACKEND_DIRS};
use crate::install::{LemonadeInstallManifest, platform_binary_name};
use crate::process::{
    LemonadeProcessEnvironment, run_lemonade_model_load, run_lemonade_pull,
    serve_direct_llama_server, spawn_lemond, wait_for_lemonade_cli_status,
};
use crate::state::{free_local_port, resolve_lemonade_model_ref, terminate_pid};
use crate::{
    DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_MODEL_GGUF, DEFAULT_MODEL_REPO_DIR, ROCM_BACKEND_NAME,
    ServeHttpRequest,
};

/// Serve a canonical Hugging Face checkpoint under its exact name. Requires an explicit
/// `:variant`; downloads the GGUF, then runs whichever packaged GPU llama-server backend
/// is installed (e.g. `vulkan` on WSL2, `rocm-stable` on native ROCm) directly on the
/// file with `--alias`. This bypasses Lemonade's model registry, whose naming rules
/// cannot preserve an `owner/repo:variant` name.
pub(crate) fn serve_hf_checkpoint(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
    checkpoint: &HfCheckpoint,
) -> Result<()> {
    // A variant is required: a bare `owner/repo` would need Lemonade's interactive
    // variant menu (which cannot be answered from this non-interactive path) and, if a
    // GGUF happened to be cached, risks silently serving the wrong quantization.
    if checkpoint.variant.is_none() {
        bail!(
            "`{model}` needs an explicit quantization variant to serve; use `owner/repo:variant`, e.g. `{model}:Q4_K_M` (see the repo's GGUF files on Hugging Face for the available variants)",
            model = request.model_ref
        );
    }
    let Some(server) = find_llama_server_binary(&runtime.manifest) else {
        bail!(
            "no GPU llama-server backend is installed under {}; run `rocm engines install lemonade`",
            runtime
                .manifest
                .runtime_dir
                .join("bin")
                .join("llamacpp")
                .display()
        );
    };
    ensure_hf_checkpoint_downloaded(request, runtime, process_env, log_path)?;
    // Never silently pick between quantizations: if the variant matches more than one
    // cached GGUF, ask the user to disambiguate instead of choosing one arbitrarily.
    let paths = AppPaths::discover()?;
    let matches = hf_checkpoint_gguf_matches(&paths, checkpoint);
    if matches.len() > 1 {
        let names = matches
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "variant `{}` matches multiple files ({names}); specify an exact quantization or filename, e.g. `{}:{}`",
            checkpoint.variant.as_deref().unwrap_or_default(),
            request.model_ref,
            matches
                .first()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or("<file>.gguf")
        );
    }
    serve_direct_llama_server(
        request,
        runtime,
        process_env,
        &server,
        log_path,
        &anyhow!(
            "serving canonical Hugging Face checkpoint `{}` directly",
            request.model_ref
        ),
    )
}

/// Ensure a canonical Hugging Face checkpoint's GGUF is in the HF hub cache, pulling
/// it through a short-lived `lemond` if needed. The bare `pull owner/repo:variant`
/// only downloads (and registers a derived name we ignore); the file is what we serve.
fn ensure_hf_checkpoint_downloaded(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    if direct_llama_model_path(&paths, &request.model_ref).is_some() {
        return Ok(());
    }
    let download_port = free_local_port()?;
    let mut child = spawn_lemond(
        &runtime.manifest,
        DEFAULT_HOST,
        download_port,
        log_path,
        process_env,
    )?;
    let result = (|| -> Result<()> {
        wait_for_lemonade_cli_status(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            Duration::from_secs(30),
            log_path,
            process_env,
        )?;
        run_lemonade_pull(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            &request.model_ref,
            log_path,
            process_env,
        )?;
        if direct_llama_model_path(&paths, &request.model_ref).is_some() {
            Ok(())
        } else {
            bail!("Lemonade did not download `{}`", request.model_ref)
        }
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    result
}

pub(crate) fn ensure_direct_llama_model_available(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    if direct_llama_model_path(&paths, &request.model_ref).is_some() {
        return Ok(());
    }

    let download_port = free_local_port()?;
    let mut child = spawn_lemond(
        &runtime.manifest,
        DEFAULT_HOST,
        download_port,
        log_path,
        process_env,
    )?;
    let result = (|| -> Result<()> {
        wait_for_lemonade_cli_status(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            Duration::from_secs(30),
            log_path,
            process_env,
        )?;
        let _ = run_lemonade_model_load(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            &request.model_ref,
            ROCM_BACKEND_NAME,
            None,
            log_path,
            process_env,
        );
        if direct_llama_model_path(&paths, &request.model_ref).is_some() {
            Ok(())
        } else {
            bail!(
                "Lemonade did not download `{}` for direct ROCm serving",
                request.model_ref
            )
        }
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    result
}

/// A canonical Hugging Face checkpoint reference: `owner/repo` with an optional
/// `:variant` suffix (a quantization label such as `BF16`/`Q4_0`, or an exact
/// `file.gguf`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HfCheckpoint {
    owner: String,
    repo: String,
    variant: Option<String>,
}

/// Parse a model reference as a canonical Hugging Face checkpoint. Returns `None` for
/// anything not of the form `owner/repo[:variant]` — built-in Lemonade aliases and
/// bare registered names all fall through unchanged.
///
/// Every component is validated against the Hugging Face id / quantization charset
/// (`[A-Za-z0-9._-]`, and never `.`/`..`). This is a security boundary as well as a
/// correctness one: the components are used to build a filesystem path into the model
/// cache, so rejecting path separators and traversal tokens prevents a crafted model
/// reference from escaping the cache directory.
pub(crate) fn parse_hf_checkpoint(model_ref: &str) -> Option<HfCheckpoint> {
    let trimmed = model_ref.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Split off an optional `:variant` suffix on the first colon; the owner/repo pair
    // never contains a colon, so any remainder is the variant.
    let (repo_path, variant) = match trimmed.split_once(':') {
        Some((path, variant)) => (path, Some(variant)),
        None => (trimmed, None),
    };
    let (owner, repo) = repo_path.split_once('/')?;
    if !is_safe_hf_component(owner) || !is_safe_hf_component(repo) {
        return None;
    }
    if variant.is_some_and(|variant| !is_safe_hf_component(variant)) {
        return None;
    }
    Some(HfCheckpoint {
        owner: owner.to_owned(),
        repo: repo.to_owned(),
        variant: variant.map(str::to_owned),
    })
}

/// A safe Hugging Face path component: a non-empty run of `[A-Za-z0-9._-]` that is not
/// a `.`/`..` traversal token. Matches the Hugging Face id and quantization-label
/// charset and, by excluding `/`, `\`, and traversal tokens, keeps a user-supplied
/// component from escaping the model cache directory when used to build a path.
fn is_safe_hf_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && component
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Packaged llama.cpp backends whose `llama-server` we can drive directly, best first.
/// GPU backends only — `cpu` is intentionally excluded so the direct-serve path never
/// silently serves on CPU under a GPU-required policy (AGENTS.md §6). An explicit
/// allowlist rather than a directory scan, so the path is built only from constants.
const DIRECT_LLAMA_SERVER_BACKENDS: [&str; 4] = ["rocm-stable", "rocm-nightly", "rocm", "vulkan"];

/// Find an installed Lemonade `llama-server` binary under `bin/llamacpp/<backend>/`,
/// checking known backends in priority order. This lets direct serving work wherever
/// Lemonade installed a backend (e.g. `vulkan` on WSL2), not just the ROCm build.
///
/// Lemonade's backend installer extracts some llama.cpp releases straight into
/// `<backend>/`, but others land one level deeper under a build-numbered directory
/// (e.g. `rocm-stable/llama-b9752/llama-server`), so both layouts are checked.
pub(crate) fn find_llama_server_binary(manifest: &LemonadeInstallManifest) -> Option<PathBuf> {
    let llamacpp_dir = manifest.runtime_dir.join("bin").join("llamacpp");
    let binary = platform_binary_name("llama-server");
    DIRECT_LLAMA_SERVER_BACKENDS
        .into_iter()
        .find_map(|backend| find_binary_in(&llamacpp_dir.join(backend), &binary))
}

/// Like [`find_llama_server_binary`], but scoped to the single backend Lemonade
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

/// Look for `binary` directly in `dir`, then one level down in each subdirectory.
fn find_binary_in(dir: &Path, binary: &str) -> Option<PathBuf> {
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

/// The backend label for a packaged llama-server, taken from the nearest ancestor
/// directory matching a known backend name (e.g. `vulkan`, `rocm-stable`) so it still
/// resolves correctly when Lemonade nests the binary under a build-numbered
/// subdirectory; falls back to the ROCm backend name.
pub(crate) fn llama_server_backend_label(server: &Path) -> String {
    server
        .ancestors()
        .filter_map(|dir| dir.file_name())
        .filter_map(|name| name.to_str())
        .find(|name| DIRECT_LLAMA_SERVER_BACKENDS.contains(name))
        .unwrap_or(ROCM_BACKEND_NAME)
        .to_owned()
}

pub(crate) fn direct_llama_model_path(paths: &AppPaths, model_ref: &str) -> Option<PathBuf> {
    let as_path = PathBuf::from(model_ref);
    if as_path.is_file() {
        return Some(as_path);
    }
    if let Some(checkpoint) = parse_hf_checkpoint(model_ref) {
        return hf_cache_roots(paths)
            .into_iter()
            .find_map(|root| find_hf_checkpoint_gguf(&root, &checkpoint));
    }
    if resolve_lemonade_model_ref(model_ref) != DEFAULT_MODEL {
        return None;
    }
    hf_cache_roots(paths)
        .into_iter()
        .find_map(find_default_qwen_gguf)
}

fn hf_cache_roots(paths: &AppPaths) -> Vec<PathBuf> {
    hf_cache_roots_from(paths, |name| std::env::var_os(name).map(PathBuf::from))
}

fn hf_cache_roots_from<F>(paths: &AppPaths, mut env_path: F) -> Vec<PathBuf>
where
    F: FnMut(&str) -> Option<PathBuf>,
{
    let mut roots = Vec::new();
    if let Some(hub_cache) = env_path("HUGGINGFACE_HUB_CACHE") {
        push_hf_cache_root(&mut roots, hub_cache);
    }
    if let Some(hf_home) = env_path("HF_HOME") {
        push_hf_cache_root(&mut roots, hf_home.join("hub"));
    }
    push_hf_cache_root(&mut roots, paths.cache_dir.join("huggingface").join("hub"));
    if let Some(home) = env_path("HOME") {
        push_hf_cache_root(
            &mut roots,
            home.join(".cache").join("huggingface").join("hub"),
        );
    }
    roots
}

fn push_hf_cache_root(roots: &mut Vec<PathBuf>, path: PathBuf) {
    if path.as_os_str().is_empty() || roots.iter().any(|existing| existing == &path) {
        return;
    }
    roots.push(path);
}

fn find_default_qwen_gguf(cache_root: PathBuf) -> Option<PathBuf> {
    let snapshots = cache_root.join(DEFAULT_MODEL_REPO_DIR).join("snapshots");
    let entries = fs::read_dir(snapshots).ok()?;
    entries
        .flatten()
        .map(|entry| entry.path().join(DEFAULT_MODEL_GGUF))
        .find(|path| path.is_file())
}

/// Locate the downloaded GGUF for a Hugging Face checkpoint inside one hub cache root,
/// selecting the file that matches the requested `:variant` (if any).
fn find_hf_checkpoint_gguf(cache_root: &Path, checkpoint: &HfCheckpoint) -> Option<PathBuf> {
    select_gguf_for_variant(
        collect_hf_checkpoint_ggufs(cache_root, checkpoint),
        checkpoint.variant.as_deref(),
    )
}

/// All cached GGUF files for a checkpoint's repo under one hub cache root (unfiltered
/// by variant).
fn collect_hf_checkpoint_ggufs(cache_root: &Path, checkpoint: &HfCheckpoint) -> Vec<PathBuf> {
    let repo_dir = format!("models--{}--{}", checkpoint.owner, checkpoint.repo);
    let snapshots = cache_root.join(repo_dir).join("snapshots");
    let mut ggufs = Vec::new();
    if let Ok(entries) = fs::read_dir(snapshots) {
        for entry in entries.flatten() {
            collect_gguf_files(&entry.path(), &mut ggufs, 0);
        }
    }
    ggufs
}

/// Every cached GGUF matching a checkpoint's variant, across all hub cache roots,
/// deduplicated by file name. Used to detect an ambiguous variant (more than one match)
/// so the caller can refuse to pick arbitrarily.
fn hf_checkpoint_gguf_matches(paths: &AppPaths, checkpoint: &HfCheckpoint) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
    for root in hf_cache_roots(paths) {
        let ggufs = collect_hf_checkpoint_ggufs(&root, checkpoint);
        for gguf in filter_ggufs_by_variant(ggufs, checkpoint.variant.as_deref()) {
            if let Some(name) = gguf.file_name().and_then(|name| name.to_str())
                && seen_names.insert(name.to_owned())
            {
                matches.push(gguf);
            }
        }
    }
    matches
}

/// Collect `*.gguf` files under a snapshot revision directory. GGUFs usually sit at the
/// snapshot root, but sharded variants live one folder down, so recurse a few levels.
fn collect_gguf_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, out, depth + 1);
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

/// Pick a single GGUF for a requested variant (the first, in sorted order, of the
/// matches). Sorting is by path — deterministic but not by mtime — which is fine
/// because the caller rejects an ambiguous variant before relying on this pick.
fn select_gguf_for_variant(ggufs: Vec<PathBuf>, variant: Option<&str>) -> Option<PathBuf> {
    filter_ggufs_by_variant(ggufs, variant).into_iter().next()
}

/// All GGUFs matching a requested variant, sorted by path. With no variant, every file
/// matches. With a variant, a case-insensitive exact filename match (`:model.gguf`)
/// wins outright; otherwise every file whose name contains the quantization label
/// (`:BF16`, `:Q4_0`) matches — more than one indicates an ambiguous variant.
fn filter_ggufs_by_variant(mut ggufs: Vec<PathBuf>, variant: Option<&str>) -> Vec<PathBuf> {
    ggufs.sort();
    ggufs.dedup();
    let Some(variant) = variant else {
        return ggufs;
    };
    let file_name = |path: &Path| -> Option<String> {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
    };
    if let Some(exact) = ggufs
        .iter()
        .find(|path| file_name(path).is_some_and(|name| name.eq_ignore_ascii_case(variant)))
    {
        return vec![exact.clone()];
    }
    let variant_lower = variant.to_ascii_lowercase();
    ggufs
        .into_iter()
        .filter(|path| {
            file_name(path).is_some_and(|name| name.to_ascii_lowercase().contains(&variant_lower))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LLAMACPP_RECIPE;
    use crate::test_support::scratch_dir;
    use std::fs;

    #[test]
    fn parses_canonical_hugging_face_checkpoints() {
        assert_eq!(
            parse_hf_checkpoint("LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
            Some(HfCheckpoint {
                owner: "LiquidAI".to_owned(),
                repo: "LFM2.5-230M-GGUF".to_owned(),
                variant: Some("Q4_0".to_owned()),
            })
        );
        assert_eq!(
            parse_hf_checkpoint("unsloth/Qwen3-0.6B-GGUF"),
            Some(HfCheckpoint {
                owner: "unsloth".to_owned(),
                repo: "Qwen3-0.6B-GGUF".to_owned(),
                variant: None,
            })
        );
    }

    #[test]
    fn rejects_non_checkpoint_model_refs() {
        // Built-in aliases and bare registered names have no owner/repo pair.
        assert!(parse_hf_checkpoint("qwen").is_none());
        assert!(parse_hf_checkpoint(DEFAULT_MODEL).is_none());
        assert!(parse_hf_checkpoint("").is_none());
        assert!(parse_hf_checkpoint("/model.gguf").is_none());
        assert!(parse_hf_checkpoint("owner/").is_none());
        assert!(parse_hf_checkpoint("owner/repo/extra").is_none());
        assert!(parse_hf_checkpoint("owner/repo:").is_none());
    }

    #[test]
    fn rejects_path_traversal_in_checkpoint_components() {
        // A crafted reference must not be able to escape the model cache directory.
        assert!(parse_hf_checkpoint("../../etc/passwd").is_none());
        assert!(parse_hf_checkpoint("..:Q4_0").is_none());
        assert!(parse_hf_checkpoint("owner/..").is_none());
        assert!(parse_hf_checkpoint("owner/repo:../secret").is_none());
        assert!(parse_hf_checkpoint(r"owner\..\..--x/repo").is_none());
        assert!(parse_hf_checkpoint("owner/repo:a/b.gguf").is_none());
    }

    #[test]
    fn selects_gguf_by_variant() {
        // Real filenames use the quant label without the `-GGUF` repo suffix, so a
        // substring match on the `:variant` selects the right file.
        let ggufs = vec![
            PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"),
            PathBuf::from("/cache/LFM2.5-230M-Q8_0.gguf"),
        ];
        assert_eq!(
            select_gguf_for_variant(ggufs.clone(), Some("q4_0")),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"))
        );
        assert_eq!(
            select_gguf_for_variant(ggufs.clone(), Some("LFM2.5-230M-Q8_0.gguf")),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q8_0.gguf"))
        );
        assert_eq!(
            select_gguf_for_variant(ggufs.clone(), None),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"))
        );
        assert_eq!(select_gguf_for_variant(ggufs, Some("Q2_K")), None);
    }

    #[test]
    fn ambiguous_variant_matches_multiple_ggufs() {
        // A partial label like `Q4` matches several quants; the caller must refuse to
        // pick. A full label or exact filename resolves to exactly one.
        let ggufs = vec![
            PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"),
            PathBuf::from("/cache/LFM2.5-230M-Q4_K_M.gguf"),
        ];
        assert_eq!(filter_ggufs_by_variant(ggufs.clone(), Some("Q4")).len(), 2);
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), Some("Q4_0")).len(),
            1
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs, Some("LFM2.5-230M-Q4_K_M.gguf")).len(),
            1
        );
    }

    #[test]
    fn find_llama_server_binary_prefers_gpu_and_skips_cpu() {
        let dir = scratch_dir("find-server");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        for backend in ["cpu", "vulkan"] {
            let backend_dir = llamacpp.join(backend);
            fs::create_dir_all(&backend_dir).unwrap();
            fs::write(backend_dir.join(platform_binary_name("llama-server")), b"x").unwrap();
        }
        let manifest = test_manifest(runtime_dir);
        // Both cpu and vulkan exist; the GPU backend is chosen and cpu is never selected.
        assert_eq!(
            find_llama_server_binary(&manifest),
            Some(
                llamacpp
                    .join("vulkan")
                    .join(platform_binary_name("llama-server"))
            )
        );
        // With only cpu present, nothing is returned (no silent CPU fallback).
        fs::remove_dir_all(llamacpp.join("vulkan")).unwrap();
        assert_eq!(find_llama_server_binary(&manifest), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_llama_server_binary_finds_build_numbered_nesting() {
        // Some Lemonade llama.cpp releases extract one level deeper than
        // `<backend>/llama-server`, e.g. `rocm-stable/llama-b9752/llama-server`.
        let dir = scratch_dir("find-server-nested");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let nested = llamacpp.join("rocm-stable").join("llama-b9752");
        fs::create_dir_all(&nested).unwrap();
        let server = nested.join(platform_binary_name("llama-server"));
        fs::write(&server, b"x").unwrap();
        let manifest = test_manifest(runtime_dir);
        assert_eq!(find_llama_server_binary(&manifest), Some(server.clone()));
        assert_eq!(llama_server_backend_label(&server), "rocm-stable");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_llama_server_binary_follows_backend_priority() {
        let dir = scratch_dir("find-server-priority");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        let install = |backend: &str| {
            let backend_dir = llamacpp.join(backend);
            fs::create_dir_all(&backend_dir).unwrap();
            fs::write(backend_dir.join(&server), b"x").unwrap();
        };
        let manifest = test_manifest(runtime_dir);

        // Nightly-only host (the resilient case that replaced the hardcoded
        // rocm-stable direct-serve path): the nightly backend is selected over vulkan.
        install("rocm-nightly");
        install("vulkan");
        assert_eq!(
            find_llama_server_binary(&manifest),
            Some(llamacpp.join("rocm-nightly").join(&server))
        );

        // With rocm-stable also present, it wins (highest priority).
        install("rocm-stable");
        assert_eq!(
            find_llama_server_binary(&manifest),
            Some(llamacpp.join("rocm-stable").join(&server))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collect_gguf_files_recurses_and_filters() {
        let dir = scratch_dir("collect-gguf");
        fs::create_dir_all(dir.join("shard")).unwrap();
        fs::write(dir.join("model-Q4_0.gguf"), b"x").unwrap();
        fs::write(dir.join("notes.txt"), b"x").unwrap();
        fs::write(dir.join("shard").join("model-Q8_0.gguf"), b"x").unwrap();
        let mut found = Vec::new();
        collect_gguf_files(&dir, &mut found, 0);
        let mut names = found
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, vec!["model-Q4_0.gguf", "model-Q8_0.gguf"]);
        fs::remove_dir_all(&dir).ok();
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

    #[test]
    fn direct_qwen_lookup_checks_huggingface_cache_env() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(&paths, |name| match name {
            "HUGGINGFACE_HUB_CACHE" => Some(PathBuf::from("hf-hub")),
            "HF_HOME" => Some(PathBuf::from("hf-home")),
            "HOME" => Some(PathBuf::from("home")),
            _ => None,
        });

        assert_eq!(
            roots,
            vec![
                PathBuf::from("hf-hub"),
                PathBuf::from("hf-home").join("hub"),
                PathBuf::from("rocm-cache").join("huggingface").join("hub"),
                PathBuf::from("home")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ]
        );
    }
}
