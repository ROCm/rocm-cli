// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, runtime_home_dir};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::backend_alignment::{LemonadeRuntime, ROCM_LLAMACPP_BACKEND_DIRS};
use crate::install::{LemonadeInstallManifest, platform_binary_name};
use crate::process::{
    LemonadeProcessEnvironment, run_lemonade_pull, spawn_lemond, wait_for_lemonade_cli_status,
};
use crate::state::{free_local_port, resolve_lemonade_model_ref, terminate_pid};
use crate::{DEFAULT_HOST, ROCM_BACKEND_NAME, ServeHttpRequest};

/// Direct serving is only available for a local GGUF or an exact checkpoint.
/// Unsupported registered recipes stay on the Lemonade router path.
#[derive(Debug)]
pub(crate) enum DirectLlamaModel {
    Local(PathBuf),
    Checkpoint(HfCheckpoint),
}

pub(crate) fn resolve_direct_llama_model(
    manifest: &LemonadeInstallManifest,
    model_ref: &str,
) -> Result<Option<DirectLlamaModel>> {
    let path = PathBuf::from(model_ref);
    if path.is_file()
        && path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return Ok(Some(DirectLlamaModel::Local(path)));
    }
    Ok(resolve_direct_llama_checkpoint(manifest, model_ref)?.map(DirectLlamaModel::Checkpoint))
}

/// Resolve the cached file without choosing an arbitrary quantization. The same
/// resolution is used before downloading and immediately before server launch.
pub(crate) fn direct_llama_model_path(
    paths: &AppPaths,
    model: &DirectLlamaModel,
) -> Result<Option<PathBuf>> {
    match model {
        DirectLlamaModel::Local(path) => Ok(path.is_file().then(|| path.clone())),
        DirectLlamaModel::Checkpoint(checkpoint) => {
            let matches = hf_checkpoint_gguf_matches(paths, checkpoint);
            if matches.len() > 1 {
                let names = matches
                    .iter()
                    .filter_map(|path| path.file_name())
                    .map(|name| name.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "variant `{}` matches multiple files ({names}); specify an exact quantization or filename",
                    checkpoint.variant.as_deref().unwrap_or_default()
                );
            }
            Ok(matches.into_iter().next())
        }
    }
}

pub(crate) fn ensure_direct_llama_model_available(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
    model: &DirectLlamaModel,
) -> Result<PathBuf> {
    let paths = AppPaths::discover()?;
    if let Some(path) = direct_llama_model_path(&paths, model)? {
        return Ok(path);
    }
    let DirectLlamaModel::Checkpoint(checkpoint) = model else {
        bail!("local GGUF `{}` is missing", request.model_ref);
    };
    let download_port = free_local_port()?;
    let mut child = spawn_lemond(
        &runtime.manifest,
        DEFAULT_HOST,
        download_port,
        log_path,
        process_env,
    )?;
    let result = (|| -> Result<PathBuf> {
        wait_for_lemonade_cli_status(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            Duration::from_secs(30),
            log_path,
            process_env,
        )?;
        // Pull the validated checkpoint, not the alias. Pull downloads without a
        // model load, so this path neither guesses a backend nor masks load errors.
        run_lemonade_pull(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            &checkpoint.model_ref(),
            log_path,
            process_env,
        )?;
        direct_llama_model_path(&paths, model)?.with_context(|| {
            format!(
                "Lemonade did not download `{}` for direct serving (searched {})",
                request.model_ref,
                describe_hf_cache_roots(&paths)
            )
        })
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

impl HfCheckpoint {
    fn model_ref(&self) -> String {
        format!(
            "{}/{}:{}",
            self.owner,
            self.repo,
            self.variant.as_deref().unwrap_or_default()
        )
    }
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

/// Find a GPU llama-server scoped to the selected backend. Generic `rocm`
/// accepts the ROCm family; Vulkan and specific ROCm channels never cross over
/// to a stale install from another backend. CPU and unknown names fail closed.
/// Both flat and one-level build-numbered extraction layouts are supported.
pub(crate) fn find_llama_server_binary_for_backend(
    manifest: &LemonadeInstallManifest,
    backend_name: &str,
) -> Option<PathBuf> {
    let llamacpp_dir = manifest.runtime_dir.join("bin").join("llamacpp");
    let binary = platform_binary_name("llama-server");
    let family: &[&str] = match backend_name {
        ROCM_BACKEND_NAME => &ROCM_LLAMACPP_BACKEND_DIRS,
        "rocm-stable" => &["rocm-stable"],
        "rocm-nightly" => &["rocm-nightly"],
        "vulkan" => &["vulkan"],
        _ => return None,
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

/// Registered model names are exact keys after known shorthand normalization.
/// Metadata is untrusted: it must name a safe HF checkpoint with an explicit
/// variant, never a local path or a model guessed from the registered name.
pub(crate) fn resolve_direct_llama_checkpoint(
    manifest: &LemonadeInstallManifest,
    model_ref: &str,
) -> Result<Option<HfCheckpoint>> {
    if let Some(checkpoint) = parse_hf_checkpoint(model_ref) {
        return require_checkpoint_variant(checkpoint, model_ref).map(Some);
    }
    let registry_path = manifest.runtime_dir.join("resources/server_models.json");
    let bytes = match fs::read(&registry_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read {}", registry_path.display()));
        }
    };
    let registry: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid model registry {}", registry_path.display()))?;
    let entries = registry
        .as_object()
        .context("Lemonade model registry must be an object")?;
    let name = resolve_lemonade_model_ref(model_ref);
    let Some(metadata) = entries.get(&name) else {
        return Ok(None);
    };
    if metadata.get("recipe").and_then(serde_json::Value::as_str) != Some(crate::LLAMACPP_RECIPE) {
        return Ok(None);
    }
    let checkpoint_ref = metadata
        .get("checkpoint")
        .and_then(serde_json::Value::as_str)
        .with_context(|| format!("registered model `{name}` has no checkpoint"))?;
    let checkpoint = parse_hf_checkpoint(checkpoint_ref)
        .with_context(|| format!("registered model `{name}` has an invalid HF checkpoint"))?;
    require_checkpoint_variant(checkpoint, &name).map(Some)
}

fn require_checkpoint_variant(checkpoint: HfCheckpoint, model_ref: &str) -> Result<HfCheckpoint> {
    if checkpoint.variant.is_none() {
        bail!(
            "`{model_ref}` needs an explicit quantization variant to serve; use `owner/repo:variant`"
        );
    }
    Ok(checkpoint)
}

pub(crate) fn direct_serve_binary(manifest: &LemonadeInstallManifest) -> Option<PathBuf> {
    if manifest.backend_recipe != crate::LLAMACPP_RECIPE {
        return None;
    }
    find_llama_server_binary_for_backend(manifest, &manifest.backend_name)
}

fn hf_cache_roots(paths: &AppPaths) -> Vec<PathBuf> {
    hf_cache_roots_from(
        paths,
        |name| std::env::var_os(name).map(PathBuf::from),
        runtime_home_dir,
    )
}

/// `home_dir` is only consulted when `HOME` is unset, which is always true on Windows
/// (Lemonade still caches under `%USERPROFILE%\.cache\huggingface\hub`); it's injected
/// so the fallback can be exercised without depending on the host's actual platform.
fn hf_cache_roots_from<F, H>(paths: &AppPaths, mut env_path: F, home_dir: H) -> Vec<PathBuf>
where
    F: FnMut(&str) -> Option<PathBuf>,
    H: FnOnce() -> Option<PathBuf>,
{
    let mut roots = Vec::new();
    if let Some(hub_cache) = env_path("HUGGINGFACE_HUB_CACHE") {
        push_hf_cache_root(&mut roots, hub_cache);
    }
    if let Some(hf_home) = env_path("HF_HOME") {
        push_hf_cache_root(&mut roots, hf_home.join("hub"));
    }
    push_hf_cache_root(&mut roots, paths.cache_dir.join("huggingface").join("hub"));
    if let Some(home) = env_path("HOME").or_else(home_dir) {
        push_hf_cache_root(
            &mut roots,
            home.join(".cache").join("huggingface").join("hub"),
        );
    }
    roots
}

/// Render the cache directories `hf_cache_roots` searches, for use in an error message
/// when none of them held the expected file.
fn describe_hf_cache_roots(paths: &AppPaths) -> String {
    render_hf_cache_roots(&hf_cache_roots(paths))
}

/// The rendering half of [`describe_hf_cache_roots`], split out so the error text can be
/// asserted against the same root list `hf_cache_roots_from` produces, without depending
/// on the host's real environment.
fn render_hf_cache_roots(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        return "no Hugging Face cache directories could be determined".to_owned();
    }
    roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn push_hf_cache_root(roots: &mut Vec<PathBuf>, path: PathBuf) {
    if path.as_os_str().is_empty() || roots.iter().any(|existing| existing == &path) {
        return;
    }
    roots.push(path);
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
    hf_checkpoint_gguf_matches_in_roots(&hf_cache_roots(paths), checkpoint)
}

fn hf_checkpoint_gguf_matches_in_roots(
    roots: &[PathBuf],
    checkpoint: &HfCheckpoint,
) -> Vec<PathBuf> {
    let ggufs = roots
        .iter()
        .flat_map(|root| collect_hf_checkpoint_ggufs(root, checkpoint))
        .collect();
    let mut matches = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
    for gguf in filter_ggufs_by_variant(ggufs, checkpoint.variant.as_deref()) {
        if let Some(name) = gguf.file_name().and_then(|name| name.to_str())
            && seen_names.insert(name.to_owned())
        {
            matches.push(gguf);
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
        } else if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
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
    // An explicit filename is not a quantization substring. If absent, download
    // that exact file rather than serve another file with the same suffix.
    if Path::new(variant)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return Vec::new();
    }
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

    fn write_registry(manifest: &LemonadeInstallManifest, registry: serde_json::Value) {
        let resources = manifest.runtime_dir.join("resources");
        fs::create_dir_all(&resources).unwrap();
        fs::write(
            resources.join("server_models.json"),
            serde_json::to_vec(&registry).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn serving_regression_registered_alias_resolves_exact_checkpoint() {
        let dir = scratch_dir("registered-alias");
        let manifest = test_manifest(dir.join("runtime"));
        write_registry(
            &manifest,
            serde_json::json!({
                "Qwen3-0.6B-GGUF": {"recipe": "llamacpp", "checkpoint": "unsloth/Qwen3-0.6B-GGUF:Q4_0"},
                "Other-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/Other-GGUF:Q8_0"}
            }),
        );
        for alias in ["qwen-smoke", "tiny", " QWEN3-0.6B-GGUF "] {
            assert_eq!(
                resolve_direct_llama_checkpoint(&manifest, alias).unwrap(),
                parse_hf_checkpoint("unsloth/Qwen3-0.6B-GGUF:Q4_0")
            );
        }
        assert_eq!(
            resolve_direct_llama_checkpoint(&manifest, "Other-GGUF").unwrap(),
            parse_hf_checkpoint("owner/Other-GGUF:Q8_0")
        );
        assert_eq!(
            resolve_direct_llama_checkpoint(&manifest, "Other").unwrap(),
            None
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_rejects_unsafe_registered_metadata() {
        let dir = scratch_dir("registered-unsafe");
        let manifest = test_manifest(dir.join("runtime"));
        for checkpoint in [
            "../../etc/passwd:Q4_0",
            "owner/repo:../secret",
            "owner/repo",
            "/tmp/model.gguf",
        ] {
            write_registry(
                &manifest,
                serde_json::json!({
                    "Unsafe-GGUF": {"recipe": "llamacpp", "checkpoint": checkpoint}
                }),
            );
            assert!(
                resolve_direct_llama_checkpoint(&manifest, "Unsafe-GGUF").is_err(),
                "{checkpoint}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_selected_backend_ignores_stale_rocm() {
        let dir = scratch_dir("selected-backend");
        let mut manifest = test_manifest(dir.join("runtime"));
        manifest.backend_name = "vulkan".to_owned();
        let base = manifest.runtime_dir.join("bin/llamacpp");
        for backend in ["rocm-stable", "vulkan", "cpu"] {
            fs::create_dir_all(base.join(backend)).unwrap();
            fs::write(
                base.join(backend)
                    .join(platform_binary_name("llama-server")),
                b"x",
            )
            .unwrap();
        }
        assert_eq!(
            direct_serve_binary(&manifest),
            Some(
                base.join("vulkan")
                    .join(platform_binary_name("llama-server"))
            )
        );
        fs::remove_dir_all(base.join("vulkan")).unwrap();
        assert_eq!(direct_serve_binary(&manifest), None);
        manifest.backend_name = "cpu".to_owned();
        assert_eq!(direct_serve_binary(&manifest), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_registered_recipe_boundaries_and_missing_registry() {
        let dir = scratch_dir("registry-boundaries");
        let manifest = test_manifest(dir.join("runtime"));
        assert!(
            resolve_direct_llama_model(&manifest, "qwen-smoke")
                .unwrap()
                .is_none()
        );
        write_registry(
            &manifest,
            serde_json::json!({
                "RecipeModel": {"recipe": "oga", "checkpoint": "not/a/gguf"},
                "NoCheckpoint": {"recipe": "llamacpp"},
                "MalformedCheckpoint": {"recipe": "llamacpp", "checkpoint": 123},
                "Exact-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:Q8_0"}
            }),
        );
        for name in [
            "RecipeModel",
            "Unknown-GGUF",
            "exact-gguf",
            "Exact",
            "../../escape",
        ] {
            assert!(
                resolve_direct_llama_model(&manifest, name)
                    .unwrap()
                    .is_none(),
                "{name}"
            );
        }
        for name in ["NoCheckpoint", "MalformedCheckpoint", "owner/repo"] {
            assert!(
                resolve_direct_llama_model(&manifest, name).is_err(),
                "{name}"
            );
        }
        for json in [b"[]".as_slice(), b"invalid".as_slice()] {
            fs::write(
                manifest.runtime_dir.join("resources/server_models.json"),
                json,
            )
            .unwrap();
            assert!(resolve_direct_llama_model(&manifest, "Exact-GGUF").is_err());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_registered_checkpoint_cache_rejects_ambiguous_variants() {
        let dir = scratch_dir("registry-cache");
        let manifest = test_manifest(dir.join("runtime"));
        let roots = vec![dir.join("hub-one"), dir.join("hub-two")];
        for root in &roots {
            let snapshot = root.join("models--owner--repo/snapshots/revision");
            fs::create_dir_all(&snapshot).unwrap();
            for file in ["model-Q4_0.gguf", "model-Q4_K_M.gguf"] {
                fs::write(snapshot.join(file), b"x").unwrap();
            }
            // A GGUF suffix on a directory is not a downloaded model file.
            fs::create_dir_all(snapshot.join("directory-Q4.gguf")).unwrap();
        }
        write_registry(
            &manifest,
            serde_json::json!({
                "Exact-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:model-Q4_0.gguf"},
                "Ambiguous-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:Q4"}
            }),
        );
        let exact = resolve_direct_llama_checkpoint(&manifest, "Exact-GGUF")
            .unwrap()
            .unwrap();
        let matches = hf_checkpoint_gguf_matches_in_roots(&roots, &exact);
        assert_eq!(
            matches.len(),
            1,
            "same filename across cache roots is one quantization"
        );
        assert_eq!(matches[0].file_name().unwrap(), "model-Q4_0.gguf");
        let ambiguous = resolve_direct_llama_checkpoint(&manifest, "Ambiguous-GGUF")
            .unwrap()
            .unwrap();
        assert_eq!(
            hf_checkpoint_gguf_matches_in_roots(&roots, &ambiguous).len(),
            2
        );
        // Path resolution must reject ambiguous matching, not pick sorted first.
        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        let snapshot = paths
            .cache_dir
            .join("huggingface/hub/models--owner--repo/snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        for file in ["model-Q4_0.gguf", "model-Q4_K_M.gguf"] {
            fs::write(snapshot.join(file), b"x").unwrap();
        }
        assert!(direct_llama_model_path(&paths, &DirectLlamaModel::Checkpoint(ambiguous)).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_local_gguf_is_explicit_not_registry_path_metadata() {
        let dir = scratch_dir("local-gguf");
        let manifest = test_manifest(dir.join("runtime"));
        let model = dir.join("model.GGUF");
        fs::write(&model, b"x").unwrap();
        let local = resolve_direct_llama_model(&manifest, model.to_str().unwrap())
            .unwrap()
            .unwrap();
        assert!(matches!(local, DirectLlamaModel::Local(ref path) if path == &model));
        write_registry(
            &manifest,
            serde_json::json!({
                "Path-GGUF": {"recipe": "llamacpp", "checkpoint": model}
            }),
        );
        assert!(resolve_direct_llama_model(&manifest, "Path-GGUF").is_err());
        let non_gguf = dir.join("model.txt");
        fs::write(&non_gguf, b"x").unwrap();
        assert!(
            resolve_direct_llama_model(&manifest, non_gguf.to_str().unwrap())
                .unwrap()
                .is_none()
        );
        fs::remove_dir_all(dir).unwrap();
    }

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
    fn serving_regression_exact_filename_never_substitutes_suffix_match() {
        assert!(
            filter_ggufs_by_variant(
                vec![PathBuf::from("/cache/other-model.gguf")],
                Some("model.gguf")
            )
            .is_empty()
        );
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
            filter_ggufs_by_variant(ggufs.clone(), Some("q4_0"))
                .into_iter()
                .next(),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"))
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), Some("LFM2.5-230M-Q8_0.gguf"))
                .into_iter()
                .next(),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q8_0.gguf"))
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), None)
                .into_iter()
                .next(),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"))
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs, Some("Q2_K"))
                .into_iter()
                .next(),
            None
        );
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
        let mut manifest = test_manifest(runtime_dir);
        manifest.backend_name = "vulkan".to_owned();
        // Both cpu and vulkan exist; the GPU backend is chosen and cpu is never selected.
        assert_eq!(
            direct_serve_binary(&manifest),
            Some(
                llamacpp
                    .join("vulkan")
                    .join(platform_binary_name("llama-server"))
            )
        );
        // With only cpu present, nothing is returned (no silent CPU fallback).
        fs::remove_dir_all(llamacpp.join("vulkan")).unwrap();
        assert_eq!(direct_serve_binary(&manifest), None);
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
        assert_eq!(direct_serve_binary(&manifest), Some(server.clone()));
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
            direct_serve_binary(&manifest),
            Some(llamacpp.join("rocm-nightly").join(&server))
        );

        // With rocm-stable also present, it wins (highest priority).
        install("rocm-stable");
        assert_eq!(
            direct_serve_binary(&manifest),
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
        let roots = hf_cache_roots_from(
            &paths,
            |name| match name {
                "HUGGINGFACE_HUB_CACHE" => Some(PathBuf::from("hf-hub")),
                "HF_HOME" => Some(PathBuf::from("hf-home")),
                "HOME" => Some(PathBuf::from("home")),
                _ => None,
            },
            // `HOME` is set above, so the fallback must never be consulted.
            || panic!("home_dir fallback must not run when HOME is set"),
        );

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

    #[test]
    fn hf_cache_roots_falls_back_to_platform_home_when_home_unset() {
        // Windows never sets `HOME`, but Lemonade still caches under
        // `%USERPROFILE%\.cache\huggingface\hub`; the lookup must fall back to the
        // platform home directory (here standing in for `runtime_home_dir`, which
        // itself resolves `USERPROFILE` on Windows) rather than skip the hub cache.
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(
            &paths,
            |_name| None,
            || Some(PathBuf::from(r"C:\Users\dev")),
        );

        assert_eq!(
            roots,
            vec![
                PathBuf::from("rocm-cache").join("huggingface").join("hub"),
                PathBuf::from(r"C:\Users\dev")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ]
        );
    }

    #[test]
    fn hf_cache_roots_skips_home_fallback_when_platform_home_unavailable() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(&paths, |_name| None, || None);

        assert_eq!(
            roots,
            vec![PathBuf::from("rocm-cache").join("huggingface").join("hub")]
        );
    }

    /// The "did not download" errors claim they searched a specific set of directories.
    /// Assert that claim against the roots `hf_cache_roots_from` actually returns, so the
    /// message cannot drift from the lookup it describes — in particular, that the
    /// Windows `USERPROFILE` fallback this fix adds is named in the text a user sees.
    #[test]
    fn hf_cache_root_description_names_every_searched_root() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(
            &paths,
            |_name| None,
            || Some(PathBuf::from(r"C:\Users\dev")),
        );
        let described = render_hf_cache_roots(&roots);

        for root in &roots {
            assert!(
                described.contains(&root.display().to_string()),
                "error text {described:?} omits searched root {}",
                root.display()
            );
        }
        assert!(
            described.contains(&PathBuf::from(r"C:\Users\dev").display().to_string()),
            "error text {described:?} must name the platform-home fallback root"
        );
    }

    #[test]
    fn hf_cache_root_description_reports_when_no_roots_resolve() {
        assert_eq!(
            render_hf_cache_roots(&[]),
            "no Hugging Face cache directories could be determined"
        );
    }
}
