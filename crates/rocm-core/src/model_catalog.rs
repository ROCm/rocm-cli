// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Model recipe/registry: the curated model catalog's types, built-in
//! catalog JSON loading, signed external recipe indexes, and artifact
//! cache-marker bookkeeping.
//!
//! Signature verification for a signed recipe index is built on
//! [`crate::crypto`]'s generic RSA primitives.

use crate::crypto::verify_rsa_pkcs1_sha256_signature;
use crate::{AppPaths, env_path_override, require_nonempty, runtime, sanitize_component};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeArtifactSourcePolicyRecord {
    pub policy: String,
    #[serde(default)]
    pub required_hosts: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeArtifactRecord {
    pub artifact_id: String,
    pub kind: String,
    pub uri: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub gated: Option<bool>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub engines: Vec<String>,
    #[serde(default)]
    pub source_policy: Option<ModelRecipeArtifactSourcePolicyRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeEndpointRecord {
    pub endpoint_mode: String,
    #[serde(default)]
    pub settings: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeUnsupportedCombinationRecord {
    pub combination: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeEngineRecord {
    pub engine: String,
    #[serde(default)]
    pub required_flags: Vec<String>,
    #[serde(default)]
    pub parser_settings: BTreeMap<String, String>,
    #[serde(default)]
    pub preferred_endpoint: Option<ModelRecipeEndpointRecord>,
    #[serde(default)]
    pub unsupported_combinations: Vec<ModelRecipeUnsupportedCombinationRecord>,
    #[serde(default)]
    pub notes: Vec<String>,
    /// Overrides the recipe `canonical_model_id` when this engine serves the model.
    /// Lets a single alias resolve to engine-specific artifacts (for example a GGUF
    /// id for Lemonade versus a Hugging Face repo id for vLLM).
    #[serde(default)]
    pub model_id_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelArtifactCacheStatus {
    pub artifact_id: String,
    pub status: String,
    pub marker_path: PathBuf,
    pub reason: String,
}

pub fn model_artifact_cache_marker_path(
    paths: &AppPaths,
    model_ref: &str,
    artifact_id: &str,
) -> PathBuf {
    let model_component = cache_marker_component("model", model_ref);
    let artifact_component = cache_marker_component("artifact", artifact_id);
    paths
        .data_dir
        .join("models")
        .join("artifacts")
        .join(&model_component)
        .join(format!("{artifact_component}.json"))
}

fn cache_marker_component(kind: &str, value: &str) -> String {
    let slug = sanitize_component(value)
        .trim_matches('-')
        .chars()
        .take(32)
        .collect::<String>();
    let slug = if slug.is_empty() {
        kind.to_owned()
    } else {
        slug
    };
    format!("{slug}--x{}", hex_encode_lower(value.as_bytes()))
}

fn hex_encode_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

pub fn model_artifact_cache_status(
    paths: &AppPaths,
    model_ref: &str,
    artifact: &ModelRecipeArtifactRecord,
) -> ModelArtifactCacheStatus {
    let marker_path = model_artifact_cache_marker_path(paths, model_ref, &artifact.artifact_id);
    if marker_path.is_file() {
        ModelArtifactCacheStatus {
            artifact_id: artifact.artifact_id.clone(),
            status: "metadata_present".to_owned(),
            marker_path,
            reason: "rocm-cli artifact cache marker exists; artifact bytes are still engine/source specific".to_owned(),
        }
    } else {
        ModelArtifactCacheStatus {
            artifact_id: artifact.artifact_id.clone(),
            status: "missing".to_owned(),
            marker_path,
            reason:
                "no rocm-cli artifact cache marker; prefetch requires an approved source policy"
                    .to_owned(),
        }
    }
}

pub fn resolve_model_recipe_artifact(
    artifact_ref: &str,
) -> Result<Option<(ModelRecipeRecord, ModelRecipeArtifactRecord)>> {
    require_nonempty(artifact_ref, "artifact_ref")?;
    let registry = load_model_recipe_registry()?;
    let artifact_ref = artifact_ref.trim();
    if let Some((model_ref, artifact_id)) = artifact_ref.split_once('#') {
        require_nonempty(model_ref, "artifact model_ref")?;
        require_nonempty(artifact_id, "artifact_id")?;
        let Some(recipe) = registry
            .recipes
            .into_iter()
            .find(|recipe| recipe.matches_ref(model_ref))
        else {
            return Ok(None);
        };
        return Ok(recipe
            .artifacts
            .iter()
            .position(|artifact| artifact.artifact_id == artifact_id)
            .map(|index| {
                let artifact = recipe.artifacts[index].clone();
                (recipe, artifact)
            }));
    }

    let mut matches = registry
        .recipes
        .into_iter()
        .filter_map(|recipe| {
            recipe
                .artifacts
                .iter()
                .position(|artifact| artifact.artifact_id == artifact_ref)
                .map(|index| {
                    let artifact = recipe.artifacts[index].clone();
                    (recipe, artifact)
                })
        })
        .collect::<Vec<_>>();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.pop()),
        _ => bail!("artifact_ref `{artifact_ref}` is ambiguous; use `<model-ref>#{artifact_ref}`"),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeRecord {
    pub canonical_model_id: String,
    pub aliases: Vec<String>,
    pub task: String,
    pub source: String,
    pub revision: String,
    pub loader: String,
    pub trust_remote_code: bool,
    pub dtype: String,
    pub device_policy: String,
    #[serde(default)]
    pub min_gpu_mem_gb: Option<u32>,
    #[serde(default)]
    pub recommended_system_ram_gb: Option<u32>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub artifact_hint: Option<String>,
    #[serde(default)]
    pub artifacts: Vec<ModelRecipeArtifactRecord>,
    #[serde(default)]
    pub engine_recipes: Vec<ModelRecipeEngineRecord>,
    #[serde(default)]
    pub manual_alternatives: Vec<String>,
    #[serde(default)]
    pub featured: bool,
    pub chat_template_mode: String,
    pub preferred_engines: Vec<String>,
    pub warnings: Vec<String>,
}

impl ModelRecipeRecord {
    pub fn matches_ref(&self, model_ref: &str) -> bool {
        let normalized = normalize_model_ref(model_ref);
        normalize_model_ref(&self.canonical_model_id) == normalized
            || self
                .aliases
                .iter()
                .any(|alias| normalize_model_ref(alias) == normalized)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelCatalogPlatform {
    pub label: String,
    pub engines: Vec<String>,
    #[serde(default)]
    pub gfx_families: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeIndexDocument {
    pub schema_version: u32,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub generated_at_unix_ms: Option<u128>,
    #[serde(default)]
    pub platforms: Vec<ModelCatalogPlatform>,

    recipes: Vec<ModelRecipeRecord>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModelRecipeRegistry {
    pub recipes: Vec<ModelRecipeRecord>,
    pub platforms: Vec<ModelCatalogPlatform>,
    pub source: ModelRecipeRegistrySource,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModelRecipeRegistrySource {
    BuiltIn,
    SignedIndex {
        index_path: PathBuf,
        signature_path: PathBuf,
        public_key_path: PathBuf,
    },
}

impl ModelRecipeIndexDocument {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!(
                "model recipe index schema_version {} is unsupported; expected 1",
                self.schema_version
            );
        }
        if self.recipes.is_empty() {
            bail!("model recipe index must contain at least one recipe");
        }

        let mut refs = BTreeMap::<String, String>::new();
        for recipe in &self.recipes {
            require_nonempty(&recipe.canonical_model_id, "canonical_model_id")?;
            require_nonempty(&recipe.task, "task")?;
            require_nonempty(&recipe.source, "source")?;
            require_nonempty(&recipe.revision, "revision")?;
            require_nonempty(&recipe.loader, "loader")?;
            require_nonempty(&recipe.dtype, "dtype")?;
            require_nonempty(&recipe.device_policy, "device_policy")?;
            require_nonempty(&recipe.chat_template_mode, "chat_template_mode")?;
            validate_model_device_policy(&recipe.device_policy)?;
            insert_unique_model_ref(
                &mut refs,
                &recipe.canonical_model_id,
                &recipe.canonical_model_id,
            )?;
            for alias in &recipe.aliases {
                require_nonempty(alias, "alias")?;
                insert_unique_model_ref(&mut refs, alias, &recipe.canonical_model_id)?;
            }
            for artifact in &recipe.artifacts {
                validate_model_recipe_artifact(artifact, &recipe.canonical_model_id)?;
            }
            let mut engines = BTreeMap::<String, String>::new();
            for engine_recipe in &recipe.engine_recipes {
                validate_model_recipe_engine_record(engine_recipe, &recipe.canonical_model_id)?;
                let normalized = normalize_model_ref(&engine_recipe.engine);
                if let Some(existing) = engines.insert(normalized, engine_recipe.engine.clone()) {
                    bail!(
                        "engine recipe for `{}` on `{}` is duplicated by `{existing}`",
                        engine_recipe.engine,
                        recipe.canonical_model_id
                    );
                }
            }
        }

        Ok(())
    }
}

pub fn builtin_model_recipe_registry() -> ModelRecipeRegistry {
    let doc = builtin_model_catalog_document();
    ModelRecipeRegistry {
        recipes: doc.recipes.clone(),
        platforms: doc.platforms.clone(),
        source: ModelRecipeRegistrySource::BuiltIn,
    }
}

pub fn load_model_recipe_registry() -> Result<ModelRecipeRegistry> {
    let configured_index = env_path_override("ROCM_CLI_MODEL_RECIPE_INDEX_PATH");
    if configured_index.is_none() && runtime::env_flag("ROCM_CLI_REQUIRE_MODEL_RECIPE_SIGNATURE") {
        bail!(
            "signed model recipe index is required but ROCM_CLI_MODEL_RECIPE_INDEX_PATH is not configured"
        );
    }
    let Some(index_path) = configured_index else {
        return Ok(builtin_model_recipe_registry());
    };

    let signature_path = env_path_override("ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH")
        .unwrap_or_else(|| model_recipe_index_signature_path(&index_path));
    let public_key_path = env_path_override("ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH")
        .context(
            "signed model recipe index requires ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH",
        )?;
    let document = load_signed_model_recipe_index(&index_path, &signature_path, &public_key_path)?;
    let platforms = if document.platforms.is_empty() {
        builtin_model_catalog_document().platforms.clone()
    } else {
        document.platforms
    };
    Ok(ModelRecipeRegistry {
        recipes: document.recipes,
        platforms,
        source: ModelRecipeRegistrySource::SignedIndex {
            index_path,
            signature_path,
            public_key_path,
        },
    })
}

pub fn resolve_model_recipe(model_ref: &str) -> Result<Option<ModelRecipeRecord>> {
    Ok(load_model_recipe_registry()?
        .recipes
        .into_iter()
        .find(|recipe| recipe.matches_ref(model_ref)))
}

pub fn load_signed_model_recipe_index(
    index_path: &Path,
    signature_path: &Path,
    public_key_path: &Path,
) -> Result<ModelRecipeIndexDocument> {
    verify_model_recipe_index_signature(index_path, signature_path, public_key_path)?;
    let document = load_model_recipe_index(index_path)?;
    document.validate()?;
    Ok(document)
}

pub fn load_model_recipe_index(index_path: &Path) -> Result<ModelRecipeIndexDocument> {
    let bytes = fs::read(index_path)
        .with_context(|| format!("failed to read model recipe index {}", index_path.display()))?;
    let document =
        serde_json::from_slice::<ModelRecipeIndexDocument>(&bytes).with_context(|| {
            format!(
                "failed to parse model recipe index {}",
                index_path.display()
            )
        })?;
    document.validate()?;
    Ok(document)
}

pub fn model_recipe_index_signature_path(index_path: &Path) -> PathBuf {
    let mut signature = index_path.as_os_str().to_os_string();
    signature.push(".sig");
    PathBuf::from(signature)
}

pub fn verify_model_recipe_index_signature(
    index_path: &Path,
    signature_path: &Path,
    public_key_path: &Path,
) -> Result<()> {
    if !signature_path.is_file() {
        bail!(
            "model recipe index signature is missing: {}",
            signature_path.display()
        );
    }
    if !public_key_path.is_file() {
        bail!(
            "model recipe index public key is missing: {}",
            public_key_path.display()
        );
    }
    let public_key_pem = fs::read_to_string(public_key_path).with_context(|| {
        format!(
            "failed to read model recipe index public key: {}",
            public_key_path.display()
        )
    })?;
    let signature = fs::read(signature_path).with_context(|| {
        format!(
            "failed to read model recipe index signature: {}",
            signature_path.display()
        )
    })?;
    let payload = fs::read(index_path).with_context(|| {
        format!(
            "failed to read model recipe index: {}",
            index_path.display()
        )
    })?;
    verify_rsa_pkcs1_sha256_signature(&public_key_pem, &payload, &signature, "model recipe index")
}

fn validate_model_device_policy(policy: &str) -> Result<()> {
    match policy {
        "gpu_required" | "gpu_preferred" | "cpu_only" => Ok(()),
        other => bail!(
            "model recipe device_policy `{other}` is unsupported; expected gpu_required, gpu_preferred, or cpu_only"
        ),
    }
}

fn insert_unique_model_ref(
    refs: &mut BTreeMap<String, String>,
    model_ref: &str,
    canonical_model_id: &str,
) -> Result<()> {
    let normalized = normalize_model_ref(model_ref);
    if let Some(existing) = refs.insert(normalized, canonical_model_id.to_owned()) {
        bail!(
            "model recipe ref `{model_ref}` is duplicated by `{existing}` and `{canonical_model_id}`"
        );
    }
    Ok(())
}

fn validate_model_recipe_artifact(
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    require_nonempty(&artifact.artifact_id, "artifact_id")?;
    require_nonempty(&artifact.kind, "artifact kind")?;
    require_nonempty(&artifact.uri, "artifact uri")?;
    if let Some(sha256) = artifact.sha256.as_deref()
        && (sha256.len() != 64 || !sha256.chars().all(|ch| ch.is_ascii_hexdigit()))
    {
        bail!(
            "artifact `{}` for `{canonical_model_id}` has invalid sha256",
            artifact.artifact_id
        );
    }
    if let Some(source_policy) = &artifact.source_policy {
        validate_model_recipe_artifact_source_policy(source_policy, artifact, canonical_model_id)?;
    }
    Ok(())
}

fn validate_model_recipe_artifact_source_policy(
    source_policy: &ModelRecipeArtifactSourcePolicyRecord,
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    require_nonempty(&source_policy.policy, "artifact source_policy policy")?;
    for host in &source_policy.required_hosts {
        require_nonempty(host, "artifact source_policy required_host")?;
        if host.contains('/') || host.contains('@') || host.contains(':') {
            bail!(
                "artifact `{}` for `{canonical_model_id}` has invalid source_policy required_host `{host}`",
                artifact.artifact_id
            );
        }
    }
    for note in &source_policy.notes {
        require_nonempty(note, "artifact source_policy note")?;
    }

    if !source_policy.required_hosts.is_empty() {
        let Some(host) = recipe_artifact_url_host(&artifact.uri) else {
            bail!(
                "artifact `{}` for `{canonical_model_id}` declares required source hosts but its uri is not HTTP(S)",
                artifact.artifact_id
            );
        };
        if !source_policy
            .required_hosts
            .iter()
            .any(|required| required.eq_ignore_ascii_case(&host))
        {
            bail!(
                "artifact `{}` for `{canonical_model_id}` uri host `{host}` is not allowed by source_policy",
                artifact.artifact_id
            );
        }
    }

    match source_policy.policy.as_str() {
        "direct_https_sha256" => {
            if !artifact.uri.starts_with("https://") {
                bail!(
                    "artifact `{}` for `{canonical_model_id}` source_policy direct_https_sha256 requires an HTTPS uri",
                    artifact.artifact_id
                );
            }
            validate_prefetch_integrity_metadata(artifact, canonical_model_id)?;
        }
        "huggingface_public" => {
            if artifact.gated.unwrap_or(false) {
                bail!(
                    "artifact `{}` for `{canonical_model_id}` source_policy huggingface_public cannot be used for a gated artifact",
                    artifact.artifact_id
                );
            }
            validate_huggingface_source_policy_uri(source_policy, artifact, canonical_model_id)?;
            validate_prefetch_integrity_metadata(artifact, canonical_model_id)?;
        }
        "huggingface_authenticated" => {
            validate_huggingface_source_policy_uri(source_policy, artifact, canonical_model_id)?;
            validate_prefetch_integrity_metadata(artifact, canonical_model_id)?;
        }
        "manual_only" => {}
        other => bail!(
            "artifact `{}` for `{canonical_model_id}` has unsupported source_policy `{other}`",
            artifact.artifact_id
        ),
    }
    Ok(())
}

fn validate_prefetch_integrity_metadata(
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    if artifact.sha256.is_none() {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy requires sha256 metadata",
            artifact.artifact_id
        );
    }
    if artifact.size_bytes.is_none() {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy requires size_bytes metadata",
            artifact.artifact_id
        );
    }
    Ok(())
}

fn validate_huggingface_source_policy_uri(
    source_policy: &ModelRecipeArtifactSourcePolicyRecord,
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    if !artifact.uri.starts_with("https://") {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy {} requires an HTTPS Hugging Face uri",
            artifact.artifact_id,
            source_policy.policy
        );
    }
    if !recipe_artifact_uri_is_huggingface(&artifact.uri) {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy {} requires a Hugging Face uri",
            artifact.artifact_id,
            source_policy.policy
        );
    }
    Ok(())
}

fn recipe_artifact_uri_is_huggingface(uri: &str) -> bool {
    recipe_artifact_url_host(uri).is_some_and(|host| {
        host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co")
    })
}

fn recipe_artifact_url_host(uri: &str) -> Option<String> {
    let rest = uri
        .strip_prefix("https://")
        .or_else(|| uri.strip_prefix("http://"))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim();
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let host = authority
        .strip_prefix('[')
        .and_then(|value| value.split_once(']').map(|(host, _)| host))
        .unwrap_or_else(|| authority.split(':').next().unwrap_or_default())
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn validate_model_recipe_engine_record(
    engine_recipe: &ModelRecipeEngineRecord,
    canonical_model_id: &str,
) -> Result<()> {
    require_nonempty(&engine_recipe.engine, "engine recipe engine")?;
    for flag in &engine_recipe.required_flags {
        require_nonempty(flag, "engine required flag")?;
    }
    for (key, value) in &engine_recipe.parser_settings {
        require_nonempty(key, "engine parser setting key")?;
        require_nonempty(value, "engine parser setting value")?;
    }
    if let Some(endpoint) = engine_recipe.preferred_endpoint.as_ref() {
        require_nonempty(&endpoint.endpoint_mode, "engine preferred endpoint mode")?;
        for (key, value) in &endpoint.settings {
            require_nonempty(key, "engine endpoint setting key")?;
            require_nonempty(value, "engine endpoint setting value")?;
        }
    }
    for item in &engine_recipe.unsupported_combinations {
        require_nonempty(&item.combination, "engine unsupported combination")?;
        require_nonempty(&item.reason, "engine unsupported combination reason")?;
    }
    for note in &engine_recipe.notes {
        require_nonempty(note, "engine recipe note")?;
    }
    if let Some(model_id_override) = engine_recipe.model_id_override.as_deref() {
        require_nonempty(model_id_override, "engine model id override")?;
    }
    if engine_recipe.required_flags.is_empty()
        && engine_recipe.parser_settings.is_empty()
        && engine_recipe.preferred_endpoint.is_none()
        && engine_recipe.unsupported_combinations.is_empty()
        && engine_recipe.notes.is_empty()
        && engine_recipe.model_id_override.is_none()
    {
        bail!(
            "engine recipe for `{}` on `{canonical_model_id}` must not be empty",
            engine_recipe.engine
        );
    }
    Ok(())
}

pub fn builtin_model_recipes() -> Vec<ModelRecipeRecord> {
    builtin_model_catalog_document().recipes.clone()
}

/// The curated fallback catalog shipped inside the binary. It is authored as JSON
/// (`model_catalog.json`) using the same schema as an external signed recipe
/// index, so the offline default and hosted indexes share one format. Parsed once
/// and cached; a malformed catalog is a test-time bug guarded by a unit test.
fn builtin_model_catalog_document() -> &'static ModelRecipeIndexDocument {
    static CATALOG: std::sync::OnceLock<ModelRecipeIndexDocument> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let document =
            serde_json::from_str::<ModelRecipeIndexDocument>(include_str!("model_catalog.json"))
                .expect("built-in model catalog JSON must parse");
        document
            .validate()
            .expect("built-in model catalog must satisfy the recipe index schema");
        document
    })
}

pub fn resolve_builtin_model_recipe(model_ref: &str) -> Option<ModelRecipeRecord> {
    builtin_model_recipes()
        .into_iter()
        .find(|recipe| recipe.matches_ref(model_ref))
}

/// Returns the ordered platform definitions from the registry.
pub fn model_catalog_platforms(registry: &ModelRecipeRegistry) -> Vec<ModelCatalogPlatform> {
    registry.platforms.clone()
}

/// The label of the hardware platform a recipe targets, derived from its first
/// preferred engine matched against the catalog's platform definitions.
pub fn model_recipe_target_platform_label(
    recipe: &ModelRecipeRecord,
    platforms: &[ModelCatalogPlatform],
) -> String {
    let engine = recipe
        .preferred_engines
        .first()
        .map(|e| e.trim().to_ascii_lowercase())
        .unwrap_or_default();
    platforms
        .iter()
        .find(|p| p.engines.iter().any(|e| e.eq_ignore_ascii_case(&engine)))
        .map_or_else(|| engine.clone(), |p| p.label.clone())
}

/// Whether the given normalized TheRock family matches a platform's gfx targets.
pub fn platform_matches_gfx_family(platform: &ModelCatalogPlatform, gfx_family: &str) -> bool {
    platform
        .gfx_families
        .iter()
        .any(|f| f.eq_ignore_ascii_case(gfx_family))
}

/// Whether a recipe appears in the curated `rocm model` short list.
///
/// Driven by the `featured` field in the catalog JSON. Hidden recipes stay fully
/// resolvable for `rocm serve` and the crate's smoke tests; only the user-facing
/// `rocm model` list omits them.
pub const fn model_recipe_featured(recipe: &ModelRecipeRecord) -> bool {
    recipe.featured
}

pub fn normalize_model_ref(model_ref: &str) -> String {
    model_ref.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature};
    use crate::test_support::temp_app_paths;

    #[test]
    fn builtin_model_catalog_json_parses_and_validates() {
        // Guards the embedded catalog: malformed JSON, a bad device policy, or a
        // duplicate alias/canonical id would fail the shared index schema here
        // instead of panicking at runtime.
        let document =
            serde_json::from_str::<ModelRecipeIndexDocument>(include_str!("model_catalog.json"))
                .expect("catalog JSON parses");
        document
            .validate()
            .expect("catalog satisfies the index schema");
        assert!(
            document.recipes.len() >= 10,
            "curated catalog is non-trivial"
        );
        // The default Lemonade assistant must remain resolvable from the catalog.
        assert!(
            document
                .recipes
                .iter()
                .any(|recipe| recipe.canonical_model_id == "Qwen3-4B-Instruct-2507-GGUF"),
            "built-in assistant recipe present"
        );
    }

    #[test]
    fn builtin_catalog_authors_vllm_tool_call_parsers() {
        // vLLM does not auto-detect a tool-call parser; the correct value is sourced
        // from explicit per-model recipe metadata (never guessed at runtime). This
        // guards the authored parser for well-known chat families so a regression is
        // caught here rather than as an HTTP 400 in the TUI chat tab.
        let document =
            serde_json::from_str::<ModelRecipeIndexDocument>(include_str!("model_catalog.json"))
                .expect("catalog JSON parses");
        let tool_call_parser = |model_id: &str| -> Option<String> {
            document
                .recipes
                .iter()
                .find(|recipe| recipe.canonical_model_id == model_id)?
                .engine_recipes
                .iter()
                .find(|engine_recipe| engine_recipe.engine == "vllm")
                .and_then(|engine_recipe| {
                    let flags = &engine_recipe.required_flags;
                    assert!(
                        flags.iter().any(|flag| flag == "--enable-auto-tool-choice"),
                        "{model_id}: --tool-call-parser must be paired with --enable-auto-tool-choice"
                    );
                    let index = flags.iter().position(|flag| flag == "--tool-call-parser")?;
                    flags.get(index + 1).cloned()
                })
        };
        // Reported repro: a lemonade-preferred Qwen forced onto vLLM must still
        // carry the Qwen-family parser.
        assert_eq!(
            tool_call_parser("Qwen/Qwen2.5-1.5B-Instruct").as_deref(),
            Some("hermes")
        );
        assert_eq!(
            tool_call_parser("Qwen/Qwen3-32B-FP8").as_deref(),
            Some("hermes")
        );
        assert_eq!(
            tool_call_parser("meta-llama/Llama-3.2-3B-Instruct").as_deref(),
            Some("llama3_json")
        );
    }

    #[test]
    fn model_recipe_target_platform_groups_by_engine() {
        let registry = builtin_model_recipe_registry();
        let platforms = model_catalog_platforms(&registry);
        // The (hidden) built-in assistant is a Lemonade recipe → Ryzen AI (Strix Halo).
        let strix = resolve_builtin_model_recipe("qwen").expect("qwen assistant");
        assert_eq!(
            model_recipe_target_platform_label(&strix, &platforms),
            "AMD Ryzen AI - Strix Halo (Lemonade / llama.cpp)"
        );
        let mi300x = resolve_builtin_model_recipe("qwen3.6-27b").expect("qwen3.6-27b");
        assert_eq!(
            model_recipe_target_platform_label(&mi300x, &platforms),
            "AMD Instinct - MI300X, MI350X, MI355X (vLLM)"
        );
        // vLLM recipes land on the Instinct platform.
        let llama = resolve_builtin_model_recipe("llama-3.2-3b-instruct").expect("llama");
        assert_eq!(
            model_recipe_target_platform_label(&llama, &platforms),
            "AMD Instinct - MI300X, MI350X, MI355X (vLLM)"
        );
    }

    #[test]
    fn featured_catalog_is_curated_but_hidden_stay_resolvable() {
        // Current popular models are featured in the curated list — GGUF for
        // Strix Halo (served by their owner/repo:variant id) and BF16 for MI300X.
        for alias in ["qwen3.6", "gemma-4", "qwen3.6-27b", "gemma-4-31b"] {
            let recipe = resolve_builtin_model_recipe(alias).unwrap_or_else(|| panic!("{alias}"));
            assert!(model_recipe_featured(&recipe), "{alias} should be featured");
        }
        // The Strix Halo entries carry an explicit GGUF quant variant so they are
        // directly servable via Lemonade.
        assert_eq!(
            resolve_builtin_model_recipe("qwen3.6")
                .unwrap()
                .canonical_model_id,
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
        );
        // ...while the default assistant, smoke paths, and superseded models stay
        // resolvable for `rocm serve` but are hidden from the curated list.
        for alias in ["qwen", "tiny-gpt2", "qwen3.5", "glm-5"] {
            let recipe = resolve_builtin_model_recipe(alias).unwrap_or_else(|| panic!("{alias}"));
            assert!(!model_recipe_featured(&recipe), "{alias} should be hidden");
        }
    }

    #[test]
    fn builtin_recipe_resolves_alias_and_canonical_model() {
        let qwen = resolve_builtin_model_recipe("qwen").expect("qwen alias should resolve");
        assert_eq!(qwen.canonical_model_id, "Qwen3-4B-Instruct-2507-GGUF");
        assert_eq!(qwen.dtype, "gguf");
        assert_eq!(qwen.device_policy, "gpu_required");
        assert_eq!(qwen.preferred_engines, vec!["lemonade"]);

        let qwen35 = resolve_builtin_model_recipe("qwen3.5").expect("qwen3.5 alias should resolve");
        assert_eq!(qwen35.canonical_model_id, "Qwen/Qwen3.5-4B");
        assert_eq!(qwen35.preferred_engines, vec!["vllm"]);
        let lemonade_qwen =
            resolve_builtin_model_recipe("lemonade-qwen").expect("lemonade qwen alias");
        assert_eq!(
            lemonade_qwen.canonical_model_id,
            "Qwen3-4B-Instruct-2507-GGUF"
        );
        assert_eq!(lemonade_qwen.preferred_engines, vec!["lemonade"]);
        assert_eq!(lemonade_qwen.device_policy, "gpu_required");
        assert!(
            qwen35
                .warnings
                .iter()
                .any(|warning| warning.contains("qwen3_5"))
        );

        let tiny = resolve_builtin_model_recipe("sshleifer/tiny-gpt2")
            .expect("canonical tiny model should resolve");
        assert_eq!(tiny.canonical_model_id, "sshleifer/tiny-gpt2");
        assert_eq!(tiny.device_policy, "gpu_required");
        assert_eq!(tiny.min_gpu_mem_gb, Some(2));
    }

    #[test]
    fn builtin_recipe_records_remote_code_policy() {
        let glm = resolve_builtin_model_recipe("glm-5").expect("glm alias should resolve");
        assert!(glm.trust_remote_code);
        assert_eq!(glm.device_policy, "gpu_required");
        assert!(
            glm.warnings
                .iter()
                .any(|item| item.contains("trust_remote_code"))
        );
    }

    #[test]
    fn model_recipe_index_validates_artifact_metadata() -> Result<()> {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.artifacts[0].uri =
            "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned();
        recipe.artifacts[0].source_policy = Some(ModelRecipeArtifactSourcePolicyRecord {
            policy: "huggingface_public".to_owned(),
            required_hosts: vec!["huggingface.co".to_owned()],
            notes: vec!["test metadata only".to_owned()],
        });
        recipe.engine_recipes.push(ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            parser_settings: BTreeMap::from([("reasoning_parser".to_owned(), "qwen3".to_owned())]),
            preferred_endpoint: Some(ModelRecipeEndpointRecord {
                endpoint_mode: "openai".to_owned(),
                settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
            }),
            unsupported_combinations: vec![ModelRecipeUnsupportedCombinationRecord {
                combination: "native Windows GPU serving".to_owned(),
                reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
            }],
            notes: vec!["metadata only; adapter protocol does not consume this yet".to_owned()],
            model_id_override: None,
        });
        let index = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        };

        index.validate()?;

        let artifact = &index.recipes[0].artifacts[0];
        assert_eq!(artifact.kind, "huggingface");
        let expected_sha = "a".repeat(64);
        assert_eq!(artifact.sha256.as_deref(), Some(expected_sha.as_str()));
        assert_eq!(artifact.engines, vec!["vllm"]);
        assert_eq!(
            artifact
                .source_policy
                .as_ref()
                .map(|policy| policy.policy.as_str()),
            Some("huggingface_public")
        );
        let settings = index.recipes[0]
            .engine_recipes
            .first()
            .expect("vllm settings should validate");
        assert_eq!(
            settings.parser_settings.get("reasoning_parser"),
            Some(&"qwen3".to_owned())
        );
        assert_eq!(
            settings
                .preferred_endpoint
                .as_ref()
                .map(|endpoint| endpoint.endpoint_mode.as_str()),
            Some("openai")
        );
        Ok(())
    }

    #[test]
    fn model_recipe_index_rejects_invalid_artifact_source_policy() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.artifacts[0].uri =
            "https://example.invalid/Qwen/Test-1B/model.safetensors".to_owned();
        recipe.artifacts[0].source_policy = Some(ModelRecipeArtifactSourcePolicyRecord {
            policy: "huggingface_authenticated".to_owned(),
            required_hosts: vec!["huggingface.co".to_owned()],
            notes: Vec::new(),
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("source policy host mismatch should be rejected")
        .to_string();

        assert!(error.contains("source_policy"));
        assert!(error.contains("not allowed"));
    }

    #[test]
    fn model_recipe_index_source_policy_requires_integrity_metadata() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.artifacts[0].uri = "https://example.invalid/model.bin".to_owned();
        recipe.artifacts[0].sha256 = None;
        recipe.artifacts[0].source_policy = Some(ModelRecipeArtifactSourcePolicyRecord {
            policy: "direct_https_sha256".to_owned(),
            required_hosts: vec!["example.invalid".to_owned()],
            notes: Vec::new(),
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("source policy should require sha256")
        .to_string();

        assert!(error.contains("requires sha256"));
    }

    #[test]
    fn model_recipe_index_rejects_empty_engine_recipe() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.engine_recipes.push(ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            ..ModelRecipeEngineRecord::default()
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("empty engine recipe should be rejected")
        .to_string();

        assert!(error.contains("engine recipe for `vllm`"));
        assert!(error.contains("must not be empty"));
    }

    #[test]
    fn model_recipe_index_rejects_duplicate_engine_recipes() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.engine_recipes = vec![
            ModelRecipeEngineRecord {
                engine: "vllm".to_owned(),
                notes: vec!["first".to_owned()],
                ..ModelRecipeEngineRecord::default()
            },
            ModelRecipeEngineRecord {
                engine: "VLLM".to_owned(),
                notes: vec!["second".to_owned()],
                ..ModelRecipeEngineRecord::default()
            },
        ];

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("duplicate engine recipes should be rejected")
        .to_string();

        assert!(error.contains("engine recipe for `VLLM`"));
        assert!(error.contains("duplicated"));
    }

    #[test]
    fn model_recipe_index_requires_unsupported_combination_reason() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.engine_recipes.push(ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            unsupported_combinations: vec![ModelRecipeUnsupportedCombinationRecord {
                combination: "native Windows GPU serving".to_owned(),
                reason: String::new(),
            }],
            ..ModelRecipeEngineRecord::default()
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("unsupported combinations need reasons")
        .to_string();

        assert!(error.contains("engine unsupported combination reason"));
    }

    #[test]
    fn model_artifact_cache_status_uses_deterministic_marker_without_creating_dirs() -> Result<()> {
        let (root, paths) = temp_app_paths("artifact-cache-status");
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        let artifact = recipe.artifacts.remove(0);

        let missing = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);
        assert_eq!(missing.status, "missing");
        assert!(
            missing
                .marker_path
                .to_string_lossy()
                .contains("hf-main--x68662d6d61696e.json")
        );
        assert!(
            missing
                .marker_path
                .to_string_lossy()
                .contains("qwen-test-1b")
        );
        assert!(!paths.data_dir.exists());

        let parent = missing.marker_path.parent().expect("marker has parent");
        fs::create_dir_all(parent)?;
        fs::write(&missing.marker_path, "{}")?;
        let present = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);

        fs::remove_dir_all(root).ok();
        assert_eq!(present.status, "metadata_present");
        Ok(())
    }

    #[test]
    fn model_artifact_cache_marker_path_includes_model_identity() {
        let (_root, paths) = temp_app_paths("artifact-cache-model-scope");

        let first = model_artifact_cache_marker_path(&paths, "Qwen/Test-1B", "hf-main");
        let second = model_artifact_cache_marker_path(&paths, "Qwen/Other-1B", "hf-main");

        assert_ne!(first, second);
        assert!(first.to_string_lossy().contains("qwen-test-1b"));
        assert!(second.to_string_lossy().contains("qwen-other-1b"));
    }

    #[test]
    fn model_artifact_cache_marker_path_is_collision_proof_for_similar_refs() {
        let (_root, paths) = temp_app_paths("artifact-cache-collision-proof");

        let dash = model_artifact_cache_marker_path(&paths, "Qwen/Test-1B", "hf-main");
        let underscore = model_artifact_cache_marker_path(&paths, "Qwen/Test_1B", "hf-main");
        let case_variant = model_artifact_cache_marker_path(&paths, "qwen/test-1b", "hf-main");

        assert_ne!(dash, underscore);
        assert_ne!(dash, case_variant);
        assert!(
            dash.to_string_lossy()
                .contains("--x5177656e2f546573742d3142")
        );
        assert!(
            underscore
                .to_string_lossy()
                .contains("--x5177656e2f546573745f3142")
        );
        assert!(
            case_variant
                .to_string_lossy()
                .contains("--x7177656e2f746573742d3162")
        );
    }

    #[test]
    fn model_recipe_index_rejects_duplicate_aliases() {
        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: None,
            generated_at_unix_ms: None,
            platforms: Vec::new(),
            recipes: vec![
                sample_recipe_with_artifact("Qwen/Test-1B", &["shared-alias"]),
                sample_recipe_with_artifact("Qwen/Other-1B", &["shared-alias"]),
            ],
        }
        .validate()
        .expect_err("duplicate aliases should be rejected")
        .to_string();

        assert!(error.contains("duplicated"));
        assert!(error.contains("shared-alias"));
    }

    #[test]
    fn load_model_recipe_index_reads_local_fixture() -> Result<()> {
        let (root, _paths) = temp_app_paths("recipe-index-fixture");
        let index_path = root.join("recipes.json");
        let document = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"])],
        };
        fs::create_dir_all(&root)?;
        fs::write(&index_path, serde_json::to_vec_pretty(&document)?)?;

        let loaded = load_model_recipe_index(&index_path)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(loaded.source.as_deref(), Some("fixture"));
        assert_eq!(loaded.recipes[0].canonical_model_id, "Qwen/Test-1B");
        assert_eq!(loaded.recipes[0].artifacts.len(), 1);
        Ok(())
    }

    #[test]
    fn model_recipe_index_signature_path_is_detached_sidecar() {
        assert_eq!(
            model_recipe_index_signature_path(Path::new("recipes/index.json")),
            PathBuf::from("recipes/index.json.sig")
        );
    }

    #[test]
    fn model_recipe_index_signature_accepts_generated_key_and_rejects_tamper() -> Result<()> {
        let (root, _paths) = temp_app_paths("recipe-index-generated-signature");
        fs::create_dir_all(&root)?;
        let private_key = root.join("recipe-private.pem");
        let public_key = root.join("recipe-public.pem");
        let index_path = root.join("recipes.json");
        let signature_path = model_recipe_index_signature_path(&index_path);
        let document = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"])],
        };

        generate_test_signing_key(&private_key, &public_key)?;
        fs::write(&index_path, serde_json::to_vec_pretty(&document)?)?;
        sign_test_payload(&private_key, &index_path, &signature_path)?;

        load_signed_model_recipe_index(&index_path, &signature_path, &public_key)?;

        let tampered = ModelRecipeIndexDocument {
            source: Some("tampered".to_owned()),
            ..document
        };
        fs::write(&index_path, serde_json::to_vec_pretty(&tampered)?)?;
        let error = load_signed_model_recipe_index(&index_path, &signature_path, &public_key)
            .unwrap_err()
            .to_string();

        assert!(error.contains("model recipe index signature verification failed"));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    fn generate_test_signing_key(private_key: &Path, public_key: &Path) -> Result<()> {
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        fs::write(private_key, private_pem.as_bytes())?;
        fs::write(public_key, public_pem.as_bytes())?;
        Ok(())
    }

    fn sign_test_payload(private_key: &Path, payload: &Path, signature: &Path) -> Result<()> {
        let private_pem = fs::read_to_string(private_key)?;
        let payload_bytes = fs::read(payload)?;
        let produced = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload_bytes)?;
        fs::write(signature, produced)?;
        Ok(())
    }

    fn sample_recipe_with_artifact(
        canonical_model_id: &str,
        aliases: &[&str],
    ) -> ModelRecipeRecord {
        ModelRecipeRecord {
            canonical_model_id: canonical_model_id.to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            task: "chat".to_owned(),
            source: "signed_recipe_index".to_owned(),
            revision: "main".to_owned(),
            loader: "transformers".to_owned(),
            trust_remote_code: false,
            dtype: "bfloat16".to_owned(),
            device_policy: "gpu_required".to_owned(),
            min_gpu_mem_gb: Some(12),
            recommended_system_ram_gb: Some(16),
            quantization: Some("none".to_owned()),
            artifact_hint: None,
            artifacts: vec![ModelRecipeArtifactRecord {
                artifact_id: "hf-main".to_owned(),
                kind: "huggingface".to_owned(),
                uri: canonical_model_id.to_owned(),
                revision: Some("main".to_owned()),
                sha256: Some("a".repeat(64)),
                size_bytes: Some(1024),
                license: Some("apache-2.0".to_owned()),
                gated: Some(false),
                quantization: Some("none".to_owned()),
                engines: vec!["vllm".to_owned()],
                source_policy: None,
            }],
            engine_recipes: Vec::new(),
            manual_alternatives: Vec::new(),
            featured: false,
            chat_template_mode: "auto".to_owned(),
            preferred_engines: vec!["vllm".to_owned()],
            warnings: Vec::new(),
        }
    }
}
