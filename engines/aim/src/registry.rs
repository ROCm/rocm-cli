// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! AIM image metadata (ROCMAI-553 finding): the image config carries structured
//! `Env`/`Labels`. Read it from the local daemon once the image is present (any
//! registry), or, as a best-effort pre-pull hint, from Docker Hub's v2 API via
//! two plain HTTP calls. The Docker Hub lookup is only ever a hint: callers must
//! tolerate its failure (offline, rate-limited, or a non-Docker-Hub registry).

use anyhow::{Context, Result};
use serde_json::Value;
use std::time::Duration;

const REGISTRY_HOST: &str = "registry-1.docker.io";
const AUTH_HOST: &str = "auth.docker.io";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
pub(crate) const DEFAULT_ORG: &str = "amdenterpriseai";

#[derive(Debug, Default, Clone)]
pub(crate) struct AimImageMetadata {
    pub accelerator_type: Option<String>,
    pub pytorch_rocm_arch: Vec<String>,
    pub canonical_name: Option<String>,
}

/// Normalizes a user- or catalog-supplied image reference so the registry
/// metadata lookup and the real `docker pull`/`run` always resolve the exact
/// same image. A bare (org-less) name defaults to [`DEFAULT_ORG`], matching
/// this engine's own catalog convention — Docker's own bare-name default
/// (`library/`) would silently point at an unrelated official image instead.
pub(crate) fn normalize_image_ref(image_ref: &str) -> String {
    if image_ref.contains('/') {
        image_ref.to_owned()
    } else {
        format!("{DEFAULT_ORG}/{image_ref}")
    }
}

/// Splits a normalized (see [`normalize_image_ref`]) `org/repo[:tag][@digest]`
/// ref into `(repo, reference)`. A digest wins over a tag (registries resolve
/// by digest), and the tag is dropped from `repo` so `repo:tag@sha256:...`
/// doesn't leave `:tag` in the name used for the token scope and manifest URL.
/// A tag's `:` is only looked for after the last `/`, so a `host:port/repo` ref
/// isn't mis-split by a naive first-`:` search.
fn split_image_ref(image_ref: &str) -> (String, String) {
    let (name, digest) = match image_ref.split_once('@') {
        Some((name, digest)) => (name, Some(digest)),
        None => (image_ref, None),
    };
    let tag_sep = name
        .rfind(':')
        .filter(|&i| i > name.rfind('/').unwrap_or(0));
    let (repo, tag) = match tag_sep {
        Some(i) => (&name[..i], Some(&name[i + 1..])),
        None => (name, None),
    };
    let reference = digest.or(tag).unwrap_or("latest");
    (repo.to_owned(), reference.to_owned())
}

/// Whether the (normalized) ref lives on Docker Hub, the only registry
/// [`fetch_image_metadata`] can read. The first path component is a registry
/// host when it looks like one (contains `.` or `:`, or is `localhost`).
fn is_docker_hub_ref(normalized: &str) -> bool {
    let first = normalized.split('/').next().unwrap_or_default();
    !(first.contains('.') || first.contains(':') || first == "localhost")
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(DEFAULT_TIMEOUT)
        .timeout(DEFAULT_TIMEOUT)
        .build()
}

fn anonymous_pull_token(repo: &str) -> Result<String> {
    let url = format!(
        "https://{AUTH_HOST}/token?service=registry.docker.io&scope=repository:{repo}:pull"
    );
    let body: Value = agent()
        .get(&url)
        .set("User-Agent", "rocm-cli")
        .call()
        .with_context(|| format!("failed to fetch registry auth token for {repo}"))?
        .into_json()
        .with_context(|| format!("failed to parse registry auth token response for {repo}"))?;
    body.get("token")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .with_context(|| format!("registry auth response for {repo} had no token"))
}

/// Best-effort pre-pull metadata hint from Docker Hub. Errors for any other
/// registry; callers treat every error as "no hint".
pub(crate) fn fetch_image_metadata(image_ref: &str) -> Result<AimImageMetadata> {
    let normalized = normalize_image_ref(image_ref);
    if !is_docker_hub_ref(&normalized) {
        anyhow::bail!("`{normalized}` is not a Docker Hub image; skipping the registry lookup");
    }
    let (repo, reference) = split_image_ref(&normalized);
    let token = anonymous_pull_token(&repo)?;

    let manifest_url = format!("https://{REGISTRY_HOST}/v2/{repo}/manifests/{reference}");
    let manifest: Value = agent()
        .get(&manifest_url)
        .set("User-Agent", "rocm-cli")
        .set("Authorization", &format!("Bearer {token}"))
        .set(
            "Accept",
            "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
        )
        .call()
        .with_context(|| format!("failed to fetch manifest for {image_ref}"))?
        .into_json()
        .with_context(|| format!("failed to parse manifest for {image_ref}"))?;

    let config_digest = manifest
        .get("config")
        .and_then(|config| config.get("digest"))
        .and_then(Value::as_str)
        .with_context(|| format!("manifest for {image_ref} had no config digest"))?;

    let blob_url = format!("https://{REGISTRY_HOST}/v2/{repo}/blobs/{config_digest}");
    let config: Value = agent()
        .get(&blob_url)
        .set("User-Agent", "rocm-cli")
        .set("Authorization", &format!("Bearer {token}"))
        .call()
        .with_context(|| format!("failed to fetch config blob for {image_ref}"))?
        .into_json()
        .with_context(|| format!("failed to parse config blob for {image_ref}"))?;

    Ok(parse_image_config(
        config.get("config").unwrap_or(&Value::Null),
    ))
}

/// Metadata from the local daemon's copy of the image, or `None` if the image
/// isn't present locally.
pub(crate) fn local_image_metadata(image_ref: &str) -> Option<AimImageMetadata> {
    let config = crate::docker::image_config(image_ref).ok().flatten()?;
    Some(parse_image_config(&config))
}

/// Parses an image `Config` object (`{"Env": [...], "Labels": {...}}`), the
/// shape shared by Docker Hub's config blob (under `config`) and
/// `docker image inspect`'s `.Config`.
fn parse_image_config(config: &Value) -> AimImageMetadata {
    let mut metadata = AimImageMetadata::default();

    let env_entries = config
        .get("Env")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    for entry in env_entries {
        if let Some((key, value)) = entry.split_once('=') {
            match key {
                "AIM_ACCELERATOR_TYPE" => metadata.accelerator_type = Some(value.to_owned()),
                "PYTORCH_ROCM_ARCH" => {
                    metadata.pytorch_rocm_arch = value.split(';').map(ToOwned::to_owned).collect();
                }
                _ => {}
            }
        }
    }

    if let Some(labels) = config.get("Labels").and_then(Value::as_object) {
        metadata.canonical_name = labels
            .get("com.amd.aim.model.canonicalName")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_image_ref_defaults_org_for_bare_name() {
        assert_eq!(
            normalize_image_ref("aim-meta-llama-llama-3-2-1b-instruct:0.11.1"),
            "amdenterpriseai/aim-meta-llama-llama-3-2-1b-instruct:0.11.1"
        );
    }

    #[test]
    fn normalize_image_ref_preserves_explicit_org_or_registry() {
        assert_eq!(
            normalize_image_ref("amdenterpriseai/aim-foo:latest"),
            "amdenterpriseai/aim-foo:latest"
        );
        assert_eq!(
            normalize_image_ref("myregistry.example.com:5000/aim-foo:latest"),
            "myregistry.example.com:5000/aim-foo:latest"
        );
    }

    #[test]
    fn split_image_ref_preserves_explicit_org() {
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo:latest"),
            ("amdenterpriseai/aim-foo".to_owned(), "latest".to_owned())
        );
    }

    #[test]
    fn split_image_ref_defaults_tag_to_latest() {
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo"),
            ("amdenterpriseai/aim-foo".to_owned(), "latest".to_owned())
        );
    }

    #[test]
    fn split_image_ref_handles_digest_refs() {
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo@sha256:abcd1234"),
            (
                "amdenterpriseai/aim-foo".to_owned(),
                "sha256:abcd1234".to_owned()
            )
        );
    }

    #[test]
    fn split_image_ref_strips_the_tag_when_a_digest_is_present() {
        // `repo:tag@sha256:...` must not leave `:tag` in the repo name, or the
        // token scope and manifest URL point at a repository that doesn't exist.
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo:0.12.0-preview@sha256:abcd1234"),
            (
                "amdenterpriseai/aim-foo".to_owned(),
                "sha256:abcd1234".to_owned()
            )
        );
    }

    #[test]
    fn is_docker_hub_ref_rejects_other_registries() {
        assert!(is_docker_hub_ref("amdenterpriseai/aim-foo:1"));
        assert!(!is_docker_hub_ref("myregistry.example.com:5000/aim-foo:1"));
        assert!(!is_docker_hub_ref("ghcr.io/org/aim-foo:1"));
        assert!(!is_docker_hub_ref("localhost/aim-foo"));
    }

    #[test]
    fn fetch_image_metadata_skips_non_docker_hub_registries_without_network() {
        let error = fetch_image_metadata("myregistry.example.com:5000/aim-foo:1")
            .expect_err("a private registry is not looked up");
        assert!(
            error.to_string().contains("not a Docker Hub image"),
            "{error}"
        );
    }

    #[test]
    fn split_image_ref_handles_registry_host_with_port() {
        // The first `:` here belongs to the registry host's port, not a tag
        // separator — a naive `split_once(':')` would mis-parse this as
        // repo="myregistry.example.com", tag="5000/aim-foo:v1".
        assert_eq!(
            split_image_ref("myregistry.example.com:5000/aim-foo:v1"),
            (
                "myregistry.example.com:5000/aim-foo".to_owned(),
                "v1".to_owned()
            )
        );
    }

    #[test]
    fn parse_image_config_extracts_known_fields() {
        let config = serde_json::json!({
            "Env": [
                "PATH=/usr/bin",
                "AIM_ACCELERATOR_TYPE=gpu",
                "PYTORCH_ROCM_ARCH=gfx1100;gfx1151",
            ],
            "Labels": {
                "com.amd.aim.model.canonicalName": "Llama-3.2-1B-Instruct",
            },
        });
        let metadata = parse_image_config(&config);
        assert_eq!(metadata.accelerator_type.as_deref(), Some("gpu"));
        assert_eq!(metadata.pytorch_rocm_arch, vec!["gfx1100", "gfx1151"]);
        assert_eq!(
            metadata.canonical_name.as_deref(),
            Some("Llama-3.2-1B-Instruct")
        );
    }
}
