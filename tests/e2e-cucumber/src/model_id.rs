// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/// Whether a response's model id identifies the model requested by the scenario.
///
/// Lemonade may report only a concrete GGUF artifact filename, while vLLM echoes
/// a fully qualified model id. Quantization and GGUF suffixes are normalized, but
/// organizations must match when both ids include one.
pub fn model_ids_match(response: &str, expected: &str) -> bool {
    let (response_org, response_base) = normalize_model_id(response);
    let (expected_org, expected_base) = normalize_model_id(expected);
    if response_base.is_empty() || expected_base.is_empty() {
        return false;
    }
    if let (Some(response_org), Some(expected_org)) = (response_org, expected_org)
        && response_org != expected_org
    {
        return false;
    }
    response_base.contains(&expected_base) || expected_base.contains(&response_base)
}

fn normalize_model_id(id: &str) -> (Option<String>, String) {
    let without_variant = id.split_once(':').map_or(id, |(model, _)| model);
    let (org, model) = without_variant
        .rsplit_once('/')
        .map_or((None, without_variant), |(org, model)| (Some(org), model));
    let mut base = model.to_ascii_lowercase();
    if let Some(stripped) = base.strip_suffix(".gguf") {
        base = stripped.to_owned();
    }
    base = base.replace("-gguf", "");
    for quant in [
        "-ud-q4_k_xl",
        "-q4_0",
        "-q4_k_m",
        "-q4_k_s",
        "-q5_k_m",
        "-q8_0",
        "-f16",
        "-fp16",
    ] {
        base = base.replace(quant, "");
    }
    (
        org.map(str::to_ascii_lowercase),
        base.trim_matches(['-', '_', '.']).to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::model_ids_match;

    #[test]
    fn matches_lemonade_artifact_without_organization() {
        assert!(model_ids_match(
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"
        ));
    }

    #[test]
    fn matches_same_organization_and_normalized_model() {
        assert!(model_ids_match(
            "unsloth/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "UNSLOTH/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
        ));
    }

    #[test]
    fn rejects_same_model_from_different_organization() {
        assert!(!model_ids_match(
            "another-owner/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
        ));
    }
}
