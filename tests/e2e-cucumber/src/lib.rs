// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

pub mod capability;
pub mod expectation;
pub mod mock_server;
pub mod model_id;

pub fn chat_response_is_successful(response: &serde_json::Value) -> bool {
    response
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
}

// The report generator lives in its own lean crate (only maud + serde) so xtask
// can reuse it without pulling this harness's heavy tree. Re-export it under the
// original path so `e2e_cucumber::report::{generate, evaluate_xfail}` call sites
// keep working.
pub use e2e_report as report;

#[cfg(test)]
mod tests {
    use super::chat_response_is_successful;

    #[test]
    fn chat_success_requires_non_empty_choices_array() {
        assert!(!chat_response_is_successful(&serde_json::json!({})));
        assert!(!chat_response_is_successful(
            &serde_json::json!({"choices": null})
        ));
        assert!(!chat_response_is_successful(
            &serde_json::json!({"choices": {}})
        ));
        assert!(!chat_response_is_successful(
            &serde_json::json!({"choices": []})
        ));
        assert!(chat_response_is_successful(
            &serde_json::json!({"choices": [{}]})
        ));
    }
}
