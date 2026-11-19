// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The suite's chat round trip against a served endpoint: discover the
//! scenario's model over `/models`, then POST one chat completion to it.
//!
//! The round trip runs as its own task on the runtime's worker threads, not on
//! the thread that polls the scenarios. cucumber polls every in-flight scenario
//! from a single task, and most steps run the `rocm` binary synchronously, so
//! that thread is routinely blocked for many seconds while some other scenario's
//! command runs. A request polled from there has its timeout clock running the
//! whole time and nobody to read the reply, so a server that answered in
//! milliseconds would be reported as a client timeout. On its own task the
//! budget times the server, which is what it is for.

use std::time::Duration;

use crate::model_id::model_ids_match;

/// Discover the served model id over `{endpoint}/models` and POST one chat
/// completion to it, returning the decoded response. `tools` is merged into the
/// request body for the tool-definitions scenario.
///
/// `expected_model` is the model the scenario served. The id sent is the entry
/// of `/models` that identifies it (see [`pick_model`]), because a router such
/// as Lemonade's lists every model it has loaded, in no order the scenario
/// controls, and the first entry may belong to someone else. `None` keeps the
/// old behaviour of taking the first entry, for a scenario that names no model.
///
/// Makes one attempt per entry of `attempt_budgets`, each bounded by its entry,
/// and moves to the next only on a TRANSPORT failure of either request (the
/// `send`, not the decode). A malformed reply is a server-contract violation
/// rather than a flake, so it panics on the spot. Panics with every attempt's
/// diagnostics once the attempts run out.
///
/// The exchange runs on the runtime's worker threads (see the module docs), so
/// a step blocking the thread that awaits this does not count against the
/// budgets.
///
/// # Panics
///
/// If every attempt fails in transport, a reply is not JSON, `/models` names no
/// model, or none of the models it names is `expected_model`.
pub async fn request_chat_completion(
    endpoint: &str,
    expected_model: Option<&str>,
    prompt: &str,
    tools: Option<serde_json::Value>,
    attempt_budgets: &[Duration],
) -> serde_json::Value {
    let exchange = tokio::spawn(exchange(
        endpoint.to_owned(),
        expected_model.map(str::to_owned),
        prompt.to_owned(),
        tools,
        attempt_budgets.to_vec(),
    ));
    match exchange.await {
        Ok(response) => response,
        // Re-raise the exchange's own panic so the step fails with its
        // diagnostics rather than a bare "task panicked".
        Err(error) => match error.try_into_panic() {
            Ok(payload) => std::panic::resume_unwind(payload),
            Err(error) => panic!("the chat request task did not finish: {error}"),
        },
    }
}

async fn exchange(
    endpoint: String,
    expected_model: Option<String>,
    prompt: String,
    tools: Option<serde_json::Value>,
    attempt_budgets: Vec<Duration>,
) -> serde_json::Value {
    let attempts = attempt_budgets.len();
    let models_url = format!("{endpoint}/models");
    let chat_url = format!("{endpoint}/chat/completions");
    let mut diagnostics = Vec::new();

    for (attempt, budget) in (1..).zip(attempt_budgets) {
        let budget_secs = budget.as_secs_f64();
        // A fresh client per attempt: the pooled connection of a timed-out
        // attempt is not worth reusing.
        let client = reqwest::Client::builder()
            .timeout(budget)
            .build()
            .expect("failed to build HTTP client");

        let started = std::time::Instant::now();
        let models = match client.get(&models_url).send().await {
            Ok(response) => response,
            // Fall through to the next attempt rather than panicking here: a
            // panic would also throw away the diagnostics of the attempts
            // already recorded, which is the whole point of collecting them.
            Err(error) => {
                diagnostics.push(format!(
                    "attempt {attempt} (budget {budget_secs}s): GET {models_url} failed after \
                     {:.1}s: {}",
                    started.elapsed().as_secs_f64(),
                    describe_request_error(&error)
                ));
                continue;
            }
        };
        let models: serde_json::Value = models
            .json()
            .await
            .unwrap_or_else(|e| panic!("GET {models_url} returned non-JSON: {e}"));
        let model = pick_model(&models, expected_model.as_deref())
            .unwrap_or_else(|reason| panic!("GET {models_url}: {reason}"));

        let mut body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        });
        if let Some(tools) = tools.clone() {
            body["tools"] = tools;
        }

        // Time the POST on its own: the discovery round trip ahead of it is
        // cheap and constant, and mixing it in would blur the number that
        // actually gets compared against the timeout.
        let started = std::time::Instant::now();
        match client.post(&chat_url).json(&body).send().await {
            Ok(response) => {
                // Say so when an earlier attempt failed. A rescued flake is
                // otherwise invisible — the scenario just goes green — and then
                // nobody can tell from the logs whether cold starts are getting
                // slower until the retry stops being enough.
                if !diagnostics.is_empty() {
                    eprintln!(
                        "chat request succeeded on attempt {attempt} of {attempts} after an \
                         earlier failure:\n{}",
                        diagnostics.join("\n")
                    );
                }
                return response
                    .json()
                    .await
                    .unwrap_or_else(|e| panic!("POST {chat_url} returned non-JSON: {e}"));
            }
            Err(error) => diagnostics.push(format!(
                "attempt {attempt} (budget {budget_secs}s): POST {chat_url} failed after \
                 {:.1}s: {}",
                started.elapsed().as_secs_f64(),
                describe_request_error(&error)
            )),
        }
    }

    panic!(
        "chat request failed after {attempts} attempt(s):\n{}",
        diagnostics.join("\n")
    );
}

/// The id in a `/models` reply to send the chat to.
///
/// With an `expected` model, the entry that IS it: an id equal to it, else the
/// first one [`model_ids_match`] accepts (Lemonade lists catalog names, vLLM the
/// repo id, and the two spell one model differently). Taking `data[0]` instead
/// sent serve-57's chat to whichever model its router listed first, and the
/// reply then named that model rather than the scenario's. With no `expected`
/// model, the first entry.
///
/// # Errors
///
/// The reason, with every id the server listed, when the reply names no model
/// or none of them is `expected`.
fn pick_model(models: &serde_json::Value, expected: Option<&str>) -> Result<String, String> {
    let ids: Vec<&str> = models["data"]
        .as_array()
        .map(|data| data.iter().filter_map(|m| m["id"].as_str()).collect())
        .unwrap_or_default();
    let Some(first) = ids.first() else {
        return Err(format!("no model id in response: {models}"));
    };
    let Some(expected) = expected else {
        return Ok((*first).to_owned());
    };
    ids.iter()
        .find(|id| **id == expected)
        .or_else(|| ids.iter().find(|id| model_ids_match(id, expected)))
        .map(|id| (*id).to_owned())
        .ok_or_else(|| {
            format!(
                "the server does not list the scenario's model '{expected}'; it lists {} \
                 model(s): {}",
                ids.len(),
                ids.join(", ")
            )
        })
}

/// A transport failure spelled out well enough to triage from a CI log alone.
///
/// `reqwest::Error`'s `Display` prints only its KIND — a client timeout and a
/// refused/reset connection both read as "error sending request for url (…)",
/// with the real cause reachable only through `source()`. That ambiguity cost a
/// full log archaeology once (a 10s client timeout that read as a dead server),
/// so classify the error and unwind the source chain into the panic message.
fn describe_request_error(error: &reqwest::Error) -> String {
    use std::{error::Error as _, fmt::Write as _};

    let kind = if error.is_timeout() {
        " [client timeout — the harness gave up, the server may still be working]"
    } else if error.is_connect() {
        " [connect failure — nothing accepted the connection]"
    } else {
        ""
    };
    let mut detail = format!("{error}{kind}");
    let mut source = error.source();
    while let Some(cause) = source {
        // Infallible: writing into a String never errors.
        let _ = write!(detail, "\n  caused by: {cause}");
        source = cause.source();
    }
    detail
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{pick_model, request_chat_completion};
    use crate::chat_response_is_successful;
    use crate::mock_server::MockServer;

    /// chat-06's failure (issue #661): the request and a step that blocks its
    /// thread are polled from one task, as cucumber polls every scenario, and
    /// the block outlasts the request's budget. The server answers at once, so
    /// the request must succeed. Polled from that task itself, it timed out.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_blocked_scenario_thread_does_not_time_out_a_request_the_server_answers() {
        let mock = MockServer::start("TestModel/E2E-1B").await;
        let budget = Duration::from_millis(500);
        let blocking_step = async {
            // What a step running `rocm` synchronously does to every other
            // scenario polled from the same task.
            std::thread::sleep(budget * 4);
        };
        let endpoint = mock.base_url();
        let budgets = [budget];

        let (response, ()) = tokio::join!(
            request_chat_completion(&endpoint, Some("TestModel/E2E-1B"), "Hello", None, &budgets),
            blocking_step,
        );

        assert!(chat_response_is_successful(&response), "{response}");
    }

    fn listing(ids: &[&str]) -> serde_json::Value {
        let data: Vec<_> = ids
            .iter()
            .map(|id| serde_json::json!({"id": id, "object": "model"}))
            .collect();
        serde_json::json!({"object": "list", "data": data})
    }

    /// serve-57: the router lists another model first. The chat goes to the
    /// scenario's model, not to `data[0]`.
    #[test]
    fn the_scenarios_model_is_picked_even_when_the_router_lists_another_first() {
        let models = listing(&["Qwen3-0.6B-GGUF", "Qwen3.5-0.8B-GGUF"]);

        assert_eq!(
            pick_model(&models, Some("Qwen3.5-0.8B-GGUF")).as_deref(),
            Ok("Qwen3.5-0.8B-GGUF")
        );
    }

    /// vLLM lists the repo id while the scenario may carry a catalog spelling:
    /// the normalised match still finds it, and an exact id wins over it.
    #[test]
    fn a_differently_spelled_id_of_the_same_model_is_picked_and_an_exact_id_wins() {
        let spelled = listing(&["other/Model-A", "unsloth/Qwen3-0.6B-Q4_0.gguf"]);
        assert_eq!(
            pick_model(&spelled, Some("Qwen3-0.6B-GGUF")).as_deref(),
            Ok("unsloth/Qwen3-0.6B-Q4_0.gguf")
        );

        let both = listing(&["Qwen3-0.6B-Q4_0.gguf", "Qwen3-0.6B-GGUF"]);
        assert_eq!(
            pick_model(&both, Some("Qwen3-0.6B-GGUF")).as_deref(),
            Ok("Qwen3-0.6B-GGUF")
        );
    }

    /// A server that does not list the model fails the request with every id
    /// it did list, rather than quietly chatting with a different model.
    #[test]
    fn a_missing_model_names_every_model_the_server_listed() {
        let models = listing(&["Qwen3-0.6B-GGUF", "Llama-3.2-1B-GGUF"]);

        let reason = pick_model(&models, Some("Qwen3.5-0.8B-GGUF")).expect_err("must not match");

        assert!(reason.contains("Qwen3.5-0.8B-GGUF"), "{reason}");
        assert!(reason.contains("2 model(s)"), "{reason}");
        assert!(
            reason.contains("Qwen3-0.6B-GGUF, Llama-3.2-1B-GGUF"),
            "{reason}"
        );
    }

    /// A scenario that names no model keeps taking the first entry, and an
    /// empty listing is still an error.
    #[test]
    fn without_an_expected_model_the_first_entry_is_used() {
        let models = listing(&["first", "second"]);
        assert_eq!(pick_model(&models, None).as_deref(), Ok("first"));
        assert!(pick_model(&listing(&[]), None).is_err());
    }
}
