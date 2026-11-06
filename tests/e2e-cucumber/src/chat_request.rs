// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The suite's chat round trip against a served endpoint: discover the model
//! over `/models`, then POST one chat completion to it.
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

/// Discover the served model id over `{endpoint}/models` and POST one chat
/// completion to it, returning the decoded response. `tools` is merged into the
/// request body for the tool-definitions scenario.
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
/// If every attempt fails in transport, a reply is not JSON, or `/models` names
/// no model.
pub async fn request_chat_completion(
    endpoint: &str,
    prompt: &str,
    tools: Option<serde_json::Value>,
    attempt_budgets: &[Duration],
) -> serde_json::Value {
    let exchange = tokio::spawn(exchange(
        endpoint.to_owned(),
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
        let model = models["data"][0]["id"]
            .as_str()
            .unwrap_or_else(|| panic!("no model id in response: {models}"))
            .to_string();

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

    use super::request_chat_completion;
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
            request_chat_completion(&endpoint, "Hello", None, &budgets),
            blocking_step,
        );

        assert!(chat_response_is_successful(&response), "{response}");
    }
}
