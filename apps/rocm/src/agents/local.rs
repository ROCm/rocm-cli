// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Use a model served by `rocm serve --managed` as an agent's backend
//! (PROTOTYPE).
//!
//! A managed service already publishes an Anthropic-compatible `/v1/messages`
//! endpoint (a translation front over the engine's OpenAI API), so Claude Code
//! needs no new protocol: the credential proxy simply forwards to that
//! endpoint instead of a cloud provider. No cloud key, gateway header or host
//! agent setting is involved in this mode.

use anyhow::{Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord};

/// The service an agent run is pointed at.
pub(crate) struct LocalTarget {
    pub service_id: String,
    /// Base URL of the service's Anthropic-compatible endpoint.
    pub anthropic_url: String,
    /// The name the engine serves the model under; every request is answered by
    /// it whatever model name the agent sends.
    pub model: String,
    /// The service's endpoint key, when it requires one. The proxy sends it as
    /// `x-api-key`; the agent never sees it.
    pub endpoint_key: Option<String>,
}

pub(crate) fn resolve(paths: &AppPaths, wanted: Option<&str>) -> Result<LocalTarget> {
    let records = crate::managed_service_status::load_managed_services(paths)?;
    let (record, anthropic_url) = pick(&records, wanted)?;
    Ok(LocalTarget {
        service_id: record.service_id.clone(),
        anthropic_url,
        model: record.served_model_name().to_owned(),
        endpoint_key: crate::endpoint_keys::endpoint_api_key(paths, &record.service_id),
    })
}

/// The ready service to use and its Anthropic endpoint URL.
///
/// With no `wanted` id the choice is made only when it is unambiguous: a
/// machine can host several servers, and silently picking one would send the
/// agent's prompts to a model the user did not mean.
pub(crate) fn pick<'a>(
    records: &'a [ManagedServiceRecord],
    wanted: Option<&str>,
) -> Result<(&'a ManagedServiceRecord, String)> {
    let usable = |record: &ManagedServiceRecord| {
        (record.status == "ready")
            .then(|| record.anthropic_endpoint_url.clone())
            .flatten()
    };
    if let Some(id) = wanted {
        let Some(record) = records.iter().find(|r| r.service_id == id) else {
            bail!("no local server `{id}`; `rocm services list` shows the ones that exist");
        };
        return match usable(record) {
            Some(url) => Ok((record, url)),
            None if record.status != "ready" => bail!(
                "local server `{id}` is {}, not ready; `rocm services list` shows its state",
                record.status
            ),
            None => bail!(
                "local server `{id}` has no Anthropic-compatible endpoint yet; \
                 `rocm services list` shows it once it is up"
            ),
        };
    }
    let mut candidates: Vec<_> = records
        .iter()
        .filter_map(|r| usable(r).map(|url| (r, url)))
        .collect();
    match candidates.len() {
        0 => bail!(
            "no ready local model server. Start one with `rocm serve <model> --managed`, \
             then check `rocm services list`"
        ),
        1 => Ok(candidates.remove(0)),
        _ => {
            let ids: Vec<_> = candidates
                .iter()
                .map(|(r, _)| r.service_id.as_str())
                .collect();
            bail!(
                "several local servers are ready ({}); choose one with --service <id>",
                ids.join(", ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, status: &str, anthropic: Option<&str>) -> ManagedServiceRecord {
        let paths = AppPaths {
            config_dir: "/nonexistent/config".into(),
            data_dir: "/nonexistent/data".into(),
            cache_dir: "/nonexistent/cache".into(),
        };
        let mut record = ManagedServiceRecord::new(
            &paths,
            id,
            "lemonade",
            "org/model",
            "org/model",
            "127.0.0.1",
            9000,
            "managed",
            1,
            None,
            None,
            None,
        );
        status.clone_into(&mut record.status);
        record.anthropic_endpoint_url = anthropic.map(str::to_owned);
        record
    }

    #[test]
    fn the_only_ready_server_is_chosen_without_naming_it() {
        let records = [
            record("a", "stopped", Some("http://127.0.0.1:1")),
            record("b", "ready", Some("http://127.0.0.1:2")),
        ];
        let (picked, url) = pick(&records, None).unwrap();
        assert_eq!(
            (picked.service_id.as_str(), url.as_str()),
            ("b", "http://127.0.0.1:2")
        );
    }

    #[test]
    fn several_ready_servers_are_never_guessed_between() {
        let records = [
            record("a", "ready", Some("http://127.0.0.1:1")),
            record("b", "ready", Some("http://127.0.0.1:2")),
        ];
        let error = pick(&records, None).unwrap_err().to_string();
        assert!(
            error.contains("a, b") && error.contains("--service <id>"),
            "{error}"
        );
        assert_eq!(pick(&records, Some("b")).unwrap().0.service_id, "b");
    }

    #[test]
    fn no_ready_server_says_how_to_start_one() {
        let error = pick(&[], None).unwrap_err().to_string();
        assert!(error.contains("rocm serve <model> --managed"), "{error}");
        let starting = [record("a", "starting", Some("http://127.0.0.1:1"))];
        assert!(pick(&starting, None).is_err());
    }

    #[test]
    fn a_named_server_that_is_not_usable_says_why() {
        let records = [
            record("up", "ready", None),
            record("loading", "starting", Some("http://127.0.0.1:1")),
        ];
        let loading = pick(&records, Some("loading")).unwrap_err().to_string();
        assert!(loading.contains("starting, not ready"), "{loading}");
        let no_endpoint = pick(&records, Some("up")).unwrap_err().to_string();
        assert!(
            no_endpoint.contains("no Anthropic-compatible endpoint"),
            "{no_endpoint}"
        );
        let missing = pick(&records, Some("nope")).unwrap_err().to_string();
        assert!(missing.contains("no local server `nope`"), "{missing}");
    }
}
