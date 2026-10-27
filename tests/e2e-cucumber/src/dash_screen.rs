// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure matching over captured dashboard screens and config files.
//!
//! The cucumber binary has no unit-test harness, so the logic that the
//! dashboard scenarios depend on lives here, where a failing case can be shown.

use serde_json::{Map, Value};

/// Fewest cells after the model cell of an Observe instances row.
///
/// The cells are `TOK/S TOK/W TTFT TPOT POWER QUEUE KV%` (`instances.rs`). A
/// cell with a unit (`12.00 tok/W`) splits into more than one, so a live row
/// can have more.
pub const MIN_OBSERVE_METRIC_CELLS: usize = 7;

/// The cells after the model cell of an Observe instances row whose model
/// starts with `model_prefix`. Free text that merely mentions the model, or a
/// row that was cut short, does not match.
///
/// The row carries no status or port. Use [`instance_detail_matches`] for those.
pub fn observe_row_cells<'a>(line: &'a str, model_prefix: &str) -> Option<Vec<&'a str>> {
    let mut cells = line.split_whitespace();
    cells.find(|cell| cell.starts_with(model_prefix))?;
    let rest: Vec<&str> = cells.collect();
    (rest.len() >= MIN_OBSERVE_METRIC_CELLS).then_some(rest)
}

/// The port of an endpoint URL such as `http://127.0.0.1:11435/v1`.
pub fn endpoint_port(endpoint: &str) -> Option<String> {
    let authority = endpoint.split("://").nth(1)?.split('/').next()?;
    let port = authority.rsplit_once(':')?.1;
    (!port.is_empty() && port.chars().all(|c| c.is_ascii_digit())).then(|| port.to_string())
}

/// True when the screen holds a matching instance-detail summary.
///
/// The status badge must be `status` with `port: <port>` on the same line, and
/// a `model:` cell must be followed by a name starting with `model_prefix`
/// (`render_summary` in `instances.rs`).
pub fn instance_detail_matches(screen: &str, model_prefix: &str, status: &str, port: &str) -> bool {
    // Panel borders can touch a label (`│model:`), so treat them as separators.
    let screen = strip_box_drawing(screen);
    let screen = screen.as_str();
    let status_and_port = screen.lines().any(|line| {
        let cells: Vec<&str> = line.split_whitespace().collect();
        cells.contains(&status)
            && cells
                .windows(2)
                .any(|pair| pair[0] == "port:" && pair[1] == port)
    });
    let model = screen.lines().any(|line| {
        let mut cells = line.split_whitespace();
        cells.any(|cell| cell == "model:")
            && cells
                .next()
                .is_some_and(|name| name.starts_with(model_prefix))
    });
    status_and_port && model
}

/// Replace the box-drawing characters of the terminal UI with spaces.
fn strip_box_drawing(screen: &str) -> String {
    screen
        .chars()
        .map(|c| {
            if ('\u{2500}'..='\u{257f}').contains(&c) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// Set only the `dashboard.daemon.*` keys of `update` on `existing`, leaving
/// every other key (an active runtime, for example) as it was.
///
/// # Panics
/// Panics when `update` has no `dashboard.daemon` object; that is a bug in the
/// calling step, not a runtime condition.
pub fn merge_dashboard_daemon_config(existing: Option<Value>, update: &Value) -> Value {
    let mut root = match existing {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    let updates = update
        .pointer("/dashboard/daemon")
        .and_then(Value::as_object)
        .expect("update must carry dashboard.daemon keys");
    let dashboard = object_entry(&mut root, "dashboard");
    let daemon = object_entry(dashboard, "daemon");
    for (key, value) in updates {
        daemon.insert(key.clone(), value.clone());
    }
    Value::Object(root)
}

/// Turn Lemonade discovery off unless the config already says otherwise.
///
/// The dashboard daemon probes `127.0.0.1:13305` by default. A scenario that
/// does not exercise discovery must not pick up a real Lemonade server that
/// happens to listen there, so the isolated config starts with discovery off.
/// A scenario that wants discovery writes `enable_lemonade` itself, and that
/// value is kept.
pub fn with_lemonade_discovery_default_off(existing: Option<Value>) -> Value {
    if existing
        .as_ref()
        .and_then(|config| config.pointer("/dashboard/daemon/enable_lemonade"))
        .is_some()
    {
        return existing.expect("checked above");
    }
    merge_dashboard_daemon_config(
        existing,
        &serde_json::json!({ "dashboard": { "daemon": { "enable_lemonade": false } } }),
    )
}

/// The object stored under `key`, replacing a missing or non-object value.
fn object_entry<'a>(parent: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    let slot = parent
        .entry(key.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !slot.is_object() {
        *slot = Value::Object(Map::new());
    }
    slot.as_object_mut().expect("slot is an object")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MANAGED_ROW: &str =
        "Qwen3-0.6B-GGUF         —        —        —        —        —      —       —";
    const DETAIL: &str = "│ READY   id: svc-1  port: 11435  tp: 1                │\n\
                          │ model: Qwen3-0.6B-GGUF  gpus: 0  tok/W: —            │";

    #[test]
    fn observe_row_matches_a_full_managed_row() {
        let cells = observe_row_cells(MANAGED_ROW, "Qwen3-0.6B").expect("row matches");
        assert_eq!(cells.len(), MIN_OBSERVE_METRIC_CELLS);
        let with_units = "Qwen3-0.6B-GGUF 41.2 tok/s 0.52 tok/W 50ms 12ms 80W 0/0 3%";
        assert!(observe_row_cells(with_units, "Qwen3-0.6B").is_some());
    }

    #[test]
    fn observe_row_rejects_other_model_truncated_row_and_prose() {
        let other = MANAGED_ROW.replace("Qwen3-0.6B-GGUF", "Phi-3-mini");
        assert!(observe_row_cells(&other, "Qwen3-0.6B").is_none());
        let truncated = "Qwen3-0.6B-GGUF         —        —        —";
        assert!(observe_row_cells(truncated, "Qwen3-0.6B").is_none());
        let prose = "loaded Qwen3-0.6B-GGUF on the default port";
        assert!(observe_row_cells(prose, "Qwen3-0.6B").is_none());
    }

    #[test]
    fn endpoint_port_reads_the_serving_port() {
        assert_eq!(
            endpoint_port("http://127.0.0.1:11435/v1").as_deref(),
            Some("11435")
        );
        assert_eq!(endpoint_port("http://localhost/v1"), None);
        assert_eq!(endpoint_port("not a url"), None);
    }

    #[test]
    fn detail_matches_ready_managed_model_on_the_served_port() {
        assert!(instance_detail_matches(
            DETAIL,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
    }

    /// Captured from a real run: the panel border touches the `model:` label.
    const REAL_DETAIL: &str = "│ ╭─╮Node╭─╮Instance · lemonade-qwen3-0-6b-gguf-1792015518018╭───╮──────╮ │\n\
        │ │      │ READY   id: lemonade-qwen3-0-6b-gguf-1792015518018  port: 11435  tp: 0     │      │ │\n\
        │ │  —   │model: Qwen3-0.6B-GGUF  gpus: -  tok/W: -  gen: -                           │      │ │\n\
        │ │      │partition: -  quantization: -  vram: 0 / 0 MiB                              │      │ │";

    #[test]
    fn detail_matches_the_screen_a_real_run_rendered() {
        assert!(instance_detail_matches(
            REAL_DETAIL,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
        assert!(!instance_detail_matches(
            REAL_DETAIL,
            "Qwen3-0.6B",
            "READY",
            "11436"
        ));
        assert!(!instance_detail_matches(
            &REAL_DETAIL.replace("READY", "STARTING"),
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
    }

    #[test]
    fn detail_rejects_wrong_status_port_or_model() {
        let starting = DETAIL.replace("READY", "STARTING");
        assert!(!instance_detail_matches(
            &starting,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
        let stopped = DETAIL.replace("READY", "STOPPED");
        assert!(!instance_detail_matches(
            &stopped,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
        let other_port = DETAIL.replace("11435", "8000");
        assert!(!instance_detail_matches(
            &other_port,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
        let other_model = DETAIL.replace("Qwen3-0.6B-GGUF", "Phi-3-mini");
        assert!(!instance_detail_matches(
            &other_model,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
        // A status word elsewhere on screen does not stand in for the badge on
        // the port line.
        let elsewhere = "READY\n│ id: svc-1  port: 11435  tp: 1 │\n│ model: Qwen3-0.6B-GGUF │";
        assert!(!instance_detail_matches(
            elsewhere,
            "Qwen3-0.6B",
            "READY",
            "11435"
        ));
    }

    #[test]
    fn dashboard_config_merge_keeps_other_keys() {
        let existing = json!({
            "runtime": { "active": "therock-7.1" },
            "dashboard": { "theme": "dark", "daemon": { "lemonade_port": 1 } }
        });
        let update = json!({
            "dashboard": { "daemon": { "enable_lemonade": false } }
        });
        let merged = merge_dashboard_daemon_config(Some(existing), &update);
        assert_eq!(merged["runtime"]["active"], "therock-7.1");
        assert_eq!(merged["dashboard"]["theme"], "dark");
        assert_eq!(merged["dashboard"]["daemon"]["lemonade_port"], 1);
        assert_eq!(merged["dashboard"]["daemon"]["enable_lemonade"], false);
    }

    #[test]
    fn lemonade_discovery_defaults_off_and_keeps_other_keys() {
        let existing = json!({ "runtime": { "active": "therock-7.1" } });
        for start in [None, Some(existing.clone()), Some(json!("garbage"))] {
            let merged = with_lemonade_discovery_default_off(start);
            assert_eq!(merged["dashboard"]["daemon"]["enable_lemonade"], false);
        }
        let merged = with_lemonade_discovery_default_off(Some(existing));
        assert_eq!(merged["runtime"]["active"], "therock-7.1");
    }

    #[test]
    fn lemonade_discovery_default_keeps_an_explicit_choice() {
        for enabled in [true, false] {
            let existing = json!({
                "dashboard": { "daemon": { "enable_lemonade": enabled, "lemonade_port": 9 } }
            });
            let merged = with_lemonade_discovery_default_off(Some(existing));
            assert_eq!(merged["dashboard"]["daemon"]["enable_lemonade"], enabled);
            assert_eq!(merged["dashboard"]["daemon"]["lemonade_port"], 9);
        }
    }

    #[test]
    fn dashboard_config_merge_starts_from_nothing_or_garbage() {
        let update = json!({ "dashboard": { "daemon": { "lemonade_port": 7 } } });
        for existing in [
            None,
            Some(json!("not an object")),
            Some(json!({"dashboard": 3})),
        ] {
            let merged = merge_dashboard_daemon_config(existing, &update);
            assert_eq!(merged["dashboard"]["daemon"]["lemonade_port"], 7);
        }
    }
}
