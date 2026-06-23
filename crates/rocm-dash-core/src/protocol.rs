// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! NDJSON command/event protocol between rocmdashd and rocmdash.
//! See `../wiki/comparisons/ctux-vs-rocm-dash.md` (resolved decisions).

use serde::{Deserialize, Serialize};

use crate::bench_schema::BenchmarkRow;
use crate::metrics::{Instance, Snapshot};

pub const PROTOCOL_VERSION: u32 = 1;

/// Sent by the TUI to the daemon. One per line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Command {
    Hello {
        protocol_version: u32,
        client: String,
        token: Option<String>,
    },
    Subscribe,
    RequestSnapshot,
    RescanInstances,
    Pause,
    Resume,
    Goodbye,
}

/// Sent by the daemon to the TUI. One per line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Welcome {
        protocol_version: u32,
        daemon_version: String,
        host: String,
    },
    Snapshot(Snapshot),
    InstanceDiscovered(Instance),
    InstanceGone {
        container_id: String,
    },
    BenchmarkRowsAppended {
        rows: Vec<BenchmarkRow>,
    },
    Warning {
        message: String,
    },
    Error {
        message: String,
    },
    Bye,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_hello_roundtrips() {
        let cmd = Command::Hello {
            protocol_version: PROTOCOL_VERSION,
            client: "rocmdash/0.1.0".into(),
            token: None,
        };
        let s = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&s).unwrap();
        match back {
            Command::Hello {
                protocol_version, ..
            } => assert_eq!(protocol_version, PROTOCOL_VERSION),
            _ => panic!("unexpected"),
        }
    }

    #[test]
    fn event_welcome_roundtrips() {
        let ev = Event::Welcome {
            protocol_version: PROTOCOL_VERSION,
            daemon_version: "0.1.0".into(),
            host: "gpu-host-01".into(),
        };
        let s = serde_json::to_string(&ev).unwrap();
        let back: Event = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, Event::Welcome { .. }));
    }

    /// Regression guard: serde's internally-tagged enums reject newtype
    /// variants containing sequences. Every `Event` variant must be either
    /// a struct variant or wrap a struct — never `Vec<T>` directly.
    #[test]
    fn every_event_variant_round_trips_through_json() {
        use crate::bench_schema::BenchmarkRow;
        use crate::metrics::{Instance, Snapshot};

        let variants = vec![
            Event::Welcome {
                protocol_version: 1,
                daemon_version: "v".into(),
                host: "h".into(),
            },
            Event::Snapshot(Snapshot::default()),
            Event::InstanceDiscovered(Instance::default()),
            Event::InstanceGone {
                container_id: "c1".into(),
            },
            Event::BenchmarkRowsAppended {
                rows: vec![BenchmarkRow::default()],
            },
            Event::Warning {
                message: "w".into(),
            },
            Event::Error {
                message: "e".into(),
            },
            Event::Bye,
        ];
        for ev in variants {
            let s = serde_json::to_string(&ev).unwrap_or_else(|e| panic!("serialize {ev:?}: {e}"));
            let back: Event =
                serde_json::from_str(&s).unwrap_or_else(|e| panic!("deserialize {s}: {e}"));
            assert_eq!(
                std::mem::discriminant(&ev),
                std::mem::discriminant(&back),
                "variant changed across round-trip: {s}"
            );
        }
    }
}
