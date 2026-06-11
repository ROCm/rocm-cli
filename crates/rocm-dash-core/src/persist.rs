//! On-disk session format. Used by the daemon to write each broadcast Event
//! and by the TUI replay mode to read them back.
//!
//! One [`PersistedEntry`] per NDJSON line. The wallclock `ts_us` records when
//! the entry was written (microseconds since the UNIX epoch) so the replayer
//! can pace playback against real-time deltas — including events that don't
//! carry their own timestamp.

use serde::{Deserialize, Serialize};

use crate::protocol::Event;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEntry {
    /// Microseconds since UNIX epoch when the daemon wrote the entry.
    pub ts_us: u64,
    pub event: Event,
}

impl PersistedEntry {
    /// Stamp `event` with the current wallclock time. Falls back to 0 if the
    /// system clock predates the epoch (won't happen in practice).
    pub fn now(event: Event) -> Self {
        let ts_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        Self { ts_us, event }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Event;

    #[test]
    fn round_trips_through_json() {
        let entry = PersistedEntry {
            ts_us: 1_700_000_000_000_000,
            event: Event::Welcome {
                protocol_version: 1,
                daemon_version: "0.1.0".into(),
                host: "host".into(),
            },
        };
        let s = serde_json::to_string(&entry).unwrap();
        let back: PersistedEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back.ts_us, entry.ts_us);
        assert!(matches!(back.event, Event::Welcome { .. }));
    }

    #[test]
    fn now_stamps_current_time() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        let e = PersistedEntry::now(Event::Bye);
        assert!(e.ts_us >= before);
        // Within a reasonable window.
        assert!(e.ts_us < before + 10_000_000);
    }
}
