//! Pure reducer. `State::apply(StateEvent) -> Vec<SideEffect>`.
//! See `../wiki/concepts/tea-reducer-pattern.md`.

use std::collections::{HashMap, VecDeque};

use crate::bench_schema::BenchmarkRow;
use crate::metrics::{Instance, Snapshot};

/// Maximum sparkline history we keep in-state.
pub const SNAPSHOT_RING_CAP: usize = 300;

/// Maximum benchmark rows kept in memory (FIFO).
pub const BENCH_RING_CAP: usize = 10_000;

#[derive(Debug, Default)]
pub struct State {
    pub latest: Option<Snapshot>,
    pub history: VecDeque<Snapshot>,
    pub instances: HashMap<String, Instance>,
    pub bench_rows: VecDeque<BenchmarkRow>,
    pub paused: bool,
}

#[derive(Debug, Clone)]
pub enum StateEvent {
    Tick(Snapshot),
    InstanceUpserted(Instance),
    InstanceRemoved(String),
    BenchmarkRows(Vec<BenchmarkRow>),
    Pause,
    Resume,
    Reset,
}

#[derive(Debug, Clone)]
pub enum SideEffect {
    Persist,
    BroadcastSnapshot,
    BroadcastInstance(String),
    BroadcastInstanceRemoved(String),
    BroadcastBenchRows(usize),
}

impl State {
    pub fn apply(&mut self, event: StateEvent) -> Vec<SideEffect> {
        match event {
            StateEvent::Tick(snap) => {
                if self.paused {
                    return Vec::new();
                }
                self.history.push_back(snap.clone());
                while self.history.len() > SNAPSHOT_RING_CAP {
                    self.history.pop_front();
                }
                self.latest = Some(snap);
                vec![SideEffect::BroadcastSnapshot]
            }
            StateEvent::InstanceUpserted(inst) => {
                let id = inst.container_id.clone();
                self.instances.insert(id.clone(), inst);
                vec![SideEffect::BroadcastInstance(id)]
            }
            StateEvent::InstanceRemoved(id) => {
                self.instances.remove(&id);
                vec![SideEffect::BroadcastInstanceRemoved(id)]
            }
            StateEvent::BenchmarkRows(rows) => {
                let n = rows.len();
                for r in rows {
                    self.bench_rows.push_back(r);
                    while self.bench_rows.len() > BENCH_RING_CAP {
                        self.bench_rows.pop_front();
                    }
                }
                vec![SideEffect::BroadcastBenchRows(n), SideEffect::Persist]
            }
            StateEvent::Pause => {
                self.paused = true;
                Vec::new()
            }
            StateEvent::Resume => {
                self.paused = false;
                Vec::new()
            }
            StateEvent::Reset => {
                self.latest = None;
                self.history.clear();
                self.instances.clear();
                self.bench_rows.clear();
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn snap_at(secs: i64) -> Snapshot {
        Snapshot {
            timestamp: chrono::DateTime::<Utc>::from_timestamp(secs, 0).unwrap(),
            ..Snapshot::default()
        }
    }

    #[test]
    fn tick_pushes_history_and_broadcasts() {
        let mut s = State::default();
        let fx = s.apply(StateEvent::Tick(snap_at(1)));
        assert_eq!(s.history.len(), 1);
        assert!(s.latest.is_some());
        assert!(matches!(fx.as_slice(), [SideEffect::BroadcastSnapshot]));
    }

    #[test]
    fn pause_drops_ticks() {
        let mut s = State::default();
        s.apply(StateEvent::Pause);
        let fx = s.apply(StateEvent::Tick(snap_at(1)));
        assert!(fx.is_empty());
        assert_eq!(s.history.len(), 0);
    }

    #[test]
    fn history_caps_at_ring_size() {
        let mut s = State::default();
        for i in 0..(SNAPSHOT_RING_CAP + 5) as i64 {
            s.apply(StateEvent::Tick(snap_at(i)));
        }
        assert_eq!(s.history.len(), SNAPSHOT_RING_CAP);
    }

    #[test]
    fn instance_lifecycle() {
        let mut s = State::default();
        let inst = Instance {
            container_id: "c1".into(),
            ..Instance::default()
        };
        s.apply(StateEvent::InstanceUpserted(inst));
        assert!(s.instances.contains_key("c1"));
        s.apply(StateEvent::InstanceRemoved("c1".into()));
        assert!(!s.instances.contains_key("c1"));
    }
}
