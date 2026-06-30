// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Rolling snapshot history kept by the daemon so late-joining clients can hydrate.

use std::collections::VecDeque;

use rocm_dash_core::metrics::Snapshot;

pub struct SnapshotRing {
    cap: usize,
    inner: VecDeque<Snapshot>,
}

impl SnapshotRing {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: VecDeque::with_capacity(cap),
        }
    }

    pub fn push(&mut self, snap: Snapshot) {
        if self.inner.len() == self.cap {
            self.inner.pop_front();
        }
        self.inner.push_back(snap);
    }

    pub fn latest(&self) -> Option<&Snapshot> {
        self.inner.back()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Snapshot> {
        self.inner.iter()
    }

    /// Cloned snapshot of the ring contents, oldest first.
    pub fn snapshot(&self) -> Vec<Snapshot> {
        self.inner.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    fn at(t: i64) -> Snapshot {
        Snapshot {
            timestamp: DateTime::<Utc>::from_timestamp(t, 0).unwrap(),
            ..Snapshot::default()
        }
    }

    #[test]
    fn caps_at_capacity() {
        let mut r = SnapshotRing::new(3);
        for i in 0..10 {
            r.push(at(i));
        }
        assert_eq!(r.len(), 3);
        assert_eq!(r.latest().unwrap().timestamp.timestamp(), 9);
    }
}
