// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Rolling benchmark-row history kept by the daemon so late-joining clients can hydrate.

use std::collections::VecDeque;

use rocm_dash_core::bench_schema::BenchmarkRow;

pub struct BenchRing {
    cap: usize,
    inner: VecDeque<BenchmarkRow>,
}

impl BenchRing {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: VecDeque::with_capacity(cap),
        }
    }

    pub fn push(&mut self, row: BenchmarkRow) {
        if self.inner.len() == self.cap {
            self.inner.pop_front();
        }
        self.inner.push_back(row);
    }

    pub fn iter(&self) -> impl Iterator<Item = &BenchmarkRow> {
        self.inner.iter()
    }

    /// Cloned snapshot of the ring contents, oldest first.
    pub fn snapshot(&self) -> Vec<BenchmarkRow> {
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

    fn row(run: u32) -> BenchmarkRow {
        BenchmarkRow {
            run,
            ..BenchmarkRow::default()
        }
    }

    #[test]
    fn caps_at_capacity() {
        let mut r = BenchRing::new(3);
        for i in 0..10 {
            r.push(row(i));
        }
        assert_eq!(r.len(), 3);
        let snap = r.snapshot();
        assert_eq!(snap.first().unwrap().run, 7);
        assert_eq!(snap.last().unwrap().run, 9);
    }
}
