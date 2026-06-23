// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Shared primitives for async collectors that scrape N targets concurrently.
//!
//! Today: [[VllmPrometheusCollector]] uses this pattern hand-rolled in the
//! daemon's runner. As Strix-Halo's `LlamaSlotsCollector` and any future
//! per-instance scrapers come online, they'll reuse `parallel_scrape` and
//! the `WarningBus` instead of re-implementing the JoinSet glue.

use std::future::Future;
use std::sync::Arc;

use tokio::task::JoinSet;

/// Run `f` against every `target` concurrently and collect `(target, result)`
/// pairs in completion order (NOT input order — order is not guaranteed).
///
/// The closure is invoked with each `target`. The returned `Future` must be
/// `'static + Send`, which the caller typically achieves by cloning anything
/// captured into the future (cheap, since reqwest's `Client` is `Arc`-shared).
///
/// Empty input returns an empty vec without spawning anything.
pub async fn parallel_scrape<T, R, F, Fut>(targets: Vec<T>, f: F) -> Vec<(T, R)>
where
    T: Clone + Send + 'static,
    R: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync,
    Fut: Future<Output = R> + Send + 'static,
{
    if targets.is_empty() {
        return Vec::new();
    }
    let mut joins = JoinSet::new();
    for t in targets {
        let tag = t.clone();
        let fut = f(t);
        joins.spawn(async move { (tag, fut.await) });
    }
    let mut out = Vec::with_capacity(joins.len());
    while let Some(res) = joins.join_next().await {
        if let Ok(pair) = res {
            out.push(pair);
        }
    }
    out
}

/// Collect warnings across a single runner tick from concurrent collector
/// failures, then drain into the snapshot. Cheap clone (Arc-backed) so it
/// can be passed into spawned tasks.
#[derive(Debug, Default, Clone)]
pub struct WarningBus {
    inner: Arc<std::sync::Mutex<Vec<String>>>,
}

impl WarningBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, msg: impl Into<String>) {
        if let Ok(mut v) = self.inner.lock() {
            v.push(msg.into());
        }
    }

    /// Drain all accumulated warnings, replacing the bus with an empty vec.
    pub fn drain(&self) -> Vec<String> {
        match self.inner.lock() {
            Ok(mut v) => std::mem::take(&mut *v),
            Err(_) => Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().map_or(true, |v| v.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn parallel_scrape_runs_all_targets_concurrently() {
        let targets = vec![1u32, 2, 3, 4, 5];
        let out = parallel_scrape(targets.clone(), |n| async move { n * 10 }).await;
        assert_eq!(out.len(), targets.len());
        let mut values: Vec<u32> = out.iter().map(|(t, _)| *t).collect();
        values.sort_unstable();
        assert_eq!(values, targets);
        for (n, mapped) in &out {
            assert_eq!(*mapped, n * 10);
        }
    }

    #[tokio::test]
    async fn parallel_scrape_empty_input_is_empty_output() {
        let out: Vec<(u32, u32)> = parallel_scrape(Vec::new(), |n| async move { n }).await;
        assert!(out.is_empty());
    }

    #[test]
    fn warning_bus_collects_then_drains() {
        let bus = WarningBus::new();
        bus.push("docker: unreachable");
        bus.push("vllm: 502 on :8000");
        let v = bus.drain();
        assert_eq!(v.len(), 2);
        assert!(bus.is_empty());
        // Second drain returns empty.
        assert!(bus.drain().is_empty());
    }

    #[tokio::test]
    async fn warning_bus_is_clone_safe_across_tasks() {
        let bus = WarningBus::new();
        let mut joins = JoinSet::new();
        for i in 0..10 {
            let b = bus.clone();
            joins.spawn(async move { b.push(format!("warn {i}")) });
        }
        while joins.join_next().await.is_some() {}
        let v = bus.drain();
        assert_eq!(v.len(), 10);
    }
}
