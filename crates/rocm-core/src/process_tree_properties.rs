// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the process-tree traversal that decides which PIDs a
//! [`KillScope::Tree`](crate::KillScope::Tree) stop signals — `rocm services
//! stop` and `rocmd`'s managed-service stop both reach it through
//! `terminate_verified`.
//!
//! Tables are generated over a deliberately tiny PID alphabet so that shared
//! parents, cycles and self-parents come up in a large share of cases instead
//! of being a measure-zero corner a uniform generator never samples. A
//! consistent process tree has no cycle, but `collect_process_tree` assembles
//! its table from one `/proc/<pid>/stat` read per PID, each at a different
//! moment, so a cycle from a PID reused during the scan is rare but cannot be
//! ruled out; the walk has to terminate on it rather than spin. A self-parent
//! cannot come from `/proc` at all and is included as the one-node case of the
//! same cycle.
//! `generator_reaches_the_interesting_shapes` checks the generator keeps
//! producing each of these shapes.

use super::walk_process_tree;
use proptest::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write as _;
use std::sync::mpsc;
use std::time::Duration;

/// Case count: `default`, unless `PROPTEST_CASES` (proptest's own knob, which
/// an explicit count would otherwise override) asks for a different run.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// PIDs are drawn from here. `0` and `1` are included so tables can take
/// init's real shape (`/proc` reports `ppid == 0` for it) alongside the rest.
const ALPHABET: u32 = 6;

/// A `(pid, ppid)` table of one to six PIDs. `pid` is unique (there is one
/// `/proc/<pid>/stat` per PID) but `ppid` is unconstrained, so shared parents,
/// cycles and self-parents are common.
fn arb_table() -> impl Strategy<Value = Vec<(u32, u32)>> {
    let alphabet: Vec<u32> = (0..ALPHABET).collect();
    proptest::sample::subsequence(alphabet, 1..=ALPHABET as usize).prop_flat_map(|pids| {
        let len = pids.len();
        proptest::collection::vec(0..ALPHABET, len)
            .prop_map(move |ppids| pids.iter().copied().zip(ppids).collect())
    })
}

/// Run the walk on its own thread and give up after a bound, so a walk that
/// never terminates on a cycle fails the suite with the offending input
/// instead of hanging it until the CI job times out. The process exits rather
/// than returning a failure: a returned failure would make proptest shrink by
/// re-running the same input, leaving another spinning thread behind each
/// time. Under nextest that ends one test; under a plain `cargo test` it ends
/// the whole test binary, which only happens once this suite is failing.
fn walk_bounded(root: u32, children: &HashMap<u32, Vec<u32>>) -> Vec<u32> {
    let (sender, receiver) = mpsc::channel();
    let owned = children.clone();
    std::thread::spawn(move || {
        let _ = sender.send(walk_process_tree(root, &owned));
    });
    if let Ok(order) = receiver.recv_timeout(Duration::from_secs(5)) {
        return order;
    }
    // Straight to the stream: the test harness's output capture would swallow
    // `eprintln!` along with the process.
    let _ = writeln!(
        std::io::stderr(),
        "walk_process_tree did not terminate within 5s: root = {root}, children = {children:?}"
    );
    std::process::exit(101);
}

/// Mirror of the `children` map `collect_process_tree` builds from `/proc`.
fn children_of(table: &[(u32, u32)]) -> HashMap<u32, Vec<u32>> {
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for (pid, ppid) in table {
        children.entry(*ppid).or_default().push(*pid);
    }
    children
}

/// Independent oracle: breadth-first reachability from `root`. Deliberately a
/// different algorithm from the implementation's depth-first stack walk.
fn reachable(root: u32, children: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([root]);
    while let Some(pid) = queue.pop_front() {
        if !seen.insert(pid) {
            continue;
        }
        if let Some(kids) = children.get(&pid) {
            queue.extend(kids.iter().copied());
        }
    }
    seen
}

/// Whether `pid`'s parent links lead back to `pid` itself after at least
/// `min_hops` steps. With `min_hops == 2` a self-parent does not count, so the
/// result is about genuine multi-process cycles.
fn on_cycle(pid: u32, parent: &HashMap<u32, u32>, min_hops: usize) -> bool {
    let mut cursor = pid;
    for hops in 1..=parent.len() {
        let Some(next) = parent.get(&cursor).copied() else {
            return false;
        };
        if next == pid {
            return hops >= min_hops;
        }
        cursor = next;
    }
    false
}

proptest! {
    // The walk is a pure function over at most six PIDs, so a deep run is
    // cheap.
    #![proptest_config(ProptestConfig::with_cases(cases(4096)))]

    /// The kill set is exactly the PIDs reachable from the root — no stranger
    /// adopted in, no descendant left behind — each PID appears once, and the
    /// root leads the walk.
    #[test]
    fn walk_matches_reachability_oracle(table in arb_table(), root in 0..ALPHABET) {
        let children = children_of(&table);
        let order = walk_bounded(root, &children);

        prop_assert_eq!(order.first().copied(), Some(root), "root must lead the walk");

        let mut unique = HashSet::new();
        for pid in &order {
            prop_assert!(unique.insert(*pid), "PID {} would be signalled twice", pid);
        }
        prop_assert_eq!(unique, reachable(root, &children));
    }

    /// Every member other than the root comes after its parent, as
    /// `walk_process_tree` documents.
    #[test]
    fn parents_are_listed_before_their_children(table in arb_table(), root in 0..ALPHABET) {
        let parent: HashMap<u32, u32> = table.iter().copied().collect();
        let order = walk_bounded(root, &children_of(&table));
        let position: HashMap<u32, usize> = order
            .iter()
            .enumerate()
            .map(|(index, pid)| (*pid, index))
            .collect();
        for (index, pid) in order.iter().enumerate().skip(1) {
            let ppid = parent.get(pid).copied();
            let parent_index = ppid.and_then(|ppid| position.get(&ppid).copied());
            prop_assert!(
                parent_index.is_some_and(|parent_index| parent_index < index),
                "PID {} is listed at {} but its parent {:?} is at {:?}",
                pid,
                index,
                ppid,
                parent_index
            );
        }
    }
}

/// Guards the suite from going vacuous: a generator that stopped producing
/// these shapes would pass every property above while testing none of the
/// behaviour that matters. Floors are roughly half the rates the generator
/// reaches today, so they trip on a real regression rather than on noise.
#[test]
fn generator_reaches_the_interesting_shapes() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    const DRAWS: u32 = 1024;
    let mut runner = TestRunner::deterministic();
    let strategy = (arb_table(), 0..ALPHABET);
    let (mut shared_parents, mut self_parents, mut cycles, mut root_on_cycle) =
        (0_u32, 0_u32, 0_u32, 0_u32);
    for _ in 0..DRAWS {
        let (table, root) = strategy
            .new_tree(&mut runner)
            .expect("generate table")
            .current();
        let parent: HashMap<u32, u32> = table.iter().copied().collect();
        if children_of(&table).values().any(|kids| kids.len() > 1) {
            shared_parents += 1;
        }
        if table.iter().any(|(pid, ppid)| pid == ppid) {
            self_parents += 1;
        }
        if table.iter().any(|(pid, _)| on_cycle(*pid, &parent, 2)) {
            cycles += 1;
        }
        // A PID has one parent, so a cycle the root is not on cannot be
        // reached from it: a root on a cycle, its own parent included, is the
        // only shape in which the walk's `seen` check stops it revisiting a
        // PID. Self-parents already have their own floor, so this one counts
        // only cycles of two or more.
        if on_cycle(root, &parent, 2) {
            root_on_cycle += 1;
        }
    }
    eprintln!(
        "generator reach over {DRAWS} draws: shared parents {shared_parents}, \
         self-parents {self_parents}, cycles of two or more {cycles}, \
         root on a cycle of two or more {root_on_cycle}"
    );
    for (shape, count, floor) in [
        ("a shared parent", shared_parents, 280),
        ("a self-parent", self_parents, 220),
        ("a cycle of two or more PIDs", cycles, 120),
        ("a root on a cycle of two or more PIDs", root_on_cycle, 55),
    ] {
        assert!(
            count >= floor,
            "generator rarely produces {shape}: {count}/{DRAWS}, floor {floor}"
        );
    }
}
