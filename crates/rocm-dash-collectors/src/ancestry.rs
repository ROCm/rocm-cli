// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! GPU process → managed service, by process ancestry.
//!
//! `amd-smi process` reports the host PIDs of GPU compute processes. For a
//! managed `rocm serve` service those are the engine's workers, which descend
//! from the PID rocm-cli recorded for the engine (or for its supervisor), so a
//! walk up `/proc/<pid>/stat` parent links reaches a recorded root. This is the
//! managed-service counterpart of [`crate::cgroup`], which only recognises
//! docker/containerd ids and can never name a managed service.
//!
//! The matcher is pure: the `/proc/<pid>/stat` reader is injected, so the
//! whole thing is testable on any host. Only [`owner_of_pid`] reads `/proc`,
//! and it is `cfg`-gated with a non-Linux stub.

/// How many parent links to follow before giving up. Real engines sit a few
/// levels deep (launcher, server, worker); the bound only stops a corrupt or
/// adversarial parent chain from looping.
const MAX_DEPTH: usize = 32;

/// The lowest PID that can be a root.
///
/// PID 0 means "not recorded" and PID 1 is init, an ancestor of every process on
/// the host: a root there would own all of them, including GPU processes that
/// belong to no service.
pub const MIN_ROOT_PID: u32 = 2;

/// A recorded process that identifies one managed service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Root {
    /// The recorded PID (engine or supervisor).
    pub pid: u32,
    /// The kernel start time recorded with it, when there is one. A PID can be
    /// recycled, so a root whose live start time differs is a different process
    /// that merely has the same number, and must not claim anything.
    pub start_ticks: Option<u64>,
    /// The managed service this root belongs to.
    pub owner: String,
}

/// `(parent pid, start time in clock ticks)` from the contents of
/// `/proc/<pid>/stat`, or `None` when it does not parse.
///
/// The process name (field 2) is parenthesised and may itself contain spaces
/// and parentheses, so the fields after it are counted from the *last* `)`.
/// Field numbers below are the `proc(5)` ones: state is 3, ppid is 4 and the
/// start time is 22.
#[must_use]
pub fn parse_stat(stat: &str) -> Option<(u32, u64)> {
    let rest = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ppid = fields.get(1)?.parse().ok()?;
    let start = fields.get(19)?.parse().ok()?;
    Some((ppid, start))
}

/// The managed service that owns `pid`, if `pid` is, or descends from, one of
/// `roots`. `read_stat` returns the contents of `/proc/<pid>/stat` for a PID.
///
/// A root counts only when its recorded start time (if it has one) matches the
/// live process, so a recycled PID never claims a stranger's GPU memory. An
/// unreadable link ends the walk with no owner rather than a guess.
pub fn owner_of<F>(pid: u32, roots: &[Root], read_stat: F) -> Option<String>
where
    F: Fn(u32) -> Option<String>,
{
    let mut current = pid;
    for _ in 0..MAX_DEPTH {
        let (parent, start) = parse_stat(&read_stat(current)?)?;
        if let Some(root) = roots
            .iter()
            .find(|r| r.pid == current && r.pid >= MIN_ROOT_PID)
            && root.start_ticks.is_none_or(|recorded| recorded == start)
        {
            return Some(root.owner.clone());
        }
        if parent == 0 || parent == current {
            return None;
        }
        current = parent;
    }
    None
}

/// [`owner_of`] against the live `/proc`.
#[cfg(target_os = "linux")]
#[must_use]
pub fn owner_of_pid(pid: u32, roots: &[Root]) -> Option<String> {
    owner_of(pid, roots, |p| {
        std::fs::read_to_string(format!("/proc/{p}/stat")).ok()
    })
}

/// Non-Linux stub: there is no `/proc`, so ancestry is never available.
#[cfg(not(target_os = "linux"))]
#[must_use]
pub const fn owner_of_pid(_pid: u32, _roots: &[Root]) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A `/proc/<pid>/stat` line: `comm` is field 2, then state, ppid, and 17
    /// filler fields so the start time lands at field 22.
    fn stat(pid: u32, comm: &str, ppid: u32, start: u64) -> String {
        let filler = vec!["0"; 17].join(" ");
        format!("{pid} ({comm}) S {ppid} {filler} {start} 0 0")
    }

    fn table(entries: &[(u32, u32, u64)]) -> HashMap<u32, String> {
        entries
            .iter()
            .map(|&(pid, ppid, start)| (pid, stat(pid, "proc", ppid, start)))
            .collect()
    }

    fn root(pid: u32, start: Option<u64>, owner: &str) -> Root {
        Root {
            pid,
            start_ticks: start,
            owner: owner.into(),
        }
    }

    #[test]
    fn parse_stat_reads_ppid_and_start_time_even_when_the_name_is_awkward() {
        assert_eq!(parse_stat(&stat(10, "vllm", 4, 777)), Some((4, 777)));
        // Spaces and parentheses inside the process name.
        assert_eq!(
            parse_stat(&stat(10, "my (odd) name", 4, 777)),
            Some((4, 777))
        );
        assert_eq!(parse_stat("garbage"), None);
        assert_eq!(parse_stat("10 (short) S 4"), None);
    }

    #[test]
    fn a_worker_resolves_to_the_service_whose_engine_is_its_ancestor() {
        // 100 (supervisor) -> 200 (engine) -> 300 (TP worker) -> 301 (helper)
        let t = table(&[
            (100, 1, 5),
            (200, 100, 6),
            (300, 200, 7),
            (301, 300, 8),
            (1, 0, 1),
        ]);
        let roots = [root(200, Some(6), "svc-a")];
        let read = |p: u32| t.get(&p).cloned();
        assert_eq!(owner_of(301, &roots, read), Some("svc-a".into()));
        assert_eq!(owner_of(300, &roots, read), Some("svc-a".into()));
        // The root itself counts, and a process outside the tree does not.
        assert_eq!(owner_of(200, &roots, read), Some("svc-a".into()));
        assert_eq!(owner_of(100, &roots, read), None);
    }

    #[test]
    fn two_services_are_told_apart() {
        let t = table(&[(10, 1, 1), (11, 10, 2), (20, 1, 3), (21, 20, 4), (1, 0, 1)]);
        let roots = [root(10, None, "a"), root(20, None, "b")];
        let read = |p: u32| t.get(&p).cloned();
        assert_eq!(owner_of(11, &roots, read), Some("a".into()));
        assert_eq!(owner_of(21, &roots, read), Some("b".into()));
    }

    #[test]
    fn a_recycled_pid_does_not_claim_a_stranger_s_children() {
        // The recorded engine had start time 6, but pid 200 is now a different
        // process (start 99) that happens to own a GPU worker.
        let t = table(&[(200, 1, 99), (300, 200, 7), (1, 0, 1)]);
        let roots = [root(200, Some(6), "svc-a")];
        assert_eq!(owner_of(300, &roots, |p| t.get(&p).cloned()), None);
        // With no recorded start time the root is taken at its word.
        let loose = [root(200, None, "svc-a")];
        assert_eq!(
            owner_of(300, &loose, |p| t.get(&p).cloned()),
            Some("svc-a".into())
        );
    }

    #[test]
    fn a_root_at_init_or_zero_never_owns_anything() {
        // Every process descends from init, so a root recorded as PID 1 would
        // claim every GPU process on the host, whoever it belongs to.
        let t = table(&[(300, 200, 7), (200, 1, 6), (1, 0, 1)]);
        let read = |p: u32| t.get(&p).cloned();
        // PID 0 documents a state the walk never reaches (a parent of 0 ends it
        // before `current` is assigned); PID 1 is the one the guard decides.
        for pid in [0, 1] {
            let roots = [root(pid, None, "svc")];
            assert_eq!(owner_of(300, &roots, read), None, "root pid {pid}");
        }
        // A real root beside a bogus one still works.
        let roots = [root(1, None, "bogus"), root(200, None, "real")];
        assert_eq!(owner_of(300, &roots, read), Some("real".into()));
        // A process outside every service tree has no owner even with a bogus root.
        let stray = table(&[(900, 1, 50), (1, 0, 1)]);
        assert_eq!(
            owner_of(900, &[root(1, None, "bogus")], |p| stray.get(&p).cloned()),
            None
        );
    }

    #[test]
    fn owner_of_pid_is_total_on_every_platform() {
        // A PID that cannot exist: unreadable on Linux, a stub elsewhere. Never
        // a panic, never an owner.
        assert_eq!(owner_of_pid(u32::MAX, &[]), None);
        assert_eq!(owner_of_pid(u32::MAX, &[root(2, None, "svc")]), None);
    }

    #[test]
    fn unreadable_links_and_cycles_end_without_an_owner() {
        let roots = [root(1000, None, "svc")];
        // The worker's parent has vanished.
        let t = table(&[(300, 200, 7)]);
        assert_eq!(owner_of(300, &roots, |p| t.get(&p).cloned()), None);
        // A parent chain that loops must terminate.
        let looped = table(&[(5, 6, 1), (6, 5, 2)]);
        assert_eq!(owner_of(5, &roots, |p| looped.get(&p).cloned()), None);
        // A process that is its own parent, and init.
        let selfref = table(&[(7, 7, 1)]);
        assert_eq!(owner_of(7, &roots, |p| selfref.get(&p).cloned()), None);
        assert_eq!(owner_of(5, &[], |_| None), None);
    }
}
