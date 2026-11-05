// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which device memory pool the serve drain-wait judges, and the floor it
//! demands of it.
//!
//! The wait between serve scenarios (`wait_for_free_vram` in the serving
//! steps) exists to see a just-killed engine hand its device memory back. It
//! has to look at the pool the engine allocates from. On a discrete card that
//! is VRAM. On an APU such as Strix Halo the VRAM total is only the fixed BIOS
//! carveout (512 MiB on the CI hosts) while the engine draws on GTT-backed
//! system memory (about 100 GiB there), so a floor taken from the carveout can
//! never be met and every GPU serve spent its whole drain deadline waiting.
//! `.github/scripts/gpu-preflight.sh` makes the same switch for the same reason.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Upper bound on the free-memory floor (MiB).
///
/// Sized so the largest single
/// scenario model can allocate its engine's memory share without tripping its
/// startup check: Qwen3.6-27B plus vLLM's default fraction-of-total KV
/// reservation needs the MI300X mostly clear. Only a data-center GPU has this
/// much, so on smaller pools the floor is capped to 90% of the pool total (see
/// [`required_free_mib`]) — otherwise the check could never pass. Note the cap
/// no longer sits above what vLLM asks for: rocm-cli used to pin utilization
/// below it, and now defers to vLLM's own (higher) default, so the wait has
/// little headroom left and can time out on a device that is merely holding
/// display memory. The wait is best-effort and the serve proceeds regardless,
/// so this costs time rather than correctness; sizing a deliberate margin is
/// follow-up work.
pub const MAX_FREE_FLOOR_MIB: u64 = 150_000;

/// A VRAM total below this is an APU's BIOS carveout, not private device memory.
///
/// The engine's allocations come out of GTT instead. It is the
/// 8 GiB floor the Strix Halo preflight lanes pass as
/// `GPU_PREFLIGHT_MIN_FREE_GIB`, which `gpu-preflight.sh` compares the VRAM
/// total against for the same purpose. Every discrete card the lanes run on is
/// far above it (48 GiB and up).
pub const CARVEOUT_MAX_MIB: u64 = 8 * 1024;

/// Whether a VRAM total is an APU's carveout rather than private device memory.
///
/// Both the decision to read GTT at all ([`sample_device_memory`]) and the
/// choice of pool ([`select_drain_target`]) go through this, so they cannot
/// disagree.
#[must_use]
pub const fn is_carveout(vram_total_mib: u64) -> bool {
    vram_total_mib < CARVEOUT_MAX_MIB
}

/// One attempt to read from a device tool.
///
/// `Absent` and `Unreadable` are kept apart on purpose. With no tool installed
/// there is nothing to wait on (a mock or local run). A tool that is present
/// but stalled, failed, or printed something unparseable is a device that
/// could not be read right now, which must never be mistaken for either "no
/// tooling" or "drained".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe<T> {
    /// The tool is not installed.
    Absent,
    /// The tool is there but gave no usable reading.
    Unreadable,
    Read(T),
}

impl<T> Probe<T> {
    /// Turn a raw reading into a parsed one; a reading `f` rejects is `Unreadable`.
    #[must_use]
    pub fn parse<U>(self, f: impl FnOnce(T) -> Option<U>) -> Probe<U> {
        match self {
            Self::Absent => Probe::Absent,
            Self::Unreadable => Probe::Unreadable,
            Self::Read(value) => f(value).map_or(Probe::Unreadable, Probe::Read),
        }
    }

    /// Fall back to `next` when this probe produced no reading.
    ///
    /// A reading from either wins. Without one, the result is `Absent` only if
    /// every tool tried was absent; if any tool was present and gave nothing,
    /// it is `Unreadable`.
    #[must_use]
    pub fn or_else(self, next: impl FnOnce() -> Self) -> Self {
        if matches!(self, Self::Read(_)) {
            return self;
        }
        match (self, next()) {
            (_, read @ Self::Read(_)) => read,
            (Self::Absent, Self::Absent) => Self::Absent,
            _ => Self::Unreadable,
        }
    }
}

/// One memory pool's size and free space, in MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool {
    pub total_mib: u64,
    pub free_mib: u64,
}

/// What the device tooling reported for GPU 0.
///
/// `gtt` is `None` when GTT was not read: a discrete card, or a host with no
/// tool that reports it. A GTT read
/// that failed never gets here (the sample is `Unreadable`, see
/// [`sample_device_memory`]), so every pool here has a non-zero total, VRAM and
/// GTT alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    pub vram: Pool,
    pub gtt: Option<Pool>,
}

/// Which pool a [`DrainTarget`] was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Vram,
    Gtt,
}

/// The pool to wait on, its current state, and the floor it must reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainTarget {
    pub kind: PoolKind,
    pub total_mib: u64,
    pub free_mib: u64,
    pub floor_mib: u64,
}

impl DrainTarget {
    /// Whether the pool has handed back enough memory.
    #[must_use]
    pub const fn drained(&self) -> bool {
        self.free_mib >= self.floor_mib
    }
}

/// The free-memory floor to wait for on a pool of `total_mib`: the
/// model-sized ceiling, but never more than 90% of the pool.
#[must_use]
pub fn required_free_mib(total_mib: u64) -> u64 {
    MAX_FREE_FLOOR_MIB.min(total_mib / 100 * 90)
}

/// What one sample means for the drain wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Judgement {
    /// No device tool is installed: nothing to wait on (mock/local run).
    NoTooling,
    /// A tool is present but gave no usable reading. This is not "no tooling"
    /// and not "drained": the wait keeps trying.
    Unreadable,
    /// A reading, and the pool and floor it is held to.
    Target(DrainTarget),
}

/// Turn a sample into what the wait does with it.
#[must_use]
pub fn judge(sample: Probe<DeviceMemory>) -> Judgement {
    match sample {
        Probe::Absent => Judgement::NoTooling,
        Probe::Unreadable => Judgement::Unreadable,
        Probe::Read(memory) => Judgement::Target(select_drain_target(memory)),
    }
}

/// Choose the pool the engine allocates from and the floor to hold it to.
///
/// A VRAM total that [`is_carveout`] means the GTT pool is used, when one was
/// read. Without GTT data the answer is VRAM, which is what every host got
/// before this choice existed; [`sample_device_memory`] only leaves GTT out
/// when the host has no tool that reports it, never when a read failed.
#[must_use]
pub fn select_drain_target(memory: DeviceMemory) -> DrainTarget {
    let (kind, pool) = match memory.gtt {
        Some(gtt) if is_carveout(memory.vram.total_mib) => (PoolKind::Gtt, gtt),
        _ => (PoolKind::Vram, memory.vram),
    };
    DrainTarget {
        kind,
        total_mib: pool.total_mib,
        free_mib: pool.free_mib,
        floor_mib: required_free_mib(pool.total_mib),
    }
}

/// Read `(total, free)` in MiB from `rocm-smi --showmeminfo <pool> --csv` output.
///
/// `total_column` and `used_column` are substrings of the two header
/// names (`Total Memory` and `Total Used Memory` for VRAM; the `GTT ` forms for
/// GTT); the first data row is the first GPU. Values are bytes in the output.
#[must_use]
pub fn parse_rocm_smi_csv_pool(
    text: &str,
    total_column: &str,
    used_column: &str,
) -> Option<(u64, u64)> {
    let mut lines = text.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split(',').collect();
    let total_idx = cols.iter().position(|c| c.contains(total_column))?;
    let used_idx = cols.iter().position(|c| c.contains(used_column))?;
    let row = lines.next()?;
    let vals: Vec<&str> = row.split(',').collect();
    let total: u64 = vals.get(total_idx)?.trim().parse().ok()?;
    let used: u64 = vals.get(used_idx)?.trim().parse().ok()?;
    let mib = 1024 * 1024;
    Some((total / mib, total.saturating_sub(used) / mib))
}

/// Assemble one device-memory sample from a VRAM probe and a GTT probe.
///
/// `gtt` runs only when the VRAM total [`is_carveout`], so a discrete-card host
/// never makes the extra call. On a carveout host the GTT reading is what the
/// engine's memory comes out of, so:
///
/// - a GTT reading makes the sample;
/// - a GTT probe that ran but gave nothing usable (a stall, a failure, output
///   not understood, or a zero total, which is a failed read rather than an
///   empty pool) makes the whole sample `Unreadable`. Judging the tiny
///   carveout instead could report "drained" while the engine still holds its
///   GTT-backed memory. `gpu-preflight.sh` treats a zero GTT total the same way,
///   as an unreadable sample to retry;
/// - no tool that reports GTT at all (`Absent`) leaves `gtt` empty, and the
///   wait judges VRAM as it did before GTT was considered.
pub fn sample_device_memory(
    vram: impl FnOnce() -> Probe<(u64, u64)>,
    gtt: impl FnOnce() -> Probe<(u64, u64)>,
) -> Probe<DeviceMemory> {
    let (vram_total, vram_free) = match vram() {
        // A zero total is a failed read, not a pool: its floor would be zero and
        // every sample would count as drained.
        Probe::Read((total, free)) if total > 0 => (total, free),
        Probe::Read(_) | Probe::Unreadable => return Probe::Unreadable,
        Probe::Absent => return Probe::Absent,
    };
    let vram = Pool {
        total_mib: vram_total,
        free_mib: vram_free,
    };
    if !is_carveout(vram_total) {
        return Probe::Read(DeviceMemory { vram, gtt: None });
    }
    match gtt() {
        Probe::Read((total_mib, free_mib)) if total_mib > 0 => Probe::Read(DeviceMemory {
            vram,
            gtt: Some(Pool {
                total_mib,
                free_mib,
            }),
        }),
        Probe::Read(_) | Probe::Unreadable => Probe::Unreadable,
        Probe::Absent => Probe::Read(DeviceMemory { vram, gtt: None }),
    }
}

/// How long one `amd-smi` or `rocm-smi` call may run before it is an unreadable sample.
///
/// These tools can stall on a wedged driver, and the drain wait is best-effort:
/// a stalled probe must cost one sample, not the worker.
pub const SMI_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

const SMI_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// How long to wait for the reader after the child has exited. A grandchild
/// that inherited the pipe can keep it open, so the reader is abandoned
/// rather than joined.
const SMI_DRAIN_GRACE: Duration = Duration::from_millis(200);

/// Run `command` and return its stdout if it exits successfully within `timeout`.
///
/// A program that is not installed is [`Probe::Absent`]. Any other spawn
/// failure, a non-zero exit, or a timeout is [`Probe::Unreadable`]; on a
/// timeout the child is killed and reaped before this returns.
///
/// Only the direct child is killed. A grandchild that inherited the stdout
/// pipe keeps the reader thread blocked in `read_to_end` until it exits, so a
/// timed-out probe can leak one thread. The probes here (`amd-smi`,
/// `rocm-smi`) do not fork helpers that outlive them, and the drain wait makes
/// a handful of calls, so this is accepted rather than guarded.
///
/// Written against `std` alone because `rocm_core::process::run_with_timeout`
/// is reachable here only as a dev-dependency, and this module is library code.
pub fn bounded_stdout(mut command: Command, timeout: Duration) -> Probe<String> {
    let mut child = match command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Probe::Absent,
        Err(_) => return Probe::Unreadable,
    };
    let Some(mut pipe) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Probe::Unreadable;
    };
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = pipe.read_to_end(&mut bytes);
        let _ = sender.send(bytes);
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(SMI_POLL_INTERVAL),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Probe::Unreadable;
            }
        }
    };
    if !status.success() {
        return Probe::Unreadable;
    }
    receiver
        .recv_timeout(SMI_DRAIN_GRACE)
        .map_or(Probe::Unreadable, |bytes| {
            Probe::Read(String::from_utf8_lossy(&bytes).into_owned())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pools `gpu-preflight.sh` printed on a Strix Halo lane host
    /// (`VRAM 203321344/536870912, GTT 41787392/107374182400`, used/total in
    /// bytes), and the same reading in MiB.
    fn strix_halo() -> DeviceMemory {
        DeviceMemory {
            vram: Pool {
                total_mib: 512,
                free_mib: 318,
            },
            gtt: Some(Pool {
                total_mib: 102_400,
                free_mib: 102_360,
            }),
        }
    }

    fn mi300x() -> DeviceMemory {
        DeviceMemory {
            vram: Pool {
                total_mib: 196_592,
                free_mib: 196_309,
            },
            gtt: Some(Pool {
                total_mib: 1_000_000,
                free_mib: 1_000_000,
            }),
        }
    }

    #[test]
    fn a_carveout_host_is_judged_by_gtt_and_a_healthy_strix_halo_counts_as_drained() {
        let target = select_drain_target(strix_halo());
        assert_eq!(target.kind, PoolKind::Gtt);
        assert_eq!(target.total_mib, 102_400);
        assert_eq!(target.floor_mib, 92_160);
        assert!(
            target.drained(),
            "an idle Strix Halo host must not wait out the drain deadline: {target:?}"
        );
    }

    #[test]
    fn a_carveout_host_whose_gtt_is_mostly_held_is_not_drained() {
        let mut memory = strix_halo();
        memory.gtt = Some(Pool {
            total_mib: 102_400,
            free_mib: 60_000,
        });
        let target = select_drain_target(memory);
        assert_eq!(target.kind, PoolKind::Gtt);
        assert!(!target.drained(), "{target:?}");
    }

    #[test]
    fn a_discrete_card_is_still_judged_by_vram_with_the_model_sized_cap() {
        let target = select_drain_target(mi300x());
        assert_eq!(target.kind, PoolKind::Vram);
        assert_eq!(target.total_mib, 196_592);
        assert_eq!(target.floor_mib, MAX_FREE_FLOOR_MIB);
        assert!(target.drained());
    }

    #[test]
    fn a_discrete_card_holding_most_of_its_vram_is_not_drained() {
        let mut memory = mi300x();
        memory.vram.free_mib = 40_000;
        assert!(!select_drain_target(memory).drained());
    }

    #[test]
    fn a_host_without_gtt_data_keeps_the_vram_behaviour() {
        let mut memory = strix_halo();
        memory.gtt = None;
        let target = select_drain_target(memory);
        assert_eq!(target.kind, PoolKind::Vram);
        assert_eq!(
            target.floor_mib, 450,
            "the floor the CI log showed for the carveout"
        );
        assert!(!target.drained());
    }

    #[test]
    fn a_vram_total_at_the_carveout_limit_is_private_memory() {
        let mut memory = strix_halo();
        memory.vram = Pool {
            total_mib: CARVEOUT_MAX_MIB,
            free_mib: CARVEOUT_MAX_MIB,
        };
        assert_eq!(select_drain_target(memory).kind, PoolKind::Vram);
        assert!(is_carveout(CARVEOUT_MAX_MIB - 1));
        assert!(!is_carveout(CARVEOUT_MAX_MIB));
    }

    /// Run `sample_device_memory` with canned probes, reporting whether the GTT
    /// probe was consulted.
    fn sample(vram: Probe<(u64, u64)>, gtt: Probe<(u64, u64)>) -> (Probe<DeviceMemory>, bool) {
        let mut gtt_asked = false;
        let result = sample_device_memory(
            || vram,
            || {
                gtt_asked = true;
                gtt
            },
        );
        (result, gtt_asked)
    }

    #[test]
    fn a_carveout_host_reads_gtt_and_the_sample_carries_it() {
        let (result, gtt_asked) = sample(Probe::Read((512, 318)), Probe::Read((102_400, 102_360)));
        assert!(gtt_asked, "a carveout host must have its GTT read");
        assert_eq!(result, Probe::Read(strix_halo()));
    }

    #[test]
    fn a_discrete_host_never_makes_the_gtt_call() {
        let (result, gtt_asked) = sample(
            Probe::Read((196_592, 196_309)),
            Probe::Read((1_000_000, 1_000_000)),
        );
        assert!(
            !gtt_asked,
            "GTT must not be read when VRAM is private memory"
        );
        let Probe::Read(memory) = result else {
            panic!("a readable VRAM probe makes a sample: {result:?}");
        };
        assert_eq!(memory.gtt, None);
        assert_eq!(select_drain_target(memory).kind, PoolKind::Vram);
    }

    #[test]
    fn a_carveout_host_whose_gtt_probe_failed_is_unreadable_not_judged_by_vram() {
        // VRAM alone would read as "drained" here (free is above its floor), so
        // falling back to it would hide a GTT-backed allocation still held.
        let (result, gtt_asked) = sample(Probe::Read((512, 500)), Probe::Unreadable);
        assert!(gtt_asked);
        assert_eq!(result, Probe::Unreadable);
    }

    #[test]
    fn a_zero_gtt_total_is_a_failed_reading_not_a_pool() {
        let (result, _) = sample(Probe::Read((512, 500)), Probe::Read((0, 0)));
        assert_eq!(result, Probe::Unreadable);
    }

    #[test]
    fn a_zero_vram_total_is_a_failed_reading_not_a_drained_host() {
        // With no GTT tool the sample would otherwise be judged by a VRAM pool
        // whose floor is zero, which is drained by definition.
        let (result, gtt_asked) = sample(Probe::Read((0, 0)), Probe::Absent);
        assert_eq!(result, Probe::Unreadable);
        assert!(!gtt_asked);
    }

    #[test]
    fn a_carveout_host_with_no_gtt_tool_keeps_the_vram_behaviour() {
        let (result, gtt_asked) = sample(Probe::Read((512, 318)), Probe::Absent);
        assert!(gtt_asked);
        let Probe::Read(memory) = result else {
            panic!("a host without a GTT tool still has a VRAM sample: {result:?}");
        };
        assert_eq!(memory.gtt, None);
        assert_eq!(select_drain_target(memory).kind, PoolKind::Vram);
    }

    #[test]
    fn a_vram_probe_that_is_absent_or_unreadable_is_passed_through_without_asking_for_gtt() {
        assert_eq!(
            sample(Probe::Absent, Probe::Read((102_400, 102_360))),
            (Probe::Absent, false)
        );
        assert_eq!(
            sample(Probe::Unreadable, Probe::Read((102_400, 102_360))),
            (Probe::Unreadable, false)
        );
    }

    #[test]
    fn an_absent_tool_is_no_tooling_but_an_unreadable_one_is_not() {
        assert_eq!(judge(Probe::Absent), Judgement::NoTooling);
        assert_eq!(judge(Probe::Unreadable), Judgement::Unreadable);
    }

    #[test]
    fn a_reading_is_judged_against_the_pool_the_engine_allocates_from() {
        assert_eq!(
            judge(Probe::Read(strix_halo())),
            Judgement::Target(select_drain_target(strix_halo()))
        );
    }

    #[test]
    fn a_fallback_tool_supplies_the_reading_the_first_could_not() {
        assert_eq!(Probe::Unreadable.or_else(|| Probe::Read(7)), Probe::Read(7));
        assert_eq!(Probe::Absent.or_else(|| Probe::Read(7)), Probe::Read(7));
        assert_eq!(
            Probe::Read(1).or_else(|| panic!("the fallback must not run after a reading")),
            Probe::Read(1)
        );
    }

    #[test]
    fn a_fallback_is_absent_only_when_every_tool_was_absent() {
        assert_eq!(Probe::<u8>::Absent.or_else(|| Probe::Absent), Probe::Absent);
        assert_eq!(
            Probe::<u8>::Absent.or_else(|| Probe::Unreadable),
            Probe::Unreadable
        );
        assert_eq!(
            Probe::<u8>::Unreadable.or_else(|| Probe::Absent),
            Probe::Unreadable
        );
    }

    #[test]
    fn parse_keeps_absent_and_unreadable_and_rejects_what_it_cannot_read() {
        assert_eq!(Probe::<u8>::Absent.parse(Some), Probe::Absent);
        assert_eq!(Probe::<u8>::Unreadable.parse(Some), Probe::Unreadable);
        assert_eq!(Probe::Read(3_u8).parse(|n| Some(n + 1)), Probe::Read(4));
        assert_eq!(Probe::Read(3_u8).parse(|_| None::<u8>), Probe::Unreadable);
    }

    #[test]
    fn rocm_smi_csv_gtt_is_read_in_mib() {
        let text = "device,GTT Total Memory (B),GTT Total Used Memory (B)\n\
                    card0,107374182400,41787392\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "GTT Total Memory", "GTT Total Used Memory"),
            Some((102_400, 102_360))
        );
    }

    #[test]
    fn rocm_smi_csv_vram_keeps_its_original_column_match() {
        let text = "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n\
                    card0,536870912,203321344\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "Total Memory", "Total Used Memory"),
            Some((512, 318))
        );
    }

    #[test]
    fn rocm_smi_csv_without_the_pool_is_none() {
        let text = "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n\
                    card0,536870912,203321344\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "GTT Total Memory", "GTT Total Used Memory"),
            None
        );
        assert_eq!(
            parse_rocm_smi_csv_pool("", "Total Memory", "Total Used Memory"),
            None
        );
    }

    #[cfg(unix)]
    fn shell(script: &str, args: &[&str]) -> Command {
        let mut command = Command::new("sh");
        command.arg("-c").arg(script).arg("sh").args(args);
        command
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_that_succeeds_returns_its_stdout() {
        let out = bounded_stdout(shell("echo hello", &[]), Duration::from_secs(30));
        assert_eq!(out, Probe::Read("hello\n".to_owned()));
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_that_exits_non_zero_is_unreadable() {
        let out = bounded_stdout(shell("echo hello; exit 3", &[]), Duration::from_secs(30));
        assert_eq!(out, Probe::Unreadable);
    }

    #[test]
    fn a_probe_whose_program_is_not_installed_is_absent() {
        let out = bounded_stdout(
            Command::new("rocm-cli-test-no-such-smi-tool"),
            Duration::from_secs(30),
        );
        assert_eq!(out, Probe::Absent);
    }

    /// A tool that never returns must cost one sample, not the worker, and the
    /// child must not be left running or unreaped. A zombie still answers
    /// `kill -0`, so a failing `kill -0` shows the child was reaped, not merely
    /// signalled.
    #[cfg(unix)]
    #[test]
    fn a_probe_that_outlives_its_deadline_is_killed_reaped_and_unreadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("pid");
        let started = Instant::now();
        let out = bounded_stdout(
            shell(
                "echo $$ > \"$1\"; exec sleep 60",
                &[pid_file.to_str().expect("utf-8 path")],
            ),
            Duration::from_secs(3),
        );
        let elapsed = started.elapsed();
        assert_eq!(
            out,
            Probe::Unreadable,
            "a timed-out probe is an unreadable sample"
        );
        assert!(
            elapsed < Duration::from_secs(30),
            "returned after {elapsed:?}, far past the 3s deadline"
        );
        // The shell creates the file and then writes the pid; give a loaded
        // runner a moment to finish the write rather than read it half-done.
        let mut pid = String::new();
        for _ in 0..100 {
            pid = std::fs::read_to_string(&pid_file)
                .unwrap_or_default()
                .trim()
                .to_owned();
            if !pid.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(
            !pid.is_empty(),
            "the child wrote its pid before the deadline"
        );
        let still_there = Command::new("kill")
            .args(["-0", &pid])
            .stderr(Stdio::null())
            .status()
            .expect("kill runs")
            .success();
        assert!(!still_there, "child {pid} was left running or unreaped");
    }

    /// Windows runs only the deadline half; the reaped-child check above needs
    /// `kill -0`.
    #[cfg(windows)]
    #[test]
    fn a_probe_that_outlives_its_deadline_returns_unreadable_in_time() {
        let mut command = Command::new("ping.exe");
        command.args(["-n", "60", "127.0.0.1"]);
        let started = Instant::now();
        let out = bounded_stdout(command, Duration::from_millis(1500));
        assert_eq!(out, Probe::Unreadable);
        assert!(started.elapsed() < Duration::from_secs(20));
    }
}
