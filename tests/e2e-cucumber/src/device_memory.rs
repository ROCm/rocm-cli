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
/// far above it (48 GiB and up). The xtask test
/// `the_serve_drain_carveout_limit_is_the_apu_lanes_preflight_floor` fails if
/// this stops matching the lanes' floor.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe<T> {
    /// The tool is not installed.
    Absent,
    /// The tool is there but gave no usable reading.
    Unreadable,
    /// The tool ran and exited successfully, but its output was not understood.
    /// Like `Unreadable` for every decision, it also carries what the tool
    /// printed (see [`output_hint`]) so a log can show what the parser met.
    Unparsed(String),
    Read(T),
}

impl<T> Probe<T> {
    /// Turn a raw reading into a parsed one; a reading `f` rejects is `Unreadable`.
    #[must_use]
    pub fn parse<U>(self, f: impl FnOnce(T) -> Option<U>) -> Probe<U> {
        match self {
            Self::Absent => Probe::Absent,
            Self::Unreadable => Probe::Unreadable,
            Self::Unparsed(hint) => Probe::Unparsed(hint),
            Self::Read(value) => f(value).map_or(Probe::Unreadable, Probe::Read),
        }
    }

    /// Fall back to `next` when this probe produced no reading.
    ///
    /// A reading from either wins. Without one, the result is `Absent` only if
    /// every tool tried was absent; if either tool ran and printed something not
    /// understood the result keeps what it printed (`Unparsed`, the first tool's
    /// if both); otherwise any tool that was present and gave nothing makes it
    /// `Unreadable`.
    #[must_use]
    pub fn or_else(self, next: impl FnOnce() -> Self) -> Self {
        if matches!(self, Self::Read(_)) {
            return self;
        }
        match (self, next()) {
            (_, read @ Self::Read(_)) => read,
            (Self::Absent, Self::Absent) => Self::Absent,
            (Self::Unparsed(hint), _) | (_, Self::Unparsed(hint)) => Self::Unparsed(hint),
            _ => Self::Unreadable,
        }
    }
}

impl Probe<String> {
    /// Parse a tool's stdout with `f`. Output `f` rejects is not `Unreadable`
    /// but `Unparsed`, carrying the first line of it (see [`output_hint`]) so the
    /// log says what the tool printed; `source` names the command.
    #[must_use]
    pub fn parse_text<U>(self, source: &str, f: impl FnOnce(&str) -> Option<U>) -> Probe<U> {
        match self {
            Self::Absent => Probe::Absent,
            Self::Unreadable => Probe::Unreadable,
            Self::Unparsed(hint) => Probe::Unparsed(hint),
            Self::Read(text) => {
                f(&text).map_or_else(|| Probe::Unparsed(output_hint(source, &text)), Probe::Read)
            }
        }
    }
}

/// The longest first line of tool output [`output_hint`] repeats.
pub const HINT_MAX_CHARS: usize = 120;

/// What a tool printed, as one bounded, printable line for a log.
///
/// It is `source`, then the first non-blank line of `text` with anything outside
/// printable ASCII replaced by `?` and cut to [`HINT_MAX_CHARS`] characters.
/// Meant for the header row of a CSV a parser could not match, so the next run
/// shows the real column names.
#[must_use]
pub fn output_hint(source: &str, text: &str) -> String {
    let Some(line) = text.lines().map(str::trim).find(|line| !line.is_empty()) else {
        return format!("{source} printed no output");
    };
    let clean: String = line
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() || c == ' ' {
                c
            } else {
                '?'
            }
        })
        .collect();
    let shown = if clean.chars().count() > HINT_MAX_CHARS {
        format!(
            "{}...",
            clean.chars().take(HINT_MAX_CHARS).collect::<String>()
        )
    } else {
        clean
    };
    format!("{source} printed {shown:?}")
}

/// One memory pool's size and free space, in MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool {
    pub total_mib: u64,
    pub free_mib: u64,
}

impl Pool {
    #[must_use]
    pub const fn new(total_mib: u64, free_mib: u64) -> Self {
        Self {
            total_mib,
            free_mib,
        }
    }
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

    /// `"<free> MiB free of <total> MiB[ GTT], floor <floor> MiB"`. VRAM is the
    /// unlabelled default, so the line keeps the shape it always had on discrete
    /// cards; only the GTT pool is named.
    #[must_use]
    pub fn describe(&self) -> String {
        let pool = match self.kind {
            PoolKind::Vram => "",
            PoolKind::Gtt => " GTT",
        };
        format!(
            "{} MiB free of {} MiB{pool}, floor {} MiB",
            self.free_mib, self.total_mib, self.floor_mib
        )
    }
}

/// The free-memory floor to wait for on a pool of `total_mib`: the
/// model-sized ceiling, but never more than 90% of the pool.
#[must_use]
pub fn required_free_mib(total_mib: u64) -> u64 {
    MAX_FREE_FLOOR_MIB.min(total_mib / 100 * 90)
}

/// What one sample means for the drain wait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Judgement {
    /// No device tool is installed: nothing to wait on (mock/local run).
    NoTooling,
    /// A tool is present but gave no usable reading. This is not "no tooling"
    /// and not "drained": the wait keeps trying.
    Unreadable,
    /// As `Unreadable`, and the tool's output was not understood: what it
    /// printed, for the report (see [`output_hint`]).
    Unparsed(String),
    /// A reading, and the pool and floor it is held to.
    Target(DrainTarget),
}

/// Turn a sample into what the wait does with it.
#[must_use]
pub fn judge(sample: Probe<DeviceMemory>) -> Judgement {
    match sample {
        Probe::Absent => Judgement::NoTooling,
        Probe::Unreadable => Judgement::Unreadable,
        Probe::Unparsed(hint) => Judgement::Unparsed(hint),
        Probe::Read(memory) => Judgement::Target(select_drain_target(memory)),
    }
}

/// Consecutive samples with no reading, from a tool that is installed and has
/// never once produced one, after which the wait stops.
///
/// A tool that never reads is broken or wedged, and waiting out the whole
/// deadline on every serve would add that deadline to each of them. A tool that
/// has read before is not given up on: its next sample may well succeed.
pub const UNREAD_SAMPLES_BEFORE_GIVING_UP: u32 = 3;

/// The least time worth starting a probe with. With less than this left before
/// the deadline the wait ends on what it has rather than spawn a probe that
/// would be killed before it could answer.
pub const MIN_PROBE_BUDGET: Duration = Duration::from_secs(1);

/// How long one probe may run: [`SMI_PROBE_TIMEOUT`], but never past `deadline`.
///
/// A sample makes up to three probes in a row, so without this the last sample
/// could run several probe timeouts beyond the wait's deadline.
#[must_use]
pub fn probe_budget(now: Instant, deadline: Instant) -> Duration {
    SMI_PROBE_TIMEOUT.min(deadline.saturating_duration_since(now))
}

/// How the drain wait ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// No device tool is installed (mock/local run): nothing was waited on.
    NoTooling,
    /// The pool handed back enough memory.
    Drained(DrainTarget),
    /// The deadline passed with the pool still below its floor.
    NotDrained(DrainTarget),
    /// No usable reading at the end: `samples` samples in a row gave none, and
    /// `last_reading` is the most recent one the wait did get, if any. `hint` is
    /// what a tool printed that was not understood, when one did (the latest
    /// such sample, kept until a reading arrives). This is not "not drained":
    /// the harness did not observe that.
    Unreadable {
        last_reading: Option<DrainTarget>,
        samples: u32,
        hint: Option<String>,
    },
}

/// The drain wait's decision state, fed one [`Judgement`] per sample.
#[derive(Debug, Default)]
pub struct DrainWait {
    samples: u32,
    unread_in_a_row: u32,
    last_reading: Option<DrainTarget>,
    hint: Option<String>,
}

impl DrainWait {
    /// Take the next sample. `Some` ends the wait; `None` means keep sampling.
    pub fn observe(&mut self, judgement: Judgement) -> Option<Verdict> {
        let first = self.samples == 0;
        self.samples += 1;
        match judgement {
            Judgement::NoTooling if first => Some(Verdict::NoTooling),
            Judgement::Target(target) => {
                self.unread_in_a_row = 0;
                self.last_reading = Some(target);
                // A reading shows the tool's output is understood after all.
                self.hint = None;
                target.drained().then_some(Verdict::Drained(target))
            }
            Judgement::Unparsed(hint) => {
                self.hint = Some(hint);
                self.unread()
            }
            // A tool that vanishes mid-wait is a sample nobody could read.
            Judgement::NoTooling | Judgement::Unreadable => self.unread(),
        }
    }

    fn unread(&mut self) -> Option<Verdict> {
        self.unread_in_a_row += 1;
        (self.last_reading.is_none() && self.unread_in_a_row >= UNREAD_SAMPLES_BEFORE_GIVING_UP)
            .then(|| self.unreadable())
    }

    /// The verdict when the deadline passes: not drained if the last sample was
    /// a reading, unreadable if it was not.
    #[must_use]
    pub fn timed_out(&self) -> Verdict {
        match self.last_reading {
            Some(target) if self.unread_in_a_row == 0 => Verdict::NotDrained(target),
            _ => self.unreadable(),
        }
    }

    fn unreadable(&self) -> Verdict {
        Verdict::Unreadable {
            last_reading: self.last_reading,
            samples: self.unread_in_a_row,
            hint: self.hint.clone(),
        }
    }
}

/// The one-line `device state:` report for a verdict, after `elapsed_secs` of waiting.
///
/// It says only what the harness observed: "NOT drained" is for a reading below
/// its floor, never for a sample that gave none, and a carveout-sized VRAM pool
/// judged without GTT data says why that cannot show whether the engine's
/// memory came back.
#[must_use]
pub fn describe_verdict(verdict: Verdict, elapsed_secs: u64) -> String {
    match verdict {
        Verdict::NoTooling => "device state: no GPU tooling (mock/local run)".to_owned(),
        Verdict::Drained(target) => format!("device state: drained ({})", target.describe()),
        Verdict::NotDrained(target) => {
            let hint = if target.kind == PoolKind::Vram && is_carveout(target.total_mib) {
                "; only the BIOS carveout was read because no GTT reading was available (is \
                 rocm-smi installed?), so this wait could not follow the memory the engine uses"
            } else {
                ""
            };
            format!(
                "device state: NOT drained after {elapsed_secs}s ({}) — a previous engine may \
                 still be holding the GPU{hint}",
                target.describe()
            )
        }
        Verdict::Unreadable {
            last_reading: None,
            samples,
            hint,
        } => format!(
            "device state: could not read device memory ({samples} samples in {elapsed_secs}s; \
             the GPU tool is installed but gave no usable reading{})",
            hint_suffix(hint.as_deref())
        ),
        Verdict::Unreadable {
            last_reading: Some(target),
            hint,
            ..
        } => format!(
            "device state: could not read device memory at the end of {elapsed_secs}s (last \
             reading: {}{})",
            target.describe(),
            hint_suffix(hint.as_deref())
        ),
    }
}

/// `"; <hint>"` for a report line, or nothing without one.
fn hint_suffix(hint: Option<&str>) -> String {
    hint.map_or_else(String::new, |hint| format!("; {hint}"))
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
pub fn parse_rocm_smi_csv_pool(text: &str, total_column: &str, used_column: &str) -> Option<Pool> {
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
    Some(Pool::new(total / mib, total.saturating_sub(used) / mib))
}

/// Read VRAM from `amd-smi metric -m` output.
///
/// The `TOTAL_VRAM:` and `FREE_VRAM:` lines look like `TOTAL_VRAM: 196592 MB`;
/// the number in front of the unit is taken as MiB. The first GPU's lines win. A
/// zero total parses here; it is [`sample_device_memory`] that treats it as a
/// failed read.
#[must_use]
pub fn parse_amd_smi_vram(text: &str) -> Option<Pool> {
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(name))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|number| number.parse::<u64>().ok())
    };
    Some(Pool::new(field("TOTAL_VRAM:")?, field("FREE_VRAM:")?))
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
///   empty pool) makes the whole sample `Unreadable` (`Unparsed`, with what the
///   tool printed, when it ran but its output was not understood). Judging the tiny
///   carveout instead could report "drained" while the engine still holds its
///   GTT-backed memory. `gpu-preflight.sh` treats a zero GTT total the same way,
///   as an unreadable sample to retry;
/// - no tool that reports GTT at all (`Absent`) leaves `gtt` empty, and the
///   wait judges VRAM as it did before GTT was considered.
pub fn sample_device_memory(
    vram: impl FnOnce() -> Probe<Pool>,
    gtt: impl FnOnce() -> Probe<Pool>,
) -> Probe<DeviceMemory> {
    let vram = match vram() {
        // A zero total is a failed read, not a pool: its floor would be zero and
        // every sample would count as drained.
        Probe::Read(pool) if pool.total_mib > 0 => pool,
        Probe::Read(_) | Probe::Unreadable => return Probe::Unreadable,
        Probe::Unparsed(hint) => return Probe::Unparsed(hint),
        Probe::Absent => return Probe::Absent,
    };
    if !is_carveout(vram.total_mib) {
        return Probe::Read(DeviceMemory { vram, gtt: None });
    }
    match gtt() {
        Probe::Read(pool) if pool.total_mib > 0 => Probe::Read(DeviceMemory {
            vram,
            gtt: Some(pool),
        }),
        Probe::Read(_) | Probe::Unreadable => Probe::Unreadable,
        Probe::Unparsed(hint) => Probe::Unparsed(hint),
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
    fn sample(vram: Probe<Pool>, gtt: Probe<Pool>) -> (Probe<DeviceMemory>, bool) {
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
        let (result, gtt_asked) = sample(
            Probe::Read(Pool::new(512, 318)),
            Probe::Read(Pool::new(102_400, 102_360)),
        );
        assert!(gtt_asked, "a carveout host must have its GTT read");
        assert_eq!(result, Probe::Read(strix_halo()));
    }

    #[test]
    fn a_discrete_host_never_makes_the_gtt_call() {
        let (result, gtt_asked) = sample(
            Probe::Read(Pool::new(196_592, 196_309)),
            Probe::Read(Pool::new(1_000_000, 1_000_000)),
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
        let (result, gtt_asked) = sample(Probe::Read(Pool::new(512, 500)), Probe::Unreadable);
        assert!(gtt_asked);
        assert_eq!(result, Probe::Unreadable);
    }

    #[test]
    fn a_zero_gtt_total_is_a_failed_reading_not_a_pool() {
        let (result, _) = sample(
            Probe::Read(Pool::new(512, 500)),
            Probe::Read(Pool::new(0, 0)),
        );
        assert_eq!(result, Probe::Unreadable);
    }

    #[test]
    fn a_zero_vram_total_is_a_failed_reading_not_a_drained_host() {
        // With no GTT tool the sample would otherwise be judged by a VRAM pool
        // whose floor is zero, which is drained by definition.
        let (result, gtt_asked) = sample(Probe::Read(Pool::new(0, 0)), Probe::Absent);
        assert_eq!(result, Probe::Unreadable);
        assert!(!gtt_asked);
    }

    #[test]
    fn a_carveout_host_with_no_gtt_tool_keeps_the_vram_behaviour() {
        let (result, gtt_asked) = sample(Probe::Read(Pool::new(512, 318)), Probe::Absent);
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
            sample(Probe::Absent, Probe::Read(Pool::new(102_400, 102_360))),
            (Probe::Absent, false)
        );
        assert_eq!(
            sample(Probe::Unreadable, Probe::Read(Pool::new(102_400, 102_360))),
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
            Some(Pool::new(102_400, 102_360))
        );
    }

    #[test]
    fn rocm_smi_csv_vram_keeps_its_original_column_match() {
        let text = "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n\
                    card0,536870912,203321344\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "Total Memory", "Total Used Memory"),
            Some(Pool::new(512, 318))
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

    /// A Strix Halo host whose GTT pool has only `free_mib` MiB free of 102 400.
    fn strix_holding(free_mib: u64) -> DeviceMemory {
        DeviceMemory {
            gtt: Some(Pool {
                total_mib: 102_400,
                free_mib,
            }),
            ..strix_halo()
        }
    }

    /// A sample whose GTT pool is nearly all held: below its 92 160 MiB floor.
    fn undrained() -> Judgement {
        Judgement::Target(select_drain_target(strix_holding(1_000)))
    }

    /// The healthy Strix Halo reading: above its floor.
    fn drained() -> Judgement {
        Judgement::Target(select_drain_target(strix_halo()))
    }

    #[test]
    fn a_target_is_described_with_its_pool_and_only_gtt_is_named() {
        assert_eq!(
            select_drain_target(strix_halo()).describe(),
            "102360 MiB free of 102400 MiB GTT, floor 92160 MiB"
        );
        assert_eq!(
            select_drain_target(mi300x()).describe(),
            "196309 MiB free of 196592 MiB, floor 150000 MiB"
        );
    }

    #[test]
    fn a_drained_sample_ends_the_wait_and_the_line_names_the_pool_it_read() {
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(undrained()), None);
        let verdict = wait.observe(drained()).expect("a drained sample ends it");
        assert_eq!(verdict, Verdict::Drained(select_drain_target(strix_halo())));
        assert_eq!(
            describe_verdict(verdict, 6),
            "device state: drained (102360 MiB free of 102400 MiB GTT, floor 92160 MiB)"
        );
    }

    #[test]
    fn no_tooling_ends_the_wait_on_the_first_sample_only() {
        assert_eq!(
            DrainWait::default().observe(Judgement::NoTooling),
            Some(Verdict::NoTooling)
        );
        assert_eq!(
            describe_verdict(Verdict::NoTooling, 0),
            "device state: no GPU tooling (mock/local run)"
        );
        // A tool that disappears mid-wait is a sample nobody could read, not a
        // reason to report a mock run.
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(undrained()), None);
        for _ in 0..UNREAD_SAMPLES_BEFORE_GIVING_UP * 2 {
            assert_eq!(wait.observe(Judgement::NoTooling), None);
        }
    }

    #[test]
    fn a_tool_that_never_reads_is_given_up_on_and_called_unreadable_not_undrained() {
        let mut wait = DrainWait::default();
        for _ in 1..UNREAD_SAMPLES_BEFORE_GIVING_UP {
            assert_eq!(wait.observe(Judgement::Unreadable), None);
        }
        let verdict = wait
            .observe(Judgement::Unreadable)
            .expect("the last allowed unread sample ends the wait");
        assert_eq!(
            verdict,
            Verdict::Unreadable {
                last_reading: None,
                samples: UNREAD_SAMPLES_BEFORE_GIVING_UP,
                hint: None,
            }
        );
        let line = describe_verdict(verdict, 24);
        assert_eq!(
            line,
            "device state: could not read device memory (3 samples in 24s; the GPU tool is \
             installed but gave no usable reading)"
        );
        assert!(!line.contains("NOT drained"), "{line}");
    }

    #[test]
    fn a_tool_that_has_read_is_waited_on_through_unreadable_samples() {
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(Judgement::Unreadable), None);
        assert_eq!(wait.observe(Judgement::Unreadable), None);
        assert_eq!(wait.observe(undrained()), None);
        for _ in 0..UNREAD_SAMPLES_BEFORE_GIVING_UP * 3 {
            assert_eq!(wait.observe(Judgement::Unreadable), None);
        }
        let verdict = wait.timed_out();
        assert_eq!(
            verdict,
            Verdict::Unreadable {
                last_reading: Some(select_drain_target(strix_holding(1_000))),
                samples: UNREAD_SAMPLES_BEFORE_GIVING_UP * 3,
                hint: None,
            }
        );
        assert_eq!(
            describe_verdict(verdict, 120),
            "device state: could not read device memory at the end of 120s (last reading: \
             1000 MiB free of 102400 MiB GTT, floor 92160 MiB)"
        );
    }

    #[test]
    fn the_deadline_after_a_reading_is_not_drained_and_after_a_miss_it_is_not() {
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(undrained()), None);
        let target = select_drain_target(strix_holding(1_000));
        assert_eq!(wait.timed_out(), Verdict::NotDrained(target));
        assert_eq!(
            describe_verdict(wait.timed_out(), 120),
            "device state: NOT drained after 120s (1000 MiB free of 102400 MiB GTT, floor \
             92160 MiB) — a previous engine may still be holding the GPU"
        );
        // The most recent sample was one nobody could read: that is not a claim
        // about the device.
        assert_eq!(wait.observe(Judgement::Unreadable), None);
        assert_eq!(
            wait.timed_out(),
            Verdict::Unreadable {
                last_reading: Some(target),
                samples: 1,
                hint: None,
            }
        );
    }

    #[test]
    fn an_undrained_carveout_judged_without_gtt_says_it_could_not_follow_the_engines_memory() {
        // The real assembly path: amd-smi read the carveout, no tool reports GTT.
        let (memory, _) = sample(Probe::Read(Pool::new(512, 318)), Probe::Absent);
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(judge(memory)), None);
        let line = describe_verdict(wait.timed_out(), 120);
        assert!(
            line.starts_with(
                "device state: NOT drained after 120s (318 MiB free of 512 MiB, floor 450 MiB)"
            ),
            "{line}"
        );
        assert!(
            line.contains(
                "only the BIOS carveout was read because no GTT reading was available (is \
                 rocm-smi installed?)"
            ),
            "{line}"
        );
        // No such note when the pool judged is the one the engine uses.
        for held in [
            Verdict::NotDrained(select_drain_target(strix_holding(1_000))),
            Verdict::NotDrained(select_drain_target(DeviceMemory {
                vram: Pool {
                    total_mib: 196_592,
                    free_mib: 1_000,
                },
                gtt: None,
            })),
        ] {
            let line = describe_verdict(held, 120);
            assert!(!line.contains("carveout"), "{line}");
        }
    }

    /// What `amd-smi metric -m` prints, per GPU, with the lines the parser reads
    /// among others it must not confuse with them.
    const AMD_SMI_GPU0: &str = "GPU: 0\n    MEM_USAGE:\n        TOTAL_VRAM: 196592 MB\n        \
                                USED_VRAM: 283 MB\n        FREE_VRAM: 196309 MB\n        \
                                TOTAL_VISIBLE_VRAM: 196592 MB\n        FREE_VISIBLE_VRAM: 196309 MB\n";

    #[test]
    fn amd_smi_vram_reads_the_total_and_free_lines_in_mib() {
        assert_eq!(
            parse_amd_smi_vram(AMD_SMI_GPU0),
            Some(Pool::new(196_592, 196_309))
        );
    }

    #[test]
    fn amd_smi_vram_takes_the_first_gpu_and_ignores_indentation_and_units() {
        let two = format!("{AMD_SMI_GPU0}GPU: 1\n  TOTAL_VRAM: 1 MB\n  FREE_VRAM: 2 MB\n");
        assert_eq!(parse_amd_smi_vram(&two), Some(Pool::new(196_592, 196_309)));
        // Tabs, wide gaps, any unit word and a missing space after the colon all
        // read the same way: the number after the name is the value.
        assert_eq!(
            parse_amd_smi_vram("\tTOTAL_VRAM:   512   MiB\n   FREE_VRAM:318\n"),
            Some(Pool::new(512, 318))
        );
        assert_eq!(
            parse_amd_smi_vram("TOTAL_VRAM: 512\r\nFREE_VRAM: 318 MB\r\n"),
            Some(Pool::new(512, 318))
        );
    }

    #[test]
    fn amd_smi_vram_needs_both_fields_as_numbers() {
        assert_eq!(parse_amd_smi_vram("TOTAL_VRAM: 512 MB\n"), None);
        assert_eq!(parse_amd_smi_vram("FREE_VRAM: 318 MB\n"), None);
        assert_eq!(
            parse_amd_smi_vram("TOTAL_VRAM: N/A\nFREE_VRAM: 318 MB\n"),
            None
        );
        assert_eq!(parse_amd_smi_vram("no such fields here"), None);
        assert_eq!(parse_amd_smi_vram(""), None);
        // The visible-VRAM lines are different fields, not the ones asked for.
        assert_eq!(
            parse_amd_smi_vram("TOTAL_VISIBLE_VRAM: 512 MB\nFREE_VISIBLE_VRAM: 318 MB\n"),
            None
        );
    }

    #[test]
    fn a_zero_amd_smi_total_parses_and_is_then_rejected_as_a_failed_read() {
        let zero = "TOTAL_VRAM: 0 MB\nFREE_VRAM: 0 MB\n";
        let parsed = parse_amd_smi_vram(zero).expect("parses");
        assert_eq!(parsed, Pool::new(0, 0));
        let (result, gtt_asked) = sample(Probe::Read(parsed), Probe::Absent);
        assert_eq!(result, Probe::Unreadable);
        assert!(!gtt_asked);
    }

    #[test]
    fn output_hint_is_the_first_nonblank_line_bounded_and_printable() {
        assert_eq!(
            output_hint(
                "rocm-smi gtt",
                "\n  \ndevice,VRAM Total Memory (B)\ncard0,1\n"
            ),
            "rocm-smi gtt printed \"device,VRAM Total Memory (B)\""
        );
        assert_eq!(output_hint("tool", ""), "tool printed no output");
        assert_eq!(output_hint("tool", " \n\t\n"), "tool printed no output");
        // Non-ASCII and control characters are replaced, not repeated.
        assert_eq!(
            output_hint("tool", "GTT \u{1b}[31mT\u{f6}tal\u{0}"),
            "tool printed \"GTT ?[31mT?tal?\""
        );
        // Quotes and backslashes are escaped by the quoting.
        assert_eq!(
            output_hint("tool", "a\"b\\c"),
            "tool printed \"a\\\"b\\\\c\""
        );
        let long = "x".repeat(HINT_MAX_CHARS + 50);
        let hint = output_hint("tool", &long);
        assert_eq!(
            hint,
            format!("tool printed \"{}...\"", "x".repeat(HINT_MAX_CHARS))
        );
        let exact = "y".repeat(HINT_MAX_CHARS);
        assert_eq!(
            output_hint("tool", &exact),
            format!("tool printed \"{exact}\"")
        );
    }

    #[test]
    fn output_a_parser_rejects_is_unparsed_with_what_it_printed_not_plain_unreadable() {
        let header = "device,GTT Total (B),GTT Used (B)\ncard0,107374182400,41787392\n";
        let probe = Probe::Read(header.to_owned()).parse_text("rocm-smi gtt", |text| {
            parse_rocm_smi_csv_pool(text, "GTT Total Memory", "GTT Total Used Memory")
        });
        assert_eq!(
            probe,
            Probe::Unparsed(
                "rocm-smi gtt printed \"device,GTT Total (B),GTT Used (B)\"".to_owned()
            )
        );
        // Right headers still read, and a tool that is absent or failed is not
        // given a hint it has not earned.
        let good = "device,GTT Total Memory (B),GTT Total Used Memory (B)\ncard0,1048576,0\n";
        assert_eq!(
            Probe::Read(good.to_owned()).parse_text("x", |t| {
                parse_rocm_smi_csv_pool(t, "GTT Total Memory", "GTT Total Used Memory")
            }),
            Probe::Read(Pool::new(1, 1))
        );
        assert_eq!(
            Probe::<String>::Absent.parse_text("x", |_| Some(0_u8)),
            Probe::Absent
        );
        assert_eq!(
            Probe::<String>::Unreadable.parse_text("x", |_| Some(0_u8)),
            Probe::Unreadable
        );
    }

    #[test]
    fn a_fallback_tool_that_printed_something_unknown_keeps_what_it_printed() {
        let unparsed = || Probe::<u8>::Unparsed("amd-smi printed \"GPU: 0\"".to_owned());
        assert_eq!(
            unparsed().or_else(|| Probe::Absent),
            unparsed(),
            "an absent fallback leaves the first tool's output visible"
        );
        assert_eq!(Probe::Absent.or_else(unparsed), unparsed());
        assert_eq!(Probe::Unreadable.or_else(unparsed), unparsed());
        assert_eq!(unparsed().or_else(|| Probe::Unreadable), unparsed());
        assert_eq!(unparsed().or_else(|| Probe::Read(7)), Probe::Read(7));
        // The first tool's output wins when both printed something.
        assert_eq!(
            unparsed().or_else(|| Probe::Unparsed("rocm-smi printed \"x\"".to_owned())),
            unparsed()
        );
    }

    #[test]
    fn a_carveout_host_whose_gtt_output_was_not_understood_carries_what_it_printed() {
        let hint = "rocm-smi gtt printed \"device,GTT Total (B)\"";
        let (result, gtt_asked) = sample(
            Probe::Read(Pool::new(512, 318)),
            Probe::Unparsed(hint.to_owned()),
        );
        assert!(gtt_asked);
        assert_eq!(result, Probe::Unparsed(hint.to_owned()));
        // And a VRAM probe's unparsed output passes through without asking for GTT.
        let (result, gtt_asked) = sample(Probe::Unparsed(hint.to_owned()), Probe::Absent);
        assert_eq!(result, Probe::Unparsed(hint.to_owned()));
        assert!(!gtt_asked);
        assert_eq!(judge(result), Judgement::Unparsed(hint.to_owned()));
    }

    #[test]
    fn what_a_tool_printed_reaches_the_final_line_when_it_never_reads() {
        let hint = "rocm-smi --showmeminfo gtt --csv printed \"device,GTT Total (B)\"";
        let mut wait = DrainWait::default();
        for _ in 1..UNREAD_SAMPLES_BEFORE_GIVING_UP {
            assert_eq!(wait.observe(Judgement::Unparsed(hint.to_owned())), None);
        }
        let verdict = wait
            .observe(Judgement::Unparsed(hint.to_owned()))
            .expect("the third unread sample ends the wait");
        assert_eq!(
            describe_verdict(verdict, 9),
            "device state: could not read device memory (3 samples in 9s; the GPU tool is \
             installed but gave no usable reading; rocm-smi --showmeminfo gtt --csv printed \
             \"device,GTT Total (B)\")"
        );
    }

    #[test]
    fn what_a_tool_printed_is_kept_through_later_plain_failures_and_dropped_by_a_reading() {
        let hint = "rocm-smi printed \"hdr\"";
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(Judgement::Unparsed(hint.to_owned())), None);
        // A later timeout does not erase the evidence of the earlier sample.
        assert_eq!(wait.observe(Judgement::Unreadable), None);
        let Verdict::Unreadable { hint: kept, .. } = wait.timed_out() else {
            panic!("two unread samples are unreadable at the deadline");
        };
        assert_eq!(kept.as_deref(), Some(hint));
        // A reading shows the output is understood after all.
        assert_eq!(wait.observe(undrained()), None);
        assert_eq!(wait.observe(Judgement::Unreadable), None);
        let Verdict::Unreadable {
            hint: after,
            last_reading,
            ..
        } = wait.timed_out()
        else {
            panic!("the last sample was unreadable");
        };
        assert_eq!(after, None);
        assert!(last_reading.is_some());
    }

    #[test]
    fn a_hint_follows_the_last_reading_on_the_line_when_the_deadline_finds_it_unreadable() {
        let mut wait = DrainWait::default();
        assert_eq!(wait.observe(undrained()), None);
        assert_eq!(
            wait.observe(Judgement::Unparsed("t printed \"h\"".to_owned())),
            None
        );
        assert_eq!(
            describe_verdict(wait.timed_out(), 120),
            "device state: could not read device memory at the end of 120s (last reading: \
             1000 MiB free of 102400 MiB GTT, floor 92160 MiB; t printed \"h\")"
        );
    }

    #[test]
    fn a_probe_gets_the_probe_timeout_or_the_time_left_whichever_is_less() {
        let now = Instant::now();
        assert_eq!(
            probe_budget(now, now + Duration::from_secs(100)),
            SMI_PROBE_TIMEOUT
        );
        assert_eq!(
            probe_budget(now, now + Duration::from_secs(4)),
            Duration::from_secs(4)
        );
        assert_eq!(
            probe_budget(now + Duration::from_secs(5), now),
            Duration::ZERO
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
