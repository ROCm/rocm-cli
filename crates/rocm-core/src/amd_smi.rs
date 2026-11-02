// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Locating and launching the `amd-smi` binary.
//!
//! Location prefers the binary shipped in the newest managed ROCm SDK runtime
//! recorded in the registry, then falls back to the home-directory venv and
//! legacy locations, then to the bare `amd-smi` name for a `PATH` lookup.
//!
//! Launching goes through [`amd_smi_json`], which is the one supported way for
//! a synchronous caller to start the binary: it runs the `/dev/kfd` pre-flight
//! that keeps `amd-smi` away from a device that can park a process in
//! uninterruptible sleep, and the pre-flight protects nothing unless every
//! launch site goes through it.

use std::ffi::{OsStr, OsString};
#[cfg(target_os = "linux")]
use std::fs;
use std::path::{Path, PathBuf};
// `Command` is this module's own — [`amd_smi_json_with`] builds one — but
// nothing here runs it: the wait, the kill and both pipe drains belong to
// [`crate::process::run_with_timeout`], which is why no production `Stdio`
// import comes with it (the helper sets the child's stdio itself, and its docs
// say callers should not). `Stdio` and `thread` are left to the KFD open probe
// and its Linux-only test, hence the `cfg`s: an unused import is a hard error
// on the Windows lane's `-D warnings` build.
use std::process::Command;
#[cfg(all(test, target_os = "linux"))]
use std::process::Stdio;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::managed_runtime::{
    TheRockFamilyManifest, managed_sdk_tool_path, managed_therock_environment_records,
    sort_records_newest_install_first,
};
use crate::{default_data_dir, runtime_home_dir};
#[cfg(target_os = "linux")]
use crate::{has_usable_amd_gpu, host_path, is_wsl_host};

pub fn resolve_amd_smi_binary() -> OsString {
    if let Some(path) = default_data_dir()
        .map(|data_dir| data_dir.join("runtimes").join("registry"))
        .and_then(|registry_dir| resolve_amd_smi_binary_in_registry(&registry_dir))
    {
        return path;
    }
    resolve_amd_smi_binary_in_home(runtime_home_dir().as_deref())
}

/// Locate `amd-smi` inside the bin directories of the newest managed ROCm SDK
/// runtime recorded in the registry. The binary ships with the TheRock wheel
/// (under the SDK `bin_path` and/or the venv `install_root/bin`) and is not on
/// `PATH`, so the home-directory fallbacks below never find it.
fn resolve_amd_smi_binary_in_registry(registry_dir: &Path) -> Option<OsString> {
    let mut records = managed_therock_environment_records(registry_dir);
    sort_records_newest_install_first(&mut records);
    records.into_iter().find_map(|(_, record)| {
        amd_smi_bin_dirs_for_record(&record)
            .iter()
            .find_map(|bin_dir| managed_sdk_tool_path(bin_dir, "amd-smi"))
            .map(PathBuf::into_os_string)
    })
}

fn amd_smi_bin_dirs_for_record(record: &TheRockFamilyManifest) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(sdk) = record.rocm_sdk.as_ref() {
        if let Some(bin_path) = sdk.bin_path.as_ref() {
            dirs.push(bin_path.clone());
        }
        for bin_path in &sdk.bin_paths {
            if !dirs.contains(bin_path) {
                dirs.push(bin_path.clone());
            }
        }
    }
    if let Some(install_root) = record.install_root.as_ref() {
        let install_bin = install_root.join("bin");
        if !dirs.contains(&install_bin) {
            dirs.push(install_bin);
        }
    }
    dirs
}

fn resolve_amd_smi_binary_in_home(home_dir: Option<&Path>) -> OsString {
    if let Some(home_dir) = home_dir {
        let venv_bin = home_dir.join("rocm_venvs").join("default").join("bin");
        if let Some(path) = managed_sdk_tool_path(&venv_bin, "amd-smi") {
            return path.into_os_string();
        }

        let legacy_bin = home_dir.join(".rocm").join("bin");
        if let Some(path) = managed_sdk_tool_path(&legacy_bin, "amd-smi") {
            return path.into_os_string();
        }
    }

    "amd-smi".into()
}

/// Device node of the AMD kernel fusion driver. Bare-metal Linux only: no other
/// platform exposes one, and WSL — which *is* `target_os = "linux"` — is not an
/// exception to hand-wave past. It reaches the GPU through `/dev/dxg` and the
/// Windows host driver and has no `/dev/kfd` at all, which is why the WSL
/// diagnosis catalog skips every bare-metal driver check (see `docs/wsl.md` and
/// [`crate::examine::WslFacts`]).
const KFD_DEVICE: &str = "/dev/kfd";

/// Name of the E2E-only seam that moves the pre-flight's device node, so the
/// gate's two branches are both reachable from a Gherkin scenario.
///
/// Honoured only by a build carrying the `e2e-test-hooks` feature, which no
/// released artefact does; see [`preflight_kfd_device`].
#[cfg(feature = "e2e-test-hooks")]
pub(crate) const TEST_KFD_DEVICE_ENV: &str = "ROCM_E2E_KFD_DEVICE";

/// The device node [`amd_smi_json`] runs its pre-flight against.
///
/// [`KFD_DEVICE`] in every shipped build. An `e2e-test-hooks` build lets
/// [`TEST_KFD_DEVICE_ENV`] name a different path, which is what makes the gate
/// testable from the outside at all.
///
/// Why a seam rather than the existing [`crate::hardware_root`] re-rooting:
/// [`kfd_readable`] deliberately keeps the *real* node's veto, so a simulated
/// root can only ever make the gate decline, never allow. That asymmetry is the
/// invariant that stops a simulated root launching `amd-smi` on a machine with
/// no GPU, and it is correct — but it also means no simulated root can exercise
/// the *allowed* branch on a GPU-less runner, which is where the per-PR lane
/// runs. A path override sidesteps that without touching the invariant: the
/// production build has no override, so the veto and the real device are all
/// there is.
///
/// Safe to be an override rather than a bypass. It relocates the probe; it does
/// not skip it. A build with the feature on and the variable unset — every unit
/// test in this crate, and every E2E scenario that does not set it — gates on
/// `/dev/kfd` exactly as production does.
#[cfg(feature = "e2e-test-hooks")]
fn preflight_kfd_device() -> PathBuf {
    std::env::var_os(TEST_KFD_DEVICE_ENV).map_or_else(|| PathBuf::from(KFD_DEVICE), Into::into)
}

/// Production counterpart of [`preflight_kfd_device`]: the constant, with no
/// environment consulted. See that function for why the seam exists.
#[cfg(not(feature = "e2e-test-hooks"))]
fn preflight_kfd_device() -> PathBuf {
    PathBuf::from(KFD_DEVICE)
}

/// Whether `/dev/kfd` can be opened for reading — the pre-flight every
/// `amd-smi` launch goes through, and it is **mandatory** before every launch.
///
/// What it actually detects is **inaccessible**, not **unhealthy**. `kfd_open`
/// refuses when the node is absent, when permission is denied, when the driver
/// exposes no KFD GPU node, and while a reset is in progress; a driver that is
/// merely version-mismatched or wedged after a fault still opens. Those are the
/// states in which `amd-smi` blocks in uninterruptible kernel sleep — no signal
/// escapes, so the process survives both Ctrl-C and `SIGKILL`, and no timeout
/// rescues it, because killing a D-state child is a no-op. This gate removes
/// the reachable half of that hazard; it does not claim to see the rest.
///
/// The probe does not assume `kfd_open` returns promptly. An `open` that blocks
/// — whatever the reason, including a state this gate has not anticipated —
/// is bounded by [`KFD_OPEN_TIMEOUT`] and read as "not readable", so the gate
/// declines rather than inheriting the stall it exists to prevent. See
/// [`device_readable_within_budget`].
///
/// Off Linux there is no such device and no such kernel state, so the check is
/// `true` without probing anything. That wave-through is load-bearing rather
/// than pedantic: `amd-smi.exe` is a supported Windows binary (see
/// [`crate::managed_runtime::managed_sdk_tool_path`]), so a check that simply
/// failed to find `/dev/kfd` there would silently disable GPU detection on
/// every Windows host.
///
/// WSL needs a wave-through for the same reason and cannot get it from `cfg`:
/// it compiles as Linux yet has no `/dev/kfd` (see [`KFD_DEVICE`]), so the real
/// open fails on every WSL host. But WSL identity alone is not the condition.
/// `b4f98bc2` replaced exactly such a bare identity check in the dashboard
/// pre-flight with [`has_usable_amd_gpu`], because `/dev/dxg` is vendor-agnostic
/// and a WSL host whose ROCDXG plumbing is not ready is one where `amd-smi`
/// still hangs — a different hazard from the KFD one, which an "is this WSL"
/// test cannot tell apart from a healthy host. So the substitution is
/// `is_wsl_host() && has_usable_amd_gpu()`, the same conjunction
/// `apps/rocm/src/dash.rs::gpu_reachable_for_preflight` feeds the dashboard, and
/// `serve`/`rocmd`/the dashboard keep the single cross-surface verdict that
/// commit established. On bare metal `is_wsl_host()` short-circuits, so
/// [`has_usable_amd_gpu`] is never consulted and behaviour is a plain
/// `/dev/kfd` open.
///
/// Hardware-probe routing (see [`crate::hardware_root`]): like the dashboard
/// pre-flight, this reads `host_path("/dev/kfd")` *and* keeps the real node's
/// veto. A simulated E2E root can therefore make the gate decline on a machine
/// that has a GPU, but can never make it allow a launch on a machine that does
/// not.
#[cfg(target_os = "linux")]
fn kfd_readable(device: &Path) -> bool {
    kfd_readable_with(
        // `&&` short-circuits: `has_usable_amd_gpu` runs on WSL only.
        is_wsl_host() && has_usable_amd_gpu(),
        device,
        &host_path(device),
    )
}

/// [`kfd_readable`] with its two non-syscall inputs injected, so the
/// wave-through and the simulated-root routing can both be pinned from a test
/// on an ordinary Linux host rather than only on WSL or under an E2E root.
///
/// `wsl_gpu_reachable` may only ever be the WSL conjunction described on
/// [`kfd_readable`]; it substitutes for the device open entirely, so widening it
/// to any fail-open verdict would disable the gate on bare metal.
#[cfg(target_os = "linux")]
fn kfd_readable_with(wsl_gpu_reachable: bool, real_device: &Path, host_device: &Path) -> bool {
    wsl_gpu_reachable
        || (device_readable(real_device)
            && (host_device == real_device || device_readable(host_device)))
}

/// How long the pre-flight waits for the device `open` itself before giving up
/// on it and declining the launch.
///
/// The gate exists to avoid spawning `amd-smi` against a device that can park a
/// process in uninterruptible sleep. Performing the probe with a plain blocking
/// `open` would make the gate's own safety rest on an *assumption* — that
/// `kfd_open` always returns promptly, and that only amd-smi's later ioctls can
/// enter D-state — and that assumption cannot be verified against a
/// reset-in-progress device from any test. If it were ever wrong, the gate
/// would have relocated the hazard into a synchronous call with *less*
/// protection than the spawn it replaced, since there is no child to kill.
///
/// Bounding the open removes the need to be right about that: a `/dev/kfd` open
/// that has not returned in this long is treated as "not readable", which is the
/// declining answer, so the unverifiable case fails in the safe direction. Short
/// because a healthy `kfd_open` is a handful of microseconds and the only thing
/// a longer wait buys is a longer stall on a host that is already unhealthy.
///
/// **It is additive to every [`amd_smi_json`] `timeout`, not inside it.** The
/// pre-flight finishes before anything is spawned, so a caller that passes a
/// 10s timeout can wait up to 12s in the worst case. Callers that document a
/// worst-case latency have to include this; see
/// `apps/rocm/src/gpu_policy.rs::AMD_SMI_SERVE_PROBE_TIMEOUT`.
#[cfg(target_os = "linux")]
const KFD_OPEN_TIMEOUT: Duration = Duration::from_secs(2);

/// How many device-open probes may be parked at once before the pre-flight
/// starts declining outright.
///
/// Every probe that overruns [`KFD_OPEN_TIMEOUT`] leaves a thread blocked in
/// `open` until the kernel lets it go, which against the wedged device this gate
/// exists to refuse may be never. `rocmd`'s telemetry watcher and `serve`'s
/// probes both poll, so without a cap a host stuck in that state accumulates one
/// parked thread per poll until the process runs out of its thread budget.
///
/// Small on purpose. The only thing a larger budget buys is more parked threads
/// on a host that is already answering "not readable"; a healthy `kfd_open`
/// returns in microseconds, so even one slot is enough for every caller this
/// crate has, and 8 leaves headroom for concurrent callers without ever
/// approaching a thread limit.
#[cfg(target_os = "linux")]
const MAX_INFLIGHT_KFD_PROBES: usize = 8;

/// The process-wide budget every production probe claims from.
#[cfg(target_os = "linux")]
static KFD_PROBE_BUDGET: ProbeBudget = ProbeBudget::new(MAX_INFLIGHT_KFD_PROBES);

/// A cap on concurrently parked device-open probes.
///
/// A counter rather than a `Semaphore` because the release is not scoped to the
/// caller: a slot is held by the probe *thread*, which outlives the call that
/// started it whenever the probe overran, and is freed if and when the kernel
/// finally returns from `open`. So a host that recovers goes back to its full
/// budget without anything having to notice, while one that never does keeps
/// answering from the declining branch rather than spawning threads forever.
#[cfg(target_os = "linux")]
struct ProbeBudget {
    in_flight: AtomicUsize,
    limit: usize,
}

#[cfg(target_os = "linux")]
impl ProbeBudget {
    const fn new(limit: usize) -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            limit,
        }
    }

    /// Take one slot, or `None` when the budget is already spent.
    ///
    /// `fetch_update` rather than `fetch_add` + compensating decrement: the
    /// latter is momentarily over the limit, which two callers racing at the
    /// boundary can both observe.
    fn claim(&'static self) -> Option<ProbeSlot> {
        self.in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |in_flight| {
                (in_flight < self.limit).then_some(in_flight + 1)
            })
            .ok()
            .map(|_| ProbeSlot(self))
    }
}

/// One claimed slot of a [`ProbeBudget`], released on drop.
///
/// Moved *into* the probe thread so the release happens when the `open` returns
/// rather than when the caller stops waiting — the whole point of the cap is to
/// count threads that are still parked.
#[cfg(target_os = "linux")]
struct ProbeSlot(&'static ProbeBudget);

#[cfg(target_os = "linux")]
impl Drop for ProbeSlot {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Release);
    }
}

/// Whether `device` can be opened for reading, giving up after
/// [`KFD_OPEN_TIMEOUT`].
#[cfg(target_os = "linux")]
fn device_readable(device: &Path) -> bool {
    device_readable_within_budget(device, KFD_OPEN_TIMEOUT, &KFD_PROBE_BUDGET)
}

/// [`device_readable`] with the bound and the budget injected, so a test can pin
/// the timeout path against a device that really does block in `open` (a FIFO
/// with no writer) without stalling the suite for the production bound, and can
/// exhaust a budget of its own without spending the production one.
///
/// The `open` runs on its own thread and the caller waits on a channel, because
/// there is no portable way to interrupt a blocking `open` once it has entered
/// the kernel. A probe that overruns leaves that thread parked — the same
/// trade-off [`crate::process::run_with_timeout`] makes when it abandons rather
/// than joins its readers, and for the same reason: one leaked thread is
/// strictly cheaper than a blocked caller, and a device that can park a thread
/// forever is exactly the device this whole gate exists to refuse. The opened
/// handle is dropped on the probe thread, so a late success closes its own
/// descriptor rather than leaking one.
///
/// # Why both failure branches decline
///
/// The parked threads are bounded by `budget`, and a probe that cannot claim a
/// slot returns `false` without opening anything. A thread that cannot be
/// spawned at all returns `false` too.
///
/// Both of those used to fall back to a direct blocking `open` on the calling
/// thread, on the reasoning that refusing a launch because the process was out
/// of threads would be a worse failure than the one being guarded against. That
/// reasoning does not survive contact with the escalation it has to handle. The
/// states in which this process runs out of threads are not independent of the
/// state this gate refuses: the way a thread budget is exhausted *here* is
/// precisely one parked `open` per poll against a wedged device, so the fallback
/// fires exactly when the device is at its most dangerous, and performs the
/// unbounded blocking `open` the bound exists to prevent — on the caller's own
/// thread, where there is nothing left to kill. A refusal is recoverable and
/// visible: `amd_smi_json` says it declined and names the device, the GPU-aware
/// paths degrade to service state alone, and the next poll on a recovered host
/// finds its slots freed and allows the launch again. A hang is neither.
///
/// So both branches answer `false` — "not readable" — which is the same
/// declining answer the timeout gives, for the same reason.
#[cfg(target_os = "linux")]
fn device_readable_within_budget(
    device: &Path,
    timeout: Duration,
    budget: &'static ProbeBudget,
) -> bool {
    let Some(slot) = budget.claim() else {
        return false;
    };
    let (sender, receiver) = std::sync::mpsc::channel();
    let probe_path = device.to_path_buf();
    let spawned = thread::Builder::new()
        .name("rocm-kfd-open-probe".to_owned())
        .spawn(move || {
            // Held until the `open` returns, however long that is, so the budget
            // counts threads that are still parked rather than calls still
            // waiting. A failed spawn drops the closure, and with it the slot.
            let _slot = slot;
            let readable = fs::OpenOptions::new().read(true).open(&probe_path).is_ok();
            // The receiver is gone whenever the probe overran; dropping the
            // send result is the whole handling.
            let _ = sender.send(readable);
        })
        .is_ok();
    if !spawned {
        return false;
    }
    // Both error arms — timeout and a probe thread that died without sending —
    // mean "no readable answer", which the gate must read as not readable.
    receiver.recv_timeout(timeout).unwrap_or(false)
}

/// Off-Linux counterpart of [`kfd_readable`]: see that function's docs for why
/// the wave-through is load-bearing rather than a stub.
///
/// `const` because it has to be. The Windows lane builds with `-D warnings` and
/// runs clippy, so `clippy::missing_const_for_fn` makes this a hard compile
/// error there while the Linux lane — where the real body opens a device and
/// cannot be `const` — never sees the lint. A `cfg`-selected pair of functions
/// is linted per branch, not per name.
#[cfg(not(target_os = "linux"))]
const fn kfd_readable(_device: &Path) -> bool {
    true
}

/// Run `amd-smi <args>` and parse its stdout as JSON, declining to launch at all
/// unless the `/dev/kfd` pre-flight in [`kfd_readable`] passes.
///
/// This is the supported way to launch `amd-smi` from a synchronous caller. The
/// pre-flight has to happen at every launch site or it protects nothing, and
/// routing the launch through one function is what keeps that true as sites are
/// added.
///
/// `timeout` is defence in depth, not the mitigation: see
/// [`crate::process::run_with_timeout`], which kills the timed-out child and
/// waits on it — neither of which dislodges a process in uninterruptible sleep,
/// and the `wait` of an undislodgeable one does not return either. It bounds
/// the ordinary failures — a slow or wedged-but-signalable `amd-smi` — and the
/// pre-flight above, not the timeout, is what keeps the D-state case away.
///
/// # Latency
///
/// `timeout` is not the worst case for the call. The pre-flight runs first and
/// is bounded separately by [`KFD_OPEN_TIMEOUT`] (2s on Linux), so the worst
/// case is `KFD_OPEN_TIMEOUT + timeout` — 12s for a 10s `timeout`, 4s for the
/// 2s one `rocmd`'s watcher passes. A caller that documents its own worst case
/// has to add the pre-flight's bound to the one it passes here.
pub fn amd_smi_json(args: &[&str], timeout: Duration) -> Result<serde_json::Value> {
    amd_smi_json_with(
        &resolve_amd_smi_binary(),
        // [`preflight_kfd_device`], not the constant: identical in every shipped
        // build, and the one seam an E2E scenario can move so that both sides of
        // the gate are reachable from the lane that runs per PR.
        &preflight_kfd_device(),
        args,
        timeout,
    )
}

fn amd_smi_json_with(
    binary: &OsStr,
    kfd_device: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<serde_json::Value> {
    if !kfd_readable(kfd_device) {
        bail!(
            "declined to launch `amd-smi {}`: {} could not be opened for reading (absent, \
             permission denied, no KFD GPU node, or a reset in progress), and amd-smi can \
             block unkillably in uninterruptible sleep against a device in that state",
            args.join(" "),
            kfd_device.display()
        );
    }

    let mut command = Command::new(binary);
    // No stdio here: `run_with_timeout` sets null stdin and piped stdout/stderr
    // itself, since capturing both is its contract, and its docs say callers
    // should not set them.
    command.args(args);
    let output = crate::process::run_with_timeout(command, timeout, "amd-smi")
        .with_context(|| format!("failed to launch amd-smi {}", args.join(" ")))?;

    if !output.status.success() {
        // Truncated, not dropped. This message reaches `rocmd`'s snapshot note
        // and from there the JSONL logs, so the quote has to be bounded at the
        // point that knows these characters are raw child output — the same
        // `MAX_QUOTED_OUTPUT_CHARS` `run_with_timeout` bounds its own quote
        // with, shared rather than re-declared so the two cannot drift. The
        // head is where a failing `amd-smi` says what went wrong, and the
        // command is named before it either way.
        //
        // The stderr-then-stdout precedence is `command_failure_detail`'s, taken
        // from the shared helper rather than re-spelled here; only the
        // both-blank fallback differs, naming the exit status instead of the
        // generic "no output".
        bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            crate::examine::truncate_to_chars(
                crate::command_failure_detail_or_else(&output, || format!(
                    "exit status {}",
                    output.status
                )),
                crate::process::MAX_QUOTED_OUTPUT_CHARS,
            )
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse amd-smi {} json", args.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::unix_time_millis;
    use anyhow::Result;
    use std::fs;

    /// Serialise a test that reads a device node through [`kfd_readable`].
    ///
    /// The gate resolves its device through `host_path`, so in an
    /// `e2e-test-hooks` build another test setting `ROCM_CLI_TEST_HOST_ROOT`
    /// process-wide would re-root this test's own tempdir device out from under
    /// it, and the gate would decline a device the test had just created. This
    /// is the same lock `fix.rs` holds for the same reason; without the feature
    /// it is uncontended and costs nothing.
    #[cfg(target_os = "linux")]
    fn host_root_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The gate has to stop the *spawn*, not merely report a failure after the
    /// fact: the launch that already happened is the launch that hangs. The
    /// fake `amd-smi` leaves a marker file, so its absence is direct evidence
    /// nothing ran — and running the same fixture against a readable device
    /// proves the fake would otherwise have executed, which is what stops the
    /// blocked case from passing vacuously.
    ///
    /// Linux-only because `kfd_readable` is deliberately a no-op elsewhere;
    /// off Linux there is no gate to exercise.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_gate_stops_amd_smi_before_it_is_spawned() -> Result<()> {
        // WSL compiles as Linux but is waved through deliberately (see
        // `the_kfd_preflight_waves_through_wsl_hosts_only_when_the_gpu_is_reachable`),
        // so there is no gate to exercise on such a host and the blocked case
        // could never hold.
        if is_wsl_host() {
            return Ok(());
        }
        let _host_root = host_root_guard();

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-gate-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let marker = temp_root.join("amd-smi-ran");
        let fake_amd_smi = temp_root.join("amd-smi");
        // Not `fs::write` + `set_permissions`: this process is about to execute
        // the file, and a write fd leaked into a parallel test's fork would make
        // the first run fail with `ETXTBSY`. See `fake_executable`.
        crate::fake_executable::write_fake_executable(
            &fake_amd_smi,
            format!("#!/bin/sh\ntouch '{}'\necho '{{}}'\n", marker.display()),
        )?;
        let readable_device = temp_root.join("stand-in-kfd");
        fs::write(&readable_device, b"")?;

        let timeout = Duration::from_secs(10);
        let args = ["list", "--json"];

        let blocked = amd_smi_json_with(
            fake_amd_smi.as_os_str(),
            &temp_root.join("absent-kfd"),
            &args,
            timeout,
        );
        let ran_while_blocked = marker.exists();

        let allowed = amd_smi_json_with(fake_amd_smi.as_os_str(), &readable_device, &args, timeout);
        let ran_while_allowed = marker.exists();

        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            blocked.is_err(),
            "an unreadable KFD device must stop the launch"
        );
        assert!(
            !ran_while_blocked,
            "amd-smi was spawned even though the pre-flight failed"
        );
        assert!(
            allowed.is_ok(),
            "a readable device must let the launch through: {allowed:?}"
        );
        assert!(
            ran_while_allowed,
            "the fake amd-smi never ran at all, so the blocked case proves nothing"
        );
        Ok(())
    }

    /// No host off Linux has a `/dev/kfd`, so the pre-flight must wave the
    /// launch through instead of reading the absent node as "unsafe".
    /// `amd-smi.exe` is a supported Windows binary, so treating its absence as
    /// a failure would disable GPU detection on every Windows host.
    #[test]
    #[cfg(not(target_os = "linux"))]
    fn the_kfd_preflight_waves_through_hosts_that_have_no_kfd() {
        assert!(kfd_readable(Path::new("/definitely/not/a/device/node")));
    }

    /// WSL has no `/dev/kfd` either, but unlike Windows it cannot be waved
    /// through by `cfg`: it compiles as Linux, so the real open runs, always
    /// fails, and declines every routed `amd-smi` launch on a supported
    /// platform — silently, since both `serve` probes drop the error with
    /// `.ok()?`. Same wave-through as the off-Linux case above, decided at
    /// runtime instead.
    ///
    /// The *conjunction* is what is pinned, not WSL identity: a WSL host whose
    /// ROCDXG plumbing is not ready is one where `amd-smi` still hangs, which is
    /// why `b4f98bc2` replaced a bare identity check with `has_usable_amd_gpu()`
    /// in the dashboard pre-flight. The unready-WSL case below is the one that
    /// distinguishes this from that mistake; the bare-metal case is the control
    /// that stops the whole test passing with the gate deleted.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_preflight_waves_through_wsl_hosts_only_when_the_gpu_is_reachable() {
        let absent = Path::new("/definitely/not/a/device/node");
        assert!(
            kfd_readable_with(true, absent, absent),
            "a ready WSL host has no /dev/kfd, so its absence must not decline the launch"
        );
        assert!(
            !kfd_readable_with(false, absent, absent),
            "an unready WSL host — or a bare-metal one with no readable KFD — must be declined"
        );
    }

    /// The simulated-root routing, which `kfd_readable` applies to the real
    /// `/dev/kfd` on the production path.
    ///
    /// Both halves are load-bearing and they fail in opposite directions. A
    /// simulated root that hides the device must decline (so an E2E lane can
    /// manufacture the declined branch on a machine that does have a GPU), and
    /// a simulated root that *invents* a device must not allow (so no test hook
    /// can ever start a real `amd-smi` on a machine that has no GPU — the veto
    /// `hardware_root`'s module contract requires of this probe).
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_gate_honours_a_simulated_root_but_keeps_the_real_device_veto() -> Result<()> {
        let _host_root = host_root_guard();
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-root-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let present = temp_root.join("present-kfd");
        fs::write(&present, b"")?;
        let absent = temp_root.join("absent-kfd");

        let real_present_root_absent = kfd_readable_with(false, &present, &absent);
        let real_absent_root_present = kfd_readable_with(false, &absent, &present);
        let both_present = kfd_readable_with(false, &present, &present);
        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            !real_present_root_absent,
            "a simulated root that hides the device must decline the launch"
        );
        assert!(
            !real_absent_root_present,
            "a simulated root must never be able to start amd-smi on a host with no real device"
        );
        assert!(
            both_present,
            "the ordinary case must still pass, or the two above prove nothing"
        );
        Ok(())
    }

    /// The gate's own probe must not be able to hang.
    ///
    /// `kfd_readable` was reviewed as resting on an unverified assumption: that
    /// `open("/dev/kfd")` always returns promptly, so that only amd-smi's later
    /// ioctls can enter uninterruptible sleep. If that were ever wrong the gate
    /// would have moved the hazard into a synchronous call with *less*
    /// protection than the spawn it replaced — no child to kill, no timeout.
    /// Rather than assert the assumption, the probe is bounded; this pins that.
    ///
    /// A FIFO is the fixture that makes the blocking case real rather than
    /// simulated: `open(O_RDONLY)` on a FIFO with no writer blocks in the kernel
    /// indefinitely, which is the observable shape of the hazard. The deadline
    /// assertion is what distinguishes a bounded probe from an unbounded one —
    /// without it, deleting the timeout would hang the suite rather than fail
    /// this test, and a hang is not a readable failure. The readable control in
    /// the same test stops the whole thing passing vacuously, e.g. if
    /// `device_readable_within_budget` were mutated to return `false`
    /// unconditionally.
    ///
    /// Runs against a budget of its own rather than [`KFD_PROBE_BUDGET`]: it
    /// deliberately parks a probe thread for the lifetime of the test binary,
    /// and spending a production slot would make every later test in this
    /// process share a quietly smaller budget.
    ///
    /// The FIFO is created by shelling out to `mkfifo(1)` rather than calling
    /// `libc::mkfifo`: the workspace lints deny `unsafe_code`, and one test
    /// fixture does not justify an exception. The tool is coreutils, so it is
    /// present wherever this test runs; a host without it skips rather than
    /// fails, since a missing fixture is not evidence about the probe.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_probe_declines_a_device_whose_open_blocks_instead_of_hanging() -> Result<()> {
        static BUDGET: ProbeBudget = ProbeBudget::new(MAX_INFLIGHT_KFD_PROBES);

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-open-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let Some(fifo) = make_fifo(&temp_root, "blocking-kfd") else {
            let _ = fs::remove_dir_all(&temp_root);
            return Ok(());
        };

        let probe_timeout = Duration::from_millis(300);
        // The probe runs on a thread the test watches with its own, much looser
        // deadline, rather than being called inline. A mutant that drops the
        // bound from `device_readable_within_budget` would block forever on this
        // FIFO; called inline that would hang the whole test binary, and a hang
        // is a CI timeout rather than a readable failure. Watched, it fails here
        // with a message naming the cause.
        let (verdict_tx, verdict_rx) = std::sync::mpsc::channel();
        thread::Builder::new()
            .name("kfd-open-probe-test".to_owned())
            .spawn(move || {
                let _ =
                    verdict_tx.send(device_readable_within_budget(&fifo, probe_timeout, &BUDGET));
            })
            .context("could not start the watched probe thread")?;
        let watchdog = probe_timeout * 20;
        let blocked = verdict_rx.recv_timeout(watchdog);

        let readable = temp_root.join("ordinary-kfd");
        fs::write(&readable, b"")?;
        let allowed = device_readable_within_budget(&readable, probe_timeout, &BUDGET);
        // The overrun probe's thread is still parked in `open` and stays that
        // way until this test binary exits — deliberately, and the same leak
        // `device_readable_within_budget` documents. Unlinking the FIFO does not
        // release it, and nothing here needs it released.
        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(
            blocked,
            Ok(false),
            "a device whose open never returns must read as not readable within its \
             {probe_timeout:?} bound, so the gate declines; `Err` here means the probe had still \
             not returned after {watchdog:?} — an unbounded open relocates the D-state hang into \
             the gate itself"
        );
        assert!(
            allowed,
            "an ordinary readable device must still pass, or the blocked case proves nothing"
        );
        Ok(())
    }

    /// Parked probes are bounded, and a probe that cannot claim a slot declines
    /// rather than opening anything on the calling thread.
    ///
    /// The escalation this pins is the one the review named: against a wedged
    /// device every poll leaves a thread parked in `open`, so an unbounded probe
    /// accumulates them until the process runs out of threads, at which point a
    /// fallback to a direct blocking open would reinstate exactly the hang the
    /// gate exists to refuse.
    ///
    /// Non-vacuous by construction. The device the capped probe is pointed at is
    /// an ordinary readable file, so an unbounded implementation — or one whose
    /// exhausted branch fell back to a blocking open — answers `true`. Only a
    /// probe that refuses on an exhausted budget answers `false`. The control
    /// re-runs the same device against a budget with a free slot and must get
    /// `true`, so a mutant returning `false` unconditionally fails too.
    ///
    /// Uses a budget of its own, with a limit of one: the single parked FIFO
    /// probe below spends it, and it is never freed, so sharing the production
    /// budget would silently shrink it for every other test in this process.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_probe_budget_declines_instead_of_parking_threads_without_bound() -> Result<()> {
        static ONE_SLOT: ProbeBudget = ProbeBudget::new(1);
        static SPARE: ProbeBudget = ProbeBudget::new(1);

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-budget-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let Some(fifo) = make_fifo(&temp_root, "blocking-kfd") else {
            let _ = fs::remove_dir_all(&temp_root);
            return Ok(());
        };
        let readable = temp_root.join("ordinary-kfd");
        fs::write(&readable, b"")?;

        let probe_timeout = Duration::from_millis(300);
        // Spends the only slot and never gives it back: the thread stays parked
        // in `open` on a FIFO with no writer, which is precisely the state the
        // cap is counting.
        assert!(
            !device_readable_within_budget(&fifo, probe_timeout, &ONE_SLOT),
            "a blocking open must read as not readable"
        );

        let with_budget_spent = device_readable_within_budget(&readable, probe_timeout, &ONE_SLOT);
        let with_a_free_slot = device_readable_within_budget(&readable, probe_timeout, &SPARE);
        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            !with_budget_spent,
            "with every probe slot held by a parked open, the pre-flight must decline rather \
             than open on the calling thread — that fallback is the hang this gate refuses"
        );
        assert!(
            with_a_free_slot,
            "the same readable device must pass when a slot is free, or the decline above \
             proves nothing about the budget"
        );
        Ok(())
    }

    /// A FIFO under `root`, or `None` on a host with no `mkfifo(1)`.
    ///
    /// Shelled out rather than calling `libc::mkfifo`: the workspace lints deny
    /// `unsafe_code` and one test fixture does not justify an exception. A
    /// missing coreutils is not evidence about the probe, so the caller skips.
    #[cfg(target_os = "linux")]
    fn make_fifo(root: &Path, name: &str) -> Option<PathBuf> {
        let fifo = root.join(name);
        let made = Command::new("mkfifo")
            .arg(&fifo)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        matches!(&made, Ok(status) if status.success()).then_some(fifo)
    }

    /// Pins the production wiring the injected helpers cannot: that
    /// [`amd_smi_json`] gates on [`KFD_DEVICE`] itself.
    ///
    /// Every other gate test reaches `amd_smi_json_with` with a device of its
    /// own choosing, so swapping the constant out of the public wrapper — which
    /// disables the gate for every production caller — leaves them all green.
    /// This compares the wrapper's observable verdict against the gate read
    /// directly, so the two can only agree while the wrapper passes the real
    /// device through.
    ///
    /// Non-vacuous on *every* host in an `e2e-test-hooks` build: the empty
    /// simulated root below has no `dev/kfd`, so the gate declines whatever the
    /// machine has, while a wrapper passing any other path — `/` being the
    /// obvious mutant — re-roots to the root directory itself, which opens.
    /// Without the feature it is non-vacuous on any host with no readable
    /// `/dev/kfd` (every non-GPU CI runner) and asserts only the absence of a
    /// spurious decline on a GPU host.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_public_amd_smi_entry_point_gates_on_the_real_kfd_device() {
        // Acquired by name rather than through `host_root_guard()`: this is the
        // one test here that also calls `RestoredEnvVar::set`, and the
        // env-mutation contract guard matches TEXT -- it recognises a
        // serializer by the `_TEST_LOCK` suffix, which the helper's name hides.
        // Same lock, spelled so the guard can see it.
        let _host_root = crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        #[cfg(feature = "e2e-test-hooks")]
        let _simulated_host = {
            let root = tempfile::tempdir().expect("a simulated host root");
            let variable = crate::test_env::RestoredEnvVar::set(
                crate::hardware_root::TEST_HOST_ROOT_ENV,
                root.path(),
            );
            (variable, root)
        };

        let gate_allows = kfd_readable(Path::new(KFD_DEVICE));

        // A deliberately bogus subcommand: if the gate allows the launch this
        // still fails, but with a *launch* error rather than a decline, and it
        // is the decline that is under test. The timeout is short because the
        // pre-flight has already passed by the time anything is spawned.
        let error = amd_smi_json(&["--rocm-cli-gate-probe"], Duration::from_secs(5)).err();
        let declined = error.is_some_and(|error| {
            let message = format!("{error:#}");
            message.contains("declined to launch") && message.contains(KFD_DEVICE)
        });

        assert_eq!(
            declined, !gate_allows,
            "amd_smi_json must decline exactly when the gate on {KFD_DEVICE} does; a wrapper \
             that passes some other path declines on neither host"
        );
    }

    /// The E2E path seam is an override, not a bypass: with the variable unset
    /// the pre-flight device is `/dev/kfd` on every build, and with it set the
    /// gate still *runs* — against the named path — rather than being skipped.
    ///
    /// Worth pinning because the seam's whole justification is that it cannot
    /// weaken a shipped build. Both halves are load-bearing: a `preflight_kfd_device`
    /// that read the variable unconditionally would fail the unset half here and
    /// hand any process on the host a way to disable the gate, and one that
    /// returned a readable path outright — the bypass this deliberately is not —
    /// would fail the declining half.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_e2e_kfd_path_seam_moves_the_gate_without_disabling_it() {
        // The same serializer the simulated-root tests take, rather than a
        // second one: both mutate the environment the pre-flight reads, and one
        // lock for the pair cannot deadlock against itself.
        let _env = crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        assert_eq!(
            preflight_kfd_device(),
            Path::new(KFD_DEVICE),
            "with no seam set, every build must gate on the real device"
        );

        #[cfg(feature = "e2e-test-hooks")]
        {
            let scratch = tempfile::tempdir().expect("a scratch dir");
            let absent = scratch.path().join("no-such-kfd");
            let _seam = crate::test_env::RestoredEnvVar::set(TEST_KFD_DEVICE_ENV, &absent);

            assert_eq!(
                preflight_kfd_device(),
                absent,
                "an e2e-test-hooks build must honour the seam"
            );

            let error = amd_smi_json(&["--rocm-cli-gate-probe"], Duration::from_secs(5))
                .expect_err("an unopenable seam path must decline the launch");
            let message = format!("{error:#}");
            assert!(
                message.contains("declined to launch")
                    && message.contains(&absent.display().to_string()),
                "the seam must move the gate, not remove it; got: {message}"
            );
        }
    }

    /// A failing `amd-smi` that prints a payload must not paste the payload into
    /// the error. `rocmd` puts this message in its snapshot note and the
    /// GPU-metrics watcher copies that note into rewrite-the-whole-file JSONL
    /// logs on every poll, which is how a 151K-character log line was produced.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_failing_amd_smi_cannot_paste_its_whole_output_into_the_error() -> Result<()> {
        const PAYLOAD_CHARS: usize = 200_000;

        if is_wsl_host() {
            return Ok(());
        }

        let _host_root = host_root_guard();

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-amd-smi-quote-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let fake_amd_smi = temp_root.join("amd-smi");
        // See the sibling gate test: planting an executable this process then
        // runs has to go through `fake_executable` or it races `ETXTBSY`.
        crate::fake_executable::write_fake_executable(
            &fake_amd_smi,
            format!("#!/bin/sh\nyes x | head -c {PAYLOAD_CHARS} | tr -d '\\n' >&2\nexit 1\n"),
        )?;
        let readable_device = temp_root.join("stand-in-kfd");
        fs::write(&readable_device, b"")?;

        let error = amd_smi_json_with(
            fake_amd_smi.as_os_str(),
            &readable_device,
            &["list", "--json"],
            Duration::from_secs(30),
        )
        .expect_err("a child that exits 1 must fail the call");
        let message = format!("{error:#}");
        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            message.chars().count() < PAYLOAD_CHARS / 10,
            "the failure message quoted {} characters of child output unbounded",
            message.chars().count()
        );
        assert!(
            message.contains("amd-smi list --json failed"),
            "the bound must not cost the message its subject: {message}"
        );
        assert!(
            message.contains("[truncated]"),
            "an over-long quote must say it was cut, not look like the whole story: {message}"
        );
        Ok(())
    }

    #[test]
    fn resolve_amd_smi_binary_prefers_home_rocm_venv_path() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-amd-smi-{}", unix_time_millis()));
        let bin_dir = temp_root.join("rocm_venvs").join("default").join("bin");
        fs::create_dir_all(&bin_dir)?;
        let amd_smi_path = bin_dir.join("amd-smi");
        fs::write(&amd_smi_path, b"#!/bin/sh\nexit 0\n")?;

        let resolved = resolve_amd_smi_binary_in_home(Some(Path::new(&temp_root)));

        let _ = fs::remove_file(&amd_smi_path);
        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(resolved, amd_smi_path.into_os_string());
        Ok(())
    }

    #[test]
    fn resolve_amd_smi_binary_in_registry_uses_newest_runtime_sdk_bin() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-amd-smi-registry-{}", unix_time_millis()));
        let registry_dir = temp_root.join("runtimes/registry");
        fs::create_dir_all(&registry_dir)?;

        // Older runtime: amd-smi only under the venv install_root/bin.
        let old_root = temp_root.join("release-wheel-gfx94x-dcgpu-7-13-0");
        let old_bin = old_root.join("bin");
        fs::create_dir_all(&old_bin)?;
        fs::write(old_bin.join("amd-smi"), b"#!/bin/sh\nexit 0\n")?;
        fs::write(
            registry_dir.join("old.json"),
            serde_json::to_vec(&serde_json::json!({
                "runtime_id": "therock-stable:gfx94X-dcgpu",
                "install_root": old_root,
                "installed_at_unix_ms": 1_000_u128,
                "rocm_sdk": { "import_ok": true },
            }))?,
        )?;

        // Newer runtime: amd-smi under the SDK devel bin_path.
        let new_bin =
            temp_root.join("release-wheel-gfx94x-dcgpu-7-14-0a20260611/_rocm_sdk_devel/bin");
        fs::create_dir_all(&new_bin)?;
        let new_amd_smi = new_bin.join("amd-smi");
        fs::write(&new_amd_smi, b"#!/bin/sh\nexit 0\n")?;
        fs::write(
            registry_dir.join("new.json"),
            serde_json::to_vec(&serde_json::json!({
                "runtime_id": "therock-stable:gfx94X-dcgpu",
                "installed_at_unix_ms": 2_000_u128,
                "rocm_sdk": { "import_ok": true, "bin_path": new_bin },
            }))?,
        )?;

        let resolved = resolve_amd_smi_binary_in_registry(&registry_dir);

        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(resolved, Some(new_amd_smi.into_os_string()));
        Ok(())
    }

    #[test]
    fn resolve_amd_smi_binary_in_registry_breaks_install_time_ties_on_the_runtime_key() -> Result<()>
    {
        // Same shape as the active-runtime fallback test: both pairings of file
        // name and runtime key, so a read-order pick is wrong in one of them.
        for (first_file_key, second_file_key) in [("aaa", "bbb"), ("bbb", "aaa")] {
            let temp_root = std::env::temp_dir().join(format!(
                "rocm-cli-amd-smi-tie-{first_file_key}-{}-{}",
                std::process::id(),
                unix_time_millis()
            ));
            let registry_dir = temp_root.join("runtimes/registry");
            fs::create_dir_all(&registry_dir)?;
            for (file, key) in [("first", first_file_key), ("second", second_file_key)] {
                let bin = temp_root.join(key).join("bin");
                fs::create_dir_all(&bin)?;
                fs::write(bin.join("amd-smi"), b"#!/bin/sh\nexit 0\n")?;
                fs::write(
                    registry_dir.join(format!("{file}.json")),
                    serde_json::to_vec(&serde_json::json!({
                        "runtime_key": key,
                        "runtime_id": "therock-stable:gfx94X-dcgpu",
                        "installed_at_unix_ms": 2_000_u128,
                        "rocm_sdk": { "import_ok": true, "bin_path": bin },
                    }))?,
                )?;
            }

            let resolved = resolve_amd_smi_binary_in_registry(&registry_dir);
            let expected = temp_root.join("aaa").join("bin").join("amd-smi");
            let _ = fs::remove_dir_all(&temp_root);

            assert_eq!(
                resolved,
                Some(expected.into_os_string()),
                "files first={first_file_key}, second={second_file_key}"
            );
        }
        Ok(())
    }
}
